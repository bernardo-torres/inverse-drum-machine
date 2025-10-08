import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import pandas as pd
import rootutils
import yaml
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from idm import CHECKPOINT_TYPE_BEST, CHECKPOINT_TYPE_LAST
from idm.config_resolvers import *
from idm.inference import find_checkpoint
from idm.run_utils import (
    RankedLogger,
    log_hyperparameters,
    print_config_tree,
    task_wrapper,
)

# Setup project root for consistent path management
log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()


def instantiate_components(cfg: Optional[DictConfig], component_type: str) -> List[Any]:
    """
    Instantiates a list of components (e.g., callbacks, loggers) from a config.

    Args:
        cfg: A DictConfig object containing component configurations.
        component_type: A string name for the component type (used for logging).

    Returns:
        A list of instantiated components.
    """
    components: List[Any] = []

    if not cfg:
        log.warning(f"No {component_type} configs found! Skipping...")
        return components

    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{component_type.capitalize()} config must be a DictConfig!")

    for _, conf in cfg.items():
        if isinstance(conf, DictConfig) and "_target_" in conf:
            log.info(f"Instantiating {component_type} <{conf._target_}>")
            components.append(hydra.utils.instantiate(conf))

    return components


@task_wrapper
class TrainingPipeline:
    """Encapsulates the entire training and evaluation pipeline."""

    def __init__(self, cfg: DictConfig):
        """Initializes the pipeline with the given Hydra configuration.

        Args:
            cfg: A DictConfig configuration composed by Hydra.
        """
        self.cfg = cfg
        self.cfg_data = OmegaConf.to_container(cfg, resolve=True)
        self.trainer: Optional[Trainer] = None
        self.model: Optional[LightningModule] = None
        self.datamodule: Optional[LightningDataModule] = None
        self.callbacks: Optional[List[Callback]] = None
        self.logger: Optional[List[Logger]] = None

    def _instantiate_objects(self) -> None:
        """Instantiates all necessary objects from the configuration."""
        log.info(f"Instantiating datamodule <{self.cfg.data._target_}>")
        self.datamodule = hydra.utils.instantiate(self.cfg.data)

        log.info(f"Instantiating model <{self.cfg.model._target_}>")
        self.model = hydra.utils.instantiate(self.cfg.model)

        log.info("Instantiating callbacks...")
        self.callbacks = instantiate_components(self.cfg.get("callbacks"), "callback")

        log.info("Instantiating loggers...")
        self.logger = instantiate_components(self.cfg.get("logger"), "logger")

        log.info(f"Instantiating trainer <{self.cfg.trainer._target_}>")
        self.trainer = hydra.utils.instantiate(
            self.cfg.trainer, callbacks=self.callbacks, logger=self.logger
        )

    def _run_tuner(self) -> None:
        """Runs the Lightning tuner to find the optimal batch size."""
        if not self.cfg.get("tuner"):
            return

        log.info("Starting batch size tuning!")
        tuner = Tuner(self.trainer)
        new_batch_size = tuner.scale_batch_size(
            self.model, datamodule=self.datamodule, **self.cfg.tuner
        )

        # Set a slightly smaller batch size for stability
        safe_batch_size = new_batch_size - min(new_batch_size // 10, 1)
        self.datamodule.batch_size = safe_batch_size
        self.cfg.data.batch_size = safe_batch_size  # Update config as well
        self.cfg_data["data"]["batch_size"] = safe_batch_size  # Update resolved config

        log.info(f"Finished batch size tuning! New batch size: {safe_batch_size}")

        # Re-instantiate trainer and model after tuning
        self.trainer = hydra.utils.instantiate(
            self.cfg.trainer, callbacks=self.callbacks, logger=self.logger
        )
        self.model = hydra.utils.instantiate(self.cfg.model)

    def _log_hyperparameters(self) -> None:
        """Logs hyperparameters if a logger is configured."""
        if not self.logger:
            return

        log.info("Logging hyperparameters!")
        object_dict = {
            "cfg": self.cfg,
            "datamodule": self.datamodule,
            "model": self.model,
            "callbacks": self.callbacks,
            "logger": self.logger,
            "trainer": self.trainer,
        }
        log_hyperparameters(object_dict)

    def run_training(self) -> None:
        """Executes the training phase."""
        if not self.cfg.get("train"):
            return

        log.info("Starting training!")
        ckpt_path = find_checkpoint(self.cfg.paths.output_dir, type=CHECKPOINT_TYPE_LAST)
        if ckpt_path:
            log.info(f"Resuming training from checkpoint: {ckpt_path}")

        self.trainer.fit(model=self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def run_testing(self) -> None:
        """Executes the testing phase."""
        if not self.cfg.get("test"):
            return

        log.info("Starting testing!")
        # Use the best checkpoint from the trainer if available
        ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if not ckpt_path:
            log.warning("Best checkpoint not found in trainer, searching output directory.")
            ckpt_path = find_checkpoint(
                self.cfg.paths.output_dir, type=CHECKPOINT_TYPE_BEST, filename_contains="val"
            )
            if not ckpt_path:
                ckpt_path = find_checkpoint(self.cfg.paths.output_dir, type=CHECKPOINT_TYPE_LAST)
                log.info("Falling back to last checkpoint found for testing.")

        if not ckpt_path:
            log.warning("No best checkpoint found! Using current model weights for testing.")
        else:
            log.info(f"Using checkpoint for testing: {ckpt_path}")

        self.trainer.test(model=self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def run(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Runs the complete pipeline: setup, train, and test.

        Returns:
            A tuple containing the final metrics and a dictionary of instantiated objects.
        """
        if self.cfg.get("seed"):
            L.seed_everything(self.cfg.seed, workers=True)

        self._instantiate_objects()
        self._log_hyperparameters()
        self._run_tuner()

        if self.cfg.get("tuner"):
            # Re-log hyperparameters if they were changed by the tuner
            self._log_hyperparameters()

        # Save the fully resolved config for reproducibility
        resolved_config_path = Path(self.cfg.paths.output_dir) / ".hydra" / "config_resolved.yaml"
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        with open(resolved_config_path, "w") as f:
            yaml.dump(resolved_config, f)

        self.run_training()
        self.run_testing()

        # Gather final metrics
        metric_dict = self.trainer.callback_metrics
        object_dict = {
            "cfg": self.cfg,
            "datamodule": self.datamodule,
            "model": self.model,
            "callbacks": self.callbacks,
            "logger": self.logger,
            "trainer": self.trainer,
        }

        return metric_dict, object_dict


def save_and_process_results(cfg: DictConfig, metric_dict: Dict[str, Any]) -> None:
    """Saves metrics to a CSV file and generates LaTeX output.

    Args:
        cfg: The Hydra configuration object.
        metric_dict: A dictionary containing the final metrics.
    """
    output_dir = Path(cfg.paths.output_dir)
    # Convert tensor metrics to NumPy for saving
    for key, value in metric_dict.items():
        if hasattr(value, "numpy"):
            metric_dict[key] = value.numpy().round(2)

    # Save metrics to a CSV file
    dataset_name = cfg.data.get("name", "test")
    result_path = output_dir / f"test_metrics_{dataset_name}.csv"
    pd.DataFrame.from_dict(metric_dict, orient="index").to_csv(result_path)
    log.info(f"Test metrics saved to: {result_path}")

    # Generate LaTeX summary of results
    # parse_test_results_for_latex(cfg, stage_str="test", result_path=str(result_path))


@hydra.main(config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training, controlled by Hydra.

    Args:
        cfg: The configuration object composed by Hydra.

    Returns:
        The value of the optimized metric, if specified.
    """
    if cfg.get("extras") and cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)

    # The original script had logic for 'dora', which is not a standard library.
    # This refactor assumes a standard Hydra setup. If 'dora' is a custom
    # tool you use, its logic can be integrated here.
    # For example:
    # if cfg.get("dora"):
    #     from dora import get_xp
    #     xp = get_xp()
    #     cfg.paths.hash = xp.sig

    pipeline = TrainingPipeline(cfg)
    metric_dict, _ = pipeline.run()

    # Save results and get the primary metric for hyperparameter optimization
    save_and_process_results(cfg, metric_dict)


if __name__ == "__main__":

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that includes a traceback."""
        log_file = file if hasattr(file, "write") else sys.stderr
        traceback.print_stack(file=log_file)
        log_file.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback
    main()
