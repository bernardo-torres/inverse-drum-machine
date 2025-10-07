import functools
import logging
import traceback
from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import rich
import rich.syntax
import rich.tree
import yaml
from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only
from omegaconf import DictConfig, OmegaConf


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes with their rank
        prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank of the process it's
        being logged from. If `'rank'` is provided, then the log will only occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)


log = RankedLogger(__name__, rank_zero_only=True)


#


def task_wrapper(task_func: Callable) -> Callable:
    """
    A robust decorator for ML tasks that handles exceptions, manages resources,
    and saves failure artifacts for easier debugging.

    Features:
    - Ensures resources like loggers (e.g., wandb) are properly closed.
    - Logs exceptions to a file for detailed analysis.
    - Optionally allows runs to fail gracefully without crashing a multirun (controlled by `cfg.ignore_exceptions`).
    - On failure, saves the exact configuration and creates a `.FAILED` marker file.
    """

    @functools.wraps(task_func)
    def wrap(cfg: DictConfig) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        output_dir = Path(cfg.paths.output_dir)
        try:
            # Execute the decorated task function (e.g., train)
            return task_func(cfg=cfg)

        except Exception as ex:
            log.error(f"Task execution failed with exception: {ex}")
            log.exception("")  # Logs the full traceback

            # Create failure artifacts for easy debugging
            failure_log_path = output_dir / "logs"
            failure_log_path.mkdir(exist_ok=True, parents=True)

            # 1. Save the exception traceback to a file
            with open(failure_log_path / "failure_traceback.log", "w") as f:
                f.write(traceback.format_exc())

            # 2. Save the failed configuration
            with open(failure_log_path / "failure_config.yaml", "w") as f:
                yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

            # 3. Create a marker file to easily identify failed runs
            (output_dir / ".FAILED").touch()

            # Conditionally re-raise the exception
            # For hyperparameter sweeps (e.g., Optuna), you might want to set this to False
            if not cfg.get("ignore_exceptions", False):
                raise ex

            return None, None  # Return None on graceful failure

        finally:
            # This block always runs, whether the task succeeded or failed
            log.info(f"Output dir: {output_dir}")

            # Safely close wandb run if it exists and is active
            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb run.")
                    wandb.finish()

    return wrap


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    # lets get config keys except for the ones above
    cfg_keys = list(cfg.keys())
    for key in ["model", "data", "trainer"]:
        cfg_keys.remove(key)

    # save the rest of the config
    for key in cfg_keys:
        hparams[key] = cfg[key]

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
