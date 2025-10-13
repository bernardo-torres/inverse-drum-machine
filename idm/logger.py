import argparse
import io
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import lightning.pytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

try:
    import wandb
except:
    pass

matplotlib.use("Agg")  # Use a non-interactive backend

metric_limits = {
    "si_sdr": (-10, 30),
    "nsdr": (-10, 25),
    "ms_stft": (0, 4),
    "lsd": (0, 4),
    "2f_peaq": (0, 70),
    "2f": (0, 70),
    "ms_stft_resynth": (0, 4),
    "sample_multiresolutionstftloss": (0, 5),
    "sample_lsd": (0, 7),
    "sample_pes": (0, 0.8),
    "sample_eps": (0, 0.8),
}


default_groups = [
    "val/f1_track",
    "val/f1_agg",
    "test/f1_track",
    "test/f1_agg",
    "test_f1_track",
    "test_f1_agg",
    "val/precision_track",
    "val/recall_track",
    "val/precision_agg",
    "val/recall_agg",
    "test/precision_track",
    "test/recall_track",
    "val/precision",
    "val/recall",
    "test/precision",
    "test/recall",
    "val/separation_SI_SDR_masked_per_class",
    "val/separation_LSD_masked_per_class",
    "val/separation_2f_masked_per_class",
    "val/sample_reconstruction",
    "test/sample_reconstruction",
    "val_tot_onsets_per_silent_tracks",
    "test_tot_onsets_per_silent_tracks",
] + [
    f"test{stage}{metric}_{metric_type}_per_class"
    for stage in [
        "/",
        "-gt_samples_",
        "-gt_onsets_",
        "-9_class_",
        "-gt_onsets-gt_samples_",
        "-gt_onsets_9_class_",
        "-gt_onsets-gt_samples-9_class_",
        "-gt_kit_labels_",
        "-gt_onsets-gt_kit_labels_",
        "-gt_samples-gt_kit_labels_",
        "-gt_onsets-gt_samples-gt_kit_labels_",
    ]
    for metric in [
        "separation_SI_SDR",
        "separation_nSDR",
        "separation_LSD",
        "separation_2f",
        "separation_EPS",
        "separation_PES",
    ]
    for metric_type in ["masked", "synth"]
]


def create_spectrogram(audio, sr=None, figsize=(8, 4), return_fig=False, **kwargs):
    """Generate and cache spectrogram."""
    fig, ax = plt.subplots(figsize=figsize)
    stft = np.abs(librosa.stft(audio))
    D = librosa.amplitude_to_db(stft, ref=np.max(stft) + 1e-10)
    D = np.maximum(D, D.max() - 80.0)  # Clip at -80 dB
    try:
        img = librosa.display.specshow(D, y_axis="log", x_axis="time", ax=ax)
    except:
        # Display empty plot
        D = np.zeros((1025, 431))
        img = librosa.display.specshow(D, y_axis="log", x_axis="time", ax=ax)
    plt.colorbar(img, format="%+2.0f dB")  # Pass the img reference to colorbar
    # plt.title("Spectrogram")
    if return_fig:
        return fig

    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    label_fontsize = kwargs.get("label_fontsize", 12)
    tick_fontsize = kwargs.get("tick_fontsize", 10)
    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    # ax.tick_params(axis='y', which='major', labelsize=tick_fontsize)

    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf


def log_audio_logger(pl_module, audio, name):
    if pl_module.logger is None:
        return
    # Let's normalize the audio
    audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    audio = audio * 0.95  # To avoid clipping

    loggers = pl_module.loggers
    for logger in loggers:
        if isinstance(logger, lightning.pytorch.loggers.tensorboard.TensorBoardLogger):
            logger.experiment.add_audio(
                name, audio, pl_module.current_epoch, sample_rate=pl_module.sampling_rate
            )
        elif isinstance(logger, CustomFileLogger):
            logger.log_audio(audio, name, pl_module.sampling_rate, step=pl_module.global_step)
        elif isinstance(pl_module.logger, lightning.pytorch.loggers.wandb.WandbLogger):
            if not name.startswith("audio"):
                _name = f"audio_{name}"
            logger.log_metrics(
                {_name: [wandb.Audio(audio, sample_rate=pl_module.sampling_rate)]},
                step=pl_module.global_step + 1,
            )


def log_image_logger(pl_module, img, name):
    if not hasattr(pl_module, "logger") or pl_module.logger is None:
        return

    # if isinstance(pl_module.logger, lightning.pytorch.loggers.tensorboard.TensorBoardLogger):
    #     pl_module.logger.experiment.add_figure(name, img, pl_module.global_step)
    #     pl_module.logger.experiment.flush()
    #     import time
    #     time.sleep(0.1)
    loggers = pl_module.loggers
    for logger in loggers:
        if isinstance(logger, lightning.pytorch.loggers.tensorboard.TensorBoardLogger):
            logger.experiment.add_figure(name, img, pl_module.global_step)
        elif isinstance(logger, CustomFileLogger):
            logger.log_image(name, img, step=pl_module.global_step)
        elif isinstance(pl_module.logger, lightning.pytorch.loggers.wandb.WandbLogger):
            _img = fig_to_numpy(img)
            if not name.startswith("image"):
                _name = f"image_{name}"
            logger.log_metrics({_name: [wandb.Image(_img)]}, step=pl_module.global_step + 1)


def log_html_logger(pl_module, html_path, name):
    if not hasattr(pl_module, "logger"):
        return
    if pl_module.logger is None:
        return

    loggers = pl_module.loggers
    for logger in loggers:
        if isinstance(logger, CustomFileLogger):
            # logger.log_html(html_path, name, step=pl_module.global_step)
            continue
        elif isinstance(logger, lightning.pytorch.loggers.wandb.WandbLogger):
            if not os.path.exists(html_path):
                warnings.warn(f"HTML file {html_path} does not exist.")
                return
            if not name.startswith("html"):
                _name = f"html_{name}"
            logger.log_metrics({_name: wandb.Html(open(html_path))}, step=pl_module.global_step + 1)


def fig_to_numpy(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


class CustomFileLogger(Logger):
    """Custom Lightning logger that saves metrics, images, and audio to files.

    Handles plotting of metrics automatically and saves them as matplotlib figures.
    """

    def __init__(
        self,
        save_dir: str,
        name: str = "default",
        version: int | str | None = "0.1",
        metric_plot_kwargs: dict | None = None,
        groups: list | None = None,
        image_format: str = "png",
        audio_format: str = "wav",
    ):
        """Initialize the logger.

        Args:
            save_dir: Directory to save all logs
            name: Name of the experiment
            version: Version of the experiment
            metric_plot_kwargs: Dictionary of kwargs for matplotlib plotting
            image_format: Format to save images (png, jpg, etc.)
            audio_format: Format to save audio files (wav, mp3, etc.)
        """
        super().__init__()
        self._save_dir = str(save_dir)
        self._name = name
        self._version = version
        self.metric_plot_kwargs = metric_plot_kwargs or {}
        if groups is None:
            groups = default_groups
        self.groups = {group: [] for group in groups} if groups else {}
        self.image_format = image_format
        self.audio_format = audio_format
        self.metric_history = {}
        self._logger = logging.getLogger(__name__)

        # Initialize experiment only on rank 0
        self._setup_experiment()

    @property
    def save_dir(self) -> str | None:
        """Return the root directory where experiment logs get saved."""
        return self._save_dir

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return self._name

    @property
    def version(self) -> int | str:
        """Return the experiment version."""
        return self._version

    @property
    def log_dir(self) -> str:
        """Return the directory for this run."""
        return self.save_dir

    @rank_zero_experiment
    def _setup_experiment(self) -> None:
        """Initialize experiment directories."""
        self.metrics_dir = Path(self.save_dir) / "metrics"
        self.img_folder = Path(self.save_dir) / "images"
        self.audio_dir = Path(self.save_dir) / "audio"

        for dir_path in [self.metrics_dir, self.img_folder, self.audio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics and create/update plots."""
        if step is None:
            step = 0

        for metric_name, value in metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = {"steps": [], "values": []}

            self.metric_history[metric_name]["steps"].append(step)
            self.metric_history[metric_name]["values"].append(value)

            # If the metric belongs to a group, plot the group
            found_groups = self._check_metric_group(metric_name)
            if len(found_groups) > 0:
                for group_name in found_groups:
                    self._plot_group(group_name)

            self._plot_metric(metric_name)

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | argparse.Namespace) -> None:
        """Log hyperparameters to a file."""
        if isinstance(params, argparse.Namespace):
            params = vars(params)

        hyperparams_file = Path(self.save_dir) / "hyperparameters.txt"
        with open(hyperparams_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

    @rank_zero_only
    def log_image(
        self, name: str, image: Tensor | np.ndarray = None, step: int | None = None
    ) -> None:
        """Save image to file."""
        if step is not None:
            filename = f"{name}_step_{step}.{self.image_format}"
        else:
            filename = f"{name}.{self.image_format}"

        filename = self.img_folder / Path(filename)
        os.makedirs(filename.parent, exist_ok=True)
        if image is None:
            plt.savefig(filename)
        elif isinstance(image, matplotlib.figure.Figure):
            image.savefig(filename)
        else:
            if isinstance(image, Tensor):
                image = image.cpu().numpy()
                plt.imsave(filename, image)

    @rank_zero_only
    def log_audio(
        self,
        audio: Tensor | np.ndarray,
        name: str,
        sample_rate: int,
        step: int | None = None,
    ) -> None:
        """Save audio to file."""
        if isinstance(audio, Tensor):
            audio = audio.cpu().numpy()

        if step is not None:
            filename = f"{name}_step_{step}.{self.audio_format}"
        else:
            filename = f"{name}.{self.audio_format}"

        os.makedirs((self.audio_dir / filename).parent, exist_ok=True)

        sf.write(self.audio_dir / filename, audio, sample_rate)

    def _check_metric_group(self, metric_name: str) -> None:
        """Check if the metric belongs to any group.

        Metrics eg val/f1_KD and val/f1_SD would belong to group f1 if it exists.
        """
        # Let's check if the metric belongs to any group, e.g the group name is fully contained in the metric name
        found_groups = []
        for group_name in self.groups:
            if group_name.replace("/", "_") in metric_name.replace("/", "_"):
                if metric_name not in self.groups[group_name]:
                    self.groups[group_name].append(metric_name)
                found_groups.append(group_name)
        return found_groups

    def _plot_group(self, group_name: str) -> None:
        """Create and save plot for a group of metrics."""

        # Let's plot all metrics in the group. If there is only one value, plot it as a bar plot
        metrics = self.groups[group_name]
        if "f1_track" in group_name:
            pass
        if len(metrics) == 0:
            return

        plt.figure(figsize=(10, 6))
        if len(self.metric_history[metrics[0]]["steps"]) > 1:
            for metric_name in metrics:
                plt.plot(
                    self.metric_history[metric_name]["steps"],
                    self.metric_history[metric_name]["values"],
                    label=metric_name,
                    **self.metric_plot_kwargs,
                )
                plt.legend()
        else:
            bar_width = 0.2
            # if only one value is present, plot all metrics in the group in the same bar plot
            values = [self.metric_history[metric_name]["values"][0] for metric_name in metrics]
            # Lets use a categorical color map for the bars
            bar_colors = plt.get_cmap("tab20")(np.arange(len(values)))
            # Let's get only the last after the last "/"
            _metrics_labels = ["/".join(metr.split("/")[1:]) for metr in metrics]
            _metrics_labels = [met.replace("/", "") for met in _metrics_labels]
            plt.bar(_metrics_labels, values, color=bar_colors)
            # Put values on top of the bars
            for i, value in enumerate(values):
                val_str = f"{value:.2f}" if np.abs(np.round(value, 2)) > 0 else ""
                plt.text(i, value, val_str, ha="center", va="bottom", color="black", fontsize=7)

            for metric_, limits in metric_limits.items():
                if metric_ in metrics[0].lower():
                    plt.ylim(limits)

        # if "PES" in group_name:
        #     pass

        plt.title(f"{group_name}")
        plt.xlabel("Steps")
        plt.ylabel("Value")
        # plt.legend()
        plt.grid(True)

        _group_name = group_name.replace("/", "_")
        dirname = "aggregated_metrics"
        os.makedirs(self.metrics_dir / dirname, exist_ok=True)
        plt.savefig(self.metrics_dir / dirname / f"{_group_name}.png")
        plt.close()

    def _plot_metric(self, metric_name: str) -> None:
        """Create and save plot for a specific metric."""

        _metric_name = metric_name.replace("/", "_")
        if "per" in _metric_name:
            dirname = _metric_name.split("_per")[0]
            figname = "_".join(_metric_name.split("_per")[1:])
        else:
            dirname = "individual_metrics"
            figname = _metric_name
        os.makedirs(self.metrics_dir / dirname, exist_ok=True)

        if len(self.metric_history[metric_name]["steps"]) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.metric_history[metric_name]["steps"],
                self.metric_history[metric_name]["values"],
                **self.metric_plot_kwargs,
            )

            plt.title(f"{metric_name}")
            plt.xlabel("Steps")
            plt.ylabel(metric_name)
            plt.grid(True)

            plt.savefig(self.metrics_dir / dirname / f"{figname}.png")
            plt.close()
            # We also cleanup the text file if it exists
            if (self.metrics_dir / dirname / f"{figname}.txt").exists():
                os.remove(self.metrics_dir / dirname / f"{figname}.txt")
        else:
            # if only one value is present, save it as a text file
            with open(self.metrics_dir / dirname / f"{figname}.txt", "w") as f:
                f.write(f"{metric_name}: {self.metric_history[metric_name]['values'][0]}")

    @rank_zero_only
    def save(self) -> None:
        """Save any pending data."""
        for metric_name in self.metric_history:
            self._plot_metric(metric_name)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize the logging."""
        with open(Path(self.save_dir) / "status.txt", "w") as f:
            f.write(f"Experiment finished with status: {status}\n")
        self.save()

    def experiment(self):
        """Return the experiment object."""
        return None
