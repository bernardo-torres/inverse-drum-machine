import base64
import os
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from IPython.display import HTML, Audio, display

from idm.callbacks.base import BaseCallback
from idm.logger import (
    create_spectrogram,
    log_audio_logger,
    log_html_logger,
)
from idm.utils import cpu_numpy


class AudioTable:
    def __init__(
        self,
        path,
        name,
        sr=None,
        display_audio=True,
        include_spectrograms=False,
        plot_waveform=False,
        normalize_audio=True,
        **kwargs,
    ):
        """Initialize an AudioTable for saving model audio outputs.

        Args:
            path (str): Path where the HTML table will be saved
            name (str): Name of the table
            sr (int): Sampling rate for the audio files
            display_audio (bool): Whether to display audio in the table
            include_spectrograms (bool): Whether to include spectrograms alongside audio
            plot_waveform (bool): Whether to plot the waveform
        """
        self._path = path
        self.name = name
        self.sr = sr
        self.display_audio = display_audio
        self.include_spectrograms = include_spectrograms
        self.plot_waveform = plot_waveform
        self.normalize_audio = normalize_audio
        self.kwargs = kwargs
        self.clear()

    def path(self, batch_idx=None):
        path = self._compose_path(batch_idx)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _compose_path(self, batch_idx):
        return (
            os.path.join(self._path, f"batch_idx_{batch_idx}", f"{self.name}.html")
            if batch_idx is not None
            else os.path.join(self._path, f"{self.name}.html")
        )

    def clear(self):
        """Clear all data from the table."""
        self.data = OrderedDict()
        self.columns = []

    def _create_spectrogram(self, audio_data):
        """Generate spectrogram and return HTML."""
        audio_data = cpu_numpy(audio_data)
        if audio_data.ndim == 2:
            audio_data = audio_data[0]
        # Create spectrogram
        buf = create_spectrogram(audio_data, self.sr, **self.kwargs)

        # Convert to base64 for HTML embedding
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return img_base64
        # return f'<img src="data:image/png;base64,{img_base64}" style="width:400px">'

    def _create_waveform(self, audio_data):
        """Generate waveform plot and return base64 encoded image."""
        import io

        figsize = self.kwargs.get("figsize", (4, 1))
        plt.figure(figsize=figsize)
        plt.plot(audio_data, linewidth=1)
        y2 = 1 if figsize[0] > 10 else 2
        y1 = -1 if audio_data.min() < 0 else 0
        plt.ylim(y1, y2)
        x1 = -5000 if figsize[0] > 10 else -100
        x2 = len(audio_data) + 2000 if figsize[0] > 10 else len(audio_data)
        plt.xlim(x1, x2)
        # plt.axis('off')
        # plt.tick_params(axis='both', which='both',
        #            bottom=False, top=False, left=False, right=False,
        #            labelbottom=False, labelleft=False)
        plt.margins(x=0.1, y=0.1)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return img_base64

    def _get_audio_html(self, generated_data, gt_data, width=120, height=30, skip_audio=False):
        if generated_data is None:
            return ""

        def normalize_audio(data):
            if data is None:
                return None

            data = cpu_numpy(data)
            if not isinstance(data, np.ndarray):
                raise ValueError(
                    "Unsupported audio data type. Must be numpy array or torch tensor."
                )

            max_val = np.max(np.abs(data))
            if max_val < 1e-4:
                return None
            if not self.normalize_audio:
                # We will only normalize if max abs >1
                if max_val >= 1:
                    data = data / max_val * 0.95
                return np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

            if max_val > 0:
                data = data / max_val
            # data = np.clip(data + 1e-7, -1, 1)
            return data * 0.95

        # generated_data = normalize_audio(generated_data) if not self.plot_waveform else generated_data
        generated_data = normalize_audio(generated_data)
        # if generated_data is None:
        #     return ""
        gt_data = normalize_audio(gt_data)

        # Initialize flags for HTML elements
        has_gen = (
            generated_data is not None
            and isinstance(generated_data, np.ndarray)
            and not (generated_data == 0).all()
        )
        has_gt = gt_data is not None and isinstance(gt_data, np.ndarray) and not skip_audio

        # Create audio elements if display_audio is True
        gen_audio_html = ""
        gt_audio_html = ""
        if self.display_audio:
            if has_gen:
                gen_audio = Audio(generated_data, autoplay=False, rate=self.sr, normalize=False)
                gen_audio_base64 = base64.b64encode(gen_audio.data).decode("utf-8")
                gen_audio_html = (
                    f'<audio controls style="width:{width}px;height:{height}px">'
                    f'<source src="data:audio/wav;base64,{gen_audio_base64}" type="audio/wav">'
                    f"</audio>"
                )

            if has_gt:
                gt_audio = Audio(gt_data, autoplay=False, rate=self.sr, normalize=False)
                gt_audio_base64 = base64.b64encode(gt_audio.data).decode("utf-8")
                gt_audio_html = (
                    f'<audio controls style="width:{width}px;height:{height}px">'
                    f'<source src="data:audio/wav;base64,{gt_audio_base64}" type="audio/wav">'
                    f"</audio>"
                )

        # Create spectrograms if enabled
        spec_html = ""
        if self.include_spectrograms:
            if has_gen and not has_gt:
                gen_spec_base64 = self._create_spectrogram(generated_data)
                spec_html = (
                    f'<div style="display:flex;justify-content:center;width:100%">'
                    f'<img src="data:image/png;base64,{gen_spec_base64}" style="width:{width*2}px;height:60px">'
                    "</div>"
                )
            elif has_gen and has_gt:
                gen_spec_base64 = self._create_spectrogram(generated_data)
                gt_spec_base64 = self._create_spectrogram(gt_data)
                spec_html = (
                    f'<div style="display:flex;justify-content:space-between;width:100%;gap:10px">'
                    f'<img src="data:image/png;base64,{gen_spec_base64}" style="width:{width}px;height:60px">'
                    f'<img src="data:image/png;base64,{gt_spec_base64}" style="width:{width}px;height:60px">'
                    "</div>"
                )
            elif not has_gen and has_gt:
                gt_spec_base64 = self._create_spectrogram(gt_data)
                spec_html = (
                    f'<div style="display:flex;justify-content:space-between;width:100%;gap:10px">'
                    f'<div style="display:flex;justify-content:center;width:100%">'
                    f'<img src="data:image/png;base64,{gt_spec_base64}" style="width:{width*2}px;height:60px">'
                    "</div>"
                )

        # Create waveform plots if enabled
        wave_html = ""
        if self.plot_waveform:
            if has_gen and not has_gt:
                gen_wave_base64 = self._create_waveform(generated_data)
                wave_html = (
                    f'<div style="display:flex;justify-content:center;width:100%;min-height:80px">'  # Added min-height
                    f'<img src="data:image/png;base64,{gen_wave_base64}" style="width:{width*2}px;height:60px">'  # Increased height
                    "</div>"
                )
            if has_gt and has_gen:
                gt_wave_base64 = self._create_waveform(gt_data)
                gen_wave_base64 = self._create_waveform(generated_data)
                wave_html = (
                    f'<div style="display:flex;justify-content:space-between;width:100%;gap:10px;min-height:80px">'  # Added min-height
                    f'<img src="data:image/png;base64,{gen_wave_base64}" style="width:{width}px;height:60px">'  # Increased height
                    f'<img src="data:image/png;base64,{gt_wave_base64}" style="width:{width}px;height:60px">'  # Increased height
                    "</div>"
                )
            elif has_gt and not has_gen:
                gt_wave_base64 = self._create_waveform(gt_data)
                wave_html = (
                    f'<div style="display:flex;justify-content:center;width:100%;min-height:80px">'  # Added min-height
                    f'<img src="data:image/png;base64,{gt_wave_base64}" style="width:{width*2}px;height:60px">'  # Increased height
                    "</div>"
                )

        # Make sure we closed all opened figures
        plt.close("all")

        # Combine all elements
        if has_gt:
            gen_div = (
                f'<div style="flex:1">'
                f'<p style="margin:2px;font-size:0.9em">Generated</p>'
                f"{gen_audio_html}</div>"
            )

            gt_div = (
                f'<div style="flex:1">'
                f'<p style="margin:2px;font-size:0.9em">Ground Truth</p>'
                f"{gt_audio_html}</div>"
            )

            audio_div = (
                (
                    f'<div style="display:flex;justify-content:space-between;width:100%;gap:10px">'
                    f"{gen_div}{gt_div}</div>"
                )
                if self.display_audio
                else ""
            )

            return (
                f'<div style="display:flex;flex-direction:column;align-items:center;padding:5px">'
                f"{audio_div}{spec_html}{wave_html}</div>"
            )
        else:
            gen_div = (
                (
                    f'<div style="width:100%;text-align:center">'
                    f'<p style="margin:2px;font-size:0.9em">Generated</p>'
                    f"{gen_audio_html}</div>"
                )
                if self.display_audio and has_gen
                else ""
            )

            return (
                f'<div style="display:flex;flex-direction:column;align-items:center;padding:5px">'
                f"{gen_div}{spec_html}{wave_html}</div>"
            )

    def save(self, width=120, height=30, subdir=None, skip_gt_audio=False):
        """Save the table to HTML file."""
        if not self.data:  # Don't save empty tables
            return

        # Create DataFrame
        audio_df = pd.DataFrame(columns=self.columns, index=self.data.keys())

        # Populate with audio HTML
        for row_name, col_dict in self.data.items():
            skip_audio = False
            audio_row = []
            for col_name in self.columns:
                audio_data = col_dict.get(col_name)
                if row_name == "GT" and skip_gt_audio:
                    skip_audio = True

                # Handle both tuple and single audio data formats
                if isinstance(audio_data, tuple):
                    gen_data, gt_data = audio_data
                    audio_html = self._get_audio_html(gen_data, gt_data, width, height)
                else:
                    # Fallback for old format or single audio data
                    audio_html = self._get_audio_html(audio_data, None, width, height)

                audio_row.append(audio_html)
            audio_df.loc[row_name] = audio_row

        # Convert to HTML
        html_table = audio_df.to_html(escape=False)
        html_table = html_table.replace("<th>", '<th style="text-align:center;">')

        # Add CSS for better layout
        css = """
        <style>
            table { 
                border-collapse: collapse; 
                width: 100%; 
            }
            th, td { 
                border: 1px solid #ddd; 
                text-align: center;
                vertical-align: middle;
                padding: 5px;
            }
            th { 
                background-color: #f4f4f4; 
                font-weight: bold;
                padding: 8px;
            }
            audio { 
                margin: 2px 0;
            }
            img {
                margin: 2px 0;
                object-fit: cover;
            }
        </style>
        """
        html_table = css + html_table

        # Save
        with open(self.path(subdir), "w") as f:
            f.write(html_table)

    def display(self):
        """Display the table in a Jupyter notebook."""
        if os.path.exists(self.path):
            with open(self.path) as f:
                display(HTML(f.read()))

    def delete(self):
        """Delete the HTML table file."""
        if os.path.exists(self.path):
            os.remove(self.path)

    def merge_table(self, other_table_path):
        """Merge another table into this one."""
        if not os.path.exists(other_table_path):
            return

        with open(other_table_path) as f:
            soup = BeautifulSoup(f, "lxml")
            other_table = soup.find("table")

        if os.path.exists(self.path):
            with open(self.path) as f:
                soup = BeautifulSoup(f, "lxml")
                table = soup.find("table")

            # Append rows from other table (skip header)
            for row in other_table.find_all("tr")[1:]:
                table.append(row)

            with open(self.path, "w") as f:
                f.write(str(soup))
        else:
            # If this table doesn't exist yet, just copy the other table
            with open(self.path, "w") as f:
                f.write(str(other_table))

    def add_row(self, row_name, column_dict):
        """Add a new row to the table.

        Args:
            row_name (str): Name of the row
            column_dict (dict): Dictionary mapping column names to audio data
        """
        if row_name not in self.data:
            self.data[row_name] = defaultdict(lambda: None)

        for col_name, audio_data in column_dict.items():
            if col_name.startswith("Generated_"):
                class_name = col_name.replace("Generated_", "")
                gt_col_name = f"GT_{class_name}"
                gt_data = column_dict.get(gt_col_name)
                self.data[row_name][class_name] = (audio_data, gt_data)
                if class_name not in self.columns:
                    self.columns.append(class_name)
            elif col_name.startswith("GT_"):
                continue
            else:
                # raise ValueError("Column names must start with 'Generated_'")
                # Handle any other type of data (like envelopes)
                # print("Adding other data")
                class_name = col_name
                self.data[row_name][class_name] = (audio_data, None)
                # Store both generated and GT data together
                if class_name not in self.columns:
                    self.columns.append(class_name)

    def save(self, width=120, height=30, subdir=None, skip_gt_audio=False):
        """Save the table to HTML file."""
        if not self.data:  # Don't save empty tables
            return

        # Create DataFrame
        audio_df = pd.DataFrame(columns=self.columns, index=self.data.keys())

        # Populate with audio HTML
        for row_name, col_dict in self.data.items():
            skip_audio = False
            audio_row = []
            for col_name in self.columns:
                audio_data = col_dict.get(col_name)
                if row_name == "GT" and skip_gt_audio:
                    skip_audio = True

                if isinstance(audio_data, tuple):
                    gen_data, gt_data = audio_data
                    audio_html = self._get_audio_html(gen_data, gt_data, width, height, skip_audio)
                else:
                    audio_html = self._get_audio_html(audio_data, None, width, height, skip_audio)

                audio_row.append(audio_html)
            audio_df.loc[row_name] = audio_row

        # Convert to HTML and clean up the newlines
        html_table = audio_df.to_html(escape=False)
        html_table = html_table.replace("<th>", '<th style="text-align:center;">')
        html_table = html_table.replace("\n", "")

        # Add CSS for better layout (single line)
        css = "<style>table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;text-align:center;vertical-align:middle;padding:5px}th{background-color:#f4f4f4;font-weight:bold;padding:8px}audio{margin:2px 0}img{margin:2px 0;object-fit:cover}</style>"

        html_table = css + html_table

        # Clean up any remaining multiple spaces
        html_table = " ".join(html_table.split())

        # Save
        with open(self.path(subdir), "w") as f:
            f.write(html_table)

    def display(self):
        """Display the table in a Jupyter notebook."""
        if os.path.exists(self.path):
            with open(self.path) as f:
                display(HTML(f.read()))

    def delete(self):
        """Delete the HTML table file."""
        if os.path.exists(self.path):
            os.remove(self.path)

    def merge_table(self, other_table_path):
        """Merge another table into this one."""
        if not os.path.exists(other_table_path):
            return

        with open(other_table_path) as f:
            soup = BeautifulSoup(f, "lxml")
            other_table = soup.find("table")

        if os.path.exists(self.path):
            with open(self.path) as f:
                soup = BeautifulSoup(f, "lxml")
                table = soup.find("table")

            # Append rows from other table (skip header)
            for row in other_table.find_all("tr")[1:]:
                table.append(row)

            with open(self.path, "w") as f:
                f.write(str(soup))
        else:
            # If this table doesn't exist yet, just copy the other table
            with open(self.path, "w") as f:
                f.write(str(other_table))


class LogAudioCallback(BaseCallback):
    def __init__(
        self,
        wav_folder="wav",
        sr=None,
        log_reconstructed=False,
        log_samples=False,
        log_reference=False,
        log_stems=False,
        log_gt_samples=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wav_folder = wav_folder
        self.sr = sr
        self.log_stems = log_stems
        self.log_reconstructed = log_reconstructed
        self.log_samples = log_samples
        self.log_reference = log_reference
        self.log_gt_samples = log_gt_samples

        # Track which files we've logged ground truth for in the current epoch
        self.epoch_gt_files = set()

        # Track the current epoch being processed
        self.current_epoch = None

        # Tables will be initialized in on_validation_start
        self.current_reconstructions = None
        self.all_reconstructions = None
        self.current_samples = None
        self.all_samples = None
        # self.inverse_class_mapping = None
        self.stems_table = None

    def on_validation_start(self, trainer, pl_module):
        self.wav_folder = os.path.join(pl_module.output_dir, "audio")
        os.makedirs(self.wav_folder, exist_ok=True)
        os.makedirs(os.path.join(self.wav_folder, "html"), exist_ok=True)

        # self.inverse_class_mapping = {v: k for k, v in pl_module.class_mapping.items()}

        # Initialize tables if they don't exist
        if self.current_reconstructions is None:
            self.current_reconstructions = AudioTable(
                os.path.join(self.wav_folder, "html"),
                name="reconstructions",
                sr=pl_module.sampling_rate,
                include_spectrograms=True,
                figsize=(14, 4),
            )
            self.all_reconstructions = AudioTable(
                os.path.join(self.wav_folder, "html"),
                name="all_reconstructions",
                sr=pl_module.sampling_rate,
            )
            self.current_samples = AudioTable(
                os.path.join(self.wav_folder, "html"),
                name="samples",
                sr=pl_module.sampling_rate,
                include_spectrograms=True,
            )
            self.all_samples = AudioTable(
                os.path.join(self.wav_folder, "html"),
                name="all_samples",
                sr=pl_module.sampling_rate,
            )
        self.stems_table = AudioTable(
            os.path.join(self.wav_folder, "html"),
            name=f"stems_epoch_{pl_module.current_epoch}",
            sr=pl_module.sampling_rate,
            include_spectrograms=True,
        )

        # Check if we're starting a new epoch
        if self.current_epoch != pl_module.current_epoch:
            self.current_epoch = pl_module.current_epoch
            # self.epoch_gt_files.clear()  # Reset ground truth tracking for new epoch
            self.current_reconstructions.clear()
            self.current_samples.clear()

    def on_test_start(self, trainer, pl_module):
        self.on_validation_start(trainer, pl_module)

    def _shared_step(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0, stage="val"
    ):
        input_mix = batch["mix"][0]
        synthesized_mix = outputs["output"][0]
        samples = outputs["samples"]
        fn = batch["audio_fn"][0]

        is_new_gt = fn not in self.epoch_gt_files
        reconstruction_data = OrderedDict()
        stem_data = OrderedDict()

        if self.log_reference:
            # Store the ground truth mix
            reconstruction_data["GT"] = {"Generated_mix": input_mix}
            if is_new_gt:
                log_audio_logger(
                    pl_module, cpu_numpy(input_mix), f"{stage}_gt/batch_idx_{batch_idx}/mix/"
                )
            self.epoch_gt_files.add(fn)

        if self.log_reconstructed:
            synth_audio = synthesized_mix
            if synth_audio.ndim == 2:
                audio_dict = defaultdict(lambda: torch.tensor(0))
                for i, _clas in enumerate(pl_module.train_classes):
                    # eval_class = self.inverse_class_mapping[_clas]
                    # Add "Generated_" prefix to keys
                    audio_dict[f"Generated_{_clas}"] = synth_audio[i]
                reconstruction_data[pl_module.current_epoch] = audio_dict
            else:
                reconstruction_data[pl_module.current_epoch] = {"Generated_mix": synth_audio}
                log_audio_logger(
                    pl_module, synth_audio, f"{stage}/batch_idx_{batch_idx}/synthesized/"
                )

        if self.log_stems:
            stems = outputs.get("stems")
            if stems is not None:
                stems = stems[0]
                stem_dict = {"Generated_mix": synthesized_mix, "GT_mix": input_mix}
                stem_dict.update(
                    {
                        "Generated_" + clas: stems[i]
                        for i, clas in enumerate(pl_module.train_classes)
                    }
                )
                self.stems_table.add_row(fn, stem_dict)

        # Add reconstructions directly to tables
        for row_name, column_dict in reconstruction_data.items():
            self.current_reconstructions.add_row(row_name, column_dict)
            self.all_reconstructions.add_row(row_name, column_dict)

        # Handle samples
        samples_data = OrderedDict()

        if self.log_gt_samples and batch["gt_sources"] is not None:
            gt_sources = batch["gt_sources"]
            gt_sources = {k: v[0] for k, v in gt_sources.items()}
            samples_data["GT"] = {}
            for key, source in gt_sources.items():
                # Add "GT_" prefix to keys
                samples_data["GT"][f"GT_{key}"] = source
            self.epoch_gt_files.add(fn)
            if is_new_gt:
                concat_sources = cpu_numpy(torch.cat(list(gt_sources.values()), dim=-1))
                log_audio_logger(
                    pl_module, concat_sources, f"{stage}_gt/batch_idx_{batch_idx}/gt_samples/"
                )

        if self.log_samples:
            source_waveform = outputs["samples"][0]
            samples_data[pl_module.current_epoch] = {}
            for i in range(source_waveform.shape[0]):
                clas = pl_module.train_classes[i]
                # Add "Generated_" prefix to keys
                samples_data[pl_module.current_epoch][f"Generated_{clas}"] = source_waveform[i]

            concat_samples = np.hstack(
                [cpu_numpy(source_waveform[i]) for i in range(source_waveform.shape[0])]
            )
            log_audio_logger(pl_module, concat_samples, f"{stage}/batch_idx_{batch_idx}/samples/")

            # Add samples directly to tables
            for row_name, column_dict in samples_data.items():
                self.current_samples.add_row(row_name, column_dict)
                self.all_samples.add_row(row_name, column_dict)

        # Save all tables
        if not self.log_stems and self.log_reconstructed:
            self.current_reconstructions.save(subdir=f"{batch_idx}", skip_gt_audio=not is_new_gt)
            self.all_reconstructions.save(subdir=f"{batch_idx}")
            # Save html to logger
            log_html_logger(
                pl_module,
                self.current_reconstructions.path(batch_idx),
                f"{stage}/batch_idx_{batch_idx}/reconstructions/",
            )
        if self.log_samples:
            self.current_samples.save(subdir=f"{batch_idx}")
            self.all_samples.save(subdir=f"{batch_idx}")

            log_html_logger(
                pl_module,
                self.current_samples.path(batch_idx),
                f"{stage}/batch_idx_{batch_idx}/samples/",
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._check(None, epoch=trainer.current_epoch):
            return
        return self._shared_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        return self._shared_epoch_end(pl_module, "test")

    def _shared_epoch_end(self, pl_module, stage):
        # We will save log stems table
        self.stems_table.save(subdir=f"{stage}_stems")
        if self.log_stems:
            log_html_logger(pl_module, self.stems_table.path(f"{stage}_stems"), f"{stage}/stems/")
