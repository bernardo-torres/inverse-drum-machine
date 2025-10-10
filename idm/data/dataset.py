import contextlib
import csv
import os
import sys
import warnings
from collections import defaultdict
from typing import Any

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

# Assumed external dependencies from the original script
from idm import (
    # drum_kit_map_egmd,
    drum_kit_map_stemgmd,
    model_version_to_train_class_list,
    stem_gmd_drum_class_to_symbol,
    stem_gmd_single_hits_map,
)
from idm.run_utils import RankedLogger
from idm.utils import (
    get_metadata_torchaudio,
    get_normalizing_function,
    load_audio,
)

log = RankedLogger(__name__, rank_zero_only=True)


@contextlib.contextmanager
def with_audio_settings(dataset, **settings):
    """Context manager that temporarily changes any audio settings of a dataset.

    Args:
        dataset: The dataset object that has load_audio_kwargs attribute
        **settings: Any number of keyword arguments to temporarily set
                   (e.g., sample_rate_target=44000, mono=True)

    Yields:
        The dataset with temporarily modified audio settings
    """
    # Initialize backup dictionary
    original_settings = {}

    # Check if load_audio_kwargs exists
    has_kwargs = hasattr(dataset, "_load_audio_kwargs")

    if has_kwargs and settings:
        # Backup and change each specified setting
        for key, value in settings.items():
            if key in dataset._load_audio_kwargs:
                # Only backup if the setting exists
                original_settings[key] = dataset._load_audio_kwargs[key]
                # Apply the new setting
                dataset._load_audio_kwargs[key] = value

    try:
        # Yield control back to the with block
        yield dataset
    finally:
        # Restore all original settings
        if has_kwargs:
            for key, value in original_settings.items():
                dataset._load_audio_kwargs[key] = value


def _invert_map(mapping: dict[Any, Any]) -> dict[Any, Any]:
    """Inverts a dictionary's key-value pairs."""
    return {v: k for k, v in mapping.items()}


def _filter_dataframe_by_path(
    data: pd.DataFrame,
    filter_patterns: str | list | dict,
    key: str = "Full Audio Path",
) -> pd.DataFrame:
    """
    Filters a DataFrame based on string patterns in a specified column.

    Args:
        data: The pandas DataFrame to filter.
        filter_patterns: A string, list of strings, or dict of patterns to apply.
        key: The DataFrame column to apply the filter on.

    Returns:
        The filtered pandas DataFrame.
    """
    if filter_patterns is None:
        return data

    if isinstance(filter_patterns, str):
        return data[data[key].str.contains(filter_patterns, na=False)]
    if isinstance(filter_patterns, (list, ListConfig)):
        for pattern in filter_patterns:
            data = data[data[key].str.contains(pattern, na=False)]
        return data
    if isinstance(filter_patterns, (dict, DictConfig)):
        for pattern, action in filter_patterns.items():
            if action == "remove":
                data = data[~data[key].str.contains(pattern, na=False)]
            elif action == "keep":
                data = data[data[key].str.contains(pattern, na=False)]
            elif action is not None:
                raise ValueError("Filter action must be 'remove' or 'keep'.")
        return data
    raise TypeError("filter_audio_fn must be a string, list, or dict.")


def load_drum_events(filename):
    """
    Loads drum event data from a two-column, tab-delimited file.

    Args:
        filename (str): Path to the annotation file.

    Returns:
        tuple: A tuple containing:
            - list: A list of event times (floats).
            - list: A list of event labels (strings).
    """
    onsets_times = []
    onsets_labels = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            onsets_times.append(float(row[0]))
            onsets_labels.append(row[1])
    return onsets_times, onsets_labels


class MixOrMultitrackDataset(Dataset):
    """
    Dataset for loading audio mixtures and/or their corresponding stems.

    This class handles loading from metadata files, dynamic mixing of stems,
    annotation loading, and an optional HDF5-based caching system for fast
    data retrieval.

    Dynamic mixing allows combining individual stems (e.g., 'HH_OHH', 'HH_CHH')
    into a single evaluation class ('HH') on the fly, based on a class mapping.
    """

    def __init__(
        self,
        # Data source parameters
        mix_metadata_file: str | None = None,
        multitrack_metadata_file: str | None = None,
        mix_metadata: pd.DataFrame | None = None,
        multitrack_metadata: pd.DataFrame | None = None,
        dataset_root: str = "",
        annotation_root: str = "",
        # Audio processing parameters
        sample_rate_target: int = 44100,
        duration: float | None = None,
        mono: bool = True,
        normalize: bool = True,
        normalizing_function: str = "maxabs",
        random_crop: bool = False,
        backend: str = "torchaudio",
        # Class and annotation parameters
        version: str = "full",
        label_map: dict[str, str] | None = None,
        eval_classes: list[str] | None = None,
        # Dynamic mixing and source separation parameters
        return_stems: bool = True,
        dynamic_mixing: bool = False,
        dynamic_mixing_classes: list[str] | None = None,
        get_gt_sources: bool | str = False,
        sample_duration: float = 1.0,
        # Caching parameters
        cache_hdf5: bool = False,
        cache_path: str | None = None,
        cache_dir: str = "cache",
        cache_multitrack: bool = False,
        optimize_silent_tracks: bool = False,
        keep_in_memory: bool = False,
        # Miscellaneous
        name: str | None = None,
        filter_audio_fn: str | list | dict | None = None,
        dataset_size_multiplier: int = 1,
        skip_audio_loading: bool = False,
        **load_audio_kwargs,
    ):
        """
        Initializes the dataset, loads metadata, and sets up caching.
        """
        self.dataset_root = dataset_root
        self.annotation_root = annotation_root
        self.sample_rate_target = sample_rate_target
        self.label_map = label_map
        self.filter_audio_fn = filter_audio_fn
        self.duration = duration
        self.random_crop = random_crop
        self.dataset_size_multiplier = dataset_size_multiplier
        self.normalize = normalize
        self.normalizing_function = normalizing_function
        self.mono = mono
        self.name = name if name is not None else "dataset"
        self.return_stems = return_stems
        self.dynamic_mixing = dynamic_mixing
        self.dynamic_mixing_classes = dynamic_mixing_classes
        self.get_gt_sources = get_gt_sources
        self.sample_duration = sample_duration
        self.skip_audio_loading = skip_audio_loading
        self.eval_classes = eval_classes
        self.gt_sources: dict[Any, dict[str, torch.Tensor]] = {}

        self._load_audio_kwargs = {
            "sample_rate_target": self.sample_rate_target,
            "mono": self.mono,
            "normalize": self.normalize,
            "normalizing_function": self.normalizing_function,
            "backend": backend,
            "duration": self.duration,
            **load_audio_kwargs,
        }

        # Caching attributes
        self.cache_hdf5 = cache_hdf5
        self.cache_dir = cache_dir
        self.cache_path = cache_path
        self.use_cached_data = False

        self._load_and_filter_metadata(
            mix_metadata, mix_metadata_file, multitrack_metadata, multitrack_metadata_file
        )
        self._initialize_class_mappings(version)

        if self.cache_hdf5 and not self.skip_audio_loading:
            self._initialize_caching(cache_multitrack, optimize_silent_tracks, keep_in_memory)

    def _load_and_filter_metadata(self, mix_meta, mix_meta_file, multi_meta, multi_meta_file):
        """Loads, validates, and filters mix and multitrack metadata."""

        # Load multitrack metadata
        self.multitrack_metadata_df = None
        if multi_meta is not None:
            self.multitrack_metadata_df = multi_meta
        elif multi_meta_file is not None:
            self.multitrack_metadata_df = pd.read_csv(multi_meta_file, header=0)

        # Load mix metadata
        if mix_meta is not None:
            self.mix_metadata_df = mix_meta
        elif mix_meta_file is not None:
            self.mix_metadata_df = pd.read_csv(mix_meta_file, header=0)
        elif multi_meta_file is not None:
            # Derive mix metadata from multitrack metadata
            multi_df = pd.read_csv(multi_meta_file, header=0)
            self.mix_metadata_df = pd.DataFrame(
                {
                    "Full Audio Path": multi_df["Mixture File"],
                    "Audio File": multi_df["Mixture File"].apply(os.path.basename),
                }
            )
        else:
            raise ValueError("Either mix or multitrack metadata must be provided.")

        # Apply filters on the mixture audio paths if specified (eg for train/val/test splits)
        self.mix_metadata_df = _filter_dataframe_by_path(
            self.mix_metadata_df, self.filter_audio_fn, "Full Audio Path"
        ).reset_index(drop=True)
        if self.multitrack_metadata_df is not None:
            self.multitrack_metadata_df = _filter_dataframe_by_path(
                self.multitrack_metadata_df, self.filter_audio_fn, "Mixture File"
            ).reset_index(drop=True)

        # Convert to numpy and create column index maps for performance
        self.mix_metadata = self.mix_metadata_df.to_numpy()
        self.col_indices = {"mix": self._create_column_index_map(self.mix_metadata_df.columns)}

        self._multitrack_path_to_idx_map = {}
        if self.multitrack_metadata_df is not None:
            self.multitrack_metadata = self.multitrack_metadata_df.to_numpy()
            self.col_indices["multitrack"] = self._create_column_index_map(
                self.multitrack_metadata_df.columns
            )
            # Create a map for O(1) lookup of multitrack rows by mixture path
            mixture_file_col = self.col_indices["multitrack"]["Mixture File"]
            self._multitrack_path_to_idx_map = {
                path: i for i, path in enumerate(self.multitrack_metadata[:, mixture_file_col])
            }

        # Cache annotation file paths
        self.cached_annotations = []
        if "Annotation Path" in self.col_indices["mix"]:
            annot_path_idx = self.col_indices["mix"]["Annotation Path"]
            for row in self.mix_metadata:
                annot_path = os.path.join(self.annotation_root, row[annot_path_idx])
                self.cached_annotations.append(annot_path if os.path.exists(annot_path) else None)
        else:
            self.cached_annotations = [None] * len(self.mix_metadata)

    def _create_column_index_map(self, columns: pd.Index) -> dict[str, int]:
        """Creates a mapping from column name to its integer index."""
        col_map = {name: i for i, name in enumerate(columns)}
        if "stemgmd" in self.name.lower():
            # Apply name remapping for stemgmd compatibility
            for new_name, old_name in stem_gmd_single_hits_map.items():
                if old_name in col_map:
                    col_map[new_name] = col_map[old_name]
        return col_map

    def _initialize_class_mappings(self, version: str):
        """Sets up mappings for training classes, evaluation classes, and dynamic mixing."""
        class_mapping = model_version_to_train_class_list.get(version, {})
        self.merge_mapping = {k: k for k in class_mapping}
        self.all_possible_classes = np.array(list(set(self.merge_mapping.values())))

        if self.multitrack_metadata_df is not None:
            if self.dynamic_mixing and self.dynamic_mixing_classes:
                self.all_possible_classes = np.array(self.dynamic_mixing_classes)
                # Filter the merge mapping to only include classes for dynamic mixing
                self.merge_mapping = {
                    k: v for k, v in self.merge_mapping.items() if v in self.all_possible_classes
                }
                # Normalizing during load can be problematic with silent stems
                if self._load_audio_kwargs["normalize"]:
                    self._load_audio_kwargs["normalize"] = False
                    warnings.warn(
                        "Dynamic mixing is enabled. Disabling normalization during audio "
                        "loading to avoid issues with silent stems."
                    )
            self.inverse_merge_mapping = _invert_map(self.merge_mapping)

        self.all_possible_classes = np.sort(self.all_possible_classes)

    def _initialize_caching(
        self, cache_multitrack: bool, optimize_silent: bool, keep_in_memory: bool
    ):
        raise NotImplementedError("Caching functionality is not implemented in this snippet.")

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.mix_metadata) * self.dataset_size_multiplier

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Retrieves a single data item from the dataset.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A dictionary containing the mixture audio, stems, annotations, and metadata.
        """
        # Account for dataset_size_multiplier
        original_idx = idx % len(self.mix_metadata)

        mix_row = self.mix_metadata[original_idx]
        rel_path = mix_row[self.col_indices["mix"]["Full Audio Path"]]
        audio_file = os.path.join(self.dataset_root, rel_path)

        # Determine drum kit
        drum_kit_name = mix_row[self.col_indices["mix"]["drum_kit"]]
        if "stemgmd" in self.name.lower():
            drum_kit = drum_kit_map_stemgmd.get(drum_kit_name, drum_kit_name)
        elif "egmd" in self.name.lower():
            drum_kit = drum_kit_map_egmd.get(drum_kit_name, drum_kit_name)
        else:
            drum_kit = drum_kit_name

        # Determine random offset if needed, but don't load audio yet
        offset_sec = (
            self._get_random_frame_offset(audio_file, original_idx) if self.random_crop else 0.0
        )

        # Load annotations
        annotation_file = self.cached_annotations[original_idx]
        onsets_dict = self._load_annotations(annotation_file, offset_sec) if annotation_file else {}
        onsets_dict = onsets_dict or {}  # Ensure it's a dict

        # Load stems if required
        stems = None
        if (self.multitrack_metadata is not None) and (self.return_stems or self.dynamic_mixing):
            stems = self._get_item_multitrack(rel_path, offset_sec)

        # Load mixture audio
        if self.dynamic_mixing:
            if stems is None:
                raise ValueError("Stems must be available for dynamic mixing.")
            mixture_audio = sum(stems.values())
        else:
            # Random crop is False because offset_sec is already calculated
            mixture_audio = self._get_mixture_audio(
                audio_file, original_idx, offset_sec=offset_sec, random_crop=False
            )

        # Load ground truth source snippets if required
        gt_sources = self._get_gt_sources_for_item(stems, onsets_dict, drum_kit, drum_kit_name)

        # Filter onsets to only include classes relevant for training/evaluation
        final_onsets_dict = defaultdict(list)
        for class_name in self.all_possible_classes:
            final_onsets_dict[class_name] = onsets_dict.get(class_name, [])

        onsets_mask = torch.tensor(
            [len(final_onsets_dict.get(c, [])) > 0 for c in self.all_possible_classes],
            dtype=torch.bool,
        )

        return {
            "mix": mixture_audio,
            "drum_kit": drum_kit,
            "onsets_dict": final_onsets_dict,
            "onset_mask": onsets_mask,
            "audio_fn": os.path.basename(audio_file),
            "audio_file": audio_file,
            "all_possible_classes": list(self.all_possible_classes),
            "stems": stems if self.return_stems else None,
            "gt_sources": gt_sources,
        }

    def _get_random_frame_offset(self, audio_file: str, idx: int) -> float:
        """Calculates a random start time in seconds for cropping."""
        if self.skip_audio_loading or self.duration is None or self.duration <= 0:
            return 0.0

        if self.use_cached_data:
            start, end = self.track_boundaries[idx]["start"], self.track_boundaries[idx]["end"]
            track_len_frames = end - start
            if track_len_frames <= 0:
                return 0.0

            duration_frames = int(self.duration * self.sample_rate_target)
            if duration_frames >= track_len_frames:
                return 0.0

            frame_offset = np.random.randint(0, track_len_frames - duration_frames)
            return frame_offset / self.sample_rate_target

        # Fallback to reading metadata from the audio file directly
        try:
            _, frame_offset_native, _, sr_native, _ = get_metadata_torchaudio(
                audio_file, self.duration, random_crop=True, offset_sec=0
            )
            return frame_offset_native / sr_native
        except Exception as e:
            log.warning(f"Could not get metadata for {audio_file}, returning 0 offset: {e}")
            return 0.0

    def _get_mixture_audio(self, mixture_file: str, idx: int, **kwargs) -> torch.Tensor | None:
        """Loads mixture audio, using cache if available."""
        if self.skip_audio_loading:
            return None

        if self.use_cached_data:
            # This logic assumes a single contiguous block of audio in HDF5
            bounds = self.track_boundaries[idx]
            start_frame = bounds["start"] + int(
                kwargs.get("offset_sec", 0) * self.sample_rate_target
            )
            end_frame = bounds["end"]
            if self.duration is not None and self.duration > 0:
                end_frame = min(
                    start_frame + int(self.duration * self.sample_rate_target), end_frame
                )

            if self.cached_audio is not None:  # In-memory cache
                audio = self.cached_audio[start_frame:end_frame].copy()
            else:  # On-disk HDF5 cache
                if not H5PY_AVAILABLE:
                    raise ImportError("h5py is required for HDF5 caching but is not installed.")
                with h5py.File(self.cache_path, "r") as hf:
                    audio = hf["audio"][start_frame:end_frame]

            # Pad if the loaded segment is shorter than the requested duration
            if self.duration and audio.shape[-1] < int(self.duration * self.sample_rate_target):
                pad_width = int(self.duration * self.sample_rate_target) - audio.shape[-1]
                audio = np.pad(
                    audio, ((0, 0), (0, pad_width)) if audio.ndim == 2 else (0, pad_width)
                )

            if self._load_audio_kwargs.get("normalize", False):
                norm_fn = get_normalizing_function(self.normalizing_function)
                audio = norm_fn(audio)

            return torch.from_numpy(audio).float()

        return self._load_audio(mixture_file, **kwargs)["audio"]

    def _load_audio(self, audio_file: str, **kwargs) -> dict[str, Any]:
        """Wrapper for the external load_audio utility."""
        final_kwargs = self._load_audio_kwargs.copy()
        final_kwargs.update(kwargs)
        try:
            audio, sr, frame_offset = load_audio(audio_file, **final_kwargs)
            return {"audio": audio, "frame_offset": frame_offset}
        except Exception as e:
            sys.stderr.write(f"ERROR reading {audio_file}: {e}\n")
            raise

    def _load_annotations(
        self, annotation_file: str, offset_seconds: float
    ) -> dict[str, np.ndarray]:
        """Loads and processes annotation files."""
        # This function assumes external utilities `read_annotations_multilabel`
        # and `rename_key` exist and function as in the original code.
        # onsets_tuple = mir_eval.io.load_labeled_events(annotation_file)
        # onsets_dict = read_annotations_multilabel(onsets_tuple)
        # onsets_dict = rename_key(onsets_dict, self.label_map)
        # onsets_times, onsets_labels = mir_eval.io.load_events(annotation_file)
        onsets_times, onsets_labels = load_drum_events(annotation_file)
        onsets_dict = defaultdict(list)
        for time, label in zip(onsets_times, onsets_labels):
            onsets_dict[label].append(time)

        processed_dict = {}
        target_duration = (
            self.duration if self.duration is not None and self.duration > 0 else float("inf")
        )
        for key, onsets in onsets_dict.items():
            onsets = np.array(onsets)
            mask = (onsets >= offset_seconds) & (onsets < offset_seconds + target_duration)
            processed_dict[key] = onsets[mask] - offset_seconds

        return processed_dict

    def _get_item_multitrack(
        self, mixture_rel_path: str, offset_sec: float
    ) -> dict[str, torch.Tensor] | None:
        """Retrieves all relevant stems for a given mixture file."""
        if self.skip_audio_loading:
            return None

        # Use pre-computed map for efficient O(1) lookup
        multitrack_idx = self._multitrack_path_to_idx_map.get(mixture_rel_path)

        if multitrack_idx is None:
            log.warning(f"Could not find {mixture_rel_path} in multitrack metadata.")
            return self._create_empty_stems()

        return self._load_multitrack_stems(multitrack_idx, offset_sec)

    def _load_multitrack_stems(self, idx: int, offset_sec: float) -> dict[str, torch.Tensor]:
        """Loads, combines, and returns stems for a single multitrack recording."""
        stems_audio: dict[str, torch.Tensor] = {}
        target_duration = int(self.duration * self.sample_rate_target) if self.duration else None

        for original_class, separation_class in self.merge_mapping.items():
            if separation_class not in self.all_possible_classes:
                continue

            filepath_val = self.multitrack_metadata[idx][
                self.col_indices["multitrack"][original_class]
            ]
            if pd.isna(filepath_val):
                continue

            stem_file = os.path.join(self.dataset_root, filepath_val)
            class_audio = self._load_audio(stem_file, random_crop=False, offset_sec=offset_sec)[
                "audio"
            ]

            if not stems_audio:  # First stem loaded determines expected shape
                if target_duration is None:
                    target_duration = class_audio.shape[-1]

            # Ensure consistent length
            if class_audio.shape[-1] != target_duration:
                class_audio = self._conform_audio_to_duration(
                    class_audio, target_duration, stem_file
                )

            if separation_class not in stems_audio:
                stems_audio[separation_class] = torch.zeros_like(class_audio)

            stems_audio[separation_class] += class_audio

        # Fill in any missing stems with silence
        if stems_audio and target_duration:
            for s_class in self.all_possible_classes:
                if s_class not in stems_audio:
                    shape = (target_duration,) if self.mono else (2, target_duration)
                    stems_audio[s_class] = torch.zeros(shape)
            return stems_audio

        return self._create_empty_stems()  # Return empty stems if none were found

    @staticmethod
    def _conform_audio_to_duration(
        audio: torch.Tensor, duration: int, filename: str
    ) -> torch.Tensor:
        """Pads or truncates audio to a target duration."""
        if audio.shape[-1] > duration:
            return audio[..., :duration]
        if audio.shape[-1] < duration:
            pad_shape = list(audio.shape)
            pad_shape[-1] = duration - audio.shape[-1]
            return torch.cat([audio, torch.zeros(pad_shape, dtype=audio.dtype)], dim=-1)
        return audio

    def _create_empty_stems(self) -> dict[str, torch.Tensor]:
        """Creates a dictionary of silent stems."""
        if self.duration is not None and self.duration > 0:
            duration_samples = int(self.duration * self.sample_rate_target)
        else:
            duration_samples = int(10 * self.sample_rate_target)  # Fallback

        shape = (duration_samples,) if self.mono else (2, duration_samples)
        return {c: torch.zeros(shape) for c in self.all_possible_classes}

    def _get_gt_sources_for_item(self, stems, onsets_dict, drum_kit, drum_kit_name):
        """Dispatcher for retrieving ground truth source snippets."""
        if not self.get_gt_sources:
            return None
        if self.get_gt_sources == "from_stems":
            if stems is None:
                raise ValueError("Stems must be available for 'from_stems' GT source extraction.")
            return self._extract_gt_sources_from_stems(stems, onsets_dict)
        if self.get_gt_sources == "from_files" or os.path.exists(str(self.get_gt_sources)):
            self._load_gt_sources_from_files(drum_kit, drum_kit_name)
            return self.gt_sources.get(drum_kit)
        return None

    def _extract_gt_sources_from_stems(self, stems_audio, onsets_dict):
        """Extracts single-hit audio samples from stems based on onset times."""
        gt_sources = {}
        sample_len = int(self.sample_duration * self.sample_rate_target)

        for class_name in self.all_possible_classes:
            onsets = onsets_dict.get(class_name)
            if not onsets or not isinstance(onsets, np.ndarray) or onsets.size == 0:
                continue

            source_stem = stems_audio.get(class_name)
            if source_stem is None:
                continue

            # Average all hits of a given class to get a representative sample
            summed_source = torch.zeros(sample_len, dtype=source_stem.dtype)
            hits_found = 0
            for onset_time in onsets:
                start_idx = int(onset_time * self.sample_rate_target)
                start_idx = max(0, start_idx - int(0.02 * self.sample_rate_target))  # Pre-roll
                end_idx = start_idx + sample_len

                if end_idx > source_stem.shape[-1]:
                    continue

                extracted_hit = source_stem[..., start_idx:end_idx]
                summed_source += extracted_hit
                hits_found += 1

            if hits_found > 0:
                gt_sources[class_name] = summed_source / hits_found

        return gt_sources

    def _load_gt_sources_from_files(self, drum_kit, drum_kit_name):
        """Loads single-hit audio samples from a directory structure."""
        if drum_kit in self.gt_sources:
            return  # Already loaded for this kit

        self.gt_sources[drum_kit] = {}
        source_dir = self.get_gt_sources

        for note_name, note_code in stem_gmd_drum_class_to_symbol.items():
            merged_code = self.merge_mapping.get(note_code)
            if merged_code is None or merged_code in self.gt_sources[drum_kit]:
                continue

            # Assumes a file naming convention like "kick-1.wav"
            filepath = os.path.join(source_dir, drum_kit_name, f"{note_name}-1.wav")
            if not os.path.exists(filepath):
                continue

            src = self._load_audio(
                filepath,
                duration=self.sample_duration,
                random_crop=False,
            )["audio"]
            self.gt_sources[drum_kit][merged_code] = src.squeeze()
