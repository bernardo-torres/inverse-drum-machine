from pathlib import Path
from typing import Any

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from idm import split_to_filter_map

# Assuming the dataset class from the previous refactor is available
from idm.data.dataset import MixOrMultitrackDataset


def get_dataset(
    dataset_split: str,
    sample_rate_target: int,
    version: str,
    root_path: str = "",
    dataset_root: str = None,
    annotation_root: str | Path | None = None,
    mix_metadata_file: str | Path | None = None,
    multitrack_metadata_file: str | Path | None = None,
    normalize: bool = False,
    mono: bool = True,
    gt_sources_path: str | None = None,
    duration: int = -1,
    **kwargs,
) -> MixOrMultitrackDataset:
    """Creates and configures the evaluation dataset based on the specified split."""
    # Using a dictionary mapping is a safe refactor that improves readability

    if dataset_split not in split_to_filter_map:
        raise ValueError(
            f"Unknown dataset split '{dataset_split}'. Valid options: {list(split_to_filter_map.keys())}"
        )

    filter_audio_fn = split_to_filter_map[dataset_split]

    root_path = Path(root_path) if root_path else Path.cwd()
    data_root = Path(dataset_root)
    mix_metadata_file = (
        data_root / "stemgmd_mapping.csv" if mix_metadata_file is None else mix_metadata_file
    )
    metadata_df = pd.read_csv(mix_metadata_file, header=0)
    multitrack_metadata_file = (
        data_root / "stemgmd_separation_mapping.csv"
        if multitrack_metadata_file is None
        else multitrack_metadata_file
    )
    multitrack_metadata_df = pd.read_csv(multitrack_metadata_file, header=0)

    split_map = {"eval_session": "test", "val": "validation", "test": "test", "train": "train"}
    split_key = next((k for k in split_map if k in dataset_split), None)
    if split_key is None:
        raise ValueError(f"Split type could not be determined from '{dataset_split}'")

    metadata_subset = metadata_df[metadata_df["split"] == split_map[split_key]]
    on_train = "train" in split_key
    if not on_train:
        # For validation and test, only use the official eval sessions
        metadata_subset.sort_values("duration", inplace=True)
    return MixOrMultitrackDataset(
        name=f"stemgmd_{dataset_split}",
        normalizing_function="maxabs",
        sample_rate_target=sample_rate_target,
        version=version,
        duration=duration,
        mono=mono,
        random_crop=on_train,
        dataset_root=str(data_root),
        annotation_root=data_root / "annotations" if annotation_root is None else annotation_root,
        mix_metadata=metadata_subset,
        multitrack_metadata=multitrack_metadata_df,
        filter_audio_fn=filter_audio_fn,
        get_gt_sources=(
            Path(gt_sources_path) if gt_sources_path else "data_drum_sources/Stem_GMD_single_hits"
        ),
        # cache_dir=str(root_path / "cache"),
        cache_hdf5=False,
        normalize=normalize,
        **kwargs,
    )


class DataModule(LightningDataModule):
    """
    LightningDataModule for loading, splitting, and serving drum audio data.

    This module handles predefined data splits, dataset instantiation with
    complex configurations (including caching and dynamic mixing), and provides
    dataloaders for training, validation, and testing stages.
    """

    def __init__(
        self,
        # Dataset source and structure
        dataset_class: type = MixOrMultitrackDataset,  # Replace with actual dataset class
        mix_metadata_file: str | list[str] | None = None,
        multitrack_metadata_file: str | list[str] | None = None,
        dataset_root: str = "",
        annotation_root: str = "",
        name: str = "dataset",
        version: str = "full",
        # Data splitting and partitioning
        split_strategy: str | list[float] = "predefined",
        train_split: str = "train_train_kits",
        val_split: str = "val_train_kits",
        test_split: str = "test_train_kits",
        val_dataset: Dataset | None = None,  # Optional override for validation dataset
        test_dataset: Dataset | None = None,  # Optional override for test dataset
        cache_partitions: list[str] = ["train", "val", "test"],
        # Audio processing
        sample_rate_target: int = 44100,
        duration: float | None = None,
        random_crop: bool = False,
        normalizing_function: str = "maxabs",
        mono: bool = True,
        backend: str = "soundfile",
        # Advanced features
        dynamic_mixing: bool = False,
        dynamic_mixing_classes: list[str] | None = None,
        get_gt_sources: str | None = None,
        return_stems: bool = True,
        # Dataloader settings
        batch_size: int = 32,
        batch_size_val: int = 1,
        num_workers: int = 0,
        num_workers_val: int = 0,
        shuffle: bool = True,  # Shuffle training data
        # Caching
        cache_hdf5: bool = False,
        keep_in_memory: bool = False,
        cache_dir: str = "cache",
        # Miscellaneous/debugging
        # filter_audio_fn: str | list | dict | None = None,
        dataset_size_multiplier: int = 1,
        skip_audio_loading: bool = False,
        **dataloader_kwargs,
    ):
        super().__init__()
        # Automatically save all __init__ arguments to self.hparams
        self.save_hyperparameters()

        # Placeholders for the datasets
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = self.hparams.val_dataset
        self.test_dataset: Dataset | None = self.hparams.test_dataset
        self.dataloader_kwargs = dataloader_kwargs

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        """
        Loads metadata and splits the data into train, validation, and test sets.
        This method is called by PyTorch Lightning on of these.
        """
        # We only need to set up datasets for 'fit' stage
        # if stage not in ("fit", None):
        #     return

        split_strategy = self.hparams.split_strategy
        if split_strategy == "predefined":
            self._setup_predefined_split()
        elif split_strategy == "overfit" or split_strategy == "debug":
            self._setup_overfit_split()
        else:
            raise ValueError(f"Split strategy '{split_strategy}' not supported.")

    def _get_dataset_kwargs(self) -> dict[str, Any]:
        """Constructs the shared keyword arguments for dataset instantiation."""
        # Define keys to be passed from hparams to the dataset
        hparam_keys = [
            # "dataset_root",
            # "annotation_root",
            "backend",
            # "filter_audio_fn",
            "dataset_size_multiplier",
            "version",
            # "get_gt_sources",
            "skip_audio_loading",
            "sample_rate_target",
            # "normalizing_function",
            # "mono",
            "dynamic_mixing",
            "dynamic_mixing_classes",
            "return_stems",
            "keep_in_memory",
            "cache_dir",
            "duration",
        ]
        return {key: self.hparams[key] for key in hparam_keys}

    def _setup_predefined_split(self) -> None:
        """Configures train/validation split from a 'split' column in metadata."""
        if not self.hparams.mix_metadata_file:
            raise ValueError("mix_metadata_file is required for 'predefined' split.")

        ds_kwargs = self._get_dataset_kwargs()

        self.train_dataset = get_dataset(
            dataset_split=self.hparams.train_split,
            dataset_root=self.hparams.dataset_root,
            **ds_kwargs,
        )
        self.val_dataset = get_dataset(
            dataset_split=self.hparams.val_split,
            dataset_root=self.hparams.dataset_root,
            **ds_kwargs,
        )

        self.test_dataset = get_dataset(
            dataset_split=self.hparams.test_split,
            dataset_root=self.hparams.dataset_root,
            **ds_kwargs,
        )

        # Verify no data leakage
        train_files = set(self.train_dataset.mix_metadata[:, 1])
        val_files = self.val_dataset.mix_metadata[:, 1]
        test_files = self.test_dataset.mix_metadata[:, 1]
        if train_files.intersection(val_files):
            raise RuntimeError("Overlap detected between train and validation sets.")
        if train_files.intersection(test_files):
            raise RuntimeError("Overlap detected between train and test sets.")
        if set(val_files).intersection(test_files):
            raise RuntimeError("Overlap detected between validation and test sets.")

    def _setup_overfit_split(self) -> None:
        """Configures train and validation sets to use the same data for overfitting checks."""
        # Use a small subset of the training data for both
        if not self.hparams.mix_metadata_file:
            raise ValueError("mix_metadata_file is required for 'overfit' split.")

        ds_kwargs = self._get_dataset_kwargs()

        self.train_dataset = get_dataset(
            dataset_split=self.hparams.train_split,
            dataset_root=self.hparams.dataset_root,
            mix_metadata_file=self.hparams.mix_metadata_file,
            **ds_kwargs,
        )
        self.train_dataset.mix_metadata = self.train_dataset.mix_metadata[: self.hparams.batch_size]
        self.val_dataset = self.train_dataset
        self.test_dataset = self.train_dataset

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
            collate_fn=unified_custom_collate,
            drop_last=True,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_val,
            num_workers=self.hparams.num_workers_val,
            shuffle=False,
            collate_fn=unified_custom_collate,
            drop_last=False,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_val,
            num_workers=self.hparams.num_workers_val,
            shuffle=False,
            collate_fn=unified_custom_collate,
            drop_last=False,
            **self.dataloader_kwargs,
        )


def unified_custom_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function to handle batches of dictionaries with complex types.
    - Stacks tensors using default_collate.
    - Handles `onsets_dict` by creating padded lists of tensors.
    - Preserves lists of strings.
    - Handles `None` values by returning `None` for that key in the batch.
    """
    if not batch:
        return {}

    first_item = batch[0]
    collated_batch = {}

    for key in first_item.keys():
        items = [d.get(key) for d in batch]

        # If all items for this key are None, the batched item is None
        if all(item is None for item in items):
            collated_batch[key] = None
            continue

        # Special handling for `onsets_dict`
        if key == "onsets_dict":
            all_onset_keys = set()
            for d in items:
                if d:
                    all_onset_keys.update(d.keys())

            collated_dict = {
                onset_key: [
                    torch.tensor(item.get(onset_key, []), dtype=torch.float32) for item in items
                ]
                for onset_key in sorted(list(all_onset_keys))
            }
            collated_batch[key] = collated_dict

        # Handle other dictionary types
        elif isinstance(first_item[key], dict):
            sub_keys = first_item[key].keys()
            collated_batch[key] = {
                sub_key: default_collate([d[key][sub_key] for d in batch]) for sub_key in sub_keys
            }

        # Handle lists of strings (e.g., file paths)
        elif isinstance(first_item[key], list) and all(isinstance(s, str) for s in first_item[key]):
            collated_batch[key] = items

        # Default case: use PyTorch's default collate
        else:
            # Filter out Nones before collating, as default_collate doesn't handle them
            valid_items = [item for item in items if item is not None]
            if not valid_items:
                collated_batch[key] = None
            else:
                try:
                    collated_batch[key] = default_collate(valid_items)
                except TypeError:
                    # Fallback for non-tensorable types like lists of strings
                    collated_batch[key] = items

    return collated_batch
