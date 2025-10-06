from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

# Assuming the dataset class from the previous refactor is available
from idm.data.dataset import MixOrMultitrackDataset

# Placeholder for Hydra/OmegaConf config types
ListConfig = list


class DataModule(pl.LightningDataModule):
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
        mix_metadata_file: Optional[Union[str, List[str]]] = None,
        multitrack_metadata_file: Optional[Union[str, List[str]]] = None,
        dataset_root: str = "",
        annotation_root: str = "",
        name: str = "dataset",
        version: str = "full",
        # Data splitting and partitioning
        split: Union[str, List[float]] = "predefined",
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        cache_partitions: List[str] = ["train", "val", "test"],
        # Audio processing
        sample_rate_target: int = 44100,
        duration: Optional[float] = None,
        random_crop: bool = False,
        normalizing_function: str = "maxabs",
        mono: bool = True,
        backend: str = "soundfile",
        # Advanced features
        dynamic_mixing: bool = False,
        dynamic_mixing_classes: Optional[List[str]] = None,
        get_gt_sources: Optional[str] = None,
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
        filter_audio_fn: Optional[Union[str, list, dict]] = None,
        dataset_size_multiplier: int = 1,
        skip_audio_loading: bool = False,
        **dataloader_kwargs,
    ):
        super().__init__()
        # Automatically save all __init__ arguments to self.hparams
        self.save_hyperparameters()

        # Placeholders for the datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = self.hparams.val_dataset
        self.test_dataset: Optional[Dataset] = self.hparams.test_dataset

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads metadata and splits the data into train, validation, and test sets.
        This method is called by PyTorch Lightning on of these.
        """
        # We only need to set up datasets for 'fit' stage
        if stage not in ("fit", None):
            return

        split_strategy = self.hparams.split
        if split_strategy == "predefined":
            self._setup_predefined_split()
        elif split_strategy == "overfit":
            self._setup_overfit_split()
        elif split_strategy in ["random", "none"] or isinstance(
            split_strategy, (list, tuple, ListConfig)
        ):
            # Implement other split strategies or use pre-assigned datasets
            pass
        else:
            raise ValueError(f"Split strategy '{split_strategy}' not supported.")

    def _get_dataset_kwargs(self) -> Dict[str, Any]:
        """Constructs the shared keyword arguments for dataset instantiation."""
        # Define keys to be passed from hparams to the dataset
        hparam_keys = [
            "dataset_root",
            "annotation_root",
            "backend",
            "filter_audio_fn",
            "dataset_size_multiplier",
            "version",
            "get_gt_sources",
            "skip_audio_loading",
            "sample_rate_target",
            "normalizing_function",
            "mono",
            "dynamic_mixing",
            "dynamic_mixing_classes",
            "return_stems",
            "keep_in_memory",
            "cache_dir",
        ]
        return {key: self.hparams[key] for key in hparam_keys}

    def _setup_predefined_split(self) -> None:
        """Configures train/validation split from a 'split' column in metadata."""
        if not self.hparams.mix_metadata_file:
            raise ValueError("mix_metadata_file is required for 'predefined' split.")

        full_metadata = pd.read_csv(self.hparams.mix_metadata_file, header=0)
        multitrack_metadata = (
            pd.read_csv(self.hparams.multitrack_metadata_file, header=0)
            if self.hparams.multitrack_metadata_file
            else None
        )

        # Filter metadata for train and validation sets
        train_meta = full_metadata[full_metadata["split"] == "train"].copy()
        val_meta = full_metadata[full_metadata["split"] == "validation"].copy()

        # Verify no data leakage
        train_files = set(train_meta["Full Audio Path"])
        val_files = set(val_meta["Full Audio Path"])
        if train_files.intersection(val_files):
            raise RuntimeError("Overlap detected between train and validation sets.")

        val_meta = val_meta.sort_values("duration")
        ds_kwargs = self._get_dataset_kwargs()

        self.train_dataset = self.hparams.dataset_class(
            name=f"{self.hparams.name}_train",
            mix_metadata=train_meta,
            multitrack_metadata=multitrack_metadata,
            random_crop=self.hparams.random_crop,
            duration=self.hparams.duration,
            cache_hdf5="train" in self.hparams.cache_partitions and self.hparams.cache_hdf5,
            **ds_kwargs,
        )

        self.val_dataset = self.hparams.dataset_class(
            name=f"{self.hparams.name}_val",
            mix_metadata=val_meta,
            multitrack_metadata=multitrack_metadata,
            random_crop=False,  # Validation should be deterministic
            duration=None,  # Typically evaluate on full files
            cache_hdf5="val" in self.hparams.cache_partitions and self.hparams.cache_hdf5,
            **ds_kwargs,
        )

    def _setup_overfit_split(self) -> None:
        """Configures train and validation sets to use the same data for overfitting checks."""
        # Use a small subset of the training data for both
        if not self.hparams.mix_metadata_file:
            raise ValueError("mix_metadata_file is required for 'overfit' split.")

        full_metadata = pd.read_csv(self.hparams.mix_metadata_file, header=0)
        overfit_meta = full_metadata.head(self.hparams.batch_size)  # Use one batch

        ds_kwargs = self._get_dataset_kwargs()

        # Instantiate the same dataset for both train and val
        overfit_dataset = self.hparams.dataset_class(
            name=f"{self.hparams.name}_overfit",
            mix_metadata=overfit_meta,
            random_crop=self.hparams.random_crop,
            duration=self.hparams.duration,
            cache_hdf5=False,  # Caching is likely not beneficial for overfitting
            **ds_kwargs,
        )
        self.train_dataset = overfit_dataset
        self.val_dataset = overfit_dataset

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        if self.samples_dataset_train:
            self.train_dataset.set_audio_samples_dataset(self.samples_dataset_train)

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
            collate_fn=unified_custom_collate,
            drop_last=True,
            **self.hparams.dataloader_kwargs,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_val,
            num_workers=self.hparams.num_workers_val,
            shuffle=False,
            collate_fn=unified_custom_collate,
            drop_last=False,
            **self.hparams.dataloader_kwargs,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_val,
            num_workers=self.hparams.num_workers_val,
            shuffle=False,
            collate_fn=unified_custom_collate,
            drop_last=False,
            **self.hparams.dataloader_kwargs,
        )


def unified_custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
