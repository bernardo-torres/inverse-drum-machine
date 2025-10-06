import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from sklearn.metrics import accuracy_score

from idm.decoder.decoder import Decoder
from idm.feature_extractor.encoder import BaseEncoder
from idm.run_utils import RankedLogger
from idm.utils import convert_onset_dict_to_activations, cpu_numpy

log = RankedLogger(__name__, rank_zero_only=True)


class BaseTrainer(LightningModule):
    """A base trainer class for PyTorch Lightning models.

    This class provides a basic structure for training, including optimizer and
    scheduler configuration, and a default media folder setup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module = torch.nn.L1Loss(),
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        monitor: str = "val/loss",
        sampling_rate: int = 44100,
        output_dir: str = "results",
        **kwargs,
    ):
        """Initializes the BaseTrainer.

        Args:
            optimizer: The optimizer class to use for training.
            loss: The loss function module.
            scheduler: An optional learning rate scheduler.
            monitor: The metric to monitor for the learning rate scheduler.
            sampling_rate: The audio sampling rate.
            output_dir: The directory to save media files (images, audio).
        """
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss
        self.monitor = monitor
        self.sampling_rate = sampling_rate
        self.output_dir = output_dir

    def _setup_media_folder(self):
        self.img_folder = os.path.join(self.output_dir, "images")
        os.makedirs(self.img_folder, exist_ok=True)

    def on_fit_start(self):
        """Called at the beginning of training to set up media folders."""
        self._setup_media_folder()

    def on_test_start(self):
        """Called at the beginning of testing to set up media folders."""
        self._setup_media_folder()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is None:
            return optimizer
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.monitor}


class InverseDrumMachine(BaseTrainer):
    """The main LightningModule for the Inverse Drum Machine model.

    This module integrates an encoder-decoder architecture for joint drum
    transcription and synthesis. It handles the complete training, validation,
    and testing logic, including loss calculations and metric logging.
    """

    def __init__(
        self,
        encoder: BaseEncoder = None,
        decoder: Decoder = None,
        loss: nn.Module = None,
        transcription_loss: nn.Module = None,
        embedding_loss: nn.Module = None,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        augmentations: nn.Module = nn.Identity(),
        n_drum_kits: int = 6,
        version: str = "full",
        sampling_rate: int = 44100,
        train_classes: Optional[List[str]] = None,
        use_target_embeddings: bool = True,
        gt_sequencing: bool = False,
        smooth_targets: bool = False,
        fix_gt_sources: bool = False,
        embedding_size: int = 15,  # not used but kept for backwards compatibility
        **kwargs,
    ):
        super().__init__(optimizer=optimizer, scheduler=scheduler, loss=loss, **kwargs)
        # Core model components
        self.encoder = encoder
        self.decoder = decoder
        self.augmentations = augmentations
        self.transcription_loss = transcription_loss
        self.embedding_loss = embedding_loss
        self.emedding_size = embedding_size

        self.train_classes = sorted(train_classes) if train_classes is not None else None
        if self.train_classes is None:
            switcher = {"basic": 5, "minimal": 3, "full": 9}
            self.n_classes = switcher.get(version, 5)
        else:
            self.n_classes = len(self.train_classes)

        # Experimental configuration parameters
        self.n_classes = len(self.train_classes)
        self.n_drum_kits = n_drum_kits
        self.sampling_rate = sampling_rate
        self.conditioning_size = self.n_classes + self.n_drum_kits
        self.register_buffer("conditioning_vector", torch.zeros(self.conditioning_size))

        # Training flags
        self.use_target_embeddings = use_target_embeddings
        self.gt_sequencing = gt_sequencing
        self.smooth_targets = smooth_targets
        self.fix_gt_sources = fix_gt_sources

        # Validation Metrics
        self.kit_preds: List[int] = []
        self.kit_targets: List[int] = []

    def forward(self, mix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Performs a full forward pass through the encoder and decoder."""
        encoder_outs = self.encoder(mix)
        return self.decoder(**encoder_outs, extra_returns="stems")

    def get_conditioning(self, *args, **kwargs):
        return self._get_conditioning_vector(*args, **kwargs)

    def _get_conditioning_vector(
        self,
        batch_size: int = 1,
        drum_kit_ids: torch.Tensor = None,
        embedding: Optional[torch.Tensor] = None,
        kit_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Constructs the full conditioning vector for the decoder.

        This vector combines a one-hot encoding of the instrument class with
        either a one-hot or predicted embedding for the drum kit.

        Args:
            batch_size: The current batch size.
            drum_kit_ids: The ground truth drum kit indices. Shape: `(B,)`.
            embedding: The predicted drum kit embedding from the encoder.
                                 Shape: `(B, n_drum_kits)`.
            kit_embedding: For backwards compatibility, same as `embedding`.

        Returns:
            The combined conditioning vector. Shape: `(B, K, D_cond)`.
        """
        # Start with one-hot vectors for the instrument classes
        # Shape: (B, K, K) where K is n_classes
        class_one_hot = torch.eye(self.n_classes, device=self.device).expand(batch_size, -1, -1)
        embedding = embedding if embedding is not None else kit_embedding

        if embedding is None:
            # Use one-hot encoding for the drum kit
            kit_one_hot = F.one_hot(drum_kit_ids, num_classes=self.n_drum_kits).float()
            # Expand to match the shape for concatenation: (B, D_kit) -> (B, K, D_kit)
            kit_conditioning = kit_one_hot.unsqueeze(1).expand(-1, self.n_classes, -1)
        else:
            # Use the predicted embedding from the encoder
            # Expand to match shape: (B, D_kit) -> (B, K, D_kit)
            kit_conditioning = embedding.unsqueeze(1).expand(-1, self.n_classes, -1)

        # Concatenate class and kit conditioning vectors
        return torch.cat([class_one_hot, kit_conditioning], dim=-1)

    def _shared_step(self, batch: Dict, stage: str) -> Dict:
        """Performs a shared step for training, validation, and testing."""
        mix = batch["mix"]
        drum_kit_ids = batch["drum_kit"]
        batch_size = mix.shape[0]

        if stage == "train":
            mix = self.augmentations(mix)

        # Encoder
        encoder_outs = self.encoder(mix)
        predicted_kit_embedding = encoder_outs["embeddings"]
        embedding_logits = encoder_outs["embedding_logits"]
        predicted_activations = encoder_outs["activations"]
        onset_logits = predicted_activations["onset"]

        if stage == "train":
            # Prep ground truth onsets
            gt_onsets = convert_onset_dict_to_activations(
                batch["onsets_dict"],
                n_frames=onset_logits.shape[-1],
                instrument_classes=self.train_classes,
                activation_rate=self.encoder.frame_rate,
            )
            if self.smooth_targets:
                # Apply smoothing by adding scaled, shifted versions of the onsets
                gt_onsets = (
                    gt_onsets
                    + torch.roll(gt_onsets, shifts=1, dims=-1) * 0.5
                    + torch.roll(gt_onsets, shifts=-1, dims=-1) * 0.5
                )
                gt_onsets = torch.clamp(gt_onsets, 0, 1)  # Ensure values stay in [0, 1]

            # Decoder inputs
            decoder_inputs = encoder_outs.copy()
            if self.use_target_embeddings:
                decoder_inputs["embeddings"] = self._get_conditioning_vector(
                    batch_size, drum_kit_ids
                )
            else:
                decoder_inputs["embeddings"] = self._get_conditioning_vector(
                    batch_size, drum_kit_ids, predicted_kit_embedding
                )

            # We use gt onsets for conditioning during training
            if self.gt_sequencing:
                decoder_inputs["override_onsets"] = gt_onsets

            if self.fix_gt_sources:
                # For debugging, use ground truth audio samples directly
                samples = torch.stack(
                    [batch["gt_sources"][inst] for inst in self.train_classes], dim=1
                )
                decoder_inputs["override_samples"] = samples

        # Decoder
        decoder_outs = self.decoder(
            **decoder_inputs,
            extra_returns=["stems"] if stage != "train" else [],
        )
        output = decoder_outs["output"][..., : mix.shape[-1]]  # Ensure same length as input

        # Transcription Loss
        transcription_loss = self.transcription_loss(onset_logits, gt_onsets) * 10
        # Mixture Embedding Loss
        embedding_loss = self.embedding_loss(embedding_logits, drum_kit_ids)
        # Reconstruction Loss
        reconstruction_loss = self.train_loss(output, mix)

        total_loss = reconstruction_loss + transcription_loss + embedding_loss

        self.log(f"{stage}/loss", total_loss, batch_size=batch_size)
        self.log(f"{stage}/reconstruction_loss", reconstruction_loss, batch_size=batch_size)
        self.log(
            f"{stage}/transcription_loss", transcription_loss, prog_bar=True, batch_size=batch_size
        )
        self.log(f"{stage}/embedding_loss", embedding_loss, prog_bar=True, batch_size=batch_size)

        returns = {
            "loss": total_loss,
            "transcription_loss": transcription_loss,
            "reconstruction_loss": reconstruction_loss,
            "embedding_loss": embedding_loss,
        }
        if stage != "train":
            self.kit_preds.extend(embedding_logits.argmax(dim=-1).cpu().tolist())
            self.kit_targets.extend(drum_kit_ids.cpu().tolist())
            returns.update(
                {
                    "output": output.detach().cpu(),
                    "stems": decoder_outs.get("stems", torch.Tensor()).detach().cpu(),
                }
            )
        return returns

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")["loss"]

    def validation_step(self, batch: Dict, batch_idx: int):
        outs = self._shared_step(batch, "val")
        embeddings = cpu_numpy(outs["embeddings"].reshape(-1, outs["embeddings"].shape[-1]))
        target_embeddings = cpu_numpy(
            outs["target_embeddings"].reshape(-1, outs["target_embeddings"].shape[-1])
        )
        pred_classes = embeddings[:, self.n_classes :].argmax(axis=1)
        target_classes = target_embeddings[:, self.n_classes :].argmax(axis=1)
        self.kit_preds.extend(pred_classes.tolist())
        self.kit_targets.extend(target_classes.tolist())

    def on_validation_epoch_end(self):
        """Calculates and logs metrics at the end of the validation epoch."""
        if self.kit_preds:
            acc = accuracy_score(self.kit_targets, self.kit_preds)
            self.log("val/embedding_accuracy", acc, prog_bar=True)
            self.kit_preds.clear()
            self.kit_targets.clear()

    def test_step(self, batch: Dict, batch_idx: int):
        """The test step for the model."""
        self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        """Calculates and logs metrics at the end of the test epoch."""
        if self.kit_preds:
            acc = accuracy_score(self.kit_targets, self.kit_preds)
            self.log("test/embedding_accuracy", acc, prog_bar=True)
            self.kit_preds.clear()
            self.kit_targets.clear()
