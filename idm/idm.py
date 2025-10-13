import os

import torch
import torch.nn as nn
from lightning import LightningModule
from sklearn.metrics import accuracy_score

from idm.decoder.decoder import Decoder
from idm.feature_extractor.encoder import BaseEncoder
from idm.run_utils import RankedLogger
from idm.synthesis_conditioning.embedding import get_conditioning_vector
from idm.utils import convert_onset_dict_to_activations

log = RankedLogger(__name__, rank_zero_only=True)


class BaseTrainer(LightningModule):
    """A base trainer class for PyTorch Lightning models.

    This class provides a basic structure for training, including optimizer and
    scheduler configuration, and a default media folder setup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        monitor: str = "val/loss",
        sampling_rate: int = 44100,
        output_dir: str = "results",
        **kwargs,
    ):
        """Initializes the BaseTrainer.

        Args:
            optimizer: The optimizer class to use for training.
            scheduler: An optional learning rate scheduler.
            monitor: The metric to monitor for the learning rate scheduler.
            sampling_rate: The audio sampling rate.
            output_dir: The directory to save media files (images, audio).
        """
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        augmentations: nn.Module = nn.Identity(),
        n_drum_kits: int = 6,
        version: str = "full",
        sampling_rate: int = 44100,
        train_classes: list[str] | None = None,
        use_target_embeddings: bool = True,
        gt_sequencing: bool = False,
        smooth_targets: bool = False,
        fix_gt_sources: bool = False,
        embedding_size: int = 15,  # not used but kept for backwards compatibility
        **kwargs,
    ):
        super().__init__(optimizer=optimizer, scheduler=scheduler, **kwargs)
        # Core model components
        self.encoder = encoder
        self.decoder = decoder
        self.augmentations = augmentations
        self.transcription_loss = transcription_loss
        self.embedding_loss = embedding_loss
        self.emedding_size = embedding_size
        self.reconstruction_loss = loss

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
        self.kit_preds: list[int] = []
        self.kit_targets: list[int] = []

    def forward(self, mix: torch.Tensor) -> dict[str, torch.Tensor]:
        """Performs a full forward pass through the encoder and decoder."""
        encoder_outs = self.encoder(mix)
        return self.decoder(**encoder_outs, extra_returns="stems")

    def get_conditioning(self, *args, **kwargs):
        return get_conditioning_vector(
            n_classes=self.n_classes, n_drum_kits=self.n_drum_kits, device=self.device, **kwargs
        )

    def _shared_step(self, batch: dict, stage: str) -> dict:
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
        decoder_inputs = encoder_outs.copy()
        decoder_inputs["embeddings"] = get_conditioning_vector(
            batch_size=batch_size,
            n_classes=self.n_classes,
            n_drum_kits=self.n_drum_kits,
            drum_kit_ids=None,
            embedding=predicted_kit_embedding,
        )

        gt_onsets = convert_onset_dict_to_activations(
            batch["onsets_dict"],
            batch_size=batch_size,
            n_frames=onset_logits.shape[-1],
            instrument_classes=self.train_classes,
            activation_rate=self.encoder.frame_rate,
            device=self.device,
        )

        if stage == "train":
            # Prep ground truth onsets

            if self.smooth_targets:
                # Apply smoothing by adding scaled, shifted versions of the onsets
                gt_onsets = (
                    gt_onsets
                    + torch.roll(gt_onsets, shifts=1, dims=-1) * 0.5
                    + torch.roll(gt_onsets, shifts=-1, dims=-1) * 0.5
                )
                gt_onsets = torch.clamp(gt_onsets, 0, 1)  # Ensure values stay in [0, 1]

            # Decoder inputs

            if self.use_target_embeddings:
                # decoder_inputs["embeddings"] = self._get_conditioning_vector(
                #     batch_size, drum_kit_ids
                # )
                decoder_inputs["embeddings"] = get_conditioning_vector(
                    batch_size=batch_size,
                    n_classes=self.n_classes,
                    n_drum_kits=self.n_drum_kits,
                    drum_kit_ids=drum_kit_ids,
                    embedding=None,
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
        reconstruction_loss = self.reconstruction_loss(output, mix)

        total_loss = reconstruction_loss + transcription_loss + embedding_loss

        returns = {
            "loss": total_loss,
            "transcription_loss": transcription_loss,
            "reconstruction_loss": reconstruction_loss,
            "embedding_loss": embedding_loss,
            "batch_size": batch_size,
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

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        returns = self._shared_step(batch, "train")
        # Log step-level metrics (for progress monitoring)

        recons_loss = returns["reconstruction_loss"]
        trans_loss = returns["transcription_loss"]
        emb_loss = returns["embedding_loss"]
        total_loss = returns["loss"]
        batch_size = returns["batch_size"]
        self.log("train/loss", total_loss, batch_size=batch_size, on_step=True, on_epoch=False)

        self.log(
            "train/reconstruction_loss",
            recons_loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/transcription_loss",
            trans_loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/embedding_loss",
            emb_loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
        )

        # Log epoch-level metrics (for CustomFileLogger)
        self.log(
            "train/loss_epoch",
            total_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/reconstruction_loss_epoch",
            recons_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/transcription_loss_epoch",
            trans_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/embedding_loss_epoch",
            emb_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch: dict, batch_idx: int, stage: str = "val"):
        returns = self._shared_step(batch, stage)
        recons_loss = returns["reconstruction_loss"]
        trans_loss = returns["transcription_loss"]
        emb_loss = returns["embedding_loss"]
        total_loss = returns["loss"]
        batch_size = returns["batch_size"]
        self.log(
            f"{stage}/loss",
            total_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}/reconstruction_loss",
            recons_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}/transcription_loss",
            trans_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}/embedding_loss",
            emb_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        return returns

    def on_validation_epoch_end(self):
        """Calculates and logs metrics at the end of the validation epoch."""
        if self.kit_preds:
            acc = accuracy_score(self.kit_targets, self.kit_preds)
            self.log("val/embedding_accuracy", acc, prog_bar=True)
            self.kit_preds.clear()
            self.kit_targets.clear()

    def test_step(self, batch: dict, batch_idx: int):
        """The test step for the model."""
        return self.validation_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self):
        """Calculates and logs metrics at the end of the test epoch."""
        if self.kit_preds:
            acc = accuracy_score(self.kit_targets, self.kit_preds)
            self.log("test/embedding_accuracy", acc, prog_bar=True)
            self.kit_preds.clear()
            self.kit_targets.clear()
