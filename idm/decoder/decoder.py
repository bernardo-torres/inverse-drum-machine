from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from idm.components import IdentityWithKwargs
from idm.synthesis_conditioning.embedding import get_conditioning_vector
from idm.synthesis_conditioning.peak_picking import PeakPicking
from idm.utils import get_act_functional


class Decoder(nn.Module):
    """Decodes latent representations into a final audio waveform.

    This module takes activations and embeddings, processes them to generate onsets,
    synthesizes the audio samples, and
    sequences them into a coherent audio output. It can optionally apply
    post-processing effects.

    Attributes:
        peak_picking: A module to detect peaks in the onset activation signals.
        sample_synth: A module that synthesizes a short audio sample from an embedding.
        sequencer: A module that arranges synthesized samples onto a timeline based on onsets.
        processors: An optional module for applying post-synthesis effects (e.g., filters, gains).
        sampling_rate: The audio sampling rate in Hz.
    """

    def __init__(
        self,
        peak_picking: nn.Module = IdentityWithKwargs(),
        sample_synth: nn.Module = nn.Identity(),
        sequencer: nn.Module = nn.Identity(),
        processors: nn.Module | None = None,
        sampling_rate: int = 16000,
        sample_duration: float = 1.0,
        activation: str = "sigmoid",
    ):
        """Initializes the Decoder module.

        Args:
            peak_picking: The module for detecting peaks in onset activations.
            sample_synth: The module for synthesizing audio samples.
            sequencer: The module for sequencing samples into a track.
            processors: Optional post-processing module.
            sampling_rate: The target audio sampling rate.
            sample_duration: The duration of synthesized samples in seconds.
            activation: The name of the activation function to apply to onset logits.
        """
        super().__init__()
        self.peak_picking = peak_picking
        self.peak_picking_val = PeakPicking()  # Separate peak picking for validation
        self.sample_synth = sample_synth
        self.sequencer = sequencer
        self.processors = processors if processors is not None else nn.Identity()

        # Determine if mixing should happen after sequencing. If processors are
        # present, they operate on stems, so mixing is done after.
        self.mix_after_seq = isinstance(self.processors, nn.Identity)

        self.sampling_rate = sampling_rate
        self.sample_duration_samples = int(sample_duration * sampling_rate)
        self.activation = get_act_functional(activation)
        self.delay_compensation = 0

    def forward(
        self,
        activations: dict[str, torch.Tensor],
        embeddings: torch.Tensor | None = None,
        frame_features: torch.Tensor | None = None,
        extra_returns: list[str] | None = None,
        apply_sigmoid: bool = True,
        activation_rate: int | None = None,
        override_onsets: torch.Tensor | None = None,
        override_samples: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Generates audio from activations and embeddings. Both activations and audio samples
        can be optionally overridden.

        Args:
            activations: A dictionary containing at least 'onset' and 'velocity' tensors.
                         Shape: (B, K, M) where K is the number of instruments.
            embeddings: Latent representations for sample synthesis. Shape: (B, K, D).
            frame_features: Additional features for post-processing modules.
            extra_returns: A list of strings specifying extra tensors to return,
                           e.g., ["stems", "processors"].
            apply_sigmoid: If True, applies the sigmoid activation to onset logits.
            activation_rate: The rate of the activation signal, used for validation peak picking.
            override_onsets: An optional tensor to use directly as onset events,
                               bypassing the internal peak picking logic.
            override_samples: An optional tensor of pre-synthesized samples to use.

        Returns:
            A dictionary containing at least the final mixed 'output', the
            synthesized 'samples', and the processed 'onsets'. May also include
            'stems' or processor parameters if requested.
        """
        extra_returns = extra_returns or []
        returns = {}
        return_stems = "stems" in extra_returns

        # Select the appropriate peak picking function based on training/validation mode.
        if self.training:
            pick_peaks = self.peak_picking
        else:
            pick_peaks = partial(self.peak_picking_val, activation_rate=activation_rate)

        # Use ground truth onsets if provided
        if override_onsets is not None:
            onsets = override_onsets
            # When using GT onsets, they are not logits, so we skip activation.
            onsets = pick_peaks(self.activation(onsets)) if apply_sigmoid else pick_peaks(onsets)
        else:
            onset_logits = activations["onset"]
            onsets = self.activation(onset_logits) if apply_sigmoid else onset_logits
            onsets = pick_peaks(onsets)

        B, K, M = onsets.shape
        # Synthesize one-shot samples unless they are provided externally.
        if override_samples is not None:
            samples = override_samples
        else:
            # We synthesize samples for all instruments in the batch at once.
            if embeddings.ndim == 2:
                embeddings = get_conditioning_vector(
                    n_classes=K,
                    embedding=embeddings,
                    batch_size=B,
                )
            embeddings_flat = rearrange(embeddings, "b k d -> (b k) d")
            noise = torch.randn(
                embeddings_flat.size(0), 1, self.sample_duration_samples, device=embeddings.device
            )
            # Synthesize samples for all K instruments in the batch.
            samples_flat = self.sample_synth(noise, embeddings_flat)
            samples = rearrange(samples_flat, "(b k) 1 t -> b k t", b=onsets.shape[0])

        # Modulate onsets by velocity.
        a = onsets * activations["velocity"]

        # Place samples on the timeline. If no processors are used and stems are not
        # requested, the sequencer can directly mix the output.
        output_stems = self.sequencer(
            a,
            sources=samples,
            mix=(self.mix_after_seq and not return_stems),
            delay_compensation=self.delay_compensation,
        )

        # Apply post-processing if modules are defined.
        if not isinstance(self.processors, nn.Identity):
            processed_output = self.processors(output_stems, frame_features)
            if "processors" in extra_returns:
                returns.update(self.processors.get_params(frame_features))
        else:
            processed_output = output_stems

        # Mix the stems to get the final output if not already mixed.
        if not self.mix_after_seq or return_stems:
            if return_stems:
                returns["stems"] = processed_output
            output = processed_output.sum(dim=1)
        else:
            output = processed_output

        returns.update({"output": output, "samples": samples, "onsets": onsets})
        return returns
