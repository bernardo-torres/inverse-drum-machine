from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F


class BaseEncoder(nn.Module):
    """Abstract base class for encoders.

    This class provides a common structure for encoders, including handling an
    input transformation (e.g., STFT, Mel spectrogram) and calculating the
    resulting feature frame rate.
    """

    def __init__(self, transform: Optional[nn.Module] = None, sampling_rate: Optional[int] = None):
        """Initializes the BaseEncoder.

        Args:
            transform: An optional nn.Module to apply to the input waveform
                before the main encoder backbone.
            sampling_rate: The sampling rate of the input audio in Hz.
        """
        super().__init__()
        self.transform = transform if transform is not None else nn.Identity()
        self.sampling_rate = sampling_rate
        self._frame_rate = None  # Cached frame rate value
        self.downsampling_factor = None

    @property
    def frame_rate(self) -> Optional[float]:
        """Calculates and returns the frame rate of the encoder's output features.

        The frame rate is determined by the downsampling performed by the input
        transform and the encoder's backbone network. The value is cached after
        the first calculation.

        Returns:
            The output frame rate in Hz, or raises a ValueError if it cannot be determined.
        """
        if self._frame_rate is None:
            self._compute_frame_rate()
        return self._frame_rate

    def _compute_frame_rate(self):
        """Internal method to calculate the frame rate based on module properties."""
        if self._frame_rate is not None:
            return

        # Attempt to get frame rate from the backbone module
        if hasattr(self, "backbone"):
            if hasattr(self.backbone, "frame_rate"):
                self._frame_rate = self.backbone.frame_rate
                return
            if hasattr(self.backbone, "downsampling_factor"):
                self.downsampling_factor = self.backbone.downsampling_factor

        # Calculate frame rate from downsampling factor if available
        if self.downsampling_factor is not None:
            self._frame_rate = self.sampling_rate / self.downsampling_factor
            return

        # Attempt to get frame rate from the transform module
        if not isinstance(self.transform, nn.Identity) and hasattr(
            self.transform, "get_frame_rate"
        ):
            self._frame_rate = self.transform.get_frame_rate(self.sampling_rate)
            return

        raise ValueError(
            "Could not determine encoder frame rate. Please set it manually, "
            "provide a downsampling factor, or use a transform with a 'get_frame_rate' method."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Abstract forward method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method.")


class Encoder(BaseEncoder):
    """A comprehensive encoder for audio processing.

    This module processes a raw audio waveform through a series of stages:
    1.  **Transform**: Converts the waveform into a time-frequency representation.
    2.  **Backbone**: A deep neural network that extracts high-level features.
    3.  **Pooling**: Aggregates frame-level features into a global embedding (e.g., for classification).
    4.  **Transcription Head**: Predicts frame-wise activations or events.

    This module groups the feature extraction and synthesis conditioning into a
    single encoder model. Parameters for processors applied after decoding are not
    included here, as they are typically self-contained within the processor modules.
    """

    def __init__(
        self,
        backbone: nn.Module = nn.Identity(),
        pooling: nn.Module = nn.Identity(),
        transcription_head: Optional[nn.Module] = None,
        transform: Optional[nn.Module] = None,
        sampling_rate: Optional[int] = None,
        embedding_norm: Optional[nn.Module] = None,
        embedding_one_hot: bool = False,
    ):
        """Initializes the Encoder.

        Args:
            backbone: The main feature extraction network.
            pooling: A module to pool frame features into a single embedding.
            transcription_head: A module to predict activations from frame features.
            transform: The input waveform-to-spectrogram transformation.
            sampling_rate: The audio sampling rate in Hz.
            embedding_norm: An optional normalization/activation layer (e.g., Softmax)
                            applied to the final embedding.
            embedding_one_hot: If True, converts the embedding to a one-hot vector
                               during evaluation.
        """
        super().__init__(transform=transform, sampling_rate=sampling_rate)
        self.backbone = backbone
        self.pooling = pooling
        self.transcription_head = (
            transcription_head if transcription_head is not None else nn.Identity()
        )
        self.embedding_norm = embedding_norm
        self.embedding_one_hot = embedding_one_hot

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Performs the full forward pass of the encoder.

        Args:
            x: The input raw audio waveform. Shape: `(B, T_audio)`.

        Returns:
            A dictionary containing various outputs:
            - `frame_features`: Features from the backbone. Shape: `(B, D, T_feat)`.
            - `activations`: Output from the transcription head.
            - `embeddings`: The final pooled and normalized embedding. Shape: `(B, D_emb)`.
            - `embedding_logits`: The embedding before normalization.
            - `attention_weights`: Attention weights from the pooling mechanism, if available.
            - `activation_rate`: The frame rate of the features and activations.
        """
        logits = None

        # Feature extraction
        input_features = self.transform(x)
        frame_features = self.backbone(input_features)

        # Synthesis conditioning
        # 1.Pooling/mixture embedding
        embeddings, attention_weights = None, None
        if not isinstance(self.pooling, nn.Identity):
            pooling_output = self.pooling(frame_features)
            if isinstance(pooling_output, tuple) and len(pooling_output) == 2:
                embeddings, attention_weights = pooling_output
            else:
                embeddings = pooling_output
        if self.embedding_norm is not None and embeddings is not None:
            logits = embeddings
            embeddings = self.embedding_norm(embeddings)
            # 5. Optionally convert embedding to one-hot during evaluation
        if self.embedding_one_hot and not self.training and embeddings is not None:
            # Get the index of the max value and convert to a one-hot vector
            pred_class = embeddings.argmax(dim=1)
            embeddings = F.one_hot(pred_class, num_classes=embeddings.size(1)).to(embeddings.dtype)

        # 2. Transcription
        activations = self.transcription_head(frame_features)

        return {
            "frame_features": frame_features,
            "activations": activations,
            "embeddings": embeddings,
            "embedding_logits": logits,
            "attention_weights": attention_weights,
            "activation_rate": self.frame_rate,
        }
