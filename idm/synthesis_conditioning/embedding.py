import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_conditioning_vector(
    batch_size: int = 1,
    n_classes: int = None,
    n_drum_kits: int = None,
    drum_kit_ids: torch.Tensor = None,
    embedding: torch.Tensor | None = None,
    kit_embedding: torch.Tensor | None = None,
    device: torch.device = torch.device("cuda"),
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

    embedding = embedding if embedding is not None else kit_embedding

    class_one_hot = torch.eye(n_classes, device=device).expand(batch_size, -1, -1)

    if embedding is None:
        if n_drum_kits is None:
            raise ValueError(
                "Either n_drum_kits or embedding must be provided for creating the conditioning vector."
            )
        # Use one-hot encoding for the drum kit
        kit_one_hot = F.one_hot(drum_kit_ids, num_classes=n_drum_kits).float()
        # Expand to match the shape for concatenation: (B, D_kit) -> (B, K, D_kit)
        kit_conditioning = kit_one_hot.unsqueeze(1).expand(-1, n_classes, -1)
    else:
        # Use the predicted embedding from the encoder
        # Expand to match shape: (B, D_kit) -> (B, K, D_kit)
        kit_conditioning = embedding.unsqueeze(1).expand(-1, n_classes, -1)

    # Concatenate class and kit conditioning vectors
    return torch.cat([class_one_hot, kit_conditioning], dim=-1)


class DrumKitClassifier(nn.Module):
    """Classifies a drum kit from a sequence of audio features.

    This module takes a sequence of feature frames, aggregates them over time,
    and passes them through a classifier to predict the drum kit ID. It is
    designed to operate on the output of an encoder backbone.
    """

    def __init__(self, in_features: int, n_drum_kits: int = 10):
        """Initializes the DrumKitClassifier.

        Args:
            in_features: The number of features in the input tensor.
            n_drum_kits: The number of possible drum kits to classify.
        """
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        # Pool features across the time dimension to get a single summary vector.
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, n_drum_kits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a feature sequence to predict a drum kit.

        Args:
            x: The input feature tensor of shape `(B, M, D_z)`, where B is the
               batch size, M is the sequence length, and D_z is `in_features`.

        Returns:
            The raw logit scores for each drum kit. Shape: `(B, n_drum_kits)`.
        """
        x = self.norm(x)
        x = rearrange(x, "b t c -> b c t")

        # (B, D_z, M) -> (B, D, 1) -> (B, D)
        x = self.pooling(x).squeeze(-1)

        x = self.classifier(x)
        return x
