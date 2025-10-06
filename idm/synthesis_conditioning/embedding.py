import torch
import torch.nn as nn
from einops import rearrange


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
