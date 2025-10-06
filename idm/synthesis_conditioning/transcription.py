from typing import Dict

import torch
from einops import rearrange
from torch import nn

from idm.utils import get_act_functional, get_act_module


class TranscriptionHead(nn.Module):
    """Predicts musical events from high-level features.

    This module takes a sequence of feature frames and predicts two parallel
    outputs for each frame and instrument class:
    1.  **Onset**: The probability of a note onset (a binary event).
    2.  **Velocity**: The intensity or volume of the note.

    It uses a combination of convolutions, an optional recurrent sequence model,
    and MLPs to generate these predictions.

    Attributes:
        sequence_model: A recurrent model (e.g., GRU) to capture temporal dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        kernel_size: int = 3,
        sequence_model: nn.Module = nn.Identity(),
        activation: str = "elu",
        onset_activation: str = "sigmoid",
        velocity_activation: str = "relu",
    ):
        """Initializes the TranscriptionHead module.

        Args:
            input_dim: The feature dimension of the input tensor.
            n_classes: The number of output classes (e.g., instrument types).
            kernel_size: The kernel size for the initial convolutional layer.
            sequence_model: An optional recurrent layer (e.g., nn.GRU) for
                            modeling temporal context.
            activation: The main activation function for intermediate layers.
            onset_activation: The activation function for the final onset predictions.
            velocity_activation: The activation function for the final velocity predictions.
        """
        super().__init__()

        # Initial convolution to process input features
        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size, stride=1, padding="same")
        self.activation = get_act_module(activation)()

        self.sequence_model = sequence_model
        # Determine the feature dimension after the sequence model
        recurrent_multiplier = 1
        if isinstance(sequence_model, nn.GRU):
            recurrent_multiplier = 2 if sequence_model.bidirectional else 1
        recurrent_dim = input_dim * recurrent_multiplier

        # Prediction Heads
        self.velocity_mlp = nn.Linear(input_dim, n_classes)
        self.onset_mlp = nn.Linear(recurrent_dim, n_classes)

        # Final activation functions for the outputs
        self.onset_act = get_act_functional(onset_activation)
        self.velocity_act = get_act_functional(velocity_activation)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Processes input features to predict onsets and velocities.

        Args:
            x: The input feature tensor. Shape: `(B, M, D_z)`.

        Returns:
            A dictionary containing:
            - `onset`: The predicted onset activations. Shape: `(B, K, M)`.
            - `velocity`: The predicted velocity values. Shape: `(B, K, M)`.
        """
        # (B, M, D_z) -> (B, D_z, M) for 1D convolution
        x = x.permute(0, 2, 1)
        x = self.conv(x).permute(
            0, 2, 1
        )  # (B, D_z, M) -> (B, M, D_z) for linear layers and sequence model
        x = self.activation(x)

        # Predict velocity from the pre-sequence features
        velocity = self.velocity_mlp(x)

        # Apply the sequence model to capture temporal context
        if not isinstance(self.sequence_model, nn.Identity):
            x, _ = self.sequence_model(x)

        onset = self.onset_mlp(x)

        # Apply final activations and reshape to (B, K, M)
        onset = self.onset_act(onset)
        onset = rearrange(onset, "b t c -> b c t")

        velocity = self.velocity_act(velocity)
        velocity = rearrange(velocity, "b t c -> b c t")

        return {"onset": onset, "velocity": velocity}
