# from https://github.com/jorshi/drumblender
import torch
import torch.nn as nn
from einops import rearrange


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer. Takes an embedding -- usually shared between layers -- and applies a linear
    transformation to get the affine parameters of the FiLM transformation.

    Args:
        film_embedding_size (int): Size of the FiLM embedding.
        input_channels (int): Number of input channels.
        use_batch_norm (bool, optional): Whether to use batch normalization.
            Defaults to True.
    """

    def __init__(self, film_embedding_size: int, input_channels: int, use_batch_norm: bool = True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.norm = nn.BatchNorm1d(input_channels, affine=False)
        self.net = nn.Linear(film_embedding_size, input_channels * 2)
        # TODO init with identity

    def forward(self, x: torch.Tensor, film_embedding: torch.Tensor):
        film = self.net(film_embedding)
        gamma, beta = film.chunk(2, dim=-1)
        if self.use_batch_norm:
            x = self.norm(x)
        return gamma[..., None] * x + beta[..., None]


class TFiLM(nn.Module):
    """Temporal Feature-wise Linear Modulation layer.

    Derives affine parameters from a decimated version of the input signal, and applies them to the input. Allows the
    model to learn longer-range temporal dependencies.
    """

    def __init__(self, channels: int, block_size: int):
        super().__init__()
        self.block_size = block_size

        self.pool = nn.MaxPool1d(block_size)
        self.block_size = block_size

        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            num_layers=1,
        )
        self.proj = nn.Linear(channels, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, length = x.shape
        n_blocks = length // self.block_size
        assert n_blocks > 0, "Input length must be greater than block size."
        assert length == n_blocks * self.block_size, "Input length must be divisible by block size."

        x_decimated = self.pool(x)
        x_decimated = rearrange(x_decimated, "b c t -> t b c")

        affine, _ = self.lstm(x_decimated)
        affine = self.proj(affine)
        affine = rearrange(affine, "t b c -> b c t 1")
        gamma, beta = affine.chunk(2, dim=1)

        x = rearrange(x, "b c (n k) -> b c n k", k=self.block_size)
        x = gamma * x + beta
        x = rearrange(x, "b c n k -> b c (n k)")

        return x
