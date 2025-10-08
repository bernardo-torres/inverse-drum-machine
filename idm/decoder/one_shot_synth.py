# Mostly taken from https://github.com/jorshi/drumblender

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from idm.decoder.film import FiLM, TFiLM
from idm.utils import get_act_functional, resample

# def _get_activation(activation: str):
#     if activation == "gated":
#         return GatedActivation()
#     return getattr(nn, activation)()


class Pad(nn.Module):
    """Pad a tensor with zeros according to causal or non-causal 1D padding scheme.

    Args:
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation factor.
        causal (bool, optional): Whether to use causal padding. Defaults to True.
        noise (bool, optional): Whether to pad with white noise. Defaults to False.
    """

    def __init__(self, kernel_size: int, dilation: int, causal: bool = True, noise=False):
        super().__init__()
        pad = dilation * (kernel_size - 1)
        if not causal:
            pad //= 2
            self.padding = (pad, pad)
        else:
            self.padding = (pad, 0)

        self.noise = noise

    def forward(self, x):
        return nn.functional.pad(x, self.padding)


class _DilatedResidualBlock(nn.Module):
    """Temporal convolutional network internal block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation factor.
        causal (bool, optional): Whether to use causal padding. Defaults to True.
        norm (Literal["batch", "instance", None], optional): Normalization type.
        activation (str, optional): Activation function in `torch.nn` or "gated".
            Defaults to "GELU".
        film_conditioning (bool, optional): Whether to use FiLM conditioning. Defaults
            to False.
        film_embedding_size (int, optional): Size of the FiLM embedding. Defaults to
            None.
        film_batch_norm (bool, optional): Whether to use batch normalization in FiLM.
            Defaults to True.
        use_temporal_film (bool, optional): Whether to use TFiLM conditioning. Defaults
            to False.
        temporal_film_block_size (int, optional): TFiLM block size. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool = True,
        pad: bool = True,
        norm: Literal["batch", "instance", None] = None,
        activation: str = "GELU",
        film_conditioning: bool = False,
        film_embedding_size: int | None = None,
        film_batch_norm: bool = True,
        use_temporal_film: bool = False,
        temporal_film_block_size: int | None = None,
    ):
        super().__init__()

        self.pad = pad

        if film_conditioning and (
            film_embedding_size is None
            or not isinstance(film_embedding_size, int)
            or film_embedding_size < 1
        ):
            raise ValueError("FiLM conditioning requires a valid embedding size (int >= 1).")

        if use_temporal_film and (
            temporal_film_block_size is None
            or not isinstance(temporal_film_block_size, int)
            or temporal_film_block_size < 1
        ):
            raise ValueError("TFiLM conditioning requires a valid block size (int >= 1).")

        net = []

        pre_activation_channels = out_channels * 2 if activation == "gated" else out_channels

        if norm is not None:
            if norm not in ("batch", "instance"):
                raise ValueError("Invalid norm type (must be batch or instance)")
            _Norm = nn.BatchNorm1d if norm == "batch" else nn.InstanceNorm1d
            net.append(_Norm(in_channels))

        if self.pad:
            net.append(Pad(kernel_size, dilation, causal))

        net.extend(
            [
                nn.Conv1d(
                    in_channels,
                    pre_activation_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=0,
                )
            ]
        )
        self.net = nn.Sequential(*net)

        self.film = (
            FiLM(film_embedding_size, pre_activation_channels, film_batch_norm)
            if film_conditioning
            else None
        )

        self.activation = (
            get_act_functional(activation, in_features=pre_activation_channels)
            if activation == "snake"
            else get_act_functional(activation)
        )

        self.tfilm = TFiLM(out_channels, temporal_film_block_size) if use_temporal_film else None
        self.residual = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, film_embedding: torch.Tensor | None = None):
        activations = self.net(x)
        if self.film is not None:
            activations = self.film(activations, film_embedding)
        y = self.activation(activations)

        if self.tfilm is not None:
            y = self.tfilm(y)

        return (
            y + self.residual(x)
            if self.pad
            else y + self.residual(x)[..., x.shape[-1] - y.shape[-1] :]
        )


class TCN(nn.Module):
    """Temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        dilation_base (int, optional): Base of the dilation factor. Defaults to 2.
        num_layers (int, optional): Number of layers. Defaults to 8.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        causal (bool, optional): Whether to use causal padding. Defaults to True.
        norm (Literal["batch", "instance", None], optional): Normalization type.
        activation (str, optional): Activation function in `torch.nn` or "gated".
            Defaults to "GELU".
        film_conditioning (bool, optional): Whether to use FiLM conditioning. Defaults
            to False.
        film_embedding_size (int, optional): FiLM embedding size. Defaults to None.
        film_batch_norm (bool, optional): Whether to use batch normalization in FiLM.
            Defaults to True.
        use_temporal_film (bool, optional): Whether to use TFiLM conditioning. Defaults
            to False.
        temporal_film_block_size (int, optional): TFiLM block size. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dilation_base: int = 2,
        dilation_blocks: int | None = None,
        num_layers: int = 8,
        kernel_size: int = 3,
        causal: bool = True,
        norm: Literal["batch", "instance", None] = None,
        activation: str = "GELU",
        film_conditioning: bool = False,
        film_embedding_size: int | None = None,
        film_batch_norm: bool = True,
        use_temporal_film: bool = False,
        temporal_film_block_size: int | None = None,
        noise_padding: bool = False,  # Pad left with noise
        pad_input: bool = True,
    ):
        super().__init__()

        self.in_projection = nn.Conv1d(in_channels, hidden_channels, 1)
        self.out_projection = nn.Conv1d(hidden_channels, out_channels, 1)

        self.kernel_size = kernel_size
        self.noise_padding = noise_padding
        self.num_layers = num_layers
        self.pad_input = pad_input

        net = []
        self.dilations = [dilation_base ** (n % dilation_blocks) for n in range(num_layers)]
        dilation_blocks = dilation_blocks if dilation_blocks is not None else num_layers
        for n in range(num_layers):
            dilation = self.dilations[n]
            net.append(
                _DilatedResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    causal=causal,
                    pad=(
                        not self.noise_padding and self.pad_input
                    ),  # if Noise padding, we add all padding before
                    norm=norm,
                    activation=activation,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_temporal_film=use_temporal_film,
                    temporal_film_block_size=temporal_film_block_size,
                )
            )

        self.net = nn.ModuleList(net)
        if self.noise_padding:
            self.register_buffer(
                "noise",
                torch.randn(1, 1, get_tcn_input_padding(kernel_size, self.num_layers, causal=True)),
            )

    def forward(self, x: torch.Tensor, film_embedding: torch.Tensor | None = None):
        if self.noise_padding:
            # We pad the input with noise so that the output of the TCN is the same length as the input
            # pad_amount = get_tcn_input_padding([self.kernel_size] * len(self.dilations), self.dilations, causal=True)
            # noise = torch.randn(x.shape[0], x.shape[1], pad_amount, device=x.device)
            noise = torch.randn(x.shape[0], 1, x.shape[-1].to(x.device))
            x = torch.cat([noise, x], dim=-1)
        # Let's normalize the input to have a mean of 0 and std of 1

        x = self.in_projection(x)
        for layer in self.net:
            x = layer(x, film_embedding)
        x = self.out_projection(x)

        return x


def get_tcn_input_padding(kernel_size: int, num_layers: int, causal: bool = True):
    """Get the padding required to keep the input and output lengths the same for a stack of dilated convolutions.

    Takes into account the kernel size and dilation factor for every layer.
    """
    padding = 0
    dilations = [2 ** (n % num_layers) for n in range(num_layers)]
    kernel_sizes = [kernel_size] * num_layers
    for kernel_size, dilation in zip(kernel_sizes, dilations):
        padding += dilation * (kernel_size - 1)
    return padding // 2 if not causal else padding


class TransientTCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        dilation_base: int = 2,
        dilation_blocks: int | None = None,
        num_layers: int = 8,
        kernel_size: int = 13,
        film_conditioning: bool = False,
        film_embedding_size: int | None = None,
        film_batch_norm: bool = True,
        transient_conditioning: bool = False,
        transient_conditioning_channels: int = 32,
        transient_conditioning_length: int = 24000,
        n_tracks: int = 1,
        noise_padding: bool = False,
        pad_input: bool = True,
        lowpass_input: bool = False,
        norm: Literal["batch", "instance", None] = None,
        activation: str = "GELU",
        normalize_output: bool = False,
        sample_rate: int = None,
    ):
        super().__init__()
        self.input_type = "waveform"
        self.synth_params = ["tcn_embedding"]
        self.logit_transform = {"tcn_embedding": nn.Identity()}
        self.splits = {"tcn_embedding": (n_tracks, 1)}
        self.lowpass_input = lowpass_input
        self.normalize_output = normalize_output
        self.sample_rate = sample_rate

        self.tcn = TCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dilation_base=dilation_base,
            dilation_blocks=dilation_blocks,
            num_layers=num_layers,
            kernel_size=kernel_size,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
            norm=norm,
            noise_padding=noise_padding,
            pad_input=pad_input,
            activation=activation,
        )

        if transient_conditioning:
            p = (
                torch.randn(1, transient_conditioning_channels, transient_conditioning_length)
                / transient_conditioning_channels
            )
            self.transient_conditioning = torch.nn.Parameter(p, requires_grad=True)

    def get_envelope(self, embedding, envelope_length):
        return torch.ones(embedding.size(0), 1, envelope_length, device=embedding.device)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor | None = None):
        if hasattr(self, "transient_conditioning"):
            cond = repeat(self.transient_conditioning, "1 c l -> b c l", b=x[0].size(0))
            cond = torch.nn.functional.pad(cond, (0, x[0].size(-1) - cond.size(-1)))
            x = torch.cat([x, cond], dim=1)

        if self.lowpass_input:
            x = torch.stack(
                [
                    F.lowpass_biquad(
                        x[:, ch, :], sample_rate=self.sample_rate, cutoff_freq=7800, Q=0.707
                    )
                    for ch in range(x.size(1))
                ],
                dim=1,
            )

        x = self.tcn(x, embedding)
        if self.normalize_output:
            x = x / (x.abs().max() + 1e-6) * 0.95
        return x


class TransientTCNWithEnvelope(TransientTCN):

    def __init__(
        self,
        num_envelope_points: int = 10,  # If envelope mode is piecewise
        scale: float = 40.0,  # If envelope mode is exponential. Alpha range is [0,  scale with]
        sample_rate: int = None,  # Used to get time in seconds for exponential envelope
        envelope_mode: str = "piecewise",
        envelope_first: bool = False,
        interpolation_method: str = "window",
        film_embedding_size: int = 128,
        **kwargs,
    ):
        super().__init__(film_embedding_size=film_embedding_size, **kwargs)
        # Let's compute the downsampling factor from how many envelope points we have
        self.envelope_mode = envelope_mode
        self.envelope_first = envelope_first

        if envelope_mode == "exponential":
            self.emb_to_gain_latent = nn.Sequential(
                nn.Linear(film_embedding_size, film_embedding_size),
                nn.ReLU(),
                nn.Linear(film_embedding_size, 1),
            )
            self.register_buffer("scale", torch.tensor(scale) / 2)
        elif envelope_mode == "piecewise":
            self.emb_to_gain_latent = nn.Sequential(
                nn.Linear(film_embedding_size, film_embedding_size),
                nn.ReLU(),
                nn.Linear(film_embedding_size, num_envelope_points),
            )
        elif envelope_mode == "none":
            self.emb_to_gain_latent = nn.Identity()

        self.interpolation_method = interpolation_method
        self.activation = get_act_functional("exp_sigmoid")
        self.sample_rate = sample_rate

    def get_envelope(self, embedding, envelope_length):
        gain_logits = self.emb_to_gain_latent(embedding)
        gain = self.activation(gain_logits)
        # We want to pass gain_logits as (batch, frames, latent_size) with frames = num_envelope_points
        envelope = gain.unsqueeze(-1)  # (batch, num_envelope_points, 1)
        if self.envelope_mode == "exponential":
            envelope = torch.exp(
                -envelope
                * self.scale
                * torch.arange(envelope_length, device=embedding.device)
                / self.sample_rate
            )
        elif self.envelope_mode == "piecewise":
            envelope = resample(
                envelope, envelope_length, method=self.interpolation_method
            ).permute(0, 2, 1)

        elif self.envelope_mode == "none":
            envelope = torch.ones(embedding.size(0), 1, envelope_length, device=embedding.device)

        return envelope

    def forward(self, audio, embedding=None):
        envelope = self.get_envelope(embedding, audio.size(-1))
        if self.envelope_first:
            audio = audio * envelope
        tcn_out = super().forward(audio, embedding=embedding)
        # Embedding is (batch, embedding size)
        if not self.envelope_first:
            tcn_out = tcn_out * envelope
        return tcn_out
