import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTransform(nn.Module):
    """Base class for time-frequency transforms."""

    def __init__(
        self, padding_divisor: Optional[int] = None, pad_left_right: bool = False, **fft_kwargs
    ):
        super().__init__()
        self.fft_kwargs = fft_kwargs
        self.padding_divisor = (
            padding_divisor if padding_divisor is not None else fft_kwargs.get("hop_length", 1)
        )
        self.pad_left_right = pad_left_right
        self.left_right_pad_amount = 0
        self.hop_length = fft_kwargs.get("hop_length", None)
        self.numpy_input = False

    def _handle_padding(self, x: torch.Tensor, adjust_padding: bool = True) -> torch.Tensor:
        """Handle both left/right padding and padding to multiple of padding_divisor."""
        _x = x

        if self.pad_left_right:
            # Pad by window_size - hop_length zeros on both sides
            pad_length = self.kwargs.get("n_fft", 0) - self.kwargs.get("hop_length", 0)
            _x = F.pad(_x, (pad_length, pad_length), mode="constant", value=0)
            self.left_right_pad_amount = pad_length

        if adjust_padding and self.padding_divisor > 1:
            # Calculate padding needed to make length multiple of padding_divisor
            length = _x.size(-1)
            target_length = (
                (length + self.padding_divisor - 1) // self.padding_divisor
            ) * self.padding_divisor
            pad_amount = target_length - length
            if pad_amount > 0:
                _x = F.pad(_x, (0, pad_amount), mode="constant", value=0)

        return _x

    def _reshape_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """Reshape input to 2D and keep original shape."""

        if not isinstance(x, torch.Tensor):
            # Convert from numpy
            x = torch.from_numpy(x)
            self.numpy_input = True
        else:
            self.numpy_input = False

        x_shape = x.shape
        if len(x_shape) == 1:
            x = x.unsqueeze(0)
        if len(x_shape) > 2:
            x = x.view(-1, x_shape[-1])

        return x, x_shape

    def _reshape_output(self, x: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """Reshape output back to original dimensions."""
        if len(original_shape) > 2:
            x = x.view(*original_shape[:-1], *x.shape[-2:])
        if self.numpy_input:
            x = x.numpy()
        return x

    def get_frame_rate(self, sample_rate: int) -> float:
        """Get frame rate in Hz."""
        if self.hop_length is None:
            raise ValueError("hop_length must be specified to calculate frame rate")
        return sample_rate / self.hop_length

    def get_n_frames(self, signal_length: int) -> int:
        """Calculate number of frames for a given signal length."""
        if self.hop_length is None:
            raise ValueError("hop_length must be specified to calculate number of frames")
        return (signal_length + self.hop_length - 1) // self.hop_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class STFT(BaseTransform):
    """Short-Time Fourier Transform."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        magnitude: bool = True,
        center: bool = True,
        log: bool = False,
        padding_divisor: Optional[int] = None,
        pad_left_right: bool = False,
    ):
        fft_kwargs = {
            "n_fft": n_fft,
            "hop_length": hop_length if hop_length is not None else n_fft // 4,
            "win_length": win_length if win_length is not None else n_fft,
            "window": window,
            "center": center,
            "return_complex": True,
        }
        self.ifft_kwargs = copy.deepcopy(fft_kwargs)
        self.ifft_kwargs.pop("return_complex")
        super().__init__(
            padding_divisor=padding_divisor, pad_left_right=pad_left_right, **fft_kwargs
        )
        self.magnitude = magnitude
        self.log = log

    def forward(
        self, x: torch.Tensor, adjust_padding: bool = True, return_complex: bool = False
    ) -> torch.Tensor:
        """x: Tensor of shape [batch, time]. If not, will be reshaped and all dims (except last) will be transferred to batch."""

        x, x_shape = self._reshape_input(x)

        # Handle padding
        x = self._handle_padding(x, adjust_padding)

        # Compute STFT
        stft = torch.stft(x, **self.fft_kwargs)

        # Post-process
        if self.magnitude:
            stft = stft.abs()
            if self.log:
                stft = torch.log(stft + 1e-6)

        # Reshape back
        stft = self._reshape_output(stft, x_shape)

        return stft

    def inverse(
        self, X: torch.Tensor, length: Optional[int] = None, n_iter: int = 32
    ) -> torch.Tensor:
        """Inverse STFT with optional Griffin-Lim phase reconstruction."""
        # Handle complex input
        if X.size(-1) == 2:
            X = torch.view_as_complex(X)

        # Handle batched instruments
        original_shape = X.shape
        if len(original_shape) == 4:
            X = X.view(-1, *original_shape[-2:])

        # If magnitude-only, perform Griffin-Lim
        if not X.is_complex():
            X = self._griffin_lim(X, n_iter)

        # Inverse STFT
        x = torch.istft(X, **self.ifft_kwargs, length=length)

        # Reshape if necessary
        if len(original_shape) == 4:
            x = x.view(*original_shape[:-2], -1)

        return x

    def _griffin_lim(self, S: torch.Tensor, n_iter: int) -> torch.Tensor:
        """Griffin-Lim algorithm for phase reconstruction."""
        angles = torch.exp(2j * torch.pi * torch.rand_like(S))
        X = S * angles

        for _ in range(n_iter):
            x = torch.istft(X, **self.kwargs)
            X = torch.stft(x, **self.kwargs)
            angles = torch.exp(1j * torch.angle(X))
            X = S * angles

        return X
