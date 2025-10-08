import torch
import torchaudio

from idm.feature_extractor.stft import BaseTransform


class MelSpectrogram(BaseTransform):
    """Mel Spectrogram transform using torchaudio."""

    def __init__(
        self,
        sample_rate: int = None,
        n_fft: int = 1024,
        n_mels: int = 128,
        hop_length: int | None = None,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        log: bool = False,
        padding_divisor: int | None = None,
        pad_left_right: bool = False,
        center: bool = True,
    ):
        kwargs = {
            "n_fft": n_fft,
            "hop_length": hop_length if hop_length is not None else n_fft // 4,
            "win_length": win_length if win_length is not None else n_fft,
            "center": center,
        }
        super().__init__(padding_divisor=padding_divisor, pad_left_right=pad_left_right, **kwargs)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=self.hop_length,
            win_length=kwargs["win_length"],
            f_min=f_min,
            f_max=f_max,
            center=center,
        )
        self.log = log

    def forward(self, x: torch.Tensor, adjust_padding: bool = True) -> torch.Tensor:
        x, x_shape = self._reshape_input(x)
        x = self._handle_padding(x, adjust_padding)
        mel = self.mel_transform(x)
        if self.log:
            mel = torch.log(mel + 1e-6)
        mel = self._reshape_output(mel, x_shape)
        return mel
