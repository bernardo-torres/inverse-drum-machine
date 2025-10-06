import torch
import torch.nn as nn


def fast_conv1d(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Performs fast 1D convolution in the frequency domain.

    This function implements a convolution operation using the overlap-add method
    with FFTs, which is efficient when the kernel is much shorter than the signal.
    The operation is exactly convolution - the kernel doesn't need to be flipped.

    Note:
        This implementation is adapted from https://github.com/keunwoochoi/DrummerNet
        and assumes a single-channel input signal.

    Args:
        signal: The input signal tensor of shape `(B, 1, T_sig)`.
        kernel: The convolution kernel tensor of shape `(T_ker,)` or `(B, T_ker)`.
                Batch size must be 1 or match the signal's batch size.

    Returns:
        The convolved signal of shape `(B, 1, T_sig)`.
    """
    batch, ch, sig_len = signal.shape
    if ch != 1:
        raise ValueError(f"fast_conv1d assumes a single channel, but got {ch}.")

    if kernel.dim() == 1:
        kernel = kernel.unsqueeze(0)

    kernel_batch, kernel_len = kernel.shape
    if not (kernel_batch == batch or kernel_batch == 1):
        raise ValueError(
            f"Kernel batch size ({kernel_batch}) must be 1 or match signal batch size ({batch})."
        )

    # Use the next power of 2 for FFT length for efficiency
    fft_len = 2 << (kernel_len - 1).bit_length()
    step_size = fft_len - kernel_len + 1

    # Pre-compute the FFT of the zero-padded kernel
    kernel_fft = torch.fft.rfft(nn.functional.pad(kernel, (0, fft_len - kernel_len)))

    # Pad the signal to be a multiple of the step size
    pad_len = (step_size - sig_len % step_size) % step_size
    signal = nn.functional.pad(signal, (0, pad_len))
    _, _, padded_len = signal.shape

    # Process the signal in overlapping blocks
    result = torch.zeros(batch, 1, padded_len + kernel_len - 1, device=signal.device)
    for i in range(0, sig_len, step_size):
        # Extract and pad a block of the signal
        block = signal[:, :, i : i + step_size]
        block = nn.functional.pad(block, (0, fft_len - step_size))

        # Perform convolution in the frequency domain
        block_fft = torch.fft.rfft(block, norm="ortho")
        conv_fft = block_fft * kernel_fft
        conv_block = torch.fft.irfft(conv_fft, norm="ortho")

        # Add the result back to the corresponding position (overlap-add)
        result[:, :, i : i + fft_len] += conv_block

    return result[:, :, :sig_len]


class FastDrumSynthesizer(nn.Module):
    """Synthesizes drum tracks by convolving onsets with instrument samples."""

    def __init__(self, *args):
        """Initializes the FastDrumSynthesizer."""
        super().__init__()

    def forward(
        self, onsets: torch.Tensor, sources: torch.Tensor, delay_compensation: int = 0
    ) -> torch.Tensor:
        """Generates audio for each instrument track.

        Args:
            onsets: A tensor of onset activations. Shape: `(B, K, T)`.
            sources: A tensor of one-shot audio samples for each instrument.
                     Shape: `(B, K, R)`.
            delay_compensation: Currently unused. Included for compatibility.

        Returns:
            A tensor of synthesized audio tracks. Shape: `(B, K, T)`.
        """
        if sources is None:
            raise ValueError("The 'sources' tensor must be provided.")

        tracks = []
        # Convolve each instrument's onset signal with its corresponding audio sample
        # This currently uses a for loop over instruments, which is efficient enough
        # There's definitely some improvements to be made here
        # We tried other methods (e.g., grouped conv1d) were tested but found slower.
        for i in range(onsets.shape[1]):
            # Convolves (B, 1, T) with (B, R) to produce (B, 1, T)
            track = fast_conv1d(onsets[:, i : i + 1, :], sources[:, i, :])
            tracks.append(track)

        return torch.cat(tracks, dim=1)


class Mixer(nn.Module):
    """A simple module to sum audio tracks into a single mixture."""

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        """Mixes instrument tracks by summing them.

        Args:
            tracks: A tensor of audio tracks. Shape: `(B, K, T)`.

        Returns:
            Mixed audio. Shape: `(B, 1, T)`.
        """
        return tracks.sum(dim=1, keepdim=True)


class FastDrumSequencerWithMixer(nn.Module):
    """A complete sequencer that synthesizes and optionally mixes drum tracks."""

    def __init__(self, upsampler: nn.Module = nn.Identity(), conv: str = "fft_loop", **kargs):
        """Initializes the FastDrumSequencerWithMixer.

        Args:
            upsampler: An optional module to upsample the input activations.
            conv: The convolution method to use. Currently only "fft_loop" is supported.
        """
        super().__init__()
        if conv == "fft_loop":
            self.sequencer = FastDrumSynthesizer()
        else:
            raise NotImplementedError(f"Convolution method '{conv}' is not implemented.")

        self.mixer = Mixer()
        self.upsampler = upsampler
        self.sequencer_type = "waveform"  # For compatibility checks

    def forward(
        self,
        activations: torch.Tensor,
        sources: torch.Tensor,
        mix: bool = True,
        delay_compensation: int = 0,
    ) -> torch.Tensor:
        """Processes activations and one-shot sources to generate drum tracks or a final mix.

        Args:
            activations: The input onset activations. Shape: `(B, K, T_low_res)`.
            sources: The one-shot audio samples for each instrument.
            mix: If True, returns a single mixed track. If False, returns the
                 individual instrument tracks (stems).
            delay_compensation: Currently unused. Included for compatibility.

        Returns:
            If mix is True, a tensor of the mixed audio. Shape: `(B, 1, T)`.
            If mix is False, a tensor of the audio stems. Shape: `(B, K, T)`.
        """
        activations = self.upsampler(activations)  # Upsample to sampling rate
        tracks = self.sequencer(activations, sources=sources, delay_compensation=delay_compensation)
        return self.mixer(tracks) if mix else tracks


class ZeroInserter(nn.Module):
    """Upsamples a signal by inserting zeros between samples (zero-order hold)."""

    def __init__(self, insertion_rate: int):
        """Initializes the ZeroInserter.

        Args:
            insertion_rate: The factor by which to upsample the signal. For
                            example, a rate of 2 will double the length.
        """
        super().__init__()
        if insertion_rate < 1:
            raise ValueError("Insertion rate must be a positive integer.")
        self.insertion_rate = insertion_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples the input tensor by inserting zeros.

        Args:
            x: The input tensor to be upsampled. Shape: `(B, K, T)`.

        Returns:
            The upsampled tensor. Shape: `(B, K, T * insertion_rate)`.
        """
        if self.insertion_rate == 1:
            return x

        b, k, t = x.shape
        # Create a tensor of zeros to insert between samples
        zeros = torch.zeros(b, k, t, self.insertion_rate - 1, device=x.device)
        # Reshape input to (B, K, T, 1) to allow concatenation
        x_reshaped = x.unsqueeze(-1)
        # Concatenate samples and zeros, then reshape to the final upsampled length
        interleaved = torch.cat([x_reshaped, zeros], dim=-1)
        return interleaved.view(b, k, t * self.insertion_rate)
