import auraloss
import torch
import torch.nn as nn


class nSDR(auraloss.time.SNRLoss):
    def __init__(self, energy_threshold=1e-4, invert=False, frame_length=None, **kwargs):
        super().__init__(**kwargs)
        self.energy_threshold = energy_threshold
        self.invert = 1 if invert else -1
        self.frame_length = frame_length

    def forward(self, input, target):
        values = super().forward(input, target) * self.invert
        return mask_energy(values, target, self.energy_threshold).mean()


def mask_energy(input, target, energy_threshold=1e-4):
    energy = torch.sum(target**2, dim=-1, keepdim=True)
    mask = energy > energy_threshold
    return mask * input


class LogSpectralDistance(nn.Module):
    """Log Spectral Distance (LSD) metric.

    Computes the average Log Spectral Distance between two audio signals. Based on
    https://github.com/jorshi/drumblender/blob/main/drumblender/metrics.py
    """

    def __init__(
        self, n_fft=8092, hop_length=64, filter_energy_threshold=1e-3, eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.name = "LSD"
        self.filter_energy_threshold = filter_energy_threshold

    def _log_spectral_power_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the log spectral power magnitude of the input tensor."""
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=x.device),
            return_complex=True,
        )
        return torch.log(torch.square(torch.abs(X)) + self.eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the Log Spectral Distance between input tensors `x` and `y`.

        x is the estimated signal and y is the reference signal.
        """
        assert x.shape == y.shape, "Input tensors must have the same shape"
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if y.ndim == 2:
            y = y.unsqueeze(1)
        # assert x.ndim == 3 and x.shape[1] == 1, "Only mono audio is supported"
        # Let's merge channel and batch dimensions
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])

        # # Remove channel dimension (since it is mono audio with shape [batch, 1, samples])
        # x = x.squeeze(1)
        # y = y.squeeze(1)

        # Compute log spectral power magnitudes
        # if self.filter_energy_threshold > 0:
        #     # Dont compute the LSD if the reference is silent
        #     if energy(y) < self.filter_energy_threshold:
        #         # TODO check if return 0.0 is correct, possibly not
        #         return torch.tensor(torch.nan)
        X = self._log_spectral_power_mag(x)
        Y = self._log_spectral_power_mag(y)

        # Mean of the squared difference along the frequency axis
        lsd = torch.mean(torch.square(X - Y), dim=-2)

        # Mean of the square root over the temporal axis
        lsd = torch.mean(torch.sqrt(lsd), dim=-1)

        # Return the average Log Spectral Distance over the batch
        if lsd.isnan().any():
            pass
        return torch.mean(lsd)


class SilenceEvaluation(nn.Module):
    """Class to compute Predicted Energy at Silence (PES) and Energy at Predicted Silence (EPS) metrics from.

    Schulze-Forster, Kilian, et al. "Phoneme level lyrics alignment and text-informed
    singing voice separation." IEEE/ACM Transactions on Audio, Speech, and Language
    Processing 29 (2021): 2382-2395.

    The energy of a frame is computed as:
        energy = 10 * log10(sum(frame ** 2) + eps)

    The silent threshold can be used to set a threshold for silence detection.
    If we normalize the PES by the average energy of non-silent frames, we get the normalized PES (nPES).
    Intuitively, we can think of it as "how much I have predicted signal when my ground truth is silent,
    compared to how much energy I have when my ground truth is not silent".
    """

    def __init__(
        self,
        silent_threshold=1e-3,
        hop_length=512,
        pes=True,
        eps=False,
        scale="log",
        norm_by_avg_non_silence_energy=False,
        epsilon=1e-8,
    ):
        super().__init__()

        # TODO: silent_threshold argument is currently useless
        self.silent_threshold_log = -60
        self.silent_threshold_lin = 1e-6
        self.silent_threshold = (
            self.silent_threshold_log if scale == "log" else self.silent_threshold_lin
        )
        self.epsilon = epsilon
        self.hop_length = hop_length
        self.pes = pes
        self.eps = eps
        self.scale = scale
        self.norm_by_avg_non_silence_energy = norm_by_avg_non_silence_energy
        self.name = ("PES", "EPS")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # if x.ndim == 3:

        # Lets merge channel and batch dimensions
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        # Remove channel dimension [batch, 1, samples]
        # x = x.squeeze(1)
        # y = y.squeeze(1)

        # Compute the energy of the input tensors
        # unfold the input tensor to compute the energy of each frame
        x = x.unfold(1, self.hop_length, self.hop_length).reshape(-1, self.hop_length)
        y = y.unfold(1, self.hop_length, self.hop_length).reshape(-1, self.hop_length)

        x_energy = energy(x, eps=self.epsilon, reduce=False, scale=self.scale)
        x_energy = torch.clamp(x_energy, min=self.silent_threshold)
        y_energy = energy(y, eps=self.epsilon, reduce=False, scale=self.scale)
        y_energy = torch.clamp(y_energy, min=self.silent_threshold)

        # Compute the predicted silence mask
        x_silence = x_energy <= self.silent_threshold
        y_silence = y_energy <= self.silent_threshold

        non_y_silence = ~y_silence
        # How much energy on average we predict with actual ground truth energy
        norm_factor = (
            torch.mean(x_energy[non_y_silence])
            if torch.any(non_y_silence)
            else torch.tensor(1.0, device=x.device)
        )

        # Compute the predicted energy at silence (PES) and energy at predicted silence (EPS)
        PES = (
            torch.mean(x_energy[y_silence])
            if torch.any(y_silence)
            else torch.zeros(1, device=x.device)
        )
        EPS = (
            torch.mean(y_energy[x_silence])
            if torch.any(x_silence)
            else torch.zeros(1, device=x.device)
        )

        if self.norm_by_avg_non_silence_energy:
            PES = PES / norm_factor

            # TOdo add norm factor to EPS

        if self.pes and self.eps:
            return (PES, EPS)
        elif self.pes:
            return PES
        elif self.eps:
            return EPS
        return (PES, EPS)


def energy(window, eps=1e-8, reduce=True, scale="log"):

    if scale == "log":
        en = 10 * torch.log10(torch.sum(window**2, axis=-1) + eps)
    elif scale == "linear":
        en = torch.sum(window**2, axis=-1)
    if reduce:
        return torch.mean(en)
    return en


def prepare_inputs(targets, estimates):
    """Prepare input tensors by ensuring they are float type and adding channel dimension if missing.

    Args:
        targets (torch.Tensor): Reference signals to compare, shape (batch_size, n_channels, n_samples) or (batch_size, n_samples)
        estimates (torch.Tensor): Estimated signals to compare, shape (batch_size, n_channels, n_samples) or (batch_size, n_samples)

    Returns:
        torch.Tensor: Prepared reference signals, shape (batch_size, n_channels, n_samples)
        torch.Tensor: Prepared estimated signals, shape (batch_size, n_channels, n_samples)
    """

    targets = targets.float()
    estimates = estimates.float()
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    if targets.ndim == 2:
        targets = targets.unsqueeze(1)
    if estimates.ndim == 1:
        estimates = estimates.unsqueeze(0)
    if estimates.ndim == 2:
        estimates = estimates.unsqueeze(1)
    return targets, estimates


def segment_and_compute_metric(
    metric_func,
    estimates,
    targets,
    eval_frame_length=16000,
    reduce=True,
    filter_silent_frames=False,
):
    """Segment the input signals and compute the given metric for each segment.

    Args:
        metric_func (Callable): The metric function to compute.
        targets (torch.Tensor): The reference signals, shape (batch_size, n_channels, n_samples) or (batch_size, n_samples)
        estimates (torch.Tensor): The estimated signals, shape (batch_size, n_channels, n_samples) or (batch_size, n_samples)
        eval_frame_length (int): The length of each evaluation frame in samples.

    Returns:
        torch.Tensor: Metric values for each example in the batch, shape (batch_size,)
    """
    targets, estimates = prepare_inputs(targets, estimates)
    discard_last_frame = False
    if eval_frame_length <= 0 or eval_frame_length is None:
        eval_frame_length = targets.shape[-1]
        discard_last_frame = True if estimates.shape[-1] > targets.shape[-1] else False
    segmented_targets, n_eval_frames = segment_signal(targets, eval_frame_length)
    segmented_estimates, _ = segment_signal(
        estimates, eval_frame_length, discard_last_frame=discard_last_frame
    )

    batch_size = targets.shape[0]

    if filter_silent_frames:
        # Filter out silent frames based on target frame energy
        segmented_targets = torch.stack(segmented_targets, dim=1)
        segmented_estimates = torch.stack(segmented_estimates, dim=1)
        target_frame_energy = energy(segmented_targets, reduce=False)
        silent_frames = target_frame_energy < SILENCE_THRESHOLD_DB
        silent_frames_expanded = silent_frames.unsqueeze(-1).expand_as(segmented_targets)
        # Fetch only non-silent frames
        # Calculate number of non-silent frames
        num_non_silent = (~silent_frames).sum()
        # Filter and then reshape to preserve dimensions
        non_silent_targets = segmented_targets[~silent_frames_expanded]
        non_silent_estimates = segmented_estimates[~silent_frames_expanded]

        # Reshape to [1, 1, num_non_silent, size]
        if num_non_silent != n_eval_frames:
            n_eval_frames = num_non_silent
            pass
        segmented_targets = non_silent_targets.reshape(batch_size, num_non_silent, 1, -1)
        segmented_estimates = non_silent_estimates.reshape(batch_size, num_non_silent, 1, -1)
    else:
        segmented_targets = torch.stack(segmented_targets, dim=1)
        segmented_estimates = torch.stack(segmented_estimates, dim=1)

    batch_size = targets.shape[0]
    metric_values_frames = torch.zeros((batch_size, n_eval_frames), device=targets.device)

    for n in range(n_eval_frames):
        metric_frame = metric_func(segmented_estimates[:, n], segmented_targets[:, n])
        metric_values_frames[:, n] = metric_frame

    if reduce:
        return torch.mean(metric_values_frames, dim=1), torch.std(metric_values_frames, dim=1)
    return metric_values_frames


# TODO: implement this with unfold
def segment_signal(signal, eval_frame_length, discard_last_frame=False):
    """Segment the signal into frames of given length. Pad the last frame if necessary.

    Args:
        signal (torch.Tensor): Input signal of shape (batch_size, n_channels, n_samples) or (batch_size, n_samples)
        eval_frame_length (int): Length of each evaluation frame in samples

    Returns:
        list of torch.Tensor: List of segmented frames
        int: Number of evaluation frames
    """
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    if signal.ndim == 2:
        signal = signal.unsqueeze(1)
    if eval_frame_length <= 0 or eval_frame_length is None:
        return [signal], 1
    batch_size, n_channels, n_samples = signal.shape
    # Pad if necessary so last frame has the same length as the others
    n_pad = eval_frame_length - (n_samples % eval_frame_length)
    if n_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, n_pad), mode="constant", value=0)
    n_eval_frames = int(np.ceil(n_samples / eval_frame_length))
    segmented_signal = [
        signal[:, :, n * eval_frame_length : (n + 1) * eval_frame_length]
        for n in range(n_eval_frames)
    ]
    if discard_last_frame:
        segmented_signal = segmented_signal[:-1]
        n_eval_frames -= 1
    return segmented_signal, n_eval_frames
