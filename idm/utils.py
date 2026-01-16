import functools
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile

from idm import SILENCE_THRESHOLD
from idm.run_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def convert_onset_dict_to_activations(
    onsets_dict: dict[str, list[torch.Tensor]],
    n_frames: int,
    instrument_classes: list[str],
    batch_size: int = 1,
    label_mapping: dict[str, str] | None = None,
    activation_rate: float = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Converts a dictionary of onset times into a multi-channel activation tensor.

    Args:
        onsets_dict: A dictionary where keys are instrument names and values are lists
                     of tensors, each containing onset times in seconds for a batch item.
        n_frames: The total number of frames for the output activation tensor.
        instrument_classes: An ordered list of instrument class names that defines
                            the channel order in the output tensor.
        batch_size: The number of items in the batch.
        label_mapping: An optional dictionary to map instrument names in `onsets_dict`
                       to those in `instrument_classes`.
        activation_rate: The rate of the activation signal in Hz (frames per second).

    Returns:
        A binary tensor of shape `(B, K, M)` where B is the batch size, K is the
        number of instrument classes, and M is `n_frames`.`1` indicates an onset.
    """
    n_classes = len(instrument_classes)
    activations = torch.zeros((batch_size, n_classes, n_frames), device=device)

    if label_mapping is None:
        label_mapping = {key: key for key in instrument_classes}

    for class_idx, instrument_name in enumerate(instrument_classes):
        # onsets_per_batch is a list of tensors for the current instrument
        instrument_name = label_mapping.get(instrument_name, instrument_name)
        onsets_per_batch = onsets_dict.get(instrument_name, [])
        if len(onsets_per_batch) == 0:
            continue
        if not isinstance(onsets_per_batch[0], list):
            assert batch_size == 1, "Batch size must be 1 if onsets are not a list"
            onsets_per_batch = [onsets_per_batch]
        for batch_idx, onset_times in enumerate(onsets_per_batch):
            if isinstance(onset_times, list):
                onset_times = torch.tensor(onset_times, device=device)
            if onset_times.numel() == 0:
                continue
            # Convert onset times (seconds) to frame indices
            onset_frames = (onset_times * activation_rate).long()
            # Clamp values to be within the valid frame range
            if onset_frames[-1] >= n_frames:
                log.warning(
                    f"Clamping onset frame {onset_frames[-1]} to {n_frames - 1}. "
                    "This may indicate a mismatch between activation_rate and n_frames."
                )
            onset_frames = torch.clamp(onset_frames, 0, n_frames - 1)
            # Set activation to 1 at the specified frame indices
            activations[batch_idx, class_idx, onset_frames] = 1

    return activations


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

          Args:
            x: Input tensor.
            exponent: In nonlinear regime (away from x=0), the output varies by this
              factor for every change of x by 1.0.
            max_value: Limiting value at x=inf.
            threshold: Limiting value at x=-inf. Stabilizes training when outputs are
              pushed to 0.

          Returns:
            A tensor with pointwise nonlinearity applied.
    """

    x = x.type(torch.float32)
    exponent = torch.tensor(exponent, dtype=torch.float32, device=x.device)
    return max_value * torch.sigmoid(x) ** torch.log(exponent) + threshold


def get_act_module(act_name):
    # act_name = act_name.lower()
    if act_name == "relu":
        return nn.ReLU
    elif act_name == "elu":
        return nn.ELU
    elif act_name in ("lrelu", "leakyrelu"):
        return nn.LeakyReLU
    elif act_name == "sigmoid":
        return nn.Sigmoid
    elif act_name == "tanh":
        return nn.Tanh
    elif act_name == "none":
        return nn.Identity
    # elif act_name == "gated":
    #     return GatedActivation
    else:
        return getattr(nn, act_name)


def get_act_functional(act_name, **kwargs):
    act_name = act_name.lower()
    if act_name == "relu":
        return F.relu
    elif act_name == "elu":
        return F.elu
    elif act_name in ("lrelu", "leakyrelu"):
        return F.leaky_relu
    elif act_name == "sigmoid":
        return F.sigmoid
    elif act_name == "tanh":
        return F.tanh
    elif act_name == "none":
        return lambda x: x
    elif act_name == "exp_sigmoid":
        return functools.partial(exp_sigmoid, exponent=10.0, max_value=2, threshold=1e-7)
    # elif act_name == "log_scale_tanh":
    #     return log_scale_tanh
    # elif act_name == "gated":
    #     return GatedActivation()
    # elif act_name == "snake":
    #     return Snake(**kwargs)
    # elif act_name == "snake_cuda":
    #     from idm.utils.alias_free_activation.cuda.activation1d import Activation1d

    #     snake = Snake(alpha_logscale=True, **kwargs)
    #     return Activation1d(snake)
    # elif act_name == "gumbel_sigmoid":
    #     return functools.partial(gumbel_sigmoid, hard=True, **kwargs)
    else:
        return getattr(F, act_name)


# From DDSP
def upsample_with_windows(
    inputs: torch.Tensor, n_timesteps: int, add_endpoint: bool = True
) -> torch.Tensor:
    """Upsample a series of frames using using overlapping hann windows. Good for amplitude envelopes.

    Args:
      inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
      n_timesteps: The time resolution of the output signal.
      add_endpoint: Hold the last timestep for an additional step as the endpoint.
        Then, n_timesteps is divided evenly into n_frames segments. If false, use
        the last timestep as the endpoint, producing (n_frames - 1) segments with
        each having a length of n_timesteps / (n_frames - 1).

    Returns:
      Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].

    Raises:
      ValueError: If input does not have 3 dimensions.
      ValueError: If attempting to use function for downsampling.
      ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
        true) or n_frames - 1 (if add_endpoint is false).
    """
    inputs = inputs.type(torch.float32)
    crop_length = n_timesteps

    if len(inputs.shape) != 3:
        raise ValueError(
            "Upsample_with_windows() only supports 3 dimensions, " f"not {inputs.shape}."
        )

    # Mimic behavior of tf.image.resize.
    # For forward (not endpointed), hold value for last interval.
    if add_endpoint:
        inputs = torch.cat([inputs, inputs[:, -1:, :]], dim=1)

    n_frames = int(inputs.shape[1])
    n_intervals = n_frames - 1

    if n_frames >= n_timesteps:
        raise ValueError(
            "Upsample with windows cannot be used for downsampling"
            f"More input frames ({n_frames}) than output timesteps ({n_timesteps})"
        )

    if n_timesteps % n_intervals != 0.0:
        # minus_one = "" if add_endpoint else " - 1"
        # raise ValueError(
        #     "For upsampling, the target number of timesteps must be divisible "
        #     "by the number of input frames{}. (timesteps:{}, frames:{}, "
        #     "add_endpoint={}).".format(minus_one, n_timesteps, n_frames, add_endpoint)
        # )
        # Add necessary padding to make n_timesteps divisible by n_intervals.

        n_timesteps = n_intervals * (n_timesteps // n_intervals + 1)
        assert n_timesteps > crop_length

    # Constant overlap-add, half overlapping windows.
    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length, device=inputs.device)  # [window]

    # Transpose for overlap_and_add.
    x = inputs.permute(0, 2, 1)  # [batch_size, n_channels, n_frames]

    # Broadcast multiply.
    # Add dimension for windows [batch_size, n_channels, window, n_frames].
    x = x[:, :, None, :]
    window = window[None, None, :, None]
    x_windowed = x * window

    n_channels = x.shape[1]
    x_windowed = x_windowed.reshape((-1, n_channels * window_length, n_frames))

    # overlap and add
    x = torch.nn.functional.fold(
        x_windowed,
        output_size=(1, n_timesteps + window_length),
        kernel_size=(1, window_length),
        stride=(1, hop_size),
    )
    # x is [batch_size, n_channels, 1, n_timesteps]

    x = x.squeeze(2)  # [batch_size, n_channels n_timesteps]

    # Transpose back.
    x = x.permute(0, 2, 1)  # [batch_size, n_timesteps, n_channels]

    # Trim the rise and fall of the first and last window.
    return x[:, hop_size:-hop_size, :][:, :crop_length, :]


def resample(
    inputs: torch.Tensor, n_timesteps: int, method: str = "bilinear", add_endpoint: bool = True
) -> torch.Tensor:
    """Interpolates a tensor from n_frames to n_timesteps.
    Args:
      inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
        [batch_size, n_frames], [batch_size, n_frames, channels], or
        [batch_size, n_frames, n_freq, channels].
      n_timesteps: Time resolution of the output signal.
      method: Type of resampling, must be in ['nearest', 'bilinear', 'bicubic',
        'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
        'window' uses overlapping windows (only for upsampling) which is smoother
        for amplitude envelopes with large frame sizes.
      add_endpoint: Hold the last timestep for an additional step as the endpoint.
        Then, n_timesteps is divided evenly into n_frames segments. If false, use
        the last timestep as the endpoint, producing (n_frames - 1) segments with
        each having a length of n_timesteps / (n_frames - 1).
    Returns:
      Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
        [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
        [batch_size, n_timesteps, n_freqs, channels].
    Raises:
      ValueError: If method is 'window' and input is 4-D.
      ValueError: If method is not one of 'nearest', 'bilinear', 'bicubic', or
        'window'.
    """
    inputs = inputs.type(torch.float32)
    is_1d = len(inputs.shape) == 1
    is_2d = len(inputs.shape) == 2
    is_4d = len(inputs.shape) == 4

    # Ensure inputs are at least 3d.
    if is_1d:
        inputs = inputs[None, :, None]
    elif is_2d:
        inputs = inputs[:, :, None]

    # resample
    if method == "window":
        outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
    elif method in ["nearest", "bilinear", "bicubic"]:
        outputs = inputs[:, :, :, None] if not is_4d else inputs

        outputs = outputs.permute(0, 2, 1, 3)  # [batch_size, n_channels, n_frames, optional]
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=[n_timesteps, outputs.shape[3]],
            mode=method,
            align_corners=not add_endpoint,
        )
        outputs = outputs.permute(0, 2, 1, 3)  # [batch_size, n_frames, n_channels, optional]
        outputs = outputs[:, :, :, 0] if not is_4d else outputs
    else:
        raise ValueError(
            "Method ({}) is invalid. Must be one of {}.".format(
                method, "['nearest', 'bilinear', 'bicubic', 'window']"
            )
        )

    # Return outputs to the same dimensionality of the inputs.
    if is_1d:
        outputs = outputs[0, :, 0]
    elif is_2d:
        outputs = outputs[:, :, 0]

    return outputs


def cpu_numpy(x):
    """Helper function to convert tensors to numpy arrays on cpu."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return [cpu_numpy(xx) for xx in x]
    elif isinstance(x, dict):
        return {k: cpu_numpy(v) for k, v in x.items()}
    else:
        return x


def pad_multiple_of(x, div):
    """Compute and pad so that `x.dim[-1] % div` becomes `0`.

    Args:
        x (torch.Tensor): (..., time) tensor to be padded
        div (int): number to divide
    """
    time_length = x.shape[-1]
    pad_length = (div - (time_length % div)) % div
    if pad_length == 0:
        return x
    else:
        return F.pad(x, (0, pad_length))


def max_abs_norm(x):
    """Normalizes audio by the maximum absolute value, independently for each batch element.

    Args:
        x: Input audio tensor. Shape [Channels, time]
    """
    if isinstance(x, np.ndarray):
        max_val = np.max(np.abs(x), axis=-1, keepdims=True)
        return x if max_val == 0 else x / max_val

    max_val, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
    if x.shape[0] > 2 and x.ndim > 1:
        raise ValueError(f"Undefined behavior for input channel size {x.shape[0]}")
    if len(max_val) == 2:
        # Normalize each channel independently
        return x / torch.max(max_val, dim=0).values
    return x / torch.max(max_val) if max_val != 0 else x


def no_norm(x):
    """Returns the audio without any normalization."""
    return x


def l1_norm(x):
    """Normalizes audio by the L1 norm."""
    return x / torch.norm(x, p=1, dim=-1, keepdim=True)


def l2_norm(x):
    """Normalizes audio by the L2 norm."""
    return x / torch.norm(x, p=2, dim=-1, keepdim=True)


def get_normalizing_function(normalizing_function):
    """Returns the appropriate normalizing function based on a string identifier."""
    normalizers = {
        "no": no_norm,
        # "abssum": sum_abs_norm,
        # "sqrsum": sum_squares_norm,
        "maxabs": max_abs_norm,
        "l1": l1_norm,
        "l2": l2_norm,
    }
    if normalizing_function in normalizers:
        return normalizers[normalizing_function]
    else:
        raise ValueError(
            f"Invalid normalization parameter! Should be one of {list(normalizers.keys())}"
        )


def is_silent_track(data, threshold=SILENCE_THRESHOLD, return_torch=False):
    """Check if a track is silent based on a threshold.

    Args:
        data: Track data
        threshold: Threshold for silence
        return_torch: Whether to keep the output's type or return torch.Tensor

    Returns:
        Boolean indicating whether the track is silent
    """
    if isinstance(data, np.ndarray) and return_torch:
        data = torch.tensor(data)
    elif isinstance(data, np.ndarray):
        return np.max(np.abs(data)) < threshold
    if torch.abs(data).max() < threshold:
        return True
    return False


def activity_detector(audio: torch.Tensor, threshold: float = 1e-4) -> bool:
    """Checks if the audio tensor is silent or not."""
    return not is_silent_track(audio, threshold=threshold, return_torch=True)


def load_audio(
    path,
    duration,
    sample_rate_target=44100,
    backend="torchaudio",
    resampling_method="sinc_interp_hann",
    random_crop=False,
    mono=True,
    normalize=True,
    normalizing_function="l2",
    offset_sec=0,
):
    """Load and process an audio segment using a specified backend.

    Parameters:
    - path (str): Path to the audio file.
    - duration (float): Duration of the audio segment to load in seconds.
    - sample_rate_target (int): Target sample rate for resampling.
    - backend (str): Audio processing backend, supports 'torchaudio' for now.
    - resampling_method (str): Resampling method for torchaudio.transforms.Resample or soxr.resample.
    - random_crop (bool): If True, crops a random segment of the specified duration.
    - mono (bool): If True, converts the audio to mono.
    - normalize (bool): If True, normalizes the audio waveform. (currently normalized to [-1, 1] range)

    Returns:
    - a tuple of:
        - audio (Tensor): The processed audio waveform.
        - sample_rate_target (int): The sample rate of the processed audio.
        - frame_offset (int): The offset in frames from the beginning of the audio file.
    """
    if duration == -1:
        duration = None
    args = [path, duration, sample_rate_target, random_crop, mono, resampling_method, offset_sec]
    audio_dict = None
    # Check if audio is a file and exists
    try:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        if backend == "torchaudio":
            audio_dict = load_audio_torchaudio(*args)
        elif backend == "scipy-soxr":  # load with scipy and resample with soxr
            if not LOADED_SCIPY_SOXR:
                raise ImportError(
                    "Failed to import scipy.io.wavfile and soxr. Make sure to install the required packages."
                )
            args[-2] = (
                "VHQ" if resampling_method not in ["LQ", "MQ", "HQ", "VHQ"] else resampling_method
            )
            audio_dict = load_audio_scipy_soxr(*args)
        elif backend == "soundfile":
            args[-2] = (
                "HQ" if resampling_method not in ["LQ", "MQ", "HQ", "VHQ"] else resampling_method
            )
            audio_dict = load_audio_soundfile(*args)
        else:
            raise NotImplementedError(f"Backend '{backend}' is not supported.")
    except Exception as e:
        print(
            f"Failed to load audio {path} with backend {backend} and args {args},  exception: {e}"
        )
        return None, None

    audio = (
        audio_dict["waveform"].unsqueeze(0)
        if audio_dict["waveform"].ndim == 1
        else audio_dict["waveform"]
    )
    # HEre expects tensor to be torch,
    if duration and audio.size(1) < int(sample_rate_target * duration):
        padding = torch.full(
            (audio.shape[0], int(sample_rate_target * duration) - audio.size(1)), 0.0
        )
        audio = torch.cat((audio, padding), dim=1)

    # Check if it is all 0
    # if torch.all(audio == 0):
    if not activity_detector(audio):
        normalize = False
    if normalize:
        try:
            normalizer = get_normalizing_function(normalizing_function)
            # waveform = torch.nn.functional.normalize(waveform, dim=1)
            audio = normalizer(audio)
        except Exception as e:
            print(f"Failed to normalize audio: {e}")

    frame_offset = audio_dict.get("frame_offset", 0)
    return (audio.squeeze(), sample_rate_target, frame_offset)


def get_metadata_torchaudio(path, duration, random_crop, offset_sec=0):
    try:
        metadata = torchaudio.info(path)
        total_frames = metadata.num_frames
        sample_rate = metadata.sample_rate

        # Determine the number of frames to load
        if duration == -1:
            duration = None
        num_frames = int(duration * sample_rate) if duration else total_frames

        if random_crop and duration:
            max_start_frame = total_frames - num_frames - int(offset_sec * sample_rate)
            max_start_frame = max(max_start_frame, 1)
            frame_offset = torch.randint(0, max_start_frame, (1,)).item() + int(
                offset_sec * sample_rate
            )
        else:
            frame_offset = offset_sec * sample_rate

        return metadata, int(frame_offset), num_frames, sample_rate, total_frames

    except Exception as e:
        print(f"Failed to load audio metadata: {e}")
        return None


def load_audio_torchaudio(
    path, duration, sample_rate_target, random_crop, mono, resampling_method, offset_sec
):
    metadata, frame_offset, num_frames, sample_rate, total_frames = get_metadata_torchaudio(
        path, duration, random_crop, offset_sec
    )
    # Load the specified frames
    # frame_offset = int(offset_sec * sample_rate) + frame_offset
    waveform, sample_rate = torchaudio.load(
        path,
        frame_offset=frame_offset,
        num_frames=num_frames if frame_offset + num_frames <= total_frames else total_frames,
    )

    if waveform.size(0) > 1 and mono:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != sample_rate_target:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=sample_rate_target, resampling_method=resampling_method
        )
        if activity_detector(waveform):
            waveform = resampler(waveform)
        else:
            waveform = torch.zeros(int(np.ceil(num_frames * sample_rate_target / sample_rate)))
        # Also convert frame_offset to the new sample rate
        frame_offset = int(frame_offset * sample_rate_target / sample_rate)

    return {"waveform": waveform, "frame_offset": frame_offset}


def load_audio_scipy_soxr(
    path, duration, sample_rate_target, random_crop, mono, resampling_method, offset_sec
):
    sample_rate, audio = wavfile.read(path)
    audio = audio.astype(float) / 32768.0
    if audio.ndim > 1 and mono:
        audio = audio.mean(axis=1, keepdims=True)
        # audio = audio.reshape(1, -1)

    audio = audio.T

    if random_crop and duration:
        max_start_frame = audio.shape[-1] - int(duration * sample_rate_target)
        start_frame = torch.randint(0, max_start_frame, (1,)).item()
        audio = audio[:, start_frame : start_frame + int(duration * sample_rate_target)]
    elif duration:
        # load using offset_sec
        start_frame = int(offset_sec * sample_rate)
        audio = audio[:, start_frame : start_frame + int(duration * sample_rate_target)]
    else:
        start_frame = 0
        duration = audio.shape[-1] / sample_rate

    if sample_rate != sample_rate_target:
        if audio.shape[0] == 1:
            audio = audio[0]
        audio = soxr.resample(audio, sample_rate, sample_rate_target, quality=resampling_method)

        start_frame = int(start_frame * sample_rate_target / sample_rate)

    return {"waveform": torch.tensor(audio), "frame_offset": start_frame}


def load_audio_soundfile(
    path, duration, sample_rate_target, random_crop, mono, resampling_method, offset_sec
):
    info = sf.info(path)
    sample_rate = info.samplerate
    total_frames = info.frames
    if random_crop and duration:
        max_start_frame = total_frames - int(duration * sample_rate)
        max_start_frame = max(max_start_frame, 1)
        start_frame = torch.randint(0, max_start_frame, (1,)).item() + int(offset_sec * sample_rate)
    else:
        start_frame = int(offset_sec * sample_rate)

    if duration:
        num_frames = int(duration * sample_rate)
    else:
        num_frames = total_frames - start_frame

    wv, sample_rate = sf.read(
        path,
        start=start_frame,
        frames=num_frames,
        always_2d=True,
        dtype="float32",
    )

    if wv.ndim > 1 and mono:
        wv = wv.mean(axis=1, keepdims=True)

    if sample_rate != sample_rate_target:
        wv = soxr.resample(wv, sample_rate, sample_rate_target, quality=resampling_method)
        start_frame = int(start_frame * sample_rate_target / sample_rate)
    wv = torch.from_numpy(wv)
    # soundfile returns (time, channels), we need (channels, time)
    wv = wv.permute(1, 0)[:, : int(duration * sample_rate_target) if duration else -1]

    return {"waveform": wv, "frame_offset": start_frame}
