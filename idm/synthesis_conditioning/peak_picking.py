import librosa
import torch
import torch.nn as nn

from idm.utils import convert_onset_dict_to_activations, cpu_numpy


def pickpeaks(
    impulse,
    sampling_rate=None,
    div_max=20,
    div_avg=10,
    div_wait=16,
    div_thre=5,
    normalize=False,
    pre_normalize_threshold=0.05,
):
    """Peak-picking with batch support.
    Implementation follows "Deep Unsupervised Drum Transcription" by Keunwoo Choi and Kyunghyun Cho, ISMIR 2019.

    sampling_rate is for audio sample domain.
    """
    if impulse.ndim == 2:
        impulse = impulse[:, None, :]

    results = []
    for b in range(impulse.shape[0]):
        class_results = []
        for c in range(impulse.shape[1]):
            signal = impulse[b, c].copy()
            if normalize:
                signal_max = signal.max()
                if signal_max > pre_normalize_threshold:
                    signal = signal / max(impulse[b, c].max(), 1e-8)
            peaks = librosa.util.peak_pick(
                signal,
                pre_max=sampling_rate // div_max,
                post_max=sampling_rate // div_max,
                pre_avg=sampling_rate // div_avg,
                post_avg=sampling_rate // div_avg,
                delta=1.0 / div_thre,
                wait=sampling_rate // div_wait,
            )
            class_results.append(librosa.samples_to_time(peaks, sr=sampling_rate))
        results.append(class_results)
    return results


class PeakPicking(nn.Module):
    def __init__(
        self,
        activation_rate=None,
        classes=None,
        div_max=20,
        div_avg=10,
        div_wait=16,
        div_thre=5,
        normalize=False,
    ):
        super().__init__()
        self.activation_rate = activation_rate
        self.classes = classes
        self.div_max = div_max
        self.div_avg = div_avg
        self.div_wait = div_wait
        self.div_thre = div_thre
        self.normalize = normalize

    def forward(self, impulse, activation_rate=None):
        peaks = pickpeaks(
            cpu_numpy(impulse),
            sampling_rate=self.activation_rate if activation_rate is None else activation_rate,
            div_max=self.div_max,
            div_avg=self.div_avg,
            div_wait=self.div_wait,
            div_thre=self.div_thre,
            normalize=self.normalize,
        )

        num_frames = impulse.shape[-1]
        classes = self.classes
        if self.classes is None:
            classes = list(range(impulse.shape[1]))
        onsets_dict = {inst: torch.tensor(peaks[0][i]) for i, inst in enumerate(classes)}

        activations = convert_onset_dict_to_activations(
            onsets_dict,
            num_frames,
            classes,
            batch_size=impulse.shape[0],
            label_mapping={inst: inst for inst in classes},
            activation_rate=self.sampling_rate if activation_rate is None else activation_rate,
        ).to(impulse.device)

        return activations
