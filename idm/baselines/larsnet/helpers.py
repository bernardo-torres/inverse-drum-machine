import soxr
import torch


# def aggregate_larsnet_outputs(x, eval_classes):
class AggregateLarsnetOutputs(torch.nn.Module):
    def __init__(self, eval_classes):
        super().__init__()
        self.eval_classes = eval_classes  # Ex KD, SD, HH, CY, etc.

    def forward(self, x):
        # Input is a dictionary with keys as classes and values as tensors
        return torch.stack([x[cls].mean(dim=0) for cls in self.eval_classes], dim=0).unsqueeze(0)


class MonoToStereo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.shape[1] == 2:
            return x
        return x.repeat(1, 2, 1)


class BandwidthRemover(torch.nn.Module):
    "We use a high quality resampler to avoid aliasing artifacts"

    def __init__(self, fcut, sample_rate, upsample=True, downsample=True):
        super().__init__()
        self.fcut = fcut
        self.sample_rate = sample_rate
        self.upsample = upsample
        self.downsample = downsample

    def forward(self, x):
        """x: Tensor of shape [1, time] or [1, chan, time]
        non batched inputs only for now"""

        x = x.squeeze(0)
        if x.ndim > 2:
            raise ValueError("Input must be mono or stereo")
        if x.ndim == 2:
            x = x.permute(1, 0)  # [time, chan]

        x = x.numpy()
        if self.downsample:
            x = soxr.resample(x, self.sample_rate, self.fcut, quality="VHQ")
        if self.upsample:
            x = soxr.resample(x, self.fcut, self.sample_rate, quality="VHQ")
        x = torch.tensor(x).unsqueeze(0)
        if x.ndim == 3:  # stereo
            x = x.permute(0, 2, 1)
        else:  # mono
            x = x.unsqueeze(1)

        return x
