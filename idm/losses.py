import abc

import auraloss
from torch.nn.modules.loss import _Loss


class BaseLoss(_Loss, abc.ABC):
    """Base class for losses."""

    # Define abstract init with weight

    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight
        self.input_type = None  # To be set by subclasses

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Computes the loss."""
        raise NotImplementedError

    @property
    def input_type(self):
        if self._input_type is None:
            raise NotImplementedError("Subclasses must set the 'input_type'.")
        return self._input_type

    @input_type.setter
    def input_type(self, value):
        self._input_type = value


class MultiScaleLoss(BaseLoss):
    def __init__(self, weight=1.0, dims=None, **kwargs):
        super().__init__(weight=weight)
        self.dims = dims
        self.input_type = "waveform"
        self.loss = auraloss.freq.MultiResolutionSTFTLoss(**kwargs)

    def forward(self, value, target):

        if value.ndim == 2:
            value = value.unsqueeze(1)
        if target.ndim == 2:
            target = target.unsqueeze(1)
        return self.loss(value, target)
