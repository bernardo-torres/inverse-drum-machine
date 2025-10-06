from typing import Optional

import torch
import torch.nn as nn

from idm.utils import get_act_module


def fc(ch_in=256, ch_out=256, activation=torch.nn.LeakyReLU()):
    layers = [
        torch.nn.Linear(ch_in, ch_out),
        # torch.nn.LayerNorm(ch_out),  # normalization is done over the last dimension
        activation,
    ]
    return torch.nn.Sequential(*layers)


def fc_stack(ch_in=256, ch_hidden=256, ch_out=256, layers=2, activation=torch.nn.LeakyReLU()):
    activation = get_act_module(activation)()
    proj_up = [fc(ch_in, ch_hidden, activation=activation)]
    proj_down = [fc(ch_hidden, ch_out, activation=torch.nn.Identity())]
    proj = [fc(ch_hidden, ch_hidden, activation=activation) for i in range(layers - 2)]
    return torch.nn.Sequential(*proj_up + proj + proj_down)


class IdentityWithKwargs(nn.Identity):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


class BatchRandomGain(nn.Module):
    def __init__(
        self,
        min_gain: float = 0.5,
        max_gain: float = 1.5,
        p: Optional[float] = None,
    ):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device

        vol = torch.empty(batch_size, device=device)
        vol.uniform_(self.min_gain, self.max_gain)
        mask = torch.rand_like(vol).ge(self.p)
        vol[mask] = 1
        vol = vol.unsqueeze(-1).expand_as(x.view(batch_size, -1)).view_as(x)
        return vol * x
