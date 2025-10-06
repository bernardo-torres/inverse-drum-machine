""" This code was adapted from the NMF toolbox available at https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/

[1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.
    """

import torch

EPS = 2.0**-52


def NEMA(A: torch.Tensor, lamb: torch.Tensor = torch.tensor(0.9)):
    """This function takes a 3D tensor of shape (batch, R, M) and applies a non-linear exponential moving average (NEMA)
    to each row along the time axis. This filter introduces exponentially decaying slopes.

    The difference equation of that filter would be:
    y(n) = max( x(n), y(n-1)*(decay) + x(n)*(1-decay) )

    Parameters
    ----------
    A: torch.Tensor
        The tensor with time series in its rows (shape: [batch, R, M])

    lamb: torch.Tensor / float
        The decay parameter in the range [0 ... 1], this can be
        given as a column-vector (shape: [batch, R]) with individual decays per row
        or as a scalar.

    Returns
    -------
    filtered: torch.Tensor
        The result after application of the NEMA filter (same shape as input)
    """

    # Prevent instable filter
    lamb = torch.clamp(lamb, min=0.0, max=0.9999999) if isinstance(lamb, torch.Tensor) else max(0.0, min(lamb, 0.9999999))

    # Get input dimensions
    batch_size, num_rows, num_cols = A.shape

    # Initialize the filtered tensor
    filtered = A.clone()

    # Apply the NEMA filter iteratively along the time axis
    for k in range(1, num_cols):
        store_row = filtered[:, :, k].clone()
        filtered[:, :, k] = lamb * filtered[:, :, k - 1] + filtered[:, :, k] * (1 - lamb)
        filtered[:, :, k] = torch.max(filtered[:, :, k], store_row)

    return filtered
