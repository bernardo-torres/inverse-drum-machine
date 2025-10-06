import torch
import torch.nn as nn

from idm.components import fc_stack


class ParamsDecoder(nn.Module):
    """Abstract base class for decoders that process and split parameters.

    This class defines a common interface for modules that take a tensor or a
    dictionary of tensors, process them, and output a dictionary of tensors
    corresponding to different "splits" or parameter groups.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the ParamsDecoder."""
        super().__init__(*args, **kwargs)
        self.splits = None
        self.logit_transform = None

    def set_splits(self, splits: dict):
        """Configures the parameter splits for the decoder.

        The splits define how the output is structured and which modules are
        created.

        Args:
            splits: A dictionary where keys are parameter names and values are
                tuples of (split_size, output_size). `split_size` is the number
                of parallel modules (e.g., number of instrument classes), and
                `output_size` is the feature dimension for each module's output.
                An integer value is treated as `(value, 1)`.
        """
        processed_splits = {}
        for name, split in splits.items():
            # If the split is a single integer, format it as (integer, 1)
            if isinstance(split, int):
                split = (split, 1)
            # Ensure the split is a tuple or list of length 2
            assert (
                len(split) == 2
            ), f"Split for '{name}' must be a tuple/list of 2 elements, got {split}"
            processed_splits[name] = split

        self.splits = processed_splits
        self.instantiate_modules()

    def instantiate_modules(self):
        """Instantiates the necessary modules for the forward pass.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError

    def set_logits_transform(self, transform: dict):
        """Applies a final transformation to the output logits.

        Args:
            transform: A dictionary where keys match the split names and
                values are callable activation functions to be applied to the
                corresponding outputs.
        """
        assert isinstance(transform, dict), "Transform must be a dictionary."
        assert set(transform.keys()) == set(
            self.splits.keys()
        ), "Transform keys must match split keys."

        for key, value in transform.items():
            assert callable(value), f"Transform for '{key}' must be a callable function."

        self.logit_transform = transform


class MLPLatentDecoder(ParamsDecoder):
    """Decodes a latent tensor sequence using parallel MLP stacks.

    This module processes a latent tensor by feeding it through multiple parallel
    MLP stacks. Each stack corresponds to a "split" defined in the configuration,
    allowing for the generation of different parameters from a shared latent space.

    Attributes:
        latent_size: The feature dimension of the input latent tensor.
        hidden_size: The feature dimension of the MLP hidden layers.
        activation: The activation function used in the MLPs.
        n_layers: The number of layers in each MLP stack.
    """

    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        activation: str = "leakyrelu",
        n_layers: int = 3,
        splits: dict = {"gain": (11, 1)},
    ):
        """Initializes the MLPLatentDecoder.

        Args:
            latent_size: Size of the input latent tensor's feature dimension.
            hidden_size: Size of the hidden layers of the MLPs.
            activation: Activation function to use in the MLPs.
            n_layers: Number of layers in the MLPs.
            splits: A dictionary defining the output splits. The format is
                `{"split_name": (n_modules, output_size)}`. For example,
                `{"gain": (11, 1)}` creates 11 parallel MLPs for the "gain"
                parameter, each with an output size of 1.
        """
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.n_layers = n_layers

        if splits is not None:
            self.set_splits(splits)

    def instantiate_modules(self):
        """Creates the MLP stacks for each defined split."""
        self.latent_mlps = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        fc_stack(
                            ch_in=self.latent_size,
                            ch_hidden=self.hidden_size,
                            ch_out=out_size,
                            layers=self.n_layers,
                            activation=self.activation,
                        )
                        for _ in range(split_size)
                    ]
                )
                for name, (split_size, out_size) in self.splits.items()
            }
        )

    def forward(self, x: torch.Tensor | dict) -> dict:
        """Processes the input latent tensor through the MLP stacks.

        Args:
            x: An input tensor of shape `(B, T, D)` where D is `latent_size`,
               or a dictionary of tensors for each split.

        Returns:
            A dictionary where keys are split names and values are the
            processed output tensors of shape `(B, n_modules, T, output_size)`.
        """
        input_dict = {}
        if isinstance(x, torch.Tensor):
            # If input is a single tensor, use it for all splits
            input_dict = dict.fromkeys(self.splits.keys(), x)
        elif isinstance(x, dict):
            input_dict = x
        else:
            raise ValueError("Input must be a tensor or a dictionary of tensors.")

        # Process each input tensor through its corresponding MLP list
        output_dict = {}
        for name, mlp_list in self.latent_mlps.items():
            # Stack the outputs of the parallel MLPs along a new dimension
            output_dict[name] = torch.stack([mlp(input_dict[name]) for mlp in mlp_list], dim=1)

        return output_dict


class MLPLatentDecoderWithDownsampling(MLPLatentDecoder):
    """An MLPLatentDecoder that first downsamples the input tensor in time.

    This module extends `MLPLatentDecoder` by adding an average pooling layer
    to reduce the temporal resolution of the input latent sequence before it is
    processed by the MLPs.
    """

    def __init__(self, downsampling_factor: int | dict = 1, *args, **kwargs):
        """Initializes the MLPLatentDecoderWithDownsampling.

        Args:
            downsampling_factor: The factor by which to downsample the time
                dimension. Can be an integer applied to all splits or a
                dictionary mapping split names to specific factors. A factor of
                -1 triggers adaptive pooling to an output size of 1.
            *args: Positional arguments passed to the parent `MLPLatentDecoder`.
            **kwargs: Keyword arguments passed to the parent `MLPLatentDecoder`.
        """
        self.downsampling_factor = downsampling_factor
        super().__init__(*args, **kwargs)

    def instantiate_modules(self):
        """Creates downsampling modules and then the parent MLP stacks."""
        downsamplers = {}
        downsampling_factors = self.downsampling_factor

        # If a single factor is given, create a dict to apply it to all splits
        if isinstance(downsampling_factors, int):
            downsampling_factors = dict.fromkeys(self.splits.keys(), downsampling_factors)

        # Create a downsampler for each split that has a defined factor
        for name in self.splits.keys():
            factor = downsampling_factors.get(name)
            if factor is None or factor == 1:
                downsamplers[name] = nn.Identity()
            elif factor == -1:
                # Adaptive pooling to reduce time dimension to 1
                downsamplers[name] = nn.AdaptiveAvgPool1d(output_size=1)
            else:
                downsamplers[name] = nn.AvgPool1d(kernel_size=factor, stride=factor)

        self.downsamplers = nn.ModuleDict(downsamplers)

        # Instantiate the MLPs from the parent class
        super().instantiate_modules()

    def forward(self, x: torch.Tensor) -> dict:
        """Downsamples the input and then processes it with the parent's forward method.

        Args:
            x: An input tensor of shape `(B, T, D)`.

        Returns:
            A dictionary of processed output tensors from the parent's forward method.
        """
        # (B, T, D) -> (B, D, T) for 1D pooling
        x_permuted = x.permute(0, 2, 1)

        # Apply the specific downsampler for each split
        downsampled_dict = {
            name: self.downsamplers[name](x_permuted).permute(0, 2, 1)
            for name in self.splits.keys()
        }

        # Pass the dictionary of downsampled tensors to the parent's forward method
        return super().forward(downsampled_dict)
