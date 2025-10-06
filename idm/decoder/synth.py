import torch.nn as nn

from idm.synthesis_conditioning.params_decoder import (
    MLPLatentDecoderWithDownsampling,
)
from idm.utils import get_act_functional, resample


class Synth(nn.Module):
    """Base class for synthesizer components. Subclasses must define the `input_type` and `synth_params` properties,
    which specify the type of input the synthesizer expects and the parameters it outputs, respectively.
    Args:
        logit_transform (callable | dict, optional): Function or dictionary of functions to transform
            the output parameters. Defaults to identity function.
        *args: Additional arguments for the latent to parameters decoder.
        downsampling_factor (int, optional): Factor by which to downsample the latent representations in time.
            Defaults to -1 (AveragePooling1d).
        **kwargs: Additional keyword arguments for the latent to parameters decoder.
    """

    def __init__(
        self,
        logit_transform=lambda x: x,
        *args,
        downsampling_factor=-1,
        **kwargs,
    ):
        super().__init__()
        self.input_type = None
        self.synth_params = None
        self.logit_transform = logit_transform
        self.latent_to_params = MLPLatentDecoderWithDownsampling(
            *args, downsampling_factor=downsampling_factor, **kwargs
        )

    @property
    def input_type(self):
        if self._input_type is None:
            raise NotImplementedError("Subclasses must set the 'input_type'.")
        return self._input_type

    @input_type.setter
    def input_type(self, value):
        self._input_type = value

    @property
    def synth_params(self):
        if self._synth_params is None:
            raise NotImplementedError("Subclasses must set the 'synth_params'.")
        return self._synth_params

    @synth_params.setter
    def synth_params(self, value):
        self._synth_params = value

    def get_params(self, latents):
        """Map latents to synth parameters, applying any necessary transformations.

        Args:
            latents (torch.Tensor): Latent representations.

        Returns:
            dict | torch.Tensor: Synth parameters after applying transformations.
        """
        return self.transform_logits(self.latent_to_params(latents))

    def transform_logits(self, params):
        """Apply the logit transformation to the parameters.
        Args:
            params (dict | torch.Tensor): Synth parameters before transformation.
        Returns:
            dict | torch.Tensor: Synth parameters after transformation.
        """
        if isinstance(self.logit_transform, dict):
            return {k: self.logit_transform[k](v) for k, v in params.items()}
        else:
            return self.logit_transform(params)

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError


class MultiTrackTimeVaryingGain(Synth):
    """_summary_

    Args:
        Synth (_type_): _description_
    """

    def __init__(
        self,
        interpolate=True,
        n_tracks=1,
        downsampling_factor=-1,
        interpolation_method="window",
        *args,
        **kwargs,
    ):
        self.splits = {"gain": (n_tracks, 1)}
        self.logit_transform = {"gain": get_act_functional("exp_sigmoid")}
        kwargs["splits"] = self.splits
        kwargs["logit_transform"] = self.logit_transform
        super().__init__(*args, downsampling_factor=downsampling_factor, **kwargs)
        self.splits = {"gain": (n_tracks, 1)}
        self.input_type = "waveform"
        self.synth_params = ["gain"]

        self.interpolate = interpolate
        self.interpolation_method = interpolation_method

    def forward(self, audio, latents=None):
        """Apply time-varying gain to multi-track audio.
        Args:
            audio (torch.Tensor): Input audio of shape (batch, n_tracks, n_samples).
            latents (torch.Tensor): Latent representations
        """
        # gains = self.latent_to_params(latents)["gain"]  # batch, n_tracks
        gains = self.get_params(latents)["gain"]  # batch, n_tracks

        if gains.ndim == 4:
            gains = gains[:, :, :, 0]
        assert (
            gains.shape[1] == audio.shape[1]
        ), f"Number of tracks mismatch: {gains.shape[1]} != {audio.shape[1]}"
        if self.interpolate:
            # Interpolate synth_params to match the audio length
            n_timesteps = audio.shape[-1]
            gains = resample(
                gains.permute(0, 2, 1), n_timesteps, method=self.interpolation_method
            ).permute(0, 2, 1)
        return audio * gains
