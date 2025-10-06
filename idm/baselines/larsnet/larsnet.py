from pathlib import Path
from typing import Union

import torch
import torchaudio as ta
import yaml
from tqdm import tqdm

from idm.baselines.larsnet.unet import UNet, UNetUtils, UNetWaveform
from idm.idm import BaseTrainer
from idm.utils import cpu_numpy


class LarsNet(BaseTrainer):

    def __init__(
        self,
        wiener_filter: bool = False,
        wiener_exponent: float = 1.0,
        config: Union[str, Path] = "config.yaml",
        return_stft: bool = False,
        device: str = "cpu",
        # sampling_rate: int = 44100,
        version: str = "full",
        train_classes=None,
        skip_test=False,
        verbose=False,
        mono=False,
        **kwargs,
    ):
        kwargs.pop("sampling_rate", None)
        super().__init__(**kwargs)

        with open(config) as f:
            config = yaml.safe_load(f)

        self.models = {}
        # self.device = device
        self.to(device)
        self.mono = mono
        self.wiener_filter = wiener_filter
        self.wiener_exponent = wiener_exponent
        self.return_stft = return_stft
        self.stems = config["inference_models"].keys()
        self.utils = UNetUtils(device=self.device)
        self.verbose = verbose
        self.sr = config["global"]["sr"]
        self.sampling_rate = self.sr
        self.skip_test = skip_test
        self.train_classes = train_classes

        if wiener_filter:
            print(f"> Applying Wiener filter with α={self.wiener_exponent}")

        print("Loading UNet models...")
        pbar = tqdm(self.stems)
        for stem in pbar:
            checkpoint_path = Path(config["inference_models"][stem])
            pbar.set_description(f"{stem} {checkpoint_path.stem}")

            F = config[stem]["F"]
            T = config[stem]["T"]

            if self.wiener_filter or self.return_stft:
                model = UNet(input_size=(2, F, T), device=self.device)
            else:
                model = UNetWaveform(input_size=(2, F, T), device=self.device)

            checkpoint = torch.load(str(checkpoint_path), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.models[stem] = model

    @staticmethod
    def _fix_dim(x):
        if x.dim() == 1:
            x = x.repeat(2, 1)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x

    def _verbose_msg(self, msg="Separate drums..."):
        if self.verbose:
            print(msg)
            pbar = tqdm(self.models.items())
        else:
            pbar = self.models.items()
        return pbar

    def separate(self, x):
        with torch.no_grad():
            out = {}
            x = x.to(self.device)

            pbar = self._verbose_msg()
            for stem, model in pbar:
                pbar.set_description(stem) if self.verbose else None
                y, __ = model(x)
                out[stem] = y.squeeze(0).detach()

        return out

    def separate_wiener(self, x):
        with torch.no_grad():
            out = {}
            mag_pred = []

            x = self._fix_dim(x).to(self.device)
            mag, phase = self.utils.batch_stft(x)

            pbar = self._verbose_msg()
            for stem, model in pbar:
                pbar.set_description(stem) if self.verbose else None
                __, mask = model(mag)
                mag_pred.append((mask * mag) ** self.wiener_exponent)

            pred_sum = sum(mag_pred)

            for stem, pred in zip(self.stems, mag_pred):
                wiener_mask = pred / (pred_sum + 1e-7)
                y = self.utils.batch_istft(mag * wiener_mask, phase, trim_length=x.size(-1))
                out[stem] = y.squeeze(0).detach()

        return out

    def separate_stft(self, x):
        with torch.no_grad():
            out = {}

            x = self._fix_dim(x).to(self.device)
            mag, phase = self.utils.batch_stft(x)

            pbar = self._verbose_msg("Separate drum magnitude...")
            for stem, model in pbar:
                pbar.set_description(stem) if self.verbose else None
                mag_pred, __ = model(mag)
                stft = torch.polar(mag_pred, phase)
                out[stem] = stft.squeeze(0).detach()

        return out

    def forward(self, x, **kwargs):
        if isinstance(x, (str, Path)):
            x, sr_ = ta.load(str(x))
            if sr_ != self.sr:
                x = ta.functional.resample(x, sr_, self.sr)

        if self.return_stft:
            return self.separate_stft(x)
        elif self.wiener_filter:
            return self.separate_wiener(x)
        else:
            return self.separate(x)

    def validation_step(self, batch, batch_idx, stage="val"):
        mix = batch["mix"]
        outs = self.forward(mix)
        return cpu_numpy(outs)

    def test_step(self, batch, batch_idx):
        if self.skip_test:
            return
        return self.validation_step(batch, batch_idx, stage="test")
