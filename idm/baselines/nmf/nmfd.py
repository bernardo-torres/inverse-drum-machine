import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnmf.nmf import NMFD

from idm.baselines.nmf.utils import EPS, NEMA
from idm.idm import BaseTrainer, convert_onset_dict_to_activations
from idm.inference import wiener_mask
from idm.utils import cpu_numpy


class NMFDSeparator(BaseTrainer):
    """NMF-based source separation model.
    Args:
        transform (nn.Module): TF transform to apply to the input signal waveform (e.g., mag STFT, identity)
        weiner_filter (bool): whether to apply weiner filter
        weiner_exponent (float): weiner filter exponent

        see torchnmf's documentation for the rest of the arguments https://pytorch-nmf.readthedocs.io/en/stable/

    """

    def __init__(
        self,
        transform: nn.Module,
        wiener_filter: bool = False,
        wiener_exponent: float = 1.0,
        R: int = None,
        lamb=0.55,
        apply_nema=True,
        beta: float = 1,
        tol: float = 0.0001,
        max_iter: int = 200,
        verbose: bool = False,
        alpha: float = 0,
        l1_ratio=0,
        trainable_W=False,
        trainable_H=False,
        random_W_init=False,
        constant_H_init=False,
        template_length=5,
        cascade=False,
        force_crop_template=False,
        compensate_delay=True,
        EPS=EPS,
        train_classes=None,
        skip_test=False,
        version="full",
        high_precision=False,
        sampling_rate=16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.activation_rate = self.transform.get_frame_rate(sampling_rate)
        self.train_classes = train_classes
        self.lamb = lamb
        self.apply_nema = apply_nema
        self.wiener_filter = wiener_filter
        self.wiener_exponent = wiener_exponent
        self.EPS = EPS
        self.skip_test = skip_test
        self.high_precision = high_precision

        # Delay compensation for the left padding added before the TF transform of the samples
        # See transform code for details
        self.compensate_delay = compensate_delay
        self.delay_compensation = 0
        if hasattr(self.transform, "left_right_pad_amount") and self.compensate_delay:
            self.delay_compensation = (
                self.transform.left_right_pad_amount // self.transform.hop_length
            )

        self.fit_done = False
        self.net = None

        if self.apply_nema:
            #     H = NEMA(H, self.lamb)
            self.onset_relaxation = "nema"

        self.R = R
        self.trainable_W = trainable_W
        self.trainable_H = trainable_H
        self.random_W_init = random_W_init
        self.constant_H_init = constant_H_init
        self.template_length = template_length  # Only used for random initialization
        self.cascade = cascade
        self.force_crop_template = force_crop_template
        self.fit_kwargs = {
            "beta": beta,
            "tol": tol,
            "max_iter": max_iter,
            "verbose": verbose,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }

        self.dummy_parameter = nn.Parameter(
            torch.zeros(1), requires_grad=True
        )  # Just so our trainer does not bug when registering optimizers

    def _pad(self, x):
        return F.pad(x, (self.pad, self.pad), mode="constant", value=0)

    def fit(self, S, W, H):
        with torch.set_grad_enabled(True):
            self.net = NMFD(
                W=W, H=H, trainable_W=self.trainable_W, trainable_H=self.trainable_H
            ).to(S.device)

            self.net.fit(S, **self.fit_kwargs)

    def reconstruct_separately(self, W, H):
        """Reconstructs the sources separately using the learned W and H.

        Args:
            W (torch.Tensor): source magnitudes, shape (K, R, T)
            H (torch.Tensor): activations, shape (1, R, M)
        Returns:
            torch.Tensor: reconstructed stems, shape (1, R, K, M)
        """
        R = W.shape[1]
        isolated_outs = []
        for r in range(R):
            _W = W[:, r].unsqueeze(1)
            _H = H[:, r].unsqueeze(1)
            isolated_outs.append(self.net.reconstruct(H=_H, W=_W))
        return torch.stack(isolated_outs, dim=1)  # (B, R, K, M)

    # Todo add computation of stft directly in the forward method
    def forward(
        self, x=None, sources=None, ext_onsets=None, refit=True, return_keys=[], mix=False, **kwargs
    ):
        #   def forward(self, x=None, sources=None, ext_onsets=None, refit=True, return_keys=[], mix=False):
        """
        Args:
            x (torch.Tensor): mixture signal, shape (batch, time) or (batch, K, M) if self.transform is identity
            sources (torch.Tensor): source signals, shape (batch, R, source_time) or (batch, R, K, T) if self.transform is identity
            ext_onsets (dict): dictionary containing the onset activations, shape (batch, R, M)
            refit (bool): whether to refit the model
            return_keys (list): list of keys to return
            mix (bool): whether to return the mixture signal or the separated sources

        R = number of sources, equal to NMF rank
        K = number of frequencies of magnitude spectrogram
        T = number of frames for the templates/sources
        M = number of frames for the mixture spectrogram

        Returns:
            dict: dictionary containing the output
            Possible keys are:
                - "spec_input": mixture signal in the frequency domain, shape (batch, K, M)
                - "source_specs": source/template magnitudes (W), shape (batch, R, K, T)
                - "activations": activations of the NMF model (H), shape (batch, R, M)
                - "spec_output": separated source magnitudes, shape (batch, R, K, T) if mix is False else (batch, K, T)
        """

        if x is None:
            return self.net()

        S = self.transform(x)

        if S.dim() == 2:
            S = S.unsqueeze(0)
        elif S.dim() == 3 and S.shape[0] > 1:
            raise ValueError("Batch processing not supported")

        _, K, M = S.shape

        if self.constant_H_init:
            if sources is not None:
                self.R = sources.shape[1]
            assert (
                self.R is not None
            ), "Number of sources must be specified for constant initialization"
            H = torch.ones(1, self.R, M, device=S.device)
        else:
            if isinstance(ext_onsets, dict):
                num_frames = M
                H = convert_onset_dict_to_activations(
                    ext_onsets,
                    num_frames,
                    self.train_classes,
                    batch_size=1,
                    label_mapping={inst: inst for inst in self.train_classes},
                    activation_rate=self.activation_rate,
                ).to(S.device)
            elif isinstance(ext_onsets, torch.Tensor):
                H = ext_onsets

        _, R, _M = H.shape

        if self.random_W_init:
            # Nonnegative random initialization
            W = torch.rand(R, K, self.template_length, device=S.device)
        else:
            W = self.transform(sources, adjust_padding=False)

        if W.ndim == 4:
            W = W[0]  # Remove batch dimension

        W = W.permute(1, 0, 2)  # (K, R, T)
        outs = {}
        if not self.fit_done or refit:
            if self.force_crop_template:
                W = W[..., : self.template_length + self.delay_compensation]
            _K, _R, T = W.shape
            assert (
                _K == K
            ), f"Number of frequencies in target and template must match, got {K} and {S.shape[1]}"
            assert (
                _M == M
            ), f"Number of frames in target and activations must match, got {M} and {H.shape[-1]}"
            assert (
                _R == R
            ), f"Number of sources in target and template must match, got {R} and {sources.shape[1]}"

            H = torch.roll(H, -self.delay_compensation, dims=-1)
            H = H[..., : M - T + 1]
            H = torch.where(H == 0, torch.tensor(self.EPS, device=H.device), H)
            if self.trainable_W and not self.random_W_init:
                W = torch.where(W == 0, torch.tensor(self.EPS, device=W.device), W)
            outs["init_H"] = H
            outs["init_W"] = W.permute(1, 0, 2)
            # Non linear exponential moving average on initial activations
            if self.apply_nema:
                H = NEMA(H, self.lamb)

            if self.high_precision:
                H = H.double()
                W = W.double()
                S = S.double()

            self.fit(S, W, H)

            # Let's reconstruct for each component
            reconstructions = self.reconstruct_separately(self.net.W, self.net.H)
            if reconstructions.isnan().any():
                pass
            if self.cascade:
                outs["pre_cascade_spec_output"] = reconstructions
                outs["pre_cascade_sources"] = self.net.W.detach().permute(1, 0, 2)
                # reconstructions = wiener_filter(S, reconstructions, self.wiener_exponent)
                # In the cross talk we use the learned W and the score-informed H
                wiener_separated = wiener_mask(reconstructions, S, self.wiener_exponent)
                reconstructions = self.cascaded_cross_talk(wiener_separated, self.net.W, H)
            self.fit_done = True

        if mix:
            outs["spec_output"] = self.net()
        else:
            outs["spec_output"] = reconstructions

        if "spec_input" in return_keys:
            outs["spec_input"] = S
        if "source_specs" in return_keys:
            outs["source_specs"] = self.net.W.detach().permute(1, 0, 2)
        if "activations" in return_keys:
            outs["activations"] = self.net.H.squeeze().detach()

        return outs

    def cascaded_cross_talk(self, V_sep, W_learned, H_init):
        """Reruns the NMFD model on each of the extracted stems to remove cross-talk.

        Uses the previously learned W and score-informed H.
        Args:
            V (torch.Tensor): separated sources, shape (1, R, K, M)
            W (torch.Tensor): source magnitudes, shape (K, R, T)
            H (torch.Tensor): activations, shape (1, R, M)
        Returns:
            torch.Tensor: separated sources, shape (1, R, K, M)
        """

        isolated_outs = []
        stem_outs = []
        R = V_sep.shape[1]
        assert (
            V_sep.shape[1] == W_learned.shape[1]
        ), "Number of sources in target and template must match"

        for r in range(R):
            self.fit(V_sep[:, r], W_learned, H_init)
            V_sep_out = self.reconstruct_separately(self.net.W, self.net.H)  # (1, R, K, M)
            stem_out = V_sep_out[:, r]  # (1, K, M)  # this one will be returned
            isolated_outs.append(V_sep_out)
            stem_outs.append(stem_out)

        hh_idx = self.eval_classes.index("HH")
        # Sum HH contributions from all reseparated stems
        stem_outs[hh_idx] = torch.sum(
            torch.stack([x[:, hh_idx] for x in isolated_outs]), dim=0
        )  # (1, K, M)
        return torch.stack(stem_outs, dim=1)  # (1, R, K, M)

    def validation_step(self, batch, batch_idx, stage="val"):
        mix = batch["mix"]
        sources = batch["gt_sources"]
        sources = torch.stack([sources[inst] for inst in self.train_classes], dim=1)

        outs = self.forward(
            x=mix,
            sources=sources,
            ext_onsets=batch["onsets_dict"],
            refit=True,
            return_keys=["spec_output"],
        )

        return cpu_numpy(outs)

    def test_step(self, batch, batch_idx):
        if self.skip_test:
            return
        return self.validation_step(batch, batch_idx, stage="test")
