from abc import abstractmethod

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class BaseCallback(Callback):
    def __init__(
        self,
        *args,
        every_n_epochs=1,
        batch_log_step=10,
        batch_idx_to_log=None,
        max_figures=2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.every_n_epochs = every_n_epochs  # All inherited classes will have this attribute but it is up to them to use it
        self.batch_log_step = batch_log_step
        self.batch_idx_to_log = (
            batch_idx_to_log if batch_idx_to_log is not None else [100000000000000]
        )
        self.max_figures = max_figures

    def check_epoch(self, current_epoch):
        if current_epoch is None:
            return True
        return current_epoch % self.every_n_epochs == 0

    def one_element_model_inference(
        self, pl_module, batch, batch_idx=0, return_keys=[], mix_out_sources=True
    ):
        """Perform inference on one element of the batch."""
        idx = self.batch_element_to_log
        mix = batch["mix"][idx].unsqueeze(0)
        sources = (
            batch["audio_samples"][idx].unsqueeze(0)
            if batch.get("audio_samples") is not None
            else None
        )
        onsets_dict = batch["onsets_dict"]
        onsets_dict = {key: onsets_dict[key][idx] for key in onsets_dict}

        return pl_module(
            mix,
            onsets_dict=onsets_dict,
            return_keys=return_keys,
            sources=sources,
            mix_out_sources=mix_out_sources,
        )

    def _check(self, batch_idx=None, epoch=None):
        if not self.check_epoch(epoch):
            return False
        if batch_idx is None:
            return True
        if batch_idx in self.batch_idx_to_log:
            return True
        if (
            batch_idx % self.batch_log_step == 0
            and batch_idx // self.batch_log_step < self.max_figures
        ):
            return True
        return False

    def on_test_start(self, trainer, pl_module):
        self.on_validation_start(trainer, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not self._check(batch_idx=batch_idx, epoch=trainer.current_epoch):
            return
        self._shared_step(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage="val"
        )
        plt.close("all")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self._check(batch_idx=batch_idx, epoch=trainer.current_epoch):
            return
        self._shared_step(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage="test"
        )
        plt.close("all")

    @abstractmethod
    def _shared_step(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0, stage="val"
    ):
        pass
