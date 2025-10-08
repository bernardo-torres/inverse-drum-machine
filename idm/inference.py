import os
import re
from pathlib import Path

import hydra
import rootutils
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf

from idm import CHECKPOINT_TYPE_BEST, CHECKPOINT_TYPE_LAST
from idm.feature_extractor.stft import STFT

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
ROOT_PATH = Path(rootutils.find_root())

try:
    NORBERT = True
except:
    print("Norbert not found. Please install norbert to use wiener_norbert_mask.")
    NORBERT = None


def load_model(identifier, device):
    # Get root from rootutils
    """Loads a model based on an identifier string."""
    print(f"Loading model: {identifier}")
    if "oracle" in identifier or "gt" in identifier:
        return identifier, identifier  # Just return the string as a placeholder
    if "larsnet" in identifier:
        from idm.baselines.larsnet.larsnet import LarsNet

        model = LarsNet(
            wiener_filter=False,
            device=device,
            wiener_exponent=1.0,
            config=ROOT_PATH / "configs/baselines/larsnet/config.yaml",
            train_classes=[
                "CY_CR",
                "CY_RD",
                "HH_CHH",
                "HH_OHH",
                "KD",
                "SD",
                "TT_HFT",
                "TT_HMT",
                "TT_LMT",
            ],
            mono="mono" in identifier,
        )
        name = "larsnet_mono" if "mono" in identifier else "larsnet_stereo"
        return model.to(device).eval(), name
    elif "nmfd" in identifier:
        # let see if we already have already registered the resolvers
        case = identifier.split("_")[-1]
        config_path = f"configs/baselines/NMFD/nmfd_amen_{case}.yaml"
        config_path = ROOT_PATH / config_path
        ckpt_path = None
        name = f"nmfd_{case}"

    else:  # Assume it's a hash for a trained model
        # log_dir = Path("logs/train/dora_xps/grids/")
        log_dir = ROOT_PATH / "logs"
        # ckpt_path = find_ckpt_from_hash(log_dir, identifier, type="val")
        ckpt_path = find_checkpoint(
            log_dir, version_id=identifier, ckpt_type=CHECKPOINT_TYPE_BEST, filename_contains="val"
        )
        name = identifier
        if not ckpt_path:
            raise FileNotFoundError(f"Could not find checkpoint for hash {identifier} in {log_dir}")
        print(f"Found checkpoint at: {os.path.relpath(ckpt_path, ROOT_PATH)}")
        exp_dir = Path(ckpt_path).parent.parent
        config_path = exp_dir / ".hydra" / "config_resolved.yaml"
    model_cfg = OmegaConf.load(config_path).model
    model = hydra.utils.instantiate(model_cfg)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    return model.to(device).eval(), name


def load_model_from_hash(model_hash: str, logs_dir: str = "logs/train/dora_xps/grids/"):
    """
    Loads a model from a given hash.
    """
    # ckpt_path = find_ckpt_from_hash(logs_dir, model_hash)
    ckpt_path = find_checkpoint(logs_dir, version_id=model_hash, ckpt_type=CHECKPOINT_TYPE_BEST)
    if ckpt_path is None:
        raise FileNotFoundError(f"Could not find checkpoint for hash {model_hash}")

    exp_dir = Path(ckpt_path).parent.parent
    config_path = exp_dir / ".hydra" / "config_resolved.yaml"
    cfg = OmegaConf.load(config_path)

    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model


def wiener_mask(sources_transform, mix_transform, alpha=2.0):
    """
    Args:
        sources_transform: transforms of the estimated sources (real or complex) [batch, inst, freq, time]
        mix_transform: mixture transform (complex) [batch, freq, time]

    Returns:
        filtered source estimated obtained by alpha wiener filtering as in [1] [batch, inst, freq, time]


    [1] A. Liutkus and R. Badeau, “Generalized Wiener filtering with fractional power spectrograms,” in ICASSP, IEEE, 2015, pp. 266–270.
    """

    M = mix_transform
    S = torch.abs(sources_transform) ** alpha
    M_hat = torch.sum(S, dim=1)  # abs before or after sum?
    batch, inst, freq, time = S.shape

    S = torch.abs(S)  # get the magnitude if it is complex
    S_hat = []
    for i in range(inst):
        mask = S[:, i] / (M_hat + 1e-12)
        s_hat_i = mask * M  # Filtering
        S_hat.append(s_hat_i)
    return torch.stack(S_hat, dim=1)


def estimate_masks(mix_transform, estimates_transform, masking_type="soft", alpha=2.0):
    args = (estimates_transform, mix_transform)
    if masking_type == "wiener":
        return wiener_mask(*args, alpha=alpha)
    elif masking_type == "wiener_norbert":
        if NORBERT is None:
            raise ValueError(
                "Norbert not found. Please install norbert to use wiener_norbert_mask."
            )
        return wiener_norbert_mask(*args)
    else:
        raise ValueError('Invalid masking type. Choose either "soft" or "wiener".')


def find_checkpoint(
    search_dir: str,
    version_id: str | None = None,
    ckpt_type: str = CHECKPOINT_TYPE_BEST,
    filename_contains: str | None = None,
) -> str | None:
    """
    Recursively finds a checkpoint file in a directory structure.

    Args:
        search_dir: The top-level directory to start the search from (e.g., "logs/").
        version_id: An optional unique identifier (like a hash) for a specific run.
        ckpt_type: The type of checkpoint to find ("last" or "best_epoch").
        filename_contains: An optional string that must be present in the
                           checkpoint's filename (e.g., "val" for validation checkpoints).

    Returns:
        The path to the found checkpoint file as a string, or None if not found.
    """
    base_path = Path(search_dir)
    if not base_path.is_dir():
        return None

    # Define a flexible search pattern that can optionally filter by version_id
    search_prefix = f"{version_id}*/" if version_id else "**/"

    # Find all potential checkpoint files recursively
    ckpt_files = list(base_path.rglob(search_prefix + "checkpoints/*.ckpt"))
    pth_files = list(base_path.rglob(search_prefix + "weights/*.pth"))
    all_matches = ckpt_files + pth_files

    if filename_contains:
        all_matches = [p for p in all_matches if filename_contains in p.name]
    # -----------------------------------------------------------------

    if not all_matches:
        return None

    if ckpt_type == CHECKPOINT_TYPE_LAST:
        last_checkpoints = [p for p in all_matches if p.name == "last.ckpt"]
        if not last_checkpoints:
            return None
        # Return the most recently modified one
        return str(max(last_checkpoints, key=os.path.getmtime))

    if ckpt_type == CHECKPOINT_TYPE_BEST:
        epoch_checkpoints = [p for p in all_matches if p.name != "last.ckpt"]
        if not epoch_checkpoints:
            return None

        def get_epoch_from_filename(path: Path) -> int:
            match = re.search(r"epoch=(\d+)", path.name)
            return int(match.group(1)) if match else -1

        # Find the checkpoint with the highest epoch number
        best_ckpt = max(epoch_checkpoints, key=get_epoch_from_filename)

        if get_epoch_from_filename(best_ckpt) > -1:
            return str(best_ckpt)
        else:
            return None

    raise ValueError(f"Unknown ckpt_type: '{ckpt_type}'. Must be 'last' or 'best_epoch'.")


