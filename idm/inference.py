from pathlib import Path

import hydra
import rootutils
import torch
from omegaconf import OmegaConf

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
            config=ROOT_PATH / "baselines/larsnet/config.yaml",
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
        ckpt_path = find_ckpt_from_hash(log_dir, identifier, type="val")
        name = identifier
        if not ckpt_path:
            raise FileNotFoundError(f"Could not find checkpoint for hash {identifier} in {log_dir}")
        print(f"Found checkpoint at: {ckpt_path}")
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
    ckpt_path = find_ckpt_from_hash(logs_dir, model_hash)
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


def find_ckpt_from_hash(log_dir, hash_str, type="val"):
    """Recursively search for a checkpoint file in the given directory that matches the specified hash.

    The expected filename format is:
    log_dir/<hash>/<hash>_datetime/checkpoints/last.ckpt
    but we can have other variations like:
    log_dir/<hash>/<hash>/checkpoints/last.ckpt
    log_dir/<hash>_datetime/<hash>_datetime/checkpoints/last.ckpt
    so we will match the hash in the first level of the directory structure. then
    if we find multiple matches to last.ckpt, we will return the one with the latest modification time.
    """
    from pathlib import Path

    log_dir = Path(log_dir)
    hash_str = str(hash_str)

    # Search for the hash in multiple recursive levels
    matches = list(log_dir.rglob(f"{hash_str}*/checkpoints/*{type}*.ckpt"))
    matches += list(log_dir.rglob(f"{hash_str}*/weights/*{type}*.pth"))
    for path in matches:
        if "latest" in str(path):
            continue
        if path.is_file():
            return str(path)

    return None
