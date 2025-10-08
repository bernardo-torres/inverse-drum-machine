from pathlib import Path

from omegaconf import OmegaConf

if not __package__:
    pass
else:
    pass


def get_n_of_classes(classes_name_list: list[str]) -> int:
    return len(classes_name_list)


def get_next_power_of_2_for_duration(sampling_rate, duration) -> int:
    n = sampling_rate * duration / 1000  # duration is in ms
    return get_next_power_of_2(int(n))


def get_next_power_of_2(n: int) -> int:
    return 2 ** (n - 1).bit_length()


def get_tcn_causal_padding(kernel_size: int, num_layers: int):
    """Get the padding required to keep the input and output lengths the same for a stack of dilated convolutions.

    Takes into account the kernel size and dilation factor for every layer.
    """
    padding = 0
    dilations = [2 ** (n % num_layers) for n in range(num_layers)]
    kernel_sizes = [kernel_size] * num_layers
    for kernel_size, dilation in zip(kernel_sizes, dilations):
        padding += dilation * (kernel_size - 1)
    return padding


def register_resolvers():
    OmegaConf.register_new_resolver("eval", eval)

    OmegaConf.register_new_resolver("get_n_of_classes", get_n_of_classes)
    OmegaConf.register_new_resolver(
        "get_next_power_of_2_for_duration", get_next_power_of_2_for_duration
    )
    OmegaConf.register_new_resolver("hash", compute_hash)
    OmegaConf.register_new_resolver("get_next_power_of_2", get_next_power_of_2)
    OmegaConf.register_new_resolver("get_tcn_causal_padding", get_tcn_causal_padding)


def compute_hash(*cfgs):
    try:
        import hashlib

        from omegaconf import DictConfig, OmegaConf

        # First resolve any remaining interpolations in the configs
        resolved_cfgs = []
        for cfg in cfgs:
            if isinstance(cfg, DictConfig):
                # Create a copy and resolve it
                resolved_cfg = OmegaConf.create(
                    OmegaConf.to_container(cfg, resolve=True, structured_config_mode=False)
                )
                resolved_cfgs.append(resolved_cfg)
            else:
                resolved_cfgs.append(cfg)

        # Convert to YAML, sorting keys to ensure order-independence
        cfg_str = "\n".join(
            [
                OmegaConf.to_yaml(cfg, sort_keys=True) if isinstance(cfg, DictConfig) else str(cfg)
                for cfg in resolved_cfgs
            ]
        )
        # Lets remove some keys and their values that are not relevant for the hash
        keys_to_remove = ["num_workers", "num_workers_val", "output_dir"]
        for key in keys_to_remove:
            # Remove the whole line
            cfg_str = "\n".join([line for line in cfg_str.split("\n") if key not in line])

        # Save config string to a file for debugging
        with open("config_str.yaml", "w") as f:
            f.write(cfg_str)
        return hashlib.sha256(cfg_str.encode()).hexdigest()[:8]
    except:
        return "nohash"


def get_hash(ckpt_path: str):
    return Path(ckpt_path).parents[1].name
