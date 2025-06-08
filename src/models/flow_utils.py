import wandb
import functools
from omegaconf import DictConfig, OmegaConf

@functools.cache
def generate_id():
    return wandb.util.generate_id()


def register_omega_conf_resolvers():
    OmegaConf.register_new_resolver(
        "do_ip",
        lambda x: True if x == "non_symmetric" else False,
    )
    OmegaConf.register_new_resolver(
        "get_dim_atomic_rep",
        lambda x: NUM_ATOMIC_BITS if x == "analog_bits" else NUM_ATOMIC_TYPES,
    )
    OmegaConf.register_new_resolver("generate_id", generate_id)
    OmegaConf.register_new_resolver("get_flowmm_version", lambda: flowmm.__version__)