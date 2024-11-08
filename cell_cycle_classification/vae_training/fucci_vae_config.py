from typing import Optional
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from pythae.models.beta_vae.beta_vae_config import BetaVAEConfig as BetaVAEConfigBase
from pythae.models.disentangled_beta_vae.disentangled_beta_vae_config import (
    DisentangledBetaVAEConfig,
)
from pythae.models.vamp.vamp_config import VAMPConfig


@dataclass
class BetaVAEConfig(BetaVAEConfigBase, DisentangledBetaVAEConfig, VAMPConfig):
    r"""
    Parameters:
        nb_classes (int): Number of classes for classification. Default: 3

    NB:
        1 - Naming is not the best, but useful to retrieve the config from the
            model folder
        2 - The config inherits from multiple classes to have all the parameters,
            even if they are not used in the final model
    """

    kld_loss: Literal["standard", "disentangled", "vamp", "custom"] = "standard"

    gamma: float = 0.0
    delta: float = 0.0
    zeta: float = 1.0

    nb_classes: int = 3

    mean_std: Optional[dict[str, list[float]]] = None
