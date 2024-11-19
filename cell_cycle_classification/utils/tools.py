import os

from ..vae_training.fucci_vae_config import BetaVAEConfig


def get_final_model_path(model_pretrained_path: str) -> str:
    """Get list of folders inside model_pretrained_path."""
    sub_folders = [
        dir
        for dir in os.listdir(model_pretrained_path)
        if os.path.isdir(os.path.join(model_pretrained_path, dir))
    ]
    assert len(sub_folders) == 1
    return os.path.join(model_pretrained_path, sub_folders[0], "final_model")


def get_vae_config(params) -> BetaVAEConfig:
    return BetaVAEConfig(
        gamma=params.gamma,
        delta=params.delta,
        zeta=params.zeta,
        nb_classes=params.nb_classes,
        reconstruction_loss=params.reconstruction_loss,
        kld_loss=params.kld_loss,
        input_dim=(
            len(params.c_indexes) * len(params.z_indexes),
            params.input_dimensions.height,
            params.input_dimensions.width,
        ),
        latent_dim=params.latent_dim,
        beta=params.beta,
        C=params.C,
        uses_default_decoder=params.encoder_name == "default",
        uses_default_encoder=params.encoder_name == "default",
        linear_scheduling_steps=params.warmup,
        number_components=params.number_components,
    )
