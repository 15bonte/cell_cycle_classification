""""Cell cycle classifier."""

import torch.nn as nn

from .encoder_resnet import ResnetEncoder
from .model import FucciVAE
from ..utils.tools import get_final_model_path, get_vae_config


class FucciClassifier(nn.Module):
    def __init__(
        self,
        params,
        freeze_backbone=True,
    ):
        super().__init__()

        # Load VAE backbone
        if params.model_pretrained_path:
            vae = self._load_backbone(
                get_final_model_path(params.model_pretrained_path), params.pretraining
            )
            self.encoder = vae.encoder
        else:
            vae_config = get_vae_config(params)
            self.encoder = ResnetEncoder(params, vae_config)

        # Freeze encoder
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Create two fully connected layers
        self.fc1 = nn.Linear(params.latent_dim, params.latent_dim)
        self.fc2 = nn.Linear(params.latent_dim, params.nb_classes)

        # Target layer for GradCam - typically for resnet18 and resnet50
        # self.target_layer = "encoder.conv_layers.layer4[-1]"

    def _load_backbone(self, folder: str, pretraining: str):
        assert "vae" in pretraining
        return FucciVAE.load_from_folder(folder)

    def forward(self, x):
        x = self.encoder(x).embedding
        x = self.fc1(x)
        return self.fc2(x)
