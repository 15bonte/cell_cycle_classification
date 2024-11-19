"""Resnet encoder."""

from torch import nn
import torch
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

from pythae.models.nn import BaseEncoder
from pythae.models.base.base_utils import ModelOutput

from cnn_framework.utils.model_managers.utils.custom_get_encoder import get_encoder


def redefine_first_layer(model: nn.Module, input_channels: int) -> None:
    """Redefine the first layer of the model to accept input_channels number of channels."""
    if input_channels >= 3:
        original_conv1_weights = model.conv1.weight.data  # Shape: (64, 3, 7, 7)
        new_conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize the new layer with the original weights for the first 3 channels
        new_conv1.weight.data[:, :3, :, :] = original_conv1_weights
        # Initialize the remaining two channels with average of the existing channels
        new_conv1.weight.data[:, 3:, :, :] = torch.mean(
            original_conv1_weights, dim=1, keepdim=True
        ).repeat(1, input_channels - 3, 1, 1)

        model.conv1 = new_conv1

    else:
        new_conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize the channels with average of the initial channels
        new_conv1.weight.data = torch.mean(
            model.conv1.weight.data, dim=1, keepdim=True
        ).repeat(1, input_channels, 1, 1)

        model.conv1 = new_conv1


class ResnetEncoder(BaseEncoder):
    def __init__(self, params, args):
        BaseEncoder.__init__(self)

        in_channels = len(params.c_indexes) * len(params.z_indexes)

        if params.encoder_name == "resnet18":
            self.conv_layers = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Modify first layer to accept input channels number
            redefine_first_layer(self.conv_layers, in_channels)
            output_size = self.conv_layers.fc.in_features
            self.conv_layers.fc = nn.Identity()  # useless prediction to classes
            self.need_to_take_last_output = False
        elif params.encoder_name == "resnet50":  # used to match cycle_cnn backbone
            self.conv_layers = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Modify first layer to accept input channels number
            redefine_first_layer(self.conv_layers, in_channels)
            output_size = self.conv_layers.fc.in_features
            self.conv_layers.fc = nn.Identity()  # useless prediction to classes
            self.need_to_take_last_output = False
        else:
            print("Deprecated encoder name")
            self.conv_layers = get_encoder(
                params.encoder_name,
                in_channels=in_channels,
                weights=None,  # "imagenet",
                depth=params.depth,
                drop_rate=params.dropout,
            )

            # Infer size of images after convolutions
            # Create random input to infer size of output
            random_input = torch.randn(
                1,
                in_channels,
                params.input_dimensions.height,
                params.input_dimensions.width,
            )
            random_output = self.conv_layers(random_input)
            output_size = random_output[-1].flatten().size(dim=0)
            self.need_to_take_last_output = True

        self.embedding = nn.Linear(output_size, args.latent_dim)
        self.log_var = nn.Linear(output_size, args.latent_dim)

        self.fucci = nn.Sequential(nn.Linear(args.latent_dim, 2), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        if self.need_to_take_last_output:
            h1 = self.conv_layers(x)[-1].reshape(x.shape[0], -1)
        else:
            h1 = self.conv_layers(x)
        # Compute useful outputs
        embedding = self.embedding(h1)
        log_covariance = self.log_var(h1)
        fucci = self.fucci(embedding)

        return ModelOutput(
            embedding=embedding, log_covariance=log_covariance, fucci=fucci
        )
