"""Decoder that mimics Resnet18 architecture.
Adapted from https://github.com/eleannavali/resnet-18-autoencoder/tree/main."""

from typing import Callable, Optional, Type
import torch.nn as nn
from torch import Tensor

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder


def conv3x3_transposed(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    output_padding: int = 0,
) -> nn.ConvTranspose2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding=output_padding,  # output_padding is neccessary to invert conv2d with stride > 1
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1_transposed(
    in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0
) -> nn.ConvTranspose2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        output_padding=output_padding,
    )


class BasicBlockDec(nn.Module):
    """The basic block architecture of resnet-18 network."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_transposed(
            planes, inplanes, stride, output_padding=output_padding
        )
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_transposed(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResnetDecoder(BaseDecoder):
    def __init__(
        self,
        params,
        config,
        block: Type[BasicBlockDec] = BasicBlockDec,
        layers: list[int] = [2, 2, 2, 2],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        BaseDecoder.__init__(self)

        # Infer size of images after convolutions
        self.first_square_size = params.input_dimensions.height
        for _ in range(5):
            self.first_square_size = self.first_square_size // 2

        self.fc = nn.Linear(
            config.latent_dim,
            512 * self.first_square_size * self.first_square_size,
        )

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # change from 2048 to 64. It should be the shape of the output image chanel.
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=1,
            output_padding=0,
            last_block_dim=64,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockDec) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        # Note: Given the non-invertible nature of max pooling, we employed bilinear upsampling with a scale factor of 2.
        # This technique effectively achieves the desired size, emulating the inversion of max pooling.
        self.head = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(
                64,
                params.out_channels + len(params.c_indexes) * len(params.z_indexes),
                kernel_size=8,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        # FUCCI prediction from latent representation
        self.fucci = nn.Sequential(nn.Linear(params.latent_dim, 2), nn.Sigmoid())

    def _make_layer(
        self,
        block: Type[BasicBlockDec],
        planes: int,
        blocks: int,
        stride: int = 2,
        output_padding: int = 1,  # NOTE: output_padding will correct the dimensions of inverting conv2d with stride > 1.
        # More info:https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        last_block_dim: int = 0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        layers = []

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        if last_block_dim == 0:
            last_block_dim = self.inplanes // 2

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                conv1x1_transposed(
                    planes * block.expansion, last_block_dim, stride, output_padding
                ),
                norm_layer(last_block_dim),
            )

        layers.append(
            block(
                last_block_dim,
                planes,
                stride,
                output_padding,
                upsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        return nn.Sequential(*layers)

    def _forward_impl(self, z: Tensor) -> Tensor:
        fucci = self.fucci(z)

        x = self.fc(z).reshape(
            z.shape[0],
            512,
            self.first_square_size,
            self.first_square_size,
        )

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        reconstruction = self.head(x)
        output = ModelOutput(reconstruction=reconstruction, fucci=fucci)

        return output

    def forward(self, z: Tensor) -> Tensor:
        return self._forward_impl(z)
