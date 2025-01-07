import argparse
from cnn_framework.utils.parsers.vae_parser import VAEParser


class FucciVAEParser(VAEParser):
    def __init__(self):
        super().__init__()

        self.arguments_parser.add_argument(
            "--prediction_loss", help="mse or reduced_mse"
        )
        self.arguments_parser.add_argument(
            "--data_dir_additional", help="Additional data directory"
        )
        self.arguments_parser.add_argument(
            "--lr_additional", help="Additional learning rate"
        )
        self.arguments_parser.add_argument(
            "--c",
            help="Value of the KL divergence term of the ELBO we wish to approach",
        )
        self.arguments_parser.add_argument("--zeta", help="reconstruction loss weight")
        self.arguments_parser.add_argument(
            "--display_umap", action=argparse.BooleanOptionalAction
        )
        self.arguments_parser.add_argument("--pretraining", default="vae")
        self.arguments_parser.add_argument(
            "--freeze_backbone", action=argparse.BooleanOptionalAction, default=True
        )
        self.arguments_parser.add_argument("--warmup", default=0)
        self.arguments_parser.add_argument("--number_components", default=50)
