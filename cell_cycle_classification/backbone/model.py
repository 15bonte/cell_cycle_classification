import os
import numpy as np
import torch
import torch.nn.functional as F

from pytorch_metric_learning import losses

from pythae.data.datasets import BaseDataset
from pythae.models.beta_vae import BetaVAE
from pythae.models.base.base_utils import ModelOutput
from pythae.models.vamp import VAMP
from pythae.models.disentangled_beta_vae import DisentangledBetaVAE

from ..vae_training.fucci_vae_config import BetaVAEConfig

# Hard-coded values for the custom KLD loss -- not used in the final implementation
B = 12
A = 6 / B**2
OPTIMAL_MEAN = np.sqrt(B) / 2
OPTIMAL_STD = 1
OPTIMAL_KL = -3 / 4


class FucciVAE(BetaVAE, DisentangledBetaVAE, VAMP):
    def __init__(
        self,
        model_config,
        encoder=None,
        decoder=None,
    ):
        BetaVAE.__init__(self, model_config, encoder, decoder)

        # NB: DisentangledBetaVAE and VAMP are finally not used in the final implementation
        DisentangledBetaVAE.__init__(self, model_config, encoder, decoder)
        VAMP.__init__(self, model_config, encoder, decoder)

        self.zeta = model_config.zeta  # image reconstruction
        self.delta = model_config.delta  # FUCCI reconstruction
        self.gamma = model_config.gamma  # adjacent loss

        self.kld_loss = model_config.kld_loss

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        """
        Necessary to test models since pythae does not know our BetaVAEConfig
        """
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = BetaVAEConfig.from_json_file(path_to_model_config)

        return model_config

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The (modified) VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        y = (
            inputs["target"]
            if "target" in inputs
            else torch.zeros((x.shape[0], 2 * x.shape[1], x.shape[2], x.shape[3])).to(
                x.device
            )
        )  # FUCCI channels reconstruction -- not used in the final implementation
        track_ids = inputs["track_id"]

        epoch = kwargs.pop("epoch", self.warmup_epoch)

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        decoder_output = self.decoder(z)

        recon = decoder_output["reconstruction"]

        recon_x = recon[:, : x.shape[1], :, :]  # B, C, H, W
        recon_fucci = recon[:, x.shape[1] :, :, :]  # B, C, H, W

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z, epoch)

        # Reconstruct FUCCI channels
        # prediction_loss = self._fucci_reconstruction_loss(recon_fucci, y)

        # Predict average FUCCI values
        prediction_loss = self._fucci_avg_prediction_loss(
            decoder_output.fucci, inputs["fucci"]
        )
        loss = loss + prediction_loss

        # Make sure adjacent nuclei are close in latent space
        adjacent_loss = self._adjacent_loss(inputs["adjacent_dapi"], z, x, track_ids)
        loss = loss + adjacent_loss

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld * self.beta,  # others are already multiplied by scalar
            prediction_loss=prediction_loss,
            loss=loss,
            recon_x=recon_x,
            z=z,
            recon_target=recon_fucci,
            adjacent_loss=adjacent_loss,
            inactive_dim_loss=(torch.var(z, dim=0) < 0.01).sum(),
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z, epoch):
        if self.kld_loss == "standard":
            _, recon_loss, KLD = BetaVAE.loss_function(self, recon_x, x, mu, log_var, z)

        elif self.kld_loss == "vamp":
            # In VAMP, beta is hard-coded to 1
            # Here, we use self.beta to multiply the KLD loss
            _, recon_loss, KLD = VAMP.loss_function(
                self, recon_x, x, mu, log_var, z, epoch
            )
            if self.linear_scheduling > 0:
                beta = self.beta * epoch / self.linear_scheduling
                if beta > self.beta or not self.training:
                    beta = self.beta
            else:
                beta = self.beta

        elif self.kld_loss == "disentangled":
            _, recon_loss, KLD = DisentangledBetaVAE.loss_function(
                self, recon_x, x, mu, log_var, z, epoch
            )

        elif self.kld_loss == "custom":
            _, recon_loss, _ = BetaVAE.loss_function(self, recon_x, x, mu, log_var, z)
            KLD = torch.sum(
                A
                * (
                    3 * log_var.exp().pow(4)
                    + 6 * log_var.exp().pow(2) * mu.pow(2)
                    + mu.pow(4)
                    - B * mu.pow(2)
                    - B * log_var.exp()
                )
                - log_var
                - OPTIMAL_KL,
                dim=-1,
            )
            KLD = KLD.mean(dim=0)

        else:
            raise ValueError(f"Invalid KLD loss: {self.kld_loss}")

        return self.zeta * recon_loss + self.beta * KLD, recon_loss, KLD

    def _fucci_avg_prediction_loss(self, pred_fucci, fucci):
        if self.delta == 0:
            return torch.tensor(0)

        prediction_loss = F.mse_loss(
            pred_fucci.reshape(fucci.shape[0], -1),
            fucci.reshape(fucci.shape[0], -1),
            reduction="none",
        ).sum(dim=-1)

        return self.delta * prediction_loss.mean(dim=0)

    def _fucci_reconstruction_loss(self, recon_target, target):
        if self.delta == 0:
            return torch.tensor(0)

        if self.model_config.reconstruction_loss == "mse":
            prediction_loss = F.mse_loss(
                recon_target.reshape(target.shape[0], -1),
                target.reshape(target.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        if self.model_config.reconstruction_loss == "reduced_mse":
            prediction_loss = F.mse_loss(
                recon_target.reshape(target.shape[0], -1),
                target.reshape(target.shape[0], -1),
                reduction="mean",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "l1":
            prediction_loss = F.l1_loss(
                recon_target.reshape(target.shape[0], -1),
                target.reshape(target.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        return self.delta * prediction_loss.mean(dim=0)

    def _adjacent_loss(self, adjacent_dapi, z, x, labels, loss="ntxent"):
        if self.gamma == 0:
            return torch.tensor(0)

        # Remove eventual duplicates
        labels_np = labels.cpu().numpy()
        unique_values = np.unique(labels_np, return_index=True)[1]
        sorted_unique_values = torch.tensor(np.sort(unique_values))

        # Keep only the first occurrence of each label
        adjacent_dapi = adjacent_dapi[sorted_unique_values]
        z = z[sorted_unique_values]
        x = x[sorted_unique_values]
        labels = labels[sorted_unique_values]

        # Before
        before_x = adjacent_dapi[:, : x.shape[1], :, :]
        before_output = self.encoder(before_x)
        before_embedding, before_log_var = (
            before_output.embedding,
            before_output.log_covariance,
        )
        before_std = torch.exp(0.5 * before_log_var)
        before_z, _ = self._sample_gauss(before_embedding, before_std)

        # After
        after_x = adjacent_dapi[:, x.shape[1] :, :, :]
        after_output = self.encoder(after_x)
        after_embedding, after_log_var = (
            self.encoder(after_x).embedding,
            after_output.log_covariance,
        )
        after_std = torch.exp(0.5 * after_log_var)
        after_z, _ = self._sample_gauss(after_embedding, after_std)

        all_embeddings = torch.cat([before_z, after_z, z], dim=0)
        all_labels = torch.cat([labels, labels, labels], dim=0)

        if loss == "ntxent":
            adjacent_loss = losses.NTXentLoss().forward(all_embeddings, all_labels)
        elif loss == "lifted":
            adjacent_loss = losses.GeneralizedLiftedStructureLoss().forward(
                all_embeddings, all_labels
            )
        else:
            raise ValueError(f"Invalid loss: {loss}")

        adjacent_loss = self.gamma * adjacent_loss
        return adjacent_loss

    def predict(self, inputs: torch.Tensor) -> ModelOutput:
        """The input data is encoded and decoded without computing loss

        Args:
            inputs (torch.Tensor): The input data to be reconstructed, as well as to generate the embedding.

        Returns:
            ModelOutput: An instance of ModelOutput containing reconstruction and embedding
        """
        z = self.encoder(inputs).embedding
        recon = self.decoder(z)["reconstruction"]

        recon_x = recon[:, : inputs.shape[1], :, :]  # B, C, H, W
        recon_target = recon[:, inputs.shape[1] :, :, :]  # B, C, H, W

        output = ModelOutput(
            recon_x=recon_x,
            embedding=z,
            recon_target=recon_target,
        )

        return output
