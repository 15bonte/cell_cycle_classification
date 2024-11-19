"""Adapted from https://github.com/clementchadebec/benchmark_VAE/blob/6419e21558f2a6abc2da99944bddda846ded30f4/src/pythae/trainers/base_trainer/base_trainer.py#L36.
Main change is the addition of the additional_losses dictionary in the train_step and eval_step functions."""

import logging
import os
from copy import deepcopy
from typing import Optional
import torch

import torch.distributed as dist

from pythae.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class CustomTrainer(BaseTrainer):
    def train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        self.callback_handler.on_train_step_begin(
            training_config=self.training_config,
            train_loader=self.train_loader,
            epoch=epoch,
            rank=self.rank,
        )

        # set model in train model
        self.model.train()

        epoch_loss: float = 0
        additional_losses: dict[str, float] = {}

        for inputs in self.train_loader:
            inputs = self._set_inputs_to_device(inputs)

            with self.amp_context:
                model_output = self.model(
                    inputs,
                    epoch=epoch,
                    dataset_size=len(self.train_loader.dataset),
                    uses_ddp=self.distributed,
                )

            self._optimizers_step(model_output)

            loss = model_output.loss

            epoch_loss += loss.item()

            # Modified code
            additional_loss_keys = [
                key for key in model_output.keys() if "_loss" in key
            ]
            for additional_loss_key in additional_loss_keys:
                train_key = "additional/train_epoch_" + additional_loss_key
                if train_key not in additional_losses:
                    additional_losses[train_key] = 0
                additional_losses[train_key] += model_output[additional_loss_key].item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        if self.distributed:
            self.model.module.update()
        else:
            self.model.update()

        epoch_loss /= len(self.train_loader)

        # Modified code
        additional_losses = {
            key: value / len(self.train_loader)
            for key, value in additional_losses.items()
        }

        return epoch_loss, additional_losses

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.callback_handler.on_eval_step_begin(
            training_config=self.training_config,
            eval_loader=self.eval_loader,
            epoch=epoch,
            rank=self.rank,
        )

        self.model.eval()

        epoch_loss: float = 0
        additional_losses: dict[str, float] = {}

        with self.amp_context:
            for inputs in self.eval_loader:
                inputs = self._set_inputs_to_device(inputs)

                try:
                    with torch.no_grad():
                        model_output = self.model(
                            inputs,
                            epoch=epoch,
                            dataset_size=len(self.eval_loader.dataset),
                            uses_ddp=self.distributed,
                        )

                except RuntimeError:
                    model_output = self.model(
                        inputs,
                        epoch=epoch,
                        dataset_size=len(self.eval_loader.dataset),
                        uses_ddp=self.distributed,
                    )

                loss = model_output.loss

                epoch_loss += loss.item()

                # Modified code
                additional_loss_keys = [
                    key for key in model_output.keys() if "_loss" in key
                ]
                for additional_loss_key in additional_loss_keys:
                    train_key = "additional/eval_epoch_" + additional_loss_key
                    if train_key not in additional_losses:
                        additional_losses[train_key] = 0
                    additional_losses[train_key] += model_output[
                        additional_loss_key
                    ].item()

                if epoch_loss != epoch_loss:
                    raise ArithmeticError("NaN detected in eval loss")

                self.callback_handler.on_eval_step_end(
                    training_config=self.training_config
                )

        epoch_loss /= len(self.eval_loader)

        # Modified code
        additional_losses = {
            key: value / len(self.eval_loader)
            for key, value in additional_losses.items()
        }

        return epoch_loss, additional_losses

    def train(self, log_output_dir: Optional[str] = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        self.prepare_training()

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model_config
        )

        log_verbose = False

        msg = (
            f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
            " - per_device_train_batch_size: "
            f"{self.training_config.per_device_train_batch_size}\n"
            " - per_device_eval_batch_size: "
            f"{self.training_config.per_device_eval_batch_size}\n"
            f" - checkpoint saving every: {self.training_config.steps_saving}\n"
            f"Optimizer: {self.optimizer}\n"
            f"Scheduler: {self.scheduler}\n"
        )

        if self.is_main_process:
            logger.info(msg)

        # set up log file
        if log_output_dir is not None and self.is_main_process:
            log_verbose = True
            file_logger = self._get_file_logger(log_output_dir=log_output_dir)

            file_logger.info(msg)

        if self.is_main_process:
            logger.info("Successfully launched training !\n")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10
        best_model = deepcopy(
            self.model
        )  # difference from original code - enables to run with 0 epochs

        for epoch in range(1, self.training_config.num_epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            epoch_train_loss, metrics = self.train_step(epoch)
            metrics["train_epoch_loss"] = epoch_train_loss

            if self.eval_dataset is not None:
                epoch_eval_loss, eval_metrics = self.eval_step(epoch)
                metrics.update(eval_metrics)
                metrics["eval_epoch_loss"] = epoch_eval_loss
                self._schedulers_step(epoch_eval_loss)

            else:
                epoch_eval_loss = best_eval_loss
                self._schedulers_step(epoch_train_loss)

            if (
                epoch_eval_loss < best_eval_loss
                and not self.training_config.keep_best_on_train
            ):
                best_eval_loss = epoch_eval_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            if (
                self.training_config.steps_predict is not None
                and epoch % self.training_config.steps_predict == 0
                and self.is_main_process
            ):
                true_data, reconstructions, generations = self.predict(best_model)

                self.callback_handler.on_prediction_step(
                    self.training_config,
                    true_data=true_data,
                    reconstructions=reconstructions,
                    generations=generations,
                    global_step=epoch,
                )

            self.callback_handler.on_epoch_end(training_config=self.training_config)

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                if self.is_main_process:
                    self.save_checkpoint(
                        model=best_model, dir_path=self.training_dir, epoch=epoch
                    )
                    logger.info(f"Saved checkpoint at epoch {epoch}\n")

                    if log_verbose:
                        file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log(
                self.training_config,
                metrics,
                logger=logger,
                global_step=epoch,
                rank=self.rank,
            )

        final_dir = os.path.join(self.training_dir, "final_model")

        if self.is_main_process:
            self.save_model(best_model, dir_path=final_dir)

            logger.info("Training ended!")
            logger.info(f"Saved final model in {final_dir}")

        if self.distributed:
            dist.destroy_process_group()

        self.callback_handler.on_train_end(self.training_config)
