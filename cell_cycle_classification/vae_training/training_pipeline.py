"""Adapted from https://github.com/clementchadebec/benchmark_VAE/blob/6419e21558f2a6abc2da99944bddda846ded30f4/src/pythae/pipelines/training.py#L23.
Main change is the import of CustomTrainer instead of Trainer."""

import logging
from typing import Optional, Union

import numpy as np
import torch

from pythae.pipelines.training import TrainingPipeline
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.models.base.base_model import BaseAE
from pythae.trainers.base_trainer.base_training_config import BaseTrainerConfig

from .trainer import CustomTrainer

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class CustomTrainingPipeline(TrainingPipeline):
    def __init__(
        self,
        model: Optional[BaseAE],
        training_config: Optional[BaseTrainerConfig] = None,
    ):
        super().__init__(model, training_config)
        self.trainer: Optional[CustomTrainer] = None

    def __call__(
        self,
        train_data: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset],
        eval_data: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset] = None,
        callbacks: Optional[list[TrainingCallback]] = None,
    ):
        """
        Launch the model training on the provided data.

        Args:
            training_data (Union[~numpy.ndarray, ~torch.Tensor]): The training data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...)

            eval_data (Optional[Union[~numpy.ndarray, ~torch.Tensor]]): The evaluation data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...). If None, only uses train_fata for training. Default: None.

            callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
                A list of callbacks to use during training.
        """

        if isinstance(train_data, np.ndarray) or isinstance(train_data, torch.Tensor):
            logger.info("Preprocessing train data...")
            train_data = self.data_processor.process_data(train_data)
            train_dataset = self.data_processor.to_dataset(train_data)

        else:
            train_dataset = train_data

        logger.info("Checking train dataset...")
        self._check_dataset(train_dataset)

        if eval_data is not None:
            if isinstance(eval_data, np.ndarray) or isinstance(eval_data, torch.Tensor):
                logger.info("Preprocessing eval data...\n")
                eval_data = self.data_processor.process_data(eval_data)
                eval_dataset = self.data_processor.to_dataset(eval_data)

            else:
                eval_dataset = eval_data

            logger.info("Checking eval dataset...")
            self._check_dataset(eval_dataset)

        else:
            eval_dataset = None

        logger.info("Using Custom Trainer\n")
        self.trainer = CustomTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.training_config,
            callbacks=callbacks,
        )

        self.trainer.train()
