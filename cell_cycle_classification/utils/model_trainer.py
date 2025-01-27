"""Module performing both self-supervised and classification learning."""

import json
import os
import shutil
from copy import deepcopy
import pickle
from torch import optim
import torch.nn as nn
import torch.distributed as dist


from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import WandbCallback

from cnn_framework.utils.data_loader_generators.data_loader_generator import (
    DataLoaderGenerator,
)
from cnn_framework.utils.model_managers.cnn_model_manager import CnnModelManager
from cnn_framework.utils.metrics.classification_accuracy import ClassificationAccuracy
from cnn_framework.utils.data_loader_generators.classifier_data_loader_generator import (
    ClassifierDataLoaderGenerator,
)
from cnn_framework.utils.data_managers.classification_data_manager import (
    ClassificationDataManager,
)
from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.metrics.ssim import SSIM
from cnn_framework.utils.data_loader_generators.data_loader_generator import (
    get_mean_and_std,
    post_process_mean_std,
)
from cnn_framework.utils.enum import PredictMode


from ..backbone.decoder_resnet import ResnetDecoder
from ..backbone.encoder_resnet import ResnetEncoder
from ..backbone.fucci_classifier import FucciClassifier
from ..backbone.model import FucciVAE

from ..vae_training.training_pipeline import CustomTrainingPipeline

from .data_set import (
    FucciClassificationDataSet,
    FucciVAEDataSet,
)
from .euclidean_matching_metric import (
    EuclideanMatchingMetric,
)
from .tools import get_final_model_path, get_vae_config
from .umap_tools import fit_umap_on_train, predict_umap
from .vae_model_manager import VAEModelManager
from .model_params import FucciVAEModelParams


class ModelTrainer:

    def ssl_train(self, args, params: FucciVAEModelParams):
        assert args.pretraining == "vae"

        print("\n### Training VAE on current data set ###\n")
        params.update(args)  # update for VAE training
        params.check_ready()

        params.out_channels = 2 * len(params.z_indexes)  # 2: FUCCI green and red
        reconstruction_score = self._core_training_vae(
            params, data_set_class=FucciVAEDataSet, adapt_decoder=False
        )
        args.model_pretrained_path = params.models_folder
        return reconstruction_score

    def _core_training_vae(self, params, data_set_class, adapt_decoder):
        loader_generator = DataLoaderGenerator(
            params, data_set_class, DefaultDataManager
        )
        train_dl, val_dl, test_dl = loader_generator.generate_data_loader(
            shuffle_test=True  # for EuclideanMetricMatching computation
        )
        print("Note: test is shuffled for EuclideanMetricMatching computation.")

        # Create folder to save model
        os.makedirs(params.models_folder, exist_ok=True)

        # Set up the training configuration
        my_training_config = BaseTrainerConfig(
            output_dir=params.models_folder,
            num_epochs=params.num_epochs,
            learning_rate=params.learning_rate,
            per_device_train_batch_size=params.batch_size,
            per_device_eval_batch_size=params.batch_size,
            train_dataloader_num_workers=params.num_workers,
            eval_dataloader_num_workers=params.num_workers,
            steps_saving=None,
            optimizer_cls="AdamW",
            optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
            scheduler_cls="ReduceLROnPlateau",
            scheduler_params={"patience": 5, "factor": 0.5},
        )

        # Set up the model configuration
        my_vae_config = get_vae_config(params)
        vae_model = self._get_vae_model(params, my_vae_config, adapt_decoder)

        # Display number of parameters
        print(
            f"Number of parameters in encoder: {sum(p.numel() for p in vae_model.encoder.parameters())}"
        )
        print(
            f"Number of parameters in decoder: {sum(p.numel() for p in vae_model.decoder.parameters())}"
        )

        # Build the Pipeline
        pipeline = CustomTrainingPipeline(
            training_config=my_training_config, model=vae_model
        )

        model_manager = VAEModelManager(vae_model, params, SSIM)

        # Compute mean and std
        data_set_mean_std = get_mean_and_std([train_dl])
        # Transform it to mimic Pythae behavior
        new_mean_std = post_process_mean_std(data_set_mean_std, mode="isl")

        train_dl.dataset.mean_std = new_mean_std
        val_dl.dataset.mean_std = new_mean_std

        # Save in model folder (duplicated from model manager code...)
        mean_std_file = os.path.join(
            model_manager.params.models_folder, "mean_std.json"
        )
        with open(mean_std_file, "w") as write_file:
            json.dump(new_mean_std, write_file, indent=4)

        train_dl.dataset.initialize_transforms()
        val_dl.dataset.initialize_transforms()
        vae_model.model_config.mean_std = new_mean_std

        # Plot few images here
        # display_examples_pythae_loader(train_dl)

        # Create WandB callback
        is_local = params.format_now.split("-")[-1] == "local"
        callbacks = []  # the TrainingPipeline expects a list of callbacks
        wandb_cb = WandbCallback()  # Build the callback
        # SetUp the callback
        wandb_cb.setup(
            training_config=my_training_config,  # training config
            model_config=my_vae_config,  # model config
            project_name=f"vae-fucci{'-local' if is_local else ''}",  # specify your wandb project
            entity_name="cbio-bis",  # specify your wandb entity
            run_name=params.format_now,  # name of the run
        )
        callbacks.append(wandb_cb)  # Add it to the callbacks list

        # Launch the Pipeline
        try:
            pipeline(
                train_data=train_dl.dataset,  # must be torch.Tensor, np.array or torch datasets
                eval_data=val_dl.dataset if len(val_dl) else None,
                callbacks=callbacks,
            )
        except ArithmeticError:
            # If NaN detected in loss, stop training and kill everything properly
            print("\nNaN detected in loss")
            if pipeline.trainer.distributed:
                dist.destroy_process_group()
            pipeline.trainer.callback_handler.on_train_end(
                pipeline.trainer.training_config
            )
            return 0

        # Test and save images
        model_manager.predict(test_dl)
        reconstruction_score = model_manager.training_information.score
        model_manager.write_useful_information()

        model_manager.metric_class = EuclideanMatchingMetric
        model_manager.params.gamma = 1  # to ensure adjacent embedding will be computed
        model_manager.predict(test_dl, predict_mode=PredictMode.GetEmbeddingMSE)

        return reconstruction_score

    def _get_vae_model(self, params, config, adapt_decoder):
        assert adapt_decoder is False
        # If provided, load pretrained model
        if params.model_pretrained_path:
            final_model_path = get_final_model_path(params.model_pretrained_path)
            vae_model = FucciVAE.load_from_folder(final_model_path)
        else:
            assert params.encoder_name in ["resnet18", "resnet50"]
            encoder = ResnetEncoder(params, config)
            decoder = ResnetDecoder(params, config)
            vae_model = FucciVAE(encoder=encoder, decoder=decoder, model_config=config)

        return vae_model

    def classification_train(self, args, params):
        # Hard-coded classification parameters
        args.epochs = 1 if args.freeze_backbone else 10  # hard code nb epochs
        args.lr = 1e-4  # hard code to 1e-4
        args.batch_size = 64  # hard code to 64, 128 crashed
        if args.data_dir_additional:
            args.data_dir = (
                args.data_dir_additional
            )  # if specified, switch to classification data folder
        params.update(args)  # update for classification training

        # Copy mean_std from pretraining to classification
        os.makedirs(params.models_folder, exist_ok=True)
        shutil.copy(
            os.path.join(params.model_pretrained_path, "mean_std.json"),
            params.models_folder,
        )

        if args.display_umap:
            print("\n### UMAP computation ###\n")
            self._display_latent_space_umap(params)

        print("\n\n### Classification ###\n")
        return self._core_classification_train(params, args)

    def _load_backbone(self, pretraining, folder):
        assert "vae" in pretraining
        vae_model = FucciVAE.load_from_folder(folder)
        return vae_model

    def _display_latent_space_umap(self, params):
        """
        Fit a UMAP on the latent space of the model and display it for train, val and test sets.
        """
        copied_params = deepcopy(params)
        loader_generator = ClassifierDataLoaderGenerator(
            copied_params, FucciVAEDataSet, DefaultDataManager
        )

        # Load pretrained model
        final_model_path = get_final_model_path(copied_params.model_pretrained_path)
        vae_model = self._load_backbone(params.pretraining, final_model_path)
        manager = VAEModelManager(vae_model, copied_params, SSIM)

        # Run predictions
        # Limit number of samples to avoid memory issues
        loader_generator.params.train_ratio = min(
            loader_generator.params.train_ratio, 0.01
        )
        loader_generator.params.val_ratio = min(loader_generator.params.val_ratio, 0.01)
        train_dl, val_dl, test_dl = loader_generator.generate_data_loader(
            train_as_test=True
        )
        umap_files = loader_generator.params.names_train
        print(f"UMAP fit on {len(umap_files)} samples.")
        umap_model = fit_umap_on_train(train_dl, manager, copied_params, save="svg")

        # Save UMAP model with pickle
        umap_model_path = os.path.join(copied_params.output_dir, "umap_model.pkl")
        with open(umap_model_path, "wb") as file:
            pickle.dump(umap_model, file)

        predict_umap(val_dl, umap_model, copied_params, manager, "val", save="svg")
        predict_umap(test_dl, umap_model, copied_params, manager, "test", save="svg")

    def _core_classification_train(self, params, args):
        """Performs deep learning classification from pretrained model"""
        loader_generator = ClassifierDataLoaderGenerator(
            params, FucciClassificationDataSet, ClassificationDataManager
        )
        train_dl, val_dl, _ = loader_generator.generate_data_loader()

        # Load pretrained model
        model = self._get_pretrained_model(params, args)

        print(
            f"Number of parameters to train: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        manager = CnnModelManager(model, params, ClassificationAccuracy)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(params.learning_rate),
            betas=(params.beta1, params.beta2),
        )  # define the optimization
        loss_function = nn.CrossEntropyLoss()  # define the loss function

        manager.fit(
            train_dl,
            val_dl,
            optimizer,
            loss_function,
            mean_std_path=os.path.join(
                params.models_folder, "mean_std.json"
            ),  # done to avoid usual mean_std computation
        )

    def _get_pretrained_model(self, params, args):
        assert "vae" in args.pretraining
        return FucciClassifier(params, args.freeze_backbone)
