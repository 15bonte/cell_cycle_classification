from matplotlib import pyplot as plt
import numpy as np

from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.display_tools import display_progress
from cnn_framework.utils.model_managers.vae_model_manager import (
    VAEModelManager as BaseVAEModelManager,
)
from cnn_framework.utils.enum import PredictMode


class VAEModelManager(BaseVAEModelManager):

    def adjacent_embedding_computation(self, dl_element, dl_metric):
        """
        Function to generate outputs from inputs for given model.
        """
        # Compute the model output
        dl_element["data"] = dl_element["data"].to(self.device)
        embedding = self.model.predict(dl_element["data"]).embedding

        dl_element["adjacent_dapi"] = dl_element["adjacent_dapi"].to(self.device)
        # Before
        emebdding_before = self.model.predict(
            dl_element["adjacent_dapi"][:, : dl_element["data"].shape[1], :, :]
        ).embedding
        # After
        emebdding_after = self.model.predict(
            dl_element["adjacent_dapi"][:, dl_element["data"].shape[1] :, :, :]
        ).embedding

        # Update metric
        dl_metric.update(
            embedding,
            emebdding_before,
        )
        dl_metric.update(
            embedding,
            emebdding_after,
        )

    def batch_predict(
        self,
        test_dl,
        images_to_save,
        num_batches_test,
        test_metric,
        predict_mode: PredictMode,
        post_processing=None,
    ) -> list[np.ndarray]:
        all_predictions_np: list[np.ndarray] = []
        for batch_idx, dl_element in enumerate(test_dl):
            # Run prediction
            if predict_mode == PredictMode.GetEmbedding:
                predictions = self.get_embedding(dl_element)
                predictions_np = predictions.cpu().detach().numpy()
                if post_processing is not None:
                    predictions_np = post_processing(predictions_np)
            elif predict_mode == PredictMode.Standard:
                self.model_prediction(dl_element, test_metric, test_dl)
                predictions, predictions_target = (
                    dl_element.prediction["recon_x"],
                    dl_element.prediction["recon_target"],
                )
                predictions_np = predictions.cpu().detach().numpy()
            elif predict_mode == PredictMode.GetEmbeddingMSE:
                self.adjacent_embedding_computation(dl_element, test_metric)
                predictions_np = np.array([])
            else:
                raise ValueError(f"Unknown predict mode: {predict_mode}")

            all_predictions_np = all_predictions_np + [*predictions_np]

            display_progress(
                "Model evaluation in progress",
                batch_idx + 1,
                num_batches_test,
                additional_message=f"Batch #{batch_idx}",
            )

            # Save few images
            if predict_mode != PredictMode.Standard:
                continue

            # Get numpy elements
            inputs_np = dl_element["data"].cpu().detach().numpy()
            targets_np = dl_element["target"].cpu().detach().numpy()
            predictions_target = predictions_target.cpu().detach().numpy()

            for idx in range(dl_element["target"].shape[0]):
                if self.image_index in images_to_save:
                    image_id = (batch_idx * test_dl.batch_size) + idx
                    image_name = test_dl.dataset.names[image_id].split(".")[0]

                    # Get element at index
                    input_np = inputs_np[idx, ...].squeeze()
                    target_np = targets_np[idx, ...].squeeze()
                    prediction_np = predictions_np[idx, ...].squeeze()
                    prediction_target_np = predictions_target[idx, ...].squeeze()

                    dl_element_numpy = DatasetOutput(
                        input=input_np,
                        target_image=target_np,
                        prediction=prediction_target_np,
                        additional=prediction_np,
                    )

                    self.save_results(
                        f"{image_name}_{self.image_index}",
                        dl_element_numpy,
                        test_dl.dataset.mean_std,
                    )

                self.image_index += 1

        return all_predictions_np

    def write_images_to_tensorboard(self, current_batch, dl_element, name):
        # Get numpy arrays
        targets_np = dl_element["target"]
        image_indexes_np = dl_element["id"]

        # Get images name
        current_dl_file_names = [
            file_name.split(".")[0] for file_name in self.dl[name].dataset.names
        ]
        image_names = [
            current_dl_file_names[image_index] for image_index in image_indexes_np
        ]

        # Log the results images
        for i, (target_np, image_name) in enumerate(zip(targets_np, image_names)):
            # Do not save too many images
            if i == self.params.nb_tensorboard_images_max:
                break
            for channel in range(target_np.shape[0]):
                # ... log the ground truth image
                plt.imshow(target_np[channel], cmap="gray")
                self.writer.add_figure(
                    f"{name}/{image_name}/{channel}/fucci",
                    plt.gcf(),
                    current_batch,
                )
