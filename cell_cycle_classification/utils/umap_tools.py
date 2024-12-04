"""UMAP tools"""

import os
import numpy as np
import umap
import umap.plot
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import DataLoader

from cnn_framework.utils.enum import PredictMode
from cnn_framework.utils.model_managers.model_manager import ModelManager


def get_predictions_names(
    manager: ModelManager, data_loader, post_processing=None, compute_own_mean_std=False
):
    """Get predictions and names from data loader."""

    # Return empty results if data set is empty
    if len(data_loader.dataset.names) == 0:
        return None, [], [], {"areas": [], "edges": []}

    # Get predictions
    all_predictions = np.array(
        manager.predict(
            data_loader,
            predict_mode=PredictMode.GetEmbedding,
            nb_images_to_save=0,
            post_processing=post_processing,
            compute_own_mean_std=compute_own_mean_std,
        )
    )

    # Iterate over data loader to get classes, names, probabilities
    predictions, classes, names, areas, edges = [], [], [], [], []
    for dl_element in data_loader:
        # Get indexes depending on model
        try:
            indexes = dl_element["id"]  # VAE
        except TypeError:
            indexes = dl_element.index  # Pix2Pix
        indexes = indexes.detach().numpy()

        local_areas = dl_element["area"].detach().numpy()
        local_edges = dl_element["edge"].detach().numpy()

        for idx, area, edge in zip(indexes, local_areas, local_edges):
            filename = data_loader.dataset.names[idx]

            # Read probabilities and class from filename
            local_probabilities = data_loader.dataset.read_output(
                filename, one_hot=False
            )
            one_hot_probabilities = data_loader.dataset.read_output(
                filename, one_hot=True
            )
            if np.max(one_hot_probabilities) == 0:
                # This point is not labeled - skip it
                all_predictions = all_predictions[1:]
                continue

            classes.append(np.argmax(one_hot_probabilities))
            names.append(f"{filename}\n{local_probabilities}")

            predictions.append(all_predictions[0])
            all_predictions = all_predictions[1:]

            areas.append(area)
            edges.append(edge)

    assert all_predictions.shape[0] == 0  # All predictions should have been used
    assert len(predictions) == len(classes) == len(names) == len(areas)

    return predictions, classes, names, {"areas": areas, "edges": edges}


def save_htlm(data: dict, path: str) -> None:
    """Save UMAP plot as HTML."""
    fig = go.Figure(
        data=go.Scatter(
            x=data["predictions"][:, 0],
            y=data["predictions"][:, 1],
            mode="markers",
            marker=dict(color=[["green", "red", "yellow"][c] for c in data["classes"]]),
            hovertext=data["names"],
        )
    )
    fig.write_html(path)


def save_built_in(data: dict, umap_model, path: str) -> None:
    """Save UMAP plot using UMAP built-in function."""
    # Update UMAP model to use predictions
    train_embedding = umap_model.embedding_
    umap_model.embedding_ = data["predictions"]
    umap.plot.points(
        umap_model,
        labels=np.array([["G1", "S", "G2"][c] for c in data["classes"]]),
        color_key={"G1": "green", "S": "red", "G2": "yellow"},
    )
    plt.savefig(path, bbox_inches="tight", dpi=300)
    # Reset train predictions
    umap_model.embedding_ = train_embedding


def run_predictions(data_loader: DataLoader, manager, post_processing=None) -> dict:
    """Perform encoder predictions given data loader."""

    print("Run predictions")
    (predictions, classes, names, _) = get_predictions_names(
        manager, data_loader, post_processing=post_processing
    )
    predictions = np.array(predictions)

    # # Get CNN predictions directly -- deprecated
    # try:
    #     cnn_predicted_classes_test = np.array(
    #         manager.predict(test_dl, predict_mode=PredictMode.GetPrediction)
    #     )
    #     test_cnn_score = balanced_accuracy_score(
    #         cnn_predicted_classes_test, classes_test
    #     )
    #     print(f"\nCNN classifier on test (balanced): {test_cnn_score}")
    # except AttributeError:
    #     print("\nNo CNN classification output to evaluate")
    # except NotImplementedError:
    #     print("\nPrediction mode is not implemented yet (most likely for Pix2pix)")

    # Get DAPI values -- deprecated
    # dapi_train = get_dapi_values(names_train, params)

    return {
        "predictions": predictions,
        "classes": classes,
        "names": names,
    }


def fit_umap_on_train(train_dl, manager, params, save: str):
    """Fit UMAP on train data loader."""
    results_train = run_predictions(train_dl, manager)

    # Check if there is a class -1
    assert (
        min(
            int(c)
            for classes in [
                results_train["classes"],
            ]
            for c in classes
        )
        > -1
    )

    # Fit UMAP on train
    umap_model = umap.UMAP(random_state=42)
    results_train["predictions"] = umap_model.fit_transform(
        results_train["predictions"]
    )

    if save == "html":
        save_htlm(results_train, os.path.join(params.output_dir, "umap_train.html"))
    elif save == "svg":
        save_built_in(
            results_train,
            umap_model,
            os.path.join(params.output_dir, "umap_train.svg"),
        )

    # Plot UMAP on train with DAPI
    if "dapi" in results_train and len(results_train["dapi"]) > 0:
        plt.subplot(1, 2, 1)
        plt.scatter(
            results_train["dapi"],
            results_train[:, 0],
            s=10,
            c=results_train["colors"],
        )
        plt.subplot(1, 2, 2)
        plt.scatter(
            results_train["dapi"],
            results_train[:, 1],
            s=10,
            c=results_train["colors"],
        )
        plt.show()

        # Plot UMAP histograms
        _, axes = plt.subplots(2, 1)
        for i, ax in enumerate(axes):
            data = results_train[:, i]
            bin_size = 0.1
            for color_axis, category_axis in zip(
                ["green", "red", "yellow"], ["G1", "S", "G2"]
            ):
                data_color = data[np.array(results_train["colors"]) == color_axis]
                bins = int((max(data_color) - min(data_color)) / bin_size)
                if bins > 0:
                    ax.hist(
                        data_color,
                        bins=bins,
                        color=color_axis,
                        alpha=0.5,
                        label=category_axis,
                    )
            ax.set_title(f"UMAP axis {i+1}")
            ax.legend()
        plt.savefig(os.path.join(params.output_dir, "umap_histograms.png"))

    return umap_model


def predict_umap(dl, umap_model, params, manager, name, save):
    """Predict UMAP on data loader."""
    results_val = run_predictions(dl, manager, post_processing=umap_model.transform)

    file_name = f"umap_{name}.html" if save == "html" else f"umap_{name}.svg"

    if save == "html":
        save_htlm(results_val, os.path.join(params.output_dir, file_name))
    elif save == "svg":
        save_built_in(
            results_val,
            umap_model,
            os.path.join(params.output_dir, file_name),
        )
