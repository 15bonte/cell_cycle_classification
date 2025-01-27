import albumentations as A
import numpy as np
import torch

from pythae.data.datasets import DatasetOutput as DatasetOutputVAE

from cnn_framework.utils.data_sets.dataset_output import DatasetOutput
from cnn_framework.utils.enum import ProjectMethods
from cnn_framework.utils.data_sets.abstract_data_set import AbstractDataSet
from cnn_framework.utils.readers.images_reader import ImagesReader
from cnn_framework.utils.readers.utils.projection import Projection
from cnn_framework.utils.augmentations.clip import Clip

from .nucleus_id_container import NucleusIdContainer


def has_one_on_edge(arr):
    """
    Checks if a 2D array contains a >0 on its edges (first/last row or column).
    """
    # Check first and last rows
    if np.any(arr[0, :] > 0) or np.any(arr[-1, :] > 0):
        return 1

    # Check first and last columns
    if np.any(arr[:, 0] > 0) or np.any(arr[:, -1] > 0):
        return 1

    return 0


class FucciVAEDataSet(AbstractDataSet):
    """Data set for self-supervised learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        self.input_data_source = ImagesReader(
            [self.data_manager.get_microscopy_image_path],
            [
                [
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.z_indexes,
                        axis=2,  # z-stack
                    ),
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.c_indexes + [1, 2, 3],  # FUCCI x2 + mask
                        axis=1,  # channel
                    ),
                ]
            ],
        )

    def set_transforms(self):
        shared_transforms = [
            A.Normalize(
                self.mean_std["mean"],
                std=self.mean_std["std"],
                max_pixel_value=1,
            ),
            A.PadIfNeeded(
                min_height=self.params.data_set_size,
                min_width=self.params.data_set_size,
                border_mode=1,
                value=0,
                p=1,
            ),
            A.CenterCrop(
                height=self.params.data_set_size,
                width=self.params.data_set_size,
                p=1,
            ),
            A.Resize(
                self.params.input_dimensions.width,
                self.params.input_dimensions.height,
                always_apply=True,
            ),
            Clip(min_value=0, max_value=1),  # Necessary since decoder ends with sigmoid
        ]
        if self.is_train:
            self.transforms = A.Compose(
                shared_transforms
                + [
                    A.Rotate(border_mode=1),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                ]
            )
        else:
            self.transforms = A.Compose(shared_transforms)

    def generate_images(self, filename: str) -> DatasetOutput:
        image = self.input_data_source.get_image(
            filename, h5_file=self.h5_file, names=self.h5_names
        )  # YXC

        nb_stacks = len(self.params.z_indexes)
        return DatasetOutput(
            input=image[..., :nb_stacks],  # DNA image
            target_image=image[..., nb_stacks : 3 * nb_stacks],  # FUCCI image
            additional=image[..., 3 * nb_stacks :],  # mask image
        )

    def get_image_sequence(
        self, filename: str, nb_adjacent=1
    ) -> tuple[DatasetOutput, np.ndarray, np.ndarray]:
        """Get adjacent DAPI images.

        NB: transformations are not the same for adjacent DAPI images.

        Parameters
        ----------
        filename : str
            Current filename.
        default_input : np.ndarray
            Default input to return if no adjacent DAPI found.
        nb_adjacent : int, optional
            Number of adjacent DAPI images to fetch before/after. Default is 1.

        Returns
        -------
        list[DatasetOutput]
            Adjacent DAPI images.
        np.ndarray
            Sequence mask
        np.ndarray
            Pixel mask
        """
        current_index = self.h5_names[
            filename
        ]  # designed to work even without h5 files
        track_id = NucleusIdContainer(filename).track_id

        # Initialize sequence and pixel masks
        height, width = self.params.input_dimensions.to_tuple(False)
        nb_channels = len(self.params.z_indexes) * len(self.params.c_indexes)
        sequence_mask = np.ones((2 * nb_adjacent + 1), dtype=bool)
        pixel_mask = np.ones(
            (2 * nb_adjacent + 1, nb_channels, height, width), dtype=bool
        )

        image_data = []

        # Create list of indexes to fetch
        adj_indexes = np.arange(
            current_index - nb_adjacent, current_index + nb_adjacent + 1
        )

        for i, adj_index in enumerate(adj_indexes):
            image_found = False
            if adj_index in self.h5_indexes:
                adj_filename = self.h5_indexes[adj_index]
                if NucleusIdContainer(adj_filename).track_id == track_id:
                    adj_data_set_output = self.generate_images(adj_filename)
                    self.apply_transforms(adj_data_set_output)
                    image_data.append(adj_data_set_output)
                    image_found = True
            if not image_found:
                image_data.append(None)
                sequence_mask[i] = False

        # Replace None with default input
        central_data = image_data[nb_adjacent]
        assert central_data is not None
        image_data = [data if data is not None else central_data for data in image_data]

        return image_data, sequence_mask, pixel_mask

    def __getitem__(self, idx: int) -> DatasetOutputVAE:
        """Read file and generate images"""
        filename = self.names[idx]

        # Adjacent DAPI
        nb_adjacent = 1 if self.params.gamma > 0 else 0
        merged, _, _ = self.get_image_sequence(filename, nb_adjacent=nb_adjacent)
        mask = merged[nb_adjacent].additional  # CYX

        if self.params.delta > 0:
            nb_stacks = len(self.params.z_indexes)

            fucci_red_avg = np.mean(merged[nb_adjacent].target[:nb_stacks][mask > 0])
            fucci_green_avg = np.mean(
                merged[nb_adjacent].target[nb_stacks : 2 * nb_stacks][mask > 0]
            )
        else:
            fucci_red_avg = 0
            fucci_green_avg = 0

        height, width = self.params.input_dimensions.to_tuple(False)
        return DatasetOutputVAE(
            data=merged[nb_adjacent].input,
            target=(
                merged[nb_adjacent].target
                if merged[nb_adjacent].target is not None
                else torch.zeros(merged[nb_adjacent].input.shape)
            ),
            id=idx,
            adjacent_dapi=(
                np.concatenate(
                    [adj.input for idx, adj in enumerate(merged) if idx != nb_adjacent],
                    0,
                )
                if nb_adjacent > 0
                else torch.zeros(merged[nb_adjacent].input.shape)
            ),
            fucci=np.array([fucci_red_avg, fucci_green_avg]),
            track_id=NucleusIdContainer(filename).get_video_track_id(),
            phase=NucleusIdContainer(filename).phase,
            area=np.sum(mask[0] > 0) / (height * width),
            edge=has_one_on_edge(mask[0]),
            dapi=np.mean(merged[nb_adjacent].input[mask > 0]),
            fucci_red=fucci_red_avg,
            fucci_green=fucci_green_avg,
        )


class FucciClassificationDataSet(FucciVAEDataSet):
    """
    Same pre-processing as FucciVAEDataSet.
    Different output.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        self.input_data_source = ImagesReader(
            [self.data_manager.get_microscopy_image_path],
            [
                [
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.z_indexes,
                        axis=2,
                    ),
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.c_indexes,
                        axis=1,
                    ),
                ]
            ],
        )

    def generate_images(self, filename):
        # Output
        probabilities = self.read_output(filename)

        return DatasetOutput(
            input=self.input_data_source.get_image(
                filename, h5_file=self.h5_file, names=self.h5_names
            ),
            target_array=probabilities,
        )

    def __getitem__(self, idx):
        # Read file and generate images
        data_set_output = self.generate_images(self.names[idx])

        # Apply transforms
        self.apply_transforms(data_set_output)
        data_set_output.index = idx

        # Set to torch tensors
        data_set_output.to_torch()

        return data_set_output
