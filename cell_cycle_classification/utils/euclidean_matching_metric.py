"""Metric to evaluate contrastive learning performance.
Adapted from https://github.com/sthalles/SimCLR """

import torch

from cnn_framework.utils.metrics.positive_pair_matching_metric import (
    PositivePairMatchingMetric,
)


class EuclideanMatchingMetric(PositivePairMatchingMetric):
    """Adapted from SimCLR pytorch implementation"""

    def update(self, predictions, targets, adds=None, mean_std=None):

        features = torch.cat([predictions, targets], dim=0)

        # Define labels
        labels = torch.cat(
            [torch.arange(len(targets)) for _ in range(2)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # # Cosine similarity - original code distance
        # import torch.nn.functional as F
        # features = F.normalize(features, dim=1)
        # similarity_matrix = torch.matmul(features, features.T)

        # Euclidean distance
        diffs = features[:, None, :] - features[None, :, :]  # Shape: (N, N, M)
        similarity_matrix = -torch.norm(
            diffs, dim=2
        )  # Shape: (N, N), Euclidean distances between each pair

        # Discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # Select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # Select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        super().update(logits, labels, adds, mean_std)
