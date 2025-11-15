"""
RMSE evaluation node for comparing pointclouds.
"""

from typing import Union

import numpy as np

from ..core.node import Node
from ..interfaces import ErrorMetrics, Frame, PointCloud


class RMSENode(Node):
    """
    Node for computing RMSE (Root Mean Square Error) between two pointclouds.

    Compares a predicted pointcloud against a ground truth pointcloud.
    """

    def __init__(self, name: str = None, **kwargs):
        """
        Initialize RMSE node.

        Args:
            name: Node name
        """
        super().__init__(name=name or "RMSE", **kwargs)

    def process(
        self,
        predicted: Union[Frame, PointCloud],
        ground_truth: Union[Frame, PointCloud],
    ) -> ErrorMetrics:
        """
        Compute RMSE between predicted and ground truth pointclouds.

        Args:
            predicted: Predicted data as Frame or PointCloud
            ground_truth: Ground truth data as Frame or PointCloud

        Returns:
            ErrorMetrics containing RMSE and other metrics
        """
        if isinstance(predicted, Frame):
            pred_points = predicted.pointcloud.points if predicted.pointcloud else None
        else:
            pred_points = predicted.points

        if isinstance(ground_truth, Frame):
            gt_points = (
                ground_truth.pointcloud.points if ground_truth.pointcloud else None
            )
        else:
            gt_points = ground_truth.points

        if pred_points is None or gt_points is None:
            raise ValueError("Pointcloud data is None")

        if pred_points.shape[1] != 3 or gt_points.shape[1] != 3:
            raise ValueError("Pointclouds must have shape (N, 3)")

        min_size = min(len(pred_points), len(gt_points))

        pred_sample = pred_points[:min_size]
        gt_sample = gt_points[:min_size]

        squared_diffs = np.sum((pred_sample - gt_sample) ** 2, axis=1)
        rmse = np.sqrt(np.mean(squared_diffs))

        mean_error = np.mean(np.sqrt(squared_diffs))
        max_error = np.max(np.sqrt(squared_diffs))

        metrics = ErrorMetrics(
            add=float(rmse),
            metadata={
                "rmse": float(rmse),
                "mean_error": float(mean_error),
                "max_error": float(max_error),
                "num_points_predicted": len(pred_points),
                "num_points_ground_truth": len(gt_points),
                "num_points_compared": min_size,
            },
        )

        return metrics
