"""
Partial pointcloud extraction node.

This node extracts pointcloud regions corresponding to detected objects.
"""

from typing import List, Tuple

import numpy as np

from ..core.node import Node
from ..interfaces import BoundingBoxes, Frame


class PartialPointCloudExtractionNode(Node):
    """
    Node for extracting partial pointclouds from detected object regions.

    Takes bounding boxes and a frame with pointcloud, and extracts the pointcloud
    regions corresponding to the detected objects.
    """

    def __init__(self, expand_factor: float = 1.1, min_points: int = 100, **kwargs):
        """
        Initialize partial pointcloud extraction node.

        Args:
            expand_factor: Factor to expand bounding boxes (1.0 = no expansion)
            min_points: Minimum number of points required for valid extraction
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.expand_factor = expand_factor
        self.min_points = min_points

    def process(self, inputs: Tuple[Frame, BoundingBoxes]) -> List[np.ndarray]:
        """
        Extract partial pointclouds from detected object regions.

        Args:
            inputs: Tuple of (Frame with pointcloud, BoundingBoxes with detections)

        Returns:
            List of partial pointclouds (each as numpy array N x 3)
        """
        frame, bboxes = inputs

        if frame.pointcloud is None:
            raise ValueError("Input frame must contain pointcloud")

        if frame.depth is None or frame.intrinsics is None:
            raise ValueError(
                "Frame must contain depth and intrinsics for pixel-to-3D mapping"
            )

        # Extract pointclouds for each bounding box
        partial_pointclouds = []

        for i, bbox in enumerate(bboxes.boxes):
            partial_pc = self._extract_pointcloud_from_bbox(
                frame, bbox, bboxes.labels[i], bboxes.class_names
            )

            if partial_pc is not None and len(partial_pc) >= self.min_points:
                partial_pointclouds.append(partial_pc)

        return partial_pointclouds

    def _extract_pointcloud_from_bbox(
        self, frame: Frame, bbox: np.ndarray, label: int, class_names: List[str]
    ) -> np.ndarray:
        """
        Extract pointcloud from a single bounding box region.

        Args:
            frame: Input frame with pointcloud and depth
            bbox: Bounding box [x1, y1, x2, y2]
            label: Object class label
            class_names: List of class names

        Returns:
            Partial pointcloud as numpy array (N, 3) or None if extraction fails
        """
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame.depth.shape

        # Expand bounding box
        if self.expand_factor != 1.0:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            new_width = width * self.expand_factor
            new_height = height * self.expand_factor

            x1 = int(center_x - new_width / 2)
            y1 = int(center_y - new_height / 2)
            x2 = int(center_x + new_width / 2)
            y2 = int(center_y + new_height / 2)

        # Clamp to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract depth region
        depth_region = frame.depth[y1:y2, x1:x2]

        # Get camera intrinsics
        fx = frame.intrinsics[0, 0]
        fy = frame.intrinsics[1, 1]
        cx = frame.intrinsics[0, 2]
        cy = frame.intrinsics[1, 2]

        # Create coordinate grids for the region
        region_h, region_w = depth_region.shape
        u_offset = x1
        v_offset = y1

        u, v = np.meshgrid(
            np.arange(region_w) + u_offset, np.arange(region_h) + v_offset
        )

        # Flatten arrays
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_region.flatten()

        # Filter valid depths
        valid_mask = (depth_flat > 0.1) & (depth_flat < 10.0)  # reasonable depth range
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]

        if len(depth_valid) == 0:
            return None

        # Unproject to 3D coordinates
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid

        # Stack into pointcloud
        partial_pointcloud = np.stack([x, y, z], axis=1).astype(np.float32)

        return partial_pointcloud
