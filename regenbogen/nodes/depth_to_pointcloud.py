"""
Depth to pointcloud conversion node.

This node converts depth images and camera intrinsics to 3D pointclouds.
"""

import numpy as np

from ..core.node import Node
from ..interfaces import Frame


class DepthToPointCloudNode(Node):
    """
    Node for converting depth images to 3D pointclouds.

    Uses camera intrinsics to unproject depth pixels to 3D points.
    """

    def __init__(self, max_depth: float = 10.0, min_depth: float = 0.1, **kwargs):
        """
        Initialize depth to pointcloud node.

        Args:
            max_depth: Maximum depth value to consider (meters)
            min_depth: Minimum depth value to consider (meters)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.min_depth = min_depth

    def process(self, frame: Frame) -> Frame:
        """
        Convert depth image to pointcloud.

        Args:
            frame: Input frame with depth image and intrinsics

        Returns:
            Frame with pointcloud added
        """
        if frame.depth is None:
            raise ValueError("Input frame must contain depth image")

        if frame.intrinsics is None:
            # Use default intrinsics if not provided
            h, w = frame.depth.shape
            fx = fy = min(w, h)  # Rough estimate
            cx, cy = w / 2, h / 2
            intrinsics = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
            )
        else:
            intrinsics = frame.intrinsics

        # Generate pointcloud
        pointcloud = self._depth_to_pointcloud(frame.depth, intrinsics)

        # Create output frame
        output_frame = Frame(
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsics=intrinsics,
            extrinsics=frame.extrinsics,
            pointcloud=pointcloud,
            metadata=frame.metadata.copy() if frame.metadata else {},
        )

        output_frame.metadata["pointcloud_method"] = "depth_unprojection"
        output_frame.metadata["pointcloud_size"] = (
            len(pointcloud) if pointcloud is not None else 0
        )

        return output_frame

    def _depth_to_pointcloud(
        self, depth: np.ndarray, intrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Convert depth image to 3D pointcloud using camera intrinsics.

        Args:
            depth: Depth image (H, W)
            intrinsics: Camera intrinsics matrix (3, 3)

        Returns:
            Pointcloud array (N, 3) in camera coordinates
        """
        h, w = depth.shape

        # Extract intrinsic parameters
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Flatten arrays
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth.flatten()

        # Filter valid depths
        valid_mask = (depth_flat > self.min_depth) & (depth_flat < self.max_depth)
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]

        if len(depth_valid) == 0:
            return np.array([]).reshape(0, 3)

        # Unproject to 3D coordinates
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid

        # Stack into pointcloud
        pointcloud = np.stack([x, y, z], axis=1).astype(np.float32)

        return pointcloud
