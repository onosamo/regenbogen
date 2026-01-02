"""
Depth to pointcloud conversion node.

This node converts depth images and camera intrinsics to 3D pointclouds.
"""

import logging
from collections.abc import Generator

import cv2
import numpy as np

from ..core.node import Node
from ..interfaces import Frame, PointCloud

logger = logging.getLogger(__name__)


class DepthToPointCloudNode(Node):
    """
    Node for converting depth images to 3D pointclouds.

    Uses camera intrinsics to unproject depth pixels to 3D points.
    """

    def __init__(self, max_depth: float = 10.0, min_depth: float = 0.1,
                 exclude_classes: list[str] = [],
                 include_classes: list[str] | None = None,
                 **kwargs):
        """
        Initialize depth to pointcloud node.

        Args:
            max_depth: Maximum depth value to consider (meters)
            min_depth: Minimum depth value to consider (meters)
            exclude_classes: List of class names to exclude from pointcloud generation (if masks are available)
            include_classes: If provided, only include points from masks with these class names (whitelist)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.exclude_classes = exclude_classes
        self.include_classes = include_classes

    def process(
        self, frame: Frame | list[Frame] | Generator[Frame, None, None]
    ) -> Frame | list[Frame] | Generator[Frame, None, None]:
        """
        Convert depth image(s) to pointcloud(s).

        Args:
            frame: Input frame, list of frames, or generator of frames with depth image and intrinsics

        Returns:
            Frame, list of frames, or generator of frames with pointcloud added and colors stored in metadata
        """
        # Handle generator
        if hasattr(frame, "__iter__") and hasattr(frame, "__next__"):
            return (self._process_single_frame(f) for f in frame)

        # Handle list of frames
        if isinstance(frame, list):
            return [self._process_single_frame(f) for f in frame]

        # Handle single frame
        return self._process_single_frame(frame)

    def _process_single_frame(self, frame: Frame) -> Frame:
        """
        Process a single frame to convert depth to pointcloud.

        Args:
            frame: Input frame with depth image and intrinsics

        Returns:
            Frame with pointcloud added and colors stored in metadata
        """
        if frame.depth is None:
            raise ValueError("Input frame must contain depth image")

        if (self.exclude_classes or self.include_classes) and frame.masks is None:
            raise ValueError("To filter by classes, the input frame should contain masks")

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

        # If masks are available and filtering is enabled, create a mask to filter points
        if (self.exclude_classes or self.include_classes) and frame.masks is not None:
            # Start with no points included if using whitelist, all points if using blacklist
            if self.include_classes is not None:
                combined_mask = np.zeros(frame.depth.shape, dtype=bool)
            else:
                combined_mask = np.ones(frame.depth.shape, dtype=bool)

            # Reshape masks if needed
            if frame.masks.masks[0].shape != frame.depth.shape:
                resized_masks = []
                for mask in frame.masks.masks:
                    resized_mask = cv2.resize(mask.astype(np.uint8), (frame.depth.shape[1], frame.depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                    resized_masks.append(resized_mask.astype(bool))
                frame.masks.masks = np.array(resized_masks)

            if frame.masks.class_names is None:
                raise ValueError("Masks must have class_names to filter by classes")

            # Apply whitelist (include only specified classes)
            if self.include_classes is not None:
                for mask, class_name in zip(frame.masks.masks, frame.masks.class_names):
                    if class_name in self.include_classes:
                        combined_mask |= mask.astype(bool)
                logger.info(
                    f"Including only classes {self.include_classes} in pointcloud, total included points: {np.sum(combined_mask)}"
                )

            # Apply blacklist (exclude specified classes)
            if self.exclude_classes:
                for mask, class_name in zip(frame.masks.masks, frame.masks.class_names):
                    if class_name in self.exclude_classes:
                        combined_mask &= ~mask.astype(bool)
                logger.info(
                    f"Excluding classes {self.exclude_classes} from pointcloud, total excluded points: {np.sum(~combined_mask)}"
                )

            # Apply combined mask to depth
            depth_masked = np.where(combined_mask, frame.depth, 0)
        else:
            depth_masked = frame.depth

        # Generate pointcloud with colors
        pointcloud, colors = self._depth_to_pointcloud_with_colors(
            depth_masked, intrinsics, frame.rgb
        )

        logger.info(
            f"Generated pointcloud with {len(pointcloud)} points"
        )

        frame.pointcloud = PointCloud(points=pointcloud, colors=colors)

        frame.metadata["pointcloud_method"] = "depth_unprojection"
        frame.metadata["pointcloud_size"] = (
            len(pointcloud) if pointcloud is not None else 0
        )

        return frame

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

        # Unproject to 3D coordinates using standard pinhole camera model
        # This formula works for both interpretations:
        # - If depth = z-coordinate: directly use it as Z, scale x/y accordingly
        # - If depth = radial distance: scale ray (x_norm, y_norm, 1) by depth
        # Both interpretations yield the same result: (x_norm*d, y_norm*d, d)

        # Compute normalized ray directions
        x_normalized = (u_valid - cx) / fx
        y_normalized = (v_valid - cy) / fy

        # Apply depth (mathematically equivalent for both interpretations)
        z = depth_valid
        x = x_normalized * z
        y = y_normalized * z

        # Stack into pointcloud
        pointcloud = np.stack([x, y, z], axis=1).astype(np.float32)

        return pointcloud

    def _depth_to_pointcloud_with_colors(
        self, depth: np.ndarray, intrinsics: np.ndarray, rgb: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Convert depth image to 3D pointcloud with colors from RGB image.

        Args:
            depth: Depth image (H, W)
            intrinsics: Camera intrinsics matrix (3, 3)
            rgb: RGB image (H', W', 3) - may have different size than depth

        Returns:
            Tuple of (pointcloud array (N, 3), colors array (N, 3) or None)
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

        # Handle RGB colors if provided
        colors = None
        if rgb is not None:
            rgb_h, rgb_w = rgb.shape[:2]

            # Resize RGB to match depth if needed
            if (rgb_h, rgb_w) != (h, w):
                rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_resized = rgb

            # Flatten RGB
            rgb_flat = rgb_resized.reshape(-1, 3)

        # Filter valid depths
        valid_mask = (depth_flat > self.min_depth) & (depth_flat < self.max_depth)
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]

        if len(depth_valid) == 0:
            return np.array([]).reshape(0, 3), None

        # Unproject to 3D coordinates using standard pinhole camera model
        # This formula works for both interpretations:
        # - If depth = z-coordinate: directly use it as Z, scale x/y accordingly
        # - If depth = radial distance: scale ray (x_norm, y_norm, 1) by depth
        # Both interpretations yield the same result: (x_norm*d, y_norm*d, d)

        # Compute normalized ray directions
        x_normalized = (u_valid - cx) / fx
        y_normalized = (v_valid - cy) / fy

        # Apply depth (mathematically equivalent for both interpretations)
        z = depth_valid
        x = x_normalized * z
        y = y_normalized * z

        # Stack into pointcloud
        pointcloud = np.stack([x, y, z], axis=1).astype(np.float32)

        # Stack into pointcloud
        pointcloud = np.stack([x, y, z], axis=1).astype(np.float32)

        # Extract colors for valid points
        if rgb is not None:
            colors = rgb_flat[valid_mask]

        return pointcloud, colors
