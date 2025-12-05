"""
Depth and pose estimation node using Depth Anything V3.

This node uses the Depth Anything V3 model from Hugging Face to estimate
metric depth and camera poses from RGB images using a sliding window approach.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.node import Node
from ..interfaces import Frame
from ..utils.device import get_torch_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def predicted_extrinsics_to_cw(predicted_extrinsics: np.ndarray) -> np.ndarray:
    ext_3x4 = predicted_extrinsics
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :] = ext_3x4

    # Invert to get camera-to-world (what we need for visualization and alignment)
    T_cw = np.linalg.inv(T_wc)
    return T_cw


class DepthAnything3Node(Node):
    """
    Node for depth and pose estimation using Depth Anything V3.

    Uses a sliding window buffer to process frames with multi-view consistency.
    Estimates camera poses and metric depth, outputting frames with extrinsics,
    intrinsics, and depth information.

    The model supports various sizes:
    - da3nested-giant-large (recommended, 1.4B params, metric depth + poses)
    - da3-giant (1.15B params, poses)
    - da3-large (0.35B params)
    - da3-base (0.12B params)
    - da3-small (0.08B params)

    Larger models provide better accuracy but are slower.
    """

    def __init__(
        self,
        model_name: str = "da3nested-giant-large",
        device: str | None = None,
        buffer_size: int = 3,
        buffer_step: int = 1,
        estimate_poses: bool = True,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        name: str = None,
        **kwargs,
    ):
        """
        Initialize Depth Anything V3 node.

        Args:
            model_name: DA3 model variant ("da3nested-giant-large", "da3-large", etc.)
            device: Device to run on ("cuda", "cpu", or None for auto)
            buffer_size: Number of frames in sliding window (2-5 recommended)
            buffer_step: Number of frames to shift buffer each step (1 for sliding window).
                        If negative -N, uses buffer_size - N (e.g., -1 means shift by buffer_size-1)
            estimate_poses: Estimate camera poses from images
            process_res: Processing resolution
            process_res_method: Resize method ("upper_bound_resize", "lower_bound_resize")
            name: Node name
            **kwargs: Additional configuration
        """
        super().__init__(name=name or "DepthAnything3Node", **kwargs)

        self.model_name = model_name
        self.buffer_size = buffer_size

        # Handle negative buffer_step
        if buffer_step < 0:
            self.buffer_step = buffer_size + buffer_step
        else:
            self.buffer_step = buffer_step

        if self.buffer_step <= 0 or self.buffer_step > buffer_size:
            raise ValueError(
                f"buffer_step must be between 1 and buffer_size ({buffer_size}), got {buffer_step}"
            )

        self.estimate_poses = estimate_poses
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.device = device

        self.frame_buffer: list[Frame] = []

        self.previous_poses: list[np.ndarray] | None = None

        self.model = None

    def _initialize_model(self):
        """Initialize DA3 model from HuggingFace."""
        if self.model is not None:
            return

        try:
            try:
                from depth_anything_3.api import DepthAnything3
            except ImportError as e:
                raise ImportError(
                    "Depth Anything V3 is not installed. This is an optional dependency.\n\n"
                    "To use DepthAnything3Node, install the DA3 package:\n"
                    "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git\n\n"
                    "Or with uv:\n"
                    "  uv pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git\n\n"
                    "Note: DA3 requires xformers, einops, torch>=2.0.0 (already in regenbogen dependencies)."
                ) from e

            device = self.device or get_torch_device()

            logger.info(f"Loading Depth Anything V3: {self.model_name}")
            logger.info("This may take a while for first-time model download...")

            # Create model from pretrained
            hf_model_name = f"depth-anything/{self.model_name.upper()}"
            self.model = DepthAnything3.from_pretrained(hf_model_name)
            self.model = self.model.to(device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"Failed to load Depth Anything V3 model: {e}")
            raise

    def process(self, frame: Frame) -> list[Frame] | None:
        """
        Process a single frame with sliding window.

        Args:
            frame: Input frame with RGB image

        Returns:
            Frame with depth, extrinsics, intrinsics, and metadata
        """
        if self.model is None:
            self._initialize_model()

        if frame.rgb is None:
            raise ValueError("Frame must contain an RGB image")

        self.frame_buffer.append(frame)

        if len(self.frame_buffer) >= self.buffer_size:
            output_frames = self._process_buffer()

            # Slide window: remove buffer_step frames from the front
            self.frame_buffer = self.frame_buffer[self.buffer_step :]

            return output_frames
        else:
            # Skip frames until buffer is ready (return None to signal skip)
            logger.debug(
                f"Buffer not ready ({len(self.frame_buffer)}/{self.buffer_size}), "
                "skipping frame"
            )
            return None

    def _process_buffer(self) -> list[Frame]:
        """Process the frame buffer with DA3."""
        import torch

        # Convert frames to image list
        images = [f.rgb for f in self.frame_buffer]

        # Run inference
        with torch.no_grad():
            prediction = self.model.inference(
                image=images,
                extrinsics=None,
                intrinsics=None,
                align_to_input_ext_scale=True,
                process_res=self.process_res,
                process_res_method=self.process_res_method,
            )

        # Update buffer frames with predicted extrinsics and intrinsics
        if self.estimate_poses and hasattr(prediction, "extrinsics"):
            # Convert predicted poses to 4x4 matrices
            # DA3 returns world-to-camera transforms, we need camera-to-world for alignment
            predicted_poses_cw = []  # Camera-to-world (inverted from DA3 output)

            for buffer_idx in range(len(self.frame_buffer)):
                T_cw = predicted_extrinsics_to_cw(prediction.extrinsics[buffer_idx])
                predicted_poses_cw.append(T_cw)

            # Align poses with global coordinate frame if we have previous poses
            if self.previous_poses is not None and len(self.previous_poses) > 0:

                overlap_idx_new = 0  # First frame in new prediction
                overlap_idx_prev = (
                    self.buffer_step  # Frame at buffer_step in previous poses
                )

                if overlap_idx_prev < len(self.previous_poses):
                    T_prev = self.previous_poses[overlap_idx_prev]  # Previous estimate
                    T_new = predicted_poses_cw[
                        overlap_idx_new
                    ]  # New estimate of same frame

                    # Compute transform: T_prev = T_align @ T_new
                    # So: T_align = T_prev @ T_new^{-1}
                    T_align = T_prev @ np.linalg.inv(T_new)

                    # Apply alignment transform to all new poses
                    aligned_poses = [T_align @ pose for pose in predicted_poses_cw]
                    logger.debug(
                        f"Aligned poses with global frame (buffer_step={self.buffer_step}, "
                        f"translation: {np.linalg.norm(T_align[:3, 3]):.4f}m)"
                    )
                else:
                    # No overlap (buffer_step >= buffer_size), treat as new sequence
                    aligned_poses = predicted_poses_cw
                    logger.debug(
                        f"No overlap (buffer_step={self.buffer_step} >= buffer_size), "
                        "starting new coordinate frame"
                    )
            else:
                # First window - no alignment needed
                aligned_poses = predicted_poses_cw
                logger.debug("First window - initializing global coordinate frame")

            for buffer_idx, frame in enumerate(self.frame_buffer):
                frame.extrinsics = aligned_poses[buffer_idx]
                frame.intrinsics = prediction.intrinsics[buffer_idx].astype(
                    np.float64
                )
                frame.depth = prediction.depth[buffer_idx]
                frame.metadata = {
                    **frame.metadata,
                    "da3_model": self.model_name,
                    "buffer_size": self.buffer_size,
                    "pose_estimated": self.estimate_poses and frame.extrinsics is not None,
                    "is_metric_depth": "nested" in self.model_name.lower(),
                }

            self.previous_poses = aligned_poses.copy()

        return self.frame_buffer

    def reset_buffer(self):
        """Clear frame buffer."""
        self.frame_buffer = []
        logger.debug("Frame buffer cleared")
