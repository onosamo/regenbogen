"""
Depth estimation node using Depth Anything model.

This node converts RGB images to monocular depth estimates using
the Depth Anything V2 model from Hugging Face.
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from ..core.node import Node
from ..interfaces import Frame


class DepthAnythingNode(Node):
    """
    Node for monocular depth estimation using Depth Anything V2.

    This node uses the Depth Anything V2 model from Hugging Face transformers
    to estimate depth from RGB images.

    The model supports various sizes:
    - depth-anything-v2-small
    - depth-anything-v2-base
    - depth-anything-v2-large

    Larger models provide better accuracy but are slower.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the Depth Anything node.

        Args:
            model_size: Size of the model ("small", "base", or "large")
            device: Device to run the model on ("cpu", "cuda", or None for auto)
            name: Optional name for the node
            **kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)
        self.model_size = model_size
        self.device = device
        self.pipe = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Depth Anything V2 model."""
        # Map model size to actual model name
        model_name_map = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }

        if self.model_size not in model_name_map:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. "
                f"Must be one of {list(model_name_map.keys())}"
            )

        model_name = model_name_map[self.model_size]

        # Determine device
        if self.device is None:
            device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        elif self.device == "cuda":
            device = 0
        else:
            device = -1  # CPU

        print(f"Loading Depth Anything V2 model: {model_name}")
        print(f"Device: {'GPU' if device == 0 else 'CPU'}")

        # Initialize the depth estimation pipeline
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=device)

        print("Depth Anything V2 model loaded successfully!")

    def process(self, frame: Frame) -> Frame:
        """
        Process a frame to estimate depth.

        Args:
            frame: Input frame with RGB image

        Returns:
            Frame with depth data added
        """
        if self.pipe is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        if frame.rgb is None:
            raise ValueError("Frame must contain an RGB image")

        # Convert numpy array to PIL Image
        rgb_image = frame.rgb
        if rgb_image.dtype != np.uint8:
            # Normalize to 0-255 if needed
            rgb_image = (rgb_image * 255).astype(np.uint8)

        pil_image = Image.fromarray(rgb_image)

        # Run inference
        result = self.pipe(pil_image)

        # Extract depth map
        depth_pil = result["depth"]

        fx = frame.intrinsics[0, 0] if frame.intrinsics is not None else 1000.0

        # Returned results are actually disparity maps; convert to depth
        depth_array = np.array(depth_pil, dtype=np.float32)
        depth_array = fx / (depth_array + 1e-6)  # Invert disparity to get depth
        depth_array = np.clip(depth_array, 0.1, 100.0)  # Clip to a reasonable range

        # Create output frame
        output_frame = Frame(
            rgb=frame.rgb,
            depth=depth_array,
            intrinsics=frame.intrinsics,
            extrinsics=frame.extrinsics,
            pointcloud=frame.pointcloud,
            metadata=frame.metadata,
        )

        return output_frame
