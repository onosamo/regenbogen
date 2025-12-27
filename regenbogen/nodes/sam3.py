"""
Segmentation node using SAM3 (Segment Anything Model 3 with Concepts).

This node performs text-prompted or automatic instance segmentation using
the SAM3 model from Meta/Facebook with open-vocabulary concept understanding.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..core.node import Node
from ..interfaces import Frame, Masks

logger = logging.getLogger(__name__)


class SAM3Node(Node):
    """
    Node for text-prompted and automatic segmentation using SAM3.

    SAM3 (Segment Anything Model 3) can detect, segment, and track objects using
    text prompts or visual prompts (points, boxes, masks). It supports open-vocabulary
    segmentation with over 270K unique concepts.

    Key Features:
    - Text-prompted segmentation ("a player in white", "a red car", etc.)
    - Visual prompts (points, boxes) for interactive segmentation
    - Automatic mask generation when no prompt is provided
    - Open-vocabulary concept understanding
    - Supports batched inference for efficiency

    Note: Requires HuggingFace authentication and access to facebook/sam3 models.
    """

    def __init__(
        self,
        device: str | None = None,
        text_prompt: str | None = None,
        point_prompts: Optional[np.ndarray] = None,
        box_prompts: Optional[np.ndarray] = None,
        mask_threshold: float = 0.0,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initialize the SAM3 node.

        Args:
            device: Device to run the model on ("cpu", "cuda", or None for auto)
            text_prompt: Optional text prompt for concept-based segmentation
                        (e.g., "a person", "red car", "player in white")
            point_prompts: Optional point prompts as numpy array (N, 2) with [x, y] coordinates
            box_prompts: Optional box prompts as numpy array (N, 4) in [x1, y1, x2, y2] format
            mask_threshold: Threshold for binarizing output masks (default: 0.0)
            name: Optional name for the node
            **kwargs: Additional configuration parameters

        Note:
            - If no prompts are provided, the model will attempt automatic segmentation
            - Text prompts enable open-vocabulary segmentation
            - Visual prompts (points/boxes) enable interactive refinement
            - Multiple prompt types can be combined
        """
        super().__init__(name=name, **kwargs)
        self.device = device
        self.text_prompt = text_prompt
        self.point_prompts = point_prompts
        self.box_prompts = box_prompts
        self.mask_threshold = mask_threshold
        self.model = None
        self.processor = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SAM3 model from HuggingFace."""
        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        logger.info("Loading SAM3 model from facebook/sam3")
        logger.info(f"Device: {device}")

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            # Build SAM3 image model
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model, device=device)
            self.model = self.model.to(device)
            self.model.eval()

            logger.info("SAM3 model loaded successfully!")

        except ImportError as e:
            logger.error(
                f"Failed to import SAM3. Install with:\\n"
                f"  uv sync --group full\\n"
                f"Or manually:\\n"
                f"  pip install git+https://github.com/Guereak/sam3-apple-silicon.git\\n\\n"
                f"Also ensure HuggingFace authentication:\\n"
                f"  huggingface-cli login\\n"
                f"And request access at: https://huggingface.co/facebook/sam3\\n\\n"
                f"Error: {e}"
            )
            raise ImportError(
                f"SAM3 dependencies not installed. Missing: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise

    def process(
        self,
        input_data: Frame | tuple,
        text_prompt: str | None = None,
        point_prompts: Optional[np.ndarray] = None,
        box_prompts: Optional[np.ndarray] = None,
    ) -> Frame | tuple:
        """
        Process a frame to generate segmentation masks using SAM3.

        Args:
            input_data: Input frame with RGB image, or tuple (Frame, ...) from dataset loaders
            text_prompt: Optional text prompt override (uses init value if None)
            point_prompts: Optional point prompts override (N, 2) array
            box_prompts: Optional box prompts override (N, 4) array

        Returns:
            Masks containing segmentation masks, bounding boxes, and scores

        Examples:
            # Text-prompted segmentation
            masks = sam3_node.process(frame, text_prompt="a red car")

            # Interactive segmentation with points
            masks = sam3_node.process(frame, point_prompts=np.array([[100, 200]]))

            # Combined prompts
            masks = sam3_node.process(
                frame,
                text_prompt="a person",
                box_prompts=np.array([[50, 50, 200, 300]])
            )
        """
        # Handle tuple input from dataset loaders (Frame, ground_truth, obj_ids)
        if isinstance(input_data, tuple):
            frame = input_data[0]
        else:
            frame = input_data

        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        if frame.rgb is None:
            raise ValueError("Frame must contain an RGB image")

        # Use provided prompts or fall back to initialization values
        text = text_prompt if text_prompt is not None else self.text_prompt
        points = point_prompts if point_prompts is not None else self.point_prompts
        boxes = box_prompts if box_prompts is not None else self.box_prompts

        rgb_image = frame.rgb
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Set the image in the processor
        inference_state = self.processor.set_image(pil_image)

        # SAM3 primarily supports text-based prompting
        # If point or box prompts are provided, we use a generic text prompt
        if text is None:
            if boxes is not None:
                logger.warning(
                    "SAM3 doesn't support box prompts directly. Using generic text prompt."
                )
                text = "objects"
            elif points is not None:
                logger.warning(
                    "SAM3 doesn't support point prompts directly. Using generic text prompt."
                )
                text = "objects"
            else:
                # No prompts provided, use default
                text = "objects"

        # Process with text prompt
        logger.debug(f"Processing with text prompt: '{text}'")
        output = self.processor.set_text_prompt(state=inference_state, prompt=text)

        if output is None:
            logger.warning("No output from SAM3 model")
            # Return frame or tuple depending on input type
            if isinstance(input_data, tuple):
                output_data = (frame, *input_data[1:])
            else:
                output_data = frame

        # Extract masks, boxes, and scores from output
        pred_masks = output["masks"]  # Shape: (N, H, W)
        pred_boxes = output["boxes"]  # Shape: (N, 4) in [x1, y1, x2, y2]
        pred_scores = output["scores"]  # Shape: (N,)

        # Convert tensors to numpy arrays
        if isinstance(pred_masks, torch.Tensor):
            pred_masks = pred_masks.cpu().numpy()
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()

        # Convert masks to boolean
        if self.mask_threshold > 0:
            pred_masks = pred_masks > self.mask_threshold
        pred_masks = pred_masks.astype(bool)

        # Ensure masks are 3D (N, H, W) - squeeze out extra dimensions
        while pred_masks.ndim > 3:
            pred_masks = pred_masks.squeeze()
        
        # If still not 3D, ensure correct shape
        if pred_masks.ndim == 2:
            pred_masks = pred_masks[np.newaxis, ...]  # Add batch dimension
        elif pred_masks.ndim != 3:
            raise ValueError(f"Expected 3D masks (N, H, W), got shape {pred_masks.shape}")

        # Ensure boxes and scores are float32
        pred_boxes = pred_boxes.astype(np.float32)
        pred_scores = pred_scores.astype(np.float32)

        # Create class names based on prompts
        if text is not None:
            class_names = [text] * len(pred_masks)
        else:
            class_names = ["object"] * len(pred_masks)

        labels = np.array([1] * len(pred_masks), dtype=np.uint16)  # Dummy class IDs

        output_masks = Masks(
            masks=pred_masks,
            boxes=pred_boxes,
            scores=pred_scores,
            labels=labels,  # SAM3 doesn't provide class IDs, only concept-based
            class_names=class_names,
            metadata={
                "model": "sam3",
                "text_prompt": text,
                "has_point_prompts": points is not None,
                "has_box_prompts": boxes is not None,
                "num_masks": len(pred_masks),
                "mask_threshold": self.mask_threshold,
            },
        )

        # Attach masks to frame
        frame.masks = output_masks

        # Return frame or tuple depending on input type
        if isinstance(input_data, tuple):
            output_data = (frame, *input_data[1:])
        else:
            output_data = frame

        logger.info(
            f"Generated {output_masks.metadata['num_masks']} masks "
            f"(prompt: {text or 'visual/auto'})"
        )
        return output_data
