"""
Segmentation node using SAM3 (Segment Anything Model 3 with Concepts).

This node performs text-prompted or automatic instance segmentation using
the SAM3 model from Meta/Facebook with open-vocabulary concept understanding.
"""

from __future__ import annotations

import logging

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
    - Multi-prompt processing (comma-separated: "player, ball, net")
    - Visual prompts (points, boxes) for interactive segmentation
    - Automatic mask generation when no prompt is provided
    - Open-vocabulary concept understanding
    - Batched inference for efficiency

    Note: Requires HuggingFace authentication and access to facebook/sam3 models.
    Multi-prompt processing is done sequentially while reusing image features for efficiency.
    """

    def __init__(
        self,
        device: str | None = None,
        text_prompt: str | None = None,
        point_prompts: np.ndarray | None = None,
        box_prompts: np.ndarray | None = None,
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
                        Multiple prompts can be specified as comma-separated values
                        (e.g., "player, ball, net") for parallel processing
            point_prompts: Optional point prompts as numpy array (N, 2) with [x, y] coordinates
            box_prompts: Optional box prompts as numpy array (N, 4) in [x1, y1, x2, y2] format
            mask_threshold: Threshold for binarizing output masks (default: 0.0)
            name: Optional name for the node
            **kwargs: Additional configuration parameters

        Note:
            - If no prompts are provided, the model will attempt automatic segmentation
            - Text prompts enable open-vocabulary segmentation
            - Multiple comma-separated prompts are processed in parallel for efficiency
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
            error_msg = str(e)
            # Check for HuggingFace authentication errors (common in CI)
            if "401 Client Error" in error_msg or "unauthorized" in error_msg.lower():
                logger.error(
                    f"HuggingFace authentication failed. This is expected in CI environments.\\n"
                    f"To use SAM3 locally, ensure you:\\n"
                    f"  1. Login: huggingface-cli login\\n"
                    f"  2. Request access: https://huggingface.co/facebook/sam3\\n\\n"
                    f"Error: {error_msg}"
                )
                raise PermissionError(
                    f"SAM3 requires HuggingFace authentication. Error: {error_msg}"
                )
            logger.error(f"Failed to load SAM3 model: {e}")
            raise

    def process(
        self,
        input_data: Frame | tuple,
        text_prompt: str | None = None,
        point_prompts: np.ndarray | None = None,
        box_prompts: np.ndarray | None = None,
    ) -> Frame | tuple:
        """
        Process a frame to generate segmentation masks using SAM3.

        Args:
            input_data: Input frame with RGB image, or tuple (Frame, ...) from dataset loaders
            text_prompt: Optional text prompt override (uses init value if None)
                        Supports comma-separated prompts for parallel processing
                        (e.g., "player, ball, net")
            point_prompts: Optional point prompts override (N, 2) array
            box_prompts: Optional box prompts override (N, 4) array

        Returns:
            Masks containing segmentation masks, bounding boxes, and scores

        Examples:
            # Text-prompted segmentation
            masks = sam3_node.process(frame, text_prompt="a red car")

            # Multi-prompt parallel processing
            masks = sam3_node.process(frame, text_prompt="player, ball, net")

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

        # Handle multiple text prompts (comma-separated)
        if text is not None and "," in text:
            # Split by comma and strip whitespace
            text_prompts = [prompt.strip() for prompt in text.split(",")]
            logger.debug(f"Processing {len(text_prompts)} prompts sequentially: {text_prompts}")
            
            # Set image once (compute backbone features once)
            inference_state = self.processor.set_image(pil_image)
            
            # Process each prompt and collect results
            all_masks = []
            all_boxes = []
            all_scores = []
            all_class_names = []
            
            for prompt_text in text_prompts:
                # Process this prompt
                output = self.processor.set_text_prompt(state=inference_state, prompt=prompt_text)
                
                if output is None:
                    logger.warning(f"No output from SAM3 for prompt: '{prompt_text}'")
                    continue
                
                # Extract results for this prompt
                prompt_masks = output["masks"]  # Shape: (N, H, W)
                prompt_boxes = output["boxes"]  # Shape: (N, 4)
                prompt_scores = output["scores"]  # Shape: (N,)
                
                # Convert tensors to numpy
                if isinstance(prompt_masks, torch.Tensor):
                    prompt_masks = prompt_masks.cpu().numpy()
                if isinstance(prompt_boxes, torch.Tensor):
                    prompt_boxes = prompt_boxes.cpu().numpy()
                if isinstance(prompt_scores, torch.Tensor):
                    prompt_scores = prompt_scores.cpu().numpy()
                
                # Apply mask threshold
                if self.mask_threshold > 0:
                    prompt_masks = prompt_masks > self.mask_threshold
                prompt_masks = prompt_masks.astype(bool)
                
                # Ensure 3D shape
                if prompt_masks.ndim == 2:
                    prompt_masks = prompt_masks[np.newaxis, ...]
                
                # Collect results
                for i in range(len(prompt_masks)):
                    all_masks.append(prompt_masks[i])
                    all_boxes.append(prompt_boxes[i])
                    all_scores.append(prompt_scores[i])
                    all_class_names.append(prompt_text)
            
            # Convert to arrays
            if len(all_masks) == 0:
                logger.warning("No masks generated from any prompts")
                pred_masks = np.array([]).reshape(0, *frame.rgb.shape[:2])
                pred_boxes = np.array([]).reshape(0, 4)
                pred_scores = np.array([])
                class_names = []
            else:
                pred_masks = np.array(all_masks, dtype=bool)
                pred_boxes = np.array(all_boxes, dtype=np.float32)
                pred_scores = np.array(all_scores, dtype=np.float32)
                class_names = all_class_names
            
        else:
            # Single prompt or no prompts - use original processing logic
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
            logger.debug(f"Processing with single text prompt: '{text}'")
            output = self.processor.set_text_prompt(state=inference_state, prompt=text)

            if output is None:
                logger.warning("No output from SAM3 model")
                # Return frame or tuple depending on input type
                if isinstance(input_data, tuple):
                    output_data = (frame, *input_data[1:])
                else:
                    output_data = frame
                return output_data

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

        # Determine if multiple prompts were used
        text_for_metadata = text if text is not None else None
        if text is not None and "," in text:
            num_prompts = len([p.strip() for p in text.split(",")])
            is_multi_prompt = True
        else:
            num_prompts = 1 if text is not None else 0
            is_multi_prompt = False

        output_masks = Masks(
            masks=pred_masks,
            boxes=pred_boxes,
            scores=pred_scores,
            labels=labels,  # SAM3 doesn't provide class IDs, only concept-based
            class_names=class_names,
            metadata={
                "model": "sam3",
                "text_prompt": text_for_metadata,
                "num_prompts": num_prompts,
                "is_multi_prompt": is_multi_prompt,
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

        prompt_info = (
            f"{num_prompts} prompts" if is_multi_prompt 
            else (text_for_metadata or 'visual/auto')
        )
        logger.info(
            f"Generated {output_masks.metadata['num_masks']} masks "
            f"(prompt: {prompt_info})"
        )
        return output_data
