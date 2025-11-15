"""
Segmentation node using SAM2 (Segment Anything Model 2).

This node performs automatic mask generation on RGB images using
the SAM2 model from Meta/Facebook available on Hugging Face.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

from ..core.node import Node
from ..interfaces import Frame, Masks

logger = logging.getLogger(__name__)


class SAM2Node(Node):
    """
    Node for automatic mask generation using SAM2 (Segment Anything Model 2).

    This node uses the SAM2 model from Meta/Facebook via Hugging Face transformers
    to generate multiple candidate instance segmentation masks with bounding boxes.

    The model supports various sizes:
    - sam2-hiera-tiny
    - sam2-hiera-small
    - sam2-hiera-base-plus
    - sam2-hiera-large

    Larger models provide better accuracy but are slower and use more memory.

    Note: This node requires transformers >= 4.56.0 for SAM2 support.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str | None = None,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.7,
        mask_threshold: float = 0.5,
        enable_rerun_logging: bool = True,
        rerun_entity_path: str = "sam2",
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the SAM2 node.

        Args:
            model_size: Size of the model ("tiny", "small", "base-plus", or "large")
            device: Device to run the model on ("cpu", "cuda", or None for auto)
            points_per_batch: Number of points per batch for mask generation (will use closest perfect square)
            pred_iou_thresh: IoU threshold for filtering masks
            mask_threshold: Threshold for converting interpolated masks to boolean (default: 0.5)
            enable_rerun_logging: Whether to enable Rerun visualization logging
            rerun_entity_path: Base entity path for Rerun logging
            name: Optional name for the node
            **kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)
        self.model_size = model_size
        self.device = device
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.mask_threshold = mask_threshold
        self.enable_rerun_logging = enable_rerun_logging
        self.rerun_entity_path = rerun_entity_path
        self.model = None
        self.processor = None
        self._frame_counter = 0

        # Initialize Rerun logger if enabled
        self.rerun_logger = None
        if enable_rerun_logging:
            from ..utils.rerun_logger import RerunLogger

            self.rerun_logger = RerunLogger(
                recording_name="SAM2",
                enabled=True,
                spawn=True,
            )

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SAM2 model from Hugging Face."""

        model_name_map = {
            "tiny": "facebook/sam2-hiera-tiny",
            "small": "facebook/sam2-hiera-small",
            "base-plus": "facebook/sam2-hiera-base-plus",
            "large": "facebook/sam2-hiera-large",
        }

        if self.model_size not in model_name_map:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. "
                f"Must be one of {list(model_name_map.keys())}"
            )

        model_name = model_name_map[self.model_size]

        if self.device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        logger.info(f"Loading SAM2 model: {model_name}")
        logger.info(f"Device: {device}")

        try:
            from transformers import Sam2Model, Sam2Processor

            self.processor = Sam2Processor.from_pretrained(model_name)

            # NOTE: This produces a warning about model type mismatch (sam2_video vs sam2).
            # This is because:
            # 1. The pretrained weights on HuggingFace use Sam2VideoConfig (model_type='sam2_video')
            # 2. But Sam2Model expects Sam2Config (model_type='sam2')
            # 3. These are separate class hierarchies in transformers library
            #
            # We cannot fix this by providing a config on initialization because:
            # - Sam2Model.from_pretrained() loads the config from the checkpoint
            # - The checkpoint has sam2_video config baked in
            # - Manually overriding config would require re-initializing weights, losing pretrained values
            #
            # The warning is harmless - the models are compatible and work correctly for image tasks.
            # The proper fix would be for Meta/HuggingFace to provide separate checkpoints with
            # Sam2Config, or to unify the config types in the transformers library.
            self.model = Sam2Model.from_pretrained(model_name)

            self.model = self.model.to(device)
            self.model.eval()

            logger.info("SAM2 model loaded successfully!")
        except ImportError as e:
            logger.error(
                f"Failed to import SAM2 classes. "
                f"Please ensure transformers >= 4.56.0 is installed: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise

    def process(self, frame: Frame) -> Masks:
        """
        Process a frame to generate instance segmentation masks.

        This performs automatic mask generation, detecting all objects in the image
        and returning multiple candidate masks with bounding boxes and confidence scores.

        Args:
            frame: Input frame with RGB image

        Returns:
            Masks containing segmentation masks, bounding boxes, and scores
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        if frame.rgb is None:
            raise ValueError("Frame must contain an RGB image")

        # Log frame to Rerun if enabled
        if self.enable_rerun_logging and self.rerun_logger:
            self.rerun_logger.set_time_sequence("frame", self._frame_counter)
            self.rerun_logger.log_frame(
                frame,
                entity_path=f"{self.rerun_entity_path}/camera",
            )

        rgb_image = frame.rgb
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        pil_image = Image.fromarray(rgb_image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        h, w = rgb_image.shape[:2]

        num_points_side = int(np.sqrt(self.points_per_batch))
        actual_points = num_points_side * num_points_side

        x_points = np.linspace(0, w - 1, num_points_side)
        y_points = np.linspace(0, h - 1, num_points_side)
        xx, yy = np.meshgrid(x_points, y_points)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)[:actual_points]

        all_masks = []
        all_scores = []

        with torch.no_grad():
            # (1, num_prompts, 1, 2)
            point_coords = torch.from_numpy(points[:, np.newaxis, :]).to(
                dtype=torch.float32, device=self.model.device
            ).unsqueeze(0)

            point_labels = torch.ones((1, actual_points, 1), dtype=torch.int32, device=self.model.device)

            logger.debug(f"Processing {actual_points} points in batch. Point coords shape: {point_coords.shape}")

            outputs = self.model(
                **inputs,
                input_points=point_coords,
                input_labels=point_labels,
            )

            masks = outputs.pred_masks.cpu().numpy()
            scores = outputs.iou_scores.cpu().numpy()

            logger.debug(f"Output masks shape: {masks.shape}, scores shape: {scores.shape}")

            if masks.ndim == 5:  # (batch, num_prompts, num_masks_per_prompt, h, w)
                masks = masks[0]  # (num_prompts, num_masks_per_prompt, h, w)
                scores = scores[0]  # (num_prompts, num_masks_per_prompt)

                for i in range(len(masks)):
                    prompt_masks = masks[i]  # (num_masks_per_prompt, h, w)
                    prompt_scores = scores[i]  # (num_masks_per_prompt,)

                    if len(prompt_scores) > 0:
                        best_idx = np.argmax(prompt_scores)
                        best_mask = prompt_masks[best_idx]
                        best_score = float(prompt_scores[best_idx])

                        if best_score > self.pred_iou_thresh:
                            all_masks.append(best_mask)
                            all_scores.append(best_score)

            elif masks.ndim == 4:  # (batch, num_prompts, h, w)
                masks = masks[0]  # (num_prompts, h, w)
                scores = scores[0]  # (num_prompts,)

                for i in range(len(masks)):
                    mask = masks[i]
                    score = float(scores[i])

                    if score > self.pred_iou_thresh:
                        all_masks.append(mask)
                        all_scores.append(score)
            else:
                logger.error(f"Unexpected mask output shape: {masks.shape}. Expected 4D or 5D tensor.")
                raise ValueError(f"Unexpected mask dimensions: {masks.ndim}")

        if len(all_masks) == 0:
            logger.warning("No masks found with current thresholds")
            output_masks = Masks(
                masks=np.zeros((0, h, w), dtype=bool),
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                labels=None,
                class_names=["object"],
                metadata={
                    "model": f"sam2-hiera-{self.model_size}",
                    "points_per_batch": actual_points,
                    "pred_iou_thresh": self.pred_iou_thresh,
                    "num_masks": 0,
                },
            )
        else:
            pred_masks = np.stack(all_masks, axis=0)
            pred_scores = np.array(all_scores, dtype=np.float32)

            if pred_masks.shape[1:] != (h, w):
                zoom_factors = (1, h / pred_masks.shape[1], w / pred_masks.shape[2])
                pred_masks = zoom(pred_masks, zoom_factors, order=0) > self.mask_threshold

            boxes = self._masks_to_boxes(pred_masks)

            output_masks = Masks(
                masks=pred_masks.astype(bool),
                boxes=boxes,
                scores=pred_scores,
                labels=None,  # SAM2 doesn't provide class labels, only masks
                class_names=["object"],  # Generic object class
                metadata={
                    "model": f"sam2-hiera-{self.model_size}",
                    "points_per_batch": actual_points,
                    "pred_iou_thresh": self.pred_iou_thresh,
                    "mask_threshold": self.mask_threshold,
                    "num_masks": len(pred_masks),
                },
            )

        # Log masks to Rerun if enabled
        if self.enable_rerun_logging and self.rerun_logger:
            logger.info(f"Frame {self._frame_counter}: Generated {output_masks.metadata['num_masks']} masks")
            self.rerun_logger.log_masks(output_masks, entity_path=f"{self.rerun_entity_path}/segmentation")

            # Log metadata
            self.rerun_logger.log_metadata(
                {
                    "frame_id": self._frame_counter,
                    "num_masks": output_masks.metadata["num_masks"],
                    "model": output_masks.metadata["model"],
                },
                entity_path=f"{self.rerun_entity_path}/metadata/frame_{self._frame_counter}",
            )

            self._frame_counter += 1

        return output_masks

    def _masks_to_boxes(self, masks: np.ndarray) -> np.ndarray:
        """
        Convert binary masks to bounding boxes.

        Args:
            masks: Binary masks of shape (N, H, W)

        Returns:
            Bounding boxes as numpy array (N, 4) in [x1, y1, x2, y2] format
        """
        boxes = []
        for mask in masks:
            y_indices, x_indices = np.where(mask > 0)

            if len(x_indices) == 0 or len(y_indices) == 0:
                boxes.append([0, 0, 0, 0])
            else:
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                boxes.append([x_min, y_min, x_max, y_max])

        return np.array(boxes, dtype=np.float32)
