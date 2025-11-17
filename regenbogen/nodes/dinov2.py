"""
Feature extraction node using Dinov2 model.

This node computes dense feature descriptors from RGB images using
the Dinov2 model from Meta/Facebook available on Hugging Face.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from ..core.node import Node
from ..interfaces import Features, Frame

logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-8  # Small constant to prevent division by zero in normalization

# Optional imports for visualization
try:
    from sklearn.decomposition import PCA

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("scikit-learn not available, PCA visualization will be disabled")

try:
    import rerun as rr

    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False
    logger.debug("rerun-sdk not available, advanced logging will be disabled")


class Dinov2Node(Node):
    """
    Node for computing feature descriptors using Dinov2 (Vision Transformer).

    This node uses the Dinov2 model from Meta/Facebook via Hugging Face transformers
    to compute dense feature descriptors from RGB images. Dinov2 is a self-supervised
    vision transformer that provides powerful image features for downstream tasks.

    The model supports various sizes:
    - dinov2-small
    - dinov2-base
    - dinov2-large
    - dinov2-giant

    Larger models provide richer features but are slower and use more memory.

    Note: This node requires transformers library for Dinov2 support.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        output_type: str = "patch",
        enable_rerun_logging: bool = True,
        rerun_entity_path: str = "dinov2",
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the Dinov2 node.

        Args:
            model_size: Size of the model ("small", "base", "large", or "giant")
            device: Device to run the model on ("cpu", "cuda", or None for auto)
            output_type: Type of output features ("patch" for dense patch features,
                        "cls" for global CLS token, "both" for both)
            enable_rerun_logging: Whether to enable Rerun visualization logging
            rerun_entity_path: Base entity path for Rerun logging
            name: Optional name for the node
            **kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)
        self.model_size = model_size
        self.device = device
        self.output_type = output_type
        self.enable_rerun_logging = enable_rerun_logging
        self.rerun_entity_path = rerun_entity_path
        self.model = None
        self.processor = None
        self._frame_counter = 0

        if output_type not in ["patch", "cls", "both"]:
            raise ValueError(
                f"Invalid output_type: {output_type}. "
                f"Must be one of ['patch', 'cls', 'both']"
            )

        # Initialize Rerun logger if enabled
        self.rerun_logger = None
        if enable_rerun_logging:
            from ..utils.rerun_logger import RerunLogger

            self.rerun_logger = RerunLogger(
                recording_name="Dinov2",
                enabled=True,
                spawn=True,
            )

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Dinov2 model from Hugging Face."""

        model_name_map = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large",
            "giant": "facebook/dinov2-giant",
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

        logger.info(f"Loading Dinov2 model: {model_name}")
        logger.info(f"Device: {device}")

        try:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            self.model = self.model.to(device)
            self.model.eval()

            logger.info("Dinov2 model loaded successfully!")
        except ImportError as e:
            logger.error(
                f"Failed to import transformers classes. "
                f"Please ensure transformers library is installed: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Dinov2 model: {e}")
            raise

    def process(self, frame: Frame) -> Features:
        """
        Process a frame to compute feature descriptors.

        This extracts dense patch-level features from the image using the Dinov2 model.
        Each patch corresponds to a 14x14 pixel region in the input image (after resizing).

        Args:
            frame: Input frame with RGB image

        Returns:
            Features containing descriptors and optional embeddings
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

        # Preprocess image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

            # Extract features based on output_type
            if self.output_type == "cls":
                # Use CLS token as global image embedding
                cls_token = outputs.last_hidden_state[:, 0]  # (1, hidden_dim)
                descriptors = cls_token.cpu().numpy()[0]  # (hidden_dim,)
                embeddings = None
            elif self.output_type == "patch":
                # Use patch tokens as dense descriptors
                patch_tokens = outputs.last_hidden_state[:, 1:]  # (1, num_patches, hidden_dim)
                descriptors = patch_tokens.cpu().numpy()[0]  # (num_patches, hidden_dim)
                embeddings = None
            else:  # "both"
                cls_token = outputs.last_hidden_state[:, 0]
                patch_tokens = outputs.last_hidden_state[:, 1:]
                descriptors = patch_tokens.cpu().numpy()[0]  # (num_patches, hidden_dim)
                embeddings = cls_token.cpu().numpy()[0]  # (hidden_dim,)

        # Calculate spatial dimensions of feature map
        # Dinov2 uses patch size of 14x14
        patch_size = 14
        h, w = rgb_image.shape[:2]

        # Processor resizes to 518x518 for dinov2-small/base, 224x224 for others
        # Calculate based on model's expected size
        if hasattr(self.processor, 'size'):
            if isinstance(self.processor.size, dict):
                input_size = self.processor.size.get('height', 224)
            else:
                input_size = self.processor.size
        else:
            input_size = 224

        num_patches_per_side = input_size // patch_size

        output_features = Features(
            descriptors=descriptors,
            embeddings=embeddings,
            metadata={
                "model": f"dinov2-{self.model_size}",
                "output_type": self.output_type,
                "descriptor_shape": descriptors.shape,
                "feature_dim": descriptors.shape[-1],
                "num_patches": descriptors.shape[0] if len(descriptors.shape) > 1 else None,
                "num_patches_per_side": num_patches_per_side,
                "patch_size": patch_size,
                "input_size": input_size,
            },
        )

        # Log features to Rerun if enabled
        if self.enable_rerun_logging and self.rerun_logger:
            logger.info(
                f"Frame {self._frame_counter}: Computed {self.output_type} features "
                f"with shape {descriptors.shape}"
            )

            # Log feature statistics
            self.rerun_logger.log_metadata(
                {
                    "frame_id": self._frame_counter,
                    "output_type": self.output_type,
                    "descriptor_shape": str(descriptors.shape),
                    "feature_dim": int(descriptors.shape[-1]),
                    "descriptor_mean": float(np.mean(descriptors)),
                    "descriptor_std": float(np.std(descriptors)),
                    "descriptor_min": float(np.min(descriptors)),
                    "descriptor_max": float(np.max(descriptors)),
                },
                entity_path=f"{self.rerun_entity_path}/metadata/frame_{self._frame_counter}",
            )

            # Visualize feature maps if patch features
            should_visualize_features = (
                self.output_type in ["patch", "both"]
                and len(descriptors.shape) > 1
                and HAS_SKLEARN
                and HAS_RERUN
            )

            if should_visualize_features:
                # Reshape descriptors to spatial grid for visualization
                feature_dim = descriptors.shape[1]
                num_patches = descriptors.shape[0]

                # Validate that number of patches matches expected grid
                expected_patches = num_patches_per_side * num_patches_per_side
                if num_patches != expected_patches:
                    logger.warning(
                        f"Number of patches ({num_patches}) doesn't match "
                        f"expected grid size ({expected_patches}). "
                        f"Skipping feature visualization."
                    )
                else:
                    # Compute PCA for visualization (first 3 components as RGB)
                    pca = PCA(n_components=min(3, feature_dim))
                    features_pca = pca.fit_transform(descriptors)  # (num_patches, 3)

                    # Normalize to [0, 1] for visualization
                    features_pca = (features_pca - features_pca.min()) / (
                        features_pca.max() - features_pca.min() + EPSILON
                    )

                    # Reshape to spatial grid
                    features_spatial = features_pca.reshape(
                        num_patches_per_side, num_patches_per_side, -1
                    )

                    # Convert to uint8 for logging
                    features_viz = (features_spatial * 255).astype(np.uint8)

                    # Log as image
                    rr.log(
                        f"{self.rerun_entity_path}/feature_map",
                        rr.Image(features_viz),
                    )

            self._frame_counter += 1

        return output_features

    def process_masked_batch(
        self,
        rgb_image: np.ndarray,
        masks: any,  # Masks interface
    ) -> Features:
        """
        Process multiple masked regions as a batch for efficient feature extraction.

        This method extracts features from multiple masked regions of an image in a single
        forward pass, which is more efficient than processing each region individually.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            masks: Masks interface containing segmentation masks

        Returns:
            Features object with batch of embeddings (N, feature_dim) where N is number of masks
        """
        self._initialize_model()

        if len(masks.masks) == 0:
            # Return empty features
            return Features(
                embeddings=np.array([]),
                metadata={"num_masks": 0}
            )

        # Process each masked region
        batch_features = []
        for i, mask in enumerate(masks.masks):
            # Create masked RGB
            masked_rgb = rgb_image.copy()
            masked_rgb[~mask] = 0

            # Convert to PIL and preprocess
            pil_image = Image.fromarray(masked_rgb)
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get CLS token (global feature) for this mask
            cls_token = outputs.last_hidden_state[:, 0]  # (1, hidden_dim)
            feature = cls_token.cpu().numpy()[0]  # (hidden_dim,)
            batch_features.append(feature)

        # Stack all features
        embeddings = np.stack(batch_features)  # (N, hidden_dim)

        return Features(
            embeddings=embeddings,
            metadata={
                "model": f"dinov2-{self.model_size}",
                "output_type": "cls_batch",
                "num_masks": len(masks.masks),
                "feature_dim": embeddings.shape[1],
            },
        )
