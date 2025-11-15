"""
Template descriptor node for computing and storing feature descriptors from templates.

This node processes rendered templates to extract feature descriptors using DINOv2,
storing them with metadata for later matching in CNOS-style pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
import torch

from ..core.node import Node
from ..interfaces import Features, Frame

logger = logging.getLogger(__name__)


class TemplateDescriptor:
    """
    Container for template descriptors with metadata.

    Stores feature descriptors along with object and template identifiers
    for efficient querying during matching.

    Attributes:
        descriptors: Feature descriptors tensor (N, feature_dim)
        object_ids: Object IDs for each template (N,)
        template_ids: Template IDs within each object (N,)
        metadata: Additional metadata dictionary
    """

    def __init__(
        self,
        descriptors: torch.Tensor,
        object_ids: torch.Tensor,
        template_ids: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize template descriptor container.

        Args:
            descriptors: Feature descriptors (N, feature_dim)
            object_ids: Object ID for each descriptor (N,)
            template_ids: Template ID for each descriptor (N,)
            metadata: Optional metadata dictionary
        """
        self.descriptors = descriptors
        self.object_ids = object_ids
        self.template_ids = template_ids
        self.metadata = metadata or {}

    def save(self, path: str) -> None:
        """
        Save descriptors to file.

        Args:
            path: Path to save descriptors (will be saved as .pth file)
        """
        torch.save(
            {
                "descriptors": self.descriptors,
                "object_ids": self.object_ids,
                "template_ids": self.template_ids,
                "metadata": self.metadata,
            },
            path,
        )
        logger.info(f"Saved template descriptors to {path}")

    @classmethod
    def load(cls, path: str) -> TemplateDescriptor:
        """
        Load descriptors from file.

        Args:
            path: Path to load descriptors from

        Returns:
            TemplateDescriptor instance
        """
        data = torch.load(path)
        logger.info(f"Loaded template descriptors from {path}")
        return cls(
            descriptors=data["descriptors"],
            object_ids=data["object_ids"],
            template_ids=data["template_ids"],
            metadata=data.get("metadata", {}),
        )

    def get_descriptors_for_object(self, object_id: int) -> torch.Tensor:
        """
        Get all descriptors for a specific object.

        Args:
            object_id: Object ID to query

        Returns:
            Tensor of descriptors for the object (M, feature_dim)
        """
        mask = self.object_ids == object_id
        return self.descriptors[mask]

    def __len__(self) -> int:
        """Return number of templates."""
        return len(self.descriptors)


class TemplateDescriptorNode(Node):
    """
    Node for computing feature descriptors from rendered templates.

    This node takes rendered templates (Frame objects) and extracts feature
    descriptors using a feature extractor node (typically Dinov2Node). The
    descriptors are stored with object and template metadata for use in
    CNOS-style matching pipelines.

    Args:
        feature_extractor: Node for extracting features (e.g., Dinov2Node)
        device: Device to use for computation ("cpu", "cuda", or None for auto)
        cache_dir: Directory to cache computed descriptors (None = no caching)
        dataset_name: Name of dataset for cache file naming (required if cache_dir is set)
        name: Optional name for the node
        **kwargs: Additional configuration parameters
    """

    def __init__(
        self,
        feature_extractor: Node,
        device: str | None = None,
        cache_dir: str | None = None,
        dataset_name: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initialize the template descriptor node.

        Args:
            feature_extractor: Feature extraction node (e.g., Dinov2Node)
            device: Device for computation ("cpu", "cuda", or None for auto)
            cache_dir: Directory to cache computed descriptors
            dataset_name: Name of dataset for cache file naming
            name: Optional name for the node
            **kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)
        self.feature_extractor = feature_extractor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name

        if self.cache_dir and not self.dataset_name:
            logger.warning("cache_dir specified but no dataset_name provided; caching will be disabled")
            self.cache_dir = None

    def _get_cache_path(self, object_id: int) -> str | None:
        """Get cache file path for an object's descriptors."""
        if not self.cache_dir or not self.dataset_name:
            return None

        from pathlib import Path
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return str(cache_path / f"{self.dataset_name}_obj_{object_id:06d}_descriptors.pth")

    def _load_from_cache(self, object_id: int) -> TemplateDescriptor | None:
        """Try to load descriptors from cache."""
        cache_path = self._get_cache_path(object_id)
        if not cache_path:
            return None

        from pathlib import Path
        if Path(cache_path).exists():
            logger.info(f"Loading cached descriptors from {cache_path}")
            return TemplateDescriptor.load(cache_path)

        return None

    def _save_to_cache(self, descriptors: TemplateDescriptor, object_id: int) -> None:
        """Save descriptors to cache."""
        cache_path = self._get_cache_path(object_id)
        if cache_path:
            descriptors.save(cache_path)
            logger.info(f"Saved descriptors to cache: {cache_path}")

    def process_templates(
        self,
        templates: Iterator[Frame],
        object_id: int,
    ) -> TemplateDescriptor:
        """
        Process templates for a single object to extract descriptors.

        Args:
            templates: Iterator of Frame objects containing rendered templates
            object_id: ID of the object being processed

        Returns:
            TemplateDescriptor containing extracted features
        """
        # Try to load from cache first
        cached_descriptors = self._load_from_cache(object_id)
        if cached_descriptors is not None:
            return cached_descriptors

        descriptors_list = []
        template_ids = []

        logger.info(f"Processing templates for object {object_id}")

        for template_id, frame in enumerate(templates):
            # Extract features from template
            features: Features = self.feature_extractor.process(frame)

            # Use embeddings (CLS token) if available, otherwise use mean of descriptors
            if features.embeddings is not None:
                descriptor = features.embeddings
            elif features.descriptors is not None and len(features.descriptors.shape) > 1:
                # Average patch features to get global descriptor
                descriptor = np.mean(features.descriptors, axis=0)
            else:
                descriptor = features.descriptors

            # Convert to tensor
            if isinstance(descriptor, np.ndarray):
                descriptor = torch.from_numpy(descriptor).float()

            descriptors_list.append(descriptor)
            template_ids.append(template_id)

        # Stack all descriptors
        descriptors = torch.stack(descriptors_list)  # (N_templates, feature_dim)
        object_ids = torch.full(
            (len(descriptors),), object_id, dtype=torch.long
        )
        template_ids = torch.tensor(template_ids, dtype=torch.long)

        logger.info(
            f"Extracted {len(descriptors)} descriptors for object {object_id}, "
            f"shape: {descriptors.shape}"
        )

        template_descriptor = TemplateDescriptor(
            descriptors=descriptors,
            object_ids=object_ids,
            template_ids=template_ids,
            metadata={
                "object_id": object_id,
                "num_templates": len(descriptors),
                "feature_dim": descriptors.shape[1],
            },
        )

        # Save to cache
        self._save_to_cache(template_descriptor, object_id)

        return template_descriptor

    def process_multiple_objects(
        self,
        object_templates: dict[int, Iterator[Frame]],
    ) -> TemplateDescriptor:
        """
        Process templates for multiple objects.

        Args:
            object_templates: Dictionary mapping object IDs to template iterators

        Returns:
            TemplateDescriptor containing descriptors for all objects
        """
        all_descriptors = []
        all_object_ids = []
        all_template_ids = []

        for object_id, templates in object_templates.items():
            template_desc = self.process_templates(templates, object_id)
            all_descriptors.append(template_desc.descriptors)
            all_object_ids.append(template_desc.object_ids)
            all_template_ids.append(template_desc.template_ids)

        # Concatenate all descriptors
        descriptors = torch.cat(all_descriptors, dim=0)
        object_ids = torch.cat(all_object_ids, dim=0)
        template_ids = torch.cat(all_template_ids, dim=0)

        logger.info(
            f"Processed {len(object_templates)} objects, "
            f"total {len(descriptors)} templates"
        )

        return TemplateDescriptor(
            descriptors=descriptors,
            object_ids=object_ids,
            template_ids=template_ids,
            metadata={
                "num_objects": len(object_templates),
                "num_templates": len(descriptors),
                "feature_dim": descriptors.shape[1],
            },
        )

    def process(
        self,
        input_data: tuple[Iterator[Frame], int] | dict[int, Iterator[Frame]],
    ) -> TemplateDescriptor:
        """
        Process method for pipeline compatibility.

        Args:
            input_data: Either a tuple of (templates, object_id) for single object
                       or a dict mapping object IDs to template iterators

        Returns:
            TemplateDescriptor with extracted features
        """
        if isinstance(input_data, dict):
            return self.process_multiple_objects(input_data)
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            templates, object_id = input_data
            return self.process_templates(templates, object_id)
        else:
            raise ValueError(
                "Input must be either a tuple of (templates, object_id) "
                "or a dict mapping object IDs to template iterators"
            )
