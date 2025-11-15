"""
CNOS matcher node for matching query descriptors against template database.

This node implements the matching logic from CNOS pipeline, comparing query
descriptors against pre-computed template descriptors using cosine similarity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from ..core.node import Node
from ..interfaces import Features, Masks

logger = logging.getLogger(__name__)


class CNOSMatcherNode(Node):
    """
    Node for matching query proposals against template descriptors.

    This node implements the CNOS matching pipeline:
    1. Compute cosine similarity between query and template descriptors
    2. Aggregate scores per object (e.g., top-k averaging)
    3. Assign each proposal to best matching object
    4. Filter by confidence threshold and max instances

    Args:
        template_descriptors: Pre-computed template descriptors
        confidence_threshold: Minimum confidence score for matches (default: 0.5)
        max_instances: Maximum number of instances to return (default: 100)
        aggregation: Aggregation method for template scores
                    ("mean", "max", "median", "avg_5")
        device: Device for computation ("cpu", "cuda", or None for auto)
        name: Optional name for the node
        **kwargs: Additional configuration parameters
    """

    def __init__(
        self,
        template_descriptors: Any,  # TemplateDescriptor from template_descriptor.py
        confidence_threshold: float = 0.5,
        max_instances: int = 100,
        aggregation: str = "avg_5",
        device: str | None = None,
        enable_rerun_logging: bool = False,
        rerun_entity_path: str = "cnos/detections",
        name: str | None = None,
        **kwargs,
    ):
        """
        Initialize CNOS matcher node.

        Args:
            template_descriptors: TemplateDescriptor object with pre-computed features
            confidence_threshold: Minimum confidence for matches
            max_instances: Maximum number of instances to return
            aggregation: Score aggregation method
            device: Device for computation
            enable_rerun_logging: Whether to enable Rerun logging
            rerun_entity_path: Entity path for Rerun logging
            name: Optional name for the node
            **kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)
        self.template_descriptors = template_descriptors
        self.confidence_threshold = confidence_threshold
        self.max_instances = max_instances
        self.aggregation = aggregation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_rerun_logging = enable_rerun_logging
        self.rerun_entity_path = rerun_entity_path
        self.rerun_logger = None

        # Initialize Rerun logger if enabled
        if enable_rerun_logging:
            try:
                from ..utils.rerun_logger import RerunLogger
                self.rerun_logger = RerunLogger("CNOS_Matcher", enabled=True, spawn=False)
            except ImportError:
                logger.warning("Rerun not available, logging disabled")
                self.enable_rerun_logging = False

        # Move template descriptors to device
        if hasattr(self.template_descriptors, "descriptors"):
            self.template_descriptors.descriptors = (
                self.template_descriptors.descriptors.to(self.device)
            )
            self.template_descriptors.object_ids = (
                self.template_descriptors.object_ids.to(self.device)
            )

        # Get unique object IDs and organize descriptors by object
        self.unique_object_ids = torch.unique(
            self.template_descriptors.object_ids
        ).tolist()
        self.num_objects = len(self.unique_object_ids)

        # Organize descriptors by object for efficient matching
        # Shape: (num_objects, num_templates_per_object, feature_dim)
        self._organize_descriptors()

        logger.info(
            f"Initialized CNOS matcher with {self.num_objects} objects, "
            f"{len(self.template_descriptors)} total templates"
        )

    def _organize_descriptors(self) -> None:
        """Organize template descriptors by object ID for efficient matching."""
        descriptors_per_object = []
        templates_per_object = []

        for obj_id in self.unique_object_ids:
            obj_descriptors = self.template_descriptors.get_descriptors_for_object(
                obj_id
            )
            descriptors_per_object.append(obj_descriptors)
            templates_per_object.append(len(obj_descriptors))

        # Store as tensor: (num_objects, max_templates, feature_dim)
        max_templates = max(templates_per_object)
        feature_dim = descriptors_per_object[0].shape[1]

        self.ref_descriptors = torch.zeros(
            (self.num_objects, max_templates, feature_dim),
            device=self.device,
        )
        self.templates_per_object = torch.tensor(
            templates_per_object, device=self.device
        )

        for i, obj_descriptors in enumerate(descriptors_per_object):
            self.ref_descriptors[i, : len(obj_descriptors)] = obj_descriptors

        logger.debug(f"Organized descriptors shape: {self.ref_descriptors.shape}")

    def compute_similarity(
        self, query_descriptors: torch.Tensor, ref_descriptors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and reference descriptors.

        Args:
            query_descriptors: Query descriptors (N_proposals, feature_dim)
            ref_descriptors: Reference descriptors (N_objects, N_templates, feature_dim)

        Returns:
            Similarity scores (N_proposals, N_objects, N_templates)
        """
        # Normalize descriptors
        query_norm = functional.normalize(query_descriptors, p=2, dim=-1)  # (N_proposals, dim)
        ref_norm = functional.normalize(ref_descriptors, p=2, dim=-1)  # (N_objects, N_templates, dim)

        # Compute cosine similarity
        # (N_proposals, 1, dim) @ (N_objects, dim, N_templates)
        # = (N_proposals, N_objects, N_templates)
        similarity = torch.einsum("nd,otd->not", query_norm, ref_norm)

        return similarity

    def aggregate_scores(
        self, scores: torch.Tensor, templates_per_object: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate template scores per object.

        Args:
            scores: Similarity scores (N_proposals, N_objects, N_templates)
            templates_per_object: Number of valid templates per object (N_objects,)

        Returns:
            Aggregated scores (N_proposals, N_objects)
        """
        if self.aggregation == "mean":
            # Average over valid templates
            score_sum = torch.sum(scores, dim=-1)  # (N_proposals, N_objects)
            score_per_proposal_and_object = score_sum / templates_per_object[None, :]
        elif self.aggregation == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.aggregation == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.aggregation == "avg_5":
            # Top-5 averaging (CNOS default)
            k = min(5, scores.shape[-1])
            top_k_scores = torch.topk(scores, k=k, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(top_k_scores, dim=-1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return score_per_proposal_and_object

    def match_proposals(
        self, query_descriptors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Match query proposals to template objects.

        Args:
            query_descriptors: Query descriptors (N_proposals, feature_dim)

        Returns:
            Tuple of (matched_object_ids, confidence_scores)
        """
        # Compute similarity scores
        scores = self.compute_similarity(
            query_descriptors, self.ref_descriptors
        )  # (N_proposals, N_objects, N_templates)

        # Aggregate scores per object
        score_per_proposal_and_object = self.aggregate_scores(
            scores, self.templates_per_object
        )  # (N_proposals, N_objects)

        # Assign each proposal to best matching object
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # (N_proposals,)

        # Filter by confidence threshold
        idx_selected = score_per_proposal > self.confidence_threshold

        # Limit to max instances
        if idx_selected.sum() > self.max_instances:
            logger.info(
                f"Selecting top {self.max_instances} instances "
                f"from {idx_selected.sum()} candidates"
            )
            scores_selected = score_per_proposal[idx_selected]
            _, top_k_idx = torch.topk(scores_selected, k=self.max_instances)

            # Create new selection mask
            selected_indices = torch.where(idx_selected)[0][top_k_idx]
            idx_selected = torch.zeros_like(idx_selected)
            idx_selected[selected_indices] = True

        # Get matched object IDs (convert from indices to actual IDs)
        matched_indices = assigned_idx_object[idx_selected]
        matched_object_ids = torch.tensor(
            [self.unique_object_ids[i] for i in matched_indices.cpu().numpy()],
            device=self.device,
        )
        matched_scores = score_per_proposal[idx_selected]

        return matched_object_ids, matched_scores

    def process(self, input_data: tuple[Features, Masks]) -> Masks:
        """
        Process query proposals to find matches in template database.

        Args:
            input_data: Tuple of (query_features, query_masks)
                       - query_features: Features extracted from query proposals
                       - query_masks: Original masks from segmentation

        Returns:
            Masks interface with matched object IDs (as labels) and scores
        """
        query_features, query_masks = input_data

        # Extract descriptors from features
        if query_features.embeddings is not None:
            # Use CLS token embeddings (global features)
            query_descriptors = query_features.embeddings
        elif query_features.descriptors is not None:
            # Use descriptors (may need aggregation)
            descriptors = query_features.descriptors
            if len(descriptors.shape) > 2:
                # Average over patches/spatial dimensions
                query_descriptors = np.mean(descriptors, axis=1)
            else:
                query_descriptors = descriptors
        else:
            raise ValueError("Features must contain either embeddings or descriptors")

        # Convert to tensor
        if isinstance(query_descriptors, np.ndarray):
            query_descriptors = torch.from_numpy(query_descriptors).float()
        query_descriptors = query_descriptors.to(self.device)

        # Match proposals
        matched_object_ids, matched_scores = self.match_proposals(query_descriptors)

        # Filter original masks
        num_proposals = len(query_masks.masks)

        # Map selected indices back to original proposal indices
        # This requires tracking which proposals passed filtering
        all_scores = torch.zeros(num_proposals, device=self.device)

        # We need to recompute to get per-proposal assignments
        query_descriptors_all = torch.from_numpy(
            query_features.embeddings if query_features.embeddings is not None
            else query_features.descriptors
        ).float().to(self.device)

        scores = self.compute_similarity(query_descriptors_all, self.ref_descriptors)
        score_per_proposal_and_object = self.aggregate_scores(
            scores, self.templates_per_object
        )
        all_scores, assigned_indices = torch.max(score_per_proposal_and_object, dim=-1)

        # Select proposals above threshold
        selected_mask = (all_scores > self.confidence_threshold).cpu().numpy()

        # Limit to max instances
        if selected_mask.sum() > self.max_instances:
            scores_selected = all_scores[selected_mask]
            _, top_k_idx = torch.topk(scores_selected, k=self.max_instances)
            selected_indices = np.where(selected_mask)[0][top_k_idx.cpu().numpy()]
            selected_mask = np.zeros(num_proposals, dtype=bool)
            selected_mask[selected_indices] = True

        # Get final matched object IDs
        matched_indices = assigned_indices[selected_mask]
        final_object_ids = np.array(
            [self.unique_object_ids[i] for i in matched_indices.cpu().numpy()]
        )
        final_scores = all_scores[selected_mask].cpu().numpy()

        # Filter masks and boxes
        filtered_masks = query_masks.masks[selected_mask] if query_masks.masks is not None else None
        filtered_boxes = query_masks.boxes[selected_mask] if query_masks.boxes is not None else None

        logger.info(
            f"Matched {len(final_object_ids)} proposals "
            f"(confidence > {self.confidence_threshold})"
        )

        # Create result Masks interface
        result_masks = Masks(
            masks=filtered_masks if filtered_masks is not None else np.array([]),
            boxes=filtered_boxes if filtered_boxes is not None else np.array([]),
            scores=final_scores,
            labels=final_object_ids.astype(np.int32),
            class_names=None,
            metadata={
                "num_proposals": num_proposals,
                "num_matched": len(final_object_ids),
                "confidence_threshold": self.confidence_threshold,
                "aggregation": self.aggregation,
            },
        )

        # Log to Rerun if enabled
        if self.enable_rerun_logging and self.rerun_logger and len(final_object_ids) > 0:
            self.rerun_logger.log_masks(result_masks, entity_path=self.rerun_entity_path)
            self.rerun_logger.log_metadata(
                {
                    "num_detections": len(final_object_ids),
                    "confidence_threshold": self.confidence_threshold,
                    "detected_objects": final_object_ids.tolist(),
                    "detection_scores": final_scores.tolist(),
                },
                entity_path=f"{self.rerun_entity_path}/metadata",
            )

        return result_masks
