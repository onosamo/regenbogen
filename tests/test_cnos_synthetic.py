"""
Simplified CNOS pipeline test with synthetic data.

This script tests the CNOS pipeline without requiring large model downloads
or real BOP datasets. It uses synthetic data to verify the pipeline works correctly.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from regenbogen.interfaces import Features, Frame, Masks
from regenbogen.nodes import (
    CNOSMatcherNode,
    TemplateDescriptor,
    TemplateDescriptorNode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticFeatureExtractor:
    """Synthetic feature extractor for testing without DINOv2."""

    def __init__(self, feature_dim=384):
        self.feature_dim = feature_dim
        # Use fixed random state for reproducibility
        self.rng = np.random.RandomState(42)

    def process(self, frame: Frame) -> Features:
        """Extract synthetic features based on image brightness."""
        # Use image statistics to create "features" that are somewhat consistent
        brightness = np.mean(frame.rgb) / 255.0
        noise = self.rng.randn(self.feature_dim).astype(np.float32) * 0.1
        base_features = np.ones(self.feature_dim, dtype=np.float32) * brightness

        embeddings = base_features + noise
        return Features(
            embeddings=embeddings,
            metadata={"feature_dim": self.feature_dim},
        )


def create_synthetic_templates(
    object_id: int, num_templates: int = 10, seed: int = 42
) -> list[Frame]:
    """
    Create synthetic template images for testing.

    Args:
        object_id: Object ID (affects template appearance)
        num_templates: Number of templates to generate
        seed: Random seed for reproducibility

    Returns:
        List of synthetic template frames
    """
    rng = np.random.RandomState(seed + object_id)
    templates = []
    for i in range(num_templates):
        # Create synthetic image with object-specific pattern
        brightness = 100 + object_id * 30 + i * 5
        rgb = np.full((480, 640, 3), brightness, dtype=np.uint8)

        # Add some variation
        noise = rng.randint(-20, 20, (480, 640, 3))
        rgb = np.clip(rgb.astype(int) + noise, 0, 255).astype(np.uint8)

        templates.append(Frame(rgb=rgb, metadata={"template_id": i}))

    return templates


def create_synthetic_query(
    target_object_id: int, num_proposals: int = 5, seed: int = 42
) -> tuple[np.ndarray, Masks]:
    """
    Create synthetic query image with proposals.

    Args:
        target_object_id: Object ID to simulate in the query
        num_proposals: Number of proposals to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (query_rgb, query_masks)
    """
    rng = np.random.RandomState(seed + target_object_id)

    # Create query image similar to target object templates
    brightness = 100 + target_object_id * 30
    query_rgb = np.full((480, 640, 3), brightness, dtype=np.uint8)
    noise = rng.randint(-30, 30, (480, 640, 3))
    query_rgb = np.clip(query_rgb.astype(int) + noise, 0, 255).astype(np.uint8)

    # Create synthetic masks (proposals)
    masks = []
    boxes = []
    for i in range(num_proposals):
        # Create random rectangular mask
        x1 = rng.randint(0, 500)
        y1 = rng.randint(0, 380)
        w = rng.randint(50, 140)
        h = rng.randint(50, 100)
        x2 = min(x1 + w, 640)
        y2 = min(y1 + h, 480)

        mask = np.zeros((480, 640), dtype=bool)
        mask[y1:y2, x1:x2] = True
        masks.append(mask)
        boxes.append([x1, y1, x2, y2])

    query_masks = Masks(
        masks=np.array(masks),
        boxes=np.array(boxes, dtype=np.float32),
        scores=np.ones(num_proposals, dtype=np.float32),
        labels=np.zeros(num_proposals, dtype=np.int32),
    )

    return query_rgb, query_masks


def test_cnos_pipeline_synthetic():
    """Test complete CNOS pipeline with synthetic data."""
    logger.info("=" * 60)
    logger.info("Testing CNOS Pipeline with Synthetic Data")
    logger.info("=" * 60)

    # Configuration
    num_objects = 3
    templates_per_object = 10
    target_object_id = 2  # Which object to simulate in query
    feature_dim = 384

    # Step 1: Create synthetic templates for multiple objects
    logger.info(f"\n--- Creating templates for {num_objects} objects ---")
    all_template_descriptors = {}

    feature_extractor = SyntheticFeatureExtractor(feature_dim=feature_dim)
    descriptor_node = TemplateDescriptorNode(
        feature_extractor=feature_extractor,
        device="cpu",
    )

    for obj_id in range(1, num_objects + 1):
        logger.info(f"Creating templates for object {obj_id}")
        templates = create_synthetic_templates(obj_id, templates_per_object)

        # Extract descriptors
        template_descriptors = descriptor_node.process_templates(
            iter(templates), object_id=obj_id
        )
        all_template_descriptors[obj_id] = template_descriptors

        logger.info(
            f"  Object {obj_id}: {len(template_descriptors)} templates, "
            f"descriptor shape: {template_descriptors.descriptors.shape}"
        )

    # Step 2: Combine all descriptors
    logger.info("\n--- Combining template descriptors ---")
    all_descriptors = torch.cat(
        [desc.descriptors for desc in all_template_descriptors.values()], dim=0
    )
    all_object_ids = torch.cat(
        [desc.object_ids for desc in all_template_descriptors.values()], dim=0
    )
    all_template_ids = torch.cat(
        [desc.template_ids for desc in all_template_descriptors.values()], dim=0
    )

    combined_descriptors = TemplateDescriptor(
        descriptors=all_descriptors,
        object_ids=all_object_ids,
        template_ids=all_template_ids,
    )

    logger.info(f"Combined: {len(combined_descriptors)} total templates")

    # Step 3: Create synthetic query
    logger.info(f"\n--- Creating query image (target object: {target_object_id}) ---")
    query_rgb, query_masks = create_synthetic_query(
        target_object_id, num_proposals=5
    )
    logger.info(f"Query image shape: {query_rgb.shape}")
    logger.info(f"Number of proposals: {len(query_masks.masks)}")

    # Step 4: Extract features from proposals
    logger.info("\n--- Extracting features from proposals ---")
    proposal_features = []
    for i, mask in enumerate(query_masks.masks):
        # Create masked RGB
        masked_rgb = query_rgb.copy()
        masked_rgb[~mask] = 0

        # Extract features
        proposal_frame = Frame(rgb=masked_rgb)
        features = feature_extractor.process(proposal_frame)
        proposal_features.append(features.embeddings)

    proposal_descriptors = np.stack(proposal_features)
    query_features = Features(
        embeddings=proposal_descriptors,
        metadata={"num_proposals": len(query_masks.masks)},
    )

    logger.info(
        f"Extracted {len(proposal_descriptors)} proposal descriptors, "
        f"shape: {proposal_descriptors.shape}"
    )

    # Step 5: Match proposals to templates
    logger.info("\n--- Matching proposals to templates ---")
    matcher = CNOSMatcherNode(
        template_descriptors=combined_descriptors,
        confidence_threshold=0.3,  # Lower threshold for synthetic data
        max_instances=10,
        aggregation="avg_5",
        device="cpu",
    )

    matches = matcher.process((query_features, query_masks))

    # Step 6: Analyze results
    logger.info("\n" + "=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"Target object ID: {target_object_id}")
    logger.info(f"Number of matches: {len(matches.labels)}")

    if len(matches.labels) > 0:
        logger.info("\nMatched objects:")
        for obj_id, score in zip(matches.labels, matches.scores):
            correct = "✓" if obj_id == target_object_id else "✗"
            logger.info(f"  {correct} Object {obj_id}: score={score:.3f}")

        # Count correct matches
        correct_matches = sum(
            1 for obj_id in matches.labels if obj_id == target_object_id
        )
        accuracy = correct_matches / len(matches.labels) * 100

        logger.info(f"\nAccuracy: {accuracy:.1f}% ({correct_matches}/{len(matches.labels)})")

        # With synthetic data designed to match, we should get high accuracy
        if accuracy >= 50:
            logger.info("✓ Pipeline working correctly!")
        else:
            logger.warning("⚠ Lower accuracy than expected with synthetic data")
    else:
        logger.info("No matches found (confidence threshold may be too high)")

    logger.info("=" * 60)

    return matches


def test_descriptor_caching():
    """Test descriptor save/load functionality."""
    import tempfile

    logger.info("\n--- Testing Descriptor Caching ---")

    # Create synthetic templates
    templates = create_synthetic_templates(object_id=1, num_templates=5)

    # Extract descriptors
    feature_extractor = SyntheticFeatureExtractor(feature_dim=384)
    descriptor_node = TemplateDescriptorNode(
        feature_extractor=feature_extractor,
        device="cpu",
    )

    original_descriptors = descriptor_node.process_templates(iter(templates), object_id=1)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        cache_path = tmp.name

    try:
        original_descriptors.save(cache_path)
        logger.info(f"Saved descriptors to {cache_path}")

        # Load from file
        loaded_descriptors = TemplateDescriptor.load(cache_path)
        logger.info(f"Loaded descriptors from {cache_path}")

        # Verify they match
        assert torch.allclose(
            original_descriptors.descriptors, loaded_descriptors.descriptors
        )
        assert torch.equal(original_descriptors.object_ids, loaded_descriptors.object_ids)

        logger.info("✓ Caching test passed!")
    finally:
        # Clean up temporary file
        import os

        if os.path.exists(cache_path):
            os.unlink(cache_path)


if __name__ == "__main__":
    # Test pipeline
    test_cnos_pipeline_synthetic()

    # Test caching
    test_descriptor_caching()

    logger.info("\n✓ All tests completed successfully!")
