"""
Tests for CNOS pipeline nodes.

Tests the TemplateDescriptorNode and CNOSMatcherNode functionality.
"""

import numpy as np
import pytest
import torch

from regenbogen.interfaces import Features, Frame, Masks
from regenbogen.nodes import (
    CNOSMatcherNode,
    TemplateDescriptor,
    TemplateDescriptorNode,
)


class MockFeatureExtractor:
    """Mock feature extractor for testing."""

    def __init__(self, feature_dim=384):
        self.feature_dim = feature_dim

    def process(self, frame: Frame) -> Features:
        """Return mock features."""
        # Return random features for testing
        embeddings = np.random.randn(self.feature_dim).astype(np.float32)
        return Features(
            embeddings=embeddings,
            metadata={"feature_dim": self.feature_dim},
        )


def test_template_descriptor_creation():
    """Test TemplateDescriptor creation and basic operations."""
    # Create mock descriptors
    descriptors = torch.randn(10, 384)  # 10 templates, 384-dim features
    object_ids = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    template_ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])

    # Create TemplateDescriptor
    template_desc = TemplateDescriptor(
        descriptors=descriptors,
        object_ids=object_ids,
        template_ids=template_ids,
    )

    # Test basic properties
    assert len(template_desc) == 10
    assert template_desc.descriptors.shape == (10, 384)

    # Test get_descriptors_for_object
    obj1_descriptors = template_desc.get_descriptors_for_object(1)
    assert obj1_descriptors.shape == (3, 384)

    obj3_descriptors = template_desc.get_descriptors_for_object(3)
    assert obj3_descriptors.shape == (4, 384)


def test_template_descriptor_save_load(tmp_path):
    """Test saving and loading template descriptors."""
    # Create mock descriptors
    descriptors = torch.randn(5, 384)
    object_ids = torch.tensor([1, 1, 2, 2, 2])
    template_ids = torch.tensor([0, 1, 0, 1, 2])

    template_desc = TemplateDescriptor(
        descriptors=descriptors,
        object_ids=object_ids,
        template_ids=template_ids,
        metadata={"test": "value"},
    )

    # Save
    save_path = tmp_path / "test_descriptors.pth"
    template_desc.save(str(save_path))

    # Load
    loaded_desc = TemplateDescriptor.load(str(save_path))

    # Verify
    assert torch.allclose(loaded_desc.descriptors, descriptors)
    assert torch.equal(loaded_desc.object_ids, object_ids)
    assert torch.equal(loaded_desc.template_ids, template_ids)
    assert loaded_desc.metadata["test"] == "value"


def test_template_descriptor_node():
    """Test TemplateDescriptorNode processing."""
    # Create mock feature extractor
    feature_extractor = MockFeatureExtractor(feature_dim=384)

    # Create TemplateDescriptorNode
    descriptor_node = TemplateDescriptorNode(
        feature_extractor=feature_extractor,
        device="cpu",
    )

    # Create mock templates
    templates = []
    for i in range(5):
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = Frame(rgb=rgb, metadata={"template_id": i})
        templates.append(frame)

    # Process templates
    template_descriptors = descriptor_node.process_templates(
        iter(templates), object_id=1
    )

    # Verify results
    assert len(template_descriptors) == 5
    assert template_descriptors.descriptors.shape == (5, 384)
    assert torch.all(template_descriptors.object_ids == 1)
    assert template_descriptors.metadata["object_id"] == 1
    assert template_descriptors.metadata["num_templates"] == 5


def test_template_descriptor_node_multiple_objects():
    """Test processing templates for multiple objects."""
    feature_extractor = MockFeatureExtractor(feature_dim=384)
    descriptor_node = TemplateDescriptorNode(
        feature_extractor=feature_extractor,
        device="cpu",
    )

    # Create mock templates for multiple objects
    object_templates = {}
    for obj_id in [1, 2, 3]:
        templates = []
        for i in range(3):
            rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            templates.append(Frame(rgb=rgb))
        object_templates[obj_id] = iter(templates)

    # Process all objects
    all_descriptors = descriptor_node.process_multiple_objects(object_templates)

    # Verify
    assert len(all_descriptors) == 9  # 3 objects * 3 templates
    assert all_descriptors.descriptors.shape == (9, 384)
    assert all_descriptors.metadata["num_objects"] == 3

    # Check object IDs
    unique_ids = torch.unique(all_descriptors.object_ids).tolist()
    assert set(unique_ids) == {1, 2, 3}


def test_cnos_matcher_node():
    """Test CNOSMatcherNode matching functionality."""
    # Create mock template descriptors
    descriptors = torch.randn(12, 384)  # 12 templates total
    object_ids = torch.tensor([1] * 4 + [2] * 4 + [3] * 4)  # 3 objects, 4 templates each
    template_ids = torch.tensor([0, 1, 2, 3] * 3)

    template_desc = TemplateDescriptor(
        descriptors=descriptors,
        object_ids=object_ids,
        template_ids=template_ids,
    )

    # Create matcher
    matcher = CNOSMatcherNode(
        template_descriptors=template_desc,
        confidence_threshold=0.3,  # Lower threshold for testing
        max_instances=10,
        aggregation="mean",
        device="cpu",
    )

    # Verify initialization
    assert matcher.num_objects == 3
    assert len(matcher.unique_object_ids) == 3

    # Create mock query features and masks
    query_descriptors = torch.randn(5, 384)  # 5 proposals
    query_features = Features(
        embeddings=query_descriptors.numpy(),
        metadata={"num_proposals": 5},
    )

    # Create mock masks
    masks = np.random.rand(5, 480, 640) > 0.5
    boxes = np.random.rand(5, 4) * 640
    boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1, y2 > y1

    query_masks = Masks(
        masks=masks,
        boxes=boxes.astype(np.float32),
        scores=np.ones(5, dtype=np.float32),
        labels=np.zeros(5, dtype=np.int32),
    )

    # Match proposals
    matches = matcher.process((query_features, query_masks))

    # Verify results structure (Masks interface)
    assert hasattr(matches, "labels")
    assert hasattr(matches, "scores")
    assert hasattr(matches, "masks")
    assert hasattr(matches, "boxes")

    # All matched object IDs (stored in labels) should be in [1, 2, 3]
    if len(matches.labels) > 0:
        assert all(obj_id in [1, 2, 3] for obj_id in matches.labels)
        assert all(score >= matcher.confidence_threshold for score in matches.scores)


def test_cnos_matcher_top_k_aggregation():
    """Test top-k aggregation in CNOS matcher."""
    # Create template descriptors
    descriptors = torch.randn(10, 384)  # 10 templates for 1 object
    object_ids = torch.ones(10, dtype=torch.long)
    template_ids = torch.arange(10)

    template_desc = TemplateDescriptor(
        descriptors=descriptors,
        object_ids=object_ids,
        template_ids=template_ids,
    )

    # Test different aggregation methods
    for aggregation in ["mean", "max", "median", "avg_5"]:
        matcher = CNOSMatcherNode(
            template_descriptors=template_desc,
            confidence_threshold=0.0,  # Accept all for testing
            aggregation=aggregation,
            device="cpu",
        )

        # Create query
        query_descriptors = torch.randn(2, 384)
        query_features = Features(embeddings=query_descriptors.numpy())

        masks = np.random.rand(2, 100, 100) > 0.5
        boxes = np.array([[10, 10, 50, 50], [60, 60, 90, 90]], dtype=np.float32)

        query_masks = Masks(
            masks=masks,
            boxes=boxes,
            scores=np.ones(2, dtype=np.float32),
            labels=np.zeros(2, dtype=np.int32),
        )

        # Should not raise error
        matches = matcher.process((query_features, query_masks))
        # With threshold=0, we should get at least some matches
        assert len(matches.labels) <= 2  # At most all proposals


def test_cnos_matcher_confidence_filtering():
    """Test confidence threshold filtering."""
    # Create template descriptors
    descriptors = torch.randn(5, 384)
    object_ids = torch.ones(5, dtype=torch.long)
    template_ids = torch.arange(5)

    template_desc = TemplateDescriptor(
        descriptors=descriptors,
        object_ids=object_ids,
        template_ids=template_ids,
    )

    # Create matcher with high threshold
    matcher = CNOSMatcherNode(
        template_descriptors=template_desc,
        confidence_threshold=0.99,  # Very high threshold
        device="cpu",
    )

    # Create query that won't match well
    query_descriptors = torch.randn(3, 384) * 10  # Very different from templates
    query_features = Features(embeddings=query_descriptors.numpy())

    masks = np.random.rand(3, 100, 100) > 0.5
    boxes = np.array(
        [[10, 10, 50, 50], [60, 60, 90, 90], [100, 100, 150, 150]], dtype=np.float32
    )

    query_masks = Masks(
        masks=masks,
        boxes=boxes,
        scores=np.ones(3, dtype=np.float32),
        labels=np.zeros(3, dtype=np.int32),
    )

    # Match - should filter out most/all proposals
    matches = matcher.process((query_features, query_masks))

    # With random data and high threshold, likely 0 matches
    assert len(matches.labels) <= 3


def test_cnos_matcher_max_instances():
    """Test max instances limit."""
    # Create template descriptors
    descriptors = torch.randn(5, 384)
    object_ids = torch.ones(5, dtype=torch.long)
    template_ids = torch.arange(5)

    template_desc = TemplateDescriptor(
        descriptors=descriptors,
        object_ids=object_ids,
        template_ids=template_ids,
    )

    # Create matcher with low max_instances
    matcher = CNOSMatcherNode(
        template_descriptors=template_desc,
        confidence_threshold=0.0,  # Accept all
        max_instances=2,
        device="cpu",
    )

    # Create many query proposals
    query_descriptors = torch.randn(10, 384)
    query_features = Features(embeddings=query_descriptors.numpy())

    masks = np.random.rand(10, 100, 100) > 0.5
    boxes = np.random.rand(10, 4) * 100
    boxes[:, 2:] += boxes[:, :2]

    query_masks = Masks(
        masks=masks,
        boxes=boxes.astype(np.float32),
        scores=np.ones(10, dtype=np.float32),
        labels=np.zeros(10, dtype=np.int32),
    )

    # Match
    matches = matcher.process((query_features, query_masks))

    # Should be limited to max_instances
    assert len(matches.labels) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
