"""Test depth to pointcloud with class filtering."""

import numpy as np
import pytest

from regenbogen.interfaces import Frame, Masks
from regenbogen.nodes import DepthToPointCloudNode


def test_depth_to_pointcloud_include_classes():
    """Test depth to pointcloud with include_classes filtering."""
    # Create test data
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5

    # Create masks for two objects
    masks_array = np.zeros((2, 100, 100), dtype=bool)
    masks_array[0, 20:40, 20:40] = True  # bottle region
    masks_array[1, 60:80, 60:80] = True  # cup region

    masks = Masks(
        masks=masks_array,
        boxes=np.array([[20, 20, 40, 40], [60, 60, 80, 80]], dtype=np.float32),
        scores=np.array([0.9, 0.85], dtype=np.float32),
        class_names=["bottle", "cup"],
    )

    frame = Frame(rgb=rgb, depth=depth)
    frame.masks = masks

    # Test with include_classes - should only include bottle
    node = DepthToPointCloudNode(include_classes=["bottle"])
    result = node.process(frame)

    assert result.pointcloud is not None
    assert len(result.pointcloud.points) > 0
    # Should have points only from bottle region (roughly 20x20 = 400 points)
    # Allow some tolerance due to depth filtering
    assert 200 < len(result.pointcloud.points) < 600


def test_depth_to_pointcloud_exclude_classes():
    """Test depth to pointcloud with exclude_classes filtering."""
    # Create test data
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5

    # Create masks for two objects
    masks_array = np.zeros((2, 100, 100), dtype=bool)
    masks_array[0, 20:40, 20:40] = True  # bottle region
    masks_array[1, 60:80, 60:80] = True  # cup region

    masks = Masks(
        masks=masks_array,
        boxes=np.array([[20, 20, 40, 40], [60, 60, 80, 80]], dtype=np.float32),
        scores=np.array([0.9, 0.85], dtype=np.float32),
        class_names=["bottle", "cup"],
    )

    frame = Frame(rgb=rgb, depth=depth)
    frame.masks = masks

    # Test with exclude_classes - should exclude cup
    node = DepthToPointCloudNode(exclude_classes=["cup"])
    result = node.process(frame)

    assert result.pointcloud is not None
    # Should have points from all regions except cup (roughly 10000 - 400 = 9600 points)
    # Allow tolerance
    assert len(result.pointcloud.points) > 8000


def test_depth_to_pointcloud_no_filtering():
    """Test depth to pointcloud without class filtering."""
    # Create test data
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5

    frame = Frame(rgb=rgb, depth=depth)

    # Test without filtering - should include all points
    node = DepthToPointCloudNode()
    result = node.process(frame)

    assert result.pointcloud is not None
    # Should have roughly 10000 points (100x100)
    assert 9000 < len(result.pointcloud.points) < 11000


def test_depth_to_pointcloud_requires_masks_for_filtering():
    """Test that filtering requires masks."""
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5
    frame = Frame(rgb=rgb, depth=depth)

    # Should raise error when trying to filter without masks
    node = DepthToPointCloudNode(include_classes=["bottle"])
    with pytest.raises(ValueError, match="filter by classes"):
        node.process(frame)


def test_depth_to_pointcloud_requires_class_names():
    """Test that filtering requires class_names in masks."""
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.ones((100, 100), dtype=np.float32) * 1.5

    # Create masks without class_names (empty list after __post_init__)
    masks_array = np.zeros((1, 100, 100), dtype=bool)
    masks_array[0, 20:40, 20:40] = True

    masks = Masks(
        masks=masks_array,
        boxes=np.array([[20, 20, 40, 40]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
    )
    # Manually set to None after init to bypass __post_init__
    masks.class_names = None

    frame = Frame(rgb=rgb, depth=depth)
    frame.masks = masks

    # Should raise error when trying to filter without class_names
    node = DepthToPointCloudNode(include_classes=["bottle"])
    with pytest.raises(ValueError, match="class_names"):
        node.process(frame)
