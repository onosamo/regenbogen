"""
Tests for SAM2 node.
"""

import numpy as np
import pytest

from regenbogen.interfaces import Frame, Masks
from regenbogen.nodes.sam2 import SAM2Node


def test_masks_interface():
    """Test Masks interface creation and validation."""
    masks = np.random.rand(3, 480, 640) > 0.5  # 3 random masks
    boxes = np.array([[10, 20, 100, 150], [200, 300, 350, 400], [50, 60, 120, 180]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.75], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int32)

    masks_obj = Masks(
        masks=masks,
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=["object", "thing"],
    )

    assert masks_obj.masks.shape == (3, 480, 640)
    assert masks_obj.boxes.shape == (3, 4)
    assert masks_obj.scores.shape == (3,)
    assert masks_obj.labels.shape == (3,)
    assert len(masks_obj.class_names) == 2
    assert isinstance(masks_obj.metadata, dict)


def test_sam2_node_initialization():
    """Test SAM2Node initialization with different model sizes."""
    # Test valid model sizes
    for model_size in ["tiny", "small", "base-plus", "large"]:
        node = SAM2Node(model_size=model_size)
        assert node.model_size == model_size

    # Test invalid model size
    with pytest.raises(ValueError):
        SAM2Node(model_size="invalid")


def test_sam2_masks_to_boxes():
    """Test conversion of masks to bounding boxes."""
    node = SAM2Node(model_size="tiny")

    # Create test masks
    masks = np.zeros((2, 100, 100), dtype=bool)
    # First mask: box at (10, 10) to (30, 40)
    masks[0, 10:40, 10:30] = True
    # Second mask: box at (50, 50) to (80, 90)
    masks[1, 50:90, 50:80] = True

    boxes = node._masks_to_boxes(masks)

    assert boxes.shape == (2, 4)
    # Check first box
    assert boxes[0, 0] == 10  # x_min
    assert boxes[0, 1] == 10  # y_min
    assert boxes[0, 2] == 29  # x_max
    assert boxes[0, 3] == 39  # y_max

    # Check second box
    assert boxes[1, 0] == 50
    assert boxes[1, 1] == 50
    assert boxes[1, 2] == 79
    assert boxes[1, 3] == 89


def test_sam2_empty_mask():
    """Test handling of empty masks."""
    node = SAM2Node(model_size="tiny")

    # Create empty mask
    masks = np.zeros((1, 100, 100), dtype=bool)
    boxes = node._masks_to_boxes(masks)

    assert boxes.shape == (1, 4)
    # Empty mask should produce dummy box
    assert np.allclose(boxes[0], [0, 0, 0, 0])


def test_sam2_node_process():
    """Test SAM2Node processing (requires model download)."""
    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    node = SAM2Node(model_size="tiny", device="cpu")
    result = node.process(frame)

    assert isinstance(result, Masks)
    assert result.masks is not None
    assert result.boxes is not None
    assert result.scores is not None
    assert len(result.masks) == len(result.boxes)
    assert len(result.boxes) == len(result.scores)


def test_sam2_node_invalid_frame():
    """Test SAM2Node with invalid input."""
    node = SAM2Node(model_size="tiny")

    # Frame without RGB
    frame = Frame(rgb=None, depth=np.zeros((100, 100)))

    with pytest.raises(ValueError, match="Frame must contain an RGB image"):
        node.process(frame)


if __name__ == "__main__":
    # Run basic tests
    print("Running Masks interface tests...")

    test_masks_interface()
    print("✓ Masks interface test passed")

    print("\nRunning SAM2 tests...")

    test_sam2_node_initialization()
    print("✓ SAM2Node initialization test passed")

    test_sam2_masks_to_boxes()
    print("✓ Masks to boxes conversion test passed")

    test_sam2_empty_mask()
    print("✓ Empty mask handling test passed")

    test_sam2_node_invalid_frame()
    print("✓ Invalid frame test passed")

    print("\n🎉 All SAM2 tests passed successfully!")
