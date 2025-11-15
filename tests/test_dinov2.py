"""
Tests for Dinov2 node.
"""

import numpy as np
import pytest

from regenbogen.interfaces import Features, Frame
from regenbogen.nodes.dinov2 import Dinov2Node


def test_features_interface():
    """Test Features interface creation and validation."""
    descriptors = np.random.randn(196, 384).astype(np.float32)  # 196 patches, 384 dims
    embeddings = np.random.randn(384).astype(np.float32)
    keypoints = np.array([[10, 20], [30, 40]], dtype=np.float32)

    features = Features(
        descriptors=descriptors,
        embeddings=embeddings,
        keypoints=keypoints,
    )

    assert features.descriptors.shape == (196, 384)
    assert features.embeddings.shape == (384,)
    assert features.keypoints.shape == (2, 2)
    assert isinstance(features.metadata, dict)


def test_dinov2_node_initialization():
    """Test Dinov2Node initialization with different model sizes."""
    # Test valid model sizes
    for model_size in ["small", "base", "large", "giant"]:
        node = Dinov2Node(
            model_size=model_size, enable_rerun_logging=False, device="cpu"
        )
        assert node.model_size == model_size

    # Test invalid model size
    with pytest.raises(ValueError, match="Invalid model_size"):
        Dinov2Node(model_size="invalid", enable_rerun_logging=False)


def test_dinov2_output_types():
    """Test Dinov2Node initialization with different output types."""
    # Test valid output types
    for output_type in ["patch", "cls", "both"]:
        node = Dinov2Node(
            model_size="small",
            output_type=output_type,
            enable_rerun_logging=False,
            device="cpu",
        )
        assert node.output_type == output_type

    # Test invalid output type
    with pytest.raises(ValueError, match="Invalid output_type"):
        Dinov2Node(
            model_size="small", output_type="invalid", enable_rerun_logging=False
        )


def test_dinov2_node_process_patch():
    """Test Dinov2Node processing with patch features."""
    rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    node = Dinov2Node(
        model_size="small", output_type="patch", enable_rerun_logging=False, device="cpu"
    )
    result = node.process(frame)

    assert isinstance(result, Features)
    assert result.descriptors is not None
    assert len(result.descriptors.shape) == 2  # (num_patches, feature_dim)
    assert result.descriptors.shape[0] > 0  # Has patches
    assert result.descriptors.shape[1] > 0  # Has features
    assert result.embeddings is None  # No CLS token in "patch" mode
    assert "model" in result.metadata
    assert "output_type" in result.metadata
    assert result.metadata["output_type"] == "patch"


def test_dinov2_node_process_cls():
    """Test Dinov2Node processing with CLS token."""
    rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    node = Dinov2Node(
        model_size="small", output_type="cls", enable_rerun_logging=False, device="cpu"
    )
    result = node.process(frame)

    assert isinstance(result, Features)
    assert result.descriptors is not None
    assert len(result.descriptors.shape) == 1  # (feature_dim,) - single vector
    assert result.descriptors.shape[0] > 0  # Has features
    assert result.embeddings is None  # No separate embeddings in "cls" mode
    assert result.metadata["output_type"] == "cls"


def test_dinov2_node_process_both():
    """Test Dinov2Node processing with both patch and CLS features."""
    rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    node = Dinov2Node(
        model_size="small", output_type="both", enable_rerun_logging=False, device="cpu"
    )
    result = node.process(frame)

    assert isinstance(result, Features)
    assert result.descriptors is not None
    assert len(result.descriptors.shape) == 2  # Patch features
    assert result.embeddings is not None  # CLS token
    assert len(result.embeddings.shape) == 1  # Single vector
    assert result.metadata["output_type"] == "both"


def test_dinov2_node_invalid_frame():
    """Test Dinov2Node with invalid input."""
    node = Dinov2Node(
        model_size="small", enable_rerun_logging=False, device="cpu"
    )

    # Frame without RGB
    frame = Frame(rgb=None, depth=np.zeros((100, 100)))

    with pytest.raises(ValueError, match="Frame must contain an RGB image"):
        node.process(frame)


def test_dinov2_node_metadata():
    """Test that Dinov2Node includes proper metadata."""
    rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    node = Dinov2Node(
        model_size="small", output_type="patch", enable_rerun_logging=False, device="cpu"
    )
    result = node.process(frame)

    # Check metadata fields
    assert "model" in result.metadata
    assert "output_type" in result.metadata
    assert "descriptor_shape" in result.metadata
    assert "feature_dim" in result.metadata
    assert "num_patches" in result.metadata
    assert "num_patches_per_side" in result.metadata
    assert "patch_size" in result.metadata
    assert "input_size" in result.metadata

    assert result.metadata["model"] == "dinov2-small"
    assert result.metadata["patch_size"] == 14


def test_dinov2_descriptor_consistency():
    """Test that Dinov2Node produces consistent features for same input."""
    rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    node = Dinov2Node(
        model_size="small", output_type="patch", enable_rerun_logging=False, device="cpu"
    )

    result1 = node.process(frame)
    result2 = node.process(frame)

    # Features should be identical for same input
    assert np.allclose(result1.descriptors, result2.descriptors, rtol=1e-5, atol=1e-5)


def test_dinov2_different_image_sizes():
    """Test Dinov2Node with different input image sizes."""
    node = Dinov2Node(
        model_size="small", output_type="patch", enable_rerun_logging=False, device="cpu"
    )

    # Test various image sizes
    for size in [(224, 224), (480, 640), (1080, 1920)]:
        rgb = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        frame = Frame(rgb=rgb)
        result = node.process(frame)

        assert isinstance(result, Features)
        assert result.descriptors is not None
        # All should produce same number of patches (model resizes internally)
        assert result.descriptors.shape[0] > 0


def test_dinov2_normalized_input():
    """Test Dinov2Node with normalized float input."""
    rgb_normalized = np.random.rand(224, 224, 3).astype(np.float32)
    frame = Frame(rgb=rgb_normalized)

    node = Dinov2Node(
        model_size="small", output_type="patch", enable_rerun_logging=False, device="cpu"
    )
    result = node.process(frame)

    assert isinstance(result, Features)
    assert result.descriptors is not None


if __name__ == "__main__":
    # Run basic tests
    print("Running Features interface tests...")
    test_features_interface()
    print("✓ Features interface test passed")

    print("\nRunning Dinov2 initialization tests...")
    test_dinov2_node_initialization()
    print("✓ Dinov2Node initialization test passed")

    test_dinov2_output_types()
    print("✓ Dinov2Node output types test passed")

    print("\nRunning Dinov2 processing tests...")
    test_dinov2_node_process_patch()
    print("✓ Dinov2Node patch processing test passed")

    test_dinov2_node_process_cls()
    print("✓ Dinov2Node CLS processing test passed")

    test_dinov2_node_process_both()
    print("✓ Dinov2Node both processing test passed")

    test_dinov2_node_invalid_frame()
    print("✓ Dinov2Node invalid frame test passed")

    test_dinov2_node_metadata()
    print("✓ Dinov2Node metadata test passed")

    test_dinov2_descriptor_consistency()
    print("✓ Dinov2Node consistency test passed")

    test_dinov2_different_image_sizes()
    print("✓ Dinov2Node different sizes test passed")

    test_dinov2_normalized_input()
    print("✓ Dinov2Node normalized input test passed")

    print("\n🎉 All Dinov2 tests passed successfully!")
