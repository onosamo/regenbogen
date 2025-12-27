"""Tests for SAM3 node."""

import numpy as np
import pytest
import traceback

from regenbogen.interfaces import Frame

# Store import error for detailed error messages
_sam3_import_error = None


def _has_sam3():
    """Check if SAM3 is available."""
    global _sam3_import_error
    try:
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
        from sam3.model_builder import build_sam3_image_model  # noqa: F401

        return True
    except ImportError as e:
        _sam3_import_error = f"{e}\n{''.join(traceback.format_tb(e.__traceback__))}"
        return False


@pytest.mark.skipif(
    not _has_sam3(),
    reason=f"SAM3 dependencies not installed: {_sam3_import_error if _sam3_import_error else 'Unknown import error'}",
)
def test_sam3_import():
    """Test that SAM3Node can be imported."""
    from regenbogen.nodes import SAM3Node

    assert SAM3Node is not None


@pytest.mark.skipif(
    not _has_sam3(),
    reason=f"SAM3 dependencies not installed: {_sam3_import_error if _sam3_import_error else 'Unknown import error'}",
)
def test_sam3_initialization():
    """Test SAM3 node initialization."""
    from regenbogen.nodes import SAM3Node

    node = SAM3Node(
        device="cpu",
        text_prompt="an object",
        name="TestSAM3",
    )

    assert node.name == "TestSAM3"
    assert node.text_prompt == "an object"
    assert node.mask_threshold == 0.0


@pytest.mark.skipif(
    not _has_sam3(),
    reason=f"SAM3 dependencies not installed: {_sam3_import_error if _sam3_import_error else 'Unknown import error'}",
)
def test_sam3_text_prompt_processing():
    """Test SAM3 processing with text prompt."""
    from regenbogen.nodes import SAM3Node

    # Create a simple test image
    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    # Create SAM3 node with text prompt
    node = SAM3Node(
        device="cpu",
        text_prompt="an object",
    )

    # Process frame
    output_frame = node.process(frame)

    # Check output structure - should return frame with masks attached
    assert isinstance(output_frame, Frame)
    assert output_frame.masks is not None
    assert hasattr(output_frame.masks, "masks")
    assert hasattr(output_frame.masks, "boxes")
    assert hasattr(output_frame.masks, "scores")
    assert output_frame.masks.metadata["model"] == "sam3"
    assert output_frame.masks.metadata["text_prompt"] == "an object"


@pytest.mark.skipif(
    not _has_sam3(),
    reason=f"SAM3 dependencies not installed: {_sam3_import_error if _sam3_import_error else 'Unknown import error'}",
)
def test_sam3_prompt_override():
    """Test SAM3 processing with prompt override."""
    from regenbogen.nodes import SAM3Node

    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    # Create SAM3 node with default prompt
    node = SAM3Node(
        device="cpu",
        text_prompt="default prompt",
    )

    # Process with overridden prompt
    output_frame = node.process(frame, text_prompt="overridden prompt")

    assert output_frame.masks.metadata["text_prompt"] == "overridden prompt"


@pytest.mark.skipif(
    not _has_sam3(),
    reason=f"SAM3 dependencies not installed: {_sam3_import_error if _sam3_import_error else 'Unknown import error'}",
)
def test_sam3_point_prompts():
    """Test SAM3 processing with point prompts."""
    from regenbogen.nodes import SAM3Node

    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    # Create SAM3 node
    node = SAM3Node(device="cpu")

    # Process with point prompts
    points = np.array([[320, 240], [100, 100]])  # Two points
    output_frame = node.process(frame, point_prompts=points)

    assert output_frame.masks.metadata["has_point_prompts"] is True
    assert output_frame.masks.metadata["has_box_prompts"] is False


@pytest.mark.skipif(
    not _has_sam3(),
    reason=f"SAM3 dependencies not installed: {_sam3_import_error if _sam3_import_error else 'Unknown import error'}",
)
def test_sam3_box_prompts():
    """Test SAM3 processing with box prompts."""
    from regenbogen.nodes import SAM3Node

    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    # Create SAM3 node
    node = SAM3Node(device="cpu")

    # Process with box prompts
    boxes = np.array([[50, 50, 200, 200], [300, 300, 500, 450]])  # Two boxes
    output_frame = node.process(frame, box_prompts=boxes)

    assert output_frame.masks.metadata["has_box_prompts"] is True
    assert output_frame.masks.metadata["has_point_prompts"] is False


@pytest.mark.skipif(
    not _has_sam3(),
    reason=f"SAM3 dependencies not installed: {_sam3_import_error if _sam3_import_error else 'Unknown import error'}",
)
def test_sam3_tuple_input():
    """Test SAM3 processing with tuple input from dataset loader."""
    from regenbogen.nodes import SAM3Node

    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    # Simulate tuple input from BOP dataset loader
    input_tuple = (frame, [], [])  # (Frame, poses, obj_ids)

    node = SAM3Node(device="cpu", text_prompt="test")
    output = node.process(input_tuple)

    # Should return tuple with frame as first element
    assert isinstance(output, tuple)
    assert len(output) == 3
    output_frame = output[0]
    assert isinstance(output_frame, Frame)
    assert output_frame.masks is not None
    assert output_frame.masks.metadata["model"] == "sam3"
