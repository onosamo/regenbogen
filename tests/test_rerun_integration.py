"""
Test Rerun logging functionality.
"""

import numpy as np
import pytest

from regenbogen import Pipeline
from regenbogen.interfaces import Frame
from regenbogen.nodes import DepthToPointCloudNode
from regenbogen.utils import RerunLogger


def test_rerun_logger_creation():
    """Test that RerunLogger can be created."""
    logger = RerunLogger("test_recording", enabled=True)
    assert logger.recording_name == "test_recording"
    assert logger.enabled is True


def test_rerun_logger_disabled():
    """Test that RerunLogger can be disabled."""
    logger = RerunLogger("test_recording", enabled=False)
    assert logger.enabled is False


def test_pipeline_with_rerun_logging():
    """Test pipeline with Rerun logging enabled."""
    # Create a simple pipeline with Rerun logging (no viewer spawn for testing)
    pipeline = Pipeline(
        name="Test_Pipeline", enable_rerun_logging=True, rerun_spawn_viewer=False
    )

    # Add a simple node
    depth_to_pc_node = DepthToPointCloudNode(name="DepthToPointCloud")
    pipeline.add_node(depth_to_pc_node)

    # Create test data
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.random.rand(100, 100).astype(np.float32)
    intrinsics = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float64)

    frame = Frame(rgb=rgb, depth=depth, intrinsics=intrinsics)

    # Process through pipeline (should not raise errors)
    result = pipeline.process(frame)

    # Verify result
    assert hasattr(result, "pointcloud")
    assert result.pointcloud is not None


def test_pipeline_without_rerun_logging():
    """Test pipeline with Rerun logging disabled (default behavior)."""
    # Create a simple pipeline without Rerun logging
    pipeline = Pipeline(name="Test_Pipeline", enable_rerun_logging=False)

    # Add a simple node
    depth_to_pc_node = DepthToPointCloudNode(name="DepthToPointCloud")
    pipeline.add_node(depth_to_pc_node)

    # Create test data
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.random.rand(100, 100).astype(np.float32)
    intrinsics = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float64)

    frame = Frame(rgb=rgb, depth=depth, intrinsics=intrinsics)

    # Process through pipeline (should not raise errors)
    result = pipeline.process(frame)

    # Verify result
    assert hasattr(result, "pointcloud")
    assert result.pointcloud is not None


def test_rerun_logger_frame_logging():
    """Test logging Frame objects to Rerun."""
    # Create logger without spawning viewer for testing
    logger = RerunLogger("test_frame_logging", enabled=True, spawn=False)

    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.random.rand(100, 100).astype(np.float32)
    pointcloud = np.random.rand(1000, 3).astype(np.float32)

    frame = Frame(rgb=rgb, depth=depth, pointcloud=pointcloud)

    # This should not raise an error even if Rerun viewer isn't running
    logger.log_frame(frame, "test_frame")


if __name__ == "__main__":
    pytest.main([__file__])
