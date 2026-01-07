"""
Basic tests for regenbogen framework.

Tests core functionality and node operations.
"""

import numpy as np

from regenbogen import Node, Pipeline
from regenbogen.interfaces import BoundingBoxes, Frame, ObjectModel, Pose
from regenbogen.nodes import (
    DepthAnythingNode,
    DepthToPointCloudNode,
    ICPRefinementNode,
    MeshSamplingNode,
)


def test_frame_interface():
    """Test Frame interface creation and validation."""
    rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    intrinsics = np.eye(3, dtype=np.float64)

    frame = Frame(rgb=rgb, intrinsics=intrinsics)

    assert frame.rgb.shape == (480, 640, 3)
    assert frame.intrinsics.shape == (3, 3)
    assert frame.depth is None
    assert isinstance(frame.metadata, dict)


def test_object_model_interface():
    """Test ObjectModel interface creation."""
    vertices = np.random.randn(100, 3).astype(np.float32)
    faces = np.random.randint(0, 100, (50, 3), dtype=np.int32)

    model = ObjectModel(mesh_vertices=vertices, mesh_faces=faces, name="test_object")

    assert model.mesh_vertices.shape == (100, 3)
    assert model.mesh_faces.shape == (50, 3)
    assert model.name == "test_object"
    assert isinstance(model.metadata, dict)


def test_pose_interface():
    """Test Pose interface creation."""
    rotation = np.eye(3, dtype=np.float64)
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    pose = Pose(rotation=rotation, translation=translation)

    assert pose.rotation.shape == (3, 3)
    assert pose.translation.shape == (3,)
    assert isinstance(pose.scores, dict)
    assert isinstance(pose.metadata, dict)


def test_bounding_boxes_interface():
    """Test BoundingBoxes interface creation."""
    boxes = np.array([[10, 20, 100, 150], [200, 300, 350, 400]], dtype=np.float32)
    scores = np.array([0.8, 0.9], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)

    bboxes = BoundingBoxes(
        boxes=boxes, scores=scores, labels=labels, class_names=["person", "car"]
    )

    assert bboxes.boxes.shape == (2, 4)
    assert bboxes.scores.shape == (2,)
    assert bboxes.labels.shape == (2,)
    assert len(bboxes.class_names) == 2


def test_pipeline_creation():
    """Test Pipeline creation and node addition."""
    pipeline = Pipeline(name="test_pipeline")

    assert pipeline.name == "test_pipeline"
    assert len(pipeline) == 0

    # Create dummy node
    class DummyNode(Node):
        def process(self, input_data):
            return input_data

    node = DummyNode(name="dummy")
    pipeline.add_node(node)

    assert len(pipeline) == 1
    assert pipeline.get_node("dummy") is not None


def test_depth_anything_node():
    """Test DepthAnythingNode processing."""
    rgb = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb)

    node = DepthAnythingNode(model_size="small")
    result = node.process(frame)

    assert isinstance(result, Frame)
    assert result.depth is not None
    assert result.depth.shape == (240, 320)
    assert result.rgb.shape == (240, 320, 3)


def test_depth_to_pointcloud_node():
    """Test DepthToPointCloudNode processing."""
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    depth = np.random.uniform(1.0, 5.0, (100, 100)).astype(np.float32)
    intrinsics = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float64)

    frame = Frame(rgb=rgb, depth=depth, intrinsics=intrinsics)

    node = DepthToPointCloudNode()
    result = node.process(frame)

    assert isinstance(result, Frame)
    assert result.pointcloud is not None
    assert result.pointcloud.points.shape[1] == 3  # 3D points


def test_mesh_sampling_node():
    """Test MeshSamplingNode processing."""
    # Create simple triangle
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    model = ObjectModel(mesh_vertices=vertices, mesh_faces=faces, name="triangle")

    node = MeshSamplingNode(num_points=100)
    result = node.process(model)

    assert isinstance(result, ObjectModel)
    assert result.pointcloud is not None
    assert len(result.pointcloud) == 100
    assert result.pointcloud.shape[1] == 3


def test_icp_refinement_node():
    """Test ICPRefinementNode processing."""
    # Create test data
    source_pc = np.random.randn(50, 3).astype(np.float32)
    target_pc = np.random.randn(50, 3).astype(np.float32)

    model = ObjectModel(pointcloud=source_pc, name="test")
    initial_pose = Pose(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    node = ICPRefinementNode(max_iterations=5)
    result = node.process((model, target_pc, initial_pose))

    assert isinstance(result, Pose)
    assert result.rotation.shape == (3, 3)
    assert result.translation.shape == (3,)
    assert "icp_score" in result.scores


def test_pipeline_processing():
    """Test full pipeline processing."""
    # Create simple pipeline
    pipeline = Pipeline(name="test_pipeline")

    # Add depth estimation and pointcloud generation
    pipeline.add_node(DepthAnythingNode(name="depth"))
    pipeline.add_node(DepthToPointCloudNode(name="pointcloud"))

    # Create test input
    rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    intrinsics = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float64)
    frame = Frame(rgb=rgb, intrinsics=intrinsics)

    # Process through pipeline
    result = pipeline.process(frame)

    assert isinstance(result, Frame)
    assert result.depth is not None
    assert result.pointcloud is not None
    assert result.pointcloud.points.shape[1] == 3


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")

    test_frame_interface()
    print("✓ Frame interface test passed")

    test_object_model_interface()
    print("✓ ObjectModel interface test passed")

    test_pose_interface()
    print("✓ Pose interface test passed")

    test_bounding_boxes_interface()
    print("✓ BoundingBoxes interface test passed")

    test_pipeline_creation()
    print("✓ Pipeline creation test passed")

    test_depth_anything_node()
    print("✓ DepthAnythingNode test passed")

    test_depth_to_pointcloud_node()
    print("✓ DepthToPointCloudNode test passed")

    test_mesh_sampling_node()
    print("✓ MeshSamplingNode test passed")

    test_icp_refinement_node()
    print("✓ ICPRefinementNode test passed")

    test_pipeline_processing()
    print("✓ Pipeline processing test passed")

    print("\n🎉 All tests passed successfully!")
