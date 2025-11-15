"""
Tests for BOP dataset loader node.

Tests the BOPDatasetNode functionality including:
- Mock dataset creation and loading
- Streaming samples
- Loading object models
- Getting specific samples
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add parent directory to path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import core interfaces first
# Import BOPDatasetNode by loading it directly to avoid torch dependency
import importlib
import importlib.util

from regenbogen.interfaces import Frame, ObjectModel, Pose

# Create a mock regenbogen.nodes module to avoid full import
if "regenbogen.nodes" not in sys.modules:
    sys.modules["regenbogen.nodes"] = type(sys)("regenbogen.nodes")

# Now import the node module directly
bop_dataset_spec = importlib.util.spec_from_file_location(
    "regenbogen.nodes.bop_dataset",
    Path(__file__).parent.parent / "regenbogen" / "nodes" / "bop_dataset.py",
)
bop_dataset_module = importlib.util.module_from_spec(bop_dataset_spec)
sys.modules["regenbogen.nodes.bop_dataset"] = bop_dataset_module
bop_dataset_spec.loader.exec_module(bop_dataset_module)

BOPDatasetNode = bop_dataset_module.BOPDatasetNode


def create_test_bop_dataset(base_path: Path) -> Path:
    """
    Create a minimal mock BOP dataset for testing.

    Args:
        base_path: Base directory for the dataset

    Returns:
        Path to the created dataset
    """
    # Create test split directory
    test_path = base_path / "test"
    test_path.mkdir(parents=True, exist_ok=True)

    # Create two test scenes
    for scene_id in [0, 1]:
        scene_path = test_path / f"{scene_id:06d}"
        scene_path.mkdir(exist_ok=True)

        # Create RGB directory
        rgb_path = scene_path / "rgb"
        rgb_path.mkdir(exist_ok=True)

        # Create test images
        has_cv2 = False
        try:
            import cv2

            has_cv2 = True
            for img_id in [0, 1]:
                test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(
                    str(rgb_path / f"{img_id:06d}.png"),
                    cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR),
                )
        except ImportError:
            # Try using PIL as fallback
            try:
                from PIL import Image

                has_cv2 = True  # Can read with PIL
                for img_id in [0, 1]:
                    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    img = Image.fromarray(test_img, "RGB")
                    img.save(str(rgb_path / f"{img_id:06d}.png"))
            except ImportError:
                # Create empty placeholder files
                for img_id in [0, 1]:
                    (rgb_path / f"{img_id:06d}.png").touch()

        # Create depth directory (optional)
        depth_path = scene_path / "depth"
        depth_path.mkdir(exist_ok=True)

        try:
            import cv2

            for img_id in [0, 1]:
                depth_img = np.random.randint(1000, 5000, (480, 640), dtype=np.uint16)
                cv2.imwrite(str(depth_path / f"{img_id:06d}.png"), depth_img)
        except ImportError:
            pass

        # Create scene_camera.json
        scene_camera = {}
        for img_id in [0, 1]:
            scene_camera[str(img_id)] = {
                "cam_K": [
                    572.4114,
                    0.0,
                    325.2611,
                    0.0,
                    573.57043,
                    242.04899,
                    0.0,
                    0.0,
                    1.0,
                ],
                "depth_scale": 0.1,
            }

        with open(scene_path / "scene_camera.json", "w") as f:
            json.dump(scene_camera, f, indent=2)

        # Create scene_gt.json
        scene_gt = {}
        for img_id in [0, 1]:
            scene_gt[str(img_id)] = [
                {
                    "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "cam_t_m2c": [0.0, 0.0, 500.0 + scene_id * 100 + img_id * 10],
                    "obj_id": 1,
                },
                {
                    "cam_R_m2c": [0.866, -0.5, 0.0, 0.5, 0.866, 0.0, 0.0, 0.0, 1.0],
                    "cam_t_m2c": [100.0, 50.0, 600.0],
                    "obj_id": 2,
                },
            ]

        with open(scene_path / "scene_gt.json", "w") as f:
            json.dump(scene_gt, f, indent=2)

    # Create models directory (optional)
    models_path = base_path / "models"
    models_path.mkdir(exist_ok=True)

    # Create models_info.json
    models_info = {
        "1": {"diameter": 100.0, "min_x": -50.0, "min_y": -50.0, "min_z": -50.0},
        "2": {"diameter": 80.0, "min_x": -40.0, "min_y": -40.0, "min_z": -40.0},
    }

    with open(models_path / "models_info.json", "w") as f:
        json.dump(models_info, f, indent=2)

    # Create simple .ply files (optional - requires Open3D)
    try:
        import open3d as o3d

        # Create simple cube mesh for object 1
        mesh1 = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        o3d.io.write_triangle_mesh(str(models_path / "obj_000001.ply"), mesh1)

        # Create simple sphere mesh for object 2
        mesh2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        o3d.io.write_triangle_mesh(str(models_path / "obj_000002.ply"), mesh2)
    except (ImportError, Exception):
        # Skip model creation if Open3D not available
        pass

    return base_path, has_cv2


def test_bop_dataset_creation():
    """Test BOP dataset node creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        # Create BOP loader
        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            name="TestLoader",
        )

        assert bop_loader.name == "TestLoader"
        assert bop_loader.split == "test"
        assert bop_loader.dataset_path == dataset_path

        print("✓ BOP dataset node creation test passed")


def test_bop_dataset_streaming():
    """Test streaming samples from BOP dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        if not has_cv2:
            print("✓ BOP dataset streaming test skipped (OpenCV/PIL not available)")
            return

        # Create BOP loader with max_samples limit
        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            max_samples=2,
            name="TestLoader",
        )

        # Stream samples
        samples = list(bop_loader.stream_dataset())

        assert len(samples) == 2, f"Expected 2 samples, got {len(samples)}"

        for frame, gt_poses, obj_ids in samples:
            # Check frame
            assert isinstance(frame, Frame)
            assert frame.rgb is not None
            assert frame.intrinsics is not None
            assert frame.intrinsics.shape == (3, 3)

            # Check ground truth
            assert len(gt_poses) == len(obj_ids)
            assert len(gt_poses) > 0  # Should have at least one object

            for pose, obj_id in zip(gt_poses, obj_ids):
                assert isinstance(pose, Pose)
                assert pose.rotation.shape == (3, 3)
                assert pose.translation.shape == (3,)
                assert isinstance(obj_id, int)

        print("✓ BOP dataset streaming test passed")


def test_bop_dataset_specific_scene():
    """Test loading a specific scene."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        if not has_cv2:
            print(
                "✓ BOP dataset specific scene test skipped (OpenCV/PIL not available)"
            )
            return

        # Load only scene 0
        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            scene_id=0,
            name="TestLoader",
        )

        samples = list(bop_loader.stream_dataset())

        # Should have samples from scene 0 only
        for frame, _, _ in samples:
            assert frame.metadata["scene_id"] == 0

        print("✓ BOP dataset specific scene test passed")


def test_bop_dataset_get_sample():
    """Test getting a specific sample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        if not has_cv2:
            print("✓ BOP dataset get_sample test skipped (OpenCV/PIL not available)")
            return

        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            name="TestLoader",
        )

        # Get a specific sample
        result = bop_loader.get_sample(scene_id=0, image_id=0)

        assert result is not None
        frame, gt_poses, obj_ids = result

        assert isinstance(frame, Frame)
        assert frame.metadata["scene_id"] == 0
        assert frame.metadata["image_id"] == 0
        assert len(gt_poses) > 0

        print("✓ BOP dataset get_sample test passed")


def test_bop_dataset_with_depth():
    """Test loading depth maps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        if not has_cv2:
            print("✓ BOP dataset with depth test skipped (OpenCV/PIL not available)")
            return

        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            scene_id=0,
            max_samples=1,
            load_depth=True,
            name="TestLoader",
        )

        samples = list(bop_loader.stream_dataset())

        if len(samples) > 0:
            frame, _, _ = samples[0]
            # Depth may or may not be loaded depending on if cv2 is available
            if frame.depth is not None:
                assert frame.depth.dtype == np.float32
                print("✓ BOP dataset with depth test passed (depth loaded)")
            else:
                print("✓ BOP dataset with depth test passed (depth not available)")


def test_bop_dataset_load_models():
    """Test loading object models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            load_models=True,
            name="TestLoader",
        )

        # Get object models
        models = bop_loader.get_object_models()

        # Models may or may not be loaded depending on if Open3D is available
        if len(models) > 0:
            # Check that models are valid
            for obj_id, model in models.items():
                assert isinstance(model, ObjectModel)
                assert model.name != ""
                print(f"  Loaded model {obj_id}: {model.name}")

            # Test get_object_model
            model = bop_loader.get_object_model(list(models.keys())[0])
            assert model is not None
            assert isinstance(model, ObjectModel)

            print(
                f"✓ BOP dataset load models test passed ({len(models)} models loaded)"
            )
        else:
            print(
                "✓ BOP dataset load models test passed (no models available - Open3D may not be installed)"
            )


def test_bop_dataset_properties():
    """Test dataset properties."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            name="TestLoader",
        )

        # Test num_scenes property
        num_scenes = bop_loader.num_scenes
        assert num_scenes == 2  # We created 2 scenes

        # Test available_models property
        available_models = bop_loader.available_models
        assert isinstance(available_models, list)

        print(
            f"✓ BOP dataset properties test passed (scenes={num_scenes}, models={len(available_models)})"
        )


def test_bop_dataset_pipeline_integration():
    """Test integration with Pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        if not has_cv2:
            print(
                "✓ BOP dataset pipeline integration test skipped (OpenCV/PIL not available)"
            )
            return

        from regenbogen import Pipeline

        pipeline = Pipeline(name="TestPipeline")

        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            scene_id=0,
            max_samples=2,
            name="TestLoader",
        )

        pipeline.add_node(bop_loader)

        assert len(pipeline) == 1
        assert pipeline.get_node("TestLoader") is not None

        # Process through pipeline
        sample_count = 0
        for frame, gt_poses, obj_ids in bop_loader.process():
            sample_count += 1
            assert isinstance(frame, Frame)
            assert isinstance(gt_poses, list)
            assert isinstance(obj_ids, list)

        assert sample_count == 2

        print("✓ BOP dataset pipeline integration test passed")


def test_bop_dataset_metadata():
    """Test that metadata is properly set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset_path, has_cv2 = create_test_bop_dataset(dataset_path)

        if not has_cv2:
            print("✓ BOP dataset metadata test skipped (OpenCV/PIL not available)")
            return

        bop_loader = BOPDatasetNode(
            dataset_name="test",
            dataset_path=str(dataset_path),
            allow_download=False,
            split="test",
            scene_id=0,
            max_samples=1,
            name="TestLoader",
        )

        samples = list(bop_loader.stream_dataset())

        if len(samples) > 0:
            frame, gt_poses, obj_ids = samples[0]

            # Check frame metadata
            assert "dataset" in frame.metadata
            assert frame.metadata["dataset"] == "BOP"
            assert "split" in frame.metadata
            assert frame.metadata["split"] == "test"
            assert "scene_id" in frame.metadata
            assert "image_id" in frame.metadata

            # Check pose metadata
            for pose, obj_id in zip(gt_poses, obj_ids):
                assert "obj_id" in pose.metadata
                assert pose.metadata["obj_id"] == obj_id
                assert "scene_id" in pose.metadata
                assert "image_id" in pose.metadata

        print("✓ BOP dataset metadata test passed")


if __name__ == "__main__":
    # Run all tests
    print("Running BOP dataset loader tests...\n")

    test_bop_dataset_creation()
    test_bop_dataset_streaming()
    test_bop_dataset_specific_scene()
    test_bop_dataset_get_sample()
    test_bop_dataset_with_depth()
    test_bop_dataset_load_models()
    test_bop_dataset_properties()
    test_bop_dataset_pipeline_integration()
    test_bop_dataset_metadata()

    print("\n🎉 All BOP dataset loader tests passed!")
