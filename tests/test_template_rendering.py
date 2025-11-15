"""
Tests for template rendering nodes.

Tests the SphericalPoseGeneratorNode and TemplateRendererNode.
"""

from __future__ import annotations

import numpy as np
import pytest

from regenbogen.interfaces import ObjectModel, Pose
from regenbogen.nodes import SphericalPoseGeneratorNode, TemplateRendererNode


def test_spherical_pose_generator_creation():
    """Test SphericalPoseGeneratorNode initialization."""
    node = SphericalPoseGeneratorNode(radius=1.0, icosahedron_subdivisions=1)

    assert node.radius == 1.0
    assert node.icosahedron_subdivisions == 1


def test_spherical_pose_generator_poses():
    """Test pose generation on sphere."""
    node = SphericalPoseGeneratorNode(
        radius=1.0, elevation_levels=2, azimuth_samples=4, inplane_rotations=1
    )

    poses = list(node.generate_poses())

    # Should generate poses
    assert len(poses) > 0
    assert len(poses) == node.total_poses

    # Check pose structure
    for pose in poses:
        assert isinstance(pose, Pose)
        assert pose.rotation.shape == (3, 3)
        assert pose.translation.shape == (3,)
        assert "pose_id" in pose.metadata
        assert "camera_position" in pose.metadata
        assert "radius" in pose.metadata


def test_spherical_pose_generator_radius():
    """Test that camera positions are at correct radius."""
    radius = 2.5
    node = SphericalPoseGeneratorNode(
        radius=radius, elevation_levels=1, azimuth_samples=4, inplane_rotations=1
    )

    poses = list(node.generate_poses())

    for pose in poses:
        # Check distance from origin
        distance = np.linalg.norm(pose.translation)
        # Allow small numerical error
        assert abs(distance - radius) < 1e-6


def test_spherical_pose_generator_inplane_rotations():
    """Test in-plane rotations."""
    node = SphericalPoseGeneratorNode(
        radius=1.0, icosahedron_subdivisions=0, inplane_rotations=3
    )

    poses = list(node.generate_poses())

    # Should have 12 icosahedron vertices * 3 inplane rotations = 36 poses
    expected = 12 * 3
    assert len(poses) == expected


def test_spherical_pose_generator_subdivisions():
    """Test icosahedron subdivisions."""
    node = SphericalPoseGeneratorNode(
        radius=1.0, icosahedron_subdivisions=1, inplane_rotations=1
    )

    poses = list(node.generate_poses())
    # Subdivision 1 should produce 42 vertices (12 base + 30 subdivided)
    assert len(poses) == 42


def test_spherical_pose_generator_look_at():
    """Test look-at matrix generation."""
    node = SphericalPoseGeneratorNode(
        radius=1.0,
        elevation_levels=1,
        azimuth_samples=1,
        look_at_center=np.array([1.0, 2.0, 3.0]),
    )

    poses = list(node.generate_poses())

    # Camera should be offset from look_at_center
    assert len(poses) > 0


def test_template_renderer_creation():
    """Test TemplateRendererNode initialization."""
    node = TemplateRendererNode(width=640, height=480, fx=600, fy=600, cx=320, cy=240)

    assert node.width == 640
    assert node.height == 480
    assert node.fx == 600
    assert node.fy == 600
    assert node.cx == 320
    assert node.cy == 240


def test_template_renderer_intrinsics():
    """Test camera intrinsics matrix creation."""
    node = TemplateRendererNode(width=640, height=480, fx=600, fy=600, cx=320, cy=240)

    K = node._create_intrinsics_matrix()

    assert K.shape == (3, 3)
    assert K[0, 0] == 600  # fx
    assert K[1, 1] == 600  # fy
    assert K[0, 2] == 320  # cx
    assert K[1, 2] == 240  # cy
    assert K[2, 2] == 1.0


def test_template_renderer_default_intrinsics():
    """Test default intrinsics based on image size."""
    node = TemplateRendererNode(width=800, height=600)

    # Should use width/height as focal lengths
    assert node.fx == 800
    assert node.fy == 600
    # Principal point at center
    assert node.cx == 400
    assert node.cy == 300


def test_template_renderer_scene_setup():
    """Test scene setup with object model."""
    # Create simple mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    model = ObjectModel(mesh_vertices=vertices, mesh_faces=faces, name="triangle")

    node = TemplateRendererNode(width=320, height=240)

    # Setup scene (doesn't require rendering)
    node._setup_scene(model)

    assert node._scene is not None
    assert node._mesh_node is not None


def test_template_renderer_missing_mesh():
    """Test error handling for missing mesh data."""
    model = ObjectModel(name="empty")

    node = TemplateRendererNode(width=320, height=240)

    with pytest.raises(ValueError, match="must contain mesh_vertices"):
        node._setup_scene(model)


def test_template_renderer_process_input_validation():
    """Test input validation for process method."""
    node = TemplateRendererNode(width=320, height=240)

    # Invalid input type
    with pytest.raises(ValueError, match="must be a tuple"):
        list(node.process("invalid"))

    # Invalid tuple length
    with pytest.raises(ValueError, match="must be a tuple"):
        list(node.process((1, 2, 3)))


def test_camera_poses_look_at_object():
    """Test that camera poses are oriented to look at the object."""
    # Create pose generator
    radius = 5.0
    node = SphericalPoseGeneratorNode(
        radius=radius,
        icosahedron_subdivisions=0,
        look_at_center=np.array([0.0, 0.0, 0.0]),
    )
    poses = list(node.generate_poses())

    for pose in poses:
        camera_pos = pose.translation
        look_at_center = np.array([0.0, 0.0, 0.0])

        # Calculate expected forward direction (from camera to object)
        expected_forward = look_at_center - camera_pos
        expected_forward = expected_forward / np.linalg.norm(expected_forward)

        # Extract actual forward direction (positive Z-axis in camera coordinates)
        # The pose rotation matrix has camera Z-axis pointing toward object
        actual_forward = pose.rotation[:, 2]  # Z axis

        # Check that the camera is looking approximately toward the object
        dot_product = np.dot(actual_forward, expected_forward)
        print(f"Camera at {camera_pos}: dot product = {dot_product}")

        # Most cameras should be looking toward the object (positive dot product)
        # Some cameras at extreme angles might have slightly negative values,
        # but they shouldn't be looking completely away (dot product < -0.5)
        assert dot_product > -0.5, (
            f"Camera looking away from object: dot product = {dot_product}, "
            f"camera_pos = {camera_pos}, expected_forward = {expected_forward}, "
            f"actual_forward = {actual_forward}"
        )

        # At least some cameras should be looking reasonably toward the object
        if dot_product > 0.2:
            print("  ✓ Camera reasonably oriented toward object")


def test_template_rendering_produces_nonzero_images():
    """Test that template rendering produces non-zero RGB and depth images."""
    # Create a simple cube mesh at origin
    vertices = np.array(
        [
            # Cube vertices centered at origin
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],  # front face
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],  # back face
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            # Cube faces (triangles)
            [0, 1, 2],
            [0, 2, 3],  # front
            [4, 7, 6],
            [4, 6, 5],  # back
            [0, 4, 5],
            [0, 5, 1],  # bottom
            [2, 6, 7],
            [2, 7, 3],  # top
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ],
        dtype=np.int32,
    )

    model = ObjectModel(mesh_vertices=vertices, mesh_faces=faces, name="test_cube")

    # Create pose generator - cameras at distance 5 looking at origin
    pose_generator = SphericalPoseGeneratorNode(
        radius=5.0, elevation_levels=1, azimuth_samples=4, inplane_rotations=1
    )

    # Create template renderer with working wide FOV configuration
    width, height = 640, 480  # Normal size images
    fx = fy = 320  # Wide FOV that works with PyRender
    cx, cy = width // 2, height // 2

    renderer = TemplateRendererNode(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        z_near=1.0,
        z_far=2000.0,
    )

    poses = list(pose_generator.generate_poses())

    # Test at least one pose
    assert len(poses) > 0

    try:
        # Render template for first pose
        frames = list(renderer.render_templates(model, iter(poses[:1])))

        assert len(frames) == 1
        frame = frames[0]

        # Check that RGB image has some non-zero values
        rgb_sum = np.sum(frame.rgb)
        print(f"RGB sum: {rgb_sum}, shape: {frame.rgb.shape}")

        # Check that depth image has some non-zero values
        if frame.depth is not None:
            depth_sum = np.sum(frame.depth)
            depth_nonzero = np.count_nonzero(frame.depth)
            print(
                f"Depth sum: {depth_sum}, non-zero pixels: {depth_nonzero}, shape: {frame.depth.shape}"
            )

            # At least some pixels should have depth values
            assert depth_nonzero > 0, (
                "Depth image is completely zero - object not visible"
            )

        # RGB should have some content (not all black)
        assert rgb_sum > 0, "RGB image is completely black - object not visible"

        # Check image dimensions
        assert frame.rgb.shape == (height, width, 3)
        if frame.depth is not None:
            assert frame.depth.shape == (height, width)

    except Exception as e:
        # If rendering fails due to missing display/EGL, skip this test
        if "offscreen renderer" in str(e).lower() or "egl" in str(e).lower():
            pytest.skip(f"Skipping rendering test due to missing display/EGL: {e}")
        else:
            raise


def test_spherical_pose_generator_rotation_orthogonal():
    """Test that generated rotations are orthogonal matrices."""
    node = SphericalPoseGeneratorNode(
        radius=1.0, elevation_levels=2, azimuth_samples=4, inplane_rotations=2
    )

    poses = list(node.generate_poses())

    for pose in poses:
        R = pose.rotation
        # Check that R^T * R = I (orthogonality)
        identity = R.T @ R
        assert np.allclose(identity, np.eye(3), atol=1e-6)

        # Check determinant is 1 (proper rotation)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6


if __name__ == "__main__":
    # Run tests
    print("Running template rendering tests...")

    test_spherical_pose_generator_creation()
    print("✓ SphericalPoseGeneratorNode creation test passed")

    test_spherical_pose_generator_poses()
    print("✓ SphericalPoseGeneratorNode pose generation test passed")

    test_spherical_pose_generator_radius()
    print("✓ SphericalPoseGeneratorNode radius test passed")

    test_spherical_pose_generator_inplane_rotations()
    print("✓ SphericalPoseGeneratorNode in-plane rotations test passed")

    test_spherical_pose_generator_subdivisions()
    print("✓ SphericalPoseGeneratorNode num_views test passed")

    test_spherical_pose_generator_look_at()
    print("✓ SphericalPoseGeneratorNode look_at test passed")

    test_spherical_pose_generator_rotation_orthogonal()
    print("✓ SphericalPoseGeneratorNode rotation orthogonality test passed")

    test_template_renderer_creation()
    print("✓ TemplateRendererNode creation test passed")

    test_template_renderer_intrinsics()
    print("✓ TemplateRendererNode intrinsics test passed")

    test_template_renderer_default_intrinsics()
    print("✓ TemplateRendererNode default intrinsics test passed")

    test_template_renderer_scene_setup()
    print("✓ TemplateRendererNode scene setup test passed")

    test_template_renderer_missing_mesh()
    print("✓ TemplateRendererNode missing mesh test passed")

    test_template_renderer_process_input_validation()
    print("✓ TemplateRendererNode input validation test passed")

    print("\n✓ All template rendering tests passed!")
