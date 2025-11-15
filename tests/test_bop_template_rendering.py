#!/usr/bin/env python3
"""Tests for template rendering with actual BOP dataset objects."""

import numpy as np
import pytest

from regenbogen.nodes.bop_dataset import BOPDatasetNode
from regenbogen.nodes.spherical_pose_generator import SphericalPoseGeneratorNode
from regenbogen.nodes.template_renderer import TemplateRendererNode


def test_bop_object_scale_and_camera_positions():
    """Test BOP object scale and verify camera positions are outside the object."""
    # Load BOP dataset object (same as in example)
    bop_node = BOPDatasetNode(
        dataset_name="ycbv",
        split="test",
        load_models=True,
        allow_download=True
    )

    # Get the object model for object 1 (same as example)
    test_object = bop_node.get_object_model(1)

    if test_object is None:
        pytest.skip("Could not load object 1 from ycbv dataset")

    print(f"Object: {test_object.name}")
    print(f"Vertices shape: {test_object.mesh_vertices.shape}")
    print(f"Faces shape: {test_object.mesh_faces.shape}")

    # Calculate object bounding box and scale
    vertices = test_object.mesh_vertices
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_size = bbox_max - bbox_min
    object_center = (bbox_min + bbox_max) / 2

    print(f"Bounding box min: {bbox_min}")
    print(f"Bounding box max: {bbox_max}")
    print(f"Object center: {object_center}")
    print(f"Object size: {bbox_size}")
    print(f"Max dimension: {np.max(bbox_size)}")

    # Check units - BOP objects are typically in mm
    max_dimension_mm = np.max(bbox_size)
    max_dimension_m = max_dimension_mm / 1000.0

    print(f"Max dimension in mm: {max_dimension_mm:.2f}")
    print(f"Max dimension in meters: {max_dimension_m:.3f}")

    # Test with sphere radius from example (0.5m = 500mm)
    example_radius_m = 0.5
    example_radius_mm = example_radius_m * 1000  # 500mm

    print(f"Example sphere radius: {example_radius_m}m = {example_radius_mm}mm")

    # Check if cameras would be inside object
    if example_radius_mm < max_dimension_mm / 2:
        print(
            "❌ PROBLEM: Camera sphere radius is smaller than object - cameras will be inside!"
        )
        print(f"   Sphere radius: {example_radius_mm}mm")
        print(f"   Object radius: {max_dimension_mm / 2:.2f}mm")
    else:
        print("✅ OK: Camera sphere radius is larger than object")

    # Generate poses with the problematic radius
    pose_generator = SphericalPoseGeneratorNode(
        radius=example_radius_mm,  # Use mm to match object units

        icosahedron_subdivisions=0,
        look_at_center=object_center,
    )

    poses = list(pose_generator.generate_poses())
    print(f"Generated {len(poses)} poses")

    # Check if any camera positions are inside the object
    cameras_inside = 0
    for i, pose in enumerate(poses[:3]):  # Check first 3 poses
        camera_pos = pose.translation
        distance_from_center = np.linalg.norm(camera_pos - object_center)

        print(f"Camera {i}: position {camera_pos}")
        print(f"  Distance from center: {distance_from_center:.2f}mm")

        if distance_from_center < max_dimension_mm / 2:
            cameras_inside += 1
            print("  ❌ Camera is inside object!")
        else:
            print("  ✅ Camera is outside object")

    # Ensure no cameras are inside the object
    assert cameras_inside == 0, f"{cameras_inside} cameras are inside the object"


def test_bop_template_rendering_with_working_config():
    """Test template rendering with BOP object using working configuration that produces depth."""
    # Load BOP object
    bop_node = BOPDatasetNode(
        dataset_name="ycbv",
        split="test",
        load_models=True,
        allow_download=True
    )
    test_object = bop_node.get_object_model(1)

    if test_object is None:
        pytest.skip("Could not load object 1 from ycbv dataset")

    # Calculate appropriate sphere radius (3x the max dimension)
    vertices = test_object.mesh_vertices
    bbox_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    max_dimension = np.max(bbox_size)

    # Use a radius that's 3 times the object's maximum dimension
    safe_radius = max_dimension * 3

    print(f"Object max dimension: {max_dimension:.2f}mm")
    print(f"Using sphere radius: {safe_radius:.2f}mm")

    # Generate camera poses
    pose_generator = SphericalPoseGeneratorNode(
        radius=safe_radius,
    )

    # Create renderer with smaller size for faster testing
    renderer = TemplateRendererNode(width=64, height=64, fx=64, fy=64, cx=32, cy=32)

    poses = list(pose_generator.generate_poses())
    print(f"Generated {len(poses)} poses")

    try:
        # Render templates
        frames = list(
            renderer.render_templates(test_object, iter(poses[:2]))
        )  # Test 2 poses

        assert len(frames) == 2

        for i, frame in enumerate(frames):
            rgb_sum = np.sum(frame.rgb)
            depth_nonzero = (
                np.count_nonzero(frame.depth) if frame.depth is not None else 0
            )

            print(f"Frame {i}:")
            print(f"  RGB sum: {rgb_sum}")
            print(f"  Depth non-zero pixels: {depth_nonzero}")

            # With correct camera positioning, we should get non-zero images
            assert rgb_sum > 0, f"Frame {i} has zero RGB content - object not visible"
            if frame.depth is not None:
                assert depth_nonzero > 0, (
                    f"Frame {i} has zero depth content - object not visible"
                )

    except Exception as e:
        # If rendering fails due to missing display/EGL, skip this test
        if "offscreen renderer" in str(e).lower() or "egl" in str(e).lower():
            pytest.skip(f"Skipping rendering test due to missing display/EGL: {e}")
        else:
            raise


def test_unit_conversion_fixes_camera_positioning():
    """Test that proper unit conversion fixes camera positioning issues."""
    # Load BOP object
    bop_node = BOPDatasetNode(
        dataset_name="ycbv",
        split="test",
        load_models=True,
        allow_download=True
    )
    test_object = bop_node.get_object_model(1)

    if test_object is None:
        pytest.skip("Could not load object 1 from ycbv dataset")

    # Calculate object dimensions
    vertices = test_object.mesh_vertices
    bbox_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    max_dimension = np.max(bbox_size)
    object_center = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2

    # Test PROBLEMATIC radius (example value in wrong units)
    example_radius_wrong = 0.5  # Intended as meters, but treated as mm

    pose_gen_wrong = SphericalPoseGeneratorNode(
        radius=example_radius_wrong,

        icosahedron_subdivisions=0,
        look_at_center=object_center
    )

    poses_wrong = list(pose_gen_wrong.generate_poses())
    cameras_inside_wrong = sum(
        1 for pose in poses_wrong
        if np.linalg.norm(pose.translation - object_center) < max_dimension / 2
    )

    # All cameras should be inside with wrong units
    assert cameras_inside_wrong == len(poses_wrong), \
        f"Expected all cameras inside, got {cameras_inside_wrong}/{len(poses_wrong)}"

    # Test CORRECT radius (convert meters to mm)
    example_radius_correct = 0.5 * 1000  # Convert meters to mm

    pose_gen_correct = SphericalPoseGeneratorNode(
        radius=example_radius_correct,

        icosahedron_subdivisions=0,
        look_at_center=object_center
    )

    poses_correct = list(pose_gen_correct.generate_poses())
    cameras_inside_correct = sum(
        1 for pose in poses_correct
        if np.linalg.norm(pose.translation - object_center) < max_dimension / 2
    )

    # No cameras should be inside with correct units
    assert cameras_inside_correct == 0, \
        f"Expected no cameras inside, got {cameras_inside_correct}/{len(poses_correct)}"


def test_working_template_rendering_config():
    """Test template rendering with discovered working configuration."""
    # Load BOP object
    bop_node = BOPDatasetNode(
        dataset_name="ycbv",
        split="test",
        load_models=True,
        allow_download=True
    )
    test_object = bop_node.get_object_model(1)

    if test_object is None:
        pytest.skip("Could not load object 1 from ycbv dataset")

    # Object analysis
    vertices = test_object.mesh_vertices
    bbox_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    max_dimension = np.max(bbox_size)
    object_center = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2

    # Use working configuration: 500mm radius, wide FOV
    radius_mm = 500.0

    pose_generator = SphericalPoseGeneratorNode(
        radius=radius_mm,

        icosahedron_subdivisions=0,
        look_at_center=object_center
    )

    # Use normal-sized images with wide FOV configuration (scaled up from working 64x64)
    width, height = 640, 480
    # Scale the working focal length: 64x64 with fx=32 -> 640x480 with fx=320
    fx = fy = 320  # Wide FOV (key to success!)
    cx, cy = width // 2, height // 2

    renderer = TemplateRendererNode(
        width=width, height=height,
        fx=fx, fy=fy, cx=cx, cy=cy,
        z_near=1.0, z_far=2000.0
    )

    poses = list(pose_generator.generate_poses())
    pose = poses[0]

    # Verify camera positioning
    distance = np.linalg.norm(pose.translation - object_center)
    assert distance > max_dimension / 2, "Camera should be outside object"

    try:
        frames = list(renderer.render_templates(test_object, iter([pose])))
        assert len(frames) == 1

        frame = frames[0]
        rgb_sum = np.sum(frame.rgb)
        depth_nonzero = np.count_nonzero(frame.depth) if frame.depth is not None else 0

        print("Template rendering results:")
        print(f"  Image size: {width}x{height}")
        print(f"  RGB sum: {rgb_sum}")
        print(f"  Depth pixels: {depth_nonzero}/{frame.depth.size if frame.depth is not None else 0}")

        # These should be non-zero with working config
        assert rgb_sum > 0, "RGB should contain object content"
        if frame.depth is not None:
            assert depth_nonzero > 0, "Depth should contain object surface"

        print("✅ Working configuration confirmed with normal-sized images!")

    except Exception as e:
        if "offscreen renderer" in str(e).lower() or "egl" in str(e).lower():
            pytest.skip(f"Skipping: {e}")
        else:
            raise


def test_camera_distance_validation():
    """Test that validates camera distance relative to object size."""
    # Create a test object with known size (100mm cube)
    vertices = np.array(
        [
            [-50, -50, -50],
            [50, -50, -50],
            [50, 50, -50],
            [-50, 50, -50],  # front face
            [-50, -50, 50],
            [50, -50, 50],
            [50, 50, 50],
            [-50, 50, 50],  # back face
        ],
        dtype=np.float32,
    )  # 100mm cube (±50mm from center)

    faces = np.array(
        [
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

    from regenbogen.interfaces import ObjectModel

    # Create test object model for validation
    ObjectModel(
        mesh_vertices=vertices, mesh_faces=faces, name="test_100mm_cube"
    )

    object_size = 100.0  # mm
    object_radius = object_size / 2  # 50mm

    # Test different sphere radii
    test_cases = [
        (25.0, "too_small", True),  # Radius < object radius - cameras inside
        (50.0, "boundary", True),  # Radius = object radius - cameras on surface
        (75.0, "just_outside", False),  # Radius > object radius - cameras outside
        (150.0, "safe", False),  # Radius >> object radius - safely outside
    ]

    for radius, description, should_fail in test_cases:
        print(f"\nTesting radius {radius}mm ({description}):")

        pose_generator = SphericalPoseGeneratorNode(
            radius=radius,
            icosahedron_subdivisions=0,
            look_at_center=np.array([0.0, 0.0, 0.0]),
        )

        poses = list(pose_generator.generate_poses())

        cameras_inside = 0
        for pose in poses:
            distance = np.linalg.norm(pose.translation)
            if distance <= object_radius:
                cameras_inside += 1

        print(f"  Cameras inside object: {cameras_inside}/{len(poses)}")

        if should_fail:
            assert cameras_inside > 0, (
                f"Expected cameras inside object with radius {radius}mm"
            )
        else:
            assert cameras_inside == 0, (
                f"No cameras should be inside object with radius {radius}mm"
            )
