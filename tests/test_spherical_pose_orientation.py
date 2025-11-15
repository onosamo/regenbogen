"""
Unit tests for spherical pose generator orientation correctness.

This module tests that camera orientations are correctly calculated at each step:
1. Forward direction calculation points from camera to object center
2. Camera rotation matrix has Z-axis pointing toward object center
3. Final pose has correct orientation toward object center
"""

from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation

from regenbogen.nodes.spherical_pose_generator import SphericalPoseGeneratorNode


class TestForwardDirectionCalculation(unittest.TestCase):
    """Test that forward direction always points from camera to object center."""

    def setUp(self):
        """Set up test generator with default parameters."""
        self.generator = SphericalPoseGeneratorNode(
            radius=2.0,
            icosahedron_subdivisions=0,  # Use basic icosahedron for testing
            look_at_center=np.array([0.0, 0.0, 0.0]),
        )

    def test_forward_direction_basic_positions(self):
        """Test forward direction for basic axis-aligned camera positions."""
        test_cases = [
            # Camera position -> Expected forward direction
            (np.array([2.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),  # X-axis
            (np.array([0.0, 2.0, 0.0]), np.array([0.0, -1.0, 0.0])),  # Y-axis
            (np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0])),  # Z-axis
            (np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # -X-axis
            (np.array([0.0, -2.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # -Y-axis
            (np.array([0.0, 0.0, -2.0]), np.array([0.0, 0.0, 1.0])),  # -Z-axis
        ]

        for camera_pos, expected_forward in test_cases:
            with self.subTest(camera_position=camera_pos):
                forward = self.generator.calculate_forward_direction(camera_pos)
                npt.assert_allclose(forward, expected_forward, atol=1e-10)

    def test_forward_direction_diagonal_positions(self):
        """Test forward direction for diagonal camera positions."""
        # Camera at diagonal position (1,1,1) should look toward origin
        camera_pos = np.array([1.0, 1.0, 1.0])
        forward = self.generator.calculate_forward_direction(camera_pos)

        # Expected: normalized vector from (1,1,1) to (0,0,0)
        expected = -camera_pos / np.linalg.norm(camera_pos)
        npt.assert_allclose(forward, expected, atol=1e-10)

    def test_forward_direction_normalized(self):
        """Test that forward direction is always normalized."""
        test_positions = [
            np.array([5.0, 0.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
            np.array([3.0, 4.0, 5.0]),
            np.array([-2.5, 1.7, -3.3]),
        ]

        for pos in test_positions:
            with self.subTest(position=pos):
                forward = self.generator.calculate_forward_direction(pos)
                norm = np.linalg.norm(forward)
                self.assertAlmostEqual(norm, 1.0, places=10)

    def test_forward_direction_nonzero_center(self):
        """Test forward direction with non-zero object center."""
        center = np.array([1.0, 2.0, 3.0])
        generator = SphericalPoseGeneratorNode(
            radius=2.0,
            look_at_center=center,
        )

        camera_pos = np.array([4.0, 2.0, 3.0])  # 3 units along X from center
        forward = generator.calculate_forward_direction(camera_pos)

        # Should point from camera toward center
        expected = np.array([-1.0, 0.0, 0.0])  # Along -X axis
        npt.assert_allclose(forward, expected, atol=1e-10)


class TestCameraRotationCalculation(unittest.TestCase):
    """Test that camera rotation matrix has Z-axis pointing toward object center."""

    def setUp(self):
        """Set up test generator with default parameters."""
        self.generator = SphericalPoseGeneratorNode(
            radius=2.0,
            icosahedron_subdivisions=0,
            look_at_center=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),  # Use Y-up for consistency
        )

    def test_camera_rotation_z_axis_alignment(self):
        """Test that Z-axis of rotation matrix points toward object center."""
        test_positions = [
            np.array([2.0, 0.0, 0.0]),   # X-axis
            np.array([0.0, 0.0, 2.0]),   # Z-axis
            np.array([1.0, 1.0, 1.0]),   # Diagonal
            np.array([-1.5, 0.5, -2.0]), # Random position
        ]

        for camera_pos in test_positions:
            with self.subTest(camera_position=camera_pos):
                # Get rotation matrix
                rotation_matrix = self.generator._calculate_camera_rotation(camera_pos)

                # Extract Z-axis (3rd column - forward direction)
                z_axis = rotation_matrix[:, 2]

                # Expected forward direction (toward center)
                expected_forward = self.generator.calculate_forward_direction(camera_pos)

                # Check that Z-axis points toward center
                npt.assert_allclose(z_axis, expected_forward, atol=1e-10)

    def test_camera_rotation_orthogonality(self):
        """Test that rotation matrix axes are orthogonal."""
        test_positions = [
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ]

        for camera_pos in test_positions:
            with self.subTest(camera_position=camera_pos):
                rotation_matrix = self.generator._calculate_camera_rotation(camera_pos)

                # Extract axes (columns of rotation matrix)
                right_axis = rotation_matrix[:, 0]
                down_axis = rotation_matrix[:, 1]
                forward_axis = rotation_matrix[:, 2]

                # Check orthogonality (dot products should be zero)
                self.assertAlmostEqual(np.dot(right_axis, down_axis), 0.0, places=10)
                self.assertAlmostEqual(np.dot(right_axis, forward_axis), 0.0, places=10)
                self.assertAlmostEqual(np.dot(down_axis, forward_axis), 0.0, places=10)

    def test_camera_rotation_normalization(self):
        """Test that rotation matrix axes are normalized."""
        test_positions = [
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ]

        for camera_pos in test_positions:
            with self.subTest(camera_position=camera_pos):
                rotation_matrix = self.generator._calculate_camera_rotation(camera_pos)

                # Extract axes (columns of rotation matrix)
                right_axis = rotation_matrix[:, 0]
                down_axis = rotation_matrix[:, 1]
                forward_axis = rotation_matrix[:, 2]

                # Check normalization (length should be 1)
                self.assertAlmostEqual(np.linalg.norm(right_axis), 1.0, places=10)
                self.assertAlmostEqual(np.linalg.norm(down_axis), 1.0, places=10)
                self.assertAlmostEqual(np.linalg.norm(forward_axis), 1.0, places=10)

    def test_camera_rotation_right_handed(self):
        """Test that rotation matrix forms a right-handed coordinate system."""
        test_positions = [
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ]

        for camera_pos in test_positions:
            with self.subTest(camera_position=camera_pos):
                rotation_matrix = self.generator._calculate_camera_rotation(camera_pos)

                # Check determinant is positive (right-handed)
                det = np.linalg.det(rotation_matrix)
                self.assertGreater(det, 0.0)
                self.assertAlmostEqual(det, 1.0, places=10)


class TestFinalPoseOrientation(unittest.TestCase):
    """Test that final Pose objects have correct orientation toward object center."""

    def setUp(self):
        """Set up test generator with minimal poses for testing."""
        self.generator = SphericalPoseGeneratorNode(
            radius=2.0,
            icosahedron_subdivisions=0,  # Basic icosahedron (12 vertices)
            inplane_rotations=1,         # Single rotation per position
            look_at_center=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

    def test_pose_z_axis_toward_center(self):
        """Test that Z-axis of pose rotation matrix points toward object center."""
        poses = list(self.generator.generate_poses())

        for i, pose in enumerate(poses):
            with self.subTest(pose_id=i):
                # Extract Z-axis from rotation matrix (3rd column)
                z_axis = pose.rotation[:, 2]  # Z-axis (forward direction)

                # Calculate expected direction from camera to center
                expected_direction = self.generator.calculate_forward_direction(pose.translation)

                # Z-axis should point toward center
                dot_product = np.dot(z_axis, expected_direction)
                self.assertAlmostEqual(dot_product, 1.0, places=6,
                                     msg=f"Pose {i}: Z-axis not aligned with expected direction")

    def test_pose_camera_positions_on_sphere(self):
        """Test that camera positions are correctly positioned on sphere."""
        poses = list(self.generator.generate_poses())

        for i, pose in enumerate(poses):
            with self.subTest(pose_id=i):
                # Distance from center should equal radius
                distance = np.linalg.norm(pose.translation - self.generator.look_at_center)
                self.assertAlmostEqual(distance, self.generator.radius, places=6,
                                     msg=f"Pose {i}: Camera not on sphere surface")

    def test_pose_rotation_matrix_validity(self):
        """Test that rotation matrices are valid (orthogonal, unit determinant)."""
        poses = list(self.generator.generate_poses())

        for i, pose in enumerate(poses):
            with self.subTest(pose_id=i):
                R = pose.rotation

                # Check orthogonality: R @ R.T should be identity
                identity_check = R @ R.T
                npt.assert_allclose(identity_check, np.eye(3), atol=1e-6,
                                  err_msg=f"Pose {i}: Rotation matrix not orthogonal")

                # Check determinant is 1 (proper rotation, not reflection)
                det = np.linalg.det(R)
                self.assertAlmostEqual(det, 1.0, places=6,
                                     msg=f"Pose {i}: Invalid determinant {det}")

    def test_pose_consistent_with_scipy_rotation(self):
        """Test that rotation matrices work correctly with scipy Rotation."""
        poses = list(self.generator.generate_poses())

        for i, pose in enumerate(poses[:5]):  # Test first 5 poses
            with self.subTest(pose_id=i):
                # Should not raise exception when creating scipy Rotation
                try:
                    rot = Rotation.from_matrix(pose.rotation)
                    # Should be able to convert back
                    reconstructed = rot.as_matrix()
                    npt.assert_allclose(reconstructed, pose.rotation, atol=1e-10)
                except ValueError as e:
                    self.fail(f"Pose {i}: scipy Rotation failed: {e}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_forward_direction_identical_positions(self):
        """Test error when camera and target are at same position."""
        generator = SphericalPoseGeneratorNode(
            look_at_center=np.array([1.0, 2.0, 3.0])
        )

        # Camera at same position as target
        with self.assertRaises(ValueError):
            generator.calculate_forward_direction(np.array([1.0, 2.0, 3.0]))

    def test_different_object_centers(self):
        """Test with different object centers."""
        centers = [
            np.array([5.0, 0.0, 0.0]),
            np.array([0.0, -3.0, 7.0]),
            np.array([1.5, -2.5, 4.2]),
        ]

        for center in centers:
            with self.subTest(center=center):
                generator = SphericalPoseGeneratorNode(
                    radius=1.0,
                    icosahedron_subdivisions=0,
                    look_at_center=center,
                    up_vector=np.array([0.0, 1.0, 0.0]),
                )

                poses = list(generator.generate_poses())
                self.assertGreater(len(poses), 0)

                # Check first pose orientation
                pose = poses[0]
                z_axis = pose.rotation[:, 2]
                expected_direction = generator.calculate_forward_direction(pose.translation)

                dot_product = np.dot(z_axis, expected_direction)
                self.assertAlmostEqual(dot_product, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
