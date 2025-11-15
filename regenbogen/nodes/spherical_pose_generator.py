"""
Spherical pose generator node for creating camera poses on a sphere.

This node generates camera poses evenly distributed on a sphere around an object
for template rendering applications.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from ..core.node import Node
from ..interfaces import Pose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SphericalPoseGeneratorNode(Node):
    """
    Node for generating camera poses on a sphere around an object.

    This node creates camera poses evenly distributed on a sphere, useful for
    template rendering in object pose estimation pipelines like CNOS.

    Args:
        radius: Radius of the sphere around the object
        upper_hemisphere: If True, only generate poses on the upper hemisphere
        inplane_rotations: Number of in-plane rotations (yaw variations) per position
        icosahedron_subdivisions: Number of subdivisions for icosphere generation
        look_at_center: 3D point the cameras should look at (default is origin)
        up_vector: World up vector for camera orientation (if None, random yaw is used)
        **kwargs: Additional configuration parameters
    """

    def __init__(
        self,
        radius: float = 1.0,
        upper_hemisphere: bool = False,
        inplane_rotations: int = 1,
        icosahedron_subdivisions: int = 0,
        look_at_center: npt.NDArray[np.float64] | None = None,
        up_vector: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.radius = radius
        self.upper_hemisphere = upper_hemisphere
        self.inplane_rotations = inplane_rotations
        self.icosahedron_subdivisions = icosahedron_subdivisions

        # Set look at center and up vector
        self.look_at_center = (
            look_at_center
            if look_at_center is not None
            else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        self.up_vector = (
            up_vector
            if up_vector is not None
            else np.array([0.0, -1.0, 0.0], dtype=np.float64)
        )

    def _generate_icosahedron(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """
        Generate the initial icosahedron vertices and faces.

        Returns:
            Tuple of (vertices, faces) for the base icosahedron
        """
        # Create initial icosahedron vertices
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float64)

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Define icosahedron faces
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])

        return vertices, faces

    def _generate_sphere_points(self) -> list[npt.NDArray[np.float64]]:
        """
        Generate evenly distributed points on a sphere using icosphere subdivision.

        Returns:
            List of 3D points on the sphere surface
        """
        vertices, faces = self._generate_icosahedron()

        for _ in range(self.icosahedron_subdivisions):
            vertices, faces = self._subdivide_icosphere(vertices, faces)

        if self.upper_hemisphere:
            # Filter to upper hemisphere only (z >= 0)
            vertices = vertices[vertices[:, 2] >= -0.1]  # Allow slight negative z

        # Scale by radius and return as list
        points = []
        for vertex in vertices:
            point = vertex * self.radius
            points.append(point)

        return points

    def _subdivide_icosphere(self, vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int32]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """
        Subdivide icosphere by splitting each face into 4 triangles.

        Args:
            vertices: Current vertices
            faces: Current face indices

        Returns:
            New vertices and faces after subdivision
        """
        # Dictionary to store midpoints and avoid duplicates
        edge_midpoints = {}
        new_vertices = list(vertices)

        def get_midpoint(i1: int, i2: int) -> int:
            """Get midpoint between two vertices, creating if necessary."""
            edge = tuple(sorted([i1, i2]))
            if edge in edge_midpoints:
                return edge_midpoints[edge]

            # Create new vertex at midpoint
            midpoint = (vertices[i1] + vertices[i2]) / 2.0
            # Normalize to unit sphere
            midpoint = midpoint / np.linalg.norm(midpoint)

            new_vertices.append(midpoint)
            new_idx = len(new_vertices) - 1
            edge_midpoints[edge] = new_idx
            return new_idx

        # Create new faces
        new_faces = []
        for face in faces:
            v1, v2, v3 = face

            # Get midpoints
            a = get_midpoint(v1, v2)
            b = get_midpoint(v2, v3)
            c = get_midpoint(v3, v1)

            # Create 4 new faces
            new_faces.extend([
                [v1, a, c],
                [v2, b, a],
                [v3, c, b],
                [a, b, c]
            ])

        return np.array(new_vertices, dtype=np.float64), np.array(new_faces, dtype=np.int32)

    def calculate_forward_direction(
        self,
        camera_position: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the forward direction vector from camera to target.

        This is the direction the camera should "look" toward.

        Args:
            camera_position: Camera position in world space
            target_position: Target position to look at

        Returns:
            Normalized direction vector pointing from camera to target
        """
        forward_direction = self.look_at_center - camera_position

        direction_length = np.linalg.norm(forward_direction)
        if direction_length < 1e-10:
            raise ValueError("Camera and target positions are identical")

        return forward_direction / direction_length

    def _calculate_camera_rotation(
        self,
        camera_position: npt.NDArray[np.float64],
        yaw_angle: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate camera coordinate axes from forward direction.

        Args:
            forward_axis: Direction camera should look (normalized) - used for +Z axis
            up_vector: World up vector for orientation, if provided
            yaw_angle: Random yaw angle in radians, if up_vector not provided

        Returns:
            Rotation matrix (3x3) representing camera orientation
        """
        forward_axis = self.calculate_forward_direction(camera_position)
        up_vector = self.up_vector

        # Calculate X and Y axes based on up vector or yaw angle
        if up_vector is not None:
            # Use provided up vector to calculate right axis (X) via cross product
            right_axis = np.cross(up_vector, forward_axis)
            right_norm = np.linalg.norm(right_axis)

            if right_norm < 1e-6:
                # Up vector and forward axis are parallel, use alternative
                if abs(forward_axis[2]) < 0.9:
                    alt_up = np.array([0, 0, 1], dtype=np.float64)
                else:
                    alt_up = np.array([1, 0, 0], dtype=np.float64)
                right_axis = np.cross(alt_up, forward_axis)
                right_axis = right_axis / np.linalg.norm(right_axis)
            else:
                right_axis = right_axis / right_norm

            # Down axis: perpendicular to both right and forward
            down_axis = np.cross(forward_axis, right_axis)

            # Build rotation matrix with axes as columns
            rotation_matrix = np.column_stack([right_axis, down_axis, forward_axis])
            return rotation_matrix
        else:
            # Use yaw angle to determine orientation around forward axis
            if yaw_angle is None:
                yaw_angle = np.random.uniform(0, 2 * np.pi)

            # Create a temporary down vector that's not parallel to forward
            if abs(forward_axis[2]) < 0.9:
                temp_down = np.array([0, 0, 1], dtype=np.float64)
            else:
                temp_down = np.array([1, 0, 0], dtype=np.float64)

            # Calculate initial right axis
            right_initial = np.cross(temp_down, forward_axis)
            right_initial = right_initial / np.linalg.norm(right_initial)

            # Calculate initial down axis
            down_initial = np.cross(forward_axis, right_initial)

            # Apply yaw rotation around forward axis
            cos_yaw = np.cos(yaw_angle)
            sin_yaw = np.sin(yaw_angle)

            right_axis = cos_yaw * right_initial + sin_yaw * down_initial
            down_axis = -sin_yaw * right_initial + cos_yaw * down_initial

        # Build rotation matrix with axes as columns
        rotation_matrix = np.column_stack([right_axis, down_axis, forward_axis])
        return rotation_matrix

    def _apply_inplane_rotation(
        self, transform: npt.NDArray[np.float64], angle: float
    ) -> npt.NDArray[np.float64]:
        """
        Apply in-plane rotation around camera's Z axis using scipy Rotation.

        Args:
            transform: 4x4 transformation matrix
            angle: Rotation angle in radians

        Returns:
            Modified transformation matrix
        """
        # Create rotation around Z axis using scipy
        z_rotation = Rotation.from_rotvec([0, 0, angle])

        # Convert to 4x4 matrix
        inplane_rot = np.eye(4, dtype=np.float64)
        inplane_rot[:3, :3] = z_rotation.as_matrix()

        # Apply rotation in camera space
        return transform @ inplane_rot

    def generate_poses(self) -> Iterator[Pose]:
        """
        Generate camera poses on a sphere.

        Yields:
            Pose objects with rotation matrices and translations
        """
        sphere_points = self._generate_sphere_points()

        pose_count = 0
        for point in sphere_points:
            # Calculate camera position (offset from look_at_center)
            camera_position = self.look_at_center + point

            # Generate in-plane rotations (yaw variations if no up vector constraint)
            for i in range(self.inplane_rotations):
                if self.inplane_rotations > 1:
                    # Use different yaw angles for in-plane rotations
                    yaw_angle = 2 * np.pi * i / self.inplane_rotations
                    rotation = self._calculate_camera_rotation(
                        camera_position, yaw_angle=yaw_angle
                    )
                else:
                    # Use up vector for single orientation per position
                    rotation = self._calculate_camera_rotation(
                        camera_position, yaw_angle=None
                    )

                yield Pose(
                    rotation=rotation,
                    translation=camera_position,
                    metadata={
                        "pose_id": pose_count,
                        "camera_position": camera_position.tolist(),
                        "radius": self.radius,
                        "inplane_rotation": i if self.inplane_rotations > 1 else 0,
                    },
                )
                pose_count += 1

        logger.info(f"Generated {pose_count} camera poses")

    def process(self, input_data=None) -> Iterator[Pose]:
        """
        Process method for pipeline compatibility.

        Args:
            input_data: Ignored for pose generator

        Returns:
            Generator yielding Pose objects
        """
        return self.generate_poses()

    @property
    def total_poses(self) -> int:
        """Calculate total number of poses that will be generated."""
        sphere_points = self._generate_sphere_points()
        return len(sphere_points) * self.inplane_rotations
