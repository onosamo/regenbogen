"""
Mesh sampling node.

This node samples points from 3D meshes to create reference pointclouds.
"""

import numpy as np

from ..core.node import Node
from ..interfaces import ObjectModel


class MeshSamplingNode(Node):
    """
    Node for sampling points from 3D meshes.

    Creates dense pointclouds from triangle meshes using various sampling strategies.
    """

    def __init__(
        self,
        num_points: int = 10000,
        sampling_method: str = "uniform",
        add_noise: bool = False,
        noise_std: float = 0.001,
        **kwargs,
    ):
        """
        Initialize mesh sampling node.

        Args:
            num_points: Number of points to sample from mesh
            sampling_method: Sampling strategy ("uniform", "face_area_weighted")
            add_noise: Whether to add Gaussian noise to sampled points
            noise_std: Standard deviation of noise to add
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.num_points = num_points
        self.sampling_method = sampling_method
        self.add_noise = add_noise
        self.noise_std = noise_std

    def process(self, object_model: ObjectModel) -> ObjectModel:
        """
        Sample points from object mesh.

        Args:
            object_model: Input object model with mesh

        Returns:
            ObjectModel with sampled pointcloud added
        """
        if object_model.mesh_vertices is None or object_model.mesh_faces is None:
            raise ValueError("Input object model must contain mesh vertices and faces")

        # Sample points from mesh
        pointcloud = self._sample_mesh(
            object_model.mesh_vertices, object_model.mesh_faces
        )

        # Create output object model
        output_model = ObjectModel(
            mesh_vertices=object_model.mesh_vertices,
            mesh_faces=object_model.mesh_faces,
            pointcloud=pointcloud,
            name=object_model.name,
            metadata=object_model.metadata.copy() if object_model.metadata else {},
        )

        output_model.metadata["sampling_method"] = self.sampling_method
        output_model.metadata["num_sampled_points"] = len(pointcloud)
        output_model.metadata["noise_added"] = self.add_noise

        return output_model

    def _sample_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Sample points from triangle mesh.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3) as vertex indices

        Returns:
            Sampled pointcloud (num_points, 3)
        """
        if self.sampling_method == "uniform":
            return self._uniform_sampling(vertices, faces)
        elif self.sampling_method == "face_area_weighted":
            return self._face_area_weighted_sampling(vertices, faces)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def _uniform_sampling(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Uniform sampling from mesh faces.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)

        Returns:
            Uniformly sampled points (num_points, 3)
        """
        # Calculate face areas for weighted sampling
        face_areas = self._calculate_face_areas(vertices, faces)
        face_probabilities = face_areas / np.sum(face_areas)

        # Sample faces according to their areas
        face_indices = np.random.choice(
            len(faces), size=self.num_points, p=face_probabilities
        )

        # Sample points within selected faces
        sampled_points = []

        for face_idx in face_indices:
            face = faces[face_idx]
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Random barycentric coordinates
            r1, r2 = np.random.random(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2

            # Sample point using barycentric coordinates
            point = r1 * v0 + r2 * v1 + r3 * v2
            sampled_points.append(point)

        pointcloud = np.array(sampled_points, dtype=np.float32)

        # Add noise if requested
        if self.add_noise:
            noise = np.random.normal(0, self.noise_std, pointcloud.shape)
            pointcloud += noise.astype(np.float32)

        return pointcloud

    def _face_area_weighted_sampling(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """
        Face area weighted sampling (same as uniform for triangle meshes).

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)

        Returns:
            Area-weighted sampled points (num_points, 3)
        """
        # For triangle meshes, this is the same as uniform sampling
        return self._uniform_sampling(vertices, faces)

    def _calculate_face_areas(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """
        Calculate areas of triangle faces.

        Args:
            vertices: Mesh vertices (N, 3)
            faces: Mesh faces (M, 3)

        Returns:
            Face areas (M,)
        """
        # Get triangle vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Calculate cross product for area
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        return areas
