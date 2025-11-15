"""
Fine pose refinement using ICP (Iterative Closest Point).

This node refines pose estimates using ICP alignment algorithms.
"""

from typing import Tuple

import numpy as np

from ..core.node import Node
from ..interfaces import ObjectModel, Pose


class ICPRefinementNode(Node):
    """
    Node for fine pose refinement using ICP (Iterative Closest Point).

    Takes an initial pose estimate and refines it by iteratively aligning
    the object model pointcloud with the scene pointcloud.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 1.0,  # Increased for synthetic data
        outlier_rejection_threshold: float = 2.0,
        **kwargs,
    ):
        """
        Initialize ICP refinement node.

        Args:
            max_iterations: Maximum number of ICP iterations
            tolerance: Convergence tolerance for pose change
            max_correspondence_distance: Maximum distance for point correspondences
            outlier_rejection_threshold: Threshold for outlier rejection (in std devs)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        self.outlier_rejection_threshold = outlier_rejection_threshold

    def process(self, inputs: Tuple[ObjectModel, np.ndarray, Pose]) -> Pose:
        """
        Refine pose using ICP alignment.

        Args:
            inputs: Tuple of (ObjectModel, scene pointcloud, initial pose)

        Returns:
            Refined pose estimate
        """
        object_model, scene_pointcloud, initial_pose = inputs

        if object_model.pointcloud is None:
            raise ValueError("Object model must contain reference pointcloud")

        # Refine pose using ICP
        refined_pose = self._icp_refinement(
            object_model.pointcloud, scene_pointcloud, initial_pose
        )

        return refined_pose

    def _icp_refinement(
        self, source_pc: np.ndarray, target_pc: np.ndarray, initial_pose: Pose
    ) -> Pose:
        """
        Refine pose using ICP algorithm.

        Args:
            source_pc: Source pointcloud (object model)
            target_pc: Target pointcloud (scene)
            initial_pose: Initial pose estimate

        Returns:
            Refined pose
        """
        # Downsample pointclouds for faster processing
        if len(source_pc) > 1000:
            indices = np.random.choice(len(source_pc), 1000, replace=False)
            source_pc = source_pc[indices]

        if len(target_pc) > 1000:
            indices = np.random.choice(len(target_pc), 1000, replace=False)
            target_pc = target_pc[indices]

        print(f"ICP: Using {len(source_pc)} source and {len(target_pc)} target points")

        # Initialize with input pose
        current_rotation = initial_pose.rotation.copy()
        current_translation = initial_pose.translation.copy()

        # Transform source pointcloud with initial pose
        transformed_source = self._transform_pointcloud(
            source_pc, current_rotation, current_translation
        )

        best_score = float("inf")
        iteration_scores = []
        valid_correspondences = np.array(
            []
        )  # Initialize for case where no iterations succeed

        for iteration in range(self.max_iterations):
            # Find closest point correspondences
            correspondences, distances = self._find_correspondences(
                transformed_source, target_pc
            )

            if len(correspondences) < 3:
                print(
                    f"ICP: Too few correspondences ({len(correspondences)}) at iteration {iteration}"
                )
                break

            # Reject outliers
            valid_correspondences = self._reject_outliers(correspondences, distances)

            if len(valid_correspondences) < 3:
                print(
                    f"ICP: Too few valid correspondences after outlier rejection at iteration {iteration}"
                )
                break

            # Solve for pose update
            source_matched = transformed_source[valid_correspondences[:, 0]]
            target_matched = target_pc[valid_correspondences[:, 1]]

            delta_rotation, delta_translation = self._solve_pose_update(
                source_matched, target_matched
            )

            # Update pose
            current_rotation = delta_rotation @ current_rotation
            current_translation = (
                delta_rotation @ current_translation + delta_translation
            )

            # Update transformed source
            transformed_source = self._transform_pointcloud(
                source_pc, current_rotation, current_translation
            )

            # Calculate alignment score (use mean distance of valid correspondences)
            alignment_score = np.mean(
                np.linalg.norm(source_matched - target_matched, axis=1)
            )
            iteration_scores.append(alignment_score)

            # Check convergence
            pose_change = np.linalg.norm(delta_translation) + np.linalg.norm(
                delta_rotation - np.eye(3)
            )

            if pose_change < self.tolerance:
                print(f"ICP converged after {iteration + 1} iterations")
                break

            if alignment_score < best_score:
                best_score = alignment_score

        # Create refined pose
        refined_pose = Pose(
            rotation=current_rotation,
            translation=current_translation,
            scores={
                "icp_score": best_score,
                "final_correspondences": len(valid_correspondences),
                "convergence_iterations": len(iteration_scores),
            },
            metadata={
                "method": "icp_point_to_point",
                "max_iterations": self.max_iterations,
                "final_iteration": len(iteration_scores),
                "iteration_scores": iteration_scores,
                "initial_score": initial_pose.scores.get("coarse_score", 0.0),
            },
        )

        return refined_pose

    def _transform_pointcloud(
        self, pointcloud: np.ndarray, rotation: np.ndarray, translation: np.ndarray
    ) -> np.ndarray:
        """
        Transform pointcloud with rotation and translation.

        Args:
            pointcloud: Input pointcloud (N, 3)
            rotation: Rotation matrix (3, 3)
            translation: Translation vector (3,)

        Returns:
            Transformed pointcloud (N, 3)
        """
        return (rotation @ pointcloud.T).T + translation

    def _find_correspondences(
        self, source_pc: np.ndarray, target_pc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find closest point correspondences between pointclouds.

        Args:
            source_pc: Source pointcloud (N, 3)
            target_pc: Target pointcloud (M, 3)

        Returns:
            Tuple of (correspondences array (K, 2), distances array (K,))
        """
        # Compute pairwise distances (simplified for large pointclouds)
        # In practice, would use KD-tree for efficiency

        correspondences = []
        distances = []

        for i, source_point in enumerate(source_pc):
            # Find closest target point
            dists = np.linalg.norm(target_pc - source_point, axis=1)
            closest_idx = np.argmin(dists)
            closest_dist = dists[closest_idx]

            # Only include if within max correspondence distance
            if closest_dist <= self.max_correspondence_distance:
                correspondences.append([i, closest_idx])
                distances.append(closest_dist)

        correspondences = (
            np.array(correspondences) if correspondences else np.array([]).reshape(0, 2)
        )
        distances = np.array(distances) if distances else np.array([])

        return correspondences, distances

    def _reject_outliers(
        self, correspondences: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        """
        Reject outlier correspondences based on distance statistics.

        Args:
            correspondences: Correspondence pairs (N, 2)
            distances: Correspondence distances (N,)

        Returns:
            Valid correspondences after outlier rejection
        """
        if len(distances) == 0:
            return correspondences

        # Statistical outlier rejection
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + self.outlier_rejection_threshold * std_dist

        valid_mask = distances <= threshold
        return correspondences[valid_mask]

    def _solve_pose_update(
        self, source_points: np.ndarray, target_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for pose update using SVD-based method.

        Args:
            source_points: Source correspondence points (N, 3)
            target_points: Target correspondence points (N, 3)

        Returns:
            Tuple of (rotation matrix, translation vector)
        """
        # Center the point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid

        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered

        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation
        rotation = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T

        # Compute translation
        translation = target_centroid - rotation @ source_centroid

        return rotation, translation
