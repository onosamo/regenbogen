"""
Rerun logging utility for regenbogen framework.

This module provides functionality to log intermediate pipeline results to Rerun
for interactive visualization and debugging.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import rerun as rr

from ..interfaces import BoundingBoxes, Features, Frame, Masks, ObjectModel, Pose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerunLogger:
    """
    Logger for pipeline intermediate results using Rerun.

    This class handles logging various data types (images, point clouds, poses, etc.)
    to Rerun for interactive visualization during pipeline execution.
    """

    def __init__(
        self,
        recording_name: str = "regenbogen_pipeline",
        enabled: bool = True,
        spawn: bool = True,
    ):
        """
        Initialize the Rerun logger.

        Args:
            recording_name: Name for the Rerun recording
            enabled: Whether logging is enabled
            spawn: Whether to spawn the Rerun viewer
        """
        import os

        self.recording_name = recording_name
        # Disable viewer spawning in CI environments to avoid connection issues
        self.spawn = spawn and not bool(os.getenv("CI"))
        self._initialized = False
        self.enabled = enabled

    def _ensure_initialized(self):
        """Ensure Rerun is initialized."""
        if not self.enabled:
            return

        if not self._initialized:
            rr.init(self.recording_name, spawn=self.spawn)
            self._initialized = True

    def set_time_sequence(self, timeline_name: str, sequence_number: int):
        """
        Set the time sequence for timeline-based logging.

        Args:
            timeline_name: Name of the timeline (e.g., "frame")
            sequence_number: Sequence number for this point in time
        """
        if not self.enabled:
            return

        self._ensure_initialized()
        rr.set_time(timeline_name, sequence=sequence_number)

    def log_frame(
        self,
        frame: Frame,
        entity_path: str = "frame",
        log_poses: Optional[List[Pose]] = None,
        log_object_ids: Optional[List[int]] = None,
        object_models: Optional[Dict[int, "ObjectModel"]] = None,
    ):
        """
        Log a frame to Rerun with optional ground truth poses and object models.

        Args:
            frame: Frame containing RGB, depth, and intrinsics
            entity_path: Entity path for the frame
            log_poses: Optional list of ground truth poses to visualize
            log_object_ids: Optional list of object IDs corresponding to poses
            object_models: Optional dictionary of object models for visualization
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # support global frame indexing
        if frame.idx is not None:
            rr.set_time(entity_path, sequence=frame.idx)

        # Log RGB image
        if frame.rgb is not None:
            rr.log(f"{entity_path}/rgb", rr.Image(frame.rgb))

        # Log depth image
        if frame.depth is not None:
            rr.log(f"{entity_path}/depth", rr.DepthImage(frame.depth, meter=1.0))

        # Log masks
        if frame.masks is not None:
            self.log_masks(frame.masks, entity_path=entity_path)

        # Log camera intrinsics
        if frame.intrinsics is not None:
            height, width = (
                frame.rgb.shape[:2] if frame.rgb is not None else frame.depth.shape[:2]
            )
            rr.log(
                entity_path,
                rr.Pinhole(
                    resolution=[width, height],
                    focal_length=[frame.intrinsics[0, 0], frame.intrinsics[1, 1]],
                    principal_point=[frame.intrinsics[0, 2], frame.intrinsics[1, 2]],
                ),
            )

        # Log ground truth poses and object models if provided
        if log_poses and log_object_ids and object_models:
            logger.debug(
                f"Logging {len(log_poses)} poses with {len(object_models)} models available"
            )
            for pose, obj_id in zip(log_poses, log_object_ids):
                if obj_id in object_models:
                    logger.debug(f"Logging object {obj_id} at pose")
                    # Create transform from pose
                    transform = np.eye(4)
                    transform[:3, :3] = pose.rotation
                    transform[:3, 3] = pose.translation / 1000.0  # Convert mm to meters

                    # Log object model at the pose location
                    obj_entity_path = f"world/objects/obj_{obj_id:06d}"
                    rr.log(
                        obj_entity_path,
                        rr.Transform3D(
                            mat3x3=transform[:3, :3], translation=transform[:3, 3]
                        ),
                    )

                    # Log the object mesh
                    model = object_models[obj_id]
                    if model.mesh_vertices is not None and model.mesh_faces is not None:
                        vertices = model.mesh_vertices / 1000.0  # Convert mm to meters
                        faces = model.mesh_faces

                        mesh_entity_path = f"{obj_entity_path}/mesh"
                        logger.debug(
                            f"Logging mesh for object {obj_id} with {len(vertices)} vertices at {mesh_entity_path}"
                        )

                        if (
                            hasattr(model, "mesh_normals")
                            and model.mesh_normals is not None
                        ):
                            rr.log(
                                mesh_entity_path,
                                rr.Mesh3D(
                                    vertex_positions=vertices,
                                    triangle_indices=faces,
                                    vertex_normals=model.mesh_normals,
                                ),
                            )
                        else:
                            rr.log(
                                mesh_entity_path,
                                rr.Mesh3D(
                                    vertex_positions=vertices, triangle_indices=faces
                                ),
                            )
                    else:
                        logger.warning(f"Object {obj_id} has no mesh data")
                else:
                    logger.warning(f"Object {obj_id} not found in object_models")
        else:
            if not log_poses:
                logger.debug("No poses provided")
            if not log_object_ids:
                logger.debug("No object IDs provided")
            if not object_models:
                logger.debug("No object models provided")

    def log_object_model(
        self,
        model: ObjectModel,
        entity_path: str = "object_model",
        pose: Optional[Pose] = None,
        scale_factor: float = 1.0,
    ):
        """
        Log an ObjectModel to Rerun with optional pose transformation.

        Args:
            model: ObjectModel containing mesh or pointcloud data
            entity_path: Entity path for logging
            pose: Optional pose to transform the object model
            scale_factor: Scale factor to apply to the model (e.g., 0.001 for mm to meters)
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # Apply pose transformation if provided
        if pose is not None:
            transform = np.eye(4)
            transform[:3, :3] = pose.rotation
            transform[:3, 3] = pose.translation * scale_factor
            rr.log(entity_path, rr.Transform3D(mat3x4=transform[:3, :]))

        # Log mesh if available
        if model.mesh_vertices is not None and model.mesh_faces is not None:
            vertices = model.mesh_vertices * scale_factor

            if hasattr(model, "mesh_normals") and model.mesh_normals is not None:
                rr.log(
                    f"{entity_path}/mesh",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=model.mesh_faces,
                        vertex_normals=model.mesh_normals,
                    ),
                )
            else:
                rr.log(
                    f"{entity_path}/mesh",
                    rr.Mesh3D(
                        vertex_positions=vertices,
                        triangle_indices=model.mesh_faces,
                    ),
                )

        # Log pointcloud if available
        if model.pointcloud is not None:
            points = model.pointcloud * scale_factor
            rr.log(f"{entity_path}/pointcloud", rr.Points3D(points))

    def log_pose(self, pose: Pose, entity_path: str = "pose"):
        """
        Log a Pose object to Rerun.

        Args:
            pose: Pose object containing rotation and translation
            entity_path: Entity path for logging
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # Create transformation matrix
        if pose.rotation.shape == (3, 3):
            # Rotation matrix
            transform = np.eye(4)
            transform[:3, :3] = pose.rotation
            transform[:3, 3] = pose.translation
        elif pose.rotation.shape == (4,):
            # Quaternion [w, x, y, z]
            transform = np.eye(4)
            # Convert quaternion to rotation matrix (simplified)
            q = pose.rotation
            w, x, y, z = q[0], q[1], q[2], q[3]
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )
            transform[:3, :3] = R
            transform[:3, 3] = pose.translation

        rr.log(f"{entity_path}/transform", rr.Transform3D(mat3x4=transform[:3, :]))

        # Log pose scores as text
        if pose.scores:
            score_text = ", ".join([f"{k}: {v:.3f}" for k, v in pose.scores.items()])
            rr.log(f"{entity_path}/scores", rr.TextLog(score_text))

    def log_bounding_boxes(self, boxes: BoundingBoxes, entity_path: str = "detections"):
        """
        Log BoundingBoxes to Rerun.

        Args:
            boxes: BoundingBoxes object containing detection results
            entity_path: Entity path for logging
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # Log 2D bounding boxes
        box_array = []
        labels = []

        for i, (box, score, label) in enumerate(
            zip(boxes.boxes, boxes.scores, boxes.labels)
        ):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            box_array.append([x1, y1, width, height])

            # Create label with class name and score
            class_name = (
                boxes.class_names[label]
                if label < len(boxes.class_names)
                else f"class_{label}"
            )
            labels.append(f"{class_name} ({score:.2f})")

        if box_array:
            rr.log(
                f"{entity_path}/boxes",
                rr.Boxes2D(
                    array=np.array(box_array),
                    array_format=rr.Box2DFormat.XYWH,
                    labels=labels,
                ),
            )

    def log_masks(self, masks: Masks, entity_path: str = "segmentation"):
        """
        Log instance segmentation masks to Rerun.

        Args:
            masks: Masks object containing segmentation results
            entity_path: Entity path for logging
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # Log segmentation masks as an image where each instance has a unique value
        if masks.masks is not None and len(masks.masks) > 0:
            # Create instance segmentation image (H, W) with instance IDs
            # Handle different mask shapes: (N, H, W), (H, W), or higher dimensional (take last 2 dims)
            if masks.masks.ndim >= 3:
                # Take last two dimensions as H, W
                h, w = masks.masks.shape[-2:]
            elif masks.masks.ndim == 2:
                h, w = masks.masks.shape
            else:
                logger.error(f"Unexpected mask shape: {masks.masks.shape}, expected at least 2D")
                return
                
            instance_image = np.zeros((h, w), dtype=np.uint16)

            # Reshape masks to (N, H, W) if needed
            if masks.masks.ndim > 3:
                # Flatten extra dimensions
                num_masks = np.prod(masks.masks.shape[:-2])
                mask_array = masks.masks.reshape(num_masks, h, w)
            elif masks.masks.ndim == 3:
                mask_array = masks.masks
            else:
                mask_array = masks.masks[np.newaxis, ...]

            for i, mask in enumerate(mask_array):
                if masks.labels is not None and i < len(masks.labels):
                    instance_image[mask] = masks.labels[i]
                else:
                    instance_image[mask] = i + 1

            rr.log(
                f"{entity_path}/masks",
                rr.SegmentationImage(instance_image),
            )

        # Log bounding boxes with labels
        if masks.boxes is not None and len(masks.boxes) > 0:
            box_array = []
            labels = []

            for i, (box, score) in enumerate(zip(masks.boxes, masks.scores)):
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                box_array.append([x1, y1, width, height])

                # Create label with instance ID and score
                if masks.labels is not None and i < len(masks.labels):
                    label_idx = masks.labels[i]
                    class_name = (
                        masks.class_names[label_idx]
                        if masks.class_names and label_idx < len(masks.class_names)
                        else f"class_{label_idx}"
                    )
                    labels.append(f"{class_name} ({score:.2f})")
                else:
                    labels.append(f"mask_{i} ({score:.2f})")

            if box_array:
                rr.log(
                    f"{entity_path}/boxes",
                    rr.Boxes2D(
                        array=np.array(box_array),
                        array_format=rr.Box2DFormat.XYWH,
                        labels=labels,
                    ),
                )

    def log_pointcloud(
        self,
        points: np.ndarray,
        entity_path: str = "pointcloud",
        colors: Optional[np.ndarray] = None,
    ):
        """
        Log a pointcloud to Rerun.

        Args:
            points: Point cloud array (N, 3)
            entity_path: Entity path for logging
            colors: Optional colors array (N, 3)
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        if colors is not None:
            rr.log(entity_path, rr.Points3D(points, colors=colors))
        else:
            rr.log(entity_path, rr.Points3D(points))

    def log_features(self, features: Features, entity_path: str = "features"):
        """
        Log Features to Rerun.

        Args:
            features: Features object containing keypoints and descriptors
            entity_path: Entity path for logging
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # Log 2D keypoints
        if features.keypoints is not None:
            rr.log(f"{entity_path}/keypoints_2d", rr.Points2D(features.keypoints))

        # Log 3D keypoints
        if features.keypoints_3d is not None:
            rr.log(f"{entity_path}/keypoints_3d", rr.Points3D(features.keypoints_3d))

        # Log descriptor statistics as text
        if features.descriptors is not None:
            desc_info = f"Descriptors: {features.descriptors.shape}, mean: {features.descriptors.mean():.3f}"
            rr.log(f"{entity_path}/descriptor_info", rr.TextLog(desc_info))

    def log_pipeline_step(self, step_name: str, data: Any, step_index: int):
        """
        Log pipeline step results with automatic type detection.

        Args:
            step_name: Name of the pipeline step
            data: Output data from the pipeline step
            step_index: Index of the step in the pipeline
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # Set timeline for this step using the new API
        rr.set_time("step", sequence=step_index)

        entity_path = f"pipeline/{step_index:02d}_{step_name}"

        # Auto-detect data type and log appropriately
        if isinstance(data, Frame):
            self.log_frame(data, entity_path)
        elif isinstance(data, ObjectModel):
            self.log_object_model(data, entity_path)
        elif isinstance(data, Pose):
            self.log_pose(data, entity_path)
        elif isinstance(data, BoundingBoxes):
            self.log_bounding_boxes(data, entity_path)
        elif isinstance(data, Features):
            self.log_features(data, entity_path)
        elif isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[2] == 3:  # Likely RGB image
                rr.log(f"{entity_path}/image", rr.Image(data))
            elif data.ndim == 2 and len(data.shape) == 2:  # Likely depth or grayscale
                rr.log(f"{entity_path}/depth", rr.DepthImage(data))
            elif data.ndim == 2 and data.shape[1] == 3:  # Likely point cloud
                rr.log(f"{entity_path}/pointcloud", rr.Points3D(data))
        else:
            # Log as text for unknown types
            rr.log(f"{entity_path}/data", rr.TextLog(str(type(data))))

    def log_metadata(self, metadata: Dict[str, Any], entity_path: str = "metadata"):
        """
        Log metadata as text.

        Args:
            metadata: Metadata dictionary
            entity_path: Entity path for logging
        """
        if not self.enabled or not metadata:
            return

        self._ensure_initialized()

        metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        rr.log(entity_path, rr.TextLog(metadata_text))

    def log_pipeline_graph(self, nodes: list, entity_path: str = "pipeline_graph"):
        """
        Log a pipeline structure as a graph visualization to Rerun.

        Args:
            nodes: List of pipeline nodes
            entity_path: Entity path for the graph visualization
        """
        if not self.enabled or not nodes:
            return

        self._ensure_initialized()

        # Create node names and labels
        node_names = [node.name for node in nodes]
        node_labels = [f"{node.name}\n({node.__class__.__name__})" for node in nodes]

        # Create edges for sequential pipeline (each node connects to the next)
        edges = []
        for i in range(len(nodes) - 1):
            edges.append((node_names[i], node_names[i + 1]))

        # Define colors for different node types
        color_map = {
            "VideoReaderNode": [228, 26, 28, 255],  # Red
            "DepthAnythingNode": [55, 126, 184, 255],  # Blue
            "DepthToPointCloudNode": [77, 175, 74, 255],  # Green
            "ICPRefinementNode": [255, 127, 0, 255],  # Orange
            "MeshSamplingNode": [166, 86, 40, 255],  # Brown
            "PartialPointCloudExtractionNode": [247, 129, 191, 255],  # Pink
        }

        # Assign colors based on node types
        colors = []
        radii = []
        for node in nodes:
            node_type = node.__class__.__name__
            colors.append(
                color_map.get(node_type, [153, 153, 153, 255])
            )  # Default gray
            # Make source nodes (like VideoReader) larger
            if "Reader" in node_type or "Input" in node_type:
                radii.append(30)
            else:
                radii.append(20)

        # Log the graph nodes
        rr.log(
            entity_path,
            rr.GraphNodes(
                node_ids=node_names, labels=node_labels, colors=colors, radii=radii
            ),
            static=True,
        )

        # Log the graph edges
        if edges:
            rr.log(
                entity_path,
                rr.GraphEdges(edges=edges, graph_type=rr.GraphType.Directed),
                static=True,
            )

        logger.debug(
            f"Logged pipeline graph with {len(nodes)} nodes and {len(edges)} edges to '{entity_path}'"
        )

    def log_graph_dag(
        self, nodes: list, node_inputs: dict, entity_path: str = "dag_graph"
    ):
        """
        Log a directed acyclic graph (DAG) structure to Rerun showing actual connections.

        Args:
            nodes: List of Node objects
            node_inputs: Dictionary mapping nodes to their input connections
            entity_path: Entity path for the graph visualization
        """
        if not self.enabled:
            return

        self._ensure_initialized()

        # Create node mapping with integer IDs
        node_ids = list(range(len(nodes)))  # Use integer indices as node IDs
        node_labels = [f"{node.name}\n({type(node).__name__})" for node in nodes]

        # Create edges based on actual connections
        edges = []
        for target_node, inputs in node_inputs.items():
            if inputs:
                for input_name, input_ref in inputs.items():
                    if hasattr(input_ref, "__self__"):
                        source_node = input_ref.__self__
                        if source_node in nodes:
                            source_idx = nodes.index(source_node)
                            target_idx = nodes.index(target_node)
                            edges.append((source_idx, target_idx))

        # Color nodes by type
        colors = []
        for node in nodes:
            if "Source" in node.name or not node_inputs.get(node):
                colors.append([0.2, 0.8, 0.2])  # Green for source nodes
            elif any("PointCloud" in node.name for _ in [1]):
                colors.append([0.2, 0.2, 0.8])  # Blue for processing nodes
            else:
                colors.append([0.8, 0.2, 0.2])  # Red for output/computation nodes

        # Set radii based on node importance (more connections = larger)
        radii = []
        for node in nodes:
            connection_count = sum(
                1
                for inputs in node_inputs.values()
                for input_ref in inputs.values()
                if hasattr(input_ref, "__self__") and input_ref.__self__ == node
            )
            radii.append(max(0.3, 0.2 + connection_count * 0.1))

        # Log the graph nodes
        rr.log(
            entity_path,
            rr.GraphNodes(
                node_ids=node_ids, labels=node_labels, colors=colors, radii=radii
            ),
            static=True,
        )

        # Log the graph edges with connection labels
        if edges:
            rr.log(
                entity_path,
                rr.GraphEdges(edges=edges, graph_type=rr.GraphType.Directed),
                static=True,
            )

        logger.debug(
            f"Logged DAG graph with {len(nodes)} nodes and {len(edges)} connections to '{entity_path}'"
        )
