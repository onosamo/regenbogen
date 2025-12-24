"""
Core data interfaces for regenbogen framework.

These interfaces define the standardized data structures used for communication
between nodes in the pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class ObjectModel:
    """
    Object model interface containing mesh or reference pointcloud data.

    Attributes:
        mesh_vertices: Mesh vertices as numpy array (N, 3) or None
        mesh_faces: Mesh faces as numpy array (M, 3) or None
        mesh_normals: Mesh normals as numpy array (N, 3) or None
        pointcloud: PointCloud instance or None
        name: Object name/identifier
        metadata: Additional metadata dictionary
    """

    mesh_vertices: Optional[npt.NDArray[np.float32]] = None
    mesh_faces: Optional[npt.NDArray[np.int32]] = None
    mesh_normals: Optional[npt.NDArray[np.float32]] = None
    pointcloud: Optional["PointCloud"] = None
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Features:
    """
    Features interface containing descriptors and embeddings.

    Attributes:
        descriptors: Feature descriptors as numpy array or None
        keypoints: 2D keypoints as numpy array (N, 2) or None
        keypoints_3d: 3D keypoints as numpy array (N, 3) or None
        embeddings: Feature embeddings as numpy array or None
        metadata: Additional metadata dictionary
    """

    descriptors: Optional[npt.NDArray[np.float32]] = None
    keypoints: Optional[npt.NDArray[np.float32]] = None
    keypoints_3d: Optional[npt.NDArray[np.float32]] = None
    embeddings: Optional[npt.NDArray[np.float32]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose:
    """
    Pose interface containing rotation, translation and confidence scores.

    Attributes:
        rotation: Rotation matrix (3, 3) or quaternion (4,)
        translation: Translation vector (3,)
        scores: Confidence scores dictionary
        metadata: Additional metadata dictionary
    """

    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundingBoxes:
    """
    Bounding boxes interface for object detection results.

    Attributes:
        boxes: Bounding boxes as numpy array (N, 4) in [x1, y1, x2, y2] format
        scores: Confidence scores as numpy array (N,)
        labels: Class labels as numpy array (N,)
        class_names: List of class names corresponding to labels
        metadata: Additional metadata dictionary
    """

    boxes: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int32]
    class_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointCloud:
    """
    Point cloud interface.

    Attributes:
        points: 3D points as numpy array (N, 3)
        colors: Point colors as numpy array (N, 3) or None
        normals: Point normals as numpy array (N, 3) or None
        metadata: Additional metadata dictionary
    """

    points: npt.NDArray[np.float32]
    colors: Optional[npt.NDArray[np.float32]] = None
    normals: Optional[npt.NDArray[np.float32]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorMetrics:
    """
    Error metrics interface for evaluation results.

    Attributes:
        add: Average Distance Error
        add_s: Average Distance Error - Symmetric
        projection_error: 2D projection error
        runtime: Processing runtime in seconds
        metadata: Additional metadata dictionary
    """

    add: Optional[float] = None
    add_s: Optional[float] = None
    projection_error: Optional[float] = None
    runtime: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Masks:
    """
    Segmentation masks interface for instance segmentation results.

    Attributes:
        masks: Binary segmentation masks as numpy array (N, H, W) where N is number of instances
        boxes: Bounding boxes as numpy array (N, 4) in [x1, y1, x2, y2] format
        scores: Confidence scores as numpy array (N,)
        labels: Class labels as numpy array (N,) or None
        class_names: List of class names corresponding to labels
        metadata: Additional metadata dictionary
    """

    masks: npt.NDArray[np.bool_]
    boxes: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]
    labels: Optional[npt.NDArray[np.int32]] = None
    class_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Frame:
    """
    Frame interface containing RGB, depth, camera intrinsics/extrinsics, and pointcloud data.

    Attributes:
        rgb: RGB image as numpy array (H, W, 3)
        idx: Optional global frame index
        depth: Depth image as numpy array (H, W) or None
        intrinsics: Camera intrinsics matrix (3, 3) or None
        extrinsics: Camera extrinsics matrix (4, 4) or None
        pointcloud: PointCloud instance or None
        metadata: Additional metadata dictionary
    """

    rgb: npt.NDArray[np.uint8]
    idx: Optional[int] = None
    depth: Optional[npt.NDArray[np.float32]] = None
    intrinsics: Optional[npt.NDArray[np.float64]] = None
    extrinsics: Optional[npt.NDArray[np.float64]] = None
    pointcloud: Optional["PointCloud"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    masks: Optional[Masks] = None
