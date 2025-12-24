"""
Core data interfaces for regenbogen framework.

These interfaces define the standardized data structures used for communication
between nodes in the pipeline.
"""

from .interfaces import (
    BoundingBoxes,
    ErrorMetrics,
    Features,
    Frame,
    Masks,
    ObjectModel,
    PointCloud,
    Pose,
)

__all__ = [
    "BoundingBoxes",
    "ErrorMetrics",
    "Features",
    "Frame",
    "Masks",
    "ObjectModel",
    "PointCloud",
    "Pose",
]
