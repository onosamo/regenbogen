"""
🌈 regenbogen 🌈 - Framework for 3D perception pipelines

A modular framework for building 3D perception pipelines with support for
both classical and deep learning approaches to pose estimation and spatial understanding.
"""

__version__ = "0.1.0"

from .core.graph_pipeline import GraphPipeline, execute
from .core.node import Node
from .core.pipeline import Pipeline
from .interfaces import BoundingBoxes, ErrorMetrics, Features, Frame, ObjectModel, Pose

__all__ = [
    "Pipeline",
    "GraphPipeline",
    "execute",
    "Node",
    "Frame",
    "ObjectModel",
    "Features",
    "Pose",
    "BoundingBoxes",
    "ErrorMetrics",
]
