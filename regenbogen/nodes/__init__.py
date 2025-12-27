"""
Nodes for the regenbogen framework.

This module uses lazy importing to avoid loading heavy dependencies
(torch, transformers, open3d, etc.) until they are actually needed.
"""

from __future__ import annotations

import importlib
from typing import Any

# Define all available nodes
__all__ = [
    "BOPDatasetNode",
    "CNOSMatcherNode",
    "DepthToPointCloudNode",
    "ICPRefinementNode",
    "MeshSamplingNode",
    "PartialPointCloudExtractionNode",
    "RMSENode",
    "SphericalPoseGeneratorNode",
    "TemplateDescriptor",
    "TemplateDescriptorNode",
    "TemplateRendererNode",
    # Optional nodes (may not be available if dependencies are missing)
    "DepthAnythingNode",
    "DepthAnything3Node",
    "VideoReaderNode",
    "SAM2Node",
    "SAM3Node",
    "Dinov2Node",
]

# Mapping of node names to their module paths
_NODE_MODULES = {
    "BOPDatasetNode": "regenbogen.nodes.bop_dataset",
    "CNOSMatcherNode": "regenbogen.nodes.cnos_matcher",
    "DepthToPointCloudNode": "regenbogen.nodes.depth_to_pointcloud",
    "ICPRefinementNode": "regenbogen.nodes.icp_refinement",
    "MeshSamplingNode": "regenbogen.nodes.mesh_sampling",
    "PartialPointCloudExtractionNode": "regenbogen.nodes.partial_pointcloud_extraction",
    "RMSENode": "regenbogen.nodes.rmse_evaluation",
    "SphericalPoseGeneratorNode": "regenbogen.nodes.spherical_pose_generator",
    "TemplateDescriptor": "regenbogen.nodes.template_descriptor",
    "TemplateDescriptorNode": "regenbogen.nodes.template_descriptor",
    "TemplateRendererNode": "regenbogen.nodes.template_renderer",
    # Nodes with heavy dependencies
    "DepthAnythingNode": "regenbogen.nodes.depth_anything",
    "DepthAnything3Node": "regenbogen.nodes.depth_anything3",
    "VideoReaderNode": "regenbogen.nodes.video_reader",
    "SAM2Node": "regenbogen.nodes.sam2",
    "SAM3Node": "regenbogen.nodes.sam3",
    "Dinov2Node": "regenbogen.nodes.dinov2",
}

# Cache for loaded modules
_loaded = {}


def __getattr__(name: str) -> Any:
    """
    Lazy import nodes on first access.

    This significantly improves startup time by deferring imports of heavy
    dependencies (torch, transformers, open3d, sklearn) until they're needed.

    Args:
        name: The name of the node class to import

    Returns:
        The requested node class

    Raises:
        AttributeError: If the node name is not recognized
        ImportError: If the node's dependencies are not installed
    """
    if name in _loaded:
        return _loaded[name]

    if name not in _NODE_MODULES:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path = _NODE_MODULES[name]

    try:
        module = importlib.import_module(module_path)
        node_class = getattr(module, name)
        _loaded[name] = node_class
        return node_class
    except ImportError as e:
        # Provide helpful error message for missing optional dependencies
        raise ImportError(
            f"Could not import {name}. "
            f"This may require optional dependencies. "
            f"Original error: {e}"
        ) from e


def __dir__() -> list[str]:
    """Return the list of available nodes for tab completion."""
    return __all__
