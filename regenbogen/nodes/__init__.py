"""Nodes for the regenbogen framework."""

from .bop_dataset import BOPDatasetNode
from .cnos_matcher import CNOSMatcherNode
from .depth_to_pointcloud import DepthToPointCloudNode
from .icp_refinement import ICPRefinementNode
from .mesh_sampling import MeshSamplingNode
from .partial_pointcloud_extraction import PartialPointCloudExtractionNode
from .rmse_evaluation import RMSENode
from .spherical_pose_generator import SphericalPoseGeneratorNode
from .template_descriptor import TemplateDescriptor, TemplateDescriptorNode
from .template_renderer import TemplateRendererNode

__all__ = [
    "DepthToPointCloudNode",
    "PartialPointCloudExtractionNode",
    "MeshSamplingNode",
    "ICPRefinementNode",
    "BOPDatasetNode",
    "RMSENode",
    "SphericalPoseGeneratorNode",
    "TemplateRendererNode",
    "TemplateDescriptorNode",
    "TemplateDescriptor",
    "CNOSMatcherNode",
]

try:
    from .depth_anything import DepthAnythingNode  # noqa: F401

    __all__.append("DepthAnythingNode")
except ImportError:
    pass

try:
    from .video_reader import VideoReaderNode  # noqa: F401

    __all__.append("VideoReaderNode")
except ImportError:
    pass

try:
    from .sam2 import SAM2Node  # noqa: F401

    __all__.append("SAM2Node")
except ImportError:
    pass

try:
    from .dinov2 import Dinov2Node  # noqa: F401

    __all__.append("Dinov2Node")
except ImportError:
    pass
