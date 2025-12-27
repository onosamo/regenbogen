# 🌈 regenbogen 🌈

A modular framework for building 3D perception pipelines.

## Quick Start

Install from source:

```bash
git clone git@github.com:onosamo/regenbogen.git
cd regenbogen
uv sync --group full
```

## Examples

Explore complete working examples in the [`examples/`](examples/) directory:

- **[bop_dataset_example.py](examples/bop_dataset_example.py)** - Load and visualize BOP benchmark datasets with automatic download
- **[sam2_bop_example.py](examples/sam2_bop_example.py)** - Instance segmentation with SAM2 on BOP dataset images
- **[sam3_bop_example.py](examples/sam3_bop_example.py)** - Text-prompted segmentation with SAM3 using open-vocabulary concepts
- **[cnos_pipeline_example.py](examples/cnos_pipeline_example.py)** - Complete CNOS pipeline for CAD-based novel object segmentation
- **[template_rendering_example.py](examples/template_rendering_example.py)** - Render RGB-D templates from CAD models for pose estimation
- **[graph_pipeline_example.py](examples/graph_pipeline_example.py)** - Dynamic graph construction with natural branching
- **[pytorch_style_example.py](examples/pytorch_style_example.py)** - PyTorch-style pipeline construction
- **[video_processing_demo.py](examples/video_processing_demo.py)** - Process video files frame-by-frame (downloads sample video if none provided)

Run any example with:
```bash
uv run python examples/<example_name>.py
```

### 🎥 Visualizing Pipeline Results with Rerun

Regenbogen supports interactive visualization of intermediate pipeline results using [Rerun](https://rerun.io). Many examples include built-in Rerun logging - just run them to see the visualization automatically.

## Documentation

📖 **[Complete Documentation](docs/)**

- [Installation Guide](docs/source/installation.rst)
- [Tutorial: Complete 3D Pipeline](docs/source/tutorial.rst)  
- [Available Nodes](docs/source/nodes.rst)
- [System Design](docs/source/system_design.rst)
- [Graph Pipeline Documentation](docs/graph_pipeline.md)

## Current Implementation Status

### ✅ Fully Implemented
- **Core Framework**: Pipeline, Node base classes, standardized interfaces
- **GraphPipeline**: Directed acyclic graph execution with natural branching (output sharing)
- **BOPDatasetNode**: Load and stream BOP benchmark datasets with ground truth poses
- **DepthToPointCloudNode**: Depth to 3D pointcloud conversion
- **PartialPointCloudExtractionNode**: Extract pointclouds from bounding boxes  
- **MeshSamplingNode**: Sample pointclouds from triangle meshes
- **ICPRefinementNode**: Iterative Closest Point pose refinement
- **DepthAnythingNode**: Depth Anything v2 model for mono depth estimation
- **SAM2Node**: Segment Anything Model 2 for instance segmentation with automatic mask generation
- **SAM3Node**: Segment Anything Model 3 with open-vocabulary text-prompted segmentation (270K+ concepts)
- **Dinov2Node**: Dinov2 vision transformer for computing dense feature descriptors
- **VideoReaderNode**: Read frames from video files
- **RMSENode**: Evaluate pointcloud similarity
- **SphericalPoseGeneratorNode**: Generate camera poses on a sphere for template rendering
- **TemplateRendererNode**: Render RGB-D templates from CAD models using pyrender
- **TemplateDescriptorNode**: Compute and store feature descriptors from templates
- **CNOSMatcherNode**: Match query proposals to template descriptors using cosine similarity

### 🚧 Features in progress or planned
- **YOLOv8Node**: Replace synthetic detections with actual YOLOv8 model  
- **CoarsePoseEstimationNode**: Replace synthetic features with Open3D FPFH
- **PlanesFromPointcloudNode**: Find planar surfaces in the scene
- **PointcloudFeaturesNode**: Compute point feature descriptors with PointNet or similar

## Architecture

The framework follows a modular pipeline architecture where each processing step is an independent node that communicates through standardized data interfaces (`Frame`, `ObjectModel`, `Pose`, etc.).

## License

MIT License - see [LICENSE](LICENSE) file for details.
