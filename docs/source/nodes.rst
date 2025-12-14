Available Nodes
===============

This page lists all the implemented nodes in the regenbogen framework. Each node is a self-contained processing unit that can be combined in pipelines.

For detailed API documentation, see the :doc:`api` page.

Implemented Nodes
-----------------

Data Loading & I/O
~~~~~~~~~~~~~~~~~~

**BOPDatasetNode**
   Loads and streams data from BOP (Benchmark for 6D Object Pose Estimation) format datasets with automatic downloading.
   
   - **Input**: None (source node)
   - **Output**: ``(Frame, List[Pose], List[int])`` - Frame with ground truth poses and object IDs
   - **Implementation**: `regenbogen/nodes/bop_dataset.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/bop_dataset.py>`_

**VideoReaderNode**
   Reads and streams frames from video files (MP4, AVI, etc.) using OpenCV.
   
   - **Input**: None (source node)
   - **Output**: ``Frame`` - Stream of video frames
   - **Implementation**: `regenbogen/nodes/video_reader.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/video_reader.py>`_

Depth Estimation
~~~~~~~~~~~~~~~~

**DepthAnythingNode**
   Monocular depth estimation using Depth Anything V1 model from HuggingFace.
   
   - **Input**: ``Frame`` with RGB image
   - **Output**: ``Frame`` with depth map added
   - **Implementation**: `regenbogen/nodes/depth_anything.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/depth_anything.py>`_

**DepthAnything3Node**
   Monocular depth estimation with visual odometry using Depth Anything V3 (fork with optional pycolmap).
   
   - **Input**: ``Frame`` with RGB image (or stream of frames)
   - **Output**: ``Frame`` or ``List[Frame]`` with depth and camera poses
   - **Implementation**: `regenbogen/nodes/depth_anything3.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/depth_anything3.py>`_

Segmentation
~~~~~~~~~~~~

**SAM2Node**
   Automatic instance segmentation using Segment Anything Model 2 (SAM2) from Meta.
   
   - **Input**: ``Frame`` with RGB image (or tuple from dataset loader)
   - **Output**: ``Masks`` with segmentation masks, bounding boxes, and confidence scores
   - **Implementation**: `regenbogen/nodes/sam2.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/sam2.py>`_

Feature Extraction
~~~~~~~~~~~~~~~~~~

**Dinov2Node**
   Dense feature extraction using DINOv2 vision transformer with patch-level and global descriptors.
   
   - **Input**: ``Frame`` with RGB image
   - **Output**: ``Features`` with dense or global feature descriptors
   - **Implementation**: `regenbogen/nodes/dinov2.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/dinov2.py>`_

**TemplateDescriptorNode**
   Extracts and manages descriptors from template images for CNOS-style matching.
   
   - **Input**: ``Frame`` with templates
   - **Output**: ``Features`` with template descriptors
   - **Implementation**: `regenbogen/nodes/template_descriptor.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/template_descriptor.py>`_

Point Cloud Processing
~~~~~~~~~~~~~~~~~~~~~~

**DepthToPointCloudNode**
   Converts depth images and camera intrinsics to 3D point clouds.
   
   - **Input**: ``Frame`` with depth and intrinsics (or stream/list of frames)
   - **Output**: ``Frame`` with point cloud and colors added
   - **Implementation**: `regenbogen/nodes/depth_to_pointcloud.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/depth_to_pointcloud.py>`_

**PartialPointCloudExtractionNode**
   Extracts point cloud regions corresponding to detected objects using bounding boxes.
   
   - **Input**: ``(Frame, BoundingBoxes)`` - Frame with point cloud and object detections
   - **Output**: ``List[Frame]`` with partial point clouds per object
   - **Implementation**: `regenbogen/nodes/partial_pointcloud_extraction.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/partial_pointcloud_extraction.py>`_

**MeshSamplingNode**
   Samples points from 3D mesh models to create reference point clouds.
   
   - **Input**: ``ObjectModel`` with mesh data
   - **Output**: ``ObjectModel`` with sampled point cloud added
   - **Implementation**: `regenbogen/nodes/mesh_sampling.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/mesh_sampling.py>`_

Pose Estimation & Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CNOSMatcherNode**
   CNOS-style template matching for 6D pose estimation using feature descriptors.
   
   - **Input**: ``(Frame, Features)`` - Query frame and template features
   - **Output**: ``BoundingBoxes`` with detected objects and initial poses
   - **Implementation**: `regenbogen/nodes/cnos_matcher.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/cnos_matcher.py>`_

**ICPRefinementNode**
   Refines pose estimates using ICP (Iterative Closest Point) alignment.
   
   - **Input**: ``(ObjectModel, PointCloud, Pose)`` - Model, scene, and initial pose
   - **Output**: ``Pose`` - Refined pose estimate
   - **Implementation**: `regenbogen/nodes/icp_refinement.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/icp_refinement.py>`_

**SphericalPoseGeneratorNode**
   Generates camera poses evenly distributed on a sphere around an object.
   
   - **Input**: None (generator node)
   - **Output**: ``Iterator[Pose]`` - Camera poses on sphere
   - **Implementation**: `regenbogen/nodes/spherical_pose_generator.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/spherical_pose_generator.py>`_

Rendering
~~~~~~~~~

**TemplateRendererNode**
   Renders RGB-D templates from 3D mesh models using pyrender for offline rendering.
   
   - **Input**: ``(ObjectModel, Iterator[Pose])`` - Mesh and camera poses
   - **Output**: ``Iterator[Frame]`` - Rendered RGB-D frames
   - **Implementation**: `regenbogen/nodes/template_renderer.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/template_renderer.py>`_

Evaluation
~~~~~~~~~~

**RMSEEvaluationNode**
   Evaluates pose estimation accuracy using RMSE metrics.
   
   - **Input**: ``(List[Pose], List[Pose])`` - Predicted and ground truth poses
   - **Output**: ``ErrorMetrics`` with RMSE statistics
   - **Implementation**: `regenbogen/nodes/rmse_evaluation.py <https://github.com/onosamo/regenbogen/blob/main/regenbogen/nodes/rmse_evaluation.py>`_

Usage Examples
--------------

BOP Dataset Loading
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from regenbogen.nodes import BOPDatasetNode
   
   # Create BOP dataset loader for YCB-Video with automatic download
   bop_loader = BOPDatasetNode(
       dataset_name="ycbv",  # Dataset name
       split="test",
       scene_id=0,  # Load only scene 0
       max_samples=10,  # Load only first 10 samples
       load_depth=True,
       load_models=True,
       allow_download=True,  # Automatically download if not found
       name="YCB_Loader"
   )
   
   # Stream samples
   for frame, gt_poses, obj_ids in bop_loader.stream_dataset():
       print(f"Frame: {frame.rgb.shape}")
       print(f"Objects: {obj_ids}")
       for pose in gt_poses:
           print(f"  Rotation:\n{pose.rotation}")
           print(f"  Translation: {pose.translation}")
   
   # Get a specific sample
   result = bop_loader.get_sample(scene_id=0, image_id=5)
   if result is not None:
       frame, gt_poses, obj_ids = result
       print(f"Loaded sample: scene={frame.metadata['scene_id']}, image={frame.metadata['image_id']}")
   
   # Load 3D object models
   models = bop_loader.get_object_models()
   for obj_id, model in models.items():
       print(f"Model {obj_id}: {model.name}")
       print(f"  Vertices: {model.mesh_vertices.shape}")

Basic Node Usage
~~~~~~~~~~~~~~~~

.. code-block:: python

   from regenbogen.nodes import DepthAnythingNode
   from regenbogen.interfaces import Frame
   
   # Create node
   depth_node = DepthAnythingNode(model_type="vits")
   
   # Process frame
   result = depth_node.process(input_frame)

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from regenbogen import Pipeline
   from regenbogen.nodes import DepthAnythingNode, DepthToPointCloudNode
   
   # Create pipeline
   pipeline = Pipeline(name="Depth_Pipeline")
   pipeline.add_node(DepthAnythingNode())
   pipeline.add_node(DepthToPointCloudNode())
   
   # Process data
   result = pipeline.process(input_frame)

Template Rendering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from regenbogen.nodes import (
       BOPDatasetNode,
       SphericalPoseGeneratorNode,
       TemplateRendererNode,
   )
   
   # Load object model from BOP dataset
   bop_loader = BOPDatasetNode(
       dataset_name="ycbv",
       load_models=True,
       allow_download=True
   )
   object_model = bop_loader.get_object_model(object_id=1)
   
   # Generate camera poses on a sphere
   pose_generator = SphericalPoseGeneratorNode(
       radius=0.5,  # 0.5 meters from object
       elevation_levels=3,
       azimuth_samples=8,
       inplane_rotations=1
   )
   poses = pose_generator.generate_poses()
   
   # Render RGB-D templates
   renderer = TemplateRendererNode(
       width=640,
       height=480,
       fx=640,
       fy=640
   )
   templates = renderer.render_templates(object_model, poses)
   
   # Process templates
   for i, frame in enumerate(templates):
       print(f"Template {i}:")
       print(f"  RGB shape: {frame.rgb.shape}")
       print(f"  Depth shape: {frame.depth.shape}")
       # Save or use templates for pose estimation

For complete examples, see the `examples/` directory in the repository.
