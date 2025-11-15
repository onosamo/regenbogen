Available Nodes
===============

This page lists all the implemented nodes in the regenbogen framework.

Core Pipeline Nodes
-------------------

Data Loading
~~~~~~~~~~~~

**BOPDatasetNode**
   Loads and streams data from BOP (Benchmark for 6D Object Pose Estimation) format datasets.
   
   - **Input**: None (dataset loader)
   - **Output**: Iterator yielding (Frame, list of ground truth Poses, list of object IDs)
   - **Parameters**: 
     
     - dataset_name: Name of the BOP dataset (e.g., 'ycbv', 'tless')
     - dataset_path: Path to the BOP dataset root directory (optional, uses cache if None)
     - split: Dataset split to use ('train' or 'test')
     - scene_id: Specific scene ID to load (None = load all scenes)
     - max_samples: Maximum number of samples to load (None = load all)
     - load_depth: Whether to load depth maps
     - load_models: Whether to load 3D object models
     - allow_download: If True, automatically download dataset if not found
     - download_scene_ids: List of scene IDs to download (None = all scenes)
   
   - **Key Features**:
     
     - Supports standard BOP dataset format
     - Automatic dataset downloading with caching (using platformdirs)
     - Streams frames with ground truth poses
     - Loads 3D object models (requires Open3D)
     - Can get specific samples to save bandwidth/disk space
     - Works with both OpenCV and PIL for image loading
   
   - **Status**: Fully implemented

Depth Estimation
~~~~~~~~~~~~~~~~~

**DepthAnythingNode**
   Converts RGB images to monocular depth estimates using the Depth Anything model.
   
   - **Input**: Frame with RGB image
   - **Output**: Frame with depth map added
   - **Parameters**: model_type, device
   - **Status**: Placeholder implementation (generates synthetic depth)

Pointcloud Processing
~~~~~~~~~~~~~~~~~~~~~

**DepthToPointCloudNode**
   Converts depth images and camera intrinsics to 3D pointclouds.
   
   - **Input**: Frame with depth image and intrinsics
   - **Output**: Frame with pointcloud added
   - **Parameters**: max_depth, min_depth
   - **Status**: Fully implemented

**PartialPointCloudExtractionNode**
   Extracts pointcloud regions corresponding to detected objects.
   
   - **Input**: Tuple of (Frame with pointcloud, BoundingBoxes with detections)
   - **Output**: List of partial pointclouds
   - **Parameters**: expand_factor, min_points
   - **Status**: Fully implemented

Object Detection
~~~~~~~~~~~~~~~~

**SAM2Node**
   Performs automatic instance segmentation mask generation using SAM2.
   
   - **Input**: Frame with RGB image
   - **Output**: Masks containing segmentation masks, bounding boxes, and scores
   - **Parameters**: 
     
     - model_size: Size of the model ("tiny", "small", "base-plus", or "large")
     - device: Device to run the model on
     - points_per_batch: Number of points per batch for mask generation
     - pred_iou_thresh: IoU threshold for filtering masks
     - mask_threshold: Threshold for converting interpolated masks to boolean
     - enable_rerun_logging: Whether to enable Rerun visualization
     - rerun_entity_path: Base entity path for Rerun logging
   
   - **Key Features**:
     
     - Automatic mask generation for all objects in image
     - Multiple candidate masks with confidence scores
     - Bounding box extraction from masks
     - Integrated Rerun logging for visualization
   
   - **Status**: Fully implemented

Feature Extraction
~~~~~~~~~~~~~~~~~~

**Dinov2Node**
   Computes dense feature descriptors from RGB images using Dinov2 vision transformer.
   
   - **Input**: Frame with RGB image
   - **Output**: Features containing descriptors and optional embeddings
   - **Parameters**:
     
     - model_size: Size of the model ("small", "base", "large", or "giant")
     - device: Device to run the model on
     - output_type: Type of features ("patch" for dense features, "cls" for global, "both")
     - enable_rerun_logging: Whether to enable Rerun visualization
     - rerun_entity_path: Base entity path for Rerun logging
   
   - **Key Features**:
     
     - Dense patch-level feature descriptors (14x14 pixel patches)
     - Global image embeddings via CLS token
     - Self-supervised features suitable for various downstream tasks
     - PCA visualization of feature maps in Rerun
     - Integrated Rerun logging with feature statistics
   
   - **Status**: Fully implemented

Mesh Processing
~~~~~~~~~~~~~~~

**MeshSamplingNode**
   Samples points from 3D meshes to create reference pointclouds.
   
   - **Input**: ObjectModel with mesh
   - **Output**: ObjectModel with sampled pointcloud added
   - **Parameters**: num_points, sampling_method, add_noise, noise_std
   - **Status**: Fully implemented

Pose Estimation
~~~~~~~~~~~~~~~

**ICPRefinementNode**
   Refines pose estimates using ICP (Iterative Closest Point) alignment algorithms.
   
   - **Input**: Tuple of (ObjectModel, scene pointcloud, initial pose)
   - **Output**: Refined pose estimate
   - **Parameters**: max_iterations, tolerance, max_correspondence_distance, outlier_rejection_threshold
   - **Status**: Fully implemented

Template Rendering
~~~~~~~~~~~~~~~~~~

**SphericalPoseGeneratorNode**
   Generates camera poses evenly distributed on a sphere around an object.
   
   - **Input**: None (pose generator)
   - **Output**: Iterator yielding Pose objects with camera positions on sphere
   - **Parameters**: 
     
     - radius: Distance from object center to camera (in meters)
     - num_views: Total number of views (overrides elevation_levels/azimuth_samples)
     - elevation_levels: Number of elevation levels (latitude circles)
     - azimuth_samples: Number of azimuth samples per elevation level
     - inplane_rotations: Number of in-plane rotations per view
     - look_at_center: Point to look at, default is origin [0, 0, 0]
     - up_vector: Up vector for camera orientation, default is [0, -1, 0]
   
   - **Key Features**:
     
     - Generates evenly distributed viewpoints on a sphere
     - Supports multiple elevation levels and azimuth samples
     - Optional in-plane rotations for more coverage
     - Generates proper look-at matrices for camera orientation
     - Useful for template rendering in pose estimation pipelines
   
   - **Status**: Fully implemented

**TemplateRendererNode**
   Renders RGB-D templates from CAD models (meshes) using pyrender.
   
   - **Input**: Tuple of (ObjectModel with mesh, Iterator[Pose])
   - **Output**: Iterator yielding Frame objects with rendered RGB and depth images
   - **Parameters**:
     
     - width: Image width in pixels (default: 640)
     - height: Image height in pixels (default: 480)
     - fx, fy: Focal lengths in pixels (default: width, height)
     - cx, cy: Principal point coordinates (default: width/2, height/2)
     - z_near: Near clipping plane distance (default: 0.01)
     - z_far: Far clipping plane distance (default: 10.0)
     - ambient_light: Ambient light intensity [0, 1] (default: 0.5)
   
   - **Key Features**:
     
     - Offscreen rendering using pyrender
     - Generates RGB-D images from mesh models
     - Configurable camera intrinsics
     - Outputs Frame objects with full camera parameters
     - Useful for generating templates for CNOS and similar methods
   
   - **Status**: Fully implemented (requires display or EGL support)

Implementation Status
---------------------

**Fully Implemented**
   - BOPDatasetNode
   - DepthToPointCloudNode
   - PartialPointCloudExtractionNode  
   - MeshSamplingNode
   - ICPRefinementNode
   - DepthAnythingNode
   - SAM2Node
   - Dinov2Node
   - VideoReaderNode
   - SphericalPoseGeneratorNode
   - TemplateRendererNode

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
Feature Extraction with Dinov2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from regenbogen.nodes import Dinov2Node
   from regenbogen.interfaces import Frame
   import numpy as np
   
   # Create Dinov2 node for dense feature extraction
   dinov2 = Dinov2Node(
       model_size="small",
       device="cuda",
       output_type="patch",  # Dense patch features
       enable_rerun_logging=True
   )
   
   # Process RGB frame
   rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
   frame = Frame(rgb=rgb)
   
   # Extract features
   features = dinov2.process(frame)
   print(f"Descriptors shape: {features.descriptors.shape}")  # (num_patches, feature_dim)
   print(f"Feature dimension: {features.metadata['feature_dim']}")
   print(f"Number of patches: {features.metadata['num_patches']}")
   
   # Get global image embedding instead
   dinov2_global = Dinov2Node(
       model_size="small",
       device="cuda",
       output_type="cls"  # Global CLS token
   )
   features_global = dinov2_global.process(frame)
   print(f"Global embedding shape: {features_global.descriptors.shape}")  # (feature_dim,)
