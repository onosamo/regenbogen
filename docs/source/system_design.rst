⚙️ System Design — Foundation Principles
========================================

This section outlines the foundational principles that guide the architecture and design of 🌈 regenbogen 🌈. These principles ensure modularity, flexibility, and ease of use while supporting both classical and deep learning approaches to 3D perception.

1. Pipeline = Directed Graph of Nodes
--------------------------------------

A pipeline is a sequence (or DAG) of nodes.

Each node:

* has inputs (well-defined interfaces, e.g. ``Frame``, ``ObjectModel``, ``Features``, ``Pose``)
* produces outputs (same structured interfaces)

This makes it easy to swap nodes, reuse them, or combine classical and deep-learning approaches.

**Benefits:**

* **Modularity**: Each node can be developed, tested, and optimized independently
* **Reusability**: Nodes can be shared across different pipelines
* **Flexibility**: Easy to experiment with different combinations of algorithms
* **Composability**: Classical and ML approaches can be seamlessly integrated

2. Interfaces First
-------------------

All communication between nodes goes through standardized data interfaces:

* **Frame** → RGB, depth, intrinsics/extrinsics, pointcloud
* **ObjectModel** → mesh or reference pointcloud
* **Features** → descriptors, embeddings
* **Pose** → rotation, translation, score(s)
* **ErrorMetrics** → ADD, ADD-S, projection error, runtime

Interfaces are serializable (NumPy, Torch tensors, JSON metadata) so nodes don't need to know about each other's internals.

**Key Advantages:**

* **Decoupling**: Nodes operate independently through well-defined contracts
* **Interoperability**: Different implementations can work together seamlessly
* **Testability**: Each interface can be mocked and tested in isolation
* **Serialization**: Data can be saved, loaded, and transferred between systems

3. Two Execution Backends
--------------------------

**Python nodes** → simple functions, run inline with dependencies (good for ICP, Open3D, lightweight ops).

**Docker nodes** → run in isolated containers (good for heavy ML models, CUDA, special deps).

Same interface, different runtime. Backend choice is transparent to the pipeline.

**Python Backend Benefits:**

* Low overhead for lightweight operations
* Direct access to Python ecosystem (NumPy, SciPy, Open3D)
* Easy debugging and development
* Shared memory between nodes

**Docker Backend Benefits:**

* Isolated environments prevent dependency conflicts
* Support for different CUDA versions and ML frameworks
* Reproducible environments across different systems
* Resource isolation and management

4. Two Modes of Inference
--------------------------

**Offline mode** → run pipeline on a dataset (e.g. folder with images, BOP dataset).

**Online mode** → pipeline subscribes to input sources (camera stream, robot sensors, ROS2 topics).

Both modes reuse the same pipeline definition, only differ in input adapter.

**Offline Mode Features:**

* Batch processing of datasets
* Reproducible results
* Performance benchmarking
* Dataset validation and testing

**Online Mode Features:**

* Real-time processing
* Streaming data support
* Integration with robotic systems
* Live visualization and monitoring

5. Visualization & Debugging
-----------------------------

Each node can emit intermediate results (features, correspondences, pose hypotheses).

Visualization layer integrates with tools like `Rerun <https://rerun.io>`_ or Open3D for interactive inspection.

Debugging = first-class: the framework is not just for final inference, but also for understanding and comparing methods.

**Debugging Capabilities:**

* **Intermediate Results**: Inspect outputs at each pipeline stage
* **Interactive Visualization**: 3D visualization of pointclouds, poses, and matches
* **Performance Profiling**: Timing and resource usage analysis
* **Comparison Tools**: Side-by-side comparison of different approaches

**Supported Visualization Tools:**

* Open3D for 3D geometry visualization
* Rerun for interactive timeline-based debugging
* Matplotlib for 2D plots and metrics
* Custom visualization nodes for specialized needs

6. Extensibility & Modularity
------------------------------

Pipelines can represent:

* **Classical geometry-based approaches** (PPF, RANSAC, ICP)
* **Deep learning methods** (FoundationPose, SAM6D, FreeZe)
* **Hybrid pipelines** (deep features + geometric refinement)

Nodes are pluggable → can be replaced, reordered, or benchmarked independently.

**Extension Points:**

* **Custom Nodes**: Implement new algorithms following interface contracts
* **Node Composition**: Combine existing nodes in novel ways
* **Pipeline Templates**: Reusable pipeline patterns for common tasks
* **Plugin System**: Dynamic loading of node implementations

**Research-Friendly Design:**

* Easy to prototype new methods
* Compare against established baselines
* Gradual integration of research into production
* Support for ablation studies

7. Experimentation & Reproducibility
-------------------------------------

Pipelines are defined directly in Python (instead of static YAML) →

* IDE support, type hints, breakpoints, dynamic graphs
* But still possible to export/import configs (YAML/JSON) for reproducibility

Benchmarking node outputs is possible with a standard ``Evaluator`` node (wrapping BOP toolkit).

**Development Experience:**

* **IDE Integration**: Full Python development environment support
* **Type Safety**: Type hints for interface validation
* **Dynamic Configuration**: Programmatic pipeline construction
* **Interactive Development**: Jupyter notebook support

**Reproducibility Features:**

* **Config Export**: Save pipeline configurations as YAML/JSON
* **Deterministic Execution**: Seeded random number generation
* **Environment Capture**: Docker images for exact environment reproduction
* **Version Tracking**: Git integration for code and data versioning

**Benchmarking Support:**

* **BOP Toolkit Integration**: Standard evaluation metrics
* **Custom Metrics**: Define domain-specific evaluation criteria
* **Automated Testing**: Continuous integration with performance regression detection
* **Comparative Analysis**: Built-in tools for method comparison

8. Minimal Working Example
---------------------------

A simple reference pipeline:

1. **Extract object features**
2. **Extract scene features**
3. **Match & register**
4. **Refine with ICP**
5. **Evaluate**

.. code-block:: python

   from regenbogen import Pipeline, Node
   from regenbogen.nodes import (
       FeatureExtractor, 
       Matcher, 
       PoseEstimator, 
       ICPRefiner, 
       Evaluator
   )
   
   # Create pipeline
   pipeline = Pipeline()
   
   # Add nodes in sequence
   pipeline.add_node(FeatureExtractor(method="sift"))
   pipeline.add_node(Matcher(method="flann"))
   pipeline.add_node(PoseEstimator(method="pnp_ransac"))
   pipeline.add_node(ICPRefiner(max_iterations=100))
   pipeline.add_node(Evaluator(metrics=["add", "add_s"]))
   
   # Process single frame
   result = pipeline.process(input_frame)
   
   # Or process dataset
   results = pipeline.process_dataset(dataset_path)

This example demonstrates the key principles in action:

* **Modular Design**: Each step is a separate, replaceable node
* **Clear Interfaces**: Data flows through standardized formats
* **Flexibility**: Easy to swap algorithms (e.g., change "sift" to "superpoint")
* **Evaluation**: Built-in support for standard metrics

Getting Started
---------------

To begin implementing pipelines with these principles:

1. **Define your interfaces** - Start with the data structures your pipeline needs
2. **Implement core nodes** - Create basic building blocks following interface contracts
3. **Build simple pipelines** - Start with linear sequences before complex DAGs
4. **Add visualization** - Integrate debugging and inspection early
5. **Benchmark and iterate** - Use evaluation nodes to measure and improve performance

For detailed implementation examples, see the :doc:`quickstart` guide and :doc:`api` reference.