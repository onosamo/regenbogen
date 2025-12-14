System Design — Foundation Principles
========================================

This section outlines the foundational principles that guide the architecture and design of 🌈 regenbogen 🌈. These principles ensure modularity, flexibility, and ease of use while supporting both classical and deep learning approaches to 3D perception.

1. Pipeline = Directed Graph of Nodes
--------------------------------------

A pipeline is a sequence (or DAG) of nodes.

Each node:

* has inputs (well-defined interfaces, e.g. ``Frame``, ``ObjectModel``, ``Features``, ``Pose``)
* produces outputs (same structured interfaces)

Standard ``Frame`` interface carries RGB, depth, intrinsics/extrinsics, pointcloud and segmentation masks.

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

3. [Future] Two Execution Backends
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

Base pipeline class natively integrates with `Rerun <https://rerun.io>`_ for interactive inspection.

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

**Development Experience:**

* **IDE Integration**: Full Python development environment support
* **Type Safety**: Type hints for interface validation
* **Dynamic Configuration**: Programmatic pipeline construction
* **Interactive Development**: Jupyter notebook support

**[Future] Reproducibility Features:**

* **Config Export**: Save pipeline configurations as YAML/JSON
* **Deterministic Execution**: Seeded random number generation
* **Environment Capture**: Docker images for exact environment reproduction

**Benchmarking Support:**

* **BOP Toolkit Integration**: Standard evaluation metrics
* **Custom Metrics**: Define domain-specific evaluation criteria
* **Automated Testing**: Continuous integration with performance regression detection
* **Comparative Analysis**: Built-in tools for method comparison

For detailed implementation examples, see the :doc:`quickstart` guide and :doc:`api` reference.