"""
Graph-based pipeline for directed acyclic graph execution.

Supports complex pipelines with branching and merging using natural output reuse.
Can be used explicitly or connections can be discovered automatically from nodes.
"""

import inspect
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..core.node import InputReference, Node, OutputReference
from ..utils.rerun_logger import RerunLogger

logger = logging.getLogger(__name__)


def execute(*nodes: Node, return_all: bool = False) -> Any:
    """
    Execute a graph of connected nodes without explicit Pipeline.

    Args:
        *nodes: One or more nodes to execute (in any order)
        return_all: If True, return dict of all outputs; if False, return last node's output

    Returns:
        Output from the last node, or dict of all outputs if return_all=True
    """
    import inspect as insp

    pipeline = GraphPipeline.from_nodes(*nodes)

    source_nodes = pipeline._get_source_nodes()

    if source_nodes:
        try:
            source_output = source_nodes[0].process(None)
            # Check if it's a generator (streaming source)
            if source_output is not None and insp.isgenerator(source_output):

                def _stream():
                    for item in source_output:
                        results = pipeline._process_single(item)
                        if return_all:
                            yield results
                        else:
                            yield results[nodes[-1].name] if nodes else None

                return _stream()
        except Exception as e:
            logger.error(f"Error processing source node: {e}")

    # Non-streaming execution
    results = pipeline.process()

    if return_all:
        return results
    else:
        return results[nodes[-1].name] if nodes else None


class GraphPipeline:
    """
    Pipeline supporting directed acyclic graph (DAG) execution.

    Nodes connect by referencing output attributes from other nodes.
    Multiple nodes can read from the same output (natural branching).

    Can be used explicitly or connections can be discovered automatically.

    Example 1 - PyTorch style (no explicit pipeline):
        source = SourceNode(name="src")
        processor = ProcessorNode(name="proc")(source)
        output = execute(processor)

    Example 2 - Explicit connections:
        source = SourceNode(name="src")
        processor = ProcessorNode(name="proc")
        source.outputs.frame.connect(processor.inputs.frame)
        output = execute(processor)

    Example 3 - Using Pipeline class:
        pipeline = GraphPipeline()
        pipeline.add_node(source)
        pipeline.add_node(processor, inputs={"frame": source.outputs.frame})
        results = pipeline.process()
    """

    def __init__(
        self,
        name: str = "GraphPipeline",
        enable_rerun_logging: bool = True,
        rerun_recording_name: Optional[str] = None,
        rerun_spawn_viewer: bool = True,
    ):
        """
        Initialize the graph pipeline.

        Args:
            name: Name of the pipeline
            enable_rerun_logging: Whether to enable Rerun logging for intermediate results
            rerun_recording_name: Custom name for Rerun recording (defaults to pipeline name)
            rerun_spawn_viewer: Whether to spawn the Rerun viewer
        """
        self.name = name
        self.nodes: List[Node] = []
        self._node_inputs: Dict[Node, Dict[str, Any]] = {}
        self._execution_order: Optional[List[Node]] = None
        self.metadata = {}

        # Initialize Rerun logging
        self.enable_rerun_logging = enable_rerun_logging
        recording_name = (
            rerun_recording_name or f"regenbogen_{name.lower().replace(' ', '_')}"
        )
        self.rerun_logger = RerunLogger(
            recording_name, enabled=enable_rerun_logging, spawn=rerun_spawn_viewer
        )

    @classmethod
    def from_nodes(cls, *nodes: Node, name: str = "AutoPipeline") -> "GraphPipeline":
        """
        Create a pipeline from a set of nodes, discovering connections automatically.

        Args:
            *nodes: Nodes to include in pipeline
            name: Pipeline name

        Returns:
            GraphPipeline instance
        """
        pipeline = cls(name=name)

        all_nodes = set()
        for node in nodes:
            pipeline._discover_upstream(node, all_nodes)

        for node in all_nodes:
            inputs = {}

            # Check for explicit connections via .connect() in the _refs cache
            if hasattr(node.inputs, "_refs"):
                for attr_name, input_ref in node.inputs._refs.items():
                    if isinstance(input_ref, InputReference) and input_ref._source:
                        inputs[attr_name] = input_ref._source

            # Check for PyTorch-style connections
            if node._upstream_node and not inputs:
                sig = inspect.signature(node.process)
                params = list(sig.parameters.keys())
                params = [p for p in params if p != "self"]
                required_params = [
                    p
                    for p in params
                    if sig.parameters[p].default == inspect.Parameter.empty
                ]

                if required_params:
                    param_name = required_params[0]
                    output_ref = OutputReference(node._upstream_node, "result")
                    inputs[param_name] = output_ref

            if node._upstream_output and not inputs:
                sig = inspect.signature(node.process)
                params = list(sig.parameters.keys())
                params = [p for p in params if p != "self"]
                required_params = [
                    p
                    for p in params
                    if sig.parameters[p].default == inspect.Parameter.empty
                ]

                if required_params:
                    inputs[required_params[0]] = node._upstream_output

            pipeline.add_node(node, inputs=inputs if inputs else None)

        return pipeline

    def _discover_upstream(self, node: Node, visited: Set[Node]) -> None:
        """Recursively discover all upstream nodes."""
        if node in visited:
            return
        visited.add(node)

        if node._upstream_node:
            self._discover_upstream(node._upstream_node, visited)

        if node._upstream_output:
            self._discover_upstream(node._upstream_output.__self__, visited)

        # Check _refs cache for input connections
        if hasattr(node.inputs, "_refs"):
            for attr_name, input_ref in node.inputs._refs.items():
                if isinstance(input_ref, InputReference) and input_ref._source:
                    self._discover_upstream(input_ref._source.__self__, visited)

    def add_node(
        self, node: Node, inputs: Optional[Dict[str, Any]] = None
    ) -> "GraphPipeline":
        """
        Add a node to the graph pipeline.

        Args:
            node: Node to add
            inputs: Dictionary mapping input parameter names to output attributes
                   from other nodes (e.g., {"frame": bop_node.outputs.frame})

        Returns:
            Self for method chaining
        """
        if node in self.nodes:
            raise ValueError(f"Node '{node.name}' already added to pipeline")

        self.nodes.append(node)

        if inputs is not None:
            self._node_inputs[node] = inputs
        else:
            self._node_inputs[node] = {}

        self._execution_order = None

        logger.debug(f"Added node {node.name} to graph pipeline {self.name}")
        return self

    def _get_source_nodes(self) -> List[Node]:
        """Get nodes with no inputs (source nodes)."""
        return [node for node in self.nodes if not self._node_inputs.get(node)]

    def _build_dependency_graph(self) -> Dict[Node, Set[Node]]:
        """Build a dependency graph showing which nodes depend on which."""
        dependencies = {node: set() for node in self.nodes}

        for node, inputs in self._node_inputs.items():
            for input_value in inputs.values():
                if hasattr(input_value, "__self__"):
                    source_node = input_value.__self__
                    if source_node in self.nodes:
                        dependencies[node].add(source_node)

        return dependencies

    def _topological_sort(self) -> List[Node]:
        """
        Perform topological sort on the graph.

        Returns:
            List of nodes in execution order
        """
        dependencies = self._build_dependency_graph()

        in_degree = {node: len(deps) for node, deps in dependencies.items()}

        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        sorted_nodes = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)

            for dependent, deps in dependencies.items():
                if node in deps:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return sorted_nodes

    def validate(self) -> tuple:
        """
        Validate the pipeline structure.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        source_nodes = self._get_source_nodes()
        if not source_nodes:
            errors.append("No source nodes found (nodes with no inputs)")

        try:
            self._topological_sort()
        except ValueError as e:
            errors.append(str(e))

        for node in self.nodes:
            try:
                sig = inspect.signature(node.process)
                list(sig.parameters.keys())
                required_params = []

                for param_name, param in sig.parameters.items():
                    if (
                        param.default == inspect.Parameter.empty
                        and param_name != "self"
                    ):
                        required_params.append(param_name)

                if node not in self._node_inputs:
                    if required_params:
                        errors.append(
                            f"Node '{node.name}' requires inputs {required_params} but none provided"
                        )
                else:
                    node_inputs = self._node_inputs[node]
                    for param in required_params:
                        if param not in node_inputs:
                            errors.append(
                                f"Node '{node.name}' missing required input '{param}'"
                            )
            except Exception as e:
                errors.append(f"Error validating node '{node.name}': {e}")

        return len(errors) == 0, errors

    def _collect_node_inputs(
        self, node: Node, node_outputs: Dict[Node, Any]
    ) -> Dict[str, Any]:
        """
        Collect inputs for a node from connected outputs.

        Args:
            node: Node to collect inputs for
            node_outputs: Dictionary mapping nodes to their outputs

        Returns:
            Dictionary mapping parameter names to values
        """
        inputs = {}
        node_input_spec = self._node_inputs.get(node, {})

        for param_name, output_ref in node_input_spec.items():
            if hasattr(output_ref, "__self__"):
                source_node = output_ref.__self__
                attr_name = output_ref.__name__

                if source_node not in node_outputs:
                    raise RuntimeError(
                        f"Output from node '{source_node.name}' not yet available"
                    )

                output_data = node_outputs[source_node]

                # Special case: 'result' means pass the whole output
                if attr_name == "result" and not hasattr(output_data, "result"):
                    inputs[param_name] = output_data
                # If output has the requested attribute, use it
                elif hasattr(output_data, attr_name):
                    inputs[param_name] = getattr(output_data, attr_name)
                # Special case: pointcloud attribute for numpy arrays
                elif attr_name == "pointcloud" and isinstance(output_data, np.ndarray):
                    inputs[param_name] = output_data
                else:
                    # Check if it's stored on node.outputs
                    actual_output = getattr(source_node.outputs, attr_name, None)
                    if actual_output is not None and not isinstance(
                        actual_output, OutputReference
                    ):
                        inputs[param_name] = actual_output
                    # If attribute matches the parameter name and output doesn't have it, pass whole output
                    elif attr_name == param_name:
                        inputs[param_name] = output_data
                    else:
                        # As a fallback, just pass the whole output
                        inputs[param_name] = output_data

        return inputs

    def process(self, input_data: Any = None) -> Dict[str, Any]:
        """
        Execute the pipeline on input data.

        Args:
            input_data: Input data for source nodes

        Returns:
            Dictionary mapping node names to their outputs
        """
        is_valid, errors = self.validate()
        if not is_valid:
            raise ValueError(f"Invalid pipeline: {'; '.join(errors)}")

        start_time = time.time()
        logger.info(f"Starting graph pipeline {self.name} with {len(self.nodes)} nodes")

        # Log pipeline structure if Rerun logging is enabled
        if self.enable_rerun_logging:
            self.log_pipeline_graph("pipeline/structure")
            self.rerun_logger.set_time_sequence("frame", 0)

        if self._execution_order is None:
            self._execution_order = self._topological_sort()

        node_outputs: Dict[Node, Any] = {}

        for i, node in enumerate(self._execution_order):
            if not self._node_inputs.get(node):
                output = node(input_data)
            else:
                inputs = self._collect_node_inputs(node, node_outputs)

                sig = inspect.signature(node.process)
                params = list(sig.parameters.keys())
                params = [p for p in params if p != "self"]

                if len(params) == 0:
                    output = node(None)
                elif len(params) == 1:
                    output = node(inputs[params[0]])
                else:
                    # Pass multiple inputs as keyword arguments
                    output = node.process(**inputs)

            node_outputs[node] = output

            # Log intermediate results if Rerun logging is enabled
            if self.enable_rerun_logging:
                self.rerun_logger.set_time_sequence("frame", i + 1)
                if hasattr(output, "rgb"):
                    self.rerun_logger.log_frame(output, f"nodes/{node.name}")
                elif hasattr(output, "points"):
                    # Log PointCloud objects
                    self.rerun_logger.log_pointcloud(
                        output.points,
                        f"nodes/{node.name}/pointcloud",
                        colors=output.colors,
                    )

            # Store output on node.outputs
            if isinstance(output, (np.ndarray, np.generic)):
                # For numpy outputs
                setattr(node.outputs, "pointcloud", output)
            elif isinstance(output, (int, float, str, bool)):
                # For simple scalar outputs
                setattr(node.outputs, "result", output)
            else:
                # For complex outputs with attributes
                for attr_name in dir(output):
                    if not attr_name.startswith("_"):
                        try:
                            attr_value = getattr(output, attr_name)
                            if not callable(attr_value):
                                setattr(node.outputs, attr_name, attr_value)
                        except Exception as e:
                            logger.error(
                                f"Error setting output attribute '{attr_name}' on node '{node.name}': {e}"
                            )

        total_runtime = time.time() - start_time
        logger.info(f"Graph pipeline {self.name} completed in {total_runtime:.3f}s")

        # Add pipeline metadata
        results = {node.name: node_outputs[node] for node in self.nodes}

        # Log final metadata if Rerun logging is enabled
        if self.enable_rerun_logging:
            pipeline_metadata = {
                f"{self.name}_total_runtime": total_runtime,
                f"{self.name}_node_count": len(self.nodes),
                f"{self.name}_execution_order": [
                    node.name for node in self._execution_order
                ],
            }
            self.rerun_logger.log_metadata(pipeline_metadata, "pipeline/final_metadata")

        return results

    def _process_single(self, source_data: Any) -> Dict[str, Any]:
        """Process a single item through the pipeline (for streaming)."""
        if self._execution_order is None:
            self._execution_order = self._topological_sort()

        node_outputs: Dict[Node, Any] = {}

        source_nodes = self._get_source_nodes()
        if source_nodes:
            node_outputs[source_nodes[0]] = source_data

            if isinstance(source_data, (np.ndarray, np.generic)):
                setattr(source_nodes[0].outputs, "pointcloud", source_data)
                setattr(source_nodes[0].outputs, "result", source_data)
            elif isinstance(source_data, (int, float, str, bool)):
                setattr(source_nodes[0].outputs, "result", source_data)
            else:
                for attr_name in dir(source_data):
                    if not attr_name.startswith("_"):
                        try:
                            attr_value = getattr(source_data, attr_name)
                            if not callable(attr_value):
                                setattr(source_nodes[0].outputs, attr_name, attr_value)
                        except Exception as e:
                            logger.error(
                                f"Error setting output attribute '{attr_name}' on node '{source_nodes[0].name}': {e}"
                            )

        remaining_nodes = [n for n in self._execution_order if n not in source_nodes]

        for node in remaining_nodes:
            inputs = self._collect_node_inputs(node, node_outputs)

            sig = inspect.signature(node.process)
            params = list(sig.parameters.keys())
            params = [p for p in params if p != "self"]

            if len(params) == 0:
                output = node._execute(None)
            elif len(params) == 1:
                output = node._execute(inputs[params[0]])
            else:
                output = node.process(**inputs)

            node_outputs[node] = output

            if isinstance(output, (np.ndarray, np.generic)):
                setattr(node.outputs, "pointcloud", output)
                setattr(node.outputs, "result", output)
            elif isinstance(output, (int, float, str, bool)):
                setattr(node.outputs, "result", output)
            else:
                for attr_name in dir(output):
                    if not attr_name.startswith("_"):
                        try:
                            attr_value = getattr(output, attr_name)
                            if not callable(attr_value):
                                setattr(node.outputs, attr_name, attr_value)
                        except Exception as e:
                            logger.error(
                                f"Error setting output attribute '{attr_name}' on node '{node.name}': {e}"
                            )

        return {node.name: node_outputs[node] for node in self.nodes}

    def process_stream(self, input_data: Any = None):
        """
        Process streaming data through the pipeline.

        Args:
            input_data: Input data for source nodes

        Yields:
            Dictionary mapping node names to their outputs for each item
        """
        is_valid, errors = self.validate()
        if not is_valid:
            raise ValueError(f"Invalid pipeline: {'; '.join(errors)}")

        logger.info(f"Starting streaming graph pipeline {self.name}")

        source_nodes = self._get_source_nodes()
        if len(source_nodes) != 1:
            raise ValueError("Streaming mode requires exactly one source node")

        source_node = source_nodes[0]

        if self._execution_order is None:
            self._execution_order = self._topological_sort()

        stream = source_node(input_data)

        if not hasattr(stream, "__iter__"):
            raise ValueError(
                f"Source node '{source_node.name}' must return an iterable"
            )

        remaining_nodes = [n for n in self._execution_order if n != source_node]

        # Log pipeline structure if Rerun logging is enabled
        if self.enable_rerun_logging:
            self.log_pipeline_graph("pipeline/structure")

        for item_index, item in enumerate(stream):
            # Set timeline for this frame
            if self.enable_rerun_logging:
                self.rerun_logger.set_time_sequence("frame", item_index)
                # Log input frame if it has recognizable data
                if hasattr(item, "rgb"):
                    self.rerun_logger.log_frame(item, "frame")

            node_outputs = {source_node: item}

            for attr_name in dir(item):
                if not attr_name.startswith("_"):
                    try:
                        attr_value = getattr(item, attr_name)
                        if not callable(attr_value):
                            setattr(source_node.outputs, attr_name, attr_value)
                    except Exception as e:
                        logger.error(
                            f"Error setting output attribute '{attr_name}' on node '{source_node.name}': {e}"
                        )

            for i, node in enumerate(remaining_nodes):
                inputs = self._collect_node_inputs(node, node_outputs)

                sig = inspect.signature(node.process)
                params = list(sig.parameters.keys())
                params = [p for p in params if p != "self"]

                if len(params) == 0:
                    output = node(None)
                elif len(params) == 1:
                    output = node(inputs[params[0]])
                else:
                    # Pass multiple inputs as keyword arguments
                    output = node.process(**inputs)

                node_outputs[node] = output

                # Log intermediate results if Rerun logging is enabled
                if self.enable_rerun_logging:
                    if hasattr(output, "rgb"):
                        self.rerun_logger.log_frame(output, f"nodes/{node.name}")
                    elif hasattr(output, "points"):
                        # Log PointCloud objects
                        self.rerun_logger.log_pointcloud(
                            output.points,
                            f"nodes/{node.name}/pointcloud",
                            colors=output.colors,
                        )

                # Store output on node.outputs
                if isinstance(output, (np.ndarray, np.generic)):
                    setattr(node.outputs, "pointcloud", output)
                elif isinstance(output, (int, float, str, bool)):
                    setattr(node.outputs, "result", output)
                else:
                    for attr_name in dir(output):
                        if not attr_name.startswith("_"):
                            try:
                                attr_value = getattr(output, attr_name)
                                if not callable(attr_value):
                                    setattr(node.outputs, attr_name, attr_value)
                            except Exception as e:
                                logger.error(
                                    f"Error setting output attribute '{attr_name}' on node '{node.name}': {e}"
                                )

            yield {node.name: node_outputs[node] for node in self.nodes}

    def log_pipeline_graph(self, entity_path: str = "pipeline_graph"):
        """
        Log the pipeline structure as a DAG visualization to Rerun.

        Args:
            entity_path: Entity path for the graph visualization
        """
        if self.enable_rerun_logging:
            self.rerun_logger.log_graph_dag(self.nodes, self._node_inputs, entity_path)

    def __repr__(self):
        return (
            f"GraphPipeline(name='{self.name}', nodes={[n.name for n in self.nodes]})"
        )
