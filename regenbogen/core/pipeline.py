"""
Pipeline class for regenbogen framework.

The Pipeline orchestrates the execution of nodes in sequence or as a directed graph.
"""

import logging
import time
from typing import Any, Iterator, List, Optional

from ..core.node import Node
from ..utils.rerun_logger import RerunLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipeline:
    """
    Pipeline for orchestrating the execution of processing nodes.

    The pipeline manages the flow of data between nodes and provides
    utilities for debugging, profiling, and visualization.
    """

    def __init__(
        self,
        name: str = "Pipeline",
        enable_rerun_logging: bool = False,
        rerun_recording_name: Optional[str] = None,
        rerun_spawn_viewer: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            name: Name of the pipeline
            enable_rerun_logging: Whether to enable Rerun logging for intermediate results
            rerun_recording_name: Custom name for Rerun recording (defaults to pipeline name)
            rerun_spawn_viewer: Whether to spawn the Rerun viewer
        """
        self.name = name
        self.nodes: List[Node] = []
        self.metadata = {}

        # Initialize Rerun logging
        self.enable_rerun_logging = enable_rerun_logging
        recording_name = (
            rerun_recording_name or f"regenbogen_{name.lower().replace(' ', '_')}"
        )
        self.rerun_logger = RerunLogger(
            recording_name, enabled=enable_rerun_logging, spawn=rerun_spawn_viewer
        )

    def add_node(self, node: Node) -> "Pipeline":
        """
        Add a node to the pipeline.

        Args:
            node: Node to add to the pipeline

        Returns:
            Self for method chaining
        """
        if not isinstance(node, Node):
            raise TypeError(f"Expected Node instance, got {type(node)}")

        self.nodes.append(node)
        logger.debug(f"Added node {node.name} to pipeline {self.name}")
        return self

    def process(self, input_data: Any) -> Any:
        """
        Process input data through all nodes in sequence.

        Args:
            input_data: Input data for the first node

        Returns:
            Output data from the last node
        """
        if not self.nodes:
            raise ValueError("Pipeline has no nodes")

        start_time = time.time()
        current_data = input_data

        logger.info(f"Starting pipeline {self.name} with {len(self.nodes)} nodes")

        # Log input data if Rerun logging is enabled
        if self.enable_rerun_logging:
            self.log_pipeline_graph("pipeline/structure")
            self.rerun_logger.set_time_sequence("frame", 0)
            if hasattr(current_data, "rgb"):
                self.rerun_logger.log_frame(current_data, "frame")

        # Process through each node sequentially
        for i, node in enumerate(self.nodes):
            logger.debug(f"Processing node {i + 1}/{len(self.nodes)}: {node.name}")
            current_data = node(current_data)

            # Log intermediate results if Rerun logging is enabled
            if self.enable_rerun_logging:
                self.rerun_logger.set_time_sequence("frame", i + 1)
                if hasattr(current_data, "rgb"):
                    self.rerun_logger.log_frame(current_data, "frame")

        total_runtime = time.time() - start_time
        logger.info(f"Pipeline {self.name} completed in {total_runtime:.3f}s")

        # Add pipeline metadata
        if hasattr(current_data, "metadata"):
            current_data.metadata = current_data.metadata or {}
            current_data.metadata[f"{self.name}_total_runtime"] = total_runtime
            current_data.metadata[f"{self.name}_node_count"] = len(self.nodes)

            # Log final metadata if Rerun logging is enabled
            if self.enable_rerun_logging:
                self.rerun_logger.log_metadata(
                    current_data.metadata, "pipeline/final_metadata"
                )

        return current_data

    def process_stream(self, input_data: Any = None) -> Iterator[Any]:
        """
        Process a stream of data through the pipeline.

        This method is designed to work with nodes that generate multiple outputs
        (like VideoReaderNode). The first node should return an iterator/generator,
        and each item from that iterator will be processed through the remaining nodes.

        Args:
            input_data: Input data for the first node (can be None for source nodes like VideoReader)

        Yields:
            Output data from the last node for each input item
        """
        if not self.nodes:
            raise ValueError("Pipeline has no nodes")

        logger.info(
            f"Starting stream pipeline {self.name} with {len(self.nodes)} nodes"
        )

        # Get the stream from the first node
        first_node = self.nodes[0]
        remaining_nodes = self.nodes[1:]

        # Process the first node to get the stream
        stream = first_node(input_data)

        if not hasattr(stream, "__iter__"):
            raise ValueError(
                f"First node {first_node.name} must return an iterable for stream processing"
            )

        # Process each item from the stream through remaining nodes
        for item_index, item in enumerate(stream):
            start_time = time.time()
            current_data = item

            # Set timeline for this frame
            if self.enable_rerun_logging:
                self.log_pipeline_graph("pipeline/structure")
                self.rerun_logger.set_time_sequence("frame", item_index)
                self.rerun_logger.log_frame(current_data, "frame")

            # Process through remaining nodes sequentially
            for i, node in enumerate(remaining_nodes):
                logger.debug(
                    f"Processing item {item_index + 1}, node {i + 2}/{len(self.nodes)}: {node.name}"
                )
                current_data = node(current_data)

                if current_data is None:
                    logger.debug(f"Node {node.name} skipped frame (returned None)")
                    break

                if self.enable_rerun_logging:
                    if hasattr(current_data, "rgb"):
                        self.rerun_logger.log_frame(current_data, "frame")
                    logger.debug(f"logging a buffer, frame ids: {[f.idx for f in current_data]}")
                    # if output is an iterable of frames (buffer):
                    if hasattr(current_data, "__iter__") and not isinstance(
                        current_data, (str, bytes)
                    ):
                        for f in current_data:
                            if hasattr(f, "rgb"):
                                self.rerun_logger.log_frame(f, "frame")

            item_runtime = time.time() - start_time
            logger.debug(
                f"Pipeline item {item_index + 1} completed in {item_runtime:.3f}s"
            )

            # Add pipeline metadata
            if hasattr(current_data, "metadata"):
                current_data.metadata = current_data.metadata or {}
                current_data.metadata[f"{self.name}_item_runtime"] = item_runtime
                current_data.metadata[f"{self.name}_item_index"] = item_index
                current_data.metadata[f"{self.name}_node_count"] = len(self.nodes)

            yield current_data

    def process_dataset(self, dataset_path: str, **kwargs) -> List[Any]:
        """
        Process an entire dataset through the pipeline.

        Args:
            dataset_path: Path to the dataset
            **kwargs: Additional parameters for dataset loading

        Returns:
            List of processing results
        """
        # This would be implemented based on specific dataset formats
        raise NotImplementedError("Dataset processing not yet implemented")

    def get_node(self, name: str) -> Optional[Node]:
        """
        Get a node by name.

        Args:
            name: Name of the node to find

        Returns:
            Node if found, None otherwise
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def remove_node(self, name: str) -> bool:
        """
        Remove a node by name.

        Args:
            name: Name of the node to remove

        Returns:
            True if node was removed, False if not found
        """
        for i, node in enumerate(self.nodes):
            if node.name == name:
                del self.nodes[i]
                logger.debug(f"Removed node {name} from pipeline {self.name}")
                return True
        return False

    def clear(self):
        """Clear all nodes from the pipeline."""
        self.nodes.clear()
        logger.debug(f"Cleared all nodes from pipeline {self.name}")

    def log_pipeline_graph(self, entity_path: str = "pipeline_graph"):
        """
        Log the pipeline structure as a graph visualization to Rerun.

        Args:
            entity_path: Entity path for the graph visualization
        """
        if self.enable_rerun_logging:
            self.rerun_logger.log_pipeline_graph(self.nodes, entity_path)

    def __len__(self) -> int:
        """Return the number of nodes in the pipeline."""
        return len(self.nodes)

    def __repr__(self) -> str:
        node_names = [node.name for node in self.nodes]
        return f"Pipeline(name='{self.name}', nodes={node_names})"
