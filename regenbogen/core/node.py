"""
Base Node class for regenbogen framework.

All processing nodes inherit from this base class and implement the process method.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class OutputReference:
    """Reference to a node's output attribute."""

    def __init__(self, node: "Node", attr_name: str):
        self.__self__ = node
        self.__name__ = attr_name
        self._connections: List[InputReference] = []

    def connect(self, target: "InputReference") -> "OutputReference":
        """
        Connect this output to an input.

        Args:
            target: Input reference to connect to

        Returns:
            Self for chaining
        """
        if target not in self._connections:
            self._connections.append(target)
            target._source = self
        return self

    def __repr__(self):
        return f"{self.__self__.name}.outputs.{self.__name__}"


class InputReference:
    """Reference to a node's input attribute."""

    def __init__(self, node: "Node", attr_name: str):
        self.__self__ = node
        self.__name__ = attr_name
        self._source: Optional[OutputReference] = None

    def __repr__(self):
        return f"{self.__self__.name}.inputs.{self.__name__}"


class OutputPort:
    """Container for node outputs accessible as attributes."""

    def __init__(self, node: "Node"):
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_refs", {})

    def __getattr__(self, name: str) -> OutputReference:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        refs = object.__getattribute__(self, "_refs")
        if name not in refs:
            refs[name] = OutputReference(object.__getattribute__(self, "_node"), name)
        return refs[name]

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)


class InputPort:
    """Container for node inputs accessible as attributes."""

    def __init__(self, node: "Node"):
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_refs", {})

    def __getattr__(self, name: str) -> InputReference:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        refs = object.__getattribute__(self, "_refs")
        if name not in refs:
            refs[name] = InputReference(object.__getattribute__(self, "_node"), name)
        return refs[name]

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)


class Node(ABC):
    """
    Base class for all processing nodes in the regenbogen framework.

    Each node represents a single processing step in the pipeline and communicates
    through standardized data interfaces.

    Nodes declare their inputs and outputs using type annotations on process() method.

    Example:
        class MyNode(Node):
            def process(self, frame: Frame) -> ObjectModel:
                return model
    """

    def __init__(self, name: str = None, **kwargs):
        """
        Initialize the node.

        Args:
            name: Optional name for the node. If None, uses class name.
            **kwargs: Additional configuration parameters.
        """
        self.name = name or self.__class__.__name__
        self.config = kwargs
        self.metadata = {}
        self.outputs = OutputPort(self)
        self.inputs = InputPort(self)
        self._upstream_node: Optional[Node] = None
        self._upstream_output: Optional[OutputReference] = None

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and return output.

        Args:
            input_data: Input data following framework interfaces

        Returns:
            Output data following framework interfaces
        """
        pass

    def __call__(self, input_data: Any = None) -> "Node":
        """
        PyTorch-style connection: connect this node to an upstream node or output.

        Args:
            input_data: Can be:
                - None: Execute node standalone
                - Node: Connect to all outputs of that node
                - OutputReference: Connect to specific output
                - dict: Explicit input mapping
                - Any: Direct data for execution

        Returns:
            Self for chaining, or output data if executed
        """
        if input_data is None:
            start_time = time.time()
            try:
                output = self.process(None)
                runtime = time.time() - start_time
                if hasattr(output, "metadata"):
                    output.metadata = output.metadata or {}
                    output.metadata[f"{self.name}_runtime"] = runtime
                logger.debug(f"Node {self.name} completed in {runtime:.3f}s")
                return output
            except Exception as e:
                logger.error(f"Error in node {self.name}: {str(e)}")
                raise

        elif isinstance(input_data, Node):
            self._upstream_node = input_data
            return self

        elif isinstance(input_data, OutputReference):
            self._upstream_output = input_data
            return self

        elif isinstance(input_data, dict):
            for param_name, output_ref in input_data.items():
                if isinstance(output_ref, OutputReference):
                    if hasattr(self.inputs, param_name):
                        input_ref = getattr(self.inputs, param_name)
                        output_ref.connect(input_ref)
            return self

        else:
            start_time = time.time()
            try:
                output = self.process(input_data)
                runtime = time.time() - start_time
                if hasattr(output, "metadata"):
                    output.metadata = output.metadata or {}
                    output.metadata[f"{self.name}_runtime"] = runtime
                logger.debug(f"Node {self.name} completed in {runtime:.3f}s")
                return output
            except Exception as e:
                logger.error(f"Error in node {self.name}: {str(e)}")
                raise

    @staticmethod
    def connect(source: OutputReference, target: InputReference) -> None:
        """
        Explicitly connect an output to an input.

        Args:
            source: Output reference
            target: Input reference
        """
        source.connect(target)

    def _execute(self, input_data: Any) -> Any:
        """Internal execution method with timing."""
        start_time = time.time()

        try:
            output = self.process(input_data)

            runtime = time.time() - start_time
            if hasattr(output, "metadata"):
                output.metadata = output.metadata or {}
                output.metadata[f"{self.name}_runtime"] = runtime

            logger.debug(f"Node {self.name} completed in {runtime:.3f}s")

            return output

        except Exception as e:
            logger.error(f"Error in node {self.name}: {str(e)}")
            raise

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
