"""
Tests for graph-based pipeline functionality with output attribute connections.
"""

import numpy as np

from regenbogen import GraphPipeline, execute
from regenbogen.core.node import Node


class DummySourceNode(Node):
    """Test source node with no inputs."""

    def __init__(self, value: int = 42, name: str = None, **kwargs):
        self.value = value
        super().__init__(name=name or "DummySource", **kwargs)

    def process(self, input_data=None):
        class Output:
            def __init__(self, value):
                self.result = value

        return Output(self.value)


class AddNode(Node):
    """Test node that adds a value."""

    def __init__(self, amount: int = 10, name: str = None, **kwargs):
        self.amount = amount
        super().__init__(name=name or "Add", **kwargs)

    def process(self, value: int) -> int:
        return value + self.amount


class MultiplyNode(Node):
    """Test node that multiplies by a factor."""

    def __init__(self, factor: int = 2, name: str = None, **kwargs):
        self.factor = factor
        super().__init__(name=name or "Multiply", **kwargs)

    def process(self, value: int) -> int:
        return value * self.factor


def test_graph_pipeline_creation():
    """Test creating a graph pipeline."""
    pipeline = GraphPipeline(name="TestPipeline")
    assert pipeline.name == "TestPipeline"
    assert len(pipeline.nodes) == 0
    print("✓ Graph pipeline creation test passed")


def test_add_node_to_graph():
    """Test adding nodes to graph pipeline."""
    pipeline = GraphPipeline()

    source = DummySourceNode(value=10, name="Source")
    pipeline.add_node(source)

    assert len(pipeline.nodes) == 1
    assert source in pipeline.nodes
    print("✓ Add node test passed")


def test_connect_nodes_with_outputs():
    """Test connecting nodes using output attributes."""
    pipeline = GraphPipeline()

    source = DummySourceNode(value=10, name="Source")
    add = AddNode(amount=5, name="Add")

    pipeline.add_node(source)
    pipeline.add_node(add, inputs={"value": source.outputs.result})

    assert len(pipeline.nodes) == 2
    print("✓ Connect nodes with outputs test passed")


def test_validation_success():
    """Test pipeline validation with valid structure."""
    pipeline = GraphPipeline()

    source = DummySourceNode(name="Source")
    add = AddNode(name="Add")

    pipeline.add_node(source)
    pipeline.add_node(add, inputs={"value": source.outputs.result})

    is_valid, errors = pipeline.validate()
    assert is_valid
    assert len(errors) == 0
    print("✓ Validation success test passed")


def test_validation_missing_connection():
    """Test pipeline validation detects missing connections."""
    pipeline = GraphPipeline()

    source = DummySourceNode(name="Source")
    add = AddNode(name="Add")

    pipeline.add_node(source)
    pipeline.add_node(add)  # No inputs specified

    is_valid, errors = pipeline.validate()
    assert not is_valid
    assert len(errors) > 0
    print("✓ Validation missing connection test passed")


def test_simple_pipeline_execution():
    """Test executing a simple linear pipeline."""
    pipeline = GraphPipeline()

    source = DummySourceNode(value=10, name="Source")
    add = AddNode(amount=5, name="Add")
    multiply = MultiplyNode(factor=3, name="Multiply")

    pipeline.add_node(source)
    pipeline.add_node(add, inputs={"value": source.outputs.result})
    pipeline.add_node(multiply, inputs={"value": add.outputs.result})

    results = pipeline.process()

    assert "Source" in results
    assert "Add" in results
    assert "Multiply" in results
    assert results["Add"] == 15
    assert results["Multiply"] == 45
    print("✓ Simple pipeline execution test passed")


def test_branching_pipeline():
    """Test pipeline with natural branching (output reuse)."""
    pipeline = GraphPipeline()

    source = DummySourceNode(value=10, name="Source")
    add = AddNode(amount=5, name="Add")
    multiply = MultiplyNode(factor=3, name="Multiply")

    pipeline.add_node(source)
    # Both nodes read from same source output
    pipeline.add_node(add, inputs={"value": source.outputs.result})
    pipeline.add_node(multiply, inputs={"value": source.outputs.result})

    results = pipeline.process()

    assert results["Source"].result == 10
    assert results["Add"] == 15
    assert results["Multiply"] == 30
    print("✓ Branching pipeline test passed")


def test_rmse_node():
    """Test RMSE evaluation node with multiple inputs."""
    try:
        from regenbogen.interfaces import PointCloud
        from regenbogen.nodes import RMSENode

        pc1 = PointCloud(points=np.random.randn(100, 3).astype(np.float32))
        pc2 = PointCloud(points=pc1.points + 0.1)

        rmse_node = RMSENode()
        result = rmse_node.process(pc1, pc2)

        assert "rmse" in result.metadata
        assert result.metadata["rmse"] > 0
        print("✓ RMSE node test passed")
    except ImportError:
        print("✓ RMSE node test skipped (optional dependencies not installed)")


def test_output_attribute_reference():
    """Test that output attributes create proper references."""
    source = DummySourceNode(name="Test")
    ref = source.outputs.some_attribute

    assert hasattr(ref, "__self__")
    assert hasattr(ref, "__name__")
    assert ref.__self__ == source
    assert ref.__name__ == "some_attribute"
    print("✓ Output attribute reference test passed")


def test_pytorch_style_connection():
    """Test PyTorch-style chaining."""
    source = DummySourceNode(value=10, name="Source")
    processor = AddNode(amount=5, name="Processor")(source)

    assert processor._upstream_node == source

    result = execute(processor, return_all=True)
    assert result is not None
    assert result["Processor"] == 15
    print("✓ PyTorch-style connection test passed")


def test_explicit_connect():
    """Test explicit .connect() method."""
    source = DummySourceNode(value=20, name="Source")
    processor = AddNode(amount=10, name="Processor")

    source.outputs.result.connect(processor.inputs.value)

    assert processor.inputs._refs["value"]._source is not None

    result = execute(processor, return_all=True)
    assert result is not None
    assert result["Processor"] == 30
    print("✓ Explicit connect test passed")


if __name__ == "__main__":
    print("Running graph pipeline tests...\n")

    test_graph_pipeline_creation()
    test_add_node_to_graph()
    test_connect_nodes_with_outputs()
    test_validation_success()
    test_validation_missing_connection()
    test_simple_pipeline_execution()
    test_branching_pipeline()
    test_rmse_node()
    test_output_attribute_reference()
    test_pytorch_style_connection()
    test_explicit_connect()

    print("\n🎉 All graph pipeline tests passed!")
