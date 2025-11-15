"""
Example demonstrating dynamic graph construction without explicit Pipeline.

Shows three different connection styles:
1. Dynamic chaining: node2 = Node2()(node1)
2. Explicit connect method: source.outputs.frame.connect(processor.inputs.frame)
3. Static connect: Node.connect(source.outputs.frame, processor.inputs.frame)
"""

import numpy as np

from regenbogen import Node, execute
from regenbogen.interfaces import Frame


class SourceNode(Node):
    """Source node generating synthetic data."""

    def __init__(self, num_frames: int = 3, name: str = None, **kwargs):
        self.num_frames = num_frames
        super().__init__(name=name or "Source", **kwargs)

    def process(self, input_data=None):
        """Generate frames."""
        for i in range(self.num_frames):
            rgb = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            depth = np.random.uniform(1.0, 5.0, (240, 320)).astype(np.float32)
            intrinsics = np.array(
                [[200, 0, 160], [0, 200, 120], [0, 0, 1]], dtype=np.float64
            )
            yield Frame(rgb=rgb, depth=depth, intrinsics=intrinsics)


class ProcessorNode(Node):
    """Processor that extracts data from frame."""

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name=name or "Processor", **kwargs)

    def process(self, frame: Frame) -> np.ndarray:
        """Convert depth to simple pointcloud."""
        h, w = frame.depth.shape
        points = np.random.randn(min(h * w // 100, 1000), 3).astype(np.float32)
        return points


class AggregatorNode(Node):
    """Aggregates results from multiple inputs."""

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name=name or "Aggregator", **kwargs)

    def process(self, pc1: np.ndarray, pc2: np.ndarray) -> dict:
        """Combine pointclouds."""
        return {
            "pc1_size": len(pc1),
            "pc2_size": len(pc2),
            "total_points": len(pc1) + len(pc2),
        }


def example_dynamic_chaining():
    """Example 1: Dynamic chaining."""
    print("=" * 70)
    print("Example 1: Dynamic chaining")
    print("=" * 70)

    source = SourceNode(num_frames=2, name="source")
    processor = ProcessorNode(name="processor")(source)

    print("Created nodes with chaining: processor = ProcessorNode()(source)")
    print(f"Source: {source}")
    print(f"Processor: {processor}")
    print(f"Processor connected to: {processor._upstream_node}")

    print("\nExecuting without explicit Pipeline...")
    for i, results in enumerate(execute(processor, return_all=True)):
        print(f"\nFrame {i + 1}:")
        print(f"  Pointcloud shape: {results['processor'].shape}")


def example_explicit_connect():
    """Example 2: Explicit connect method."""
    print("\n" + "=" * 70)
    print("Example 2: Explicit connect with .connect() method")
    print("=" * 70)

    source = SourceNode(num_frames=2, name="source")
    proc1 = ProcessorNode(name="proc1")
    proc2 = ProcessorNode(name="proc2")

    source.outputs.frame.connect(proc1.inputs.frame)
    source.outputs.frame.connect(proc2.inputs.frame)

    print("Connected using: source.outputs.frame.connect(proc1.inputs.frame)")
    print(f"Source: {source}")
    print(f"Processor 1: {proc1}")
    print(f"Processor 2: {proc2}")

    print("\nBoth processors read from same source (natural branching)")
    print("Executing without explicit Pipeline...")

    for i, results in enumerate(execute(proc1, proc2, return_all=True)):
        print(f"\nFrame {i + 1}:")
        print(f"  Proc1 pointcloud: {results['proc1'].shape}")
        print(f"  Proc2 pointcloud: {results['proc2'].shape}")


def example_static_connect():
    """Example 3: Static connect method."""
    print("\n" + "=" * 70)
    print("Example 3: Static Node.connect() method")
    print("=" * 70)

    source = SourceNode(num_frames=2, name="source")
    processor = ProcessorNode(name="processor")

    Node.connect(source.outputs.frame, processor.inputs.frame)

    print("Connected using: Node.connect(source.outputs.frame, processor.inputs.frame)")
    print(f"Source: {source}")
    print(f"Processor: {processor}")

    print("\nExecuting without explicit Pipeline...")
    for i, results in enumerate(execute(processor, return_all=True)):
        print(f"\nFrame {i + 1}:")
        print(f"  Pointcloud shape: {results['processor'].shape}")


def example_mixed_connections():
    """Example 4: Mix different connection styles."""
    print("\n" + "=" * 70)
    print("Example 4: Mixed connection styles")
    print("=" * 70)

    source = SourceNode(num_frames=1, name="source")

    proc1 = ProcessorNode(name="proc1")(source)

    proc2 = ProcessorNode(name="proc2")
    source.outputs.frame.connect(proc2.inputs.frame)

    aggregator = AggregatorNode(name="agg")
    proc1.outputs.pointcloud.connect(aggregator.inputs.pc1)
    proc2.outputs.pointcloud.connect(aggregator.inputs.pc2)

    print("Mixed styles:")
    print("  - proc1 uses dynamic chaining")
    print("  - proc2 uses explicit .connect()")
    print("  - aggregator uses explicit .connect() for multiple inputs")

    print("\nExecuting...")
    results_list = list(execute(aggregator, return_all=True))
    if results_list:
        results = results_list[0]
        print(f"\nResults: {results['agg']}")


def main():
    print("\n🌈 Regenbogen Dynamic Graph Construction Examples 🌈\n")

    example_dynamic_chaining()
    example_explicit_connect()
    example_static_connect()
    example_mixed_connections()

    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
