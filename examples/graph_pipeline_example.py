"""
New graph pipeline example using output attributes and natural branching.

Demonstrates:
1. Loading synthetic data
2. Natural branching - multiple nodes reading from same output
3. Computing RMSE between two PointClouds
"""

import numpy as np

from regenbogen import GraphPipeline
from regenbogen.core.node import Node
from regenbogen.interfaces import Frame, ObjectModel, PointCloud
from regenbogen.nodes import RMSENode


class SyntheticDataSource(Node):
    """Source node that generates synthetic frame and model data."""

    def __init__(self, num_samples: int = 3, name: str = None, **kwargs):
        self.num_samples = num_samples
        super().__init__(name=name or "SyntheticSource", **kwargs)

    def process(self, input_data=None):
        """Generate synthetic data stream."""
        for i in range(self.num_samples):
            rgb = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            depth = np.random.uniform(1.0, 5.0, (240, 320)).astype(np.float32)
            intrinsics = np.array(
                [[200, 0, 160], [0, 200, 120], [0, 0, 1]], dtype=np.float64
            )

            frame = Frame(rgb=rgb, depth=depth, intrinsics=intrinsics)

            vertices = np.random.randn(1000, 3).astype(np.float32)
            faces = np.random.randint(0, 1000, (500, 3), dtype=np.int32)
            model = ObjectModel(
                mesh_vertices=vertices, mesh_faces=faces, name=f"object_{i}"
            )

            class Output:
                def __init__(self, frame, model):
                    self.frame = frame
                    self.model = model

            yield Output(frame, model)


class FrameToPointCloudNode(Node):
    """Convert frame depth to PointCloud."""

    def process(self, frame: Frame) -> Frame:
        """Extract frame and convert depth to PointCloud, return in Frame.pointcloud."""
        if frame.depth is None or frame.intrinsics is None:
            points = np.random.randn(1000, 3).astype(np.float32)
            frame.pointcloud = PointCloud(points=points)
            return frame

        h, w = frame.depth.shape
        fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
        cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = frame.depth.flatten()

        valid_mask = (depth_flat > 0.1) & (depth_flat < 10.0)
        u_valid = u_flat[valid_mask][:1000]
        v_valid = v_flat[valid_mask][:1000]
        depth_valid = depth_flat[valid_mask][:1000]

        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid

        points = np.stack([x, y, z], axis=1).astype(np.float32)
        frame.pointcloud = PointCloud(points=points)
        return frame


class ModelToPointCloudNode(Node):
    """Sample PointCloud from model mesh."""

    def __init__(self, num_points: int = 1000, name: str = None, **kwargs):
        self.num_points = num_points
        super().__init__(name=name or "ModelToPointCloud", **kwargs)

    def process(self, model: ObjectModel) -> PointCloud:
        """Extract model and sample PointCloud."""
        if model.mesh_vertices is None or model.mesh_faces is None:
            return PointCloud(
                points=np.random.randn(self.num_points, 3).astype(np.float32)
            )

        faces = model.mesh_faces
        vertices = model.mesh_vertices

        face_indices = np.random.choice(len(faces), size=self.num_points)

        sampled_points = []
        for face_idx in face_indices:
            face = faces[face_idx]
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            r1, r2 = np.random.random(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2

            point = r1 * v0 + r2 * v1 + r3 * v2
            sampled_points.append(point)

        return PointCloud(points=np.array(sampled_points, dtype=np.float32))


def create_new_graph_pipeline():
    """
    Create pipeline with natural branching (no split/merge nodes).

    Pipeline structure:
                   ┌─> FrameToPointCloud (reads frame) ──┐
    Source ────────┤                                      ├─> RMSE
                   └─> ModelToPointCloud (reads model) ──┘
    """
    pipeline = GraphPipeline(name="NaturalBranchingPipeline")

    source = SyntheticDataSource(num_samples=3, name="Source")
    pipeline.add_node(source)

    frame_to_pc = FrameToPointCloudNode(name="FrameToPointCloud")
    pipeline.add_node(frame_to_pc, inputs={"frame": source.outputs.frame})

    model_to_pc = ModelToPointCloudNode(num_points=1000, name="ModelToPointCloud")
    pipeline.add_node(model_to_pc, inputs={"model": source.outputs.model})

    rmse = RMSENode(name="RMSE")
    pipeline.add_node(
        rmse,
        inputs={
            "predicted": frame_to_pc.outputs.result,
            "ground_truth": model_to_pc.outputs.result,
        },
    )

    return pipeline


def run_example():
    """Run the new graph pipeline example."""
    print("🌈 Regenbogen New Graph Pipeline Example 🌈")
    print("=" * 60)
    print("\nDemonstrating natural branching without utility nodes")
    print("Multiple nodes can read from the same output\n")

    pipeline = create_new_graph_pipeline()

    print(f"Pipeline: {pipeline}")
    print(f"Nodes: {[n.name for n in pipeline.nodes]}\n")

    is_valid, errors = pipeline.validate()
    print(f"Validation: {'✓ Passed' if is_valid else '✗ Failed'}")
    if not is_valid:
        for error in errors:
            print(f"  - {error}")
        return

    print("\nProcessing streaming data...")
    try:
        for i, results in enumerate(pipeline.process_stream()):
            print(f"\n--- Sample {i + 1} ---")

            if "FrameToPointCloud" in results:
                frame_result = results["FrameToPointCloud"]
                if isinstance(frame_result, Frame) and frame_result.pointcloud:
                    print(f"  Frame PointCloud: {frame_result.pointcloud.points.shape}")

            if "ModelToPointCloud" in results:
                pc_result = results["ModelToPointCloud"]
                if isinstance(pc_result, PointCloud):
                    print(f"  Model PointCloud: {pc_result.points.shape}")

            if "RMSE" in results:
                metrics = results["RMSE"]
                print(f"  RMSE: {metrics.metadata.get('rmse', 'N/A'):.4f}")
                print(f"  Mean Error: {metrics.metadata.get('mean_error', 'N/A'):.4f}")

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_example()
