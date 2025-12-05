#!/usr/bin/env python3
"""
Demo script showing video processing with Depth Anything V3 for depth and pose estimation.

This script demonstrates:
- Video frame processing with sliding window buffer
- Automatic camera pose estimation from video
- Metric depth estimation
- Pointcloud generation with camera poses
- Rerun visualization of camera trajectory and reconstruction

REQUIREMENTS:
    This example requires Depth Anything V3 to be installed as an optional dependency.

    Recommended (uses fork with optional pycolmap):

        pip install --no-deps git+https://github.com/fedor-chervinskii/Depth-Anything-3.git

    Or with uv:

        uv pip install --no-deps git+https://github.com/fedor-chervinskii/Depth-Anything-3.git

    The fork makes pycolmap and optional (only needed for COLMAP export).
"""

import rerun as rr

from regenbogen import Pipeline
from regenbogen.nodes import DepthAnything3Node, DepthToPointCloudNode, VideoReaderNode


def main(
    video_path: str | None = None,
    max_frames: int | None = None,
    model_name: str = "da3-small",  # Use small model for demo
    buffer_size: int = 3,
    buffer_step: int = -1,
    frame_skip: int = 1,
):
    """
    Demonstrate video processing with Depth Anything V3.

    Args:
        video_path: Path to video file (None to download sample)
        max_frames: Maximum number of frames to process
        model_name: DA3 model variant (da3-small, da3-base, da3-large, da3nested-giant-large)
        buffer_size: Sliding window size for pose estimation
        buffer_step: Number of frames to shift buffer each step (1 for sliding window, -1 for buffer_size-1)
        frame_skip: Skip every N frames (1 = process all frames)
    """

    print("Setting up pipeline...")
    pipeline = Pipeline(
        name="DA3_Video_Pipeline",
        enable_rerun_logging=True,
        rerun_recording_name="regenbogen_da3_video",
        rerun_spawn_viewer=True,
    )

    pipeline.add_node(
        VideoReaderNode(
            video_path=video_path,
            max_frames=max_frames,
            frame_skip=frame_skip,
            name="VideoReader",
        )
    )

    pipeline.add_node(
        DepthAnything3Node(
            model_name=model_name,
            buffer_size=buffer_size,
            buffer_step=buffer_step,
            estimate_poses=True,
            use_pose_conditioning=True,
            name="DA3",
        )
    )

    pipeline.add_node(
        DepthToPointCloudNode(
            max_depth=10.0,
            min_depth=0.1,
            name="PointCloudGen",
        )
    )

    video_reader = pipeline.get_node("VideoReader")
    print("🎥 Video information:")
    print(f"   Total frames: {video_reader.total_frames}")
    print(f"   Video FPS: {video_reader.video_fps:.2f}")
    print()

    print("🚀 Processing video frames...")
    print(f"   First {buffer_size - 1} frames will warm up the buffer")
    print("   Subsequent frames will have estimated poses and depth")
    print()

    processed_frames = 0
    frames_with_poses = 0
    total_points = 0

    try:
        for frames in pipeline.process_stream():
            if frames is None:
                continue

            for frame in frames:

                processed_frames += 1

                # Check if frame has pose information
                has_pose = frame.extrinsics is not None
                has_pc = frame.pointcloud is not None

                if has_pose:
                    frames_with_poses += 1

                if has_pc:
                    total_points += len(frame.pointcloud)

                rr.set_time("frame", sequence=processed_frames)

                if has_pose and frame.intrinsics is not None:
                    h, w = frame.rgb.shape[:2]

                    rotation_matrix = frame.extrinsics[:3, :3]
                    translation_vector = frame.extrinsics[:3, 3]

                    rr.log(
                        f"world/keyframes/frame_{processed_frames:04d}",
                        rr.Transform3D(
                            translation=translation_vector,
                            mat3x3=rotation_matrix,
                        ),
                    )

                    rr.log(
                        f"world/keyframes/frame_{processed_frames:04d}/pinhole",
                        rr.Pinhole(
                            resolution=[w, h],
                            focal_length=[frame.intrinsics[0, 0], frame.intrinsics[1, 1]],
                            principal_point=[
                                frame.intrinsics[0, 2],
                                frame.intrinsics[1, 2],
                            ],
                            camera_xyz=rr.ViewCoordinates.RDF,  # OpenCV: Right-Down-Forward
                            image_plane_distance=0.2,
                        ),
                    )
                    if has_pc and has_pose:
                        # Get colors from metadata (computed by DepthToPointCloudNode)
                        colors = frame.metadata.get("pointcloud_colors", None)

                        # Log pointcloud in world coordinates
                        rr.log(
                            f"world/keyframes/frame_{processed_frames:04d}/pointcloud",
                            rr.Points3D(frame.pointcloud, colors=colors),
                        )

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Processing complete! Summary:")
    print(f"   Total frames processed: {processed_frames}")
    print(f"   Frames with poses: {frames_with_poses}")
    print(f"   Total points generated: {total_points:,}")
    if frames_with_poses > 0:
        print(f"   Avg points per frame: {total_points // frames_with_poses:,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process video with Depth Anything V3 for depth and pose estimation"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (default: download sample)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to process (default: 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="da3-small",
        choices=[
            "da3-small",
            "da3-base",
            "da3-large",
            "da3-giant",
            "da3nested-giant-large",
        ],
        help="DA3 model variant (default: da3-small for speed)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=3,
        help="Sliding window buffer size (default: 3)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Skip every N frames (1 = process all frames)",
    )

    args = parser.parse_args()

    main(
        video_path=args.video,
        max_frames=args.max_frames,
        model_name=args.model,
        buffer_size=args.buffer_size,
        frame_skip=args.frame_skip,
    )
