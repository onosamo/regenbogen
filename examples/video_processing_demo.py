#!/usr/bin/env python3
"""
Demo script showing the video processing pipeline with real video files using regenbogen Pipeline and Rerun logging.

This script demonstrates the complete functionality of the video monodepth pointcloud
processing pipeline using actual video files, with interactive Rerun visualization.
"""

import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to path to import regenbogen
sys.path.insert(0, str(Path(__file__).parent.parent))

from regenbogen import Pipeline
from regenbogen.nodes import DepthAnythingNode, DepthToPointCloudNode, VideoReaderNode


def demonstrate_video_pipeline(
    video_path: Optional[str] = None, max_frames: int = None, enable_rerun: bool = True
):
    """Demonstrate the complete video processing pipeline using regenbogen Pipeline."""
    print("🎬 Video Monodepth Pointcloud Pipeline Demo")
    print("=" * 45)
    print("📝 This demo shows the complete pipeline functionality")
    print("   using actual video files with stream-based Pipeline orchestration.")
    if video_path is None:
        print("📥 No video path provided - will download sample video")
    if enable_rerun:
        print("🌈 Rerun logging enabled for interactive visualization")
    print()

    # Create the pipeline with Rerun logging
    print("🏗️  Setting up regenbogen pipeline...")
    pipeline = Pipeline(
        name="Video_Processing_Pipeline",
        enable_rerun_logging=enable_rerun,
        rerun_recording_name="regenbogen_video_demo",
        rerun_spawn_viewer=enable_rerun,
    )

    # Add nodes to the pipeline - VideoReaderNode first, then processing nodes
    pipeline.add_node(
        VideoReaderNode(
            video_path=video_path, max_frames=max_frames, name="VideoReader"
        )
    )

    pipeline.add_node(
        DepthAnythingNode(
            model_size="small",
            device=None,  # Auto-detect best device
            name="DepthEstimation",
        )
    )

    pipeline.add_node(
        DepthToPointCloudNode(
            max_depth=10.0, min_depth=0.1, name="PointCloudGeneration"
        )
    )

    print(f"   ✅ Pipeline created with {len(pipeline)} nodes")
    print("   ✅ Video reader node (VideoReaderNode)")
    print("   ✅ Depth estimation node (DepthAnythingNode)")
    print("   ✅ Pointcloud generation node (DepthToPointCloudNode)")
    if enable_rerun:
        print("   ✅ Rerun logging configured")
    print()

    # Validate video file access
    print("🎥 Validating video file...")
    try:
        # Get the video reader node to check video properties
        video_reader = pipeline.get_node("VideoReader")
        print(f"   ✅ Video file: {video_path}")
        print(f"   ✅ Total frames in video: {video_reader.total_frames}")
        print(f"   ✅ Video FPS: {video_reader.video_fps:.2f}")
        if max_frames:
            print(f"   ✅ Processing limit: {max_frames} frames")
    except Exception as e:
        print(f"   ❌ Failed to validate video file: {e}")
        return False

    # Process frames through the pipeline using stream processing
    print()
    print("🖼️  Processing video frames through pipeline...")
    frames_to_process = max_frames or video_reader.total_frames
    print(f"   Processing up to {frames_to_process} frames...")
    print()

    total_points = 0
    processed_frames = 0
    frame_width, frame_height = None, None

    try:
        # Use process_stream to handle the video stream
        for result_frame in pipeline.process_stream():
            frame_num = processed_frames + 1
            total_frames_display = max_frames or video_reader.total_frames
            print(f"   📹 Frame {frame_num}/{total_frames_display}:")
            print(f"     - Input RGB: {result_frame.rgb.shape}")

            # Get frame dimensions from first frame
            if frame_width is None:
                frame_height, frame_width = result_frame.rgb.shape[:2]

            if result_frame.depth is not None:
                depth_stats = f"min: {result_frame.depth.min():.2f}, max: {result_frame.depth.max():.2f}, mean: {result_frame.depth.mean():.2f}"
                print(
                    f"     - Estimated depth: {result_frame.depth.shape} ({depth_stats})"
                )

            if result_frame.pointcloud is not None:
                num_points = len(result_frame.pointcloud)
                total_points += num_points
                print(f"     - Generated pointcloud: {num_points:,} points")

                # Show sample points
                if num_points >= 3:
                    sample_points = result_frame.pointcloud[:3]
                    print(
                        f"     - Sample XYZ points: {sample_points[0]}, {sample_points[1]}, {sample_points[2]}"
                    )

            processed_frames += 1
            print()

    except Exception as e:
        print(f"     ❌ Error processing video stream: {e}")
        return False

    # Summary
    print("✅ Video processing completed!")
    print()
    print("📊 Processing Summary:")
    print(f"   - Video frames processed: {processed_frames}")
    print(f"   - Total pointcloud points generated: {total_points:,}")
    if processed_frames > 0:
        avg_points = total_points // processed_frames
        print(f"   - Average points per frame: {avg_points:,}")
    print()

    print("🔍 Pipeline Analysis:")
    print("   - VideoReaderNode streams frames from video file")
    print("   - Each RGB frame (3 channels) → Depth map (1 channel)")
    print("   - Each depth pixel → 3D point (X, Y, Z coordinates)")
    if frame_width and frame_height:
        print(
            f"   - Frame resolution: {frame_width}x{frame_height} = {frame_width * frame_height:,} pixels"
        )
    print("   - Valid depth pixels become 3D points in pointcloud")
    print("   - Pipeline processes each frame through all nodes sequentially")
    print()

    if enable_rerun:
        print("🌈 Rerun Visualization:")
        print("   - RGB frames logged to 'video/rgb' (timeline sequence)")
        print("   - Depth maps logged to 'video/depth' (timeline sequence)")
        print("   - Pointclouds logged to 'video/pointcloud' (timeline sequence)")
        print("   - Camera intrinsics logged to 'video/camera'")
        print("   - All data appears as timeline with frame sequence numbers")
        print("   - Keep Rerun viewer open to explore results interactively")
        print()

    print("🎯 Usage:")
    print("   To process video files:")
    print("   $ python video_processing_demo.py                         # Downloads sample video")
    print("   $ python video_processing_demo.py your_video.mp4")
    print("   $ python video_processing_demo.py your_video.mp4 --max-frames 10")
    print("   $ python video_processing_demo.py your_video.mp4 --no-rerun")
    print()

    return processed_frames > 0


def main():
    """Run the demonstration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Video processing pipeline demo with regenbogen Pipeline and Rerun logging"
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Path to the video file (optional, downloads sample video if not provided)",
    )
    parser.add_argument(
        "--max-frames", type=int, help="Maximum number of frames to process"
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun logging and visualization",
    )

    args = parser.parse_args()

    try:
        success = demonstrate_video_pipeline(
            args.video_path, args.max_frames, enable_rerun=not args.no_rerun
        )
        if success:
            print("\n🎉 Demo completed successfully!")
            if not args.no_rerun:
                print("💡 Keep the Rerun viewer open to explore the results.")
            return 0
        else:
            print("\n❌ Demo failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        if not args.no_rerun:
            print("💡 If you see Rerun-related errors, try: pip install rerun-sdk")
        return 1


if __name__ == "__main__":
    sys.exit(main())
