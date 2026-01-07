#!/usr/bin/env python3


from regenbogen import Pipeline
from regenbogen.nodes import (
    DepthAnything3Node,
    DepthToPointCloudNode,
    SAM3Node,
    VideoReaderNode,
)


def main(
    video_path: str | None = None,
    max_frames: int | None = None,
    text_prompt: str = "an object",
    da3_model_name: str = "da3-small",
    frame_skip: int = 1,
):
    """
    Demonstrate video processing with Depth Anything V3.

    Args:
        video_path: Path to video file (None to download sample)
        max_frames: Maximum number of frames to process
        text_prompt: Text prompt for SAM3 segmentation
        da3_model_name: DA3 model variant (da3-small, da3-base, da3-large, da3nested-giant-large)
        frame_skip: Skip every N frames (1 = process all frames)
    """

    pipeline = Pipeline(
        name="DA3_SAM3_Video_Pipeline",
        enable_rerun_logging=True,
        rerun_recording_name="regenbogen_da3_sam3_video",
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
        SAM3Node(
            text_prompt=text_prompt,
            mask_threshold=0.0,
            name="SAM3",
        )
    )

    pipeline.add_node(
        DepthAnything3Node(
            model_name=da3_model_name,
            buffer_size=5,
            estimate_poses=True,
            name="DA3",
        )
    )

    pipeline.add_node(
        DepthToPointCloudNode(
            max_depth=10.0,
            min_depth=0.1,
            name="PointCloudGen",
            include_classes=text_prompt
        )
    )

    for _ in pipeline.process_stream():
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DA3 + SAM3 video processing example with Regenbogen"
    )
    parser.add_argument(
        "--video-path", type=str, default=None, help="Path to video file (default: download sample)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames to process (default: all)"
    )
    parser.add_argument(
        "--text-prompt", type=str, default="an object", help="Text prompt for SAM3 segmentation"
    )
    parser.add_argument(
        "--da3-model-name", type=str, default="da3-small", help="DA3 model variant (default: da3-small)"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=1, help="Skip every N frames (1 = process all frames)"
    )
    args = parser.parse_args()
    main(
        video_path=args.video_path,
        max_frames=args.max_frames,
        text_prompt=args.text_prompt,
        da3_model_name=args.da3_model_name,
        frame_skip=args.frame_skip,
    )

