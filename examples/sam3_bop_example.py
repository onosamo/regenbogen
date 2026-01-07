"""
Example demonstrating SAM3 text-prompted segmentation with BOP dataset.

This example shows how to:
1. Load BOP dataset frames
2. Apply SAM3 with text prompts for concept-based segmentation
3. Visualize results in Rerun with segmentation masks and bounding boxes
"""

import argparse
import logging

from regenbogen import Pipeline
from regenbogen.nodes import BOPDatasetNode, SAM3Node

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sam3_with_bop_example(
    dataset_name="ycbv",
    split="test",
    scene_id=48,
    max_samples=5,
    text_prompt="an object",
):
    """
    Demonstrate SAM3 text-prompted segmentation on BOP dataset with Rerun visualization.

    Args:
        dataset_name: BOP dataset name (e.g., 'ycbv', 'tless', 'lm')
        split: Dataset split ('train' or 'test')
        scene_id: Scene ID to process
        max_samples: Maximum number of samples to process
        text_prompt: Text prompt for concept-based segmentation
    """
    logger.info(f"Loading BOP dataset: {dataset_name}, scene {scene_id}")

    # Create BOP dataset loader
    bop_loader = BOPDatasetNode(
        dataset_name=dataset_name,
        split=split,
        scene_id=scene_id,
        max_samples=max_samples,
        load_depth=True,
        load_models=True,
        allow_download=True,
        download_scene_ids=[scene_id] if scene_id is not None else None,
        enable_rerun_logging=False,
        name="BOP_Loader",
    )

    # Create SAM3 node with text prompt
    logger.info(f"Initializing SAM3 model with text prompt: '{text_prompt}'...")
    sam3_node = SAM3Node(
        device=None,  # Auto-detect GPU/CPU
        text_prompt=text_prompt,
        mask_threshold=0.0,
        name="SAM3",
    )

    pipeline = Pipeline(
        name="SAM3_BOP_Pipeline",
        enable_rerun_logging=True,
        rerun_spawn_viewer=True,
        rerun_recording_name="SAM3_BOP",
    )
    pipeline.add_node(bop_loader)
    pipeline.add_node(sam3_node)

    logger.info("Processing frames with SAM3...")
    # Consume the generator to actually process the frames
    for masks in pipeline.process_stream():
        pass  # Pipeline handles all visualization automatically

    logger.info("Processing complete! Results are visualized in Rerun viewer")


def main():
    """Run the SAM3 with BOP dataset example."""
    parser = argparse.ArgumentParser(
        description="SAM3 Text-Prompted Segmentation with BOP Dataset and Rerun Visualization"
    )
    parser.add_argument(
        "--dataset-name",
        default="ycbv",
        choices=["ycbv", "tless", "lm", "lmo", "hb", "icbin", "itodd", "tudl", "tyol"],
        help="BOP dataset name (default: ycbv)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--scene-id", type=int, default=48, help="Scene ID to process (default: 48)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum samples to process (default: 5)",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="an object",
        help="Text prompt for SAM3 (e.g., 'a red object', 'a bottle', 'metallic objects')",
    )

    args = parser.parse_args()

    try:
        sam3_with_bop_example(
            dataset_name=args.dataset_name,
            split=args.split,
            scene_id=args.scene_id,
            max_samples=args.max_samples,
            text_prompt=args.text_prompt,
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
