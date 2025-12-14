"""
Example demonstrating SAM2 instance segmentation with BOP dataset and Rerun visualization.

This example shows how to:
1. Load BOP dataset frames
2. Apply SAM2 for automatic mask generation
3. Visualize results in Rerun with segmentation masks and bounding boxes
"""

import logging

from regenbogen import Pipeline
from regenbogen.nodes.bop_dataset import BOPDatasetNode
from regenbogen.nodes.sam2 import SAM2Node

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sam2_with_bop_example(
    dataset_name="ycbv",
    split="test",
    scene_id=0,
    max_samples=5,
    model_size="tiny",
):
    """
    Demonstrate SAM2 segmentation on BOP dataset with Rerun visualization.

    Args:
        dataset_name: BOP dataset name (e.g., 'ycbv', 'tless', 'lm')
        split: Dataset split ('train' or 'test')
        scene_id: Scene ID to process
        max_samples: Maximum number of samples to process
        model_size: SAM2 model size ('tiny', 'small', 'base-plus', or 'large')
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

    logger.info(f"Initializing SAM2 model (size: {model_size})...")
    sam2_node = SAM2Node(
        model_size=model_size,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        mask_threshold=0.5,
        name="SAM2",
    )

    pipeline = Pipeline(
        name="SAM2_BOP_Pipeline",
        enable_rerun_logging=True,
        rerun_spawn_viewer=True,
        rerun_recording_name="SAM2_BOP",
    )
    pipeline.add_node(bop_loader)
    pipeline.add_node(sam2_node)

    logger.info("Processing frames with SAM2...")
    for masks in pipeline.process_stream():
        pass

    logger.info("Processing complete! Results are visualized in Rerun viewer")


def main():
    """Run the SAM2 with BOP dataset example."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM2 Instance Segmentation with BOP Dataset and Rerun Visualization"
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
        "--scene-id", type=int, default=0, help="Scene ID to process (default: 0)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum samples to process (default: 5)",
    )
    parser.add_argument(
        "--model-size",
        default="tiny",
        choices=["tiny", "small", "base-plus", "large"],
        help="SAM2 model size (default: tiny)",
    )

    args = parser.parse_args()

    try:
        sam2_with_bop_example(
            dataset_name=args.dataset_name,
            split=args.split,
            scene_id=args.scene_id,
            max_samples=args.max_samples,
            model_size=args.model_size,
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
