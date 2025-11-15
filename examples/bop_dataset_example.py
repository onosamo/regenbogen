"""
Example demonstrating BOP dataset visualization with Rerun.
"""

import logging

from regenbogen.nodes.bop_dataset import BOPDatasetNode

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def visualize_with_rerun(dataset_name="ycbv", split="test", scene_id=0, max_samples=10):
    """Load BOP dataset and visualize frames with ground truth poses in Rerun."""
    bop_loader = BOPDatasetNode(
        dataset_name=dataset_name,
        split=split,
        scene_id=scene_id,
        max_samples=max_samples,
        load_depth=True,
        load_models=True,
        allow_download=True,
        download_scene_ids=[scene_id] if scene_id is not None else None,
        enable_rerun_logging=True,
        rerun_spawn_viewer=True,
        name="BOP_Loader",
    )

    logger.info("Processing BOP dataset with automatic Rerun visualization...")

    sample_count = 0
    for frame, gt_poses, obj_ids in bop_loader.process():
        sample_count += 1

    logger.info(f"Processed {sample_count} samples")


def main():
    """Run the BOP dataset visualization example."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BOP Dataset Example with Rerun Visualization"
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
        "--scene-id", type=int, default=0, help="Scene ID to visualize (default: 0)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples to visualize (default: 10)",
    )

    args = parser.parse_args()

    try:
        visualize_with_rerun(
            dataset_name=args.dataset_name,
            split=args.split,
            scene_id=args.scene_id,
            max_samples=args.max_samples,
        )
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
