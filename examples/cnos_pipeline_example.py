"""
CNOS pipeline example using node chaining.

This example demonstrates the CNOS (CAD-based Novel Object Segmentation) pipeline
using regenbogen's node-based architecture with minimal boilerplate.
"""

from __future__ import annotations

import argparse
import logging

from regenbogen.interfaces import Frame
from regenbogen.nodes import (
    BOPDatasetNode,
    CNOSMatcherNode,
    Dinov2Node,
    SAM2Node,
    SphericalPoseGeneratorNode,
    TemplateDescriptorNode,
    TemplateRendererNode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_templates_for_object(
    object_model,
    object_id: int,
    radius_mm: float = 400.0,
    icosahedron_subdivisions: int = 1,
) -> list[Frame]:
    """
    Render templates for a single object.

    Args:
        object_model: Object model from BOP dataset
        object_id: Object ID
        radius_mm: Camera distance in millimeters
        icosahedron_subdivisions: Icosphere subdivision level (1 = 42 views)

    Returns:
        List of rendered frames
    """
    logger.info(f"Rendering {object_id} with subdivision level {icosahedron_subdivisions}")

    # Generate poses and render
    pose_generator = SphericalPoseGeneratorNode(
        radius=radius_mm,
        inplane_rotations=1,
        icosahedron_subdivisions=icosahedron_subdivisions,
    )

    renderer = TemplateRendererNode(
        width=640,
        height=480,
        fx=572.4,
        fy=573.6,
    )

    poses = list(pose_generator.generate_poses())
    templates = list(renderer.render_templates(object_model, iter(poses)))

    logger.info(f"Rendered {len(templates)} templates for object {object_id}")
    return templates


def run_cnos_pipeline(
    dataset_name: str = "ycbv",
    object_ids: list[int] | None = None,
    scene_id: int = 48,
    image_id: int | None = None,
    confidence_threshold: float = 0.5,
    template_cache_dir: str | None = None,
    device: str = None,
    enable_rerun: bool = True,
) -> None:
    """
    Run CNOS pipeline with node chaining.

    Args:
        dataset_name: BOP dataset name
        object_ids: List of object IDs to process
        scene_id: Scene ID to test on
        image_id: Image ID (None = first available)
        confidence_threshold: Confidence threshold for matches
        template_cache_dir: Cache directory for templates
        device: Device for computation
        enable_rerun: Enable Rerun visualization
    """
    logger.info("=" * 60)
    logger.info("CNOS Pipeline")
    logger.info("=" * 60)

    # Initialize BOP dataset with rerun logging
    bop_loader = BOPDatasetNode(
        dataset_name=dataset_name,
        split="test",
        scene_id=scene_id,
        load_models=True,
        allow_download=True,
        enable_rerun_logging=enable_rerun,
        rerun_recording_name=f"CNOS_{dataset_name}_scene{scene_id}",
    )

    # Get object IDs
    available_objects = bop_loader.get_object_ids()
    if object_ids is None:
        object_ids = available_objects[:3]

    logger.info(f"Processing objects: {object_ids}")

    # Initialize descriptor node with caching
    feature_extractor = Dinov2Node(
        model_size="small",
        device=device,
        output_type="cls",
    )

    descriptor_node = TemplateDescriptorNode(
        feature_extractor=feature_extractor,
        device=device,
        cache_dir=template_cache_dir,
        dataset_name=dataset_name,
    )

    # Process templates for each object
    all_template_descriptors = []
    for obj_id in object_ids:
        logger.info(f"\n--- Object {obj_id} ---")

        object_model = bop_loader.get_object_model(obj_id)
        if object_model is None:
            logger.warning(f"Object {obj_id} not found, skipping")
            continue

        # Render templates
        templates = render_templates_for_object(object_model, obj_id)

        # Extract descriptors (with automatic caching)
        template_desc = descriptor_node.process_templates(iter(templates), obj_id)
        all_template_descriptors.append(template_desc)

    if not all_template_descriptors:
        logger.error("No template descriptors computed!")
        return

    # Combine descriptors
    import torch

    from regenbogen.nodes.template_descriptor import TemplateDescriptor

    combined_descriptors = TemplateDescriptor(
        descriptors=torch.cat([d.descriptors for d in all_template_descriptors]),
        object_ids=torch.cat([d.object_ids for d in all_template_descriptors]),
        template_ids=torch.cat([d.template_ids for d in all_template_descriptors]),
    )

    logger.info(f"Combined: {len(combined_descriptors)} templates")

    # Get query image
    if image_id is None:
        # Use first available image
        for frame, poses, obj_ids in bop_loader.process():
            image_id = frame.metadata["image_id"]
            query_frame = frame
            break
        logger.info(f"Using first available image: {image_id}")
    else:
        sample = bop_loader.get_sample(scene_id, image_id)
        if sample is None:
            logger.error(f"Image {image_id} not found")
            return
        query_frame, _, _ = sample

    # Process query: SAM2 -> Feature Extraction -> CNOS Matching
    logger.info("\n--- Processing Query ---")

    sam2 = SAM2Node(
        model_size="tiny",
        device=device,
        pred_iou_thresh=0.7,
    )

    proposals = sam2.process(query_frame)
    logger.info(f"Generated {len(proposals.masks)} proposals")

    if len(proposals.masks) == 0:
        logger.warning("No proposals generated!")
        return

    # Extract features from proposals
    query_features = feature_extractor.process_masked_batch(query_frame.rgb, proposals)

    # Match proposals to templates
    matcher = CNOSMatcherNode(
        template_descriptors=combined_descriptors,
        confidence_threshold=confidence_threshold,
        aggregation="avg_5",
        device=device,
        enable_rerun_logging=enable_rerun,
        rerun_entity_path="cnos/detections",
    )

    matches = matcher.process((query_features, proposals))

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info(f"Detections: {len(matches.labels)}")
    for obj_id, score in zip(matches.labels, matches.scores):
        logger.info(f"  - Object {obj_id}: score={score:.3f}")
    logger.info("=" * 60)


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(description="CNOS pipeline example")
    parser.add_argument("--dataset", type=str, default="ycbv", help="BOP dataset name")
    parser.add_argument("--object-ids", type=int, nargs="+", help="Object IDs to process")
    parser.add_argument("--scene-id", type=int, default=48, help="Scene ID")
    parser.add_argument("--image-id", type=int, help="Image ID")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--template-cache", type=str, default="/tmp/cnos_templates", help="Template cache directory")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun visualization")

    args = parser.parse_args()

    run_cnos_pipeline(
        dataset_name=args.dataset,
        object_ids=args.object_ids,
        scene_id=args.scene_id,
        image_id=args.image_id,
        confidence_threshold=args.confidence,
        template_cache_dir=args.template_cache,
        device=args.device,
        enable_rerun=not args.no_rerun,
    )


if __name__ == "__main__":
    main()
