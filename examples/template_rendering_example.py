"""
Template rendering example using BOP dataset objects.

This example demonstrates how to:
1. Load an object model from a BOP dataset
2. Generate camera poses on a sphere around the object
3. Render RGB-D templates from each pose
4. Save templates for use in pose estimation pipelines like CNOS

The templates can be used for training or inference in object pose estimation systems.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from regenbogen.interfaces import Frame
from regenbogen.nodes import (
    BOPDatasetNode,
    SphericalPoseGeneratorNode,
    TemplateRendererNode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_template(frame: Frame, output_dir: Path, template_id: int) -> None:
    """
    Save a template frame to disk.

    Args:
        frame: Frame containing RGB and depth images
        output_dir: Directory to save templates
        template_id: Unique identifier for this template
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping template saving")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save RGB image
    rgb_path = output_dir / f"template_{template_id:06d}_rgb.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))

    # Save depth image (as 16-bit PNG, depth in mm)
    if frame.depth is not None:
        depth_path = output_dir / f"template_{template_id:06d}_depth.png"
        depth_mm = (frame.depth * 1000.0).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)

    # Save metadata
    metadata_path = output_dir / f"template_{template_id:06d}_meta.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Template ID: {template_id}\n")
        f.write(f"Object: {frame.metadata.get('object_name', 'unknown')}\n")
        if 'pose_metadata' in frame.metadata:
            pose_meta = frame.metadata['pose_metadata']
            f.write(f"Pose ID: {pose_meta.get('pose_id', 'N/A')}\n")
            f.write(f"Camera Position: {pose_meta.get('camera_position', 'N/A')}\n")
            f.write(f"Radius: {pose_meta.get('radius', 'N/A')}\n")

    logger.debug(f"Saved template {template_id} to {output_dir}")


def visualize_scene_setup(
    object_model,
    poses: list,
    radius: float
) -> None:
    """
    Visualize the scene setup with mesh and camera poses in Rerun.

    Args:
        object_model: The object model to visualize
        poses: List of camera poses
        radius: Sphere radius for reference
    """
    try:
        import rerun as rr
    except ImportError:
        logger.warning("Rerun not available, skipping scene setup visualization")
        return

    # Initialize Rerun for scene setup
    rr.init("template_scene_setup", spawn=True)

    # Log the object mesh at origin
    if object_model.mesh_vertices is not None and object_model.mesh_faces is not None:
        rr.log(
            "world/object_mesh",
            rr.Mesh3D(
                vertex_positions=object_model.mesh_vertices,
                triangle_indices=object_model.mesh_faces,
                vertex_normals=object_model.mesh_normals if object_model.mesh_normals is not None else None,
            ),
        )

        # Log object bounding box for reference
        vertices = object_model.mesh_vertices
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        logger.info(f"Object bounding box: min={bbox_min}, max={bbox_max}")
        logger.info(f"Object center: {(bbox_min + bbox_max) / 2}")
        logger.info(f"Object size: {bbox_max - bbox_min}")

    # Log camera poses as frustums
    for i, pose in enumerate(poses):
        # Camera position (translation)
        cam_pos = pose.translation

        # Convert rotation matrix to quaternion for Rerun
        from scipy.spatial.transform import Rotation

        # Use the rotation matrix directly - our pose generator now creates correct orientations
        quat = Rotation.from_matrix(pose.rotation).as_quat()  # [x, y, z, w]

        # Create camera with proper intrinsics for frustum visualization
        # Use reasonable focal length and image size for visualization
        fx = fy = 640.0  # Focal length
        cx = cy = 320.0  # Principal point
        width = height = 640  # Image dimensions

        rr.log(
            f"world/cameras/camera_{i:03d}",
            rr.Transform3D(
                translation=cam_pos,
                rotation=rr.Quaternion(xyzw=quat)
            ),
        )

        # Log camera frustum using Pinhole camera
        rr.log(
            f"world/cameras/camera_{i:03d}",
            rr.Pinhole(
                focal_length=[fx, fy],
                principal_point=[cx, cy],
                resolution=[width, height],
            ),
        )

    # Log sphere reference
    sphere_points = []
    for i in range(100):
        theta = 2 * np.pi * i / 100
        for j in range(50):
            phi = np.pi * j / 50
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            sphere_points.append([x, y, z])

    rr.log(
        "world/sphere_reference",
        rr.Points3D(
            positions=sphere_points,
            colors=[0, 255, 0],  # Green for sphere
            radii=[0.005]
        ),
    )

    logger.info(f"Scene setup visualized with {len(poses)} camera poses")


def visualize_combined_scene(
    object_model,
    poses: list,
    rendered_frames: list[Frame],
    radius_mm: float
) -> None:
    """
    Visualize the complete scene: object mesh, camera poses, and rendered templates.

    Args:
        object_model: BOP dataset object model
        poses: List of camera poses
        rendered_frames: List of rendered frames
        radius_mm: Sphere radius in millimeters
    """
    try:
        import rerun as rr
    except ImportError:
        logger.warning("Rerun not available, skipping visualization")
        return

    # Initialize Rerun
    rr.init("template_rendering_pipeline", spawn=True)

    # Log the object mesh
    rr.log(
        "world/object",
        rr.Mesh3D(
            vertex_positions=object_model.mesh_vertices,
            triangle_indices=object_model.mesh_faces,
            vertex_normals=object_model.mesh_normals if object_model.mesh_normals is not None else None,
        ),
    )

    # Log object bounding box for reference
    vertices = object_model.mesh_vertices
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    logger.info(f"Object bounding box: min={bbox_min}, max={bbox_max}")
    logger.info(f"Object center: {(bbox_min + bbox_max) / 2}")
    logger.info(f"Object size: {bbox_max - bbox_min}")

    # Log camera poses and their corresponding rendered images
    for i, (pose, frame) in enumerate(zip(poses, rendered_frames)):
        # Set timeline for this camera/frame
        rr.set_time_sequence("camera_id", i)

        # Camera position and orientation
        cam_pos = pose.translation

        # Create camera entity with transform and pinhole parameters
        camera_entity = f"world/cameras/camera_{i:03d}"

        # Log camera transform
        rr.log(
            camera_entity,
            rr.Transform3D(
                translation=cam_pos,
                mat3x3=pose.rotation
            ),
        )

        # Log camera frustum using same intrinsics as renderer
        rr.log(
            camera_entity,
            rr.Pinhole(
                focal_length=[320.0, 320.0],  # Same as renderer
                principal_point=[320.0, 240.0],  # Image center
                resolution=[640, 480],
            ),
        )

        # Log rendered RGB image from this camera
        rr.log(
            f"{camera_entity}/rgb",
            rr.Image(frame.rgb),
        )

        # Log rendered depth image from this camera
        if frame.depth is not None:
            rr.log(
                f"{camera_entity}/depth",
                rr.DepthImage(frame.depth),
            )

    logger.info(f"Visualized complete scene with {len(rendered_frames)} camera poses and templates in Rerun")


def visualize_with_rerun(
    frames: list[Frame],
    object_name: str,
    show_mesh: bool = True
) -> None:
    """
    Visualize rendered templates using Rerun.

    Args:
        frames: List of rendered frames
        object_name: Name of the object
        show_mesh: Whether to display the original mesh
    """
    try:
        import rerun as rr
    except ImportError:
        logger.warning("Rerun not available, skipping visualization")
        return

    # Initialize Rerun
    rr.init(f"template_rendering_{object_name}", spawn=True)

    # Log templates
    for i, frame in enumerate(frames):
        rr.set_time_sequence("template_id", i)

        # Log RGB image
        rr.log(
            "templates/rgb",
            rr.Image(frame.rgb),
        )

        # Log depth image
        if frame.depth is not None:
            rr.log(
                "templates/depth",
                rr.DepthImage(frame.depth),
            )

        # Log camera pose
        if frame.extrinsics is not None:
            # Convert extrinsics to camera position
            cam_to_world = np.linalg.inv(frame.extrinsics)
            camera_pos = cam_to_world[:3, 3]

            # Log camera position as a point
            rr.log(
                f"cameras/cam_{i}",
                rr.Transform3D(
                    translation=camera_pos,
                    mat3x3=cam_to_world[:3, :3]
                ),
            )

    logger.info(f"Visualized {len(frames)} templates in Rerun")


def render_templates_example(
    dataset_name: str = "ycbv",
    object_id: int = 1,
    radius: float = 0.5,
    num_views: int | None = None,
    elevation_levels: int = 3,
    azimuth_samples: int = 8,
    output_dir: str | None = None,
    visualize: bool = True,
    save_templates: bool = False,
) -> list[Frame]:
    """
    Render templates for an object from BOP dataset.

    Args:
        dataset_name: Name of BOP dataset (e.g., "ycbv", "lmo", "tless")
        object_id: Object ID to render
        radius: Distance from object center to camera (in meters)
        num_views: Total number of views (overrides elevation_levels/azimuth_samples)
        elevation_levels: Number of elevation levels on sphere
        azimuth_samples: Number of azimuth samples per elevation
        output_dir: Directory to save templates (if save_templates=True)
        visualize: Whether to visualize with Rerun
        save_templates: Whether to save templates to disk

    Returns:
        List of rendered Frame objects
    """
    logger.info(f"Starting template rendering for {dataset_name} object {object_id}")

    # Step 1: Load object model from BOP dataset
    logger.info(f"Loading object model from BOP dataset: {dataset_name}")
    bop_loader = BOPDatasetNode(
        dataset_name=dataset_name,
        split="test",
        load_models=True,
        allow_download=True,
        name="BOP_Loader"
    )

    # Get the object model
    object_model = bop_loader.get_object_model(object_id)
    if object_model is None:
        raise ValueError(f"Object {object_id} not found in {dataset_name} dataset")

    logger.info(f"Loaded object: {object_model.name}")
    logger.info(f"  Vertices: {len(object_model.mesh_vertices)}")
    logger.info(f"  Faces: {len(object_model.mesh_faces)}")

    # Step 2: Generate camera poses on a sphere
    logger.info("Generating camera poses on sphere...")

    # Convert radius from meters to millimeters to match BOP object coordinates
    # BOP objects are stored in millimeter units, so we need to convert the radius
    radius_mm = radius * 1000.0  # Convert meters to millimeters
    logger.info(f"Converting radius: {radius}m → {radius_mm}mm to match object coordinate system")

    pose_generator = SphericalPoseGeneratorNode(
        radius=radius_mm,
        inplane_rotations=1,
        name="PoseGenerator"
    )

    logger.info(f"Total poses to generate: {pose_generator.total_poses}")

    # Step 3: Render templates
    logger.info("Rendering templates...")

    # Use wide FOV configuration that works (scaled up from 64x64 with fx=32, fy=32)
    # Scale factor: 640/64 = 10x, so fx/fy should be 32*10 = 320
    width, height = 640, 480
    fx = fy = 320  # Wide FOV - key to successful rendering!
    cx, cy = width // 2, height // 2  # Image center

    logger.info(f"Using wide FOV configuration: fx={fx}, fy={fy} (FOV ≈ {2 * np.arctan(width/(2*fx)) * 180/np.pi:.1f}°)")

    renderer = TemplateRendererNode(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        z_near=1.0,
        z_far=2000.0,
        name="TemplateRenderer"
    )

    # Generate poses
    poses = list(pose_generator.generate_poses())
    logger.info(f"Generated {len(poses)} poses")

    # Render templates
    rendered_frames = list(renderer.render_templates(object_model, iter(poses)))
    logger.info(f"Rendered {len(rendered_frames)} templates")

    # Step 4: Save templates if requested
    if save_templates and output_dir:
        output_path = Path(output_dir)
        logger.info(f"Saving templates to {output_path}")
        for i, frame in enumerate(rendered_frames):
            save_template(frame, output_path, i)
        logger.info(f"Saved {len(rendered_frames)} templates")

    # Step 5: Combined visualization with Rerun if requested
    if visualize:
        logger.info("Visualizing complete scene: mesh, camera poses, and rendered templates...")
        visualize_combined_scene(object_model, poses, rendered_frames, radius_mm)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Template Rendering Summary")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Object ID: {object_id}")
    logger.info(f"Object Name: {object_model.name}")
    logger.info(f"Number of templates: {len(rendered_frames)}")
    logger.info(f"Image size: {rendered_frames[0].rgb.shape[:2]}")
    logger.info(f"Sphere radius: {radius} m")
    logger.info(f"Elevation levels: {elevation_levels}")
    logger.info(f"Azimuth samples: {azimuth_samples}")
    if save_templates and output_dir:
        logger.info(f"Templates saved to: {output_dir}")
    logger.info("=" * 60)

    return rendered_frames


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Render templates for BOP dataset objects"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ycbv",
        help="BOP dataset name (e.g., ycbv, lmo, tless)"
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=1,
        help="Object ID to render"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.5,
        help="Camera distance from object center (meters)"
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="Total number of views (overrides elevation/azimuth)"
    )
    parser.add_argument(
        "--elevation-levels",
        type=int,
        default=3,
        help="Number of elevation levels on sphere"
    )
    parser.add_argument(
        "--azimuth-samples",
        type=int,
        default=8,
        help="Number of azimuth samples per elevation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save templates"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable Rerun visualization"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save templates to disk"
    )

    args = parser.parse_args()

    # Run template rendering
    render_templates_example(
        dataset_name=args.dataset,
        object_id=args.object_id,
        radius=args.radius,
        num_views=args.num_views,
        elevation_levels=args.elevation_levels,
        azimuth_samples=args.azimuth_samples,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        save_templates=args.save,
    )


if __name__ == "__main__":
    main()
