"""
Template renderer node for generating RGB-D templates from CAD models.

This node renders templates of objects represented by CAD models (meshes) from
different camera viewpoints for use in pose estimation pipelines like CNOS.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pyrender
import trimesh

from ..core.node import Node
from ..interfaces import Frame, ObjectModel, Pose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateRendererNode(Node):
    """
    Node for rendering RGB-D templates from object models.

    This node takes an ObjectModel (mesh) and a set of camera Poses, and renders
    RGB-D images from each viewpoint using pyrender. Useful for generating
    templates for pose estimation methods like CNOS.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fx: Focal length in x direction (pixels)
        fy: Focal length in y direction (pixels)
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)
        z_near: Near clipping plane distance
        z_far: Far clipping plane distance
        ambient_light: Ambient light intensity [0, 1]
        render_normals: Whether to render surface normals
        render_mask: Whether to render segmentation mask
        **kwargs: Additional configuration parameters
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        z_near: float = 0.01,
        z_far: float = 2000.0,
        ambient_light: float = 0.5,
        render_normals: bool = False,
        render_mask: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.width = width
        self.height = height

        # Set default intrinsics if not provided
        self.fx = fx if fx is not None else width
        self.fy = fy if fy is not None else height
        self.cx = cx if cx is not None else width / 2.0
        self.cy = cy if cy is not None else height / 2.0

        self.z_near = z_near
        self.z_far = z_far
        self.ambient_light = ambient_light
        self.render_normals = render_normals
        self.render_mask = render_mask

        self._scene = None
        self._renderer = None
        self._mesh_node = None

    def _create_intrinsics_matrix(self) -> npt.NDArray[np.float64]:
        """
        Create camera intrinsics matrix.

        Returns:
            3x3 intrinsics matrix
        """
        K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return K

    def _setup_scene(self, object_model: ObjectModel) -> None:
        """
        Setup pyrender scene with object mesh.

        Args:
            object_model: ObjectModel containing mesh data
        """
        if object_model.mesh_vertices is None or object_model.mesh_faces is None:
            raise ValueError("ObjectModel must contain mesh_vertices and mesh_faces")

        # Create trimesh from vertices and faces
        mesh = trimesh.Trimesh(
            vertices=object_model.mesh_vertices,
            faces=object_model.mesh_faces
        )

        # Ensure proper normals
        if not mesh.is_watertight:
            logger.warning("Mesh is not watertight, normals may be inconsistent")

        # Create pyrender mesh
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        # Create scene
        self._scene = pyrender.Scene(
            ambient_light=np.array([self.ambient_light] * 3),
            bg_color=np.array([0.0, 0.0, 0.0, 0.0])  # Transparent background
        )

        # Add mesh to scene
        self._mesh_node = self._scene.add(pyrender_mesh)

        # Add multiple directional lights from different directions
        light_intensity = 5.0
        light_poses = [
            np.array([[1, 0, 0, 100], [0, 1, 0, 100], [0, 0, 1, 100], [0, 0, 0, 1]]),  # Top-right-front
            np.array([[1, 0, 0, -100], [0, 1, 0, -100], [0, 0, 1, 100], [0, 0, 0, 1]]),  # Top-left-front
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -100], [0, 0, 0, 1]]),  # Behind
        ]

        for i, light_pose in enumerate(light_poses):
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
            self._scene.add(light, pose=light_pose)
            logger.info(f"Added light {i} at position {light_pose[:3, 3]}")

        logger.info(f"Scene setup complete with mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    def _setup_renderer(self) -> None:
        """Setup offscreen renderer with platform-specific fallbacks."""
        if self._renderer is None:
            import os
            import platform

            # Try different platforms based on OS
            platforms_to_try = []

            if platform.system() == 'Darwin':  # macOS
                platforms_to_try = [None, 'osmesa', 'egl']  # Try default first on macOS
            else:  # Linux/Windows
                platforms_to_try = ['egl', 'osmesa', None]

            renderer_created = False

            for platform_name in platforms_to_try:
                try:
                    if platform_name is not None:
                        os.environ['PYOPENGL_PLATFORM'] = platform_name
                        logger.info(f"Trying renderer platform: {platform_name}")
                    else:
                        # Remove platform override to use default (usually requires display)
                        if 'PYOPENGL_PLATFORM' in os.environ:
                            del os.environ['PYOPENGL_PLATFORM']
                        logger.info("Trying default renderer platform (may require display)")

                    self._renderer = pyrender.OffscreenRenderer(
                        viewport_width=self.width,
                        viewport_height=self.height
                    )

                    platform_desc = platform_name if platform_name else "default"
                    logger.info(f"Renderer initialized with {platform_desc}: {self.width}x{self.height}")
                    renderer_created = True
                    break

                except Exception as e:
                    platform_desc = platform_name if platform_name else "default"
                    logger.info(f"Platform {platform_desc} failed: {e}")
                    continue

            if not renderer_created:
                logger.error("Failed to initialize renderer with any platform")
                logger.info("Note: Rendering requires EGL support (Linux) or OSMesa (macOS/headless)")
                logger.info("On macOS, rendering may require running in a terminal with display access")
                raise RuntimeError("Could not initialize offscreen renderer")

    def _render_view(
        self,
        pose: Pose
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        """
        Render a single view from the given camera pose.

        Args:
            pose: Camera pose (rotation and translation)

        Returns:
            Tuple of (rgb_image, depth_image)
        """
        # Create camera with intrinsics
        camera = pyrender.IntrinsicsCamera(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            znear=self.z_near,
            zfar=self.z_far
        )

        # Create camera pose matrix
        # The pose from SphericalPoseGeneratorNode gives us camera-to-world transform
        # with Z pointing toward object, but PyRender expects OpenGL convention (camera looks down -Z)
        # So we need to flip the Z-axis (forward direction)
        camera_pose = np.eye(4, dtype=np.float64)

        # Extract axes from pose rotation
        right_axis = pose.rotation[:, 0]     # X-axis (right)
        up_axis = pose.rotation[:, 1]        # Y-axis (up)
        forward_axis = pose.rotation[:, 2]   # Z-axis (forward toward object)

        # For OpenGL/PyRender: camera looks down -Z axis
        # Transform from pose generator's convention (camera looks along +Z, Z+ forward) to OpenGL/PyRender convention (camera looks along -Z, Z- forward).
        # This is done by flipping the forward (Z) and up (Y) axes: see https://www.khronos.org/opengl/wiki/Viewing_and_Transformations and PyRender documentation.
        opengl_rotation = np.column_stack([right_axis, -up_axis, -forward_axis])

        camera_pose[:3, :3] = opengl_rotation
        camera_pose[:3, 3] = pose.translation

        # Add camera to scene
        camera_node = self._scene.add(camera, pose=camera_pose)

        # Render
        color, depth = self._renderer.render(self._scene)

        # Remove camera from scene
        self._scene.remove_node(camera_node)

        return color, depth

    def render_templates(
        self,
        object_model: ObjectModel,
        poses: Iterator[Pose]
    ) -> Iterator[Frame]:
        """
        Render templates for all given poses.

        Args:
            object_model: ObjectModel containing mesh data
            poses: Iterator of camera poses

        Yields:
            Frame objects with rendered RGB and depth images
        """
        # Setup scene and renderer
        self._setup_scene(object_model)
        self._setup_renderer()

        intrinsics = self._create_intrinsics_matrix()

        frame_count = 0
        for pose in poses:
            try:
                # Render view
                rgb, depth = self._render_view(pose)

                # Create extrinsics matrix (world to camera)
                # The pose gives us camera-to-world, but extrinsics should be world-to-camera
                camera_to_world = np.eye(4, dtype=np.float64)
                camera_to_world[:3, :3] = pose.rotation
                camera_to_world[:3, 3] = pose.translation
                extrinsics = np.linalg.inv(camera_to_world)

                # Create Frame
                frame = Frame(
                    rgb=rgb,
                    depth=depth,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    metadata={
                        "template_id": frame_count,
                        "pose_metadata": pose.metadata,
                        "object_name": object_model.name,
                        "render_settings": {
                            "width": self.width,
                            "height": self.height,
                            "fx": self.fx,
                            "fy": self.fy,
                            "cx": self.cx,
                            "cy": self.cy
                        }
                    }
                )

                frame_count += 1
                yield frame

            except Exception as e:
                logger.error(f"Error rendering frame {frame_count}: {e}")
                continue

        logger.info(f"Rendered {frame_count} template frames")

    def process(
        self,
        input_data: tuple[ObjectModel, Iterator[Pose]]
    ) -> Iterator[Frame]:
        """
        Process method for pipeline compatibility.

        Args:
            input_data: Tuple of (ObjectModel, Iterator[Pose])

        Returns:
            Generator yielding Frame objects with rendered templates
        """
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            raise ValueError(
                "Input must be a tuple of (ObjectModel, Iterator[Pose])"
            )

        object_model, poses = input_data
        return self.render_templates(object_model, poses)

    def __del__(self):
        """Clean up renderer resources."""
        if self._renderer is not None:
            try:
                self._renderer.delete()
            except Exception as e:
                logger.warning(f"Error cleaning up renderer: {e}")
