"""
BOP dataset loader node for streaming benchmark datasets.

This node loads and streams data from BOP (Benchmark for 6D Object Pose Estimation)
format datasets, including ground truth poses and object models.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from platformdirs import user_cache_dir

from ..core.node import Node
from ..interfaces import Frame, ObjectModel, Pose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BOPDatasetNode(Node):
    """
    Node for loading and streaming BOP benchmark datasets.

    BOP datasets follow a standardized format with:
    - RGB images in train/test directories
    - Depth maps (if available)
    - Camera intrinsics in scene_camera.json
    - Ground truth poses in scene_gt.json
    - Object models in models directory

    This node can:
    - Stream frames from a specific scene
    - Load ground truth object poses
    - Load 3D object models
    - Support different splits (train/test)
    - Automatically download datasets if not present (when allow_download=True)
    """

    def __init__(
        self,
        dataset_name: str = "ycbv",
        dataset_path: Optional[str] = None,
        split: str = "test",
        scene_id: Optional[int] = None,
        max_samples: Optional[int] = None,
        load_depth: bool = False,
        load_models: bool = True,
        allow_download: bool = False,
        download_scene_ids: Optional[List[int]] = None,
        enable_rerun_logging: bool = False,
        rerun_recording_name: Optional[str] = None,
        rerun_spawn_viewer: bool = True,
        name: str = None,
        **kwargs,
    ):
        """
        Initialize BOP dataset loader.

        Args:
            dataset_name: Name of the BOP dataset (e.g., 'ycbv', 'tless', 'lm')
            dataset_path: Path to the BOP dataset root directory. If None and allow_download=True,
                         will download to cache directory
            split: Dataset split to use ('train' or 'test')
            scene_id: Specific scene ID to load (None = load all scenes)
            max_samples: Maximum number of samples to load (None = load all)
            load_depth: Whether to load depth maps
            load_models: Whether to load 3D object models
            allow_download: If True, download dataset if not found locally
            download_scene_ids: List of scene IDs to download (None = all scenes)
            enable_rerun_logging: Whether to enable Rerun visualization logging
            rerun_recording_name: Name for the Rerun recording (auto-generated if None)
            rerun_spawn_viewer: Whether to spawn the Rerun viewer window
            **kwargs: Additional configuration
        """
        super().__init__(name=name or "BOPDataset", **kwargs)

        self.dataset_name = dataset_name
        self.split = split
        self.scene_id = scene_id
        self.max_samples = max_samples
        self.load_depth = load_depth
        self.load_models = load_models
        self.allow_download = allow_download
        self.download_scene_ids = download_scene_ids
        self.enable_rerun_logging = enable_rerun_logging

        # Initialize Rerun logging if enabled
        self.rerun_logger = None
        if enable_rerun_logging:
            from ..utils.rerun_logger import RerunLogger

            if rerun_recording_name is None:
                rerun_recording_name = f"BOP_{dataset_name}_{split}"
                if scene_id is not None:
                    rerun_recording_name += f"_scene{scene_id}"

            self.rerun_logger = RerunLogger(rerun_recording_name, enabled=True, spawn=rerun_spawn_viewer)

        if dataset_path is None:
            cache_dir = user_cache_dir("regenbogen", "regenbogen")
            self.dataset_path = Path(cache_dir) / "bop_datasets" / dataset_name
        else:
            self.dataset_path = Path(dataset_path)

        self._models: Optional[Dict[int, ObjectModel]] = None
        self._scenes: Optional[List[int]] = None
        self._current_sample_idx = 0
        if not self.dataset_path.exists():
            if self.allow_download:
                self._download_dataset()
            else:
                raise ValueError(
                    f"Dataset path does not exist: {self.dataset_path}\n"
                    f"Set allow_download=True to automatically download the dataset."
                )
        elif self.allow_download:
            self._ensure_split_available()

    def _ensure_split_available(self):
        """Check if the requested split is available, download if missing."""
        try:
            self._get_split_path()
        except ValueError:
            logger.info(f"Split '{self.split}' not found, downloading...")
            self._download_split()

    def _download_split(self):
        """Download only the requested split (and models if needed)."""
        logger.info(
            f"Downloading missing split: {self.dataset_name} ({self.split} split)"
        )

        try:
            repo_id = f"bop-benchmark/{self.dataset_name}"

            logger.info(f"Checking available files in {repo_id}")
            available_files = list_repo_files(repo_id, repo_type="dataset")

            files_to_download = []

            split_patterns = self._get_split_file_patterns(self.split)
            for pattern in split_patterns:
                if pattern in available_files:
                    files_to_download.append(pattern)
                    logger.info(f"Selected {self.split} file: {pattern}")
                    break

            if self.load_models and not (self.dataset_path / "models").exists():
                model_patterns = [
                    f"{self.dataset_name}_models.zip",
                    f"{self.dataset_name}_base.zip",  # Often contains models_info.json
                ]
                for pattern in model_patterns:
                    if pattern in available_files:
                        files_to_download.append(pattern)
                        logger.info(f"Found model file: {pattern}")

            if not files_to_download:
                raise ValueError(
                    f"No suitable files found for dataset '{self.dataset_name}' with split '{self.split}'. "
                    f"Available files: {[f for f in available_files if f.endswith('.zip')]}"
                )

            logger.info(
                f"Will download {len(files_to_download)} files: {files_to_download}"
            )

            for filename in files_to_download:
                try:
                    logger.info(f"Downloading: {filename}")
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(self.dataset_path),
                    )

                    logger.info(f"Extracting: {filename}")
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(self.dataset_path)

                    Path(file_path).unlink()
                    logger.info(f"Successfully downloaded and extracted: {filename}")

                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    continue

            logger.info(f"Split '{self.split}' downloaded successfully")

        except Exception as e:
            raise RuntimeError(
                f"Failed to download split '{self.split}' from HuggingFace Hub: {e}\n"
                f"Make sure the dataset '{self.dataset_name}' exists at https://huggingface.co/bop-benchmark"
            )

    def _download_dataset(self):
        """Download the BOP dataset from HuggingFace Hub."""
        logger.info(
            f"Downloading BOP Dataset: {self.dataset_name} ({self.split} split)"
        )
        logger.info(f"Cache directory: {self.dataset_path.parent}")

        self.dataset_path.mkdir(parents=True, exist_ok=True)

        try:
            repo_id = f"bop-benchmark/{self.dataset_name}"

            logger.info(f"Checking available files in {repo_id}")
            available_files = list_repo_files(repo_id, repo_type="dataset")
            logger.info(f"Found {len(available_files)} files in repository")

            files_to_download = []

            split_patterns = self._get_split_file_patterns(self.split)
            for pattern in split_patterns:
                if pattern in available_files:
                    files_to_download.append(pattern)
                    logger.info(f"Selected {self.split} file: {pattern}")
                    break

            if self.load_models:
                model_patterns = [
                    f"{self.dataset_name}_models.zip",
                    f"{self.dataset_name}_base.zip",  # Often contains models_info.json
                ]
                for pattern in model_patterns:
                    if pattern in available_files:
                        files_to_download.append(pattern)
                        logger.info(f"Found model file: {pattern}")

            if not files_to_download:
                raise ValueError(
                    f"No suitable files found for dataset '{self.dataset_name}' with split '{self.split}'. "
                    f"Available files: {[f for f in available_files if f.endswith('.zip')]}"
                )

            logger.info(
                f"Will download {len(files_to_download)} files: {files_to_download}"
            )

            for filename in files_to_download:
                try:
                    logger.info(f"Downloading: {filename}")
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(self.dataset_path),
                    )

                    logger.info(f"Extracting: {filename}")
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(self.dataset_path)

                    Path(file_path).unlink()
                    logger.info(f"Successfully downloaded and extracted: {filename}")

                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    continue

            if not any(self.dataset_path.glob("*")):
                raise ValueError(
                    "No dataset files were successfully downloaded and extracted"
                )

            logger.info(f"Dataset downloaded successfully to: {self.dataset_path}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to download dataset from HuggingFace Hub: {e}\n"
                f"Make sure the dataset '{self.dataset_name}' exists at https://huggingface.co/bop-benchmark"
            )

    def _load_json(self, filepath: Path) -> Dict:
        with open(filepath) as f:
            return json.load(f)

    def _get_split_variants(self, split: str) -> List[Tuple[str, str]]:
        """
        Get ordered list of split variants (prioritized by preference).

        Returns:
            List of (suffix, description) tuples where suffix is the part after dataset_name_{split}_
        """
        if split == "test":
            return [
                ("bop19", "BOP19 subset (smaller)"),
                ("primesense_bop19", "T-LESS specific (smaller)"),
                ("all", "Full test set"),
                ("primesense_all", "T-LESS specific (full)"),
                ("primesense", "T-LESS generic"),
                ("", "Generic fallback"),
            ]
        elif split == "train":
            return [
                ("pbr", "PBR synthetic data"),
                ("real", "Real data"),
                ("primesense", "T-LESS specific"),
                ("synt", "YCB-Video synthetic"),
                ("render_reconst", "Other variants"),
                ("", "Generic fallback"),
            ]
        return []

    def _get_split_file_patterns(self, split: str) -> List[str]:
        """Get ordered list of file patterns for a given split (prioritized by preference)."""
        patterns = []
        for suffix, _ in self._get_split_variants(split):
            if suffix:
                patterns.append(f"{self.dataset_name}_{split}_{suffix}.zip")
            else:
                patterns.append(f"{self.dataset_name}_{split}.zip")
        return patterns

    def _get_split_directory_patterns(self, split: str) -> List[str]:
        """Get ordered list of directory patterns for a given split (prioritized by preference)."""
        patterns = []
        for suffix, _ in self._get_split_variants(split):
            if suffix:
                # Add both unprefixed and dataset-prefixed variants
                patterns.append(f"{split}_{suffix}")
                patterns.append(f"{self.dataset_name}_{split}_{suffix}")
            else:
                # Add generic split name
                patterns.append(split)
                patterns.append(f"{self.dataset_name}_{split}")
        return patterns

    def _get_split_path(self) -> Path:
        """Get the path to the split directory."""
        split_path = self.dataset_path / self.split
        if split_path.exists():
            return split_path

        alternatives = self._get_split_directory_patterns(self.split)

        for alt_name in alternatives:
            alt_path = self.dataset_path / alt_name
            if alt_path.exists():
                logger.info(f"Using alternative split directory: {alt_name}")
                return alt_path

        glob_alternatives = list(self.dataset_path.glob(f"{self.split}*"))
        if glob_alternatives:
            logger.info(f"Found split directory via glob: {glob_alternatives[0].name}")
            return glob_alternatives[0]

        available = [p.name for p in self.dataset_path.iterdir() if p.is_dir()]
        raise ValueError(
            f"Split directory not found: {split_path}\n"
            f"Available directories: {available}\n"
            f"Expected patterns: {alternatives[:5]}..."
        )

    def _load_object_models(self) -> Dict[int, ObjectModel]:
        """
        Load 3D object models from the models directory.

        Returns:
            Dictionary mapping object IDs to ObjectModel instances
        """
        models = {}
        models_path = self.dataset_path / "models"

        if not models_path.exists():
            alternatives = ["models_eval", "models_cad"]
            for alt in alternatives:
                alt_path = self.dataset_path / alt
                if alt_path.exists():
                    models_path = alt_path
                    break

        if not models_path.exists():
            return models

        models_info_path = models_path / "models_info.json"
        models_info = {}
        if models_info_path.exists():
            models_info = self._load_json(models_info_path)

        for model_file in sorted(models_path.glob("obj_*.ply")):
            try:
                import open3d as o3d

                obj_id = int(model_file.stem.split("_")[1])  # obj_000001.ply -> 1

                mesh = o3d.io.read_triangle_mesh(str(model_file))

                if mesh.is_empty():
                    continue

                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                triangles = np.asarray(mesh.triangles, dtype=np.int32)

                if mesh.has_vertex_normals():
                    logger.info(f"Loading normals for model {obj_id}")
                    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
                else:
                    logger.info(f"Computing normals for model {obj_id}")
                    mesh.compute_vertex_normals()
                    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

                metadata = models_info.get(str(obj_id), {})

                models[obj_id] = ObjectModel(
                    mesh_vertices=vertices,
                    mesh_faces=triangles,
                    mesh_normals=normals,
                    name=f"obj_{obj_id:06d}",
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {e}")
                continue

        return models

    def _get_scene_ids(self) -> List[int]:
        """Get list of available scene IDs."""
        split_path = self._get_split_path()

        scene_dirs = [
            d for d in split_path.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        scene_ids = sorted([int(d.name) for d in scene_dirs])

        return scene_ids

    def _load_scene_data(
        self, scene_id: int
    ) -> Iterator[Tuple[Frame, List[Pose], List[int]]]:
        """
        Load data from a specific scene.

        Args:
            scene_id: Scene ID to load

        Yields:
            Tuples of (Frame, list of ground truth Poses, list of object IDs)
        """
        split_path = self._get_split_path()
        scene_path = split_path / f"{scene_id:06d}"

        if not scene_path.exists():
            return

        scene_camera_path = scene_path / "scene_camera.json"
        scene_gt_path = scene_path / "scene_gt.json"

        if not scene_camera_path.exists():
            return

        scene_camera = self._load_json(scene_camera_path)
        scene_gt = self._load_json(scene_gt_path) if scene_gt_path.exists() else {}

        rgb_path = scene_path / "rgb"
        if not rgb_path.exists():
            return

        image_files = sorted(rgb_path.glob("*.png")) + sorted(rgb_path.glob("*.jpg"))

        for img_idx, img_file in enumerate(image_files):
            if self.max_samples and self._current_sample_idx >= self.max_samples:
                return

            img_id = int(img_file.stem)
            img_id_str = str(img_id)

            rgb_img = cv2.imread(str(img_file))
            if rgb_img is None:
                logger.warning(f"Failed to load image {img_file}")
                continue
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            depth_img = None
            if self.load_depth:
                depth_path = scene_path / "depth" / f"{img_id:06d}.png"
                if depth_path.exists():
                    try:
                        depth_img = cv2.imread(
                            str(depth_path), cv2.IMREAD_ANYDEPTH
                        )  # 16-bit
                        if depth_img is not None:
                            depth_img = (
                                depth_img.astype(np.float32) / 10000.0
                            )  # 10s of mm to meters
                    except Exception as e:
                        logger.warning(f"Failed to load depth {depth_path}: {e}")

            cam_data = scene_camera.get(img_id_str, {})
            cam_K = cam_data.get("cam_K", None)

            intrinsics = None
            if cam_K:
                intrinsics = np.array(cam_K, dtype=np.float64).reshape(
                    3, 3
                )  # [fx,0,cx,0,fy,cy,0,0,1]
            frame = Frame(
                rgb=rgb_img.astype(np.uint8),
                depth=depth_img,
                intrinsics=intrinsics,
                metadata={
                    "dataset": "BOP",
                    "split": self.split,
                    "scene_id": scene_id,
                    "image_id": img_id,
                    "source": str(self.dataset_path.name),
                },
            )

            gt_poses = []
            gt_obj_ids = []

            if img_id_str in scene_gt:
                for gt_annotation in scene_gt[img_id_str]:
                    obj_id = gt_annotation.get("obj_id")
                    cam_R_m2c = gt_annotation.get("cam_R_m2c")
                    cam_t_m2c = gt_annotation.get("cam_t_m2c")

                    if cam_R_m2c and cam_t_m2c:
                        rotation = np.array(cam_R_m2c, dtype=np.float64).reshape(3, 3)
                        translation = np.array(cam_t_m2c, dtype=np.float64).flatten()

                        pose = Pose(
                            rotation=rotation,
                            translation=translation,
                            metadata={
                                "obj_id": obj_id,
                                "scene_id": scene_id,
                                "image_id": img_id,
                            },
                        )

                        gt_poses.append(pose)
                        gt_obj_ids.append(obj_id)

            self._current_sample_idx += 1
            yield frame, gt_poses, gt_obj_ids

    def get_object_models(self) -> Dict[int, ObjectModel]:
        """
        Get all object models for the dataset.

        Returns:
            Dictionary mapping object IDs to ObjectModel instances
        """
        if self._models is None and self.load_models:
            self._models = self._load_object_models()
        return self._models or {}

    def get_object_ids(self) -> List[int]:
        """
        Get list of available object IDs in the dataset.

        Returns:
            List of object IDs
        """
        models = self.get_object_models()
        return list(models.keys())

    def get_object_model(self, obj_id: int) -> Optional[ObjectModel]:
        """
        Get a specific object model by ID.

        Args:
            obj_id: Object ID

        Returns:
            ObjectModel instance or None if not found
        """
        models = self.get_object_models()
        return models.get(obj_id)

    def stream_dataset(self) -> Iterator[Tuple[Frame, List[Pose], List[int]]]:
        """
        Stream all samples from the dataset.

        Yields:
            Tuples of (Frame, list of ground truth Poses, list of object IDs)
        """
        if self._scenes is None:
            if self.scene_id is not None:
                self._scenes = [self.scene_id]
            else:
                self._scenes = self._get_scene_ids()

        for scene_id in self._scenes:
            yield from self._load_scene_data(scene_id)

    def get_sample(
        self, scene_id: int, image_id: int
    ) -> Optional[Tuple[Frame, List[Pose], List[int]]]:
        """
        Get a specific sample from the dataset.

        Args:
            scene_id: Scene ID
            image_id: Image ID within the scene

        Returns:
            Tuple of (Frame, list of ground truth Poses, list of object IDs) or None
        """
        original_scene_id = self.scene_id
        original_max_samples = self.max_samples
        original_idx = self._current_sample_idx

        self.scene_id = scene_id
        self.max_samples = None
        self._current_sample_idx = 0

        try:
            for frame, poses, obj_ids in self._load_scene_data(scene_id):
                if frame.metadata["image_id"] == image_id:
                    return frame, poses, obj_ids
            return None
        finally:
            self.scene_id = original_scene_id
            self.max_samples = original_max_samples
            self._current_sample_idx = original_idx

    def process(self, input_data=None) -> Iterator[Tuple[Frame, List[Pose], List[int]]]:
        """
        Process method for pipeline compatibility with optional Rerun visualization.

        Args:
            input_data: Ignored for dataset loader

        Returns:
            Generator yielding (Frame, ground truth Poses, object IDs) tuples
        """
        if self.enable_rerun_logging and self.load_models:
            # Preload object models to avoid delays in first frames
            logger.info("Preloading object models for visualization...")
            all_models = self.get_object_models()
            logger.info(f"Loaded {len(all_models)} object models")

        sample_count = 0
        for frame, gt_poses, obj_ids in self.stream_dataset():
            sample_count += 1

            # Log to Rerun if enabled
            if self.enable_rerun_logging and self.rerun_logger:
                # Set time sequence for this frame
                self.rerun_logger.set_time_sequence("frame", sample_count)

                # Get object models for visualization
                object_models = None
                if gt_poses and obj_ids and self.load_models:
                    object_models = {}
                    for obj_id in obj_ids:
                        model = self.get_object_model(obj_id)
                        if model:
                            object_models[obj_id] = model

                # Log frame with poses and models
                self.rerun_logger.log_frame(
                    frame,
                    entity_path="world/camera",
                    log_poses=gt_poses,
                    log_object_ids=obj_ids,
                    object_models=object_models
                )

                logger.debug(f"Frame {sample_count}: {len(gt_poses)} objects (IDs: {obj_ids})")

            yield frame, gt_poses, obj_ids

        if self.enable_rerun_logging:
            logger.info(f"Visualized {sample_count} samples in Rerun")

    @property
    def num_scenes(self) -> int:
        if self._scenes is None:
            self._scenes = self._get_scene_ids()
        return len(self._scenes)

    @property
    def available_models(self) -> List[int]:
        models = self.get_object_models()
        return list(models.keys())
