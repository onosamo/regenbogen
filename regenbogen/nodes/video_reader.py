"""
Video reader node for processing video files frame by frame.

This node reads frames from video files and converts them to Frame objects.
"""

import logging
from pathlib import Path
from typing import Iterator, Optional
from urllib.request import urlretrieve

import numpy as np

from ..core.node import Node
from ..interfaces import Frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
)


class VideoReaderNode(Node):
    """
    Node for reading frames from video files.

    Reads video frames and converts them to Frame objects with RGB data.
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        fps: Optional[float] = None,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize video reader node.

        Args:
            video_path: Path to the video file. If None, downloads a sample video.
            fps: Target FPS for frame extraction (None = use video's FPS)
            frame_skip: Skip every N frames (1 = process all frames)
            max_frames: Maximum number of frames to process (None = all frames)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.video_path = video_path or self._download_sample_video()
        self.fps = fps
        self.frame_skip = frame_skip
        self.max_frames = max_frames

        self._cap = None
        self._frame_count = 0
        self._processed_frames = 0

    def _download_sample_video(self) -> str:
        """
        Download sample video if no path is provided.

        Returns:
            Path to the downloaded sample video
        """
        cache_dir = Path.home() / ".cache" / "regenbogen" / "sample_videos"
        cache_dir.mkdir(parents=True, exist_ok=True)

        video_filename = "person-bicycle-car-detection.mp4"
        video_path = cache_dir / video_filename

        if not video_path.exists():
            logger.info(f"Downloading sample video from {DEFAULT_SAMPLE_VIDEO_URL}")
            logger.info(f"Saving to {video_path}")
            try:
                urlretrieve(DEFAULT_SAMPLE_VIDEO_URL, video_path)
                logger.info("Sample video downloaded successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download sample video from {DEFAULT_SAMPLE_VIDEO_URL}: {e}"
                )
        else:
            logger.info(f"Using cached sample video from {video_path}")

        return str(video_path)

    def _init_video_capture(self):
        """Initialize video capture."""
        import cv2

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _create_default_intrinsics(self) -> np.ndarray:
        """Create default camera intrinsics based on video dimensions."""
        # Use simple assumptions for focal length and principal point
        fx = fy = min(self._width, self._height)
        cx, cy = self._width / 2, self._height / 2

        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def read_frame(self) -> Optional[Frame]:
        """
        Read the next frame from video.

        Returns:
            Frame object with RGB data or None if no more frames
        """
        if self._cap is None:
            self._init_video_capture()

        if self.max_frames and self._processed_frames >= self.max_frames:
            return None

        for _ in range(self.frame_skip):
            ret, frame = self._cap.read()
            if not ret:
                return None
            self._frame_count += 1

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_frame = frame[:, :, ::-1]
        intrinsics = self._create_default_intrinsics()

        result_frame = Frame(
            rgb=rgb_frame.astype(np.uint8),
            intrinsics=intrinsics,
            metadata={
                "source": "video",
                "video_path": self.video_path,
                "frame_number": self._frame_count,
                "width": self._width,
                "height": self._height,
                "video_fps": self._video_fps,
            },
        )

        self._frame_count += 1
        self._processed_frames += 1

        return result_frame

    def read_all_frames(self) -> Iterator[Frame]:
        """
        Generator that yields all frames from the video.

        Yields:
            Frame objects with RGB data
        """
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def process(self, input_data=None) -> Iterator[Frame]:
        """
        Process method for pipeline compatibility.

        Args:
            input_data: Ignored for video reader

        Returns:
            Generator yielding Frame objects
        """
        return self.read_all_frames()

    def __del__(self):
        """Clean up video capture resources."""
        if hasattr(self, "_cap") and self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(
                    f"Error releasing video capture for {self.video_path}: {e}"
                )

    @property
    def total_frames(self) -> int:
        """Get total number of frames in video."""
        if self._cap is None:
            self._init_video_capture()
        return self._total_frames

    @property
    def video_fps(self) -> float:
        """Get video FPS."""
        if self._cap is None:
            self._init_video_capture()
        return self._video_fps
