"""
Tests for VideoReaderNode.

Tests video reading functionality including sample video download.
"""

from pathlib import Path
from unittest.mock import patch

from regenbogen.nodes import VideoReaderNode


def test_video_reader_with_path():
    """Test VideoReaderNode with a provided path."""
    test_path = "/tmp/test_video.mp4"
    node = VideoReaderNode(video_path=test_path)
    assert node.video_path == test_path


def test_video_reader_downloads_sample():
    """Test VideoReaderNode downloads sample video when no path provided."""
    with patch.object(VideoReaderNode, "_download_sample_video") as mock_download:
        mock_download.return_value = "/tmp/sample.mp4"
        node = VideoReaderNode()
        mock_download.assert_called_once()
        assert node.video_path == "/tmp/sample.mp4"


def test_download_sample_video_caching():
    """Test that sample video is cached after first download."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("regenbogen.nodes.video_reader.Path.home") as mock_home:
            mock_home.return_value = Path(tmpdir)

            cache_dir = Path(tmpdir) / ".cache" / "regenbogen" / "sample_videos"
            cache_dir.mkdir(parents=True, exist_ok=True)
            video_path = cache_dir / "person-bicycle-car-detection.mp4"
            video_path.write_text("fake video content")

            with patch("regenbogen.nodes.video_reader.urlretrieve") as mock_urlretrieve:
                node = VideoReaderNode()
                mock_urlretrieve.assert_not_called()
                assert "person-bicycle-car-detection.mp4" in node.video_path


def test_download_sample_video_downloads_when_not_cached():
    """Test that sample video is downloaded when not in cache."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("regenbogen.nodes.video_reader.Path.home") as mock_home:
            mock_home.return_value = Path(tmpdir)

            with patch("regenbogen.nodes.video_reader.urlretrieve") as mock_urlretrieve:
                VideoReaderNode()
                mock_urlretrieve.assert_called_once()
                call_args = mock_urlretrieve.call_args[0]
                assert "person-bicycle-car-detection.mp4" in call_args[0]
                assert "person-bicycle-car-detection.mp4" in str(call_args[1])
