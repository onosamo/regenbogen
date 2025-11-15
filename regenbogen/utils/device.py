"""
Utility functions for device management in PyTorch.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_torch_device(device: str | None = None) -> str:
    """
    Get the appropriate PyTorch device with auto-detection.

    Checks device availability in the following priority:
    1. If device is explicitly provided, use it
    2. Check if CUDA is available
    3. Check if MPS (Apple Silicon) is available
    4. Fall back to CPU

    Args:
        device: Explicitly specified device ("cuda", "mps", "cpu", or None for auto)

    Returns:
        Device string to use ("cuda", "mps", or "cpu")
    """
    if device is not None:
        logger.info(f"Using explicitly specified device: {device}")
        return device

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Auto-detected device: cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Auto-detected device: mps")
    else:
        device = "cpu"
        logger.info("Auto-detected device: cpu")

    return device
