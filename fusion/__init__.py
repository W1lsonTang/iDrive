"""Late-fusion utilities for the dual-channel drowsiness detection system."""

from .late_fusion import FusionResult, LateFusion
from .time_sync import ChannelBuffer, ChannelSample

__all__ = [
    "ChannelBuffer",
    "ChannelSample",
    "FusionResult",
    "LateFusion",
]
