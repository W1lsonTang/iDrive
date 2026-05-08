"""Decision-level late fusion combining visual and physiological probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .time_sync import ChannelSample


@dataclass
class FusionResult:
    """Result of a single fusion decision."""

    probability: Optional[float]
    label: int  # 1 drowsy, 0 alert, -1 unknown
    visual_prob: Optional[float]
    physio_prob: Optional[float]
    visual_weight: float
    physio_weight: float
    visual_ok: bool
    physio_ok: bool


class LateFusion:
    """Quality-aware weighted late fusion for visual + physio drowsiness probabilities."""

    def __init__(
        self,
        visual_weight: float = 0.6,
        physio_weight: float = 0.4,
        threshold: float = 0.5,
    ) -> None:
        total = float(visual_weight) + float(physio_weight)
        if total <= 0.0:
            raise ValueError("Fusion weights must sum to a positive value.")
        self.base_visual_w = float(visual_weight) / total
        self.base_physio_w = float(physio_weight) / total
        self.threshold = float(threshold)

    def fuse(
        self,
        visual: Optional[ChannelSample],
        physio: Optional[ChannelSample],
    ) -> FusionResult:
        visual_ok = visual is not None and visual.quality_ok
        physio_ok = physio is not None and physio.quality_ok

        visual_prob = visual.probability if visual is not None else None
        physio_prob = physio.probability if physio is not None else None

        w_v, w_p = self._adaptive_weights(visual_ok, physio_ok)

        if visual_ok and physio_ok and visual_prob is not None and physio_prob is not None:
            fused = w_v * float(visual_prob) + w_p * float(physio_prob)
        elif visual_ok and visual_prob is not None:
            fused = float(visual_prob)
        elif physio_ok and physio_prob is not None:
            fused = float(physio_prob)
        else:
            return FusionResult(
                probability=None,
                label=-1,
                visual_prob=visual_prob,
                physio_prob=physio_prob,
                visual_weight=w_v,
                physio_weight=w_p,
                visual_ok=visual_ok,
                physio_ok=physio_ok,
            )

        label = 1 if fused >= self.threshold else 0
        return FusionResult(
            probability=fused,
            label=label,
            visual_prob=visual_prob,
            physio_prob=physio_prob,
            visual_weight=w_v,
            physio_weight=w_p,
            visual_ok=visual_ok,
            physio_ok=physio_ok,
        )

    def _adaptive_weights(self, visual_ok: bool, physio_ok: bool) -> tuple[float, float]:
        if visual_ok and physio_ok:
            return self.base_visual_w, self.base_physio_w
        if visual_ok and not physio_ok:
            return 1.0, 0.0
        if physio_ok and not visual_ok:
            return 0.0, 1.0
        return 0.0, 0.0
