"""Request-scoped multimodal frontend wall time and workload metrics."""

import asyncio
import time
from contextlib import contextmanager

import numpy as np
import torch


def _feature_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        return sum(_feature_bytes(part) for part in value)
    # Measuring a transport must not reconstruct a tensor or synchronize a device.
    return 0


class MultimodalFrontendMetrics:
    def __init__(
        self, *, labels: dict[str, str], counter_cls, gauge_cls, histogram_cls
    ):
        # Request-custom/static labels may reuse these names; the frontend
        # dimensions must remain bounded and must not receive duplicate kwargs.
        labels = {
            name: value
            for name, value in labels.items()
            if name not in ("stage", "outcome", "modality")
        }
        self._labels = labels
        self._duration = histogram_cls(
            name="sglang:mm_frontend_stage_seconds",
            documentation="Per-request multimodal frontend stage wall time, including waits.",
            labelnames=[*labels, "stage", "outcome"],
            buckets=(
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2.5,
                5,
                10,
                30,
                60,
            ),
        )
        self._inflight = gauge_cls(
            name="sglang:mm_frontend_inflight",
            documentation="Requests currently inside a multimodal frontend stage.",
            labelnames=[*labels, "stage"],
            multiprocess_mode="livesum",
        )
        self._items = counter_cls(
            name="sglang:mm_frontend_items_total",
            documentation="Multimodal items returned by preprocessing or an encoder receiver.",
            labelnames=[*labels, "modality"],
        )
        self._bytes = counter_cls(
            name="sglang:mm_frontend_feature_bytes_total",
            documentation="Logical bytes in directly available feature tensors and arrays.",
            labelnames=[*labels, "modality"],
        )

    @contextmanager
    def record(self, stage: str):
        if stage not in ("preprocess", "hash", "dispatch"):
            raise ValueError(f"Unknown multimodal frontend stage: {stage}")
        inflight = self._inflight.labels(**self._labels, stage=stage)
        inflight.inc()
        started = time.perf_counter()
        outcome = "success"
        try:
            yield
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            inflight.inc(-1)
            self._duration.labels(**self._labels, stage=stage, outcome=outcome).observe(
                time.perf_counter() - started
            )

    def observe_inputs(self, mm_inputs) -> None:
        for item in mm_inputs.mm_items:
            modality = item.modality.name.lower()
            if modality not in ("image", "video", "audio"):
                modality = "other"
            self._items.labels(**self._labels, modality=modality).inc()
            size = _feature_bytes(item.feature) + _feature_bytes(
                item.precomputed_embeddings
            )
            self._bytes.labels(**self._labels, modality=modality).inc(size)
