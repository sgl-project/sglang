"""Guard the PaddleOCR-VL serving defaults that a refactor could silently drop.

Both settings here live in allowlists keyed by model type / architecture, so
nothing in PaddleOCR-VL's own code path breaks if an entry disappears — the
model just quietly serves slower.

A document page costs tens of milliseconds to resize + normalize + patchify, so
a single synchronous processor worker caps request throughput at
1 / preprocess_time no matter how much GPU is idle. Measured on an H200 with
1080p pages, opting into concurrent workers moved 32-way concurrent throughput
from 6.6 to 8.9 req/s and made single-stream TTFT stable (the single-worker
path alternated between ~282 ms and ~790 ms).

The opt-in lives on the class, and `QwenVLImageProcessor` grants it only to an
explicit `model_type` allowlist that PaddleOCR-VL is not on — so it is exactly
the kind of setting a refactor can silently drop.
"""

import pytest

from sglang.srt.configs.model_config import (
    multimodal_breakable_cuda_graph_supported_model_archs,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.paddleocr_vlm import PaddleOCRVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_processor_preprocesses_pages_concurrently():
    assert PaddleOCRVLImageProcessor.supports_mm_processor_concurrency is True
    assert PaddleOCRVLImageProcessor.auto_mm_processor_worker_num > 1
    assert PaddleOCRVLImageProcessor.auto_mm_io_worker_num > 1


def test_worker_count_stays_at_the_measured_optimum():
    """Two beat both one and four at 32-way concurrency on an H200, on document
    pages and on small images with long outputs alike. Past two, spreading
    request arrivals fragments GPU prefill batches faster than the extra overlap
    pays for itself."""
    assert PaddleOCRVLImageProcessor.auto_mm_processor_worker_num == 2


def test_io_worker_count_is_this_model_own():
    """Concurrency is the base default now, but the IO fan-out is not.

    Fetching a page is network-bound and cheap to overlap, so this model asks for
    more IO workers than the conservative base default. That number has to be
    declared here, not inherited.
    """
    assert PaddleOCRVLImageProcessor.__dict__["auto_mm_io_worker_num"] > (
        BaseMultimodalProcessor.auto_mm_io_worker_num
    ), "the IO fan-out must be declared on PaddleOCRVLImageProcessor itself"


def test_prefill_breakable_cuda_graph_is_allowlisted():
    """Breakable CG is the CUDA default but is switched off for every multimodal
    arch; PaddleOCR-VL opts back in so its text-only prefill keeps the graph.

    Measured on an H200 (2704-token text prompts): single-stream TTFT 16.1 ms
    without the graph, 11.5 ms with it. Image-carrying batches are rejected at
    replay and run eager either way, so this is a text/mixed-traffic win only.
    """
    assert (
        "PaddleOCRVLForConditionalGeneration"
        in multimodal_breakable_cuda_graph_supported_model_archs
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
