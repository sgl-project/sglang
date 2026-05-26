"""DSV4 attention correctness — SWA + C4/C128 dispatch coverage.

Covers eager EXTEND/DECODE plus CUDA-graph-style capture/replay for the
SWA-only (compress_ratio=0) path of `DeepseekV4AttnBackend` through flash_mla
with the production packed FP8-nope/BF16-rope SWA cache, plus
metadata-level smoke tests for the C4 (compress_ratio=4) and C128
(compress_ratio=128) paths. The C4/C128 smoke tests bypass the production
`Compressor`/`C4Indexer` modules — they pre-populate the extra K cache via
the production pack+set path and verify only that the dispatch path
(metadata population, flash_mla integration with `extra_k_cache`) produces
shape-correct finite output. Compressor math correctness is a deferred
follow-up.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_FLASH_MLA_AVAILABLE = importlib.util.find_spec("flash_mla") is not None

from common.attention_methods.dsv4_attention import (  # noqa: E402
    DSV4_PAGE_SIZE,
    DSV4AttentionCase,
    make_dsv4_cases,
    run_dsv4_attention_case,
    run_dsv4_compress_smoke_case,
)
from common.runner_modes.cuda_graph_decode_runner import (  # noqa: E402
    run_dsv4_cuda_graph_decode_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(not _FLASH_MLA_AVAILABLE, "flash_mla is required for DSV4 SWA")
class TestDSV4AttentionBackendCorrectness(CustomTestCase):
    CASES = make_dsv4_cases("dsv4")
    CUDA_GRAPH_DECODE_CASES = (
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_decode_within_window",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
        ),
        DSV4AttentionCase(
            name="runner_cuda_graph_dsv4_decode_multi_request",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(32, 96),
        ),
    )
    # C4 (compress_ratio=4) and C128 (compress_ratio=128) dispatch smoke cases.
    # They populate the extra K cache directly via `set_extra_key_buffer` and
    # verify the `forward(compress_ratio=...)` dispatch produces finite output;
    # the production `Compressor`/`C4Indexer` math path is not exercised here.
    COMPRESS_SMOKE_CASES = (
        DSV4AttentionCase(
            name="dsv4_c4_extend_smoke",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
            extend_lens=(16,),
            compress_ratio=4,
        ),
        DSV4AttentionCase(
            name="dsv4_c4_decode_smoke",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
            compress_ratio=4,
        ),
        DSV4AttentionCase(
            name="dsv4_c128_extend_smoke",
            backend="dsv4",
            forward_mode=ForwardMode.EXTEND,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128,),
            extend_lens=(16,),
            compress_ratio=128,
        ),
        DSV4AttentionCase(
            name="dsv4_c128_decode_smoke",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(128,),
            compress_ratio=128,
        ),
    )

    def test_swa_only_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_attention_case(self, case)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsv4_cuda_graph_decode_case(self, case)

    def test_compress_dispatch_smoke_cases(self):
        for case in self.COMPRESS_SMOKE_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                compress_ratio=case.compress_ratio,
            ):
                run_dsv4_compress_smoke_case(self, case)


if __name__ == "__main__":
    unittest.main()
