import inspect
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_RELEASE_FILES = (
    "python/sglang/srt/mem_cache/deepseek_v4_shared.py",
    "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
    "python/sglang/srt/layers/attention/dsv4/indexer.py",
    "python/sglang/srt/layers/attention/dsv4/compressor_v2.py",
    "python/sglang/srt/models/deepseek_v4.py",
    "python/sglang/jit_kernel/dsv4/__init__.py",
    "python/sglang/jit_kernel/dsv4/attn.py",
    "python/sglang/jit_kernel/dsv4/compress.py",
    "python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh",
    "python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh",
)
_FORBIDDEN = (
    "SGLANG_DSV4_SHARED_SIGNAL_GATHER",
    "SGLANG_DSV4_SHARED_NCCL_GATHER",
    "SGLANG_DSV4_SHARED_ACTIVE_SWA",
    "SGLANG_DSV4_SHARED_SIGNAL_FENCE",
    "SGLANG_DSV4_SHARED_ASYNC_STAGE",
    "stage_slots_with_allgather",
    "stage_pages_with_allgather",
    "prepare_active_stage",
    "get_active_swa_stage",
    "get_active_stage",
    "get_compressor_active_write_info",
    "begin_active_chunk",
    "finish_active_chunk",
    "fused_store_cache_shared_active",
    "active_kvcache",
    "active_cache",
    "active_out_loc",
    "active_indices",
    "delay_shared_page_indices",
    "_dsv4_shared_write_fence_layer",
)


class TestDSV4SharedReleaseSurface(CustomTestCase):
    def test_rejected_experimental_paths_are_absent_from_production(self):
        violations = []
        for relative_path in _RELEASE_FILES:
            source = (_REPOSITORY_ROOT / relative_path).read_text()
            for forbidden in _FORBIDDEN:
                if forbidden in source:
                    violations.append(f"{relative_path}: {forbidden}")

        self.assertEqual(violations, [])

    def test_sparse_swa_plan_does_not_read_a_cuda_scalar_on_the_host(self):
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            build_swa_token_ids,
        )

        self.assertNotIn(".item()", inspect.getsource(build_swa_token_ids))

    def test_sparse_swa_total_accepts_forward_batch_cpu_metadata_types(self):
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            compute_sparse_prefill_total_swa,
        )

        self.assertEqual(
            compute_sparse_prefill_total_swa(
                seq_lens_cpu=torch.tensor([128, 512], dtype=torch.int32),
                extend_seq_lens_cpu=[32, 256],
                swa_window_size=64,
            ),
            95 + 319,
        )


if __name__ == "__main__":
    unittest.main()
