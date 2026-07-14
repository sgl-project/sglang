# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GPU integration test for decode context-parallel (DCP) support in the
TRT-LLM sparse MLA decode path (`DeepseekSparseAttnBackend._forward_trtllm` /
`_dcp_shard_page_table_decode`, dsa_backend.py).

Drives the *real* `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
kernel plus the real `_dcp_shard_page_table_decode` helper and the real
`fixup_zero_kv_rows` zero-KV-row fixup, and checks that splitting a decode
step's top-k selection across `dcp_size` simulated ranks and merging the
partial (out, lse) results via the same base-2 online-softmax combine
`cp_lse_ag_out_rs_mla` uses (layers/dcp/comm.py, exp2/log2 convention)
reproduces the unsharded (dcp_size=1) output.

This exists because two non-obvious correctness bugs were caught only by
running the real kernel, not by static review of the DCP address-translation
convention used elsewhere in the codebase:

  1. The TRT-LLM sparse MLA kernel's returned LSE is natural-log (base-e),
     while `cp_lse_ag_out_rs_mla` (the merge every MLA-family DCP backend
     shares) expects base-2 (log2/exp2) LSE, matching FlashInfer-MLA's/
     FlashMLA's own convention. Feeding the raw LSE through unconverted
     silently produces wrong combine weights.
  2. The paged KV-cache allocator scales both `page_size` and `size` by
     `dcp_size` (mem_cache/kv_cache_configurator.py), while the physical
     `DSATokenToKVPool` tensor stays at the *undivided* row count
     (mem_cache/allocator/paged.py::PagedTokenToKVPoolAllocator.alloc: the
     dcp_size factor cancels out of `num_pages = size // page_size`) -- so
     `out_cache_loc` values range up to `dcp_size` times the physical
     tensor's row count, i.e. the cache is genuinely address-sharded, not
     replicated, and a selected physical loc's rank-local row is
     `loc // dcp_size` (owned iff `loc % dcp_size == dcp_rank`), matching
     layers/dcp/layout.py::filter_dcp_local_kv_indices's convention.
     A per-rank-buffer simulation faithful to that scheme (see
     `_per_rank_sharded_buffers` below) confirms dividing reproduces the
     unsharded ground truth (~0.4 abs diff, bf16 noise) while NOT dividing
     does not (~5.2 abs diff) -- the first version of this change dropped
     the division after validating against a single *shared* buffer for
     all simulated ranks, which cannot actually distinguish the two: with
     one shared buffer, "don't translate the index" is correct by
     construction regardless of what a real sharded deployment needs.

`_forward_trtllm` writes KV via `set_mla_kv_buffer` (memory_pool.py), which
performs neither the `dcp_size`-masked nor the `//dcp_size`-translated write
that this addressing scheme requires -- it writes the raw, unmasked
`out_cache_loc` directly. That is a separate, pre-existing gap in shared
memory-pool code (not introduced or fixed by the read-side change under test
here) that would need fixing before DCP + DSA is correct end-to-end; this
test only exercises the read/attention side.

Only the trtllm kernel invocation itself is exercised on real hardware here;
the actual cross-rank communication (cp_lse_ag_out_rs_mla's all-gather +
reduce-scatter) is single-process-simulated by holding all "ranks'" partials
in one process and applying the identical exp2/log2 combine math, since
`_forward_trtllm` only needs to be correct up to producing (out, lse) --
the communication primitive itself is exercised by
test/registered/kernels/test_dcp_lse_combine.py.

Usage:
    python -m pytest test_dsa_dcp_trtllm.py -v
    python test_dsa_dcp_trtllm.py
"""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.jit_kernel.fixup_zero_kv import fixup_zero_kv_rows
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    dsa_impl_capability,
)

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


class _FakeParallel:
    def __init__(self, dcp_size: int, dcp_rank: int):
        self.attn_dcp_size = dcp_size
        self.attn_dcp_rank = dcp_rank


def _base2_lse_merge(
    stacked_out: torch.Tensor, stacked_lse: torch.Tensor
) -> torch.Tensor:
    """Reference base-2 online-softmax combine, matching cp_lse_ag_out_rs_mla /
    correct_attn_out's exp2/log2 formula (layers/kernels/attention/dcp_kernels.py)."""
    global_lse = torch.logsumexp(stacked_lse * math.log(2), dim=0) / math.log(2)
    weight = torch.exp2(stacked_lse - global_lse).unsqueeze(-1)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.nan_to_num(stacked_out, nan=0.0, posinf=0.0, neginf=0.0)
    return (out * weight).sum(dim=0)


def _per_rank_sharded_buffers(
    physical_capacity: int,
    dcp_size: int,
    kv_cache_dim: int,
    real_page_size: int,
    device,
):
    """Build a faithful simulation of the paged allocator's DCP addressing:
    one `total_logical = physical_capacity * dcp_size`-row reference table
    (what a hypothetical unsharded/dcp_size=1 run would see -- the ground
    truth), and `dcp_size` separate `physical_capacity`-row per-rank buffers,
    where rank r's buffer holds `ref[loc]` at row `loc // dcp_size` for every
    `loc` with `loc % dcp_size == r` -- exactly the addressing
    `PagedTokenToKVPoolAllocator` implies (see module docstring).
    """
    total_logical = physical_capacity * dcp_size
    ref = torch.randn(total_logical, kv_cache_dim, dtype=torch.bfloat16, device=device)
    rank_bufs = []
    for r in range(dcp_size):
        buf = torch.zeros(
            physical_capacity, kv_cache_dim, dtype=torch.bfloat16, device=device
        )
        owned_loc = torch.arange(r, total_logical, dcp_size, device=device)
        buf[owned_loc // dcp_size] = ref[owned_loc]
        rank_bufs.append(buf)
    return ref, rank_bufs


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDsaTrtllmDcpDecode(unittest.TestCase):
    def setUp(self):
        supported, reason = dsa_impl_capability("trtllm")
        if not supported:
            self.skipTest(f"DSA trtllm decode impl not supported: {reason}")
        import flashinfer.decode  # noqa: F401

        self.d = __import__("flashinfer.decode", fromlist=["decode"])

    def _run_kernel(
        self,
        q,
        kv_cache_flat,
        real_page_size,
        kv_cache_dim,
        block_tables,
        seq_lens,
        topk,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        return_lse=False,
    ):
        num_pages = kv_cache_flat.shape[0] // real_page_size
        kv_cache = kv_cache_flat.view(num_pages, 1, real_page_size, kv_cache_dim)
        workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=q.device
        )
        return self.d.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables.unsqueeze(1).contiguous(),
            seq_lens=seq_lens,
            max_seq_len=topk,
            sparse_mla_top_k=topk,
            bmm1_scale=1.0,
            backend="trtllm-gen",
            return_lse=return_lse,
        )

    def test_dcp_shard_and_merge_matches_unsharded_ground_truth(self):
        torch.manual_seed(0)
        device = "cuda"
        batch_size = 8
        num_heads = 8
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        qk_nope_head_dim = 128
        head_dim_qk = kv_lora_rank + qk_rope_head_dim
        topk = 128
        real_page_size = 64
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        # Fixed physical per-rank capacity -- matches a real deployment where
        # each rank's physical KV pool size doesn't change with dcp_size; only
        # the addressable logical range (physical_capacity * dcp_size) does.
        physical_capacity = 32 * real_page_size

        q = torch.randn(
            batch_size, 1, num_heads, head_dim_qk, dtype=torch.bfloat16, device=device
        )
        # Valid top-k counts deliberately include values that won't divide
        # evenly across the dcp sizes under test (5, 1) to exercise the
        # zero-local-KV-row fixup path.
        valid_counts = torch.tensor([128, 100, 63, 5, 1, 40, 100, 128])

        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                ref, rank_bufs = _per_rank_sharded_buffers(
                    physical_capacity, dcp_size, kv_cache_dim, real_page_size, device
                )
                total_logical = physical_capacity * dcp_size

                topk_loc = torch.full((batch_size, topk), -1, dtype=torch.int64)
                for i, vc in enumerate(valid_counts):
                    topk_loc[i, : vc.item()] = torch.randperm(total_logical)[
                        : vc.item()
                    ]
                page_table_1 = topk_loc.to(torch.int32).to(device)
                seq_lens_full = valid_counts.to(torch.int32).to(device)

                out_full = self._run_kernel(
                    q,
                    ref,
                    real_page_size,
                    kv_cache_dim,
                    page_table_1,
                    seq_lens_full,
                    topk,
                    qk_nope_head_dim,
                    kv_lora_rank,
                    qk_rope_head_dim,
                ).squeeze(1)

                per_rank_out = []
                per_rank_lse = []
                for dcp_rank in range(dcp_size):
                    with patch(
                        "sglang.srt.layers.attention.dsa_backend.get_parallel",
                        return_value=_FakeParallel(dcp_size, dcp_rank),
                    ):
                        local_pt, local_seq_lens = (
                            DeepseekSparseAttnBackend._dcp_shard_page_table_decode(
                                None, page_table_1
                            )
                        )
                    out_r, lse_r = self._run_kernel(
                        q,
                        rank_bufs[dcp_rank],
                        real_page_size,
                        kv_cache_dim,
                        local_pt,
                        local_seq_lens,
                        topk,
                        qk_nope_head_dim,
                        kv_lora_rank,
                        qk_rope_head_dim,
                        return_lse=True,
                    )
                    out_r = out_r.squeeze(1)
                    cum_seq_lens = torch.arange(
                        batch_size + 1, dtype=torch.int32, device=device
                    )
                    # Real fixup, matching _forward_trtllm exactly.
                    fixup_zero_kv_rows(out_r, lse_r, local_seq_lens, cum_seq_lens, 1)
                    # Real base-e -> base-2 conversion, matching _forward_trtllm.
                    lse_r = lse_r * math.log2(math.e)

                    per_rank_out.append(out_r)
                    per_rank_lse.append(lse_r)

                merged = _base2_lse_merge(
                    torch.stack(per_rank_out, dim=0).float(),
                    torch.stack(per_rank_lse, dim=0).float(),
                )

                diff = (merged - out_full.float()).abs()
                self.assertTrue(
                    torch.isfinite(merged).all(),
                    msg=f"dcp_size={dcp_size}: non-finite merged output",
                )
                # bf16-precision kernel output; ~1.0 absolute tolerance is
                # rounding noise (values span roughly [-a few, +a few]).
                # Confirmed against the (wrong) not-divided variant on this
                # exact per-rank-sharded-buffer setup, which produces ~5.2
                # max abs diff -- over 5x this tolerance.
                self.assertLess(
                    diff.max().item(),
                    1.0,
                    msg=f"dcp_size={dcp_size}: max abs diff {diff.max().item()} "
                    "too large vs. unsharded ground truth",
                )


if __name__ == "__main__":
    unittest.main()
