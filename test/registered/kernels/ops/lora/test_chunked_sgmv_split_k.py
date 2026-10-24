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

"""Chunked-SGMV split-K shrink and compact-expand numerics."""

import math
from unittest.mock import patch

import pytest
import torch

import sglang.kernels.ops.lora.dense.chunked_sgmv_shrink as shrink_module
from sglang.kernels.ops.lora.dense.chunked_sgmv_expand import (
    chunked_sgmv_lora_expand_forward,
)
from sglang.kernels.ops.lora.dense.chunked_sgmv_shrink import (
    chunked_sgmv_lora_shrink_forward,
)
from sglang.srt.environ import envs
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cuda_ci

# Split-K is opt-in (SGLANG_ENABLE_LORA_CSGMV_SPLIT_K defaults to False).
register_cuda_ci(est_time=25, stage="extra-a", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BS = 8
NUM_LORAS = 5
MAX_RANK = 8


def _make_batch_info():
    return LoRABatchInfo(
        use_cuda_graph=True,
        bs=BS,
        num_segments=None,
        max_len=16,
        seg_lens=None,
        seg_indptr=torch.zeros(BS + 1, dtype=torch.int32, device="cuda"),
        weight_indices=torch.zeros(BS, dtype=torch.int32, device="cuda"),
        lora_ranks=torch.zeros(NUM_LORAS, dtype=torch.int32, device="cuda"),
        scalings=torch.ones(NUM_LORAS, dtype=torch.float, device="cuda"),
        permutation=torch.arange(BS, dtype=torch.int32, device="cuda"),
    )


def _set_segment_state(batch_info, *, active):
    if active:
        lora_ranks = [MAX_RANK] * NUM_LORAS
        weight_indices = [1, 2, 3, 4]
        seg_indptr = [0, 2, 4, 6, BS]
    else:
        lora_ranks = [0] * NUM_LORAS
        weight_indices = [0]
        seg_indptr = [0, BS]

    num_segments = len(weight_indices)
    batch_info.lora_ranks.copy_(
        torch.tensor(lora_ranks, dtype=torch.int32, device="cuda")
    )
    batch_info.weight_indices.zero_()
    batch_info.weight_indices[:num_segments].copy_(
        torch.tensor(weight_indices, dtype=torch.int32, device="cuda")
    )
    batch_info.seg_indptr.fill_(seg_indptr[-1])
    batch_info.seg_indptr[: num_segments + 1].copy_(
        torch.tensor(seg_indptr, dtype=torch.int32, device="cuda")
    )
    batch_info.num_segments = num_segments


def _set_mixed_segment_state(batch_info, *, active):
    _set_segment_state(batch_info, active=active)
    if active:
        batch_info.lora_ranks.copy_(
            torch.tensor([0, 8, 16, 32, 0], dtype=torch.int32, device="cuda")
        )
    batch_info.permutation.copy_(
        torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32, device="cuda")
    )
    batch_info.scalings.copy_(torch.tensor([1.0, -0.7, 1.0, 1.5, 0.3], device="cuda"))


def test_split_k_preserves_fp32_partials_and_reduction_tail():
    """Split boundaries and rank tails must match FP32 matmul before the final cast."""
    torch.manual_seed(5)
    batch_info = _make_batch_info()
    _set_mixed_segment_state(batch_info, active=True)
    k = 12345
    inputs = torch.randn(BS, k, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        NUM_LORAS, 96, k, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(k)
    # Cancellation loses the small terms if partials are rounded before reduction.
    starts = [0, 1792, 3584, 5376, 7168, 8960]
    small_terms = [start + 1 for start in starts] + [10753]
    weights[:, 0].zero_()
    inputs[:, starts + small_terms] = 1
    weights[:, 0, starts] = torch.tensor(
        [256, -256, 256, -256, 256, -256], device="cuda", dtype=torch.bfloat16
    )
    weights[:, 0, small_terms] = 0.125
    split_kernel = shrink_module._chunked_lora_shrink_split_k_kernel
    with (
        envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.override(False),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False),
        patch.object(
            shrink_module,
            "get_lora_shrink_config",
            return_value={"BLOCK_N": 16, "BLOCK_K": 256},
        ),
        patch.object(split_kernel, "run", wraps=split_kernel.run) as launches,
    ):
        actual = chunked_sgmv_lora_shrink_forward(
            inputs, weights, batch_info, 3, split_k=8
        )
        assert launches.call_count == 1
    for adapter, rank in enumerate((8, 16, 32), start=1):
        rows = batch_info.permutation[2 * (adapter - 1) : 2 * adapter].long()
        expected = (
            inputs[rows].float() @ weights[adapter, : 3 * rank].float().T
        ).bfloat16()
        torch.testing.assert_close(
            actual[rows, : 3 * rank], expected, rtol=0.025, atol=0.04
        )
        assert torch.equal(expected[:, 0], torch.full_like(expected[:, 0], 0.875))
        assert torch.equal(actual[rows, 0], expected[:, 0])


def test_compact_expand_preserves_ragged_slices_and_output_stride():
    """Empty and partial slices must preserve their neighbors and wider output padding."""
    torch.manual_seed(6)
    batch_info = _make_batch_info()
    _set_mixed_segment_state(batch_info, active=True)
    offsets = (0, 65, 65, 194)
    offsets_cpu = torch.tensor(offsets, dtype=torch.int32)
    offsets_gpu = offsets_cpu.cuda()
    inputs = torch.randn(BS, 96, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        NUM_LORAS, 194, 32, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(32)
    initial = torch.randn(BS, 211, device="cuda", dtype=torch.bfloat16)
    rectangular, compact = initial.clone(), initial.clone()
    chunked_sgmv_lora_expand_forward(
        inputs, weights, batch_info, offsets_gpu, 129, rectangular[:, :194]
    )
    chunked_sgmv_lora_expand_forward(
        inputs, weights, batch_info, offsets_gpu, 129, compact[:, :194], offsets_cpu
    )
    assert torch.equal(compact, rectangular)
    expected = initial.clone()
    for adapter, rank in enumerate((8, 16, 32), start=1):
        rows = batch_info.permutation[2 * (adapter - 1) : 2 * adapter].long()
        for s, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
            delta = (
                inputs[rows, s * rank : (s + 1) * rank].float()
                @ weights[adapter, lo:hi, :rank].float().T
            )
            expected[rows, lo:hi] += (delta * batch_info.scalings[adapter]).bfloat16()
    torch.testing.assert_close(compact, expected, rtol=0.03, atol=0.08)
    assert torch.equal(compact[:, 194:], initial[:, 194:])
    inactive_rows = batch_info.permutation[6:].long()
    assert torch.equal(compact[inactive_rows], initial[inactive_rows])


def test_deterministic_mode_keeps_original_shrink_path():
    """Deterministic mode must suppress both automatic and explicit split-K launches."""
    batch_info = _make_batch_info()
    _set_mixed_segment_state(batch_info, active=True)
    inputs = torch.randn(BS, 4096, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(NUM_LORAS, 96, 4096, device="cuda", dtype=torch.bfloat16) / 64
    split_kernel = shrink_module._chunked_lora_shrink_split_k_kernel
    # Installed tuning tables may disable the policy; pin the eligibility inputs.
    with (
        patch.object(
            shrink_module,
            "get_lora_shrink_config",
            return_value={"BLOCK_N": 16, "BLOCK_K": 256},
        ),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False),
    ):
        with envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.override(False):
            expected = chunked_sgmv_lora_shrink_forward(inputs, weights, batch_info, 3)
        with (
            envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.override(True),
            patch.object(split_kernel, "run", wraps=split_kernel.run) as launches,
        ):
            chunked_sgmv_lora_shrink_forward(inputs, weights, batch_info, 3)
            assert launches.call_count == 1
            assert launches.call_args.kwargs["SPLIT_K"] == 8
            chunked_sgmv_lora_shrink_forward(inputs, weights, batch_info, 3, split_k=4)
            assert launches.call_count == 2
            assert launches.call_args.kwargs["SPLIT_K"] == 4
            chunked_sgmv_lora_shrink_forward(inputs, weights, batch_info, 3, split_k=1)
            assert launches.call_count == 2
            for deterministic in (False, True):
                with (
                    envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(deterministic),
                    pytest.raises(ValueError, match="split_k must be positive"),
                ):
                    chunked_sgmv_lora_shrink_forward(
                        inputs, weights, batch_info, 3, split_k=0
                    )
            with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
                actual = chunked_sgmv_lora_shrink_forward(
                    inputs, weights, batch_info, 3, split_k=8
                )
            assert launches.call_count == 2
    for adapter, rank in enumerate((8, 16, 32), start=1):
        rows = batch_info.permutation[2 * (adapter - 1) : 2 * adapter].long()
        assert torch.equal(actual[rows, : 3 * rank], expected[rows, : 3 * rank])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
