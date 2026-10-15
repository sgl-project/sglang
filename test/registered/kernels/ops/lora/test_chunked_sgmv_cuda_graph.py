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

"""Numerical and CUDA-graph regressions for chunked-SGMV LoRA."""

import math
from unittest.mock import patch

import pytest
import torch

import sglang.kernels.ops.lora.dense.chunked_sgmv_shrink as shrink_module
from sglang.kernels.ops.lora.dense.chunked_sgmv_expand import (
    _chunked_lora_expand_kernel,
    chunked_sgmv_lora_expand_forward,
)
from sglang.kernels.ops.lora.dense.chunked_sgmv_shrink import (
    _chunked_lora_shrink_kernel,
    chunked_sgmv_lora_shrink_forward,
)
from sglang.kernels.ops.lora.dense.kv_b_lora_absorbed import (
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
)
from sglang.srt.environ import envs
from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

ATOL = 1e-3
RTOL = 1e-3
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


def _capture(call):
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def test_shrink_replay_uses_all_current_segments():
    """A one-segment graph must match eager shrink after replaying four adapters."""
    _chunked_lora_shrink_kernel._clear_cache()
    batch_info = _make_batch_info()
    inputs = torch.randn(BS, 64, dtype=torch.float16, device="cuda")
    weights = torch.randn(NUM_LORAS, MAX_RANK, 64, dtype=torch.float16, device="cuda")

    _set_segment_state(batch_info, active=True)
    expected = chunked_sgmv_lora_shrink_forward(
        inputs, weights, batch_info, num_slices=1
    ).clone()

    _set_segment_state(batch_info, active=False)
    graph, captured_output = _capture(
        lambda: chunked_sgmv_lora_shrink_forward(
            inputs, weights, batch_info, num_slices=1
        )
    )

    _set_segment_state(batch_info, active=True)
    captured_output.zero_()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured_output[:, :MAX_RANK],
        expected[:, :MAX_RANK],
        rtol=RTOL,
        atol=ATOL,
    )


def test_expand_replay_uses_all_current_segments():
    """A one-segment graph must match eager expand after replaying four adapters."""
    _chunked_lora_expand_kernel._clear_cache()
    batch_info = _make_batch_info()
    output_dim = 32
    inputs = torch.randn(BS, MAX_RANK, dtype=torch.float16, device="cuda")
    weights = torch.randn(
        NUM_LORAS, output_dim, MAX_RANK, dtype=torch.float16, device="cuda"
    )
    slice_offsets = torch.tensor([0, output_dim], dtype=torch.int32, device="cuda")
    base_output = torch.randn(BS, output_dim, dtype=torch.float16, device="cuda")
    graph_base_output = base_output.clone()

    _set_segment_state(batch_info, active=True)
    expected = chunked_sgmv_lora_expand_forward(
        inputs,
        weights,
        batch_info,
        slice_offsets,
        output_dim,
        base_output=base_output.clone(),
    ).clone()

    def run_graph():
        graph_base_output.copy_(base_output)
        return chunked_sgmv_lora_expand_forward(
            inputs,
            weights,
            batch_info,
            slice_offsets,
            output_dim,
            base_output=graph_base_output,
        )

    _set_segment_state(batch_info, active=False)
    graph, captured_output = _capture(run_graph)

    _set_segment_state(batch_info, active=True)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_output, expected, rtol=RTOL, atol=ATOL)


def test_prepare_batch_neutralizes_static_tail_segments():
    """A smaller replay batch must not expose stale adapter metadata."""

    class MockForwardBatch:
        def __init__(self, batch_size):
            self.batch_size = batch_size
            self.forward_mode = ForwardMode.DECODE

    with get_context().override_server_args(max_lora_chunk_size=16) as server_args:
        backend = ChunkedSgmvLoRABackend(
            max_loras_per_batch=NUM_LORAS,
            device=torch.device("cuda"),
            server_args=server_args,
        )
        backend.init_cuda_graph_batch_info(
            max_bs_in_cuda_graph=BS, num_tokens_per_req=1
        )
        lora_ranks = [MAX_RANK] * NUM_LORAS
        scalings = [1.0] * NUM_LORAS

        backend.prepare_lora_batch(
            forward_batch=MockForwardBatch(BS),
            weight_indices=[0, 1, 2, 3, 4, 0, 1, 2],
            lora_ranks=lora_ranks,
            scalings=scalings,
            use_cuda_graph=True,
        )
        backend.prepare_lora_batch(
            forward_batch=MockForwardBatch(2),
            weight_indices=[0, 0],
            lora_ranks=lora_ranks,
            scalings=scalings,
            use_cuda_graph=True,
        )
        torch.cuda.synchronize()

        assert backend.batch_info.num_segments == 1
        torch.testing.assert_close(
            backend.batch_info.weight_indices.cpu(),
            torch.tensor([0] * BS, dtype=torch.int32),
        )
        torch.testing.assert_close(
            backend.batch_info.seg_indptr.cpu(),
            torch.tensor([0, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32),
        )


def test_absorbed_kv_b_replay_uses_all_current_segments():
    """A one-segment graph must match eager MLA Q/V updates for four adapters."""
    batch_info = _make_batch_info()
    num_heads = 2
    qk_nope_head_dim = 16
    v_head_dim = 16
    kv_lora_rank = 32
    full_k_per_head = qk_nope_head_dim + v_head_dim

    q_nope = torch.randn(
        BS, num_heads, qk_nope_head_dim, dtype=torch.float16, device="cuda"
    )
    attn_output = torch.randn(
        BS, num_heads, kv_lora_rank, dtype=torch.float16, device="cuda"
    )
    a_buf = torch.randn(
        NUM_LORAS, MAX_RANK, kv_lora_rank, dtype=torch.float16, device="cuda"
    )
    b_buf = torch.randn(
        NUM_LORAS,
        num_heads * full_k_per_head,
        MAX_RANK,
        dtype=torch.float16,
        device="cuda",
    )
    base_q = torch.randn(
        BS, num_heads, kv_lora_rank, dtype=torch.float16, device="cuda"
    )
    base_v = torch.randn(BS, num_heads, v_head_dim, dtype=torch.float16, device="cuda")
    graph_base_q = base_q.clone()
    graph_base_v = base_v.clone()

    def run_kv_b(base_q_output, base_v_output):
        q_lora_a = step_a_q_fwd(q_nope, b_buf, batch_info, full_k_per_head)
        q_output = step_b_q_fwd(q_lora_a, a_buf, batch_info, base_q_output)
        v_lora_a = step_a_v_fwd(attn_output, a_buf, batch_info)
        v_output = step_b_v_fwd(
            v_lora_a,
            b_buf,
            batch_info,
            base_v_output,
            qk_nope_head_dim,
            v_head_dim,
        )
        return q_output, v_output

    _set_segment_state(batch_info, active=True)
    expected_q, expected_v = run_kv_b(base_q.clone(), base_v.clone())
    expected_q = expected_q.clone()
    expected_v = expected_v.clone()

    def run_graph():
        graph_base_q.copy_(base_q)
        graph_base_v.copy_(base_v)
        return run_kv_b(graph_base_q, graph_base_v)

    _set_segment_state(batch_info, active=False)
    graph, (captured_q, captured_v) = _capture(run_graph)

    _set_segment_state(batch_info, active=True)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_q, expected_q, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(captured_v, expected_v, rtol=RTOL, atol=ATOL)


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


def test_split_k_compact_replay_uses_current_metadata():
    """Captured split-K and compact expand must follow inactive and active adapter updates."""
    torch.manual_seed(7)
    batch_info = _make_batch_info()
    _set_mixed_segment_state(batch_info, active=False)
    inputs = torch.randn(BS, 4096, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(NUM_LORAS, 96, 4096, device="cuda", dtype=torch.bfloat16) / 64
    b = torch.randn(
        NUM_LORAS, 194, 32, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(32)
    offsets_cpu = torch.tensor([0, 65, 65, 194], dtype=torch.int32)
    offsets_gpu = offsets_cpu.cuda()
    base = torch.randn(BS, 211, device="cuda", dtype=torch.bfloat16)

    def run():
        h = chunked_sgmv_lora_shrink_forward(inputs, a, batch_info, 3, split_k=8)
        output = base.clone()
        chunked_sgmv_lora_expand_forward(
            h, b, batch_info, offsets_gpu, 129, output[:, :194], offsets_cpu
        )
        return output

    split_kernel = shrink_module._chunked_lora_shrink_split_k_kernel
    with (
        envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.override(False),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False),
        patch.object(split_kernel, "run", wraps=split_kernel.run) as launches,
    ):
        graph, captured = _capture(run)
        assert launches.call_count == 4
        for active in (True, False, True):
            _set_mixed_segment_state(batch_info, active=active)
            expected = run()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)


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
