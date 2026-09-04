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

"""CUDA-graph regressions for #28371: dynamic chunked-SGMV LoRA segments."""

import pytest
import torch

from sglang.kernels.ops.gemm.chunked_sgmv_expand import (
    _chunked_lora_expand_kernel,
    chunked_sgmv_lora_expand_forward,
)
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
    _chunked_lora_shrink_kernel,
    chunked_sgmv_lora_shrink_forward,
)
from sglang.kernels.ops.gemm.kv_b_lora_absorbed import (
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
)
from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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

    server_args = type("ServerArgs", (), {"max_lora_chunk_size": 16})
    backend = ChunkedSgmvLoRABackend(
        max_loras_per_batch=NUM_LORAS,
        device=torch.device("cuda"),
        server_args=server_args,
    )
    backend.init_cuda_graph_batch_info(max_bs_in_cuda_graph=BS, num_tokens_per_req=1)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
