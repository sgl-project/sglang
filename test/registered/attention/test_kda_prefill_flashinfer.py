"""Correctness tests for the FlashInfer CAKE recurrent-KDA prefill backend."""

from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b",
    runner_config="4-gpu-b200",
    disabled="CAKE KDA prefill is not in the pinned public FlashInfer build",
)

if not (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
):
    pytest.skip(
        "FlashInfer CAKE KDA prefill requires SM100 or SM103.",
        allow_module_level=True,
    )

from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (  # noqa: E402
    CakeKDAKernel,
    _get_flashinfer_kda_prefill_kernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import (  # noqa: E402
    TritonKDAKernel,
)

_available, _ = _get_flashinfer_kda_prefill_kernel()
if not _available:
    pytest.skip(
        "FlashInfer build does not expose the CAKE recurrent prefill backend.",
        allow_module_level=True,
    )

K = V = 128
LOWER_BOUND = -5.0


def _make_inputs(seq_lens, num_heads):
    num_sequences = len(seq_lens)
    total_tokens = sum(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    pool_size = num_sequences + 5
    cache_indices = torch.arange(
        pool_size - 1,
        pool_size - num_sequences - 1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    return dict(
        q=torch.randn(
            1, total_tokens, num_heads, K, device="cuda", dtype=torch.bfloat16
        ).contiguous(),
        k=torch.randn(
            1, total_tokens, num_heads, K, device="cuda", dtype=torch.bfloat16
        ).contiguous(),
        v=torch.randn(
            1, total_tokens, num_heads, V, device="cuda", dtype=torch.bfloat16
        ).contiguous(),
        g=torch.randn(
            1, total_tokens, num_heads, K, device="cuda", dtype=torch.bfloat16
        ).contiguous(),
        # Match Kimi K3's fused projection slice: raw BF16 logits with a wider
        # physical token-row pitch and unit head stride.
        beta=torch.randn(1, total_tokens, 32, device="cuda", dtype=torch.bfloat16)[
            :, :, 8 : 8 + num_heads
        ],
        A_log=(
            torch.randn(1, 1, num_heads, 1, device="cuda", dtype=torch.float32) * 0.2
        ).contiguous(),
        dt_bias=(
            torch.randn(num_heads * K, device="cuda", dtype=torch.float32) * 0.1
        ).contiguous(),
        state=(
            torch.randn(
                pool_size,
                num_heads,
                V,
                K,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        ).contiguous(),
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
    )


def _extend(kernel, data, state, seq_lens, **kwargs):
    if getattr(kernel, "supports_cake_route_telemetry", False):
        kwargs["layer_id"] = 7
    beta = data["beta"]
    if isinstance(kernel, TritonKDAKernel):
        beta = torch.sigmoid(beta)
    return kernel.extend(
        data["q"].clone(),
        data["k"].clone(),
        data["v"].clone(),
        data["g"].clone(),
        beta,
        ssm_states=state,
        cache_indices=data["cache_indices"],
        query_start_loc=data["cu_seqlens"],
        A_log=data["A_log"],
        dt_bias=data["dt_bias"],
        lower_bound=LOWER_BOUND,
        extend_seq_lens_cpu=seq_lens,
        **kwargs,
    )


@pytest.mark.parametrize(
    "num_heads,seq_lens",
    [(6, [128]), (12, [96]), (12, [64, 160])],
)
def test_kda_prefill_cake_matches_triton(num_heads, seq_lens):
    torch.manual_seed(num_heads + sum(seq_lens))
    data = _make_inputs(seq_lens, num_heads)
    cake = CakeKDAKernel()
    triton = TritonKDAKernel()

    state_ref = data["state"].clone()
    output_ref = _extend(triton, data, state_ref, seq_lens)
    state_cake = data["state"].clone()
    output_cake = _extend(cake, data, state_cake, seq_lens)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output_cake.float(), output_ref.float(), atol=1e-2, rtol=1e-2
    )
    selected = data["cache_indices"].long()
    torch.testing.assert_close(
        state_cake[selected].float(),
        state_ref[selected].float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_kda_prefill_cake_aligned_tracking_is_bitwise_identical():
    seq_lens = [64, 128]
    data = _make_inputs(seq_lens, 12)
    cake = CakeKDAKernel()
    initial_state = data["state"]
    state_untracked = initial_state.clone()
    state_tracked = initial_state.clone()
    empty_checkpoint_source = torch.empty(0, device="cuda", dtype=torch.int64)

    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("aligned tracking must not fall back to Triton"),
    ):
        output_untracked = _extend(cake, data, state_untracked, seq_lens)
        output_tracked, intermediate_tracked = _extend(
            cake,
            data,
            state_tracked,
            seq_lens,
            return_intermediate_states=True,
            track_ssm_h_src=empty_checkpoint_source,
        )
    torch.cuda.synchronize()

    assert torch.equal(output_tracked, output_untracked)
    assert torch.equal(state_tracked, state_untracked)
    assert intermediate_tracked.shape == (1, 0, 12, V, K)
    assert intermediate_tracked.dtype == torch.bfloat16
    assert intermediate_tracked.numel() == 0


def test_kda_prefill_cake_returns_native_interior_state_tracking():
    seq_lens = [65, 131]
    data = _make_inputs(seq_lens, 12)
    interior_checkpoint_source = torch.tensor([0, 4], device="cuda", dtype=torch.int64)
    checkpoint_cu_starts = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int64)
    state_ref = data["state"].clone()
    output_ref = _extend(
        TritonKDAKernel(),
        data,
        state_ref,
        seq_lens,
        return_intermediate_states=True,
    )
    state_cake = data["state"].clone()
    with patch.object(
        CakeKDAKernel,
        "_extend_triton",
        side_effect=AssertionError("native checkpoints must not fall back to Triton"),
    ):
        output_cake = _extend(
            CakeKDAKernel(),
            data,
            state_cake,
            seq_lens,
            return_intermediate_states=True,
            track_ssm_h_src=interior_checkpoint_source,
            cake_query_start_loc=data["cu_seqlens"].to(torch.int64),
            state_checkpoint_cu_starts=checkpoint_cu_starts,
            num_state_checkpoints=5,
            state_checkpoint_every_n_tokens=64,
        )
    torch.cuda.synchronize()

    out_cake, intermediate_cake = output_cake
    out_ref, intermediate_ref = output_ref
    torch.testing.assert_close(out_cake.float(), out_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        intermediate_cake.float(), intermediate_ref.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        state_cake.float(), state_ref.float(), atol=1e-2, rtol=1e-2
    )
    assert data["beta"].stride(-2) == 32


def test_kda_prefill_cake_falls_back_during_cuda_graph_capture():
    seq_lens = [96]
    data = _make_inputs(seq_lens, 12)
    initial_state = data["state"].clone()
    state_ref = initial_state.clone()
    output_ref = _extend(TritonKDAKernel(), data, state_ref, seq_lens)

    # Warm the Triton fallback before capture so graph construction itself
    # contains no JIT compilation.
    _extend(TritonKDAKernel(), data, initial_state.clone(), seq_lens)
    torch.cuda.synchronize()

    state_graph = initial_state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output_graph = _extend(CakeKDAKernel(), data, state_graph, seq_lens)

    # Replay from the same initial state and compare the captured Triton
    # fallback with an eager Triton step.
    state_graph.copy_(initial_state)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output_graph, output_ref)
    torch.testing.assert_close(state_graph, state_ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
