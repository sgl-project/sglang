"""ROCm gfx950 grouped-head split-KV verify attention tests for GQA DSpark drafts."""

import sys

import pytest
import torch
from sglang.kernels.ops.attention.dspark_draft_attn_hip import (
    verify_gqa_draft_fwd_hip,
)
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()),
    reason="ROCm gfx950 kernel",
)


@pytest.mark.parametrize(
    ("h_q", "h_kv", "head_dim", "window", "is_causal", "fp8_kv", "k_scale", "v_scale"),
    [
        (12, 2, 64, 2048, True, True, 1.0, 1.0),
        (12, 2, 64, 2048, True, True, 0.5, 0.25),
        (12, 2, 64, 2048, False, True, 1.0, 1.0),
        (12, 2, 64, 2048, True, False, 1.0, 1.0),
        (12, 2, 64, -1, True, True, 1.0, 1.0),
        (6, 2, 64, 512, True, True, 1.0, 1.0),
        (8, 2, 128, 512, True, True, 1.0, 1.0),
    ],
)
def test_verify_gqa_draft_matches_extend_attention(
    h_q: int,
    h_kv: int,
    head_dim: int,
    window: int,
    is_causal: bool,
    fp8_kv: bool,
    k_scale: float,
    v_scale: float,
) -> None:
    """Parity against extend_attention_fwd across empty, partial, full, and over-window prefixes."""
    torch.manual_seed(42)
    device = "cuda"
    l_ext = 6
    kv_lens = [0, 1, 17, 64, 512, 2048] if window != 512 else [0, 1, 17, 512, 600]
    bs = len(kv_lens)
    total_q = bs * l_ext
    total_kv = sum(kv_lens)
    pool_size = max(total_kv + 32, 4096)

    q = torch.randn((total_q, h_q, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((total_q, h_kv, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn((total_q, h_kv, head_dim), dtype=torch.bfloat16, device=device)
    k_raw = (
        torch.randn((pool_size, h_kv, head_dim), dtype=torch.bfloat16, device=device)
        * 0.5
    )
    v_raw = (
        torch.randn((pool_size, h_kv, head_dim), dtype=torch.bfloat16, device=device)
        * 0.5
    )
    k_buf = k_raw.to(torch.float8_e4m3fn) if fp8_kv else k_raw
    v_buf = v_raw.to(torch.float8_e4m3fn) if fp8_kv else v_raw

    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=device) * l_ext
    kv_indptr = torch.tensor(
        [0] + [sum(kv_lens[: i + 1]) for i in range(bs)],
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.arange(0, total_kv, dtype=torch.int64, device=device) % pool_size
    sm_scale = head_dim**-0.5

    o_ref = torch.empty_like(q)
    extend_attention_fwd(
        q,
        k,
        v,
        o_ref,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        is_causal,
        None,
        l_ext,
        k_scale,
        v_scale,
        sm_scale,
        sliding_window_size=window,
        bf16_dot=True,
    )
    o_got = torch.empty_like(q)
    # Pass max_bs > bs to exercise smaller batches slicing the shared flat scratch.
    handled = verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o_got,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        is_causal,
        None,
        l_ext,
        k_scale,
        v_scale,
        sm_scale,
        sliding_window_size=window,
        max_bs=bs + 8,
        bf16_dot=True,
    )
    assert handled
    assert (o_got.float() - o_ref.float()).abs().max().item() < 2e-2


def test_verify_gqa_draft_cuda_graph_capture_and_replay() -> None:
    """Zero host syncs in the gate/launch path allow CUDA graph capture and replay."""
    torch.manual_seed(7)
    device = "cuda"
    bs, l_ext, h_q, h_kv, d, window = 4, 6, 12, 2, 64, 2048
    kv_len = 512
    total_q = bs * l_ext
    q = torch.randn((total_q, h_q, d), dtype=torch.bfloat16, device=device)
    k = torch.randn((total_q, h_kv, d), dtype=torch.bfloat16, device=device)
    v = torch.randn((total_q, h_kv, d), dtype=torch.bfloat16, device=device)
    k_buf = (
        torch.randn((4096, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    v_buf = (
        torch.randn((4096, h_kv, d), dtype=torch.bfloat16, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=device) * l_ext
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=device) * kv_len
    kv_indices = torch.arange(0, bs * kv_len, dtype=torch.int64, device=device)
    o_got = torch.empty_like(q)
    sm_scale = d**-0.5
    k_scale, v_scale = 1.0, 1.0

    assert verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o_got,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        k_scale,
        v_scale,
        sm_scale,
        sliding_window_size=window,
        max_bs=bs + 4,
        bf16_dot=True,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        handled = verify_gqa_draft_fwd_hip(
            q,
            k,
            v,
            o_got,
            k_buf,
            v_buf,
            qo_indptr,
            kv_indptr,
            kv_indices,
            None,
            True,
            None,
            l_ext,
            1.0,
            1.0,
            sm_scale,
            sliding_window_size=window,
            max_bs=bs + 4,
            bf16_dot=True,
        )
    assert handled
    q.copy_(torch.randn_like(q))
    graph.replay()
    o_ref = torch.empty_like(q)
    extend_attention_fwd(
        q,
        k,
        v,
        o_ref,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        k_scale,
        v_scale,
        sm_scale,
        sliding_window_size=window,
        bf16_dot=True,
    )
    assert (o_got.float() - o_ref.float()).abs().max().item() < 2e-2


def test_verify_gqa_draft_rejects_unsupported_contracts() -> None:
    """Custom masks, empty rails, short windows, sinks, and logit_cap fall back cleanly."""
    device = "cuda"
    bs, l_ext, h_q, h_kv, d = 2, 6, 12, 2, 64
    q = torch.randn((bs * l_ext, h_q, d), dtype=torch.bfloat16, device=device)
    k = torch.randn((bs * l_ext, h_kv, d), dtype=torch.bfloat16, device=device)
    v = torch.randn((bs * l_ext, h_kv, d), dtype=torch.bfloat16, device=device)
    o = torch.empty_like(q)
    k_buf = torch.randn((128, h_kv, d), dtype=torch.bfloat16, device=device)
    v_buf = torch.randn((128, h_kv, d), dtype=torch.bfloat16, device=device)
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=device) * l_ext
    kv_indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device=device)
    kv_indices = torch.arange(64, dtype=torch.int64, device=device)

    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        torch.ones(1, dtype=torch.int8, device=device),
        True,
        None,
        l_ext,
        1.0,
        1.0,
    )
    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        torch.empty(0, dtype=torch.int64, device=device),
        None,
        True,
        None,
        l_ext,
        1.0,
        1.0,
    )
    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        1.0,
        1.0,
        sliding_window_size=l_ext - 1,
    )
    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        1.0,
        1.0,
        sinks=torch.zeros(h_q, dtype=torch.bfloat16, device=device),
    )
    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        1.0,
        1.0,
        logit_cap=30.0,
    )
    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        "invalid",  # type: ignore[arg-type]
        1.0,
        1.0,
    )
    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        1.0,
        1.0,
        xai_temperature_len=128,
    )
    assert not verify_gqa_draft_fwd_hip(
        q[:-1],
        k[:-1],
        v[:-1],
        o[:-1],
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        1.0,
        1.0,
    )
    assert not verify_gqa_draft_fwd_hip(
        q,
        k,
        v,
        o,
        k_buf,
        v_buf,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        True,
        None,
        l_ext,
        torch.tensor(1.0, device=device),  # type: ignore[arg-type]
        1.0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
