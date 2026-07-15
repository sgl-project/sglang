"""Correctness test for the SM100 CuTe DSL KDA prefill pipeline.

Validates ``chunk_kda_cutedsl`` (the ``kda_blackwell`` package: fused Triton
prologue -> kkt_inv_uw -> h -> o) against the token-by-token
``fused_recurrent_kda`` Triton reference. Mirrors ``test_gdn_prefill_cutedsl.py``.

KDA differs from GDN by a PER-CHANNEL decay gate (g is [T, H, K], not scalar
[T, H]); otherwise the chunk pipeline, head dims, and recurrent-state layout
[N, H, V, K] are identical.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

# CuteDSL prefill kernel only exists on Blackwell. Single-GPU kernel-unit suite,
# same slot as the GDN prefill test.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10):
    pytest.skip(
        "KDA CuteDSL prefill requires CUDA SM10x (Blackwell).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.linear.kda_blackwell import (  # noqa: E402
    chunk_kda_cutedsl,
    prepare_metadata,
)
from sglang.srt.layers.attention.fla.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.fla.kda import fused_recurrent_kda  # noqa: E402


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
def test_kda_chunk_cutedsl_correctness(num_seqs: int):
    torch.manual_seed(num_seqs)
    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to("cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    # KDA shares the head count across q/k/v (G=1). CuteDSL prefill hard-requires
    # head_k_dim == head_v_dim == 128.
    num_heads = 8
    head_dim = 128
    scale = head_dim**-0.5

    q = _l2norm(torch.randn(1, total_tokens, num_heads, head_dim, device="cuda"))
    k = _l2norm(torch.randn(1, total_tokens, num_heads, head_dim, device="cuda"))
    v = torch.randn(1, total_tokens, num_heads, head_dim, device="cuda")

    # Per-channel KDA gate. Mild gates keep the kernel's externalized per-channel
    # pre-scaling (qg2 = scale*q*exp(g_cu - g_last), unbounded >= 1) inside fp32
    # range -- this matches real Kimi-Linear retention gates. Large per-chunk gate
    # spans are a known limitation (B2 TODO: clamp / sub-chunk normalize).
    A_log = torch.randn(num_heads, device="cuda") * 0.5 - 1.5
    dt_bias = torch.randn(num_heads, head_dim, device="cuda") * 0.1
    g_raw = torch.randn(1, total_tokens, num_heads, head_dim, device="cuda")
    g_act = -A_log.exp().view(1, 1, num_heads, 1) * F.softplus(
        g_raw + dt_bias.view(1, 1, num_heads, head_dim)
    )
    beta = torch.sigmoid(torch.randn(1, total_tokens, num_heads, device="cuda")).float()

    # Recurrent-state layout [N, H, V, K] (V-major) -- identical for the recurrent
    # reference and the cutedsl ht output (no transpose needed).
    initial_state = (
        torch.randn(num_seqs, num_heads, head_dim, head_dim, device="cuda") * 0.05
    ).float()

    # --- metadata helper must match the FLA chunkers the Triton path uses ---
    chunk_indices, chunk_offsets, _, total_chunks = prepare_metadata(cu_seqlens)
    torch.cuda.synchronize()
    expected_indices = prepare_chunk_indices(cu_seqlens, 64)
    expected_offsets = prepare_chunk_offsets(cu_seqlens, 64)
    torch.testing.assert_close(chunk_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(chunk_indices[:total_chunks], expected_indices)

    # --- reference: token-by-token recurrent kernel (ground truth) ---
    ref_o, ref_state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g_act,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens.long(),
    )

    # --- cutedsl chunk prefill (q/k already L2-normed; pass g_act directly) ---
    o, ht = chunk_kda_cutedsl(
        q[0].bfloat16(),
        k[0].bfloat16(),
        v[0].bfloat16(),
        g_act[0].float(),
        beta[0].float(),
        initial_state.clone(),
        cu_seqlens,
        scale,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(o).all(), "cutedsl output has non-finite values"
    assert torch.isfinite(ht).all(), "cutedsl final state has non-finite values"

    o_error = (o.float() - ref_o[0].float()).abs()
    state_error = (ht.float() - ref_state.float()).abs()
    # bf16 MMA + Newton-Schulz inverse noise; matches the B200 e2e validation
    # (o ~5e-4, ht ~4e-3) with margin.
    assert o_error.max().item() < 1e-2
    assert o_error.mean().item() < 1e-3
    assert state_error.max().item() < 5e-2
    assert state_error.mean().item() < 5e-3


def test_kda_chunk_cutedsl_internal_gate_activation():
    """The A_log/dt_bias gate activation inside chunk_kda_cutedsl must match
    feeding a pre-activated gate."""
    torch.manual_seed(0)
    T = 256
    num_heads = 8
    head_dim = 128
    scale = head_dim**-0.5

    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    q = _l2norm(torch.randn(1, T, num_heads, head_dim, device="cuda")).bfloat16()
    k = _l2norm(torch.randn(1, T, num_heads, head_dim, device="cuda")).bfloat16()
    v = torch.randn(1, T, num_heads, head_dim, device="cuda").bfloat16()
    A_log = torch.randn(num_heads, device="cuda") * 0.5 - 1.5
    dt_bias = torch.randn(num_heads, head_dim, device="cuda") * 0.1
    g_raw = torch.randn(1, T, num_heads, head_dim, device="cuda")
    beta = torch.sigmoid(torch.randn(1, T, num_heads, device="cuda")).float()
    h0 = torch.zeros(1, num_heads, head_dim, head_dim, device="cuda")

    g_pre = -A_log.exp().view(1, num_heads, 1) * F.softplus(
        g_raw[0] + dt_bias.view(1, num_heads, head_dim)
    )
    o_pre, ht_pre = chunk_kda_cutedsl(
        q[0], k[0], v[0], g_pre.float(), beta[0], h0.clone(), cu_seqlens, scale
    )
    o_int, ht_int = chunk_kda_cutedsl(
        q[0],
        k[0],
        v[0],
        g_raw[0].float(),
        beta[0],
        h0.clone(),
        cu_seqlens,
        scale,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(o_int.float(), o_pre.float(), atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(ht_int.float(), ht_pre.float(), atol=2e-3, rtol=1e-2)


def test_kda_chunk_cutedsl_realistic_gate():
    """Real Kimi-Linear retention gates are far stronger than the mild gates in
    the tests above (exp(A_log) ~ 0.2 there vs up to ~200 in the model). Strong
    gates make the per-chunk cumulative decay span large; a chunk-global g_last
    reference would overflow the externalized exp(g_cu - g_last) pre-scaling.
    The intra-chunk matrices use a sub-chunk-normalized path instead, so this
    must stay finite and match the recurrent reference."""
    torch.manual_seed(0)
    num_heads = 8
    head_dim = 128
    scale = head_dim**-0.5
    T = 128
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    q = _l2norm(torch.randn(1, T, num_heads, head_dim, device="cuda"))
    k = _l2norm(torch.randn(1, T, num_heads, head_dim, device="cuda"))
    v = torch.randn(1, T, num_heads, head_dim, device="cuda")
    # exp(A_log) ~ exp(1.5) ~ 4.5 mean (real model reaches ~200), vs ~0.22 above.
    A_log = torch.randn(num_heads, device="cuda") * 0.5 + 1.5
    dt_bias = torch.randn(num_heads, head_dim, device="cuda") * 0.1
    g_raw = torch.randn(1, T, num_heads, head_dim, device="cuda")
    g_act = -A_log.exp().view(1, 1, num_heads, 1) * F.softplus(
        g_raw + dt_bias.view(1, 1, num_heads, head_dim)
    )
    beta = torch.sigmoid(torch.randn(1, T, num_heads, device="cuda")).float()
    h0 = torch.zeros(1, num_heads, head_dim, head_dim, device="cuda").float()

    ref_o, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g_act,
        beta=beta,
        scale=scale,
        initial_state=h0.clone(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens.long(),
    )
    o, ht = chunk_kda_cutedsl(
        q[0].bfloat16(),
        k[0].bfloat16(),
        v[0].bfloat16(),
        g_act[0].float(),
        beta[0].float(),
        h0.clone(),
        cu_seqlens,
        scale,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(o).all() and torch.isfinite(ht).all()
    assert (o.float() - ref_o[0].float()).abs().max().item() < 1e-2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
