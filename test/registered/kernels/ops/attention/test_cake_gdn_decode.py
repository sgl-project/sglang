"""Cake GDN decode / MTP verify through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backends; that the
facade output, the in-place K-last BF16 state pool and the MTP intermediate
buffer are bitwise identical to ``flashinfer.gdn_decode.
gated_delta_rule_decode_pretranspose(backend="cake_gdn")``; that the result
matches a pure-torch gated delta step within BF16 tolerance; that CUDA-graph
replay of the facade equals the eager result (PR #35400's eager-vs-graph
assertion; the route is allocation-free with a caller-owned ``output``); and
that the K-major FP32-state decode (``gated_delta_rule_decode``) matches a
torch reference with the FP32 state at 1e-3.

Rows: Qwen3-Next TP4 decode (B=4, T=1, H=4, HV=8) and verify (B=2, T=4) from
PR #35400, and the H=8 / HV=16 / T=7 verify row from PR #40656. Which rows are
promoted is FlashInfer's manifest; a ``CakeGDNUnsupportedError`` skips the
test with the reason. Skips when FlashInfer lacks the modules or the GPU is
outside sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_linear_gdn as cake_gdn
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.attention.cake_linear import (
    cake_gdn_decode,
    cake_gdn_decode_pretranspose,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = cake_gdn.HEAD_DIM
SCALE = HEAD_DIM**-0.5


@pytest.mark.parametrize(
    "op", ["attention.gdn_decode_pretranspose", "attention.gdn_decode"]
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_linear_gdn:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_gdn.FI_MODULE_DECODE, cake_gdn.FI_JIT_MODULE
    ):
        pytest.skip("installed FlashInfer lacks the Cake GDN decode modules")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_gdn.ARCHS:
        pytest.skip(f"Cake GDN is built for sm_100a/sm_103a, device is {cc}")


def _run_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:  # CakeGDNUnsupportedError
        pytest.skip(f"FlashInfer Cake GDN has no manifest row: {error}")


def _l2norm(x32):
    return x32 / torch.sqrt(x32.square().sum(-1, keepdim=True) + 1e-6)


def _make(batch, seq_len, hq, hv, device, seed):
    torch.manual_seed(seed)
    q = torch.randn(batch, seq_len, hq, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, hv, HEAD_DIM, device=device, dtype=torch.bfloat16)
    a = (torch.randn(batch, seq_len, hv, device=device) * 0.1).bfloat16()
    b = torch.randn(batch, seq_len, hv, device=device).bfloat16()
    A_log = torch.randn(hv, device=device) * 0.1
    dt_bias = torch.randn(hv, device=device) - 2.0
    pool = (
        torch.randn(batch + 3, hv, HEAD_DIM, HEAD_DIM, device=device) * 0.1
    ).bfloat16()
    indices = torch.randperm(batch + 3, device=device)[:batch].to(torch.int32)
    return dict(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, pool=pool, idx=indices
    )


def _pretranspose_reference(t, pool_before):
    """Per-token FP32 step on the K-last pool rows; returns (out, cache, final)."""
    repeats = t["v"].shape[2] // t["q"].shape[2]
    state = pool_before.index_select(0, t["idx"].long()).float()
    outs, cache = [], []
    for tok in range(t["q"].shape[1]):
        q = _l2norm(t["q"][:, tok].float()).repeat_interleave(repeats, dim=1) * SCALE
        k = _l2norm(t["k"][:, tok].float()).repeat_interleave(repeats, dim=1)
        alpha = torch.exp(
            -t["A_log"].exp()
            * torch.nn.functional.softplus(t["a"][:, tok].float() + t["dt_bias"])
        )
        beta = torch.sigmoid(t["b"][:, tok].float())
        state = state * alpha[:, :, None, None]
        delta = (
            t["v"][:, tok].float() - torch.einsum("bhk,bhvk->bhv", k, state)
        ) * beta[:, :, None]
        state = state + delta[..., None] * k[:, :, None, :]
        outs.append(torch.einsum("bhk,bhvk->bhv", q, state))
        cache.append(state)
    return torch.stack(outs, dim=1), torch.stack(cache, dim=1), state


def test_tp4_decode_row_matches_flashinfer_reference_and_graph_replay():
    _skip_unless_supported()
    device = torch.device("cuda")
    t = _make(4, 1, 4, 8, device, seed=0)
    pool_before = t["pool"].clone()
    out = torch.empty(4, 1, 8, HEAD_DIM, device=device, dtype=torch.bfloat16)
    assert cake_gdn.supports_gdn_decode_pretranspose(
        t["q"],
        t["k"],
        t["v"],
        t["pool"],
        t["idx"],
        A_log=t["A_log"],
        a=t["a"],
        dt_bias=t["dt_bias"],
        b=t["b"],
        output=out,
    )

    def launch(pool, output):
        return cake_gdn_decode_pretranspose(
            t["q"],
            t["k"],
            t["v"],
            None,
            t["A_log"],
            t["a"],
            t["dt_bias"],
            t["b"],
            scale=SCALE,
            output=output,
            use_qk_l2norm=True,
            initial_state=pool,
            initial_state_indices=t["idx"],
            backend="cake_gdn",
        )

    got, pool_ret = _run_or_skip(launch, t["pool"], out)
    assert got is out and pool_ret is t["pool"]
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose as fi_direct

    pool_fi = pool_before.clone()
    out_fi, _ = fi_direct(
        t["q"],
        t["k"],
        t["v"],
        None,
        t["A_log"],
        t["a"],
        t["dt_bias"],
        t["b"],
        scale=SCALE,
        use_qk_l2norm=True,
        initial_state=pool_fi,
        initial_state_indices=t["idx"],
        backend="cake_gdn",
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(t["pool"], pool_fi)
    expected_out, _, expected_final = _pretranspose_reference(t, pool_before)
    torch.testing.assert_close(out.float(), expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        t["pool"].index_select(0, t["idx"].long()).float(),
        expected_final,
        atol=1e-2,
        rtol=1e-2,
    )

    # Eager vs CUDA-graph replay on identical inputs must agree bitwise.
    eager_out, eager_pool = out.clone(), t["pool"].clone()
    t["pool"].copy_(pool_before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(t["pool"], out)
    t["pool"].copy_(pool_before)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_out)
    assert torch.equal(t["pool"], eager_pool)


@pytest.mark.parametrize("batch,seq_len,hq,hv", [(2, 4, 4, 8), (3, 7, 8, 16)])
def test_verify_rows_match_flashinfer_and_reference(batch, seq_len, hq, hv):
    _skip_unless_supported()
    device = torch.device("cuda")
    t = _make(batch, seq_len, hq, hv, device, seed=seq_len)
    pool_before = t["pool"].clone()
    out = torch.empty(batch, seq_len, hv, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = torch.empty(
        batch, seq_len, hv, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    assert cake_gdn.supports_gdn_decode_pretranspose(
        t["q"],
        t["k"],
        t["v"],
        t["pool"],
        t["idx"],
        A_log=t["A_log"],
        a=t["a"],
        dt_bias=t["dt_bias"],
        b=t["b"],
        output=out,
        intermediate_states_buffer=cache,
        disable_state_update=True,
    )
    kwargs = dict(
        scale=SCALE,
        use_qk_l2norm=True,
        initial_state_indices=t["idx"],
        disable_state_update=True,
        backend="cake_gdn",
    )
    _run_or_skip(
        cake_gdn_decode_pretranspose,
        t["q"],
        t["k"],
        t["v"],
        None,
        t["A_log"],
        t["a"],
        t["dt_bias"],
        t["b"],
        output=out,
        initial_state=t["pool"],
        intermediate_states_buffer=cache,
        **kwargs,
    )
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose as fi_direct

    pool_fi, cache_fi = pool_before.clone(), torch.empty_like(cache)
    out_fi, _ = fi_direct(
        t["q"],
        t["k"],
        t["v"],
        None,
        t["A_log"],
        t["a"],
        t["dt_bias"],
        t["b"],
        initial_state=pool_fi,
        intermediate_states_buffer=cache_fi,
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(cache, cache_fi)
    assert torch.equal(t["pool"], pool_before)  # disable_state_update
    expected_out, expected_cache, _ = _pretranspose_reference(t, pool_before)
    torch.testing.assert_close(out.float(), expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cache.float(), expected_cache, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch", [1, 4])
def test_kmajor_fp32_decode_matches_flashinfer_and_reference(batch):
    _skip_unless_supported()
    device = torch.device("cuda")
    hq, hv = 16, 32
    t = _make(batch, 1, hq, hv, device, seed=10 + batch)
    state = torch.randn(batch, hv, HEAD_DIM, HEAD_DIM, device=device) * 0.1
    state_before = state.clone()
    out = torch.empty(batch, 1, hv, HEAD_DIM, device=device, dtype=torch.bfloat16)
    assert cake_gdn.supports_gdn_decode_nontranspose(
        t["q"],
        t["k"],
        t["v"],
        state,
        A_log=t["A_log"],
        a=t["a"],
        dt_bias=t["dt_bias"],
        b=t["b"],
        output=out,
    )
    _run_or_skip(
        cake_gdn_decode,
        t["q"],
        t["k"],
        t["v"],
        state,
        t["A_log"],
        t["a"],
        t["dt_bias"],
        t["b"],
        scale=SCALE,
        output=out,
        use_qk_l2norm=True,
        backend="cake_gdn",
    )
    from flashinfer.gdn_decode import gated_delta_rule_decode as fi_direct

    state_fi = state_before.clone()
    out_fi, _ = fi_direct(
        t["q"],
        t["k"],
        t["v"],
        state_fi,
        t["A_log"],
        t["a"],
        t["dt_bias"],
        t["b"],
        scale=SCALE,
        use_qk_l2norm=True,
        backend="cake_gdn",
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(state, state_fi)

    q = _l2norm(t["q"][:, 0].float()).repeat_interleave(hv // hq, dim=1) * SCALE
    k = _l2norm(t["k"][:, 0].float()).repeat_interleave(hv // hq, dim=1)
    alpha = torch.exp(
        -t["A_log"].exp()
        * torch.nn.functional.softplus(t["a"][:, 0].float() + t["dt_bias"])
    )
    beta = torch.sigmoid(t["b"][:, 0].float())
    decayed = state_before * alpha[:, :, None, None]
    delta = (t["v"][:, 0].float() - torch.einsum("bhk,bhkv->bhv", k, decayed)) * beta[
        :, :, None
    ]
    expected_state = decayed + k[..., None] * delta[:, :, None, :]
    expected_out = torch.einsum("bhk,bhkv->bhv", q, expected_state)
    torch.testing.assert_close(out[:, 0].float(), expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(state, expected_state, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
