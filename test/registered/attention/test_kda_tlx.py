"""``TlxKDAKernel`` (gfx950 TLX KDA) against ``TritonKDAKernel``.

Both receive what ``KDAAttnBackend`` hands a kernel for Kimi-K3 (12 heads of
128, split views of one packed qkv, raw gate and beta, ``lower_bound=-5``, a
shuffled fp32 slot pool), so the pool plumbing is covered with the kernels.
"""

from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.tlx.kda import decode_available, prefill_available
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not decode_available(), reason="gfx950 required")
requires_prefill = pytest.mark.skipif(
    not prefill_available(),
    reason="TLX-capable Triton with async copy required",
)

if decode_available():
    from sglang.srt.layers.attention.linear import kda_backend
    from sglang.srt.layers.attention.linear.kernels import kda_tlx, kda_triton
    from sglang.srt.layers.attention.linear.kernels.kda_tlx import TlxKDAKernel
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend

HEADS, DIM, LOWER_BOUND = 12, 128, -5.0


def _rel(reference: torch.Tensor, actual: torch.Tensor) -> float:
    delta = actual.float() - reference.float()
    return float(delta.norm() / reference.float().norm().clamp_min(1e-8))


def _split_qkv(qkv: torch.Tensor):
    q, k, v = qkv.split(HEADS * DIM, dim=-1)
    return tuple(t.unflatten(-1, (HEADS, DIM)).unsqueeze(0) for t in (q, k, v))


def _case(lengths, decode, seed=20261028):
    gen = torch.Generator(device="cuda").manual_seed(seed + sum(lengths))
    tokens, width = sum(lengths), HEADS * DIM

    def randn(*shape, dtype=torch.bfloat16, scale=1.0):
        return scale * torch.randn(shape, dtype=dtype, device="cuda", generator=gen)

    slots = len(lengths) + 3
    indices = torch.randperm(slots - 1, device="cuda", generator=gen)[: len(lengths)]
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return dict(
        qkv=randn(tokens, 3 * width),
        # forward_decode passes the flat gate; forward_extend unflattens it.
        a=randn(tokens, width) if decode else randn(1, tokens, HEADS, DIM),
        b=randn(1, tokens, HEADS),
        A_log=-4.0 + randn(HEADS, dtype=torch.float32, scale=0.1),
        dt_bias=randn(width, dtype=torch.float32, scale=0.1),
        pool=randn(slots, HEADS, DIM, DIM, dtype=torch.float32, scale=0.02),
        cache_indices=(indices + 1).to(torch.int32),
        query_start_loc=torch.nn.functional.pad(
            lens.cumsum(0, dtype=torch.int32), (1, 0)
        ),
    )


def _extend(kernel, case, pool, **kwargs):
    # chunk_kda writes its output into v, so every run gets its own qkv.
    q, k, v = _split_qkv(case["qkv"].clone())
    return kernel.extend(
        q,
        k,
        v,
        case["a"],
        case["b"],
        ssm_states=pool,
        cache_indices=case["cache_indices"],
        query_start_loc=case["query_start_loc"],
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        lower_bound=LOWER_BOUND,
        beta_is_raw=True,
        **kwargs,
    )


def _decode(kernel, case, pool):
    q, k, v = _split_qkv(case["qkv"])
    return kernel.decode(
        q,
        k,
        v,
        case["a"],
        case["b"],
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        ssm_states=pool,
        cache_indices=case["cache_indices"],
        query_start_loc=case["query_start_loc"],
        lower_bound=LOWER_BOUND,
    )


def _assert_untouched(before, after, used):
    untouched = torch.ones(before.shape[0], dtype=torch.bool, device=before.device)
    untouched[used.long()] = False
    torch.testing.assert_close(after[untouched], before[untouched], atol=0, rtol=0)


@requires_prefill
@pytest.mark.parametrize(
    "lengths", [[64], [1, 63, 64, 65, 300], [4096, 17, 1000]], ids=str
)
def test_extend_matches_triton(lengths):
    case = _case(lengths, decode=False)
    triton_pool, tlx_pool = case["pool"].clone(), case["pool"].clone()
    # The chunk_kda reference, not the gfx950 HIP prefill TritonKDAKernel tries first.
    with patch.object(kda_triton, "_is_hip", False):
        expected = _extend(kda_triton.TritonKDAKernel(), case, triton_pool)
    actual = _extend(TlxKDAKernel(), case, tlx_pool)

    used = case["cache_indices"].long()
    assert _rel(expected, actual) < 1e-2
    assert _rel(triton_pool[used], tlx_pool[used]) < 1e-2
    _assert_untouched(case["pool"], tlx_pool, case["cache_indices"])


@pytest.mark.parametrize("batch", [1, 5, 64])
def test_decode_matches_triton(batch):
    case = _case([1] * batch, decode=True)
    triton_pool, tlx_pool = case["pool"].clone(), case["pool"].clone()
    expected = _decode(kda_triton.TritonKDAKernel(), case, triton_pool)
    actual = _decode(TlxKDAKernel(), case, tlx_pool)

    used = case["cache_indices"].long()
    assert actual.shape == expected.shape
    assert _rel(expected, actual) < 5e-3
    assert _rel(triton_pool[used], tlx_pool[used]) < 5e-3
    _assert_untouched(case["pool"], tlx_pool, case["cache_indices"])


@requires_prefill
@pytest.mark.parametrize(
    "kwargs, uses_tlx",
    [
        ({}, True),
        ({"is_spec_decode": True}, False),
        ({"return_intermediate_states": True}, False),
        ({"chunk_indices": torch.zeros(1, 2, dtype=torch.int32)}, False),
    ],
    ids=["covered", "spec-decode", "intermediate-states", "prefill-graph"],
)
def test_extend_routes_uncovered_calls_to_triton(kwargs, uses_tlx):
    case = _case([100, 30], decode=False)
    with (
        patch(
            "sglang.kernels.ops.attention.tlx.kda.gfx950_prefill.kda_paged_prefill",
            return_value=(None, None),
        ) as tlx,
        patch.object(kda_triton.TritonKDAKernel, "extend", return_value=None) as base,
    ):
        _extend(TlxKDAKernel(), case, case["pool"].clone(), **kwargs)
    assert tlx.called == uses_tlx
    assert base.called != uses_tlx


@requires_prefill
def test_dispatcher_selects_tlx_for_decode_and_prefill():
    tlx = LinearAttnKernelBackend.TLX
    dispatcher = kda_backend.KDAKernelDispatcher(
        decode_backend=tlx,
        prefill_backend=tlx,
        verify_backend=LinearAttnKernelBackend.TRITON,
    )
    assert isinstance(dispatcher.decode_kernel, TlxKDAKernel)
    assert isinstance(dispatcher.extend_kernel, TlxKDAKernel)
    assert type(dispatcher.verify_kernel) is kda_triton.TritonKDAKernel
    assert not dispatcher.supports_packed_decode


def test_dispatcher_falls_back_to_triton_when_tlx_is_unavailable():
    tlx = LinearAttnKernelBackend.TLX
    with (
        patch(
            "sglang.kernels.ops.attention.tlx.kda.decode_available", return_value=False
        ),
        patch(
            "sglang.kernels.ops.attention.tlx.kda.prefill_available", return_value=False
        ),
    ):
        dispatcher = kda_backend.KDAKernelDispatcher(
            decode_backend=tlx,
            prefill_backend=tlx,
            verify_backend=LinearAttnKernelBackend.TRITON,
        )
    assert dispatcher.decode_kernel is dispatcher.triton_kernel
    assert dispatcher.extend_kernel is dispatcher.triton_kernel


UNSUPPORTED_POOLS = {
    "bf16-state": lambda pool: pool.bfloat16(),
    # Same [slots, H, 128, 128] shape, but K-major within each slot.
    "transposed-state": lambda pool: pool.transpose(-1, -2),
}


@requires_prefill
@pytest.mark.parametrize("make_pool", UNSUPPORTED_POOLS.values(), ids=UNSUPPORTED_POOLS)
def test_unsupported_extend_warns_once_and_falls_back_to_triton(make_pool):
    case = _case([100, 30], decode=False)
    kernel, pool = TlxKDAKernel(), make_pool(case["pool"])
    with (
        patch(
            "sglang.kernels.ops.attention.tlx.kda.gfx950_prefill.kda_paged_prefill"
        ) as tlx,
        patch.object(kda_triton.TritonKDAKernel, "extend", return_value=None) as base,
        patch.object(kda_tlx.logger, "warning") as warning,
    ):
        for _ in range(2):
            _extend(kernel, case, pool)
    assert not tlx.called
    assert base.call_count == 2
    assert warning.call_count == 1


@pytest.mark.parametrize("make_pool", UNSUPPORTED_POOLS.values(), ids=UNSUPPORTED_POOLS)
def test_unsupported_decode_warns_once_and_falls_back_to_triton(make_pool):
    case = _case([1] * 4, decode=True)
    kernel, pool = TlxKDAKernel(), make_pool(case["pool"])
    with (
        patch(
            "sglang.kernels.ops.attention.tlx.kda.gfx950_decode.kda_recurrent_decode"
        ) as tlx,
        patch.object(kda_triton.TritonKDAKernel, "decode", return_value=None) as base,
        patch.object(kda_tlx.logger, "warning") as warning,
    ):
        for _ in range(2):
            _decode(kernel, case, pool)
    assert not tlx.called
    assert base.call_count == 2
    assert warning.call_count == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
