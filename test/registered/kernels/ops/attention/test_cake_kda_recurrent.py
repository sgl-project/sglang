"""Cake recurrent KDA decode / prefill through sglang.kernels.

Checks three things for the Cake adapter distributed by FlashInfer: the
registry resolves the explicit FlashInfer backend; the facade result (output
and the in-place state pool) is bitwise identical to calling
``flashinfer.kda.recurrent_kda(backend="cake")`` directly; and the result
matches a pure-torch KDA recurrence within BF16 tolerance. Skips (with the
reason) when the installed FlashInfer lacks the Cake KDA modules or the GPU is
outside sm_100a / sm_103a.

Decode covers the equal-head D128 unbounded-softplus route absorbed from PRs
#34946 / #34299 (raw gate + ``A_log``/``dt_bias``, BF16 beta logits, indexed
BF16 pool). Prefill covers the frozen FlashKDA packed route (B=1,
``cu_seqlens``, indexed pool) and the BF16-checkpoint form (interval a
multiple of 32, ``ceil(n / every)`` rows per sequence; parity plus FlashInfer's
initial-state row check).
Graph capture of the Cake prefill needs an eagerly warmed
``flashinfer.RecurrentKDAPrefillWorkspace`` and a caller-owned output; the
decode route is allocation-free when ``output`` is supplied.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_linear_kda as cake_kda
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.attention.cake_linear import cake_kda_recurrent
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OP = "attention.kda_recurrent"
HEAD_DIM = 128
L2_EPS = 1e-6


def test_registry_resolves_flashinfer_backend():
    spec = select_kernel(OP, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_linear_kda:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_kda.FI_MODULE,
        cake_kda.FI_MODULE_DECODE,
        cake_kda.FI_MODULE_PREFILL,
        cake_kda.FI_JIT_MODULE_DECODE,
        cake_kda.FI_JIT_MODULE_PREFILL,
    ):
        pytest.skip("installed FlashInfer lacks the Cake KDA modules")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_kda.ARCHS:
        pytest.skip(f"Cake KDA is built for sm_100a/sm_103a, device is {cc}")


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + L2_EPS)


def _step(state, q, k, v, g_raw, beta_logit, A_log, dt_bias):
    """One KDA token on a ``[H, V, K]`` FP32 state; returns (out, state)."""
    qn = _l2norm(q) * HEAD_DIM**-0.5
    kn = _l2norm(k)
    gate = -A_log.exp()[:, None] * F.softplus(g_raw + dt_bias.view(-1, HEAD_DIM))
    state = state * gate.exp()[:, None, :]
    pred = torch.einsum("hvk,hk->hv", state, kn)
    delta = (v - pred) * torch.sigmoid(beta_logit)[:, None]
    state = state + delta[:, :, None] * kn[:, None, :]
    return torch.einsum("hvk,hk->hv", state, qn), state


def test_decode_matches_flashinfer_and_reference():
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, heads, slots = 4, 8, 7
    q, k, v, g = (
        torch.randn(batch, 1, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    )
    beta = torch.randn(batch, 1, heads, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(heads, device=device) * 0.1
    dt_bias = torch.randn(heads * HEAD_DIM, device=device) - 2.0
    pool = (torch.randn(slots, heads, HEAD_DIM, HEAD_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    indices = torch.tensor([1, 3, 0, 5], device=device, dtype=torch.int32)
    pool_fi = pool.clone()
    pool_ref = pool.clone()
    assert cake_kda.supports_kda_recurrent_decode(
        q,
        k,
        v,
        g,
        beta,
        pool,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=None,
        ssm_state_indices=indices,
        beta_is_logit=True,
    )
    kwargs = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        ssm_state_indices=indices,
        beta_is_logit=True,
    )
    out, _ = cake_kda_recurrent(q, k, v, g, beta, initial_state=pool, **kwargs)
    from flashinfer.kda import recurrent_kda as fi_direct

    out_fi, _ = fi_direct(
        q, k, v, g, beta, initial_state=pool_fi, backend="cake", **kwargs
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(pool, pool_fi)

    for row in range(batch):
        slot = int(indices[row])
        ref_out, ref_state = _step(
            pool_ref[slot].float(),
            q[row, 0].float(),
            k[row, 0].float(),
            v[row, 0].float(),
            g[row, 0].float(),
            beta[row, 0].float(),
            A_log,
            dt_bias,
        )
        torch.testing.assert_close(out[row, 0].float(), ref_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(pool[slot].float(), ref_state, atol=1e-2, rtol=1e-2)
    untouched = [s for s in range(slots) if s not in indices.tolist()]
    assert torch.equal(pool[untouched], pool_ref[untouched])


def _prefill_case(lengths, heads, device):
    tokens = sum(lengths)
    q, k, v, g = (
        torch.randn(1, tokens, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    )
    beta = torch.randn(1, tokens, heads, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(heads, device=device) * 0.1
    dt_bias = torch.randn(heads * HEAD_DIM, device=device) - 2.0
    pool = (
        torch.randn(len(lengths) + 2, heads, HEAD_DIM, HEAD_DIM, device=device) * 0.1
    ).to(torch.bfloat16)
    indices = torch.arange(len(lengths), device=device, dtype=torch.int32) + 1
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
    return q, k, v, g, beta, A_log, dt_bias, pool, indices, cu_seqlens, offsets


@pytest.mark.parametrize("lengths", [(17, 65), (64,)])
def test_prefill_matches_flashinfer_and_reference(lengths):
    _skip_unless_supported()
    torch.manual_seed(1)
    device = torch.device("cuda")
    heads = 4
    q, k, v, g, beta, A_log, dt_bias, pool, indices, cu_seqlens, offsets = (
        _prefill_case(lengths, heads, device)
    )
    pool_fi = pool.clone()
    pool_ref = pool.clone()
    kwargs = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        beta_is_logit=True,
    )
    assert cake_kda.supports_kda_recurrent_prefill(
        q,
        k,
        v,
        g,
        beta,
        pool,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=None,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )
    out, _ = cake_kda_recurrent(q, k, v, g, beta, initial_state=pool, **kwargs)
    from flashinfer.kda import recurrent_kda as fi_direct

    out_fi, _ = fi_direct(
        q, k, v, g, beta, initial_state=pool_fi, backend="cake", **kwargs
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(pool, pool_fi)

    expected = torch.empty(out.shape, device=device, dtype=torch.float32)
    for seq, (start, end) in enumerate(zip(offsets, offsets[1:])):
        slot = int(indices[seq])
        state = pool_ref[slot].float()
        for token in range(start, end):
            expected[0, token], state = _step(
                state,
                q[0, token].float(),
                k[0, token].float(),
                v[0, token].float(),
                g[0, token].float(),
                beta[0, token].float(),
                A_log,
                dt_bias,
            )
        torch.testing.assert_close(pool[slot].float(), state, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(pool[0], pool_ref[0]) and torch.equal(pool[-1], pool_ref[-1])


def test_prefill_with_bf16_checkpoints_matches_flashinfer():
    """Checkpoint rows: BF16, every 32 tokens (the Cake binding's granularity).

    Rows per sequence follow FlashInfer's own tests (``tests/kda/
    test_kda_prefill_trained_gate_distribution.py``): ``ceil(n / every)`` with
    row ``starts[i]`` holding the BF16 initial state and ``starts[i] + k`` the
    state after token ``k * every`` (``k * every < n``). Output, pool and
    checkpoints are compared bitwise against the direct FlashInfer call and the
    initial-state rows against the pool before the run; an interval of 16 is
    refused by the adapter before FlashInfer's C++ check can raise.
    """
    _skip_unless_supported()
    torch.manual_seed(2)
    device = torch.device("cuda")
    lengths, heads, every = (40, 70), 4, cake_kda.CHECKPOINT_TOKEN_GRANULARITY
    q, k, v, g, beta, A_log, dt_bias, pool, indices, cu_seqlens, _ = _prefill_case(
        lengths, heads, device
    )
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + (n + every - 1) // every)
    checkpoint_cu_starts = torch.tensor(starts, device=device, dtype=torch.int64)
    checkpoints = torch.full(
        (starts[-1], heads, HEAD_DIM, HEAD_DIM),
        float("nan"),
        device=device,
        dtype=torch.bfloat16,
    )
    checkpoints_fi = checkpoints.clone()
    pool_before = pool.clone()
    pool_fi = pool.clone()
    kwargs = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=None,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        beta_is_logit=True,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=every,
    )
    admission = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
    )
    assert cake_kda.supports_kda_recurrent_prefill(
        q, k, v, g, beta, pool, checkpoint_every_n_tokens=every, **admission
    )
    # csrc/kda/cake_kda_binding_common.cuh: "checkpoint_every_n_tokens must be
    # zero or a multiple of 32" -- the adapter must refuse 16 before the C++ check.
    assert not cake_kda.supports_kda_recurrent_prefill(
        q, k, v, g, beta, pool, checkpoint_every_n_tokens=every // 2, **admission
    )
    # With checkpointing enabled FlashInfer returns ``(output, final_state,
    # state_checkpoints)``; the facade forwards that triple unchanged.
    out, _, checkpoints_ret = cake_kda_recurrent(
        q, k, v, g, beta, initial_state=pool, state_checkpoints=checkpoints, **kwargs
    )
    from flashinfer.kda import recurrent_kda as fi_direct

    out_fi, _, checkpoints_fi_ret = fi_direct(
        q,
        k,
        v,
        g,
        beta,
        initial_state=pool_fi,
        state_checkpoints=checkpoints_fi,
        backend="cake",
        **kwargs,
    )
    torch.cuda.synchronize()
    assert checkpoints_ret is checkpoints and checkpoints_fi_ret is checkpoints_fi
    assert torch.equal(out, out_fi)
    assert torch.equal(pool, pool_fi)
    assert torch.isfinite(checkpoints).all()
    assert torch.equal(checkpoints, checkpoints_fi)
    # FlashInfer's own check: row ``starts[i]`` is the sequence's BF16 initial state.
    first_rows = checkpoints[checkpoint_cu_starts[:-1]]
    assert torch.equal(first_rows, pool_before[indices.long()])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
