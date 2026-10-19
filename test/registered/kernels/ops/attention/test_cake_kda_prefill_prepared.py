"""Cake prepared KDA prefill export (BF16 / TF32) through sglang.kernels.

Prepared-runner pattern: ``prepare -> launch() -> launch() again with new
data in the same storage``, then a caller-owned CUDA-graph capture of
``launch()``. Checks that the registry resolves the explicit FlashInfer
backends; that the facade launch is bitwise identical to calling
``flashinfer.kda_prefill.prepare_{bf16,tf32}_kda_prefill`` directly; that the
output, the in-place FP32 state pool and the BF16 checkpoint rows match a
pure-torch recurrence within BF16 tolerance (both exports compute the chunked
recurrence with BF16 / TF32 tensor-core MMAs, so the FP32-typed external state
is compared at the BF16 tolerance -- see ADAPTERS_linear_mamba.md open
questions); and that a ``KDAPrefillPlanCache`` hit (same structural signature,
new tensor addresses) is served by rebinding instead of re-preparing.

Cache-miss-during-capture rule: preparation compiles and allocates, so the
first ``prepare`` for every structural signature must run eagerly; only
``launch()`` (and plan-cache hits) may be captured. Skips with the reason when
FlashInfer lacks the export or the GPU is outside sm_100a / sm_103a.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_linear_kda_prefill as cake_prefill
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.attention.cake_linear import (
    cake_kda_prefill_plan_cache,
    cake_kda_prefill_prepare_bf16,
    cake_kda_prefill_prepare_tf32,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = cake_prefill.HEAD_DIM
LOWER_BOUND = -5.0
EVERY = 64


@pytest.mark.parametrize(
    "op",
    [
        "attention.kda_prefill_prepare_bf16",
        "attention.kda_prefill_prepare_tf32",
        "attention.kda_prefill_plan_cache",
        "attention.kda_prefill_supports_fp32_checkpoints",
    ],
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith(
        "sglang.kernels.cake_kernels.attention_linear_kda_prefill:"
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_prefill.FI_MODULE,
        cake_prefill.FI_RUNTIME_MODULE,
        cake_prefill.FI_JIT_MODULE,
    ):
        pytest.skip("installed FlashInfer lacks the Cake KDA prefill export")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_prefill.ARCHS:
        pytest.skip(f"Cake KDA prefill export is built for sm_100a/sm_103a, got {cc}")


class _Case:
    def __init__(self, lengths, heads, device, seed):
        torch.manual_seed(seed)
        self.lengths = lengths
        self.heads = heads
        tokens = sum(lengths)
        shape = (1, tokens, heads, HEAD_DIM)
        self.q, self.k, self.v, self.g = (
            torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(4)
        )
        self.beta = torch.randn(shape[:-1], device=device, dtype=torch.bfloat16)
        self.A_log = torch.zeros(heads, device=device)
        self.dt_bias = torch.full((heads, HEAD_DIM), -2.0, device=device)
        self.pool = torch.randn(
            len(lengths) + 2, heads, HEAD_DIM, HEAD_DIM, device=device
        )
        self.pool.mul_(0.1)
        self.indices = torch.arange(len(lengths), device=device, dtype=torch.int32) + 1
        offsets, cp_offsets = [0], [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
            cp_offsets.append(cp_offsets[-1] + (n + EVERY - 1) // EVERY)
        self.offsets = offsets
        self.cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int64)
        self.checkpoint_cu_starts = torch.tensor(
            cp_offsets, device=device, dtype=torch.int64
        )
        self.checkpoints = torch.empty(
            (cp_offsets[-1], heads, HEAD_DIM, HEAD_DIM),
            device=device,
            dtype=torch.bfloat16,
        )
        self.out = torch.empty_like(self.q)

    def kwargs(self, pool, checkpoints, out):
        return dict(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            out=out,
            initial_state=pool,
            final_state=pool,
            lower_bound=LOWER_BOUND,
            cu_seqlens=self.cu_seqlens,
            sequence_lengths=self.lengths,
            state_indices=self.indices,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=self.checkpoint_cu_starts,
            checkpoint_every_n_tokens=EVERY,
        )

    def reference(self, pool_before):
        """FP64 recurrence; checkpoints are the states at chunk starts."""
        qn = F.normalize(self.q.float(), dim=-1).bfloat16().double()
        kn = F.normalize(self.k.float(), dim=-1).bfloat16().double()
        decay = (
            LOWER_BOUND * (self.g.double() + self.dt_bias.double()).sigmoid()
        ).exp()
        active = self.beta.float().sigmoid().double()
        expected = torch.empty_like(self.out, dtype=torch.float64)
        checkpoints, final = [], []
        for seq, (start, end) in enumerate(zip(self.offsets, self.offsets[1:])):
            state = pool_before[int(self.indices[seq])].double()
            for token in range(start, end):
                if (token - start) % EVERY == 0:
                    checkpoints.append(state.clone())
                state = state * decay[0, token, :, None, :]
                residual = (
                    self.v[0, token].double()
                    - (state * kn[0, token, :, None, :]).sum(-1)
                ) * active[0, token, :, None]
                state = state + residual[:, :, None] * kn[0, token, :, None, :]
                expected[0, token] = (state * qn[0, token, :, None, :]).sum(-1)
            final.append(state)
        expected *= HEAD_DIM**-0.5
        return expected, torch.stack(final), torch.stack(checkpoints)


def _assert_matches(case, pool_before, out, pool, checkpoints):
    expected_out, expected_final, expected_cp = case.reference(pool_before)
    torch.testing.assert_close(out.float(), expected_out.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        pool[case.indices.long()].float(), expected_final.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        checkpoints.float(), expected_cp.float(), atol=1e-2, rtol=1e-2
    )
    assert torch.equal(pool[0], pool_before[0])
    assert torch.equal(pool[-1], pool_before[-1])


@pytest.mark.parametrize("lengths", [(17, 65), (64,)])
@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("precision", ["bf16", "tf32"])
def test_prepared_launch_matches_flashinfer_and_reference(lengths, heads, precision):
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _Case(lengths, heads, device, seed=sum(lengths) + heads)
    facade = (
        cake_kda_prefill_prepare_bf16
        if precision == "bf16"
        else cake_kda_prefill_prepare_tf32
    )
    from flashinfer.kda_prefill import (
        prepare_bf16_kda_prefill,
        prepare_tf32_kda_prefill,
    )

    direct = (
        prepare_bf16_kda_prefill if precision == "bf16" else prepare_tf32_kda_prefill
    )
    assert cake_prefill.supports_kda_prepared_prefill(
        case.q,
        case.k,
        case.v,
        case.g,
        case.beta,
        A_log=case.A_log,
        dt_bias=case.dt_bias,
        out=case.out,
        initial_state=case.pool,
        final_state=case.pool,
        lower_bound=LOWER_BOUND,
        cu_seqlens=case.cu_seqlens,
        sequence_lengths=case.lengths,
        state_indices=case.indices,
        state_checkpoints=case.checkpoints,
        checkpoint_cu_starts=case.checkpoint_cu_starts,
        checkpoint_every_n_tokens=EVERY,
        compute_dtype=precision,
    )
    pool_before = case.pool.clone()
    pool_fi, cp_fi, out_fi = (
        case.pool.clone(),
        case.checkpoints.clone(),
        case.out.clone(),
    )
    call = facade(
        case.q,
        case.k,
        case.v,
        case.g,
        case.beta,
        **case.kwargs(case.pool, case.checkpoints, case.out),
    )
    call_fi = direct(
        case.q,
        case.k,
        case.v,
        case.g,
        case.beta,
        **case.kwargs(pool_fi, cp_fi, out_fi),
    )
    try:
        call.launch()
        call_fi.launch()
        torch.cuda.synchronize()
        assert torch.equal(case.out, out_fi)
        assert torch.equal(case.pool, pool_fi)
        assert torch.equal(case.checkpoints, cp_fi)
        assert torch.isfinite(case.out).all()
        _assert_matches(case, pool_before, case.out, case.pool, case.checkpoints)

        # Second launch: new activations in the same storage, pool reset.
        case.v.copy_(torch.randn_like(case.v))
        case.beta.copy_(torch.randn_like(case.beta))
        case.pool.copy_(pool_before)
        call.launch()
        torch.cuda.synchronize()
        _assert_matches(case, pool_before, case.out, case.pool, case.checkpoints)

        # Caller-owned graph capture of launch() after the eager preparation.
        case.pool.copy_(pool_before)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call.launch()
        case.pool.copy_(pool_before)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches(case, pool_before, case.out, case.pool, case.checkpoints)
    finally:
        call.close()
        call_fi.close()


def test_plan_cache_hit_rebinds_new_addresses():
    _skip_unless_supported()
    device = torch.device("cuda")
    cache = cake_kda_prefill_plan_cache(capacity=4)
    try:
        results = []
        for seed in (11, 12):
            case = _Case((33, 64), 6, device, seed=seed)
            pool_before = case.pool.clone()
            call = cake_kda_prefill_prepare_bf16(
                case.q,
                case.k,
                case.v,
                case.g,
                case.beta,
                **case.kwargs(case.pool, case.checkpoints, case.out),
                plan_cache=cache,
            )
            call.launch()
            torch.cuda.synchronize()
            _assert_matches(case, pool_before, case.out, case.pool, case.checkpoints)
            results.append(call)
        assert cache.misses == 1
        assert cache.hits == 1
        assert results[0] is results[1]  # the same prepared launch, rebound
    finally:
        cache.clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
