"""Cake GDN chunked prefill (non-CP, CP, prepared CP) through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backends; that the
facade output and the in-place K-last state pool are bitwise identical to
``flashinfer.gdn_prefill.chunk_gated_delta_rule(backend="cake_gdn")``; that
output, final state rows and BF16 checkpoint rows match a pure-torch gated
delta rule within BF16 tolerance (PR #35400's "checkpoint prefill tracks
indexed state" case); that the CP route (``use_cp=True``) equals the non-CP
route (PR #35552's invariant); and that the prepared CP runner follows
``prepare -> replay() -> replay() with new data``.

Cache-miss-during-capture rule: the non-CP route resolves ``cu_seqlens`` on
the host once per (ptr, version, numel) and the CP route keeps one prepared
plan keyed by layouts / addresses / stream; both raise when first resolved
during CUDA-graph capture, so every signature must be warmed eagerly first.
Skips with the reason when FlashInfer lacks the modules, the GPU is outside
sm_100a / sm_103a, or FlashInfer's manifest has no row for the shape
(``CakeGDNUnsupportedError`` / CP ``ValueError`` are FlashInfer's decision).
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_linear_gdn as cake_gdn
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.attention.cake_linear import (
    cake_gdn_chunk_gated_delta_rule,
    cake_gdn_cp_prefill_prepare,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = cake_gdn.HEAD_DIM
EVERY = 64


@pytest.mark.parametrize(
    "op", ["attention.gdn_chunk_gated_delta_rule", "attention.gdn_cp_prefill_prepare"]
)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_linear_gdn:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(
        cake_gdn.FI_MODULE_PREFILL,
        cake_gdn.FI_MODULE_CP,
        cake_gdn.FI_JIT_MODULE,
        cake_gdn.FI_JIT_MODULE_CP,
    ):
        pytest.skip("installed FlashInfer lacks the Cake GDN prefill modules")
    cc = torch.cuda.get_device_capability()
    if cc not in cake_gdn.ARCHS:
        pytest.skip(f"Cake GDN is built for sm_100a/sm_103a, device is {cc}")


def _run_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as error:  # CakeGDNUnsupportedError
        pytest.skip(f"FlashInfer Cake GDN has no manifest row: {error}")
    except ValueError as error:
        if "Cake GDN CP requires" in str(error):
            pytest.skip(f"FlashInfer Cake GDN CP declined the shape: {error}")
        raise


def _l2norm(x):
    x32 = x.float()
    return (x32 / torch.sqrt(x32.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


class _Case:
    """Qwen3-Next TP4 GVA shape (Hq = Hk = 4, Hv = 8) from PR #35400."""

    def __init__(self, seq_lens, device, seed, hq=4, hv=8, state_dtype=torch.bfloat16):
        torch.manual_seed(seed)
        self.seq_lens = seq_lens
        total = sum(seq_lens)
        self.q = _l2norm(torch.randn(total, hq, HEAD_DIM, device=device).bfloat16())
        self.k = _l2norm(torch.randn(total, hq, HEAD_DIM, device=device).bfloat16())
        self.v = (torch.randn(total, hv, HEAD_DIM, device=device) * 0.1).bfloat16()
        self.alpha = torch.rand(total, hv, device=device)
        self.beta = torch.rand(total, hv, device=device)
        offsets = [0]
        for n in seq_lens:
            offsets.append(offsets[-1] + n)
        self.offsets = offsets
        self.cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
        self.pool = (
            torch.randn(len(seq_lens) + 2, hv, HEAD_DIM, HEAD_DIM, device=device) * 0.1
        ).to(state_dtype)
        self.indices = torch.arange(len(seq_lens), device=device, dtype=torch.int32) + 1
        counts = [n // EVERY for n in seq_lens]
        starts = [0]
        for c in counts:
            starts.append(starts[-1] + c)
        self.checkpoint_cu_starts = torch.tensor(
            starts, device=device, dtype=torch.int32
        )
        self.checkpoints = torch.empty(
            (starts[-1], hv, HEAD_DIM, HEAD_DIM), device=device, dtype=state_dtype
        )
        self.output = torch.empty(
            total, hv, HEAD_DIM, device=device, dtype=torch.bfloat16
        )
        self.scale = HEAD_DIM**-0.5

    def reference(self, pool_before, every):
        """FP32 recurrence on the logical [H, K, V] state (pool rows are [H, V, K])."""
        hv = self.v.shape[1]
        repeats = hv // self.q.shape[1]
        q = self.q.float().repeat_interleave(repeats, dim=1)
        k = self.k.float().repeat_interleave(repeats, dim=1)
        v = self.v.float()
        out = torch.empty(self.output.shape, device=v.device, dtype=torch.float32)
        finals, checkpoints = [], []
        for seq, (start, end) in enumerate(zip(self.offsets, self.offsets[1:])):
            state = pool_before[int(self.indices[seq])].float().transpose(-1, -2)
            for t in range(start, end):
                state = self.alpha[t][:, None, None] * state
                old_v = torch.einsum("hk,hkv->hv", k[t], state)
                delta = self.beta[t][:, None] * (v[t] - old_v)
                state = state + k[t][:, :, None] * delta[:, None, :]
                out[t] = self.scale * torch.einsum("hk,hkv->hv", q[t], state)
                if every and (t - start + 1) % every == 0:
                    checkpoints.append(state.transpose(-1, -2))
            finals.append(state.transpose(-1, -2))
        return out, torch.stack(finals), checkpoints


def _assert_reference(case, pool_before, output, pool, every, checkpoints=None):
    expected_out, expected_final, expected_cp = case.reference(pool_before, every)
    torch.testing.assert_close(output.float(), expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        pool[case.indices.long()].float(), expected_final, atol=1e-2, rtol=1e-2
    )
    if checkpoints is not None:
        torch.testing.assert_close(
            checkpoints.float(), torch.stack(expected_cp), atol=1e-2, rtol=1e-2
        )
    assert torch.equal(pool[0], pool_before[0])
    assert torch.equal(pool[-1], pool_before[-1])


def test_noncp_checkpoint_prefill_matches_flashinfer_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _Case((64, 100), device, seed=1)
    pool_before = case.pool.clone()
    kwargs = dict(
        g=case.alpha,
        beta=case.beta,
        scale=case.scale,
        output_final_state=True,
        cu_seqlens=case.cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        state_indices=case.indices,
        checkpoint_cu_starts=case.checkpoint_cu_starts,
        checkpoint_every_n_tokens=EVERY,
        use_cp=False,
    )
    assert cake_gdn.supports_gdn_chunk_gated_delta_rule(
        case.q,
        case.k,
        case.v,
        case.alpha,
        case.beta,
        case.cu_seqlens,
        initial_state=case.pool,
        output=case.output,
        output_state=case.pool,
        state_indices=case.indices,
        state_checkpoints=case.checkpoints,
        checkpoint_cu_starts=case.checkpoint_cu_starts,
        checkpoint_every_n_tokens=EVERY,
        use_cp=False,
        output_final_state=True,
    )
    out, final = _run_or_skip(
        cake_gdn_chunk_gated_delta_rule,
        case.q,
        case.k,
        case.v,
        initial_state=case.pool,
        output=case.output,
        output_state=case.pool,
        state_checkpoints=case.checkpoints,
        **kwargs,
    )
    assert out is case.output and final is case.pool
    pool_fi, cp_fi = pool_before.clone(), torch.empty_like(case.checkpoints)
    out_fi = torch.empty_like(case.output)
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_direct

    fi_direct(
        case.q,
        case.k,
        case.v,
        initial_state=pool_fi,
        output=out_fi,
        output_state=pool_fi,
        state_checkpoints=cp_fi,
        backend="cake_gdn",
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(case.output, out_fi)
    assert torch.equal(case.pool, pool_fi)
    assert torch.equal(case.checkpoints, cp_fi)
    _assert_reference(
        case, pool_before, case.output, case.pool, EVERY, case.checkpoints
    )


def test_cp_route_matches_noncp_route_and_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _Case((65, 200), device, seed=2, state_dtype=torch.float32)
    pool_before = case.pool.clone()
    common = dict(
        g=case.alpha,
        beta=case.beta,
        scale=case.scale,
        output_final_state=True,
        cu_seqlens=case.cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        state_indices=case.indices,
    )
    out_noncp = torch.empty_like(case.output)
    pool_noncp = pool_before.clone()
    _run_or_skip(
        cake_gdn_chunk_gated_delta_rule,
        case.q,
        case.k,
        case.v,
        initial_state=pool_noncp,
        output=out_noncp,
        output_state=pool_noncp,
        use_cp=False,
        **common,
    )
    assert cake_gdn.supports_gdn_chunk_gated_delta_rule(
        case.q,
        case.k,
        case.v,
        case.alpha,
        case.beta,
        case.cu_seqlens,
        initial_state=case.pool,
        output=case.output,
        output_state=case.pool,
        state_indices=case.indices,
        use_cp=True,
        output_final_state=True,
    )
    _run_or_skip(
        cake_gdn_chunk_gated_delta_rule,
        case.q,
        case.k,
        case.v,
        initial_state=case.pool,
        output=case.output,
        output_state=case.pool,
        use_cp=True,
        **common,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        case.output.float(), out_noncp.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        case.pool[case.indices.long()],
        pool_noncp[case.indices.long()],
        atol=1e-2,
        rtol=1e-2,
    )
    _assert_reference(case, pool_before, case.output, case.pool, 0)


def test_prepared_cp_replays_with_new_data():
    _skip_unless_supported()
    device = torch.device("cuda")
    case = _Case((128, 72), device, seed=3, state_dtype=torch.float32)
    pool_before = case.pool.clone()
    prepared = _run_or_skip(
        cake_gdn_cp_prefill_prepare,
        case.q,
        case.k,
        case.v,
        case.alpha,
        case.beta,
        case.cu_seqlens,
        case.pool,
        output=case.output,
        output_state=case.pool,
        state_indices=case.indices,
        scale=case.scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        capture_graph=False,
    )
    torch.cuda.synchronize()
    # Preparation ran the composite once eagerly and restored the in-place pool.
    assert torch.equal(case.pool, pool_before)
    out, final = prepared.replay()
    torch.cuda.synchronize()
    assert out is case.output and final is case.pool
    _assert_reference(case, pool_before, case.output, case.pool, 0)
    # New activations in the same storage, pool reset: replay again.
    case.v.copy_((torch.randn_like(case.v.float()) * 0.1).bfloat16())
    case.alpha.copy_(torch.rand_like(case.alpha))
    case.pool.copy_(pool_before)
    prepared.replay()
    torch.cuda.synchronize()
    _assert_reference(case, pool_before, case.output, case.pool, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
