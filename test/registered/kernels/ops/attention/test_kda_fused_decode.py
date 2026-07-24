"""Correctness tests for the TP8/TP16 Kimi-K3 fused KDA decode kernel."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.kda_fused_decode import kda_fused_decode
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=90,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)

_D = 128


def _make_case(heads: int, batch: int, *, strided_state: bool = False):
    torch.manual_seed(20260724 + heads * 1000 + batch)
    device = "cuda"
    seg = heads * _D
    slots = batch + 3
    dense_slot = heads * _D * _D
    slot_stride = dense_slot + 4 * 73 if strided_state else dense_slot
    state_storage = (
        torch.randn(slots * slot_stride, device=device, dtype=torch.float32) * 0.01
    )
    state = torch.as_strided(
        state_storage,
        (slots, heads, _D, _D),
        (slot_stride, _D * _D, _D, 1),
    )
    indices = torch.randperm(slots, device=device)[:batch].to(torch.int32)
    if batch > 1:
        indices[-1] = -1
    return {
        "heads": heads,
        "batch": batch,
        "seg": seg,
        "slots": slots,
        "slot_stride": slot_stride,
        "mixed_qkv": torch.randn(batch, 3 * seg, device=device, dtype=torch.bfloat16),
        "a": torch.randn(batch, seg, device=device, dtype=torch.bfloat16),
        "beta": torch.randn(batch, heads, device=device, dtype=torch.bfloat16),
        "conv": torch.randn(slots, 3, 3 * seg, device=device, dtype=torch.bfloat16),
        "weight": torch.randn(3 * seg, 4, device=device, dtype=torch.float32) * 0.01,
        "bias": torch.randn(3 * seg, device=device, dtype=torch.float32) * 0.01,
        "A_log": torch.randn(1, 1, heads, 1, device=device, dtype=torch.float32),
        "dt_bias": torch.randn(seg, device=device, dtype=torch.float32),
        "onorm_g": torch.randn(batch, seg, device=device, dtype=torch.bfloat16),
        "onorm_weight": torch.randn(_D, device=device, dtype=torch.float32),
        "state_storage": state_storage,
        "state": state,
        "indices": indices,
    }


def _clone_state(case):
    storage = case["state_storage"].clone()
    state = torch.as_strided(
        storage,
        (case["slots"], case["heads"], _D, _D),
        (case["slot_stride"], _D * _D, _D, 1),
    )
    return storage, state


def _run_fused(case, conv, state, steps):
    weight_t = case["weight"].t().contiguous()
    seg = case["seg"]
    output = None
    for _ in range(steps):
        output = kda_fused_decode(
            case["mixed_qkv"],
            case["a"],
            case["beta"],
            conv,
            weight_t[:, :seg].contiguous(),
            weight_t[:, seg : 2 * seg].contiguous(),
            weight_t[:, 2 * seg :].contiguous(),
            case["bias"],
            case["A_log"].reshape(-1),
            case["dt_bias"],
            case["onorm_g"],
            case["onorm_weight"],
            state,
            case["indices"],
            _D**-0.5,
            1e-6,
            -5.0,
        )
    return output


def _run_fallback(case, conv, state, steps):
    output = None
    for _ in range(steps):
        qkv = causal_conv1d_update(
            case["mixed_qkv"],
            conv.transpose(-1, -2),
            case["weight"],
            case["bias"],
            activation="silu",
            conv_state_indices=case["indices"],
        )
        core = TritonKDAKernel().packed_decode(
            qkv,
            case["a"],
            case["beta"],
            A_log=case["A_log"],
            dt_bias=case["dt_bias"],
            scale=_D**-0.5,
            ssm_states=state,
            cache_indices=case["indices"],
            num_v_heads=case["heads"],
            head_v_dim=_D,
            lower_bound=-5.0,
        )
        output = rms_norm_gated(
            core,
            case["onorm_g"].view(1, case["batch"], case["heads"], _D),
            case["onorm_weight"],
            None,
            activation="sigmoid",
            eps=1e-6,
        )
    return output


class TestKdaFusedDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("KDA fused decode requires an SM90+ CUDA GPU")

    def _check(self, heads: int, batch: int, *, steps=1, strided_state=False):
        case = _make_case(heads, batch, strided_state=strided_state)
        fused_conv = case["conv"].clone()
        fallback_conv = case["conv"].clone()
        fused_storage, fused_state = _clone_state(case)
        fallback_storage, fallback_state = _clone_state(case)

        fused_out = _run_fused(case, fused_conv, fused_state, steps)
        fallback_out = _run_fallback(case, fallback_conv, fallback_state, steps)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            fused_out.float(), fallback_out.float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            fused_storage, fallback_storage, rtol=2e-3, atol=2e-4
        )
        self.assertTrue(torch.equal(fused_conv, fallback_conv))

    def test_tp16_matches_fallback(self):
        for batch in (1, 8, 64):
            with self.subTest(batch=batch):
                self._check(6, batch)

    def test_tp16_two_step_envelope_strided_state(self):
        self._check(6, 8, steps=2, strided_state=True)

    def test_tp8_specialization_remains_correct(self):
        for batch in (1, 64):
            with self.subTest(batch=batch):
                self._check(12, batch)


if __name__ == "__main__":
    unittest.main()
