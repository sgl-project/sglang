"""Regression tests for degenerate Inkling gate renormalization."""

import unittest

import torch

from sglang.kernels.ops.moe.inkling_gate_topk_renorm import (
    inkling_gate_topk_renorm,
    inkling_gate_topk_renorm_v2,
)
from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import (
    sigmoid_gate_topk_renorm,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_ROUTE_SCALE = 1.5


def _make_inputs():
    # The production gate GEMM uses a padded [T, 264] allocation and returns a
    # [:, :258] view. Keep that layout so the v2 JIT vector-load path is tested.
    logits_storage = torch.empty((2, 264), dtype=torch.float32, device="cuda")
    logits = logits_storage[:, :258]

    # Row 0 reproduces the production failure: expf(100) overflows, so all six
    # selected routed sigmoids and both shared sigmoids become exactly zero.
    logits[0].fill_(-100.0)

    # Row 1 is a healthy control with an unambiguous routed top-6.
    logits[1].fill_(-4.0)
    logits[1, :6] = torch.tensor(
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32, device="cuda"
    )
    logits[1, 256:] = torch.tensor([0.5, -0.5], dtype=torch.float32, device="cuda")

    # Unique values avoid backend-dependent top-k tie ordering on row 0.
    bias = torch.arange(256, dtype=torch.float32, device="cuda") * 1e-3
    global_scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    return logits, bias, global_scale


def _reference(logits, bias, global_scale):
    routed_logits = logits[:, :256]
    indices = torch.topk(torch.sigmoid(routed_logits) + bias, 6, dim=-1).indices
    active_logits = torch.cat(
        [routed_logits.gather(1, indices), logits[:, 256:]], dim=-1
    )
    probs = torch.sigmoid(active_logits)
    weights = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    weights *= _ROUTE_SCALE * global_scale
    return weights[:, :6], indices.to(torch.int32), weights[:, 6:]


def _unpack(packed):
    indices = packed >> 16
    weights = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
    return weights, indices


def _run_backend(backend, logits, bias, global_scale, packed):
    if backend == "jit_v1":
        result = inkling_gate_topk_renorm(
            logits, bias, global_scale, _ROUTE_SCALE, return_packed=packed
        )
        if packed:
            packed_output, shared_weights = result
            return None, None, shared_weights, packed_output
        routed_weights, shared_weights, indices = result
        return routed_weights, indices.to(torch.int32), shared_weights, None

    if backend == "jit_v2":
        return inkling_gate_topk_renorm_v2(
            logits, bias, global_scale, _ROUTE_SCALE, return_packed=packed
        )

    assert backend == "triton"
    with envs.SGLANG_OPT_USE_GATE_TOPK_JIT.override(False):
        return sigmoid_gate_topk_renorm(
            logits,
            6,
            2,
            _ROUTE_SCALE,
            global_scale,
            bias,
            return_packed_topk=packed,
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TestInklingGateTopkRenorm(CustomTestCase):
    @torch.inference_mode()
    def test_degenerate_row_is_finite(self):
        logits, bias, global_scale = _make_inputs()
        ref_routed, ref_indices, ref_shared = _reference(logits, bias, global_scale)

        for backend in ("jit_v1", "jit_v2", "triton"):
            for packed in (False, True):
                with self.subTest(backend=backend, packed=packed):
                    routed, indices, shared, packed_output = _run_backend(
                        backend, logits, bias, global_scale, packed
                    )
                    if packed:
                        routed, indices = _unpack(packed_output)

                    self.assertTrue(torch.isfinite(routed).all().item())
                    self.assertTrue(torch.isfinite(shared).all().item())
                    self.assertTrue(torch.equal(indices, ref_indices))

                    expected_routed = (
                        ref_routed.bfloat16().float() if packed else ref_routed
                    )
                    torch.testing.assert_close(
                        routed, expected_routed, rtol=2e-3, atol=2e-4
                    )
                    torch.testing.assert_close(shared, ref_shared, rtol=2e-3, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
