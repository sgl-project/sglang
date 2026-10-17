"""Inkling gate normalization must preserve tiny sigmoid ratios, not just finiteness."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.moe.inkling_gate_topk_renorm import (
    ensure_gate_gemv_fused_scratch,
    inkling_gate_gemv_fused,
    inkling_gate_topk_renorm,
    inkling_gate_topk_renorm_v2,
)
from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import (
    sigmoid_gate_topk_renorm,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_ROUTE_SCALE = 1.5
_TOPK = 6


def _make_inputs(tokens=17, n_shared=2, row_stride=264):
    # Match the padded production GEMM view and poison unused columns. Neither
    # vector loads nor normalization may accidentally include the padding.
    storage = torch.full((tokens, row_stride), float("nan"), device="cuda")
    logits = storage[:, : 256 + n_shared]
    logits.fill_(50.0)
    cases = torch.tensor(
        [[x] * 8 for x in (-100.0, -50.0, -80.0, -1000.0, -1e8, -1e30, 0.0, 100.0)]
        + [
            [-110.0, -108.0, -106.0, -104.0, -102.0, -100.0, -98.0, -96.0],
            [-100.0, -50.0, -2.0, 0.0, 2.0, 100.0, -80.0, 1.0],
            [-100.0] * 6 + [0.0, 10.0],
            [10.0] * 6 + [-100.0, -100.0],
            [-4.0, -2.0, -0.5, 0.5, 2.0, 4.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    active = cases[torch.arange(tokens, device="cuda") % cases.shape[0]]
    logits[:, 250:256] = active[:, :_TOPK]
    logits[:, 256:] = active[:, _TOPK : _TOPK + n_shared]

    # Bias selects experts 250..255 even when their sigmoid weights are tiny
    # and the unselected logits are positive. Scores have no top-k ties.
    bias = torch.arange(256, dtype=torch.float32, device="cuda") * 2.0
    global_scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    return logits, bias, global_scale


def _reference(logits, bias, global_scale):
    routed_logits = logits[:, :256]
    indices = torch.topk(torch.sigmoid(routed_logits) + bias, _TOPK, dim=-1).indices
    active = torch.cat([routed_logits.gather(1, indices), logits[:, 256:]], dim=-1)
    # Independent, high-precision model reference. In particular, do NOT add
    # an epsilon: eight equal finite logits must yield eight equal 1/8 weights.
    weights = F.logsigmoid(active.double()).softmax(dim=-1)
    weights = (weights * (_ROUTE_SCALE * global_scale.double())).float()
    return weights[:, :_TOPK], indices.to(torch.int32), weights[:, _TOPK:]


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
            logits,
            bias,
            global_scale,
            _ROUTE_SCALE,
            return_packed=packed,
            enable_pdl=is_arch_support_pdl(),
        )

    assert backend == "triton"
    with envs.SGLANG_OPT_USE_GATE_TOPK_JIT.override(False):
        return sigmoid_gate_topk_renorm(
            logits,
            _TOPK,
            logits.shape[1] - 256,
            _ROUTE_SCALE,
            global_scale,
            bias,
            return_packed_topk=packed,
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TestInklingGateTopkRenorm(CustomTestCase):
    def _check(self, output, expected, global_scale):
        routed, indices, shared, packed = output
        if packed is not None:
            routed, indices = _unpack(packed)
        ref_routed, ref_indices, ref_shared = expected
        self.assertTrue(torch.isfinite(routed).all().item())
        self.assertTrue(torch.isfinite(shared).all().item())
        self.assertTrue(torch.equal(indices, ref_indices))
        if packed is not None:
            ref_routed = ref_routed.bfloat16().float()
        # Allow one bf16 rounding boundary in packed outputs; the unquantized
        # shared weights and both fp32 outputs retain strict tolerances.
        torch.testing.assert_close(
            routed, ref_routed, rtol=8e-3 if packed is not None else 5e-5, atol=1e-7
        )
        torch.testing.assert_close(shared, ref_shared, rtol=5e-5, atol=1e-7)
        total = routed.sum(-1) + shared.sum(-1)
        torch.testing.assert_close(
            total,
            (_ROUTE_SCALE * global_scale).expand_as(total),
            rtol=4e-3 if packed is not None else 2e-5,
            atol=1e-7,
        )

    @torch.inference_mode()
    def test_normalization_across_launch_shapes(self):
        # Exercise decode, partial CTAs, and both Triton prefill tile regimes.
        for tokens in (1, 17, 129, 769, 1025):
            logits, bias, global_scale = _make_inputs(tokens)
            expected = _reference(logits, bias, global_scale)
            for backend in ("jit_v1", "jit_v2", "triton"):
                for packed in (False, True):
                    with self.subTest(tokens=tokens, backend=backend, packed=packed):
                        self._check(
                            _run_backend(backend, logits, bias, global_scale, packed),
                            expected,
                            global_scale,
                        )

    @torch.inference_mode()
    def test_random_logits_and_bias(self):
        torch.manual_seed(0)
        for offset in (0.0, -20.0, -45.0, -80.0, -100.0, -1e4):
            storage = torch.randn((129, 264), device="cuda") + offset
            logits = storage[:, :258]
            bias = torch.randn((256,), device="cuda")
            global_scale = torch.tensor([0.75], device="cuda")
            expected = _reference(logits, bias, global_scale)
            for backend in ("jit_v1", "jit_v2", "triton"):
                for packed in (False, True):
                    with self.subTest(offset=offset, backend=backend, packed=packed):
                        self._check(
                            _run_backend(backend, logits, bias, global_scale, packed),
                            expected,
                            global_scale,
                        )

    @torch.inference_mode()
    def test_triton_active_padding_and_row_stride(self):
        # Six routed + one shared leaves an inactive eighth normalization
        # lane. It must not affect either the shift or the denominator.
        for stride in (257, 264):
            logits, bias, global_scale = _make_inputs(n_shared=1, row_stride=stride)
            expected = _reference(logits, bias, global_scale)
            for packed in (False, True):
                with self.subTest(stride=stride, packed=packed):
                    self._check(
                        _run_backend("triton", logits, bias, global_scale, packed),
                        expected,
                        global_scale,
                    )

    @torch.inference_mode()
    def test_cuda_graph_replay_with_changed_inputs(self):
        for backend in ("jit_v1", "jit_v2", "triton"):
            for packed in (False, True):
                with self.subTest(backend=backend, packed=packed):
                    logits, bias, global_scale = _make_inputs()
                    original = logits.clone()
                    for _ in range(2):
                        _run_backend(backend, logits, bias, global_scale, packed)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = _run_backend(
                            backend, logits, bias, global_scale, packed
                        )
                    for step, scale in enumerate((0.75, 0.0, 1.25)):
                        logits.copy_(original.roll(step, dims=0))
                        global_scale.fill_(scale)
                        graph.replay()
                        self._check(
                            output, _reference(logits, bias, global_scale), global_scale
                        )

    @torch.inference_mode()
    def test_fused_gemv_cuda_graph(self):
        ensure_gate_gemv_fused_scratch(torch.device("cuda"))
        for tokens in (1, 4, 17):
            target, bias, global_scale = _make_inputs(tokens)
            # One-hot inputs make the GEMV exact, including in bf16. This
            # checks the full GEMV/epilogue path, not just the v2 entry point.
            x = torch.zeros((tokens, 6144), dtype=torch.bfloat16, device="cuda")
            x[:, :tokens] = torch.eye(tokens, device="cuda")
            weight = torch.zeros((264, 6144), dtype=torch.bfloat16, device="cuda")
            weight[:258, :tokens] = target.T
            actual_logits = weight[:258, :tokens].T.float()
            for packed in (False, True):
                with self.subTest(tokens=tokens, packed=packed):

                    def launch():
                        return inkling_gate_gemv_fused(
                            x,
                            weight,
                            bias,
                            global_scale,
                            _ROUTE_SCALE,
                            return_packed=packed,
                            enable_pdl=is_arch_support_pdl(),
                        )

                    self._check(
                        launch(),
                        _reference(actual_logits, bias, global_scale),
                        global_scale,
                    )
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = launch()
                    for scale in (0.75, 0.0, 1.25):
                        global_scale.fill_(scale)
                        graph.replay()
                        self._check(
                            output,
                            _reference(actual_logits, bias, global_scale),
                            global_scale,
                        )

    @torch.inference_mode()
    def test_nonfused_normalization(self):
        from sglang.srt.models.inkling_common.moe import renorm_topk_logits_scaled

        logits, bias, global_scale = _make_inputs()
        expected = _reference(logits, bias, global_scale)
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                routed, shared = renorm_topk_logits_scaled(
                    logits.to(device),
                    expected[1].to(device),
                    2,
                    _ROUTE_SCALE,
                    global_scale.to(device),
                )
                self._check(
                    (routed.cuda(), expected[1], shared.cuda(), None),
                    expected,
                    global_scale,
                )


if __name__ == "__main__":
    unittest.main()
