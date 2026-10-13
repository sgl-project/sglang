"""K3 attention overlap parity under CUDA graph capture and changed-input replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear
from sglang.srt.layers.quantization.w8a8_fp8 import (
    W8A8Fp8Config,
    W8A8Fp8LinearMethod,
)
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    KimiK3MLAAttention,
    _get_k3_dense_weight,
)
from sglang.srt.utils import is_ppu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

_H = 7168
_QKVG = 6144  # q,k,v,g slices per rank at TP8
_N_FA = 128
_N_B = 12
_BFA_W_ROWS = 144  # [f_a | b] padded to 8 rows like _merge_bfa_weights


def _make_owner(with_stream: bool):
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _randn(*shape):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(0.05)
            .to(torch.bfloat16)
        )

    qkvg_w = _randn(_QKVG, _H)

    def fused_qkvg_proj(x):
        return torch.nn.functional.linear(x, qkvg_w), None

    owner = SimpleNamespace(
        use_full_rank_gate=True,
        use_fused_input_projection=False,
        _bfa_w=_randn(_BFA_W_ROWS, _H).contiguous(),
        _bfa_scale=None,
        _bfa_f_b_w=_randn(1536, _N_FA).contiguous(),
        _bfa_fa_size=_N_FA,
        _bfa_b_size=_N_B,
        fused_qkvg_proj=fused_qkvg_proj,
        split_sizes=[3 * 1536, 1536],
        _bfa_alt_stream=torch.cuda.Stream() if with_stream else None,
        _bfa_bs_limit=128 if with_stream else 0,
    )
    return owner


def _run(owner, x):
    out = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
    return [t.clone() for t in out]


class _FP8Projection(torch.nn.Module):
    def __init__(self, in_features, out_features, generator):
        super().__init__()
        weight = torch.randn(
            out_features, in_features, device="cuda", generator=generator
        ).mul_(0.05)
        scale = weight.abs().amax(dim=1).clamp_min(1e-6).div(448)
        self.weight = torch.nn.Parameter(
            (weight / scale[:, None]).to(torch.float8_e4m3fn).t(),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(scale[:, None], requires_grad=False)
        self.quant_method = W8A8Fp8LinearMethod(W8A8Fp8Config(True))

    def forward(self, x):
        return apply_fp8_linear(x, self.weight, self.weight_scale), None


class TestKimiK3ScaledBfaWeights(CustomTestCase):
    def test_channelwise_merge_preserves_square_weight_orientation(self):
        def projection(rows, cols):
            weight = torch.arange(rows * cols).reshape(rows, cols).remainder(31)
            return SimpleNamespace(
                weight=torch.nn.Parameter(weight.to(torch.float8_e4m3fn).t(), False),
                weight_scale=torch.nn.Parameter(torch.ones(rows, 1), False),
                quant_method=W8A8Fp8LinearMethod(W8A8Fp8Config(True)),
            )

        modules = [projection(128, 128), projection(64, 128), projection(12, 128)]
        weights = [layer.weight.t().float().clone() for layer in modules]
        owner = SimpleNamespace(
            fused_qkvg_proj=modules[0], f_a_proj=modules[1], b_proj=modules[2]
        )

        self.assertTrue(KimiK3DeltaAttention._merge_scaled_bfa_weights(owner))
        offset = 0
        for layer, expected in zip(modules, weights):
            size = expected.shape[0]
            torch.testing.assert_close(
                owner._bfa_w[offset : offset + size].float(), expected
            )
            torch.testing.assert_close(layer.weight.t().float(), expected)
            self.assertEqual(layer.weight_scale.shape, (size, 1))
            offset += size
        self.assertEqual((owner._bfa_fa_size, owner._bfa_b_size), (64, 12))
        self.assertEqual(owner._bfa_w.shape, (256, 128))

    def test_fp16_input_uses_original_projections_after_fp8_merge(self):
        generator = torch.Generator().manual_seed(42)

        def projection(rows, cols):
            weight = torch.randn(rows, cols, generator=generator, dtype=torch.float16)
            return lambda x: (torch.nn.functional.linear(x, weight), None)

        owner = SimpleNamespace(
            use_full_rank_gate=True,
            _bfa_w=torch.empty(128, 8, dtype=torch.float8_e4m3fn),
            _bfa_scale=torch.ones(128, 1),
            fused_qkvg_proj=projection(4, 8),
            f_a_proj=projection(2, 8),
            b_proj=projection(1, 8),
            f_b_proj=projection(3, 2),
            split_sizes=[3, 1],
        )
        x = torch.randn(2, 8, generator=generator, dtype=torch.float16)
        qkvg = owner.fused_qkvg_proj(x)[0]
        expected = (
            qkvg[:, :3],
            owner.b_proj(x)[0],
            owner.f_b_proj(owner.f_a_proj(x)[0])[0],
            qkvg[:, 3:],
        )
        with patch("sglang.kernels.ops.gemm.kimi_k3_tiny_gemm") as tiny_gemm:
            actual = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
            tiny_gemm.assert_not_called()
        for got, ref in zip(actual, expected):
            self.assertEqual(got.dtype, torch.float16)
            self.assertTrue(torch.equal(got, ref))


class TestKimiK3BfaOverlap(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_capture_replay_matches_serial(self):
        torch.manual_seed(0)
        for T in (1, 4, 12):
            for defer_f_b in (False, True):
                with self.subTest(T=T, defer_f_b=defer_f_b):
                    x = torch.empty(T, _H, device="cuda", dtype=torch.bfloat16)
                    x.normal_(std=0.05)
                    serial_owner = _make_owner(with_stream=False)
                    owner = _make_owner(with_stream=True)
                    forward = KimiK3DeltaAttention.forward_qkvbfg_fused
                    with patch(
                        "sglang.srt.models.kimi_k3.get_is_capture_mode",
                        return_value=True,
                    ):
                        # Warm up allocations/JIT outside capture.
                        forward(owner, x, defer_f_b=defer_f_b)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = forward(owner, x, defer_f_b=defer_f_b)
                        for _ in range(3):
                            # Changed inputs expose stale reads or missing dependencies.
                            x.normal_(std=0.05)
                            serial = forward(serial_owner, x, defer_f_b=defer_f_b)
                            graph.replay()
                            torch.cuda.synchronize()
                            for got, ref, name in zip(
                                captured, serial, ("qkv", "beta", "forget_gate", "g")
                            ):
                                self.assertTrue(torch.equal(got, ref), name)

    def test_eager_stream_branch_not_taken(self):
        x = torch.randn(3, _H, device="cuda", dtype=torch.bfloat16)
        serial = _run(_make_owner(with_stream=False), x)
        overlap = _run(_make_owner(with_stream=True), x)  # capture mode False
        for got, ref in zip(overlap, serial):
            self.assertTrue(torch.equal(got, ref))

    def test_mla_gate_capture_matches_serial(self):
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        qkv_weight = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        gate_weight = torch.randn_like(qkv_weight)
        project = torch.nn.functional.linear
        owner = SimpleNamespace(
            _gate_alt_stream=torch.cuda.Stream(),
            _gate_bs_limit=128,
            g_proj=lambda value: (project(value, gate_weight), None),
        )

        def forward():
            # Both branches must wait for this in-graph input producer.
            hidden = x * 0.5
            KimiK3MLAAttention._fork_output_gate(owner, hidden)
            qkv = project(hidden, qkv_weight)
            gate = KimiK3MLAAttention._compute_output_gate(owner, hidden)
            return qkv * torch.sigmoid(gate)

        for capture_mode, breakable in ((True, False), (True, True), (False, False)):
            with (
                self.subTest(capture_mode=capture_mode, breakable=breakable),
                patch(
                    "sglang.srt.models.kimi_k3.get_is_capture_mode",
                    return_value=capture_mode,
                ),
                patch(
                    "sglang.srt.models.kimi_k3.is_in_breakable_cuda_graph",
                    return_value=breakable,
                ),
            ):
                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = forward()
                for _ in range(3):
                    x.normal_()
                    hidden = x * 0.5
                    expected = project(hidden, qkv_weight) * torch.sigmoid(
                        project(hidden, gate_weight)
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(captured, expected))

    def test_block_fp8_weight_is_dequantized_for_tiny_gemm(self):
        module = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.ones((130, 129), device="cuda", dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            weight_scale_inv=torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"),
                requires_grad=False,
            ),
            quant_method=SimpleNamespace(weight_block_size=[128, 128]),
            params_dtype=torch.bfloat16,
        )

        weight = _get_k3_dense_weight(module)

        self.assertEqual(weight.dtype, torch.bfloat16)
        torch.testing.assert_close(
            weight[[0, 0, 128, 128], [0, 128, 0, 128]].float(),
            torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"),
        )

    def test_per_tensor_fp8_weight_is_not_block_dequantized(self):
        weight = torch.nn.Parameter(
            torch.ones((2, 2), device="cuda", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module = SimpleNamespace(
            weight=weight, weight_scale=torch.ones(1, device="cuda")
        )

        self.assertEqual(_get_k3_dense_weight(module).data_ptr(), weight.data_ptr())

    def test_ppu_fused_input_projection_matches_linear_slices(self):
        def linear(rows, cols):
            weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
            return SimpleNamespace(weight=torch.nn.Parameter(weight, False))

        owner = SimpleNamespace(
            use_full_rank_gate=True,
            use_fused_input_projection=True,
            _bfa_uses_block_fp8=False,
            fused_qkvg_proj=linear(12, 8),
            f_a_proj=linear(2, 8),
            b_proj=linear(1, 8),
            f_b_proj=linear(3, 2),
            split_sizes=[9, 3],
            _bfa_w=None,
            _bfa_scale=None,
            _merge_scaled_bfa_weights=lambda: False,
        )
        KimiK3DeltaAttention._merge_bfa_weights(owner)
        x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)

        qkv, beta, forget_gate, gate = _run(owner, x)
        fused = torch.nn.functional.linear(x, owner._bfa_w)
        expected_qkv, expected_gate, f_a, expected_beta, _ = fused.split(
            [9, 3, 2, 1, 1], dim=-1
        )
        expected_forget_gate = torch.nn.functional.linear(f_a, owner.f_b_proj.weight)

        for got, ref in zip(
            (qkv, beta, forget_gate, gate),
            (expected_qkv, expected_beta, expected_forget_gate, expected_gate),
        ):
            self.assertTrue(torch.equal(got, ref))

    def test_fp8_channelwise_merge_matches_unmerged_projections(self):
        if not is_ppu():
            self.skipTest("KDA FP8 fused input projection requires PPU")

        generator = torch.Generator(device="cuda").manual_seed(42)
        qkvg = _FP8Projection(256, 128, generator)
        f_a = _FP8Projection(256, 128, generator)
        beta = _FP8Projection(256, 16, generator)
        f_b = _FP8Projection(128, 128, generator)
        owner = SimpleNamespace(
            use_full_rank_gate=True,
            use_fused_input_projection=True,
            _bfa_uses_block_fp8=False,
            fused_qkvg_proj=qkvg,
            f_a_proj=f_a,
            b_proj=beta,
            f_b_proj=f_b,
            split_sizes=[96, 32],
            _bfa_w=None,
            _bfa_scale=None,
        )
        owner._merge_scaled_bfa_weights = lambda: (
            KimiK3DeltaAttention._merge_scaled_bfa_weights(owner)
        )

        inputs = [
            torch.randn(t, 256, generator=generator, device="cuda").to(torch.bfloat16)
            for t in (8, 16, 32, 128)
        ]
        references = []
        for x in inputs:
            qkvg_out = qkvg(x)[0]
            references.append(
                (qkvg_out[:, :96], beta(x)[0], f_b(f_a(x)[0])[0], qkvg_out[:, 96:])
            )

        KimiK3DeltaAttention._merge_bfa_weights(owner)
        self.assertEqual(owner._bfa_w.shape, (384, 256))
        self.assertEqual(owner._bfa_scale.shape, (384, 1))
        self.assertEqual(qkvg.weight.data_ptr(), owner._bfa_w.data_ptr())
        for x, expected in zip(inputs, references):
            actual = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, atol=0.03, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
