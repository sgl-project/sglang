"""Isolated gfx950 numerical tests for GLM-5.3-Flash Quark MoE."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.quantization.fp8_utils import dequant_mxfp4
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestGLM53FlashQuarkMoE(CustomTestCase):
    hidden_size = 4096
    intermediate_size = 2048
    num_experts = 9
    swiglu_limit = 10.0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(7)
        cls.weights = cls._make_mxfp4_bank()
        cls.weights["w13_deq"] = cls._dequant(
            cls.weights["w13_raw"], cls.weights["s13_raw"]
        )
        cls.weights["w2_deq"] = cls._dequant(
            cls.weights["w2_raw"], cls.weights["s2_raw"]
        )
        cls.runner = AiterRunnerCore(
            SimpleNamespace(
                no_combine=False,
                activation="silu",
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
            )
        )

    @classmethod
    def _make_mxfp4_bank(cls):
        gate_weights = []
        up_weights = []
        down_weights = []
        gate_scales = []
        up_scales = []
        down_scales = []
        for expert in range(cls.num_experts):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(100 + expert)
            gate = (
                torch.randn(
                    cls.intermediate_size,
                    cls.hidden_size,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                * 0.05
            )
            up = (
                torch.randn(
                    cls.intermediate_size,
                    cls.hidden_size,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                * 0.05
            )
            down = (
                torch.randn(
                    cls.hidden_size,
                    cls.intermediate_size,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                * 0.01
            )
            gate_q, gate_s = dynamic_mxfp4_quant(gate)
            up_q, up_s = dynamic_mxfp4_quant(up)
            down_q, down_s = dynamic_mxfp4_quant(down)
            gate_weights.append(gate_q)
            up_weights.append(up_q)
            down_weights.append(down_q)
            gate_scales.append(gate_s)
            up_scales.append(up_s)
            down_scales.append(down_s)

        w13 = torch.cat([torch.stack(gate_weights), torch.stack(up_weights)], dim=1)
        w2 = torch.stack(down_weights)
        s13 = torch.cat([torch.stack(gate_scales), torch.stack(up_scales)], dim=1)
        s2 = torch.stack(down_scales)
        return {
            "w13_raw": w13,
            "w2_raw": w2,
            "s13_raw": s13,
            "s2_raw": s2,
            "w13": shuffle_weight(w13.contiguous(), (16, 16)),
            "w2": shuffle_weight(w2.contiguous(), (16, 16)),
            "s13": e8m0_shuffle(s13.view(-1, s13.shape[-1])).view_as(s13),
            "s2": e8m0_shuffle(s2.view(-1, s2.shape[-1])).view_as(s2),
        }

    @staticmethod
    def _quantize_fp8_weight(weight):
        rows, width = weight.shape
        blocks = (
            weight.float().view(rows // 128, 128, width // 128, 128).permute(0, 2, 1, 3)
        )
        scale = blocks.abs().amax(dim=(2, 3)).clamp(min=1e-12) / 448.0
        quantized = (blocks / scale[:, :, None, None]).to(torch.float8_e4m3fn)
        return (
            quantized.permute(0, 2, 1, 3).reshape(rows, width),
            scale,
        )

    @staticmethod
    def _dequantize_fp8_weight(weight, scale):
        return weight.float() * scale.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1
        )

    @staticmethod
    def _quant_dequant_fp8_activation(activation):
        tokens, width = activation.shape
        groups = activation.float().view(tokens, width // 128, 128)
        scale = groups.abs().amax(dim=-1).clamp(min=1e-12) / 448.0
        quantized = (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return (quantized.float() * scale.unsqueeze(-1)).reshape(tokens, width)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "weights"):
            del cls.weights
        if hasattr(cls, "runner"):
            del cls.runner
        torch.cuda.empty_cache()
        super().tearDownClass()

    @classmethod
    def _dequant(cls, weight, scale):
        experts, rows, packed = weight.shape
        blocks = packed // 16
        return dequant_mxfp4(
            weight.view(experts, rows, blocks, 16),
            scale,
            torch.bfloat16,
        )

    @classmethod
    def _quant_dequant_activation(cls, activation):
        quantized, scale = dynamic_mxfp4_quant(activation)
        tokens, packed = quantized.shape
        blocks = packed // 16
        return dequant_mxfp4(
            quantized.view(1, tokens, blocks, 16),
            scale.view(1, tokens, blocks),
            torch.bfloat16,
        ).squeeze(0)

    @classmethod
    def _torch_oracle(cls, hidden_states, topk_ids, topk_weights):
        w13 = cls.weights["w13_deq"]
        w2 = cls.weights["w2_deq"]
        output = torch.zeros_like(hidden_states)
        hidden_qdq = cls._quant_dequant_activation(hidden_states)
        for token in range(hidden_states.shape[0]):
            for route in range(topk_ids.shape[1]):
                expert = int(topk_ids[token, route])
                gate = F.linear(
                    hidden_qdq[token].float(),
                    w13[expert, : cls.intermediate_size].float(),
                )
                up = F.linear(
                    hidden_qdq[token].float(),
                    w13[expert, cls.intermediate_size :].float(),
                )
                gate = gate.clamp(max=cls.swiglu_limit)
                up = up.clamp(min=-cls.swiglu_limit, max=cls.swiglu_limit)
                activated = F.silu(gate) * up
                activated = cls._quant_dequant_activation(
                    activated.unsqueeze(0).bfloat16()
                ).squeeze(0)
                expert_output = F.linear(activated.float(), w2[expert].float())
                output[token] += (expert_output * topk_weights[token, route]).to(
                    output.dtype
                )
        return output

    @classmethod
    def _aiter(cls, hidden_states, topk_ids, topk_weights):
        w13 = cls.weights["w13"].view(torch.float4_e2m1fn_x2)
        w2 = cls.weights["w2"].view(torch.float4_e2m1fn_x2)
        w13.is_shuffled = True
        w2.is_shuffled = True
        quant_info = AiterMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=cls.weights["s13"],
            w2_scale=cls.weights["s2"],
            swiglu_limit=cls.swiglu_limit,
            fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
        )
        runner_input = AiterRunnerInput(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights.to(torch.float32),
            quant_type=AiterQuantType.PER_1X32,
        )
        return cls.runner.run(runner_input, quant_info, {}).hidden_states

    def _assert_numerics(self, actual, expected, max_abs=None):
        self.assertTrue(torch.isfinite(actual).all())
        actual_float = actual.float()
        expected_float = expected.float()
        cosine = F.cosine_similarity(
            actual_float.flatten().unsqueeze(0),
            expected_float.flatten().unsqueeze(0),
        ).item()
        self.assertGreater(cosine, 0.98)
        relative_l2 = (
            torch.linalg.vector_norm(actual_float - expected_float)
            / torch.linalg.vector_norm(expected_float).clamp(min=1e-12)
        ).item()
        self.assertLess(relative_l2, 0.20)
        if max_abs is not None:
            self.assertLess(
                (actual_float - expected_float).abs().max().item(),
                max_abs,
            )

    def test_top1_and_top8_match_dequantized_oracle(self):
        for tokens in (1, 8, 17, 32, 64, 128):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(tokens)
            hidden = (
                torch.randn(
                    tokens,
                    self.hidden_size,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                * 0.5
            )
            # topk=9 models eight routed experts plus one fused shared slot.
            for topk in (1, 8, 9):
                with self.subTest(tokens=tokens, topk=topk):
                    ids = torch.arange(topk, device="cuda", dtype=torch.int64).repeat(
                        tokens, 1
                    )
                    weights = torch.rand(
                        tokens,
                        topk,
                        generator=generator,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    if topk > 1:
                        weights /= weights.sum(dim=-1, keepdim=True)
                    expected = self._torch_oracle(hidden, ids, weights)
                    actual = self._aiter(hidden, ids, weights)
                    repeated = self._aiter(hidden, ids, weights)
                    self._assert_numerics(actual, expected, max_abs=0.75)
                    if topk == 1:
                        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
                    else:
                        # Stage-2 combines top-k routes with atomics; reduction
                        # order may differ while remaining BF16-equivalent.
                        torch.testing.assert_close(
                            actual, repeated, atol=2e-2, rtol=1e-2
                        )

    def test_clamp_boundary(self):
        hidden = torch.full(
            (1, self.hidden_size),
            4.0,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ids = torch.tensor([[0]], device="cuda", dtype=torch.int64)
        weights = torch.ones((1, 1), device="cuda", dtype=torch.float32)
        expected = self._torch_oracle(hidden, ids, weights)
        actual = self._aiter(hidden, ids, weights)
        self._assert_numerics(actual, expected)

    def test_plain_block_fp8_matches_separated_oracle(self):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1234)
        gate = (
            torch.randn(
                self.intermediate_size,
                self.hidden_size,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.05
        )
        up = (
            torch.randn(
                self.intermediate_size,
                self.hidden_size,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.05
        )
        down = (
            torch.randn(
                self.hidden_size,
                self.intermediate_size,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        gate_q, gate_s = self._quantize_fp8_weight(gate)
        up_q, up_s = self._quantize_fp8_weight(up)
        down_q, down_s = self._quantize_fp8_weight(down)
        w13_raw = torch.cat([gate_q, up_q], dim=0).unsqueeze(0)
        w13_scale = torch.cat([gate_s, up_s], dim=0).unsqueeze(0)
        w2_raw = down_q.unsqueeze(0)
        w2_scale = down_s.unsqueeze(0)
        w13 = shuffle_weight(w13_raw.contiguous(), (16, 16))
        w2 = shuffle_weight(w2_raw.contiguous(), (16, 16))
        quant_info = AiterMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            quant_type=AiterQuantType.PER_128X128,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            swiglu_limit=self.swiglu_limit,
            fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
        )

        gate_deq = self._dequantize_fp8_weight(gate_q, gate_s)
        up_deq = self._dequantize_fp8_weight(up_q, up_s)
        down_deq = self._dequantize_fp8_weight(down_q, down_s)
        for tokens in (1, 8, 32):
            with self.subTest(tokens=tokens):
                hidden = (
                    torch.randn(
                        tokens,
                        self.hidden_size,
                        generator=generator,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    * 0.5
                )
                hidden_qdq = self._quant_dequant_fp8_activation(hidden)
                gate_out = F.linear(hidden_qdq, gate_deq).clamp(max=self.swiglu_limit)
                up_out = F.linear(hidden_qdq, up_deq).clamp(
                    -self.swiglu_limit, self.swiglu_limit
                )
                activated = self._quant_dequant_fp8_activation(
                    (F.silu(gate_out) * up_out).bfloat16()
                )
                expected = F.linear(activated, down_deq).bfloat16()

                runner_input = AiterRunnerInput(
                    hidden_states=hidden,
                    topk_ids=torch.zeros((tokens, 1), device="cuda", dtype=torch.int32),
                    topk_weights=torch.ones(
                        (tokens, 1), device="cuda", dtype=torch.float32
                    ),
                    quant_type=AiterQuantType.PER_128X128,
                )
                actual = self.runner.run(runner_input, quant_info, {}).hidden_states
                self._assert_numerics(actual, expected, max_abs=0.75)


if __name__ == "__main__":
    unittest.main()
