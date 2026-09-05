"""Checkpoint-driven MoE validation for GLM-5.3-Flash.

This test loads selected expert tensors, never the model. Set:

  GLM53_FLASH_FP8_PATH=/path/to/zai-org/GLM-5.3-Flash
  GLM53_FLASH_MXFP4_PATH=/path/to/GLM-5.3-Flash-Quark-MXFP4-AttnFP8
"""

import json
import os
import re
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from safetensors import safe_open

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.quantization.fp8_utils import dequant_mxfp4
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.test_utils import CustomTestCase


class TestGLM53FlashMoECheckpoint(CustomTestCase):
    hidden_size = 4096
    intermediate_size = 2048
    num_experts = 288
    swiglu_limit = 10.0
    target = re.compile(
        r"^model\.language_model\.layers\.(?:[3-9]|[1-3][0-9]|4[0-4])"
        r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        source = os.environ.get("GLM53_FLASH_FP8_PATH")
        quantized = os.environ.get("GLM53_FLASH_MXFP4_PATH")
        if not source or not quantized:
            raise unittest.SkipTest(
                "set GLM53_FLASH_FP8_PATH and GLM53_FLASH_MXFP4_PATH"
            )
        cls.source = Path(source)
        cls.quantized = Path(quantized)
        cls.source_map, cls.source_headers = cls._read_headers(cls.source)
        cls.quantized_map, cls.quantized_headers = cls._read_headers(cls.quantized)
        config = json.loads((cls.quantized / "config.json").read_text())
        text_config = config.get("text_config", config)
        cls.rms_eps = text_config["rms_norm_eps"]
        cls.hc_eps = text_config["hc_eps"]
        cls.sinkhorn_iters = text_config["hc_sinkhorn_iters"]
        cls.runner = AiterRunnerCore(
            SimpleNamespace(
                no_combine=False,
                activation="silu",
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
            )
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "runner"):
            del cls.runner
        torch.cuda.empty_cache()
        super().tearDownClass()

    @staticmethod
    def _read_headers(root):
        weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        headers = {}
        for filename in sorted(set(weight_map.values())):
            with (root / filename).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
            header.pop("__metadata__", None)
            headers.update(header)
        return weight_map, headers

    @staticmethod
    def _load(root, weight_map, name, device="cuda"):
        with safe_open(root / weight_map[name], framework="pt", device=device) as shard:
            return shard.get_tensor(name)

    @classmethod
    def _expert_names(cls, layer, expert):
        base = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
        return {
            projection: (
                f"{base}.{projection}.weight",
                f"{base}.{projection}.weight_scale",
            )
            for projection in ("gate_proj", "up_proj", "down_proj")
        }

    @classmethod
    def _load_expert_bank(cls, layer, experts):
        weights = {"gate_proj": [], "up_proj": [], "down_proj": []}
        scales = {"gate_proj": [], "up_proj": [], "down_proj": []}
        for expert in experts:
            for projection, (weight_name, scale_name) in cls._expert_names(
                layer, expert
            ).items():
                weights[projection].append(
                    cls._load(cls.quantized, cls.quantized_map, weight_name)
                )
                scales[projection].append(
                    cls._load(cls.quantized, cls.quantized_map, scale_name)
                )

        w13_raw = torch.cat(
            [
                torch.stack(weights["gate_proj"]),
                torch.stack(weights["up_proj"]),
            ],
            dim=1,
        )
        s13_raw = torch.cat(
            [
                torch.stack(scales["gate_proj"]),
                torch.stack(scales["up_proj"]),
            ],
            dim=1,
        )
        w2_raw = torch.stack(weights["down_proj"])
        s2_raw = torch.stack(scales["down_proj"])
        return {
            "w13_raw": w13_raw,
            "s13_raw": s13_raw,
            "w2_raw": w2_raw,
            "s2_raw": s2_raw,
            "w13": shuffle_weight(w13_raw.contiguous(), (16, 16)),
            "w2": shuffle_weight(w2_raw.contiguous(), (16, 16)),
            "s13": e8m0_shuffle(s13_raw.view(-1, s13_raw.shape[-1])).view_as(s13_raw),
            "s2": e8m0_shuffle(s2_raw.view(-1, s2_raw.shape[-1])).view_as(s2_raw),
        }

    @classmethod
    def _load_fp8_expert_bank(cls, layer, expert, shared=False):
        if shared:
            base = f"model.language_model.layers.{layer}.mlp.shared_experts"
        else:
            base = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
        tensors = {}
        scales = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors[projection] = cls._load(
                cls.source,
                cls.source_map,
                f"{base}.{projection}.weight",
            )
            scales[projection] = cls._load(
                cls.source,
                cls.source_map,
                f"{base}.{projection}.weight_scale_inv",
            )
        w13 = torch.cat([tensors["gate_proj"], tensors["up_proj"]], dim=0).unsqueeze(0)
        s13 = torch.cat([scales["gate_proj"], scales["up_proj"]], dim=0).unsqueeze(0)
        return {
            "w13": shuffle_weight(w13.contiguous(), (16, 16)),
            "w2": shuffle_weight(
                tensors["down_proj"].unsqueeze(0).contiguous(), (16, 16)
            ),
            "s13": s13,
            "s2": scales["down_proj"].unsqueeze(0),
            "gate": tensors["gate_proj"].float()
            * scales["gate_proj"]
            .repeat_interleave(128, dim=0)
            .repeat_interleave(128, dim=1)
            .float(),
            "up": tensors["up_proj"].float()
            * scales["up_proj"]
            .repeat_interleave(128, dim=0)
            .repeat_interleave(128, dim=1)
            .float(),
            "down": tensors["down_proj"].float()
            * scales["down_proj"]
            .repeat_interleave(128, dim=0)
            .repeat_interleave(128, dim=1)
            .float(),
        }

    @staticmethod
    def _quant_dequant_fp8_activation(activation):
        tokens, width = activation.shape
        groups = activation.float().view(tokens, width // 128, 128)
        scale = groups.abs().amax(dim=-1).clamp(min=1e-12) / 448.0
        quantized = (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return (quantized.float() * scale.unsqueeze(-1)).reshape(tokens, width)

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
    def _torch_routed(cls, hidden, ids, weights, bank):
        w13 = cls._dequant(bank["w13_raw"], bank["s13_raw"])
        w2 = cls._dequant(bank["w2_raw"], bank["s2_raw"])
        hidden = cls._quant_dequant_activation(hidden)
        output = torch.zeros_like(hidden)
        for token in range(hidden.shape[0]):
            for route in range(ids.shape[1]):
                expert = int(ids[token, route])
                gate = F.linear(
                    hidden[token].float(),
                    w13[expert, : cls.intermediate_size].float(),
                ).clamp(max=cls.swiglu_limit)
                up = F.linear(
                    hidden[token].float(),
                    w13[expert, cls.intermediate_size :].float(),
                ).clamp(-cls.swiglu_limit, cls.swiglu_limit)
                activated = cls._quant_dequant_activation(
                    (F.silu(gate) * up).unsqueeze(0).bfloat16()
                ).squeeze(0)
                expert_output = F.linear(activated.float(), w2[expert].float())
                output[token] += (expert_output * weights[token, route]).to(
                    output.dtype
                )
        return output

    @classmethod
    def _aiter_routed(cls, hidden, ids, weights, bank):
        w13 = bank["w13"].view(torch.float4_e2m1fn_x2)
        w2 = bank["w2"].view(torch.float4_e2m1fn_x2)
        w13.is_shuffled = True
        w2.is_shuffled = True
        quant_info = AiterMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=bank["s13"],
            w2_scale=bank["s2"],
            swiglu_limit=cls.swiglu_limit,
            fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
        )
        runner_input = AiterRunnerInput(
            hidden_states=hidden,
            topk_ids=ids.to(torch.int32),
            topk_weights=weights.to(torch.float32),
            quant_type=AiterQuantType.PER_1X32,
        )
        return cls.runner.run(runner_input, quant_info, {}).hidden_states

    @staticmethod
    def _assert_numerics(actual, expected):
        actual_float = actual.float()
        expected_float = expected.float()
        cosine = F.cosine_similarity(
            actual_float.flatten().unsqueeze(0),
            expected_float.flatten().unsqueeze(0),
        ).item()
        relative_l2 = (
            torch.linalg.vector_norm(actual_float - expected_float)
            / torch.linalg.vector_norm(expected_float).clamp(min=1e-12)
        ).item()
        if cosine <= 0.98 or relative_l2 >= 0.20:
            raise AssertionError(
                f"MoE mismatch: cosine={cosine:.6f}, relative_l2={relative_l2:.6f}"
            )
        if not torch.isfinite(actual).all():
            raise AssertionError("MoE output contains non-finite values")

    def test_checkpoint_precision_boundary_and_shapes(self):
        targets = {
            name
            for name, metadata in self.quantized_headers.items()
            if self.target.fullmatch(name) and metadata["dtype"] == "U8"
        }
        self.assertEqual(len(targets), 42 * self.num_experts * 3)

        for name in targets:
            metadata = self.quantized_headers[name]
            scale = self.quantized_headers[f"{name}_scale"]
            projection = name.rsplit(".", 2)[-2]
            if projection in ("gate_proj", "up_proj"):
                self.assertEqual(metadata["shape"], [2048, 2048])
                self.assertEqual(scale["shape"], [2048, 128])
            else:
                self.assertEqual(metadata["shape"], [4096, 1024])
                self.assertEqual(scale["shape"], [4096, 64])
            self.assertEqual(scale["dtype"], "U8")

        shared_targets = [
            name
            for name, metadata in self.quantized_headers.items()
            if ".mlp.shared_experts." in name
            and not ".layers.45." in name
            and name.endswith(".weight")
            and metadata["dtype"] == "U8"
        ]
        self.assertEqual(len(shared_targets), 42 * 3)
        for name in shared_targets:
            self.assertEqual(self.quantized_headers[f"{name}_scale"]["dtype"], "U8")

        preserved_mtp = [
            name
            for name, metadata in self.quantized_headers.items()
            if ".layers.45.mlp." in name
            and (".experts." in name or ".shared_experts." in name)
            and name.endswith(".weight")
            and metadata["dtype"] == "F8_E4M3"
        ]
        self.assertEqual(len(preserved_mtp), (self.num_experts + 1) * 3)

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
        "requires one gfx950 GPU",
    )
    def test_source_block_fp8_routed_shared_and_mtp_experts(self):
        cases = (
            (3, 0, False),
            (43, 287, False),
            (3, 0, True),
            (45, 0, False),
            (45, 0, True),
        )
        for layer, expert, shared in cases:
            with self.subTest(layer=layer, expert=expert, shared=shared):
                bank = self._load_fp8_expert_bank(layer, expert, shared=shared)
                generator = torch.Generator(device="cuda")
                generator.manual_seed(layer * 1000 + expert + int(shared))
                hidden = (
                    torch.randn(
                        2,
                        self.hidden_size,
                        generator=generator,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    * 0.5
                )
                hidden_qdq = self._quant_dequant_fp8_activation(hidden)
                gate = F.linear(hidden_qdq, bank["gate"]).clamp(max=self.swiglu_limit)
                up = F.linear(hidden_qdq, bank["up"]).clamp(
                    -self.swiglu_limit, self.swiglu_limit
                )
                activated = self._quant_dequant_fp8_activation(
                    (F.silu(gate) * up).bfloat16()
                )
                expected = F.linear(activated, bank["down"]).bfloat16()

                quant_info = AiterMoeQuantInfo(
                    w13_weight=bank["w13"],
                    w2_weight=bank["w2"],
                    quant_type=AiterQuantType.PER_128X128,
                    w13_scale=bank["s13"],
                    w2_scale=bank["s2"],
                    swiglu_limit=self.swiglu_limit,
                    fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
                )
                runner_input = AiterRunnerInput(
                    hidden_states=hidden,
                    topk_ids=torch.zeros((2, 1), device="cuda", dtype=torch.int32),
                    topk_weights=torch.ones((2, 1), device="cuda", dtype=torch.float32),
                    quant_type=AiterQuantType.PER_128X128,
                )
                actual = self.runner.run(runner_input, quant_info, {}).hidden_states
                self._assert_numerics(actual, expected)

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
        "requires one gfx950 GPU",
    )
    def test_real_experts_match_dequantized_oracle(self):
        for layer in (3, 4, 11, 22, 43, 44):
            for expert in (0, 143, 287):
                with self.subTest(layer=layer, expert=expert):
                    bank = self._load_expert_bank(layer, [expert])
                    generator = torch.Generator(device="cuda")
                    generator.manual_seed(layer * 1000 + expert)
                    hidden = (
                        torch.randn(
                            4,
                            self.hidden_size,
                            generator=generator,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        * 0.5
                    )
                    ids = torch.zeros((4, 1), device="cuda", dtype=torch.int64)
                    weights = torch.ones((4, 1), device="cuda", dtype=torch.float32)
                    expected = self._torch_routed(hidden, ids, weights, bank)
                    actual = self._aiter_routed(hidden, ids, weights, bank)
                    self._assert_numerics(actual, expected)
                    del bank

    @classmethod
    def _load_shared_mxfp4(cls, layer):
        base = f"model.language_model.layers.{layer}.mlp.shared_experts"
        tensors = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight_name = f"{base}.{projection}.weight"
            scale_name = f"{base}.{projection}.weight_scale"
            weight = cls._load(cls.quantized, cls.quantized_map, weight_name)
            scale = cls._load(cls.quantized, cls.quantized_map, scale_name)
            rows, packed = weight.shape
            blocks = packed // 16
            tensors[projection] = cls._dequant(
                weight.view(1, rows, packed),
                scale.view(1, rows, blocks),
            ).squeeze(0)
        return tensors

    @classmethod
    def _shared_output(cls, hidden, shared):
        gate = F.linear(hidden.float(), shared["gate_proj"].float()).clamp(
            max=cls.swiglu_limit
        )
        up = F.linear(hidden.float(), shared["up_proj"].float()).clamp(
            -cls.swiglu_limit, cls.swiglu_limit
        )
        return F.linear(F.silu(gate) * up, shared["down_proj"].float()).bfloat16()

    @classmethod
    def _load_ffn_mhc(cls, layer):
        base = f"model.language_model.layers.{layer}"
        fn = cls._load(cls.quantized, cls.quantized_map, f"{base}.hc_ffn_fn").float()
        scale = cls._load(
            cls.quantized, cls.quantized_map, f"{base}.hc_ffn_scale"
        ).float()
        bias = cls._load(
            cls.quantized, cls.quantized_map, f"{base}.hc_ffn_base"
        ).float()
        norm = cls._load(
            cls.quantized,
            cls.quantized_map,
            f"{base}.post_attention_layernorm.weight",
        ).bfloat16()
        return fn, scale, bias, norm

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
        "requires one gfx950 GPU",
    )
    def test_real_top8_plus_shared_output_contract(self):
        layer = 11
        bank = self._load_expert_bank(layer, list(range(8)))
        hidden = (
            torch.randn(2, self.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.5
        )
        ids = torch.arange(8, device="cuda", dtype=torch.int64).repeat(2, 1)
        weights = torch.rand(2, 8, device="cuda", dtype=torch.float32)
        weights /= weights.sum(dim=-1, keepdim=True)
        routed_ref = self._torch_routed(hidden, ids, weights, bank)
        routed = self._aiter_routed(hidden, ids, weights, bank)
        self._assert_numerics(routed, routed_ref)

        shared = self._load_shared_mxfp4(layer)
        shared_output = self._shared_output(hidden, shared)
        combined = (routed + shared_output).bfloat16()
        combined_ref = (routed_ref + shared_output).bfloat16()
        self.assertEqual(combined.shape, (2, self.hidden_size))
        self.assertEqual(combined.dtype, torch.bfloat16)
        self._assert_numerics(combined, combined_ref)

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
        "requires one gfx950 GPU",
    )
    def test_real_mhc_moe_mhc_joint_flow(self):
        for layer in (3, 4, 43, 44):
            with self.subTest(layer=layer):
                fn, scale, bias, norm = self._load_ffn_mhc(layer)
                bank = self._load_expert_bank(layer, list(range(8)))
                shared = self._load_shared_mxfp4(layer)
                generator = torch.Generator(device="cuda")
                generator.manual_seed(5000 + layer)
                residual = (
                    torch.randn(
                        2,
                        4,
                        self.hidden_size,
                        generator=generator,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    * 0.1
                )
                residual_flat = residual.reshape(2, 4 * self.hidden_size)

                post_ref, comb_ref, moe_input_ref = mhc._mhc_pre_torch(
                    residual,
                    fn,
                    scale,
                    bias,
                    self.rms_eps,
                    self.hc_eps,
                    self.hc_eps,
                    2.0,
                    self.sinkhorn_iters,
                )
                moe_input_ref = (
                    moe_input_ref.float()
                    * torch.rsqrt(
                        moe_input_ref.float().square().mean(dim=-1, keepdim=True)
                        + self.rms_eps
                    )
                    * norm.float()
                ).bfloat16()

                moe_input, h_res, h_post, norm_fused = mhc.hc_pre(
                    residual_flat,
                    fn,
                    scale,
                    bias,
                    hc_mult=4,
                    rms_eps=self.rms_eps,
                    hc_eps=self.hc_eps,
                    sinkhorn_iters=self.sinkhorn_iters,
                    out_norm_weight=norm,
                    out_norm_eps=self.rms_eps,
                )
                if not norm_fused:
                    moe_input = (
                        moe_input.float()
                        * torch.rsqrt(
                            moe_input.float().square().mean(dim=-1, keepdim=True)
                            + self.rms_eps
                        )
                        * norm.float()
                    ).bfloat16()
                torch.testing.assert_close(
                    moe_input, moe_input_ref, atol=2e-2, rtol=2e-2
                )

                ids = torch.arange(8, device="cuda", dtype=torch.int64).repeat(2, 1)
                weights = torch.rand(
                    2,
                    8,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                weights /= weights.sum(dim=-1, keepdim=True)
                routed_ref = self._torch_routed(moe_input_ref, ids, weights, bank)
                routed = self._aiter_routed(moe_input, ids, weights, bank)
                shared_ref = self._shared_output(moe_input_ref, shared)
                shared_actual = self._shared_output(moe_input, shared)
                moe_output_ref = (routed_ref + shared_ref).bfloat16()
                moe_output = (routed + shared_actual).bfloat16()
                self._assert_numerics(moe_output, moe_output_ref)

                joint_ref = mhc._mhc_post_torch(
                    moe_output_ref, residual, post_ref, comb_ref
                ).reshape(2, 4 * self.hidden_size)
                joint = mhc.hc_post(
                    moe_output,
                    residual_flat,
                    h_post,
                    h_res,
                    hc_mult=4,
                )
                self.assertEqual(joint.shape, residual_flat.shape)
                self.assertEqual(joint.dtype, torch.bfloat16)
                self._assert_numerics(joint, joint_ref)

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
        "requires one gfx950 GPU",
    )
    def test_mhc_moe_mhc_zero_token_contract(self):
        fn, scale, bias, norm = self._load_ffn_mhc(3)
        bank = self._load_expert_bank(3, [0])
        residual = torch.empty(
            0, 4 * self.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        moe_input, h_res, h_post, norm_fused = mhc.hc_pre(
            residual,
            fn,
            scale,
            bias,
            hc_mult=4,
            rms_eps=self.rms_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            out_norm_weight=norm,
            out_norm_eps=self.rms_eps,
        )
        self.assertFalse(norm_fused)
        moe_output = self._aiter_routed(
            moe_input,
            torch.empty((0, 1), device="cuda", dtype=torch.int64),
            torch.empty((0, 1), device="cuda", dtype=torch.float32),
            bank,
        )
        joint = mhc.hc_post(moe_output, residual, h_post, h_res, hc_mult=4)
        self.assertEqual(joint.shape, (0, 4 * self.hidden_size))
        self.assertEqual(joint.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
