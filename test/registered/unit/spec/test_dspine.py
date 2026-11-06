# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.

import copy
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.configs.dspine import normalize_dspine_config, validate_dspine_runtime
from sglang.srt.speculative.dspine_components import (
    DSpineCodeBridge,
    resolve_dspine_head_topk,
    score_dspine_transitions,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def bridge_config(**overrides):
    cfg = dict(
        selected_prefix_terminal=False,
        selected_prefix_feature_writes=True,
        selected_prefix_feature_write_mode="independent",
        draft_block_causal=True,
        selected_prefix_feature_layers=[1, 2, 3],
        selected_prefix_code_bridge={"code_size": 5, "rank": 4, "eps": 1e-6},
        selected_prefix_last_write_refine={"kind": "exact_sender", "head_topk": 4},
        mask_token_id=12,
        target_layer_ids=[0, 4, 8],
    )
    cfg.update(overrides)
    return SimpleNamespace(
        hidden_size=8,
        num_hidden_layers=3,
        block_size=4,
        vocab_size=13,
        dflash_config=cfg,
    )


def make_bridge(dtype=torch.float32, **overrides):
    torch.manual_seed(13)
    bridge = DSpineCodeBridge(bridge_config(**overrides), params_dtype=dtype)
    state = dict(bridge.named_parameters()) | dict(bridge.named_buffers())
    for name, tensor in state.items():
        if name.endswith("version"):
            weight = torch.ones_like(tensor)
        elif name == "out.weight":
            weight = torch.zeros_like(tensor)
        else:
            weight = torch.randn_like(tensor) * 0.15
        bridge.load_weight(name, weight)
    bridge.prepare()
    return bridge


class TestDSpineConfig(unittest.TestCase):
    def test_native_config_is_not_mutated(self):
        source = vars(bridge_config()) | {"model_type": "qwen3"}
        before = copy.deepcopy(source)
        result = normalize_dspine_config(source)
        self.assertEqual(source, before)
        self.assertTrue(result["is_causal"])
        self.assertEqual(result["architectures"], ["DSpineDraftModel"])

    def test_speculators_format_preserves_geometry_and_indices(self):
        source = {
            "speculators_model_type": "dspine",
            "transformer_layer_config": {
                "model_type": "qwen3",
                "hidden_size": 8,
                "num_hidden_layers": 3,
                "vocab_size": 13,
            },
            "block_size": 4,
            "aux_hidden_state_layer_ids": [1, 5, 9],
            "mask_token_id": 12,
            "dspine_config": {"transfer_rank": 5, "message_size": 4, "top_k": 4},
        }
        normalized = normalize_dspine_config(source)
        cfg = normalized["dflash_config"]
        self.assertEqual(cfg["target_layer_ids"], [0, 4, 8])
        self.assertEqual(
            cfg["selected_prefix_code_bridge"], {"code_size": 5, "rank": 4, "eps": 1e-6}
        )
        self.assertEqual(cfg["selected_prefix_feature_layers"], [1, 2, 3])
        self.assertEqual(resolve_dspine_head_topk(SimpleNamespace(**normalized)), 4)
        self.assertNotIn("is_causal", source["transformer_layer_config"])

    def test_unsupported_export_modes_rejected(self):
        for key, value in (
            ("is_causal", False),
            ("has_own_lm_head", True),
            ("draft_vocab_size", 7),
            ("quantization_config", {"quant_method": "fp8"}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_dspine_config(vars(bridge_config()) | {key: value})

    def test_invalid_bridge_modes_rejected(self):
        for fields in (
            {"selected_prefix_terminal": True},
            {"selected_prefix_feature_layers": [1, 2]},
            {"selected_prefix_feature_next_token_rows": True},
            {"selected_prefix_feature_write_mode": "shared"},
            {"selected_prefix_code_bridge": {"code_size": 0, "rank": 4, "eps": 1e-6}},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                DSpineCodeBridge(bridge_config(**fields))

    def test_conflicting_refinement_width_rejected(self):
        config = bridge_config(dspine_top_k=8)
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_dspine_head_topk(config)

    def test_runtime_guards(self):
        valid = dict(
            device="cuda",
            tp_size=1,
            dp_size=1,
            pp_size=1,
            speculative_num_draft_tokens=4,
        )
        validate_dspine_runtime(SimpleNamespace(**valid), bridge_config())
        for key, value in (
            ("device", "npu"),
            ("tp_size", 2),
            ("dp_size", 2),
            ("pp_size", 2),
            ("speculative_num_draft_tokens", 3),
            ("enable_lora", True),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dspine_runtime(
                    SimpleNamespace(**(valid | {key: value})), bridge_config()
                )


class TestDSpineNumerics(unittest.TestCase):
    def test_bridge_against_independent_float64_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                bridge = make_bridge(dtype)
                hidden = torch.randn(8, 8, dtype=dtype)
                residual = torch.randn_like(hidden)
                ids = torch.tensor([2, 12, 12, 12, 7, 12, 12, 12])
                norm = SimpleNamespace(
                    variance_epsilon=1e-6, weight=torch.ones(8, dtype=dtype)
                )
                normed, output, raw, rms, _ = bridge(hidden, residual, ids, 2, norm)
                raw_ref = (hidden + residual).double().view(2, 4, 8)
                rms_ref = raw_ref.square().mean(-1).sqrt()
                read_input = (
                    (raw_ref / (rms_ref[..., None] + bridge.eps)).to(dtype).double()
                )
                projection = (
                    F.linear(read_input, bridge.read_weight.double()).to(dtype).double()
                )
                codes = (
                    math.sqrt(5)
                    * projection
                    / (projection.norm(dim=-1, keepdim=True) + bridge.eps)
                )
                codes[:, 0] = (
                    math.sqrt(5) * bridge.codebook[ids.view(2, 4)[:, 0]].double()
                )
                codes = codes.to(dtype).double()
                q = F.linear(codes, bridge.gate.weight.double()).to(dtype).double()
                s = (
                    F.linear(codes, bridge.predecessor_gate.weight.double())
                    .to(dtype)
                    .double()
                )
                v = F.linear(codes, bridge.prev.weight.double()).to(dtype).double()
                message = torch.zeros(2, 4, 4, dtype=torch.float64)
                message[:, 1:] = (
                    torch.sigmoid(q[:, 1:] + s[:, :-1] + bridge.gate.bias.double())
                    * v[:, :-1]
                )
                delta = (
                    F.linear(
                        message.to(dtype).double(),
                        bridge.feature_write_weights[2].double(),
                    )
                    .to(dtype)
                    .double()
                )
                out_ref = (raw_ref + rms_ref[..., None] * delta).to(dtype).double()
                norm_ref = (
                    (
                        out_ref
                        * torch.rsqrt(
                            out_ref.square().mean(-1, keepdim=True)
                            + norm.variance_epsilon
                        )
                    )
                    .to(dtype)
                    .double()
                )
                tolerance = 5e-6 if dtype == torch.float32 else 0.025
                torch.testing.assert_close(
                    output.double(), out_ref.view(8, 8), atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    normed.double(), norm_ref.view(8, 8), atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(
                    raw.double(), raw_ref.view(8, 8), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    rms.double(), rms_ref.view(8), atol=1e-6, rtol=1e-6
                )
                torch.testing.assert_close(
                    output.view(2, 4, 8)[:, 0], raw.view(2, 4, 8)[:, 0], atol=0, rtol=0
                )

    def test_bridge_never_crosses_request_boundaries(self):
        bridge = make_bridge()
        hidden = torch.randn(8, 8)
        ids = torch.tensor([2, 12, 12, 12, 7, 12, 12, 12])
        norm = SimpleNamespace(variance_epsilon=1e-6, weight=torch.ones(8))
        original = bridge(hidden, None, ids, 0, norm)[1]
        changed = hidden.clone()
        changed[3] += 100
        actual = bridge(changed, None, ids, 0, norm)[1]
        torch.testing.assert_close(actual[4:], original[4:], atol=0, rtol=0)

    def test_serial_refinement_matches_explicit_hidden_states(self):
        bridge = make_bridge()
        torch.manual_seed(17)
        candidates = torch.tensor(
            [
                [[1, 3, 5, 7], [2, 4, 6, 8], [0, 3, 9, 11]],
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            ]
        )
        raw, rms, query = (
            torch.randn(2, 3, 8),
            torch.rand(2, 3) + 0.1,
            torch.randn(2, 3, 4),
        )
        anchors, norm, lm = torch.tensor([10, 12]), torch.randn(8), torch.randn(13, 8)
        kwargs = dict(
            codebook=bridge.codebook,
            pred_gate=bridge.predecessor_gate.weight,
            prev=bridge.prev.weight,
            gate_bias=bridge.gate.bias,
            write=bridge.feature_write_weights[-1],
            norm=norm,
            lm_head_weight=lm,
            norm_eps=1e-6,
        )
        actual = score_dspine_transitions(
            candidates, raw, rms, query, anchors, **kwargs
        )
        reference = torch.empty(2, 3, 4, 4, dtype=torch.float64)
        for b in range(2):
            for step in range(3):
                for sender in range(4):
                    token = anchors[b] if step == 0 else candidates[b, step - 1, sender]
                    code = math.sqrt(5) * bridge.codebook[token].double()
                    gate = torch.sigmoid(
                        query[b, step].double()
                        + bridge.predecessor_gate.weight.double() @ code
                        + bridge.gate.bias.double()
                    )
                    message = gate * (bridge.prev.weight.double() @ code)
                    state = raw[b, step].double() + rms[b, step].double() * (
                        bridge.feature_write_weights[-1].double() @ message
                    )
                    normalized = (
                        state
                        * torch.rsqrt(state.square().mean() + 1e-6)
                        * norm.double()
                    )
                    reference[b, step, sender] = (
                        lm[candidates[b, step]].double() @ normalized
                    )
        torch.testing.assert_close(actual.double(), reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            actual[:, 0], actual[:, 0, :1].expand(-1, 4, -1), atol=0, rtol=0
        )

    def test_checkpoint_validation_and_reload(self):
        bridge = DSpineCodeBridge(bridge_config(), params_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "Missing"):
            bridge.prepare()
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            bridge.load_weight("prev.weight", torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            bridge.load_weight(
                "prev.weight", torch.full_like(bridge.prev.weight, float("nan"))
            )
        bridge = make_bridge()
        ptr = bridge.packed_weight.data_ptr()
        bridge.prepare()
        self.assertEqual(ptr, bridge.packed_weight.data_ptr())
        bridge.load_weight("terminal_disabled_version", torch.tensor(2))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            bridge.prepare()


if __name__ == "__main__":
    unittest.main()
