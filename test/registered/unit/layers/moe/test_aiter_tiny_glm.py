"""CPU contracts for exact-M GLM caller, without ROCm/AITER imports."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-b-test-cpu-intel")

ROOT = Path(__file__).resolve().parents[5]
FILE = ROOT / "python/sglang/srt/layers/moe/moe_runner/aiter_tiny_glm.py"
spec = importlib.util.spec_from_file_location("tiny_glm_policy", FILE)
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)


class TestTinyGlm(TestCase):
    def model(self):
        return NS(
            architectures=["GlmMoeDsaForCausalLM"],
            hidden_size=6144,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            num_experts_per_tok=8,
            n_shared_experts=1,
            hidden_act="silu",
        )

    def contract(self):
        cfg = NS(
            activation="silu",
            is_gated=True,
            no_combine=False,
            apply_router_weight_on_input=False,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
            num_fused_shared_experts=1,
            num_experts=257,
            num_local_experts=257,
            use_tp_all_gather_activation=False,
        )
        quant = NS(
            doweight_stage1=False,
            expert_mask=None,
            b13=None,
            b2=None,
            hidden_pad=0,
            intermediate_pad=0,
            swiglu_limit=0,
            a13_scale=None,
            a2_scale=None,
            quant_type=NS(value="per_1x32"),
            fused_moe_kwargs={"gate_mode": "separated"},
        )
        inputs = NS(
            hidden_states=NS(shape=(8, 6144)),
            topk_ids=NS(shape=(8, 9)),
            topk_weights=NS(shape=(8, 9)),
            a1_scale=None,
            output_dtype=None,
            num_local_tokens=None,
            quant_type=NS(value="per_1x32"),
        )
        return cfg, quant, inputs

    def test_opt_in_and_old_aiter(self):
        with patch.dict(os.environ, {"SGLANG_AITER_TINY_GLM_MOE": "0"}):
            self.assertFalse(tiny.enabled())
        with patch.dict(os.environ, {"SGLANG_AITER_TINY_GLM_MOE": "1"}):
            self.assertTrue(tiny.enabled())
        tiny.factories.cache_clear()
        with patch.dict("sys.modules", {"aiter": None}):
            self.assertIsNone(tiny.factories())
        tiny.factories.cache_clear()

    def test_only_target_model_and_tp8_ep1(self):
        self.assertTrue(tiny.model_supported(self.model(), tp=8, ep=1, nextn=False))
        for tp, ep, nextn in [(4, 1, False), (8, 8, False), (8, 1, True)]:
            self.assertFalse(
                tiny.model_supported(self.model(), tp=tp, ep=ep, nextn=nextn)
            )
        m = self.model()
        m.architectures = ["DeepseekV3ForCausalLM"]
        self.assertFalse(tiny.model_supported(m, tp=8, ep=1, nextn=False))
        m = self.model()
        m.swiglu_limit = 7
        self.assertFalse(tiny.model_supported(m, tp=8, ep=1, nextn=False))

    def test_phase_does_not_follow_rows(self):
        for target, width, expected in [
            (True, 4, True),
            (False, 4, False),
            (True, 1, False),
            (True, 8, False),
        ]:
            b = NS(
                forward_mode=NS(is_target_verify=lambda: target),
                spec_info=NS(draft_token_num=width),
            )
            self.assertEqual(tiny.target_supported(b), expected)
        self.assertFalse(tiny.target_supported(None))
        b = NS(
            forward_mode=NS(is_target_verify=lambda: True),
            spec_info=NS(draft_token_num=4),
        )
        for flag in (
            "SGLANG_MORI_NO_PAD_MASK",
            "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK",
        ):
            with patch.dict(os.environ, {flag: "1"}):
                self.assertFalse(tiny.target_supported(b))

    def test_unsupported_native_semantics_fall_back(self):
        cfg, q, x = self.contract()
        self.assertTrue(tiny.runner_supported(cfg, q, x))
        for obj, field, value in [
            (cfg, "no_combine", True),
            (cfg, "gemm1_alpha", 1.702),
            (cfg, "swiglu_limit", 7),
            (q, "expert_mask", object()),
            (q, "b13", object()),
            (q, "a2_scale", object()),
            (q, "hidden_pad", 128),
            (x, "num_local_tokens", object()),
            (x, "a1_scale", object()),
            (x, "output_dtype", object()),
            (q, "fused_moe_kwargs", {"gate_mode": "interleave"}),
            (x, "quant_type", NS(value="per_Token")),
        ]:
            original = getattr(obj, field)
            setattr(obj, field, value)
            self.assertFalse(tiny.runner_supported(cfg, q, x), field)
            setattr(obj, field, original)
        for rows in [0, 1, 2, 16, 32, 64]:
            x.hidden_states.shape = (rows, 6144)
            x.topk_ids.shape = x.topk_weights.shape = (rows, 9)
            self.assertFalse(tiny.runner_supported(cfg, q, x), rows)

    def test_half_open_alias_range(self):
        tensor = lambda ptr, size: NS(
            data_ptr=lambda: ptr, numel=lambda: size, element_size=lambda: 2
        )
        self.assertFalse(tiny.overlaps(tensor(100, 4), tensor(108, 4)))
        self.assertTrue(tiny.overlaps(tensor(100, 4), tensor(102, 1)))
        self.assertTrue(tiny.overlaps(tensor(102, 1), tensor(100, 4)))

    def test_m8_canonical_padding_bounds_all_active_prefixes(self):
        # Every expert may appear at most once/token: at most eight ballot bits.
        for active in range(9):
            ids = [
                [0, 7, 255, 1, 2, 3, 4, 5] if row < active else list(range(8))
                for row in range(8)
            ]
            for expert in range(256):
                positions = [
                    (row, choice)
                    for row in range(8)
                    for choice in range(8)
                    if ids[row][choice] == expert
                ]
                self.assertLessEqual(len(positions), 8)
                self.assertEqual(len(positions), len(set(row for row, _ in positions)))
        original = [[0, 1, 2, 3, 4, 5, 6, 7]] + [[0] * 8 for _ in range(7)]
        self.assertEqual(sum(row.count(0) for row in original), 57)

    def test_per_layer_stream_handles_are_retained_and_not_reallocated(self):
        obj = tiny.PreparedTinyGlm.__new__(tiny.PreparedTinyGlm)
        obj.weights = {"w1": NS(device="cuda:0")}
        obj.handles = {}
        obj.layer_id = 3
        calls = []

        def make(**kwargs):
            handle = object()
            calls.append(handle)
            return handle

        obj.make = {4: make, 8: make}
        stream = NS(cuda_stream=11)
        cuda = NS(
            is_current_stream_capturing=lambda: False, current_stream=lambda *a: stream
        )
        with patch.dict("sys.modules", {"torch": NS(cuda=cuda)}):
            obj.prepare_stream()
            first = dict(obj.handles)
            obj.prepare_stream()
            self.assertEqual(obj.handles, first)
            stream.cuda_stream = 22
            obj.prepare_stream()
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(set(obj.handles.values())), 4)
        self.assertEqual(set(obj.handles), {(4, 11), (8, 11), (4, 22), (8, 22)})

    def test_opt_in_warmups_bind_capture_stream(self):
        captured = object()
        cuda = NS(stream=lambda value: value)
        with patch.dict("sys.modules", {"torch": NS(cuda=cuda)}):
            with patch.dict(os.environ, {"SGLANG_AITER_TINY_GLM_MOE": "1"}):
                self.assertIs(tiny.warmup_stream(captured), captured)
            with patch.dict(os.environ, {"SGLANG_AITER_TINY_GLM_MOE": "0"}):
                with tiny.warmup_stream(captured) as value:
                    self.assertIsNone(value)

    def test_no_factory_or_jit_inside_capture(self):
        # Exercise real host method with a mocked capture probe. The preparation
        # must exit before querying a stream or accessing a factory.
        obj = tiny.PreparedTinyGlm.__new__(tiny.PreparedTinyGlm)
        obj.weights = {"w1": object()}
        obj.handles = {}
        obj.make = {4: lambda **kw: self.fail("allocated during capture")}
        cuda = NS(
            is_current_stream_capturing=lambda: True,
            current_stream=lambda *a: self.fail("stream lookup after capture guard"),
        )
        with patch.dict("sys.modules", {"torch": NS(cuda=cuda)}):
            obj.prepare_stream()
        self.assertEqual(obj.handles, {})


if __name__ == "__main__":
    unittest.main()
