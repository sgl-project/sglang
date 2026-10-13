"""CPU tests: the two model specs (DeepSeek-V4.1 FFN-only, MiniMax-M3 skeleton) through the same runtime, and the
runtime itself free of model constants."""

import ast
import re
import types
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.mono import registry, routing
from sglang.srt.layers.mono.spec import LayerMode, MonoContext
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DSV41 = types.SimpleNamespace(
    model_type="deepseek_v41",
    hidden_size=5120,
    n_routed_experts=384,
    num_experts_per_tok=6,
    moe_intermediate_size=2304,
    n_shared_experts=1,
    hc_mult=4,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    hc_pre_from_prev_sublayer=True,
    num_hidden_layers=40,
)
M3 = types.SimpleNamespace(
    hidden_size=6144,
    num_hidden_layers=60,
    num_local_experts=128,
    num_experts_per_tok=4,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    scoring_func="sigmoid",
)
CFG = types.SimpleNamespace(
    tp_size=4, speculative_algorithm="DSPARK", enforce_shared_experts_fusion=False
)


def ctx():
    return MonoContext(
        server_args=None,
        hf_config=None,
        tp_size=1,
        tp_rank=0,
        cpu_group=None,
        device=torch.device("cpu"),
    )


def batch(mode, rows):
    fm = types.SimpleNamespace(
        is_decode=lambda: mode == "decode", is_target_verify=lambda: mode == "verify"
    )
    return types.SimpleNamespace(
        forward_mode=fm, input_ids=torch.zeros(rows, dtype=torch.long), batch_size=1
    )


class TestSpecs(CustomTestCase):
    def setUp(self):
        self.dsv41 = registry.load_spec("DeepseekV41ForCausalLM")
        self.m3 = registry.load_spec("MiniMaxM3SparseForCausalLM")

    def test_registry(self):
        self.assertIs(registry.load_spec("DeepseekV4ForCausalLM"), self.dsv41)
        self.assertIs(
            registry.load_spec("MiniMaxM3SparseForConditionalGeneration"), self.m3
        )
        self.assertIsNone(registry.load_spec("LlamaForCausalLM"))

    def test_each_spec_accepts_its_model_only(self):
        import os

        os.environ.setdefault("SGLANG_USE_AITER", "1")
        self.assertEqual(self.dsv41.refuse_config(CFG, DSV41), [])
        self.assertTrue(self.dsv41.refuse_config(CFG, M3))
        self.assertEqual(self.m3.refuse_config(CFG, M3), [])
        self.assertTrue(self.m3.refuse_config(CFG, DSV41))

    def test_different_contracts_same_runtime(self):
        self.assertEqual(set(self.dsv41.layers(DSV41).values()), {LayerMode.FFN})
        self.assertEqual(set(self.m3.layers(M3).values()), {LayerMode.WHOLE})
        self.assertEqual((max(self.dsv41.widths), max(self.m3.widths)), (48, 16))
        self.assertIsNotNone(
            routing.decide_for_batch(self.dsv41, ctx(), batch("verify", 48)).step
        )
        self.assertEqual(
            routing.decide_for_batch(self.m3, ctx(), batch("verify", 48)).reason,
            "width",
        )
        self.assertIsNotNone(
            routing.decide_for_batch(self.m3, ctx(), batch("verify", 16)).step
        )

    def test_m3_skeleton_refuses_at_load(self):
        self.assertTrue(any("not vendored" in r for r in self.m3.refuse_model(None)))


class TestWoARequant(CustomTestCase):
    def test_bf16_wo_a_round_trips_to_checkpoint_mxfp8(self):
        from sglang.srt.models.deepseek_common.amd.mono.spec import _mxfp8_blocks

        g = torch.Generator().manual_seed(0)
        n, k = 64, 128
        q = (
            (torch.randn(n, k, generator=g) * 100)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        code = torch.randint(-20, 5, (n // 32, k // 32), generator=g).float()
        scale = torch.exp2(code).repeat_interleave(32, 0).repeat_interleave(32, 1)
        w = (q.float() * scale).to(
            torch.bfloat16
        )  # what SGLang holds after its load-time dequant
        q2, s2 = _mxfp8_blocks(w)
        back = q2.float() * torch.exp2(s2.float() - 127).repeat_interleave(
            32, 0
        ).repeat_interleave(32, 1)
        self.assertTrue(torch.equal(back.to(torch.bfloat16), w))


class TestRuntimeHasNoModelConstants(CustomTestCase):
    def test_runtime_sources(self):
        root = Path(registry.__file__).parent
        pattern = re.compile(
            r"\b(5120|6144|384|deepseek|minimax|mhc|hc_mult|sinkhorn|dspark)\b", re.I
        )
        for name in (
            "spec.py",
            "guards.py",
            "routing.py",
            "layer.py",
            "failstop.py",
            "check.py",
        ):
            hits = [
                ln
                for ln in (root / name).read_text().splitlines()
                if pattern.search(ln) and not ln.lstrip().startswith("#")
            ]
            self.assertEqual(hits, [], f"{name} mentions a model: {hits[:3]}")


class TestFamiliesUseSharedCommon(CustomTestCase):
    # same name, different helper: aiter's symbol namespace; the attention timeline's stamp(c, point)
    ALLOWED = {
        ("deepseek_v41/attention/device.py", "kernel_symbol"),
        ("deepseek_v41/attention/device.py", "stamp"),
    }

    def test_no_family_redefines_a_shared_helper(self):
        gfx950 = Path(registry.__file__).parents[3] / "kernels/ops/mono/gfx950"

        def top_level(path):
            tree = ast.parse(path.read_text())
            return {
                n.name
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }

        shared = set().union(*map(top_level, (gfx950 / "common").rglob("*.py")))
        families = [d for d in gfx950.iterdir() if d.is_dir() and d.name != "common"]
        self.assertTrue(families)
        clashes = sorted(
            (str(f.relative_to(gfx950)), name)
            for d in families
            for f in d.rglob("*.py")
            for name in top_level(f) & shared
            if (str(f.relative_to(gfx950)), name) not in self.ALLOWED
        )
        self.assertEqual(clashes, [], "import these from gfx950/common instead")


if __name__ == "__main__":
    unittest.main()
