"""CPU tests: the MiniMax-M3 spec through the runtime's routing, the runtime free of model constants, and kernel
families reusing the shared gfx950 helpers."""

import ast
import re
import types
import unittest
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.mono import registry, routing
from sglang.srt.layers.mono.spec import LayerMode, MonoContext
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

M3 = types.SimpleNamespace(
    hidden_size=6144,
    num_hidden_layers=60,
    num_local_experts=128,
    num_experts_per_tok=4,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    scoring_func="sigmoid",
    swiglu_alpha=1.702,
)
CFG = types.SimpleNamespace(
    tp_size=4,
    speculative_algorithm="EAGLE3",
    speculative_eagle_topk=1,
    speculative_num_steps=3,
    speculative_num_draft_tokens=4,
    disable_shared_experts_fusion=False,
    page_size=128,
    kv_cache_dtype="fp8_e4m3",
    enable_hierarchical_cache=False,
    hicache_io_backend="kernel",
    hicache_mem_layout="layer_first",
)


def ctx(server_args=None):
    return MonoContext(
        server_args=server_args,
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
        self.m3 = registry.load_spec("MiniMaxM3SparseForCausalLM")

    def test_registry(self):
        self.assertIs(
            registry.load_spec("MiniMaxM3SparseForConditionalGeneration"), self.m3
        )
        self.assertIsNone(registry.load_spec("LlamaForCausalLM"))

    def test_m3_accepts_its_model_only(self):
        import os

        os.environ.setdefault("SGLANG_USE_AITER", "1")
        with envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.override(1):
            self.assertEqual(self.m3.refuse_config(CFG, M3), [])
            other = types.SimpleNamespace(**{**vars(M3), "hidden_size": 5120})
            self.assertTrue(self.m3.refuse_config(CFG, other))

    def test_m3_routes_whole_moe_layers_up_to_16_rows(self):
        dense3 = types.SimpleNamespace(**vars(M3), moe_layer_freq=[0] * 3 + [1] * 57)
        self.assertEqual(set(self.m3.layers(M3).values()), {LayerMode.WHOLE})
        self.assertEqual(sorted(self.m3.layers(dense3)), list(range(3, 60)))
        # The widest step is the LDS budget's, which scales with TP (config's
        # MAX_WIDTH): 16 at the TP4 these tests import under, 12 at TP2.
        from sglang.kernels.ops.mono.gfx950.minimax_m3.config import TP, MAX_WIDTH

        self.assertEqual(max(self.m3.widths), MAX_WIDTH)
        if TP == 4:
            self.assertEqual(max(self.m3.widths), 16)
        self.assertEqual(
            routing.decide_for_batch(self.m3, ctx(), batch("verify", 48)).reason,
            "width",
        )
        m3_args = types.SimpleNamespace(
            speculative_algorithm="EAGLE3", speculative_num_draft_tokens=16
        )
        self.assertIsNotNone(
            routing.decide_for_batch(self.m3, ctx(m3_args), batch("verify", 16)).step
        )

    def test_m3_refuses_what_its_kernels_do_not_run(self):
        import os

        os.environ.setdefault("SGLANG_USE_AITER", "1")
        with envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.override(2):
            self.assertTrue(
                any("TOPK_FREQ" in r for r in self.m3.refuse_config(CFG, M3))
            )
        with envs.SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ.override(1):
            for field, value in (
                ("speculative_eagle_topk", 2),
                ("speculative_num_draft_tokens", 3),
                ("page_size", 64),
                ("kv_cache_dtype", "auto"),
                ("disable_shared_experts_fusion", True),
            ):
                cfg = types.SimpleNamespace(**{**vars(CFG), field: value})
                self.assertEqual(len(self.m3.refuse_config(cfg, M3)), 1, field)

    def test_m3_serves_linear_verify_and_plain_decode_only(self):
        spec_args = types.SimpleNamespace(
            speculative_algorithm="EAGLE3", speculative_num_draft_tokens=4
        )
        plain_args = types.SimpleNamespace(speculative_algorithm=None)
        verify = routing.decide_for_batch(self.m3, ctx(spec_args), batch("verify", 4))
        self.assertIsNotNone(verify.step)
        self.assertEqual(verify.step.query_len, 4)
        self.assertEqual(
            routing.decide_for_batch(
                self.m3, ctx(spec_args), batch("verify", 3)
            ).reason,
            "verify rows a request",
        )
        self.assertIsNone(
            routing.decide_for_batch(self.m3, ctx(spec_args), batch("decode", 1)).step
        )
        self.assertIsNotNone(
            routing.decide_for_batch(self.m3, ctx(plain_args), batch("decode", 1)).step
        )


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
    ALLOWED: set[tuple[str, str]] = (
        set()
    )  # (family file, helper name) a family may redefine

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
