import ast
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from safetensors.torch import save_file

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/speculative/dspark_components/dspark_lora.py"
)
spec = importlib.util.spec_from_file_location("dspark_lora_bank_test", SOURCE)
lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora)


def draft(dtype=torch.float32):
    model = torch.nn.Module()
    layer = torch.nn.Module()
    attn = torch.nn.Module()
    mlp = torch.nn.Module()
    attn.q_size = 4
    attn.kv_size = 2
    attn.qkv_proj = torch.nn.Linear(4, 8, bias=False)
    attn.o_proj = torch.nn.Linear(4, 4, bias=False)
    mlp.gate_up_proj = torch.nn.Linear(4, 12, bias=False)
    mlp.down_proj = torch.nn.Linear(6, 4, bias=False)
    layer.self_attn = attn
    layer.mlp = mlp
    model.layers = torch.nn.ModuleList([layer])
    model.fc = torch.nn.Linear(12, 4, bias=False)
    model.hidden_norm = torch.nn.LayerNorm(4)
    model.embed_tokens = torch.nn.Embedding(8, 4)
    model.lm_head = torch.nn.Linear(4, 8, bias=False)
    model._fused_kv_write_cache = object()
    return model.to(dtype=dtype)


class TestDraftAdapterBank(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def adapter(self, name, modules, zero=False):
        path = self.root / name
        path.mkdir()
        tensors = {}
        pairs = {}
        for module, (rows, cols) in modules.items():
            a = torch.randn(2, cols) / 10
            b = torch.zeros(rows, 2) if zero else torch.randn(rows, 2) / 10
            pairs[module] = (a, b)
            tensors[f"base_model.model.{module}.lora_A.weight"] = a
            tensors[f"base_model.model.{module}.lora_B.weight"] = b
        (path / "adapter_config.json").write_text(
            json.dumps(
                dict(
                    peft_type="LORA",
                    r=2,
                    lora_alpha=6,
                    target_modules=sorted({m.split(".")[-1] for m in modules}),
                )
            )
        )
        save_file(tensors, str(path / "adapter_model.safetensors"))
        return str(path), pairs

    def test_all_projection_slices_and_fc_match_independent_reference(self):
        layout = {
            "layers.0.self_attn.q_proj": (
                "layers.0.self_attn.qkv_proj.weight",
                0,
                4,
                4,
            ),
            "layers.0.self_attn.k_proj": (
                "layers.0.self_attn.qkv_proj.weight",
                4,
                6,
                4,
            ),
            "layers.0.self_attn.v_proj": (
                "layers.0.self_attn.qkv_proj.weight",
                6,
                8,
                4,
            ),
            "layers.0.self_attn.o_proj": ("layers.0.self_attn.o_proj.weight", 0, 4, 4),
            "layers.0.mlp.gate_proj": ("layers.0.mlp.gate_up_proj.weight", 0, 6, 4),
            "layers.0.mlp.up_proj": ("layers.0.mlp.gate_up_proj.weight", 6, 12, 4),
            "layers.0.mlp.down_proj": ("layers.0.mlp.down_proj.weight", 0, 4, 6),
            "fc": ("fc.weight", 0, 4, 12),
        }
        path, pairs = self.adapter(
            "all", {m: (hi - lo, cols) for m, (_, lo, hi, cols) in layout.items()}
        )
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = draft(dtype)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                bank = lora.DSparkDraftAdapterBank(model, {"all": path})
                self.assertTrue(bank.activate("all"))
                expected = {n: t.clone() for n, t in before.items()}
                for module, (key, lo, hi, _) in layout.items():
                    a, b = pairs[module]
                    expected[key][lo:hi] = (
                        before[key][lo:hi].float() + 3 * (b @ a)
                    ).to(dtype)
                for n, p in model.named_parameters():
                    torch.testing.assert_close(p, expected[n], rtol=0, atol=0)
                self.assertIsNone(model._fused_kv_write_cache)
                self.assertNotIn("lm_head.weight", bank.parameters)
                self.assertNotIn("embed_tokens.weight", bank.parameters)
                self.assertEqual(
                    bank.resident_bytes,
                    2
                    * sum(
                        t.numel() * t.element_size()
                        for n, t in before.items()
                        if n in bank.parameters
                    ),
                )

    def test_disjoint_adapters_restore_previous_changes_and_do_not_drift(self):
        rust, _ = self.adapter("rust", {"layers.0.self_attn.q_proj": (4, 4)})
        python, _ = self.adapter("python", {"fc": (4, 12)})
        model = draft(torch.bfloat16)
        base = {n: p.detach().clone() for n, p in model.named_parameters()}
        pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
        bank = lora.DSparkDraftAdapterBank(model, {"rust": rust, "python": python})
        for _ in range(20):
            for name in ("rust", "python", None):
                bank.activate(name)
                for n, p in model.named_parameters():
                    expected = bank.variants.get(name, {}).get(n, base[n])
                    torch.testing.assert_close(p, expected, rtol=0, atol=0)
                    self.assertEqual(p.data_ptr(), pointers[n])
                cache = object()
                model._fused_kv_write_cache = cache
                self.assertFalse(bank.activate(name))
                self.assertIs(model._fused_kv_write_cache, cache)
        self.assertEqual(bank.switch_count, 60)

    def test_zero_adapter_and_shared_target_weights_unchanged(self):
        zero, _ = self.adapter("zero", {"fc": (4, 12)}, zero=True)
        model = draft()
        target = torch.nn.Module()
        target.embed_tokens = model.embed_tokens
        target.lm_head = model.lm_head
        before = {n: p.clone() for n, p in model.named_parameters()}
        bank = lora.DSparkDraftAdapterBank(model, {"zero": zero})
        bank.activate("zero")
        bank.activate(None)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        self.assertIs(model.lm_head, target.lm_head)

    def test_stacked_context_kv_projection_rebuilds_after_adapter_switch(self):
        path = SOURCE.parents[2] / "models/dspark.py"
        tree = ast.parse(path.read_text())
        mixin = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DSparkDraftMixin"
        )
        method = next(
            n
            for n in mixin.body
            if isinstance(n, ast.FunctionDef) and n.name == "_stacked_ctx_kv_params"
        )
        namespace = {
            "torch": torch,
            "envs": NS(SGLANG_DSPARK_STACKED_CTX_KV=NS(get=lambda: True)),
            "can_dflash_slice_qkv_weight": lambda layer: (True, None),
        }
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(method),
                str(path),
                "exec",
            ),
            namespace,
        )
        model = draft(torch.float16)
        attn = model.layers[0].self_attn
        attn.k_norm = torch.nn.LayerNorm(2).to(dtype=torch.float16)
        attn.k_norm.variance_epsilon = 1e-6
        model._stacked_ctx_kv_params = types.MethodType(
            namespace["_stacked_ctx_kv_params"], model
        )
        adapter, _ = self.adapter("kv", {"layers.0.self_attn.k_proj": (2, 4)})
        bank = lora.DSparkDraftAdapterBank(model, {"kv": adapter})
        initial = model._stacked_ctx_kv_params()
        expected_base = initial["weight"].clone()
        for name in ("kv", None, "kv"):
            old = model._stacked_ctx_kv_params()
            bank.activate(name)
            fresh = model._stacked_ctx_kv_params()
            self.assertIsNot(fresh, old)
            torch.testing.assert_close(
                fresh["weight"], attn.qkv_proj.weight[4:], rtol=0, atol=0
            )
            if name is None:
                torch.testing.assert_close(
                    fresh["weight"], expected_base, rtol=0, atol=0
                )
            else:
                self.assertFalse(torch.equal(fresh["weight"], expected_base))
            self.assertFalse(bank.activate(name))
            self.assertIs(model._stacked_ctx_kv_params(), fresh)

    def test_bad_adapter_does_not_mutate_live_model(self):
        good, _ = self.adapter("good", {"fc": (4, 12)})
        bad, _ = self.adapter("bad", {"fc": (5, 12)})
        model = draft()
        before = {n: p.clone() for n, p in model.named_parameters()}
        with self.assertRaises(ValueError):
            lora.DSparkDraftAdapterBank(model, {"good": good, "bad": bad})
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        bank = lora.DSparkDraftAdapterBank(model, {"good": good})
        with self.assertRaises(ValueError):
            bank.activate("unknown")
        self.assertIsNone(bank.active)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)

    def test_packed_partial_update_does_not_modify_other_slices(self):
        path, _ = self.adapter(
            "k", {"layers.0.self_attn.k_proj": (2, 4), "layers.0.mlp.up_proj": (6, 4)}
        )
        model = draft()
        qkv = model.layers[0].self_attn.qkv_proj.weight.clone()
        mlp = model.layers[0].mlp.gate_up_proj.weight.clone()
        bank = lora.DSparkDraftAdapterBank(model, {"k": path})
        bank.activate("k")
        actual = model.layers[0].self_attn.qkv_proj.weight
        torch.testing.assert_close(actual[:4], qkv[:4], rtol=0, atol=0)
        torch.testing.assert_close(actual[6:], qkv[6:], rtol=0, atol=0)
        self.assertFalse(torch.equal(actual[4:6], qkv[4:6]))
        torch.testing.assert_close(
            model.layers[0].mlp.gate_up_proj.weight[:6], mlp[:6], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
