# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the DeepSeek-V4.1 spec: the spans of every layer per step type on the checkpoint's layout, the
refusal reasons for the cookbook recipe and its variants, and where the model calls the weight hooks."""

import ast
import pathlib
import types
import unittest
from unittest import mock

import sglang
from sglang.experimental.monokernels.hip.deepseek_v41 import spec as dsv41
from sglang.experimental.monokernels.spec import StepKind
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = pathlib.Path(sglang.__file__).parent
# DeepSeek-V4.1-Flash config.json (snapshot dba1be0a): the decoder layout the spans depend on
LAYERS = 40
COMPRESS = [0, 0] + [2] * 18 + [1] * 20
ENGRAM = (1, 14)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)


def _model(capture=None):
    layers = []
    for i in range(LAYERS):
        source = i in INDEX_SOURCES
        attn = types.SimpleNamespace(
            compress_ratio=COMPRESS[i],
            compressor=object() if source else None,
            indexer=object() if source else None,
        )
        mlp = types.SimpleNamespace(is_hash=False, is_nextn=False)
        layers.append(
            types.SimpleNamespace(
                hc_boundary_fused=True,
                mlp=mlp,
                self_attn=attn,
                engram=object() if i in ENGRAM else None,
            )
        )
    model = types.SimpleNamespace(
        layers=layers, start_layer=0, end_layer=LAYERS, dspark_layers_to_capture=capture
    )
    return types.SimpleNamespace(model=model)


def _ids(plan, name):
    return {i for i, spans in plan.layers.items() if any(s.name == name for s in spans)}


class TestSpans(unittest.TestCase):
    def test_decode_layer_sets(self):
        plan = dsv41.spans(_model(), StepKind.DECODE)
        k1, ffn = _ids(plan, "k1"), _ids(plan, "ffn")
        self.assertEqual(ffn, set(range(LAYERS)))
        self.assertEqual(len(k1), 30)
        self.assertTrue(k1.isdisjoint({0, *ENGRAM, *INDEX_SOURCES}), sorted(k1))
        for spans in plan.layers.values():
            self.assertEqual(spans[-1], dsv41.FFN)

    def test_verify_leaves_out_captured_layers(self):
        capture = [10, 21, 33]
        decode = _ids(dsv41.spans(_model(capture), StepKind.DECODE), "k1")
        plan = dsv41.spans(_model(capture), StepKind.TARGET_VERIFY)
        verify = _ids(plan, "k1")
        self.assertEqual(verify, decode - set(capture))
        self.assertEqual(len(verify), 27)
        self.assertTrue(
            any("DSpark captures" in n and "[10, 21, 33]" in n for n in plan.notes),
            plan.notes,
        )
        self.assertEqual(_ids(plan, "ffn"), set(range(LAYERS)))


def _cfg(**over):
    fields = dict(speculative_algorithm="DSPARK", tp_size=4)
    fields.update(over)
    return types.SimpleNamespace(**fields)


def _hf():
    return types.SimpleNamespace(**dsv41._MODEL)


class TestConfigReasons(unittest.TestCase):
    def setUp(self):
        envs = dsv41.envs
        self.patches = [
            mock.patch.object(dsv41, "is_gfx95_supported", return_value=True),
            mock.patch.object(dsv41, "get_bool_env_var", return_value=False),
            mock.patch.object(
                envs.SGLANG_RAGGED_VERIFY_MODE, "get", return_value="static"
            ),
            mock.patch.object(envs.SGLANG_DSV4_FP4_EXPERTS, "get", return_value=True),
            mock.patch.object(envs.SGLANG_DSV4_FP4_DEQUANT, "get", return_value=False),
            mock.patch.object(
                envs.SGLANG_USE_AITER_MOE_GU_ITLV, "get", return_value=True
            ),
            mock.patch.object(envs.SGLANG_DSV4_KV_LAYOUT, "get", return_value="v4"),
            mock.patch.object(envs.SGLANG_TEST_MONO_SPANS, "get", return_value=""),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_cookbook_recipe_accepted(self):
        self.assertEqual(dsv41.config_disable_reasons(_cfg(), _hf()), [])
        self.assertEqual(
            dsv41.config_disable_reasons(
                _cfg(speculative_algorithm=None, tp_size=2), _hf()
            ),
            [],
        )

    def test_refusals(self):
        def one(cfg=None, hf=None):
            reasons = dsv41.config_disable_reasons(cfg or _cfg(), hf or _hf())
            self.assertEqual(len(reasons), 1, reasons)
            return reasons[0]

        self.assertIn("DSpark", one(_cfg(speculative_algorithm="EAGLE")))
        self.assertIn("tensor parallel size 8", one(_cfg(tp_size=8)))
        bad = _hf()
        bad.n_routed_experts = 256
        self.assertIn("n_routed_experts", one(hf=bad))
        with mock.patch.object(
            dsv41.envs.SGLANG_RAGGED_VERIFY_MODE, "get", return_value="compact"
        ):
            self.assertIn("SGLANG_RAGGED_VERIFY_MODE", one())
        with mock.patch.object(
            dsv41.envs.SGLANG_TEST_MONO_SPANS, "get", return_value="k1"
        ):
            self.assertIn("k1 needs ffn", one())

    def test_every_reason_at_once(self):
        reasons = dsv41.config_disable_reasons(
            _cfg(speculative_algorithm="EAGLE", tp_size=8), _hf()
        )
        self.assertEqual(len(reasons), 2, reasons)


def _function(path: pathlib.Path, cls: str, name: str) -> str:
    """The source of method ``cls.name`` (or function ``name`` when cls is None), read without importing."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if cls is None and isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return ast.unparse(item)
    raise LookupError(f"{cls}.{name} not in {path}")


class TestHookPlacement(unittest.TestCase):
    """load_tensor sees each checkpoint tensor before renames and quantization; loaded() runs at the end of
    load_weights, before process_weights_after_loading; attach runs when the kernels are precompiled."""

    def test_deepseek_v4_calls(self):
        path = ROOT / "srt" / "models" / "deepseek_v4.py"
        src = _function(path, "DeepseekV4ForCausalLM", "load_weights")
        at = src.index("mono.load_tensor(")
        self.assertLess(at, src.index("_prepare_deepseek_v4_weights("))
        self.assertLess(at, src.index("remap_weight_name_to_dpsk_hf_format("))
        self.assertGreater(
            src.index("mono.loaded(self)"), src.index("self.post_load_weights(")
        )
        pre = _function(
            path, "DeepseekV4ForCausalLM", "precompile_kernels_after_loading"
        )
        body = ast.parse(pre).body[0].body
        self.assertEqual(ast.unparse(body[0]).strip(), "mono.attach(self)")

    def test_loader_runs_post_processing_after_load_weights(self):
        path = ROOT / "srt" / "model_loader" / "loader.py"
        src = _function(path, "DefaultModelLoader", "load_weights_and_postprocess")
        self.assertLess(
            src.index("load_weights_only("), src.index("postprocess_weights(")
        )
        self.assertIn(
            "process_weights_after_loading",
            _function(path, "DefaultModelLoader", "postprocess_weights"),
        )


if __name__ == "__main__":
    unittest.main()
