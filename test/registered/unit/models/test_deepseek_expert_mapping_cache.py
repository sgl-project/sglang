"""Exercise the integrated loader's mapping cache without constructing a model."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common import deepseek_weight_loader as loader_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Parameter:
    def __init__(self):
        self.loads = []

    def weight_loader(self, param, weight, *args, **kwargs):
        assert param is self
        self.loads.append((weight, args, kwargs))


class _Model(loader_module.DeepseekV2WeightLoaderMixin):
    def __init__(self, mapping, params):
        self.expert_params_mapping = mapping
        self.params = params
        self.model = SimpleNamespace(start_layer=0, end_layer=79)
        self.config = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=1)
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self.quant_config = None
        self.num_fused_shared_experts = 0
        self.fuse_qkv_a_proj = False
        self.stacked_params_mapping = []

    def named_parameters(self):
        return iter(self.params.items())

    def _maybe_quant_weights_to_fp8_ue8m0(self, weights, *args):
        return weights

    def post_load_weights(self, **kwargs):
        pass


def _mapping():
    return [
        (
            "experts.w13_" if shard != "w2" else "experts.w2_",
            f"experts.{e}.{proj}.",
            e,
            shard,
        )
        for e in range(3)
        for proj, shard in [("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")]
    ]


class TestDeepseekExpertMappingCache(unittest.TestCase):
    def test_cold_and_warm_dispatch_all_components(self):
        params = {
            f"model.layers.3.mlp.experts.{matrix}_{component}": _Parameter()
            for matrix in ("w13", "w2")
            for component in ("weight", "weight_scale", "weight_scale_2")
        }
        model = _Model(_mapping(), params)
        weights = [
            (f"model.layers.3.mlp.experts.{e}.{proj}.{component}", object())
            for e in range(3)
            for proj in ("gate_proj", "down_proj", "up_proj")
            for component in ("weight", "weight_scale", "weight_scale_2")
        ]
        model.do_load_weights(weights)
        first = {n: p.loads[:] for n, p in params.items()}
        cache = model._expert_mapping_start_cache
        self.assertEqual(len(cache), len(weights))
        model.do_load_weights(weights)
        self.assertIs(cache, model._expert_mapping_start_cache)
        for name, param in params.items():
            self.assertEqual(param.loads, first[name] * 2)
        self.assertEqual(sum(map(len, first.values())), 27)

    def test_changed_parameter_objects_are_resolved_again(self):
        name = "model.layers.3.mlp.experts.0.gate_proj.weight_scale_2"
        target = "model.layers.3.mlp.experts.w13_weight_scale_2"
        first, second = _Parameter(), _Parameter()
        model = _Model(_mapping(), {target: first})
        model.do_load_weights([(name, "first")])
        model.params[target] = second
        model.do_load_weights([(name, "second")])
        self.assertEqual([x[0] for x in first.loads], ["first"])
        self.assertEqual([x[0] for x in second.loads], ["second"])

    def test_overlap_priority_and_missing_target_fallthrough(self):
        mapping = [("X", "late", 1, "w1"), ("Y", "early", 2, "w2"), ("Z", "X", 3, "w3")]
        model = _Model(mapping, {})
        for target, shard in [("early-X", "w1"), ("Y-X", "w2"), ("Y-Z", "w3")]:
            with self.subTest(target=target):
                param = _Parameter()
                model.params = {target: param}
                model.do_load_weights([("early-late", object())])
                self.assertEqual(param.loads[0][1], (target,))
                self.assertEqual(param.loads[0][2]["shard_id"], shard)
        self.assertEqual(model._expert_mapping_start_cache, {"early-late": 0})

    def test_mapping_mutation_and_reordering_invalidate(self):
        mapping = [("X", "late", 1, "w1"), ("Y", "early", 2, "w2")]
        model = _Model(mapping, {"early-X": _Parameter(), "Y-late": _Parameter()})
        model.do_load_weights([("early-late", object())])
        cache = model._expert_mapping_start_cache
        mapping.reverse()
        model.do_load_weights([("early-late", object())])
        self.assertIsNot(cache, model._expert_mapping_start_cache)
        self.assertEqual(len(model.params["Y-late"].loads), 1)
        cache = model._expert_mapping_start_cache
        mapping[0] = ("Z", "early", 3, "w3")
        model.params["Z-late"] = _Parameter()
        model.do_load_weights([("early-late", object())])
        self.assertIsNot(cache, model._expert_mapping_start_cache)
        self.assertEqual(model.params["Z-late"].loads[0][2]["expert_id"], 3)

    def test_unmatched_name_uses_current_default_parameter(self):
        model = _Model(_mapping(), {})
        name = "model.layers.3.self_attn.other.weight"
        model.do_load_weights([(name, "missing")])
        self.assertEqual(model._expert_mapping_start_cache[name], len(_mapping()))
        param = _Parameter()
        model.params[name] = param
        model.do_load_weights([(name, "present")])
        self.assertEqual(param.loads, [("present", (), {})])

    def test_cache_cap_keeps_fallback_and_existing_hits(self):
        self.assertEqual(loader_module._EXPERT_MAPPING_CACHE_MAX_ENTRIES, 262_144)
        mapping = [(f"dest{i}", f"key{i}", i, "w1") for i in range(4)]
        model = _Model(mapping, {f"dest{i}": _Parameter() for i in range(4)})
        with patch.object(loader_module, "_EXPERT_MAPPING_CACHE_MAX_ENTRIES", 2):
            for _ in range(2):
                model.do_load_weights([(f"key{i}", i) for i in range(4)])
        self.assertEqual(model._expert_mapping_start_cache, {"key0": 0, "key1": 1})
        self.assertTrue(all(len(p.loads) == 2 for p in model.params.values()))
        for i in range(4):
            self.assertEqual(model.params[f"dest{i}"].loads[1][2]["expert_id"], i)

    def test_cache_is_per_model(self):
        a, b = _Model([], {}), _Model([], {})
        a.do_load_weights([("a", None)])
        b.do_load_weights([("b", None)])
        self.assertEqual(a._expert_mapping_start_cache, {"a": 0})
        self.assertEqual(b._expert_mapping_start_cache, {"b": 0})

    def test_nextn_names_and_target_layer_exclusion(self):
        target = "model.layers.3.mlp.experts.w13_weight"
        draft = "model.decoder.mlp.experts.w13_weight"
        model = _Model(_mapping(), {target: _Parameter(), draft: _Parameter()})
        weights = [
            (f"model.layers.{i}.mlp.experts.0.gate_proj.weight", i) for i in (3, 78)
        ]
        for _ in range(2):
            model.do_load_weights(weights, is_nextn=False)
            model.do_load_weights(weights, is_nextn=True)
        self.assertEqual([x[0] for x in model.params[target].loads], [3, 3])
        self.assertEqual([x[0] for x in model.params[draft].loads], [78, 78])
        self.assertEqual(len(model._expert_mapping_start_cache), 2)

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires CUDA for device-copy routing"
    )
    def test_cuda_warm_dispatch_keeps_values_and_target_addresses(self):
        # Cache only dispatch metadata. CUDA stores remain the parameter's loader.
        class TensorParameter(_Parameter):
            def __init__(self, dtype):
                super().__init__()
                self.targets = {
                    (e, s): torch.empty(8, dtype=dtype, device="cuda")
                    for e in range(3)
                    for s in ("w1", "w2", "w3")
                }

            def weight_loader(self, param, weight, name, **kwargs):
                self.targets[kwargs["expert_id"], kwargs["shard_id"]].copy_(weight)

        params = {
            f"model.layers.3.mlp.experts.{matrix}_{component}": TensorParameter(dtype)
            for matrix in ("w13", "w2")
            for component, dtype in [
                ("weight", torch.uint8),
                ("weight_scale", torch.float8_e4m3fn),
                ("weight_scale_2", torch.float32),
            ]
        }
        model = _Model(_mapping(), params)
        pointers = {
            n: {k: t.data_ptr() for k, t in p.targets.items()}
            for n, p in params.items()
        }
        for version in (1, 2):
            for e in range(3):
                for proj, shard in [
                    ("gate_proj", "w1"),
                    ("down_proj", "w2"),
                    ("up_proj", "w3"),
                ]:
                    for component, dtype in [
                        ("weight", torch.uint8),
                        ("weight_scale", torch.float8_e4m3fn),
                        ("weight_scale_2", torch.float32),
                    ]:
                        weight = torch.full(
                            (8,), version + e, dtype=torch.float32, device="cuda"
                        ).to(dtype)
                        name = f"model.layers.3.mlp.experts.{e}.{proj}.{component}"
                        model.do_load_weights([(name, weight)])
                        matrix = "w2" if shard == "w2" else "w13"
                        target_name = f"model.layers.3.mlp.experts.{matrix}_{component}"
                        actual = params[target_name].targets[e, shard]
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.uint8), weight.view(torch.uint8)
                            )
                        )
                        self.assertEqual(
                            actual.data_ptr(), pointers[target_name][e, shard]
                        )


if __name__ == "__main__":
    unittest.main()
