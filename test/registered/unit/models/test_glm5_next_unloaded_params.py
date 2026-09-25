"""`load_weights` warns when it did not fill every parameter."""

import logging
import unittest

import torch

from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.models.glm5_next import (
    _UNLOADED_REPORT_LIMIT,
    report_unloaded_params,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestReportUnloadedParams(unittest.TestCase):
    def test_a_complete_load_says_nothing(self):
        names = {"model.layers.0.self_attn.qkv_proj.weight", "model.norm.weight"}
        with self.assertNoLogs("sglang.srt.models.glm5_next", level="WARNING"):
            self.assertEqual(report_unloaded_params(names, names), [])

    def test_a_dropped_parameter_is_named(self):
        declared = {"a.weight", "b.weight"}
        missing = report_unloaded_params(declared, {"a.weight"})
        self.assertEqual(missing, ["b.weight"])

    def test_the_warning_carries_the_count_and_the_total(self):
        declared = {f"layers.{i}.mlp.experts.{i}.w1.weight" for i in range(5)}
        with self.assertLogs("sglang.srt.models.glm5_next", level="WARNING") as cm:
            report_unloaded_params(declared, set())
        message = "\n".join(cm.output)
        self.assertIn("5 of 5 parameters were not initialized", message)
        self.assertIn("not meaningful", message)

    def test_a_long_list_is_truncated_but_counted(self):
        declared = {f"layers.{i}.w" for i in range(_UNLOADED_REPORT_LIMIT + 20)}
        with self.assertLogs("sglang.srt.models.glm5_next", level="WARNING") as cm:
            missing = report_unloaded_params(declared, set())
        message = "\n".join(cm.output)
        self.assertEqual(len(missing), _UNLOADED_REPORT_LIMIT + 20)
        self.assertIn(f"+{20} more", message)

    def test_extra_loaded_names_are_not_reported(self):
        declared = {"attn.fused_qkv_a_proj_with_mqa.weight"}
        loaded = {
            "attn.fused_qkv_a_proj_with_mqa.weight",
            "attn.q_a_proj.weight",
            "attn.kv_a_proj_with_mqa.weight",
        }
        self.assertEqual(report_unloaded_params(declared, loaded), [])

    def test_the_label_distinguishes_one_pass_from_another(self):
        with self.assertLogs("sglang.srt.models.glm5_next", level="WARNING") as cm:
            report_unloaded_params({"a"}, set(), model_label="Glm5NextForCausalLM")
        self.assertIn("Glm5NextForCausalLM", "\n".join(cm.output))

    def test_the_result_is_sorted_so_a_log_is_stable(self):
        declared = {"c", "a", "b"}
        self.assertEqual(report_unloaded_params(declared, set()), ["a", "b", "c"])

    def test_iterables_are_accepted_not_only_sets(self):
        self.assertEqual(report_unloaded_params(["a", "b"], iter(["a"])), ["b"])

    def test_optional_params_are_not_reported(self):
        # The released GLM-5.3-Flash checkpoint carries no KV-cache scales.
        declared = {
            "layers.11.self_attn.attn_mha.k_scale",
            "layers.11.self_attn.attn_mha.v_scale",
            "layers.11.self_attn.o_proj.weight",
        }
        optional = {n for n in declared if n.endswith(("k_scale", "v_scale"))}
        with self.assertNoLogs("sglang.srt.models.glm5_next", level="WARNING"):
            missing = report_unloaded_params(
                declared,
                {"layers.11.self_attn.o_proj.weight"},
                optional_params=optional,
            )
        self.assertEqual(missing, [])
        self.assertEqual(
            report_unloaded_params(declared, set(), optional_params=optional),
            ["layers.11.self_attn.o_proj.weight"],
        )

    def test_kv_cache_scales_are_marked_optional(self):
        # load_weights passes the parameters marked _skip_weight_check as optional.
        layer = torch.nn.Module()
        BaseKVCacheMethod(None).create_weights(layer)
        self.assertTrue(layer.k_scale._skip_weight_check)
        self.assertTrue(layer.v_scale._skip_weight_check)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
