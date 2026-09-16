"""`load_weights` must say so when it did not fill every parameter.

Skipping a checkpoint tensor whose rewritten name misses `params_dict` is right
for a tensor this model does not own, and indistinguishable from a checkpoint
whose names this loader was not written against. In the second case the module
keeps its initialised values and the server starts on a partly-random model --
it logs that it is ready, answers /health with 200, and generates noise.

That is sgl-project/sglang#38618: 524 of 1609 tensors, 304.4B parameters, 97% of
a 628 GB file, dropped with no warning, no traceback and no non-zero exit.

These tests pin `report_unloaded_params`, the part that decides whether anything
is said at all.
"""

import logging
import unittest

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
        # The operator has to be able to tell this is not a slow load.
        self.assertIn("not meaningful", message)

    def test_a_long_list_is_truncated_but_counted(self):
        declared = {f"layers.{i}.w" for i in range(_UNLOADED_REPORT_LIMIT + 20)}
        with self.assertLogs("sglang.srt.models.glm5_next", level="WARNING") as cm:
            missing = report_unloaded_params(declared, set())
        message = "\n".join(cm.output)
        self.assertEqual(len(missing), _UNLOADED_REPORT_LIMIT + 20)
        self.assertIn(f"+{20} more", message)

    def test_extra_loaded_names_are_not_reported(self):
        """A fused parameter is recorded under the name params_dict holds.

        `q_a_proj` / `kv_a_proj_with_mqa` are the checkpoint's halves of
        `fused_qkv_a_proj_with_mqa`; recording the fused name must not make the
        halves look like a surplus or a gap.
        """
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
        """params_dict.keys() is a view, and loaded_params is a set."""
        self.assertEqual(report_unloaded_params(["a", "b"], iter(["a"])), ["b"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
