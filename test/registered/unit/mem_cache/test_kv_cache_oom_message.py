"""CPU tests for the KV-cache OOM diagnostic.

The `--mem-fraction-static` suggestion prices the weights against free memory
and cannot shrink the hybrid state cache charged afterwards, so an overflow
caused by that cache used to suggest a fraction the user already exceeded and
never named the binding term. The message now says whether the weights or the
state cache crossed the zero line.
"""

import unittest

from sglang.srt.mem_cache.kv_cache_configurator import kv_cache_oom_message
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _message(**overrides):
    kwargs = dict(
        mem_fraction_static=0.90,
        available_gpu_memory=7771.0,
        pre_model_load_memory=10000.0,
        rest_before_state_cache=20.0,
        rest_after_state_cache=18.0,
        max_mamba_cache_size=16,
        speculative_num_draft_tokens=16,
    )
    kwargs.update(overrides)
    return kv_cache_oom_message(**kwargs)


class TestKVCacheOOMDiagnostic(CustomTestCase):
    def test_weights_exhausting_the_budget_keep_the_old_advice(self):
        # The weights alone crossed zero, so the state cache is not the term
        # to blame even though it was charged afterwards.
        message = _message(rest_before_state_cache=-1.0, rest_after_state_cache=-3.0)
        self.assertIn("Loaded weights leave no GPU memory for the KV cache", message)
        self.assertIn("Raise --mem-fraction-static above 0.223", message)
        self.assertIn("minimum viable = 1 - available/pre = 0.2229", message)
        self.assertNotIn("state cache", message)
        self.assertNotIn("intermediate_ssm_state_cache", message)

    def test_no_state_cache_deduction_keeps_the_old_advice(self):
        message = _message(rest_before_state_cache=2.0, rest_after_state_cache=2.0)
        self.assertNotIn("state cache", message)
        self.assertNotIn("--max-mamba-cache-size", message)

    def test_state_cache_crossing_zero_is_named_with_its_knobs(self):
        message = _message(rest_before_state_cache=1.50, rest_after_state_cache=-7.50)
        # The weights still get the fraction advice...
        self.assertIn("Raise --mem-fraction-static above 0.223", message)
        # ...but the binding term and every relevant knob are named.
        self.assertIn("intermediate_ssm_state_cache", message)
        self.assertIn("charge after the weights was 9.00 GB", message)
        self.assertIn("more than the 1.50 GB left there", message)
        self.assertIn("--mem-fraction-static does not shrink it", message)
        self.assertIn("lower --max-mamba-cache-size (currently 16)", message)
        self.assertIn("lower --speculative-num-draft-tokens (currently 16)", message)

    def test_without_speculation_only_the_mamba_knob_is_suggested(self):
        message = _message(speculative_num_draft_tokens=None)
        self.assertIn("intermediate_ssm_state_cache", message)
        self.assertIn("lower --max-mamba-cache-size (currently 16)", message)
        self.assertNotIn("--speculative-num-draft-tokens", message)

    def test_auto_sized_mamba_pool_points_at_the_size_and_request_caps(self):
        message = _message(max_mamba_cache_size=None)
        self.assertIn("set --max-mamba-cache-size below the auto-sized pool", message)
        self.assertIn("lower --max-running-requests", message)
        self.assertNotIn("currently None", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
