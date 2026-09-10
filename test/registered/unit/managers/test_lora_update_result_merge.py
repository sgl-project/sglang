# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for merging per-rank LoRA update replies from the control fan-out."""

import unittest

from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.managers.tokenizer_control_mixin import _merge_lora_update_results
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _ok(adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(success=True, loaded_adapters=adapters or {})


def _err(message, adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(
        success=False, error_message=message, loaded_adapters=adapters or {}
    )


class TestMergeLoRAUpdateResults(CustomTestCase):
    def test_all_success_returns_first_rank_result(self):
        """On success the merge must hand back a rank's own reply: callers
        mutate result.loaded_adapters in place during LRU eviction, which a
        synthesized empty result would silently break."""
        results = [_ok({"a": "path"}), _ok({"a": "path"})]
        merged = _merge_lora_update_results(results)
        self.assertIs(merged, results[0])
        self.assertTrue(merged.success)

    def test_any_rank_failure_wins(self):
        """Regression guard for the pre-merge behavior of returning
        results[0]: a failure on a non-zero rank was reported as success,
        letting the tokenizer-side registry drift from that rank's actual
        adapter state."""
        merged = _merge_lora_update_results(
            [_ok({"a": "path"}), _err("out of memory", {"stale": "path"})]
        )
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "out of memory")
        self.assertEqual(merged.loaded_adapters, {"stale": "path"})

    def test_duplicate_error_messages_deduplicated(self):
        """All ranks usually fail identically (e.g. "already loaded"); the
        joined message must not repeat per rank, but distinct causes must all
        be kept."""
        merged = _merge_lora_update_results(
            [_err("already loaded"), _err("already loaded"), _err("bad rank")]
        )
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "already loaded | bad rank")

    def test_failure_without_message(self):
        """A rank replying success=False with error_message=None must not
        crash the join."""
        merged = _merge_lora_update_results([_err(None), _ok()])
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "")


if __name__ == "__main__":
    unittest.main()
