# Copyright 2023-2026 SGLang Team
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
"""Post-capture KV sizing must not disable itself silently.

`post_capture_kv_sizing_planned()` is a *predicate*, so every reason it answers
False is invisible at the point the operator can act on it. Most of its early
returns are genuinely inapplicable configurations (no CUDA, MLA, unified
memory, Mooncake over EFA, ...) and saying nothing is correct.

The prefill-capture-coverage check is different: it compares two knobs the
operator sets independently -- `chunked_prefill_size` (the scheduling ceiling,
via `max_prefill_buffer_tokens()`) and `cuda_graph_config.prefill.bs` (capture
coverage). Passing `--cuda-graph-max-bs-prefill` below `chunked_prefill_size`
used to turn the feature off with no diagnostic at all. That is exactly what
happened in #33847, where a CI-wide `--cuda-graph-max-bs-prefill 1024` silenced
the feature on H100-class machines, whose `chunked_prefill_size` defaults to
8192. The RFC (#33852) asks for the warning as an item that can land on its own,
independent of the larger proposal to move the check onto the headroom path.

Pinned here: the coverage shortfall warns and names both values plus the knob
that fixes it, the warning reports the *buffer ceiling* rather than the raw
chunk size (PP dynamic chunking raises the ceiling above
`chunked_prefill_size`), the remedy names a real capture bucket rather than an
off-grid number, and the two inapplicable gates stay silent so the log does not
cry wolf.

    python -m pytest test/registered/unit/server_args/test_post_capture_kv_sizing_warning.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import post_capture_kv_sizing_planned
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _run(
    *,
    prefill_backend=Backend.BREAKABLE,
    prefill_bs=(1024,),
    chunked_prefill_size=8192,
    max_prefill_tokens=8192,
    enable_dynamic_chunking=False,
    pp_size=1,
    disaggregation_mode="null",
    decode_backend=Backend.FULL,
    locked=(),
):
    """Run the predicate over a minimal stand-in and return (result, warning).

    `resolving_view` is patched rather than resolving a real `ServerArgs`,
    matching how the sibling coverage in
    `unit/disaggregation/test_mooncake_efa_allocator.py` drives this function.
    `locked` seeds `_cuda_graph_config_locked`, the set of (`phase`, `key`)
    pairs whose cuda-graph value came from an explicit source.
    """
    cfg = SimpleNamespace(
        enable_unified_memory=False,
        device="cuda",
        dcp_size=1,
        prefill_only_disable_kv_cache=False,
        enable_memory_saver=False,
        disaggregation_transfer_backend="mooncake",
        disaggregation_mode=disaggregation_mode,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=max_prefill_tokens,
        enable_dynamic_chunking=enable_dynamic_chunking,
        pp_size=pp_size,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=prefill_backend, bs=list(prefill_bs)),
            decode=SimpleNamespace(backend=decode_backend),
        ),
    )
    with (
        patch("sglang.srt.arg_groups.overrides.resolving_view", return_value=cfg),
        patch("sglang.srt.arg_groups.overrides.use_mla_backend", return_value=False),
        patch.object(
            envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING, "get", return_value=True
        ),
        patch.object(envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL, "get", return_value=None),
        # Not EFA: the EFA early return would mask the prefill check entirely.
        patch.object(envs.MOONCAKE_PROTOCOL, "get", return_value="rdma"),
        patch("sglang.srt.configs.model_config.is_deepseek_v4", return_value=False),
        patch("sglang.srt.configs.model_config.is_minimax_sparse", return_value=False),
        patch(
            "sglang.srt.arg_groups.overrides.model_config_of",
            return_value=SimpleNamespace(hf_config=SimpleNamespace()),
        ),
        patch("sglang.srt.arg_groups.overrides.logger.warning") as warning,
    ):
        planned = post_capture_kv_sizing_planned(
            SimpleNamespace(_cuda_graph_config_locked=set(locked))
        )
    return planned, warning


def _rendered(warning):
    """The log line as the operator reads it.

    `logging` defers %-formatting to the handler, so the flag names this module
    interpolates through `%s` are arguments, not part of the format string.
    """
    args = warning.call_args.args
    return args[0] % args[1:]


class TestPostCaptureKVSizingWarning(CustomTestCase):
    def test_warns_when_prefill_buckets_do_not_cover_the_buffer(self):
        """BUG REGRESSION. The #33847 configuration: the prefill graph tops out
        at 1024 while chunked prefill still admits 8192-token batches. The
        feature turned off with nothing in the log."""
        planned, warning = _run(prefill_bs=(1024,), chunked_prefill_size=8192)

        self.assertFalse(planned)
        self.assertEqual(warning.call_count, 1)
        # Both values, so the operator can see which knob is short.
        self.assertIn(8192, warning.call_args.args)
        self.assertIn(1024, warning.call_args.args)
        rendered = _rendered(warning)
        self.assertIn("Post-capture KV sizing is disabled", rendered)
        # And the flag that fixes it.
        self.assertIn("--cuda-graph-max-bs-prefill", rendered)

    def test_no_warning_when_buckets_cover_the_buffer(self):
        planned, warning = _run(prefill_bs=(1024, 8192), chunked_prefill_size=8192)

        self.assertTrue(planned)
        warning.assert_not_called()

    def test_warns_on_an_unsorted_bucket_list(self):
        """The check is on the largest bucket, not the last one."""
        planned, warning = _run(prefill_bs=(8192, 1024), chunked_prefill_size=8192)

        self.assertTrue(planned)
        warning.assert_not_called()

    def test_warning_reports_the_buffer_ceiling_not_the_chunk_size(self):
        """PP dynamic chunking grows the ceiling past chunked_prefill_size, and
        the ceiling is what the graph must cover."""
        planned, warning = _run(
            prefill_bs=(1024,),
            chunked_prefill_size=8192,
            max_prefill_tokens=16384,
            enable_dynamic_chunking=True,
            pp_size=2,
        )

        self.assertFalse(planned)
        self.assertEqual(warning.call_count, 1)
        # 1.25x of 8192 is 10240; max_prefill_tokens 16384 wins.
        self.assertIn(16384, warning.call_args.args)
        self.assertNotIn(8192, warning.call_args.args)

    def test_disabled_prefill_graph_stays_silent(self):
        """`backend == DISABLED` is genuinely inapplicable, not a coverage
        shortfall -- the RFC keeps it as a hard gate and it should not warn."""
        planned, warning = _run(
            prefill_backend=Backend.DISABLED,
            prefill_bs=(1024,),
            chunked_prefill_size=8192,
        )

        self.assertFalse(planned)
        warning.assert_not_called()

    def test_disabled_chunked_prefill_stays_silent(self):
        planned, warning = _run(prefill_bs=(1024,), chunked_prefill_size=-1)

        self.assertFalse(planned)
        warning.assert_not_called()

    def test_gate_still_returns_false_for_the_shortfall(self):
        """The warning reports the shortfall; it does not relax the gate. The
        RFC's larger proposal to move this onto the headroom path is separate."""
        planned, _ = _run(prefill_bs=(1024,), chunked_prefill_size=8192)

        self.assertFalse(planned)

    def test_advice_quotes_a_capture_bucket_not_the_raw_ceiling(self):
        """Capture buckets are quantised and `--cuda-graph-max-bs-prefill` does
        not append its own value, so advising a value between buckets is a
        no-op: the largest bucket stays short and the warning fires again.
        Advise a real bucket instead. 10000 is off the grid; the next bucket up
        is 10240."""
        planned, warning = _run(prefill_bs=(1024,), chunked_prefill_size=10000)

        self.assertFalse(planned)
        self.assertEqual(warning.call_count, 1)
        # The ceiling itself is still reported verbatim...
        self.assertIn(10000, warning.call_args.args)
        # ...but the remedy names a bucket that actually raises the ceiling.
        self.assertIn("at least 10240", _rendered(warning))

    def test_advice_targets_the_bucket_list_when_buckets_are_explicit(self):
        """With `--cuda-graph-bs-prefill` given, `max_bs` no longer generates
        the list, so raising it changes nothing. Point at the list instead."""
        planned, warning = _run(
            prefill_bs=(1024,),
            chunked_prefill_size=8192,
            locked=((Phase.PREFILL, "bs"),),
        )

        self.assertFalse(planned)
        self.assertEqual(warning.call_count, 1)
        rendered = _rendered(warning)
        self.assertIn("--cuda-graph-bs-prefill", rendered)
        self.assertNotIn("--cuda-graph-max-bs-prefill", rendered)

    def test_advice_names_the_max_bs_flag_when_buckets_are_generated(self):
        """The default: an unlocked bucket list is generated from max_bs, so
        max_bs is the knob to raise. An unrelated locked key must not divert
        the advice."""
        planned, warning = _run(
            prefill_bs=(1024,),
            chunked_prefill_size=8192,
            locked=((Phase.PREFILL, "backend"),),
        )

        self.assertFalse(planned)
        rendered = _rendered(warning)
        self.assertIn("--cuda-graph-max-bs-prefill", rendered)
        self.assertNotIn("--cuda-graph-bs-prefill", rendered)


if __name__ == "__main__":
    unittest.main()
