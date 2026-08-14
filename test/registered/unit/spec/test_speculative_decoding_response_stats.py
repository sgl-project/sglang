"""Unit coverage for request-level speculative decoding response statistics."""

from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import SglExt
from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.srt.entrypoints.openai.utils import (
    speculative_decoding_stats_from_meta,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _recv_obj():
    return SimpleNamespace(
        spec_verify_ct=[2],
        spec_num_correct_drafts=[5],
        spec_num_proposed_drafts=[9],
        spec_correct_drafts_histogram=[[0, 1, 0, 1]],
        spec_num_cap_tokens=[0],
        spec_num_block_accept_tokens=[0],
        spec_cap_lens_histogram=[[]],
        spec_verify_lens=[[6, 5]],
        spec_accept_lens=[[4, 3]],
        completion_tokens=[6],
    )


class TestSpeculativeDecodingResponseStats(unittest.TestCase):
    def _calculate(self, mode: str):
        manager = object.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(
            speculative_decoding_stats=mode,
            speculative_num_draft_tokens=99,
        )
        meta_info = {}
        manager._calculate_spec_decoding_metrics(meta_info, _recv_obj(), 0)
        return meta_info

    def test_detailed_uses_exact_width_and_raw_ordered_lengths(self):
        stats = self._calculate("detailed")["speculative_decoding_stats"]

        self.assertEqual(stats["num_verification_steps"], 2)
        self.assertEqual(stats["num_verified_draft_tokens"], 9)
        self.assertEqual(stats["num_accepted_draft_tokens"], 5)
        self.assertAlmostEqual(stats["draft_acceptance_rate"], 5 / 9)
        self.assertEqual(stats["mean_accept_length"], 3.5)
        self.assertEqual(stats["verify_lengths"], [6, 5])
        self.assertEqual(stats["accept_lengths"], [4, 3])

        self.assertEqual(len(stats["verify_lengths"]), stats["num_verification_steps"])
        self.assertEqual(len(stats["accept_lengths"]), stats["num_verification_steps"])
        self.assertEqual(
            sum(length - 1 for length in stats["verify_lengths"]),
            stats["num_verified_draft_tokens"],
        )
        self.assertEqual(
            sum(length - 1 for length in stats["accept_lengths"]),
            stats["num_accepted_draft_tokens"],
        )

    def test_summary_omits_step_arrays(self):
        stats = self._calculate("summary")["speculative_decoding_stats"]

        self.assertNotIn("verify_lengths", stats)
        self.assertNotIn("accept_lengths", stats)

    def test_none_keeps_existing_metrics_but_has_no_public_stats(self):
        meta_info = self._calculate("none")

        self.assertEqual(meta_info["spec_num_proposed_drafts"], 9)
        self.assertNotIn("speculative_decoding_stats", meta_info)

    def test_sglext_schema_and_chat_sse_serialization(self):
        internal = self._calculate("detailed")["speculative_decoding_stats"]
        stats = speculative_decoding_stats_from_meta(
            {"speculative_decoding_stats": internal}, 0
        )
        extension = SglExt(speculative_decoding_stats=[stats])

        chunk = build_sse_content(
            chunk_id="chatcmpl-test",
            created=1,
            model="test",
            index=0,
            finish_reason="stop",
            sglext=extension.model_dump(exclude_none=True),
        )
        payload = json.loads(chunk.removeprefix("data: "))

        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        record = payload["sglext"]["speculative_decoding_stats"][0]
        self.assertEqual(record["index"], 0)
        self.assertEqual(record["verify_lengths"], [6, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
