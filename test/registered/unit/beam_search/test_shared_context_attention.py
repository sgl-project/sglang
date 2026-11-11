"""Unit tests for shared-context attention support checks and the context and
decode slot layout built for a mixed-width beam batch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.beam_search import shared_context_attention
from sglang.srt.beam_search.shared_context_attention import (
    build_metadata,
    supports_shared_context,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _radix_attention(logit_cap=0.0, head_dim=4, v_head_dim=4):
    layer = RadixAttention.__new__(RadixAttention)
    torch.nn.Module.__init__(layer)
    layer.logit_cap = logit_cap
    layer.head_dim = head_dim
    layer.v_head_dim = v_head_dim
    return layer


class _FakeRunner(SimpleNamespace):
    """Weak-referenceable stand-in for ModelRunner (the verdict cache is keyed
    weakly by runner)."""

    __hash__ = object.__hash__


class _SharedContextTestCase(CustomTestCase):
    def setUp(self):
        self.enterContext(
            envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(True)
        )
        self.two_batch_overlap = False
        self.enterContext(
            patch(
                "sglang.srt.runtime_context.get_exec",
                lambda: SimpleNamespace(
                    overlap=SimpleNamespace(
                        enable_two_batch_overlap=self.two_batch_overlap
                    )
                ),
            )
        )

    @staticmethod
    def model_runner(**overrides):
        options = dict(
            decode_attention_backend_str="flashinfer",
            sliding_window_size=None,
            model_config=SimpleNamespace(
                is_encoder_decoder=False, attention_arch=AttentionArch.MHA
            ),
            kv_cache_dtype=torch.bfloat16,
            model=torch.nn.Sequential(_radix_attention()),
            # Row r of the pool maps positions 0..2 to slots 3r..3r+2.
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(60, dtype=torch.int32).view(20, 3)
            ),
        )
        return _FakeRunner(**(options | overrides))


class TestSupportsSharedContext(_SharedContextTestCase):
    def reason_logged(self, runner):
        with self.assertLogs(shared_context_attention.logger, "WARNING") as logs:
            self.assertFalse(supports_shared_context(runner))
        return "\n".join(logs.output)

    def test_supported_setup(self):
        runner = self.model_runner()
        self.assertTrue(supports_shared_context(runner))
        with self.assertNoLogs(shared_context_attention.logger, "WARNING"):
            self.assertTrue(supports_shared_context(runner))

    def test_unsupported_setups_report_their_reason(self):
        for reason, overrides in (
            ("not FlashInfer", {"decode_attention_backend_str": "triton"}),
            ("sliding-window", {"sliding_window_size": 128}),
            ("KV cache dtype", {"kv_cache_dtype": torch.float8_e4m3fn}),
            (
                "encoder-decoder",
                {
                    "model_config": SimpleNamespace(
                        is_encoder_decoder=True, attention_arch=AttentionArch.MHA
                    )
                },
            ),
            (
                "only MHA",
                {
                    "model_config": SimpleNamespace(
                        is_encoder_decoder=False, attention_arch=AttentionArch.MLA
                    )
                },
            ),
            (
                "logit caps",
                {"model": torch.nn.Sequential(_radix_attention(logit_cap=30.0))},
            ),
            (
                "v_head_dim",
                {"model": torch.nn.Sequential(_radix_attention(v_head_dim=8))},
            ),
        ):
            with self.subTest(reason=reason):
                runner = self.model_runner(**overrides)
                self.assertIn(reason, self.reason_logged(runner))

    def test_two_batch_overlap_is_unsupported(self):
        self.two_batch_overlap = True
        self.assertIn("two-batch overlap", self.reason_logged(self.model_runner()))

    def test_verdict_is_cached_and_warned_once_per_runner(self):
        runner = self.model_runner(decode_attention_backend_str="triton")
        self.reason_logged(runner)
        with self.assertNoLogs(shared_context_attention.logger, "WARNING"):
            self.assertFalse(supports_shared_context(runner))
        # A later config change does not flip a settled verdict.
        runner.decode_attention_backend_str = "flashinfer"
        self.assertFalse(supports_shared_context(runner))


class TestBuildMetadata(_SharedContextTestCase):
    @staticmethod
    def entry(leader_idx, start, end, prompt_len=1):
        return SimpleNamespace(
            leader_idx=leader_idx,
            start=start,
            end=end,
            group=SimpleNamespace(prompt_len=prompt_len),
        )

    def batch(self, entries=None, seq_lens=None, **overrides):
        if entries is None:
            # Widths 2, 2, 3 over 3 base rows followed by 4 beam rows.
            entries = [self.entry(0, 0, 1), self.entry(1, 1, 2), self.entry(2, 2, 4)]
        seq_lens = torch.tensor(
            [3, 2, 2, 3, 2, 2, 2] if seq_lens is None else seq_lens,
            dtype=torch.int32,
        )
        options = dict(
            forward_mode=ForwardMode.DECODE,
            beam_tail=SimpleNamespace(entries=entries, num_base_rows=3),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            req_pool_indices=torch.arange(10, 10 + len(seq_lens)),
            device=torch.device("cpu"),
        )
        return SimpleNamespace(**(options | overrides))

    def test_slot_layout_per_width_bucket(self):
        metadata = build_metadata(self.batch(), self.model_runner())

        self.assertEqual(metadata.num_rows, 7)
        narrow, wide = metadata.buckets
        self.assertEqual((narrow.beam_width, wide.beam_width), (2, 3))

        # Rows are the leader followed by that group's beam rows.
        self.assertEqual(narrow.row_indices.tolist(), [[0, 3], [1, 4]])
        self.assertEqual(wide.row_indices.tolist(), [[2, 5, 6]])
        self.assertEqual(narrow.row_indices.dtype, torch.int32)

        # Context slots: the leader's prompt positions, concatenated per group.
        self.assertEqual(narrow.context_slots.tolist(), [30, 33])
        self.assertEqual(narrow.context_cu_seqlens.tolist(), [0, 1, 2])
        self.assertEqual(wide.context_slots.tolist(), [36])
        self.assertEqual(wide.context_cu_seqlens.tolist(), [0, 1])

        # Decode slots: positions after the prompt, position-major then beam,
        # zero-padded to the bucket's longest decode length.
        self.assertEqual(narrow.decode_lens.tolist(), [2, 1])
        self.assertEqual(
            narrow.decode_slots.tolist(), [[31, 40, 32, 41], [34, 43, 0, 0]]
        )
        self.assertEqual(wide.decode_lens.tolist(), [1])
        self.assertEqual(wide.decode_slots.tolist(), [[37, 46, 49]])

    def test_long_prompts_span_context_slots(self):
        entries = [self.entry(0, 0, 1, prompt_len=2)]
        batch = self.batch(
            entries, seq_lens=[3, 3], req_pool_indices=torch.tensor([4, 5])
        )
        batch.beam_tail.num_base_rows = 1

        (bucket,) = build_metadata(batch, self.model_runner()).buckets

        self.assertEqual(bucket.context_slots.tolist(), [12, 13])
        self.assertEqual(bucket.context_cu_seqlens.tolist(), [0, 2])
        self.assertEqual(bucket.decode_lens.tolist(), [1])
        self.assertEqual(bucket.decode_slots.tolist(), [[14, 17]])

    def test_returns_none_when_disabled_or_unsupported(self):
        batch, runner = self.batch(), self.model_runner()
        self.assertIsNotNone(build_metadata(batch, runner))
        with envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(False):
            self.assertIsNone(build_metadata(batch, runner))
        for overrides in (
            {"decode_attention_backend_str": "triton"},
            {"kv_cache_dtype": torch.float8_e4m3fn},
        ):
            with self.subTest(overrides=overrides):
                self.assertIsNone(build_metadata(batch, self.model_runner(**overrides)))
        self.two_batch_overlap = True
        self.assertIsNone(build_metadata(batch, self.model_runner()))

    def test_returns_none_for_batches_without_a_shared_context(self):
        runner = self.model_runner()
        cases = {
            "no beam tail": {"beam_tail": None},
            "prefill batch": {"forward_mode": ForwardMode.EXTEND},
            "no cpu seq lens": {"seq_lens_cpu": None},
            "no entries": {"beam_tail": SimpleNamespace(entries=[], num_base_rows=0)},
            "width one": {
                "beam_tail": SimpleNamespace(
                    entries=[self.entry(0, 0, 0)], num_base_rows=1
                )
            },
            "row count mismatch": {
                "beam_tail": SimpleNamespace(
                    entries=[self.entry(0, 0, 1)], num_base_rows=3
                )
            },
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                self.assertIsNone(build_metadata(self.batch(**overrides), runner))

    def test_returns_none_when_a_group_has_not_decoded_yet(self):
        # The leader's sequence still equals its prompt: no decode tokens.
        batch = self.batch(seq_lens=[1, 2, 2, 3, 2, 2, 2])
        self.assertIsNone(build_metadata(batch, self.model_runner()))


if __name__ == "__main__":
    unittest.main()
