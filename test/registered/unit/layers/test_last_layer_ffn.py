import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.last_layer_ffn import eligible_batch, select_attention_rows
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        is_prefill_only=False,
        spec_info=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        lora_ids=[None, None],
        mm_inputs=None,
        extend_seq_lens_cpu=[3, 5],
        extend_prefix_lens_cpu=[0, 0],
        extend_is_final_cpu=[True, True],
        batch_size=2,
        input_ids=torch.arange(8),
        extend_seq_lens=torch.tensor([3, 5]),
        residual_stream=None,
    )


class LastLayerFFNTests(CustomTestCase):
    def test_eligibility(self):
        self.assertTrue(eligible_batch(batch()))
        cases = {
            "forward_mode": ForwardMode.DECODE,
            "tbo_split_seq_index": 0,
            "tbo_parent_token_range": (0, 8),
            "tbo_children": [],
            "return_logprob": True,
            "is_prefill_only": True,
            "spec_info": object(),
            "capture_hidden_mode": CaptureHiddenMode.FULL,
            "return_hidden_states_before_norm": True,
            "token_indices_to_pool": [1],
            "multi_item_delimiter_indices": [1],
            "lora_ids": ["adapter", None],
            "mm_inputs": [object()],
            "extend_is_final_cpu": [True, False],
            "extend_prefix_lens_cpu": [1, 0],
            "extend_seq_lens_cpu": [0, 8],
            "batch_size": 3,
            "input_ids": torch.arange(9),
            "extend_seq_lens": None,
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                b = batch()
                setattr(b, name, value)
                self.assertFalse(eligible_batch(b))

    def test_empty_and_single_token_requests_keep_original_path(self):
        for lens in ([], [1, 1]):
            b = batch()
            b.batch_size = len(lens)
            b.extend_seq_lens_cpu = lens
            b.extend_prefix_lens_cpu = [0] * len(lens)
            b.extend_is_final_cpu = [True] * len(lens)
            b.input_ids = torch.zeros(sum(lens), dtype=torch.long)
            self.assertFalse(eligible_batch(b))

    def test_tokenwise_ffn_selection_is_mathematically_equivalent(self):
        # Exact arithmetic with small integer-valued doubles isolates the
        # dependency property from reduced-precision GEMM shape effects.
        x = torch.arange(32, dtype=torch.float64).reshape(8, 4)
        residual = x + 1
        up = torch.arange(24, dtype=torch.float64).reshape(4, 6)
        down = torch.arange(24, dtype=torch.float64).reshape(6, 4)
        indices = torch.tensor([2, 7])
        full = torch.relu((x + residual) @ up) @ down
        b = batch()
        hidden, b.residual_stream = ResidualStream.from_handoff(x, residual, PLAIN_ADD)
        selected = select_attention_rows(hidden, b, indices)
        h, res = residual_batch.stream_of(b).export(selected)
        compact = torch.relu((h + res) @ up) @ down
        torch.testing.assert_close(compact, full[indices], rtol=0, atol=0)

    def test_operands_selected_without_adding(self):
        b = batch()
        x = torch.arange(32.0).reshape(8, 4)
        r = x + 100
        hidden, b.residual_stream = ResidualStream.from_handoff(x, r, PLAIN_ADD)
        idx = torch.tensor([2, 7])
        selected = select_attention_rows(hidden, b, idx)
        h, res = residual_batch.stream_of(b).export(selected)
        torch.testing.assert_close(h, x[idx])
        torch.testing.assert_close(res, r[idx])
        self.assertIsNot(h, x)
        self.assertIsNot(res, r)
        torch.testing.assert_close(r, x + 100)

    def test_no_double_indexing(self):
        x = torch.randn(2, 4)
        meta = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_return_logprob=False,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            extend_seq_lens_cpu=[3, 5],
        )
        out = LogitsProcessor._get_pruned_states(
            None, x, None, None, meta, already_pruned=True
        )
        self.assertIs(out[0], x)
        self.assertEqual(out[1:], (None, None, None, None, []))
        meta.extend_return_logprob = True
        with self.assertRaises(AssertionError):
            LogitsProcessor._get_pruned_states(
                None, x, None, None, meta, already_pruned=True
            )


if __name__ == "__main__":
    unittest.main()
