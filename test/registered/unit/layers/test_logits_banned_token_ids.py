"""--sampling-banned-token-ids: LogitsProcessor picks the logits columns of the
banned ids (full vocabulary or a draft hot-token map) and _get_logits masks
them on both of its return paths."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOWEST = torch.finfo(torch.float32).min


def _processor(vocab_size):
    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.vocab_size = vocab_size
    processor.banned_token_cols = None
    return processor


class TestBannedTokenColumns(CustomTestCase):
    def test_full_vocabulary_columns(self):
        processor = _processor(vocab_size=8)
        processor.set_banned_token_ids(token_ids=[5, 2, 5, 9], device="cpu")
        self.assertEqual(processor.banned_token_cols.tolist(), [2, 5])

        logits = torch.zeros(3, 8)
        processor._mask_banned_tokens(logits)
        self.assertTrue((logits[:, [2, 5]] == _LOWEST).all())
        self.assertTrue((logits[:, [0, 1, 3, 4, 6, 7]] == 0).all())
        self.assertTrue(torch.isfinite(torch.log_softmax(logits, dim=-1)).all())

    def test_hot_token_map_columns(self):
        processor = _processor(vocab_size=4)
        hot_token_id = torch.tensor([7, 2, 30, 11])
        processor.set_banned_token_ids(
            token_ids=[11, 30, 40], device="cpu", column_token_ids=hot_token_id
        )
        self.assertEqual(processor.banned_token_cols.tolist(), [2, 3])

    def test_no_banned_column_leaves_logits_unchanged(self):
        processor = _processor(vocab_size=4)
        processor.set_banned_token_ids(token_ids=[100], device="cpu")
        self.assertIsNone(processor.banned_token_cols)
        logits = torch.randn(2, 4)
        expected = logits.clone()
        processor._mask_banned_tokens(logits)
        self.assertTrue(torch.equal(logits, expected))


class TestGetLogitsMasksBannedTokens(CustomTestCase):
    """Both return paths of _get_logits (the fused gather + softcap path and
    the regular path) apply the mask."""

    def _get_logits(self, fused):
        processor = _processor(vocab_size=6)
        processor.set_banned_token_ids(token_ids=[1, 4], device="cpu")
        raw = torch.zeros(2, 6)
        processor.logit_scale = None
        processor.final_logit_softcapping = None
        processor.defer_verify_gather = False
        processor.do_tensor_parallel_all_gather = False
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor._gather_dp_attn_hidden_states = lambda h, m: (h, h)
        processor._compute_lm_head = lambda h, lm_head, bias, out=None: raw.clone()
        processor._can_gather_softcap_fused = lambda *args: fused
        processor._gather_softcap_fused = lambda logits, meta, use_buffer: logits
        processor._scatter_dp_attn_logits = lambda logits, local, meta: logits
        processor._copy_logits_to_buffer = lambda logits, meta, **kwargs: logits
        metadata = SimpleNamespace(next_token_logits_buffer=None)
        forward = SimpleNamespace(scoped=lambda **kwargs: nullcontext())
        with patch(
            "sglang.srt.layers.logits_processor.get_forward", return_value=forward
        ):
            return processor._get_logits(
                hidden_states=torch.zeros(2, 3), lm_head=None, logits_metadata=metadata
            )

    def test_regular_path(self):
        logits = self._get_logits(fused=False)
        self.assertTrue((logits[:, [1, 4]] == _LOWEST).all())
        self.assertTrue((logits[:, [0, 2, 3, 5]] == 0).all())

    def test_fused_gather_path(self):
        logits = self._get_logits(fused=True)
        self.assertTrue((logits[:, [1, 4]] == _LOWEST).all())
        self.assertTrue((logits[:, [0, 2, 3, 5]] == 0).all())


if __name__ == "__main__":
    unittest.main()
