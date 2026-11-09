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


class TestBannedTokensKeepTheVerifyGather(CustomTestCase):
    """Vocab-shard verify logits are never masked, so a ban turns them off,
    whichever of the ban and the speculative opt-in comes first."""

    def _eligible(self):
        processor = _processor(vocab_size=8)
        processor._sharded_verify_eligible = True
        processor.sharded_verify = False
        return processor

    def test_ban_then_opt_in(self):
        processor = self._eligible()
        processor.set_banned_token_ids(token_ids=[3], device="cpu")
        processor.enable_sharded_verify()
        self.assertFalse(processor.sharded_verify)

    def test_opt_in_then_ban(self):
        processor = self._eligible()
        processor.enable_sharded_verify()
        self.assertTrue(processor.sharded_verify)
        processor.set_banned_token_ids(token_ids=[3], device="cpu")
        self.assertFalse(processor.sharded_verify)

    def test_ban_outside_the_vocabulary_keeps_sharded_verify(self):
        processor = self._eligible()
        processor.set_banned_token_ids(token_ids=[100], device="cpu")
        processor.enable_sharded_verify()
        self.assertTrue(processor.sharded_verify)


class TestBannedTokenIdsRejectedOnMlx(CustomTestCase):
    def _validate(self, token_ids, mlx):
        from sglang.srt.arg_groups import validation_hook

        with (
            patch.object(validation_hook, "resolving_view", side_effect=lambda x: x),
            patch.dict("os.environ", {"SGLANG_USE_MLX": "1" if mlx else "0"}),
        ):
            validation_hook.validate_sampling_banned_token_ids(
                SimpleNamespace(sampling_banned_token_ids=token_ids)
            )

    def test_mlx_rejects_the_flag(self):
        with self.assertRaisesRegex(ValueError, "not supported on MLX"):
            self._validate([5], mlx=True)

    def test_mlx_without_the_flag_and_cuda_with_it_pass(self):
        self._validate(None, mlx=True)
        self._validate([5], mlx=False)


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


class TestModelRunnerSetsBannedTokenIds(CustomTestCase):
    """A target model without a LogitsProcessor on the last pipeline stage
    refuses to start instead of ignoring the flag."""

    def _set_banned_token_ids(self, model, is_last_rank):
        from sglang.srt.model_executor.model_runner import ModelRunner

        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        runner.is_draft_worker = False
        runner.device = "cpu"
        runner.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        exec_config = SimpleNamespace(
            features=SimpleNamespace(sampling_banned_token_ids=[1])
        )
        with patch(
            "sglang.srt.model_executor.model_runner.get_exec",
            return_value=exec_config,
        ):
            runner.maybe_set_banned_token_ids()

    def test_logits_processor_is_masked(self):
        model = torch.nn.Sequential(_processor(vocab_size=4))
        self._set_banned_token_ids(model=model, is_last_rank=True)
        self.assertEqual(model[0].banned_token_cols.tolist(), [1])

    def test_model_without_logits_processor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no LogitsProcessor"):
            self._set_banned_token_ids(model=torch.nn.Linear(2, 2), is_last_rank=True)

    def test_earlier_pipeline_stage_without_logits_processor_starts(self):
        self._set_banned_token_ids(model=torch.nn.Linear(2, 2), is_last_rank=False)


if __name__ == "__main__":
    unittest.main()
