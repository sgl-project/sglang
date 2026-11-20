"""Qwen4 PLE verify padding must not overwrite live request state."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4PLEVerifyPadding(CustomTestCase):
    @contextmanager
    def _case(self, pp_spec, lengths, padding_slot=-1, with_cache_loc=True):
        width, pool_size, channels, history = 3, 6, 2, 2
        conv = ShortConvPool(
            size=pool_size,
            state_shape=(channels, history),
            layer_ids=[0],
            dtype=torch.float32,
            device="cpu",
            spec_state_size=pool_size,
            speculative_num_draft_tokens=width,
        )
        ngram = NGramPool(
            size=pool_size,
            context_len=history,
            eos_token_id=17,
            device="cpu",
            spec_state_size=pool_size,
            speculative_num_draft_tokens=width,
        )
        state = conv.layer_cache(0)
        state.copy_(torch.arange(state.numel()).reshape_as(state))
        conv.layer_intermediate_cache(0).fill_(-99)
        ngram.context.copy_(
            torch.arange(ngram.context.numel()).reshape_as(ngram.context)
        )
        ngram.intermediate_context.fill_(-99)
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=3,
            input_ids=torch.arange(1, 10),
            req_pool_indices=torch.tensor([4, pool_size, 0]),
            seq_lens=torch.tensor([9, 9, 0]),
            seq_lens_sum=18,
            out_cache_loc=torch.tensor([1, 2, 3, 4, 5, 6, 0, 0, 0])
            if with_cache_loc
            else None,
            extend_seq_lens=None if lengths is None else torch.tensor(lengths),
            spec_info=SimpleNamespace(draft_token_num=width, topk=1),
        )
        router = SimpleNamespace(
            read=lambda *_: SimpleNamespace(view=conv, view_layer_id=0),
            part_view=lambda _: ngram,
            wait_for_part=lambda _: None,
        )
        backend = SimpleNamespace(
            forward_state_slots=lambda *_: SimpleNamespace(
                working=torch.tensor([3, 5, padding_slot])
            )
        )
        with (
            envs.SGLANG_ENABLE_PP_SPEC.override(pp_spec),
            patch.object(
                qwen4_exp,
                "get_req_to_token_pool",
                return_value=SimpleNamespace(size=pool_size),
            ),
            patch.object(
                qwen4_exp,
                "get_token_to_kv_pool",
                return_value=SimpleNamespace(layer_router=router),
            ),
            patch.object(qwen4_exp, "get_attn_backend", return_value=backend),
        ):
            batch = qwen4_exp._prepare_ple_batch(
                fb.input_ids, fb, ngram_size=history + 1, ngram_eos_token_id=17
            )
            yield SimpleNamespace(fb=fb, batch=batch, conv=conv, ngram=ngram)

    def test_verify_padding_masks_tokens_without_length_metadata(self):
        for padding_slot, with_cache_loc in ((-1, True), (5, True), (-1, False)):
            with (
                self.subTest(padding_slot=padding_slot, with_cache_loc=with_cache_loc),
                self._case(True, None, padding_slot, with_cache_loc) as case,
            ):
                self.assertEqual(case.batch.lengths.tolist(), [3, 3, 3])
                self.assertEqual(case.batch.state_indices.tolist(), [3, 5, 0])
                self.assertEqual(
                    case.batch.valid_tokens.tolist(), [True] * 6 + [False] * 3
                )
                self.assertEqual(case.batch.ngram_context[-1, -3:].tolist(), [17] * 3)

    def test_decode_padding_keeps_dummy_state_in_both_paths(self):
        for fast in (False, True):
            for padding_slot in (-1, 5):
                with (
                    self.subTest(fast=fast, padding_slot=padding_slot),
                    self._case(False, None, padding_slot) as case,
                    envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.override(fast),
                ):
                    case.fb.forward_mode = ForwardMode.DECODE
                    case.fb.input_ids = torch.tensor([101, 102, 103])
                    case.fb.out_cache_loc = torch.tensor([1, 2, 0])
                    case.fb.spec_info = None
                    batch = qwen4_exp._prepare_ple_batch(
                        case.fb.input_ids,
                        case.fb,
                        ngram_size=3,
                        ngram_eos_token_id=17,
                    )
                    self.assertEqual(batch.state_indices.tolist(), [3, 5, 0])
                    self.assertEqual(batch.valid_tokens.tolist(), [True, True, False])
                    torch.testing.assert_close(
                        batch.ngram_context[:2],
                        torch.cat(
                            (case.ngram.context[[3, 5]], case.fb.input_ids[:2, None]),
                            dim=1,
                        ),
                    )

    def test_unfused_verify_preserves_maximum_request_row(self):
        for pp_spec in (False, True):
            for lengths in (None, [3, 3, 0], [2, 1, 0]):
                with (
                    self.subTest(pp_spec=pp_spec, lengths=lengths),
                    self._case(pp_spec, lengths) as case,
                ):
                    batch = case.batch
                    state = case.conv.layer_cache(0)
                    before = state.clone()
                    x = torch.arange(1, 19, dtype=torch.float32).reshape(9, 2)
                    layer = SimpleNamespace(
                        layer_id=0,
                        conv_channels=2,
                        short_conv_state_len=2,
                        short_conv_dilation=1,
                        conv1d=SimpleNamespace(weight=torch.ones(2, 1, 3)),
                    )
                    qwen4_exp.Qwen4ExpPLELayer._short_conv(layer, x, case.fb, batch)
                    full = torch.cat(
                        (before[[3, 5]], x[:6].reshape(2, 3, 2).transpose(1, 2)), dim=2
                    )
                    expected = full.unfold(2, 2, 1)[:, :, 1:4].permute(0, 2, 1, 3)
                    counts = [3, 3] if lengths is None else lengths[:2]
                    valid = torch.arange(3)[None, :] < torch.tensor(counts)[:, None]
                    expected = torch.where(valid[:, :, None, None], expected, 0)
                    rows = [4, 6] if pp_spec else [0, 1]
                    torch.testing.assert_close(
                        case.conv.layer_intermediate_cache(0)[rows], expected
                    )
                    scratch = case.conv.layer_intermediate_cache(0)
                    self.assertTrue(scratch[0 if pp_spec else 2].eq(0).all())
                    untouched = [1, 2, 3, 5] if pp_spec else [3, 4, 5, 6]
                    self.assertTrue(scratch[untouched].eq(-99).all())
                    torch.testing.assert_close(state, before)

    def test_ngram_verify_preserves_maximum_request_row(self):
        for pp_spec in (False, True):
            for lengths in (None, [3, 3, 0], [2, 1, 0]):
                with (
                    self.subTest(pp_spec=pp_spec, lengths=lengths),
                    self._case(pp_spec, lengths) as case,
                ):
                    before = case.ngram.context.clone()
                    qwen4_exp._commit_ple_batch(case.batch, case.fb)
                    contexts = torch.cat(
                        (before[[3, 5]], case.fb.input_ids[:6].reshape(2, 3)), dim=1
                    )
                    expected = contexts.unfold(1, 2, 1)[:, 1:]
                    counts = [3, 3] if lengths is None else lengths[:2]
                    valid = torch.arange(3)[None, :] < torch.tensor(counts)[:, None]
                    expected = torch.where(valid[:, :, None], expected, 17)
                    rows = [4, 6] if pp_spec else [0, 1]
                    torch.testing.assert_close(
                        case.ngram.intermediate_context[rows], expected
                    )
                    scratch = case.ngram.intermediate_context
                    self.assertTrue(scratch[0 if pp_spec else 2].eq(17).all())
                    untouched = [1, 2, 3, 5] if pp_spec else [3, 4, 5, 6]
                    self.assertTrue(scratch[untouched].eq(-99).all())
                    torch.testing.assert_close(case.ngram.context, before)


if __name__ == "__main__":
    unittest.main()
