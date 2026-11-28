import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.serving_hook import handle_watermark_config
from sglang.srt.arg_groups.validation_hook import check_watermark_server_args
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.sampling.verify_accept import VerifyAccept
from sglang.srt.sampling.watermarking.config import load_watermark_config
from sglang.srt.sampling.watermarking.redaction import redact_watermark_secrets
from sglang.srt.sampling.watermarking.synthid import adapter
from sglang.srt.sampling.watermarking.synthid.algorithm import SynthIDConfig
from sglang.srt.sampling.watermarking.synthid.watermark import SynthIDWatermark
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.synthid_utils import assert_synthid_state_equal, synthid_state
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_KEYS = [11, 22, 33]


def _resolve(watermark_config, **kwargs):
    args = ServerArgs(
        model_path="dummy",
        device="cuda",
        enable_watermark=True,
        watermark_algorithm=kwargs.pop("watermark_algorithm", "synthid"),
        watermark_config=json.dumps(watermark_config),
        **kwargs,
    )
    with patch.object(adapter, "is_cuda", return_value=True):
        handle_watermark_config(args)
        check_watermark_server_args(args)
    return resolving_view(args)


def _features(**kwargs):
    return SimpleNamespace(
        **{
            "enable_watermark": True,
            "watermark_default_enabled": False,
            "watermark_enforce_all": False,
            **kwargs,
        }
    )


def _request(watermark=None, top_k=TOP_K_ALL):
    return SimpleNamespace(
        sampling_params=SamplingParams(watermark=watermark, top_k=top_k)
    )


def _info(batch_size, *, top_ks=None, top_ps=None, min_ps=None, watermark=None):
    top_ks = torch.full((batch_size,), TOP_K_ALL) if top_ks is None else top_ks
    top_ps = torch.ones(batch_size) if top_ps is None else top_ps
    min_ps = torch.zeros(batch_size) if min_ps is None else min_ps
    return SamplingBatchInfo(
        temperatures=torch.linspace(0.6, 1.4, batch_size).view(-1, 1),
        top_ps=top_ps,
        top_ks=top_ks.to(torch.int32),
        min_ps=min_ps,
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=bool((top_ps < 1).any()),
        need_top_k_sampling=bool((top_ks != TOP_K_ALL).any()),
        need_min_p_sampling=bool((min_ps > 0).any()),
        vocab_size=16,
        watermark=watermark,
    )


def _watermarks(num_slots=8):
    config = SynthIDConfig(
        keys=(1, 2, 3), ngram_len=3, num_candidates=6, context_history_size=4
    )
    watermarks = []
    for _ in range(2):
        watermark = SynthIDWatermark(config, num_slots, "cpu")
        watermark.state.seed(torch.arange(num_slots), [[] for _ in range(num_slots)])
        watermark.state.commit(torch.arange(num_slots), torch.arange(num_slots) + 3)
        watermarks.append(watermark)
    return watermarks


def _top_k_top_p(top_ks, top_ps):
    """Each row's request top-k, then top-p, over its candidates, written out:
    walk the candidates by descending probability and keep one while fewer than
    top-k are kept and the mass kept before it is at most top-p."""

    def filter_probs(probs):
        kept = torch.zeros_like(probs)
        for row, (top_k, top_p) in enumerate(zip(top_ks, top_ps, strict=True)):
            mass = 0.0
            for rank, index in enumerate(probs[row].argsort(descending=True)):
                if rank >= top_k or mass > top_p:
                    break
                kept[row, index] = probs[row, index]
                mass += probs[row, index].item()
        return kept / kept.sum(dim=-1, keepdim=True)

    return filter_probs


def _rows(candidates):
    """The rows build_rows makes from the pool's gathered candidate field."""
    return adapter.build_rows(torch.tensor([[int(c) for c in candidates]]), candidates)


class TestSynthIDConfig(CustomTestCase):
    def test_file_fields_resolve_and_keys_are_redacted(self):
        cfg = _resolve({"keys": _KEYS, "ngram_len": 3, "num_candidates": 40})
        self.assertEqual(cfg.watermark_keys, _KEYS)
        config = adapter._synthid_config(cfg)
        self.assertEqual(config.keys, tuple(_KEYS))
        self.assertEqual((config.ngram_len, config.num_candidates), (3, 40))

        loaded = load_watermark_config(json.dumps({"keys": _KEYS}), algorithm="synthid")
        self.assertNotIn("11", repr(loaded))
        self.assertEqual(redact_watermark_secrets(loaded).keys, "<redacted>")
        self.assertEqual(
            redact_watermark_secrets({"watermark_keys": _KEYS}),
            {"watermark_keys": "<redacted>"},
        )

    def test_server_checks(self):
        eagle = {
            "speculative_algorithm": "EAGLE",
            "speculative_draft_model_path": "dummy",
            "speculative_eagle_topk": 1,
            "speculative_num_steps": 3,
            "speculative_num_draft_tokens": 4,
            "speculative_use_rejection_sampling": True,
        }
        _resolve({"keys": _KEYS, "default_enabled": True}, **eagle)
        refused = {
            "requires --speculative-use-rejection-sampling": (
                {"keys": _KEYS},
                {**eagle, "speculative_use_rejection_sampling": False},
            ),
            r"\['key'\] are not supported with --watermark-algorithm synthid": (
                {"keys": _KEYS, "key": "abcd"},
                {},
            ),
            # Passed as server arguments instead of through the file.
            r"\['watermark_keys'\] are not supported with --watermark-algorithm gumbel": (
                {"key": "abcd"},
                {"watermark_algorithm": "gumbel", "watermark_keys": _KEYS},
            ),
        }
        for message, (config, kwargs) in refused.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _resolve(config, **kwargs)


class TestSynthIDRequestPolicy(CustomTestCase):
    def test_request_policy(self):
        features = _features()
        self.assertFalse(adapter.request_enabled(SamplingParams(), features=features))
        self.assertTrue(
            adapter.request_enabled(
                SamplingParams(watermark={"enabled": True}), features=features
            )
        )
        default_on = _features(watermark_default_enabled=True)
        self.assertTrue(adapter.request_enabled(SamplingParams(), features=default_on))
        self.assertFalse(
            adapter.request_enabled(
                SamplingParams(watermark={"enabled": False}), features=default_on
            )
        )
        with self.assertRaisesRegex(ValueError, "enforce_all"):
            adapter.request_enabled(
                SamplingParams(watermark={"enabled": False}),
                features=_features(watermark_enforce_all=True),
            )

    def test_rows_hold_enabled_sampling_rows(self):
        on, off = {"enabled": True}, {"enabled": False}
        requests = [_request(on), _request(off), _request(on, top_k=1), _request(on)]
        candidates, values = zip(
            *(adapter.pool_values(r, features=_features()) for r in requests)
        )
        self.assertEqual(candidates, (True, False, False, True))
        rows = adapter.build_rows(torch.tensor(values).T, list(candidates))
        self.assertEqual(rows.indices.tolist(), [0, 3])
        self.assertEqual(rows.candidates_host, [True, False, False, True])
        self.assertFalse(rows.all_on)


class TestSynthIDEngine(CustomTestCase):
    def test_seed_payload_covers_prefill_rows_only(self):
        engine = adapter.SynthIDEngine(_watermarks()[0], allow_fused_sample=False)
        rows = _rows([True, False, True, True])
        reqs = [SimpleNamespace(output_ids=ids) for ids in ([], [5], [7, 8], [9])]

        def batch(mode, *, decoding_reqs=None, num_mixed=0):
            return SimpleNamespace(
                sampling_info=SimpleNamespace(watermark=rows),
                forward_mode=mode,
                reqs=reqs,
                decoding_reqs=decoding_reqs,
                mix_running_indices_cpu=torch.zeros(num_mixed),
            )

        self.assertEqual(
            engine.seed_payload(batch(ForwardMode.EXTEND)),
            ([0, 2, 3], [[], [7, 8], [9]]),
        )
        # The trailing rows of a mixed batch decode and keep their state.
        self.assertEqual(
            engine.seed_payload(batch(ForwardMode.MIXED, num_mixed=2)), ([0], [[]])
        )
        self.assertIsNone(
            engine.seed_payload(batch(ForwardMode.EXTEND, decoding_reqs=reqs))
        )

    def test_fused_sample_is_skipped_for_a_custom_logit_processor(self):
        engine = adapter.SynthIDEngine(_watermarks()[0], allow_fused_sample=True)
        rows = _rows([True, True])
        info = _info(2, watermark=rows)
        forward_batch = SimpleNamespace(
            sampling_info=info, req_pool_indices=torch.tensor([0, 1])
        )
        sample = engine.bind_sample(forward_batch)

        self.assertTrue(sample._can_fuse(keep_logits=False))
        info.has_custom_logit_processor = True
        self.assertFalse(sample._can_fuse(keep_logits=False))

    def test_sample_watermarks_on_rows_and_commits_their_slots(self):
        watermark, twin = _watermarks()
        engine = adapter.SynthIDEngine(watermark, allow_fused_sample=False)
        rows = _rows([True, False, True, False])
        info = _info(
            4,
            top_ks=torch.tensor([3, 2, TOP_K_ALL, TOP_K_ALL]),
            top_ps=torch.tensor([0.9, 0.5, 1.0, 1.0]),
            watermark=rows,
        )
        req_pool_indices = torch.tensor([6, 1, 4, 2])
        forward_batch = SimpleNamespace(
            sampling_info=info,
            req_pool_indices=req_pool_indices,
            watermark_seed=([2], [[3, 5]]),
        )
        torch.manual_seed(0)
        logits = torch.randn(4, 16)
        original = logits.clone()

        sample = engine.bind_sample(forward_batch)
        step = sample.apply(logits, keep_logits=True)

        torch.testing.assert_close(step.logprob_logits, original, rtol=0, atol=0)
        torch.testing.assert_close(logits[[1, 3]], original[[1, 3]], rtol=0, atol=0)
        expected = original[[0, 2]].clone()
        twin.state.seed(torch.tensor([4]), [[3, 5]])
        twin.apply(
            expected,
            info.temperatures[[0, 2]],
            torch.tensor([6, 4]),
            filter_probs=_top_k_top_p([3, TOP_K_ALL], [0.9, 1.0]),
        )
        torch.testing.assert_close(logits[[0, 2]], expected, rtol=0, atol=0)
        # Row 0's request top-k ran before the tournament.
        self.assertEqual(int(torch.isfinite(logits[0]).sum()), 3)

        self.assertEqual(step.sampling_info.top_ps.tolist(), [1.0, 0.5, 1.0, 1.0])
        self.assertTrue(step.sampling_info.need_top_p_sampling)
        self.assertAlmostEqual(info.top_ps[0].item(), 0.9, places=6)

        sample.commit(torch.tensor([7, 8, 9, 10]))
        twin.state.commit(torch.tensor([6, 4]), torch.tensor([7, 9]))
        assert_synthid_state_equal(
            synthid_state(watermark.state), synthid_state(twin.state)
        )

    def test_verify_watermarks_on_rows_and_appends_their_tokens(self):
        watermark, twin = _watermarks()
        engine = adapter.SynthIDEngine(watermark, allow_fused_sample=False)
        rows = _rows([False, True, True])
        info = _info(3, top_ks=torch.tensor([TOP_K_ALL, 4, TOP_K_ALL]))
        req_pool_indices = torch.tensor([5, 3, 7])
        draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        torch.manual_seed(1)
        logits = torch.randn(9, 16)
        original = logits.clone()

        verify = engine.bind_verify(rows, full_mask=True)
        verified, _ = verify.apply(
            logits,
            sampling_info=info,
            req_pool_indices=req_pool_indices,
            draft_tokens=draft_tokens.view(-1),
            positions=torch.zeros(9, dtype=torch.int64),
            draft_token_num=3,
            custom_mask=torch.ones(27, dtype=torch.bool),
            keep_logits=True,
        )

        torch.testing.assert_close(logits, original, rtol=0, atol=0)
        torch.testing.assert_close(verified[:3], original[:3], rtol=0, atol=0)
        expected = original[3:].clone()
        twin.apply_verify(
            expected,
            info.temperatures[1:],
            torch.tensor([3, 7]),
            draft_tokens[1:],
            filter_probs=_top_k_top_p([4] * 3 + [TOP_K_ALL] * 3, [1.0] * 6),
        )
        torch.testing.assert_close(verified[3:], expected, rtol=0, atol=0)
        self.assertLessEqual(int(torch.isfinite(verified[3:6]).sum(dim=1).max()), 4)

        accept_tokens = torch.tensor([[1, 0, 0], [4, 5, 0], [10, 11, 12]])
        accept_lens = torch.tensor([1, 2, 3], dtype=torch.int32)
        verify.commit(VerifyAccept(tokens=accept_tokens, accept_lens=accept_lens))
        twin.state.commit(torch.tensor([3, 7]), accept_tokens[1:], accept_lens[1:])
        assert_synthid_state_equal(
            synthid_state(watermark.state), synthid_state(twin.state)
        )


if __name__ == "__main__":
    unittest.main()
