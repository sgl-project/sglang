"""SynthID adapter: config, request policy, server checks, rows, and the bound
sample and verify steps on CPU."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.serving_hook import handle_watermark_config
from sglang.srt.arg_groups.validation_hook import check_watermark_server_args
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.srt.sampling.watermarking.config import load_watermark_config
from sglang.srt.sampling.watermarking.interface import merge_watermark_rows
from sglang.srt.sampling.watermarking.redaction import redact_watermark_secrets
from sglang.srt.sampling.watermarking.synthid import adapter
from sglang.srt.sampling.watermarking.synthid.adapter import SynthIDRows
from sglang.srt.sampling.watermarking.synthid.algorithm import SynthIDConfig
from sglang.srt.sampling.watermarking.synthid.watermark import SynthIDWatermark
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.synthid_utils import STATE_FIELDS, synthid_state
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
    """Two watermarks with the same keys and the same state, held separately."""
    config = SynthIDConfig(
        keys=(1, 2, 3), ngram_len=3, num_candidates=6, context_history_size=4
    )
    watermarks = []
    for _ in range(2):
        watermark = SynthIDWatermark(config, num_slots, "cpu")
        watermark.seed(torch.arange(num_slots), [[] for _ in range(num_slots)])
        watermark.commit(torch.arange(num_slots), torch.arange(num_slots) + 3)
        watermarks.append(watermark)
    return watermarks


def _assert_same_state(test, actual, expected):
    actual, expected = synthid_state(actual), synthid_state(expected)
    for name in STATE_FIELDS:
        test.assertTrue(torch.equal(actual[name], expected[name]), name)


class TestSynthIDConfig(CustomTestCase):
    def test_file_fields_resolve_and_keys_are_redacted(self):
        cfg = _resolve({"keys": _KEYS, "ngram_len": 3, "num_candidates": 40})
        self.assertEqual(cfg.watermark_keys, _KEYS)
        config = adapter._synthid_config(cfg)
        self.assertEqual(config.keys, tuple(_KEYS))
        self.assertEqual((config.ngram_len, config.num_candidates), (3, 40))
        # Unset fields keep the lib defaults.
        self.assertEqual(config.context_history_size, 1024)
        self.assertFalse(config.skip_first_ngram_calls)

        loaded = load_watermark_config(json.dumps({"keys": _KEYS}), algorithm="synthid")
        self.assertNotIn("11", repr(loaded))
        self.assertIn("keys=<redacted>", repr(loaded))
        unset = load_watermark_config("{}", algorithm="synthid")
        self.assertIn("keys=None", repr(unset))
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
        drafts = {
            "EAGLE": {},
            "EAGLE3": {"speculative_algorithm": "EAGLE3"},
            "--speculative-token-map": {"speculative_token_map": "map.pt"},
            "--enable-multi-layer-eagle": {"enable_multi_layer_eagle": True},
        }
        _resolve({"keys": _KEYS})
        # A null field is unset, whichever algorithm reads it.
        _resolve({"key": "abcd", "ngram_len": None}, watermark_algorithm="gumbel")
        for name, draft in drafts.items():
            spec = {**eagle, **draft}
            with self.subTest(accepted=name):
                _resolve({"keys": _KEYS, "default_enabled": True}, **spec)
            with self.subTest(without_rejection_sampling=name):
                with self.assertRaisesRegex(
                    ValueError, "requires --speculative-use-rejection-sampling"
                ):
                    _resolve(
                        {"keys": _KEYS},
                        **{**spec, "speculative_use_rejection_sampling": False},
                    )
        refused = {
            "requires watermark config keys": ({}, {}),
            "1 to 64": ({"keys": list(range(65))}, {}),
            "distinct": ({"keys": [7, 7]}, {}),
            "ngram_len must be from 2 to 17": ({"keys": _KEYS, "ngram_len": 18}, {}),
            "num_candidates must be from 2": ({"keys": _KEYS, "num_candidates": 1}, {}),
            "context_history_size must be from 1 to 2048": (
                {"keys": _KEYS, "context_history_size": 2049},
                {},
            ),
            r"\['key'\] are not supported with --watermark-algorithm synthid": (
                {"keys": _KEYS, "key": "abcd"},
                {},
            ),
            # Restates the gumbel default: rejected all the same.
            r"\['context_window'\] are not supported": (
                {"keys": _KEYS, "context_window": 4},
                {},
            ),
            r"\['keys'\] are not supported with --watermark-algorithm gumbel": (
                {"key": "abcd", "keys": _KEYS},
                {"watermark_algorithm": "gumbel"},
            ),
            # Passed as server arguments instead of through the file.
            r"\['watermark_keys'\] are not supported with --watermark-algorithm gumbel": (
                {"key": "abcd"},
                {"watermark_algorithm": "gumbel", "watermark_keys": _KEYS},
            ),
            r"\['watermark_context_window'\] are not supported": (
                {"keys": _KEYS},
                {"watermark_context_window": 8},
            ),
            "unknown --watermark-algorithm 'nope'": (
                {"keys": _KEYS},
                {"watermark_algorithm": "nope"},
            ),
            "only with EAGLE or EAGLE3": (
                {"keys": _KEYS},
                {**eagle, "speculative_algorithm": "NGRAM"},
            ),
            "at most 8 --speculative-num-draft-tokens": (
                {"keys": _KEYS},
                {
                    **eagle,
                    "speculative_num_steps": 8,
                    "speculative_num_draft_tokens": 9,
                },
            ),
            "--rl-on-policy-target": (
                {"keys": _KEYS},
                {"rl_on_policy_target": "fsdp"},
            ),
        }
        for message, (config, kwargs) in refused.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _resolve(config, **kwargs)
        with self.subTest(message="NV mega runtime"):
            with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
                with self.assertRaisesRegex(ValueError, "NV_MEGA_RUNTIME"):
                    _resolve({"keys": _KEYS})
        with self.subTest(message="CUDA"):
            args = ServerArgs(
                model_path="dummy",
                device="cuda",
                enable_watermark=True,
                watermark_algorithm="synthid",
                watermark_config=json.dumps({"keys": _KEYS}),
            )
            handle_watermark_config(args)
            with patch.object(adapter, "is_cuda", return_value=False):
                with self.assertRaisesRegex(ValueError, "requires NVIDIA CUDA"):
                    check_watermark_server_args(args)


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
        refused = {
            "not supported with --watermark-algorithm synthid": (
                {"key": "abcd"},
                features,
            ),
            "must set enabled": ({}, features),
            "enforce_all": (
                {"enabled": False},
                _features(watermark_enforce_all=True),
            ),
            "requires the server to enable": (
                {"enabled": True},
                _features(enable_watermark=False),
            ),
        }
        for message, (watermark, request_features) in refused.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adapter.request_enabled(
                        SamplingParams(watermark=watermark), features=request_features
                    )

    def test_rows_hold_enabled_sampling_rows(self):
        on, off = {"enabled": True}, {"enabled": False}
        requests = [_request(on), _request(off), _request(on, top_k=1), _request(on)]
        rows = adapter.build_rows(requests, features=_features(), device="cpu")
        self.assertEqual(rows.indices.tolist(), [0, 3])
        self.assertEqual(rows.candidates_host, [True, False, False, True])
        self.assertFalse(rows.all_on)
        self.assertIsNone(
            adapter.build_rows(requests[1:3], features=_features(), device="cpu")
        )

        kept = rows.filter([3, 1], torch.tensor([3, 1]))
        self.assertEqual(
            (kept.indices.tolist(), kept.candidates_host), ([0], [True, False])
        )
        self.assertIsNone(rows.filter([1, 2], torch.tensor([1, 2])))
        merged = merge_watermark_rows(None, kept, 2, 2)
        self.assertEqual(merged.indices.tolist(), [2])
        merged = merge_watermark_rows(kept, None, 2, 1)
        self.assertEqual(merged.candidates_host, [True, False, False])
        everything = rows.filter([0, 3], torch.tensor([0, 3]))
        self.assertTrue(everything.all_on)
        self.assertIsNone(everything.take_index)
        # The overlapped forward may still read the old rows.
        self.assertEqual(rows.indices.tolist(), [0, 3])


class TestSynthIDEngine(CustomTestCase):
    def test_seed_payload_covers_prefill_rows_only(self):
        engine = adapter.SynthIDEngine(_watermarks()[0], allow_fused_sample=False)
        rows = SynthIDRows.from_candidates([True, False, True, True], "cpu")
        reqs = [SimpleNamespace(output_ids=ids) for ids in ([], [5], [7, 8], [9])]

        def batch(mode, *, decoding_reqs=None, num_mixed=0):
            return SimpleNamespace(
                sampling_info=SimpleNamespace(watermark=rows),
                forward_mode=mode,
                reqs=reqs,
                decoding_reqs=decoding_reqs,
                mix_running_indices_cpu=torch.zeros(num_mixed),
            )

        # A fresh request replays nothing; a resumed one, its generated tokens.
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
        self.assertIsNone(engine.seed_payload(batch(ForwardMode.DECODE)))

    def test_sample_watermarks_on_rows_and_commits_their_slots(self):
        watermark, twin = _watermarks()
        engine = adapter.SynthIDEngine(watermark, allow_fused_sample=False)
        rows = SynthIDRows.from_candidates([True, False, True, False], "cpu")
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
        twin.seed(torch.tensor([4]), [[3, 5]])
        twin.apply(
            expected,
            info.temperatures[[0, 2]],
            torch.tensor([6, 4]),
            filter_probs=adapter._request_filter(info, rows),
        )
        torch.testing.assert_close(logits[[0, 2]], expected, rtol=0, atol=0)
        # Row 0's request top-k ran before the tournament.
        self.assertEqual(int(torch.isfinite(logits[0]).sum()), 3)

        self.assertEqual(step.sampling_info.top_ps.tolist(), [1.0, 0.5, 1.0, 1.0])
        self.assertEqual(step.sampling_info.min_ps.tolist(), [0.0] * 4)
        self.assertEqual(step.sampling_info.top_ks.tolist(), info.top_ks.tolist())
        self.assertTrue(step.sampling_info.need_top_p_sampling)
        self.assertAlmostEqual(info.top_ps[0].item(), 0.9, places=6)
        self.assertIsNone(step.token_ids)

        sample.commit(torch.tensor([7, 8, 9, 10]))
        twin.commit(torch.tensor([6, 4]), torch.tensor([7, 9]))
        _assert_same_state(self, watermark, twin)

    def test_verify_watermarks_on_rows_and_appends_their_tokens(self):
        watermark, twin = _watermarks()
        engine = adapter.SynthIDEngine(watermark, allow_fused_sample=False)
        rows = SynthIDRows.from_candidates([False, True, True], "cpu")
        info = _info(3, top_ks=torch.tensor([TOP_K_ALL, 4, TOP_K_ALL]))
        req_pool_indices = torch.tensor([5, 3, 7])
        draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        torch.manual_seed(1)
        logits = torch.randn(9, 16)
        original = logits.clone()

        verify = engine.bind_verify(rows, full_mask=True)
        verified, verify_info = verify.apply(
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
            filter_probs=adapter._request_filter(info, rows, width=3),
        )
        torch.testing.assert_close(verified[3:], expected, rtol=0, atol=0)
        self.assertLessEqual(int(torch.isfinite(verified[3:6]).sum(dim=1).max()), 4)
        self.assertEqual(verify_info.top_ps.tolist(), [1.0, 1.0, 1.0])

        accept_tokens = torch.tensor([[1, 0, 0], [4, 5, 0], [10, 11, 12]])
        accept_lens = torch.tensor([1, 2, 3], dtype=torch.int32)
        engine.append_accept_tokens(rows, req_pool_indices, accept_tokens, accept_lens)
        twin.commit(torch.tensor([3, 7]), accept_tokens[1:], accept_lens[1:])
        _assert_same_state(self, watermark, twin)


class TestSynthIDFusedGate(CustomTestCase):
    def test_fused_sample_falls_back_for_a_seed_or_logprobs(self):
        fusable = SimpleNamespace(can_fuse_sample=lambda logits: True)

        def can_fuse(*, keep_logits=False, sampling_seed=None):
            info = _info(2)
            info.sampling_seed = sampling_seed
            sample = adapter.SynthIDSample(
                fusable,
                SynthIDRows.from_candidates([True, True], "cpu"),
                SimpleNamespace(sampling_info=info),
                allow_fused_sample=True,
            )
            return sample._can_fuse(torch.zeros(2, 16), keep_logits)

        self.assertTrue(can_fuse())
        self.assertFalse(can_fuse(sampling_seed=torch.tensor([1, 2])))
        self.assertFalse(can_fuse(keep_logits=True))


if __name__ == "__main__":
    unittest.main()
