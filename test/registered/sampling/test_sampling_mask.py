import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    SamplingMaskStatus,
)
from sglang.srt.layers.sampler import Sampler, _SamplingMaskCapture
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.custom_logit_processor import (
    DisallowedTokensLogitsProcessor,
    Qwen3ThinkingBudgetLogitProcessor,
)
from sglang.srt.sampling.sampling_batch_info import (
    ProcessorEntry,
    SamplingBatchInfo,
)
from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=250, stage="base-b", runner_config="2-gpu-large")
register_amd_ci(est_time=320, suite="stage-b-test-1-gpu-small-amd")

_MAX_NEW_TOKENS = 4
_TOP_P = 0.99
_TOP_K = 10
_TOP_LOGPROBS_NUM = 128
_SAMPLING_SEED = 1234
_SERVER_ARGS = (
    "--mem-fraction-static",
    "0.7",
    "--enable-custom-logit-processor",
    "--sampling-mask-max-tokens",
    "64",
)
_INVALID_SAMPLING_MASK_ERROR = (
    "return_sampling_mask requires top_k=1 for greedy sampling"
)


class TestSamplingMaskCapture(CustomTestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)
        self.sampler.sampling_mask_max_tokens = 4096
        self.sampler.tp_sync_group = None
        self.sampler.cp_sync_group = None

    def test_default_sampling_does_not_construct_capture_helpers(self):
        """Requests without masks must bypass capture-only allocations."""
        probs = torch.tensor([[0.6, 0.4]])
        info = SimpleNamespace(sampling_mask_batch_indices=None, sampling_seed=None)
        with patch.object(
            sampler_module, "partial", side_effect=AssertionError("capture helper")
        ):
            _, capture = self.sampler._sample_from_probs(
                probs=probs,
                sampling_info=info,
                positions=torch.tensor([0]),
                simple_sampling_case=True,
            )
        self.assertIsNone(capture)

    def _sample(
        self, probs, backend, *, top_k=2, top_p=0.45, min_p=0.0, requested_rows=None
    ):
        batch_size = len(probs)
        if requested_rows is None:
            requested_rows = range(batch_size)
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=True,
            need_top_p_sampling=top_p < 1.0,
            need_min_p_sampling=min_p > 0.0,
            top_ks=torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda"),
            top_ps=torch.full((batch_size,), top_p, device="cuda"),
            min_ps=torch.full((batch_size,), min_p, device="cuda"),
            sampling_mask_batch_indices=torch.tensor(requested_rows, device="cuda"),
        )
        with patch.object(
            sampler_module,
            "get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend=backend)
            ),
        ):
            return self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.zeros(batch_size, dtype=torch.int64, device="cuda"),
                simple_sampling_case=False,
            )

    def _materialize(self, sampled, capture, requested_rows):
        output = LogitsProcessorOutput(
            next_token_logits=None,
            sampling_mask_output=self.sampler._build_sampling_mask_output(
                sampled,
                capture,
                support_capture_indices=torch.arange(
                    len(requested_rows), device=sampled.device
                ),
            ),
        )
        output.sampling_mask_output.map_device_tensors(lambda tensor: tensor.cpu())
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            [
                SimpleNamespace(
                    return_sampling_mask=i in requested_rows,
                    sampling_logprobs_mode="support",
                )
                for i in range(len(sampled))
            ],
            output,
        )
        return output

    def test_min_p_capture_matches_filtered_support_and_logprob(self):
        backends = ("pytorch",) if is_hip() else ("pytorch", "flashinfer")
        for backend in backends:
            with self.subTest(backend=backend):
                probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device="cuda")
                sampled, capture = self._sample(
                    probs, backend, top_k=3, top_p=1.0, min_p=0.6
                )
                output = self.sampler._build_sampling_mask_output(
                    sampled,
                    capture,
                    support_capture_indices=torch.tensor([0], device="cuda"),
                )
                self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
                self.assertEqual(output.lengths.tolist(), [2])
                support = output.token_ids[0, :2].tolist()
                self.assertEqual(set(support), {0, 1})
                sampled_index = support.index(sampled.item())
                expected = (0.4 if sampled.item() == 0 else 0.3) / 0.7
                self.assertAlmostEqual(
                    output.support_logprobs[0, sampled_index].item(),
                    math.log(expected),
                    places=6,
                )

    def test_hard_exclusion_replay_in_mixed_batch(self):
        backends = ["pytorch"] if is_hip() else ["pytorch", "flashinfer"]
        for backend in backends:
            with self.subTest(backend=backend):
                logits = (
                    torch.tensor([[0.3, 0.2, 0.5, 0.15, 0.1]], device="cuda")
                    .log()
                    .repeat(2, 1)
                )
                original = logits.clone()
                info = SamplingBatchInfo(
                    temperatures=torch.ones(2, 1, device="cuda"),
                    top_ps=torch.full((2,), 0.9, device="cuda"),
                    top_ks=torch.full((2,), 3, dtype=torch.int32, device="cuda"),
                    min_ps=torch.zeros(2, device="cuda"),
                    is_all_greedy=False,
                    is_any_greedy=False,
                    need_top_p_sampling=True,
                    need_top_k_sampling=True,
                    need_min_p_sampling=False,
                    vocab_size=5,
                    has_custom_logit_processor=True,
                    custom_params=[{"token_ids": [2]}, None],
                    custom_logit_processor={
                        0: ProcessorEntry(
                            processor=DisallowedTokensLogitsProcessor(),
                            rows=[0],
                            indices=torch.tensor([0], device="cuda"),
                        )
                    },
                    return_sampling_masks=[True, True],
                    sampling_mask_batch_indices=torch.tensor([0, 1], device="cuda"),
                )
                logits = self.sampler._preprocess_logits(logits, info)
                with patch(
                    "sglang.srt.layers.sampler.get_exec",
                    return_value=SimpleNamespace(
                        kernel=SimpleNamespace(sampling_backend=backend)
                    ),
                ):
                    sampled, capture = self.sampler._sample_from_probs(
                        logits.softmax(-1),
                        info,
                        positions=torch.zeros(2, dtype=torch.int64, device="cuda"),
                        simple_sampling_case=False,
                    )
                output = self._materialize(sampled, capture, requested_rows=[0, 1])
                support = output.next_token_sampling_mask_idx[0]
                sampling_logprobs = output.next_token_sampling_logprobs[0]
                self.assertEqual(set(support), {0, 1, 3})
                self.assertIn(int(sampled[0]), support)
                expected = original[0, sampled[0]] - original[0, support].logsumexp(0)
                self.assertAlmostEqual(
                    sampling_logprobs[support.index(int(sampled[0]))],
                    expected.item(),
                    places=5,
                )
                self.assertIn(2, output.next_token_sampling_mask_idx[1])

    @unittest.skipIf(is_hip(), "FlashInfer is not available on ROCm")
    def test_flashinfer_joint_cutoff_ties_match_capture(self):
        batch_size = 256
        top_k = 2
        top_p = 0.45
        base_probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]], device="cuda")
        probs = base_probs.repeat(batch_size, 1)

        # Derive the threshold-based joint support independently. Both filters
        # cut at 0.2, so the tied entries must survive even though this yields
        # more support entries than top_k.
        sorted_probs = base_probs[0].sort(descending=True).values
        top_k_cutoff = sorted_probs[top_k - 1]
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        top_p_cutoff = sorted_probs[mass_before <= top_p][-1]
        expected_support = (base_probs[0] >= top_k_cutoff) & (
            base_probs[0] >= top_p_cutoff
        )
        expected_ids = expected_support.nonzero(as_tuple=True)[0].tolist()
        self.assertEqual(expected_ids, [0, 1, 2])

        sampled, capture = self._sample(probs, "flashinfer", top_k=top_k, top_p=top_p)

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), list(range(batch_size)))
        actual_support = capture.weights > 0
        self.assertTrue(
            torch.equal(actual_support, expected_support.expand_as(actual_support))
        )
        self.assertGreater(int(actual_support[0].sum().item()), top_k)
        self.assertTrue(
            bool(actual_support.gather(1, sampled.view(-1, 1)).all().item())
        )

    @unittest.skipIf(is_hip(), "FlashInfer is not available on ROCm")
    def test_flashinfer_capture_only_materializes_requested_rows(self):
        batch_size = 4
        requested_rows = [1, 3]
        probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]], device="cuda").repeat(
            batch_size, 1
        )
        with (
            patch.object(
                sampler_module,
                "top_k_renorm_prob",
                wraps=sampler_module.top_k_renorm_prob,
            ) as top_k_mock,
            patch.object(
                sampler_module,
                "top_p_renorm_prob",
                wraps=sampler_module.top_p_renorm_prob,
            ) as top_p_mock,
        ):
            sampled, capture = self._sample(
                probs, "flashinfer", requested_rows=requested_rows
            )

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), requested_rows)
        self.assertEqual(tuple(capture.weights.shape), (len(requested_rows), 5))
        self.assertEqual(tuple(top_k_mock.call_args.args[0].shape), (2, 5))
        self.assertEqual(tuple(top_p_mock.call_args.args[0].shape), (2, 5))

        output = self._materialize(sampled, capture, requested_rows)
        self.assertIsNone(output.next_token_sampling_mask_idx[0])
        self.assertEqual(set(output.next_token_sampling_mask_idx[1]), {0, 1, 2})
        self.assertIsNone(output.next_token_sampling_mask_idx[2])
        self.assertEqual(set(output.next_token_sampling_mask_idx[3]), {0, 1, 2})
        self.assertIsNone(output.next_token_sampling_logprobs[0])
        self.assertIsNotNone(output.next_token_sampling_logprobs[1])
        self.assertIsNone(output.next_token_sampling_logprobs[2])
        self.assertIsNotNone(output.next_token_sampling_logprobs[3])

    def test_pytorch_capture_compacts_requested_rows(self):
        batch_size = 4
        requested_rows = [1, 3]
        probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]], device="cuda").repeat(
            batch_size, 1
        )
        sampled, capture = self._sample(probs, "pytorch", requested_rows=requested_rows)

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), requested_rows)
        self.assertEqual(tuple(capture.weights.shape), (len(requested_rows), 5))
        self.assertEqual(tuple(capture.token_ids.shape), (len(requested_rows), 5))

        output = self._materialize(sampled, capture, requested_rows)
        for batch_row in requested_rows:
            self.assertIn(
                int(sampled[batch_row]),
                output.next_token_sampling_mask_idx[batch_row],
            )
            self.assertIsNotNone(output.next_token_sampling_logprobs[batch_row])
        self.assertIsNone(output.next_token_sampling_mask_idx[0])
        self.assertIsNone(output.next_token_sampling_mask_idx[2])


class SamplingMaskTestMixin:
    @classmethod
    def _launch_server(cls, other_args=()):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(*_SERVER_ARGS, *other_args),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post_generate(
        self,
        sampling_params,
        return_sampling_mask=True,
        return_logprob=False,
        top_logprobs_num=0,
        custom_logit_processor=None,
        stream=False,
        sampling_logprobs_mode="support",
    ):
        payload = {
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 1.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
                **sampling_params,
            },
            "return_sampling_mask": return_sampling_mask,
            "stream": stream,
        }
        if return_sampling_mask and sampling_logprobs_mode is not None:
            payload["sampling_logprobs_mode"] = sampling_logprobs_mode
        if custom_logit_processor is not None:
            payload["custom_logit_processor"] = custom_logit_processor
        if return_logprob:
            payload["return_logprob"] = True
            payload["top_logprobs_num"] = top_logprobs_num
        return requests.post(
            self.base_url + "/generate", json=payload, stream=stream, timeout=60
        )

    def _assert_sampling_masks(self, output_ids, meta_info):
        masks = meta_info["output_token_sampling_mask"]
        sampling_logprobs = meta_info["output_token_sampling_logprobs"]
        self.assertEqual(len(masks), len(output_ids))
        self.assertEqual(len(sampling_logprobs), len(output_ids))
        for token_id, mask, logprobs in zip(
            output_ids, masks, sampling_logprobs, strict=True
        ):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(set(mask)))
            self.assertEqual(len(mask), len(logprobs))
            self.assertTrue(all(math.isfinite(logprob) for logprob in logprobs))
            self.assertAlmostEqual(
                sum(math.exp(logprob) for logprob in logprobs), 1.0, delta=1e-5
            )
        return masks

    def _generate_sampling_masks(self, sampling_params):
        response = self._post_generate(sampling_params)
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        meta_info = output["meta_info"]
        output_ids = output["output_ids"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(meta_info["completion_tokens"], len(output_ids))
        self.assertEqual(
            meta_info["output_token_sampling_mask_length"], len(output_ids)
        )
        return self._assert_sampling_masks(output_ids, meta_info)


class TestSamplingMask(SamplingMaskTestMixin, CustomTestCase):
    _sampling_backend = "flashinfer"

    @classmethod
    def setUpClass(cls):
        cls._launch_server()

    def test_disallowed_tokens_with_replay(self):
        params = {
            "temperature": 1.0,
            "top_k": _TOP_K,
            "top_p": _TOP_P,
            "max_new_tokens": 1,
            "ignore_eos": True,
        }
        baseline = self._post_generate(params)
        self.assertEqual(baseline.status_code, 200, baseline.text)
        # Exclude tokens that actually belong to the unmodified sampling support.
        blocked = baseline.json()["meta_info"]["output_token_sampling_mask"][0][:2]
        self.assertTrue(blocked)
        response = self._post_generate(
            {**params, "custom_params": {"token_ids": blocked}},
            return_logprob=True,
            top_logprobs_num=_TOP_LOGPROBS_NUM,
            custom_logit_processor=DisallowedTokensLogitsProcessor.to_str(),
        )
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()
        meta = output["meta_info"]
        token = output["output_ids"][0]
        mask = meta["output_token_sampling_mask"][0]
        sampling_logprobs = meta["output_token_sampling_logprobs"][0]
        self.assertTrue(set(mask).isdisjoint(blocked))
        self.assertIn(token, mask)
        probs = {
            int(tid): math.exp(lp) for lp, tid, _ in meta["output_top_logprobs"][0]
        }
        expected = math.log(probs[token] / sum(probs[tid] for tid in mask))
        self.assertAlmostEqual(
            sampling_logprobs[mask.index(token)], expected, delta=1e-2
        )

    def test_sampling_logprobs_default_to_selected_token(self):
        response = self._post_generate(
            {"top_k": _TOP_K, "top_p": _TOP_P},
            sampling_logprobs_mode=None,
        )
        self.assertEqual(response.status_code, 200, response.text)
        meta = response.json()["meta_info"]
        self.assertTrue(meta["output_token_sampling_logprobs"])
        self.assertTrue(
            all(
                isinstance(logprob, float)
                for logprob in meta["output_token_sampling_logprobs"]
            )
        )

    def test_rejected_processors_do_not_break_generation(self):
        params = {"top_k": _TOP_K, "max_new_tokens": 1}
        for processor in (
            Qwen3ThinkingBudgetLogitProcessor.to_str(),
            "invalid processor",
        ):
            with self.subTest(processor=processor):
                response = self._post_generate(params, custom_logit_processor=processor)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(
                    "only supports DisallowedTokensLogitsProcessor", response.text
                )
                recovery = self._post_generate(params)
                self.assertEqual(recovery.status_code, 200, recovery.text)

    def test_generate_returns_sampling_mask(self):
        for params, min_size in (
            ({"top_p": _TOP_P}, 1),
            ({}, _TOP_K),
            ({"top_p": 1.0}, _TOP_K),
        ):
            with self.subTest(sampling_params=params):
                masks = self._generate_sampling_masks({"top_k": _TOP_K, **params})
                for mask in masks:
                    self.assertGreaterEqual(len(mask), min_size)

    def test_generate_returns_greedy_singleton_mask(self):
        masks = self._generate_sampling_masks({"temperature": 0.0})
        self.assertTrue(all(len(mask) == 1 for mask in masks))

    def test_sampling_mask_matches_topk_logprobs(self):
        """Check the returned mask and its aligned behavior logprobs.

        We get a wide prefix of full-vocab logprobs via ``return_logprob`` so
        cutoff ties that extend beyond ``top_k`` are visible. With
        ``temperature=1.0`` these are the sampler's distribution, so
        ``p = exp(logprob)`` are the exact probabilities. For each token, we check:

        1. the sampled token is in the returned mask,
        2. every mask token is in the returned top logprobs and at or above
           the top-k cutoff (ties at the cutoff survive, so the mask may
           exceed ``top_k``),
        3. each sampling logprob equals log(p[token] / sum(p[t] for t in mask)).
        """
        top_k, top_p = _TOP_K, _TOP_P
        response = self._post_generate(
            {"top_k": top_k, "top_p": top_p},
            return_logprob=True,
            top_logprobs_num=_TOP_LOGPROBS_NUM,
        )
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        meta_info = output["meta_info"]
        output_ids = output["output_ids"]
        sampling_masks = self._assert_sampling_masks(output_ids, meta_info)
        sampling_logprobs = meta_info["output_token_sampling_logprobs"]
        top_logprobs = meta_info["output_top_logprobs"]  # [logprob, id, text] per token

        self.assertEqual(len(top_logprobs), len(output_ids))

        for output_id, mask, mask_logprobs, step_top_logprobs in zip(
            output_ids, sampling_masks, sampling_logprobs, top_logprobs, strict=True
        ):
            probs = {
                int(tid): math.exp(logprob) for logprob, tid, _ in step_top_logprobs
            }

            mask_set = set(mask)

            self.assertTrue(mask_set.issubset(probs))
            top_k_cutoff = sorted(probs.values(), reverse=True)[top_k - 1]
            for token_id in mask_set:
                # 1e-3 slack: the kernel cuts on its own probs, not these logprobs.
                self.assertGreaterEqual(probs[token_id], top_k_cutoff * (1 - 1e-3))

            support_mass = sum(probs[token_id] for token_id in mask_set)
            for token_id, sampling_logprob in zip(mask, mask_logprobs, strict=True):
                expected_logprob = math.log(probs[token_id] / support_mass)
                self.assertAlmostEqual(sampling_logprob, expected_logprob, delta=1e-2)

    def test_chat_completions_returns_sampling_mask(self):
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Name a capital city."}],
                "temperature": 1.0,
                "top_k": _TOP_K,
                "top_p": _TOP_P,
                "max_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
                "return_sampling_mask": True,
                "sampling_logprobs_mode": "support",
                "return_meta_info": True,
                "return_token_ids": True,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)

        choice = response.json()["choices"][0]
        output_ids = choice["response_token_ids"]
        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self._assert_sampling_masks(output_ids, choice["meta_info"])

    def test_generate_streams_aligned_sampling_masks(self):
        response = self._post_generate({"top_k": _TOP_K, "top_p": _TOP_P}, stream=True)
        self.assertEqual(response.status_code, 200, response.text)

        output_ids = []
        for line in response.iter_lines():
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            chunk = json.loads(line[6:])
            output_ids = chunk["output_ids"]
            self._assert_sampling_masks(output_ids, chunk["meta_info"])

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)

    def test_generate_rejects_unbounded_sampling_mask(self):
        for params in ({"top_p": _TOP_P}, {"top_k": 65}, {"top_p": 1.0}):
            with self.subTest(sampling_params=params):
                response = self._post_generate(params)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(_INVALID_SAMPLING_MASK_ERROR, response.text)


class TestSamplingMaskPacking(CustomTestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        self.sampler.sampling_mask_max_tokens = 3
        self.sampler.tp_sync_group = None
        self.sampler.cp_sync_group = None

    def test_selected_token_must_have_positive_captured_weight(self):
        for token_ids in (None, torch.tensor([[2, 1, 0]], dtype=torch.int32)):
            with self.subTest(sorted_capture=token_ids is not None):
                capture = _SamplingMaskCapture(
                    batch_rows=torch.tensor([0]),
                    weights=torch.tensor([[0.7, 0.3, 0.0]]),
                    token_ids=token_ids,
                    selected_weight=None,
                )
                selected = torch.tensor([2 if token_ids is None else 0])
                output = self.sampler._build_sampling_mask_output(
                    selected, capture, support_capture_indices=None
                )
                self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.INVALID])

    def test_selected_mode_does_not_build_support_logprobs(self):
        capture = _SamplingMaskCapture(
            batch_rows=torch.tensor([0]),
            weights=torch.tensor([[0.6, 0.4, 0.0]]),
            token_ids=None,
            selected_weight=torch.tensor([0.6]),
        )
        output = self.sampler._build_sampling_mask_output(
            torch.tensor([0]), capture, support_capture_indices=None
        )
        self.assertIsNone(output.support_logprobs)
        self.assertAlmostEqual(output.selected_logprobs.item(), math.log(0.6))

    def test_support_logprobs_align_with_packed_token_ids(self):
        captures = (
            _SamplingMaskCapture(
                batch_rows=torch.tensor([0]),
                weights=torch.tensor([[0.1, 0.6, 0.0, 0.3]]),
                token_ids=None,
                selected_weight=torch.tensor([0.3]),
            ),
            _SamplingMaskCapture(
                batch_rows=torch.tensor([0]),
                weights=torch.tensor([[0.6, 0.3, 0.1, 0.0]]),
                token_ids=torch.tensor([[1, 3, 0, 2]], dtype=torch.int32),
                selected_weight=torch.tensor([0.3]),
            ),
        )
        for capture in captures:
            with self.subTest(sorted_capture=capture.token_ids is not None):
                output = self.sampler._build_sampling_mask_output(
                    torch.tensor([3]),
                    capture,
                    support_capture_indices=torch.tensor([0]),
                )
                self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
                self.assertEqual(output.lengths.tolist(), [3])
                self.assertEqual(output.token_ids[0, :3].tolist(), [1, 3, 0])
                torch.testing.assert_close(
                    output.support_logprobs[0, :3].exp(),
                    torch.tensor([0.6, 0.3, 0.1]),
                )

    def test_support_logprobs_only_pack_support_mode_rows(self):
        capture = _SamplingMaskCapture(
            batch_rows=torch.tensor([0, 1]),
            weights=torch.tensor(
                [
                    [0.7, 0.3, 0.0],
                    [0.1, 0.6, 0.3],
                ]
            ),
            token_ids=None,
            selected_weight=torch.tensor([0.7, 0.6]),
        )
        output = self.sampler._build_sampling_mask_output(
            torch.tensor([0, 1]),
            capture,
            support_capture_indices=torch.tensor([1]),
        )
        self.assertEqual(tuple(output.selected_logprobs.shape), (2,))
        self.assertEqual(tuple(output.support_logprobs.shape), (1, 3))
        torch.testing.assert_close(
            output.support_logprobs[0].exp(),
            torch.tensor([0.6, 0.3, 0.1]),
        )

    def test_synced_token_logprob_is_recomputed_from_capture(self):
        capture = _SamplingMaskCapture(
            batch_rows=torch.tensor([0]),
            weights=torch.tensor([[0.6, 0.2, 0.0]]),
            token_ids=torch.tensor([[2, 1, 0]], dtype=torch.int32),
            selected_weight=None,
        )
        output = self.sampler._build_sampling_mask_output(
            torch.tensor([1]),
            capture,
            support_capture_indices=torch.tensor([0]),
        )
        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
        self.assertAlmostEqual(output.selected_logprobs.item(), math.log(0.25))
        support = output.token_ids[0, :2].tolist()
        self.assertAlmostEqual(
            output.support_logprobs[0, support.index(1)].item(), math.log(0.25)
        )

    def test_greedy_device_output_survives_async_copy(self):
        from sglang.srt.managers.utils import GenerationBatchResult

        tokens = torch.tensor([3, 4, 5], device="cuda")
        output = LogitsProcessorOutput(
            next_token_logits=None,
            sampling_mask_output=self.sampler._build_greedy_sampling_mask_output(
                torch.tensor([0, 2], device="cuda"),
                tokens,
                support_capture_indices=torch.tensor([0, 1], device="cuda"),
            ),
        )
        result = GenerationBatchResult(
            logits_output=output, next_token_ids=tokens, copy_done=torch.cuda.Event()
        )
        result.copy_to_cpu(return_logprob=False)
        result.copy_done.synchronize()
        self.assertEqual(output.sampling_mask_output.token_ids.device.type, "cpu")
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            [
                SimpleNamespace(
                    return_sampling_mask=flag,
                    sampling_logprobs_mode="support",
                )
                for flag in (True, False, True)
            ],
            output,
        )
        self.assertEqual(output.next_token_sampling_mask_idx, [[3], None, [5]])
        self.assertEqual(output.next_token_sampling_logprobs, [[0.0], None, [0.0]])

    def test_overflow_never_materializes_a_partial_mask(self):
        # Simulate a top-k cutoff tie: a nominal top_k below the cap can still
        # produce more positive weights than the fixed transport can hold.
        capture = _SamplingMaskCapture(
            batch_rows=torch.tensor([0]),
            weights=torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]),
            token_ids=None,
            selected_weight=torch.tensor([0.2]),
        )

        sampling_output = self.sampler._build_sampling_mask_output(
            torch.tensor([0]),
            capture,
            support_capture_indices=torch.tensor([0]),
        )

        output = LogitsProcessorOutput(
            next_token_logits=None,
            sampling_mask_output=sampling_output,
        )
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            [
                SimpleNamespace(
                    return_sampling_mask=True,
                    sampling_logprobs_mode="support",
                )
            ],
            output,
        )
        self.assertEqual(
            output.next_token_sampling_mask_status,
            [SamplingMaskStatus.OVERFLOW],
        )
        self.assertEqual(output.next_token_sampling_mask_idx, [None])
        self.assertEqual(output.next_token_sampling_logprobs, [None])


class TestSamplingMaskDeterministic(SamplingMaskTestMixin, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # This test validates sampler/output determinism, not backend selection.
        # Pin Triton so the same deterministic path runs on CUDA and ROCm CI.
        cls._launch_server(
            ("--enable-deterministic-inference", "--attention-backend", "triton")
        )

    def test_return_sampling_mask_preserves_deterministic_sampling(self):
        sampling_params = {
            "top_k": _TOP_K,
            "top_p": 1.0,
            "sampling_seed": _SAMPLING_SEED,
        }

        outputs = []
        for return_mask in (False, True):
            response = self._post_generate(
                sampling_params, return_sampling_mask=return_mask
            )
            self.assertEqual(response.status_code, 200, response.text)
            output = response.json()
            outputs.append((output["output_ids"], output["text"]))
        self.assertEqual(outputs[0], outputs[1])


class TestSamplingMaskPytorch(TestSamplingMask):
    _sampling_backend = "pytorch"

    @classmethod
    def setUpClass(cls):
        cls._launch_server(("--sampling-backend", "pytorch"))


@unittest.skipIf(is_hip(), "The AMD sampling-mask CI suite provides only one GPU.")
class TestDistributedSamplingMask(CustomTestCase):
    def _check_parallel_config(self, *, tp_size, pp_size):
        process = None
        try:
            process = popen_launch_server(
                "Qwen/Qwen2.5-0.5B-Instruct",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--tp-size",
                    str(tp_size),
                    "--pp-size",
                    str(pp_size),
                    "--sampling-mask-max-tokens",
                    "64",
                    "--mem-fraction-static",
                    "0.5",
                    "--max-running-requests",
                    "8",
                    "--cuda-graph-max-bs-decode",
                    "8",
                ],
            )
            for return_logprob in (False, True):
                with self.subTest(return_logprob=return_logprob):
                    output = self._generate(
                        return_sampling_mask=True, return_logprob=return_logprob
                    )
                    token_ids = output["output_ids"]
                    meta = output["meta_info"]
                    masks = meta["output_token_sampling_mask"]
                    logprobs = meta["output_token_sampling_logprobs"]
                    self.assertEqual(len(token_ids), 4)
                    self.assertEqual(meta["output_token_sampling_mask_length"], 4)
                    self.assertEqual(len(masks), 4)
                    self.assertEqual(len(logprobs), 4)
                    for token_id, mask, step_logprobs in zip(
                        token_ids, masks, logprobs, strict=True
                    ):
                        self.assertIn(token_id, mask)
                        self.assertEqual(len(mask), len(set(mask)))
                        self.assertLessEqual(len(mask), 64)
                        self.assertEqual(len(mask), len(step_logprobs))
                        self.assertTrue(
                            all(math.isfinite(logprob) for logprob in step_logprobs)
                        )
                        self.assertTrue(
                            all(logprob <= 0.0 for logprob in step_logprobs)
                        )
                    if return_logprob:
                        self.assertEqual(len(meta["output_token_logprobs"]), 4)

            ordinary = self._generate(return_sampling_mask=False, return_logprob=False)
            self.assertEqual(len(ordinary["output_ids"]), 4)
            self.assertNotIn("output_token_sampling_mask", ordinary["meta_info"])
        finally:
            if process is not None:
                kill_process_tree(process.pid)
                process.wait(timeout=30)

    def _generate(self, *, return_sampling_mask, return_logprob):
        payload = {
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0.8,
                "top_k": 8,
                "top_p": 0.9,
                "max_new_tokens": 4,
                "ignore_eos": True,
            },
            "return_sampling_mask": return_sampling_mask,
            "return_logprob": return_logprob,
        }
        if return_sampling_mask:
            payload["sampling_logprobs_mode"] = "support"
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate", json=payload, timeout=120
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_tp2_sampling_mask(self):
        """Exercise status synchronization across two tensor-parallel ranks."""
        self._check_parallel_config(tp_size=2, pp_size=1)

    def test_pp2_sampling_mask(self):
        """Exercise mask transport between two live pipeline stages."""
        self._check_parallel_config(tp_size=1, pp_size=2)


if __name__ == "__main__":
    unittest.main()
