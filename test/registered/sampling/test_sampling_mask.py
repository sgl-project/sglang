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
from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=320, suite="stage-b-test-1-gpu-small-amd")

_MAX_NEW_TOKENS = 4
_TOP_P = 0.99
_TOP_K = 10
_TOP_LOGPROBS_NUM = 128
_SAMPLING_SEED = 1234
_SERVER_ARGS = (
    "--mem-fraction-static",
    "0.7",
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
                sampled, capture
            ),
        )
        output.sampling_mask_output.map_device_tensors(lambda tensor: tensor.cpu())
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            [
                SimpleNamespace(return_sampling_mask=i in requested_rows)
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
                output = self.sampler._build_sampling_mask_output(sampled, capture)
                self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
                self.assertEqual(output.lengths.tolist(), [2])
                self.assertEqual(set(output.token_ids[0, :2].tolist()), {0, 1})
                expected = (0.4 if sampled.item() == 0 else 0.3) / 0.7
                self.assertAlmostEqual(
                    output.selected_logprobs.item(), math.log(expected), places=6
                )

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
        stream=False,
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
        if return_logprob:
            payload["return_logprob"] = True
            payload["top_logprobs_num"] = top_logprobs_num
        return requests.post(
            self.base_url + "/generate", json=payload, stream=stream, timeout=60
        )

    def _assert_sampling_masks(self, output_ids, meta_info):
        masks = meta_info["output_token_sampling_mask"]
        self.assertEqual(len(masks), len(output_ids))
        self.assertEqual(
            len(meta_info["output_token_sampling_logprobs"]), len(output_ids)
        )
        for token_id, mask in zip(output_ids, masks):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(set(mask)))
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
        """Check the returned mask and its renormalized logprobs.

        We get a wide prefix of full-vocab logprobs via ``return_logprob`` so
        cutoff ties that extend beyond ``top_k`` are visible. With
        ``temperature=1.0`` these are the sampler's distribution, so
        ``p = exp(logprob)`` are the exact probabilities. For each token, we check:

        1. the sampled token is in the returned mask,
        2. every mask token is in the returned top logprobs and at or above
           the top-k cutoff (ties at the cutoff survive, so the mask may
           exceed ``top_k``),
        3. sampling_logprob == log(p[sampled] / sum(p[t] for t in mask)).
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

        for output_id, mask, mask_logprob, step_top_logprobs in zip(
            output_ids, sampling_masks, sampling_logprobs, top_logprobs
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
            expected_logprob = math.log(probs[output_id] / support_mass)
            self.assertAlmostEqual(mask_logprob, expected_logprob, delta=1e-2)

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
                output = self.sampler._build_sampling_mask_output(selected, capture)
                self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.INVALID])

    def test_synced_token_logprob_is_recomputed_from_capture(self):
        capture = _SamplingMaskCapture(
            batch_rows=torch.tensor([0]),
            weights=torch.tensor([[0.6, 0.2, 0.0]]),
            token_ids=torch.tensor([[2, 1, 0]], dtype=torch.int32),
            selected_weight=None,
        )
        output = self.sampler._build_sampling_mask_output(torch.tensor([1]), capture)
        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
        self.assertAlmostEqual(output.selected_logprobs.item(), math.log(0.25))

    def test_greedy_device_output_survives_async_copy(self):
        from sglang.srt.managers.utils import GenerationBatchResult

        tokens = torch.tensor([3, 4, 5], device="cuda")
        output = LogitsProcessorOutput(
            next_token_logits=None,
            sampling_mask_output=self.sampler._build_greedy_sampling_mask_output(
                torch.tensor([0, 2], device="cuda"), tokens
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
                SimpleNamespace(return_sampling_mask=flag)
                for flag in (True, False, True)
            ],
            output,
        )
        self.assertEqual(output.next_token_sampling_mask_idx, [[3], None, [5]])
        self.assertEqual(output.next_token_sampling_logprobs, [0.0, None, 0.0])

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
            torch.tensor([0]), capture
        )

        self.assertEqual(
            sampling_output.statuses.tolist(), [SamplingMaskStatus.OVERFLOW]
        )
        self.assertEqual(sampling_output.lengths.tolist(), [3])
        self.assertAlmostEqual(sampling_output.selected_logprobs.item(), -math.log(5))

        output = LogitsProcessorOutput(
            next_token_logits=None,
            sampling_mask_output=sampling_output,
        )
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            [SimpleNamespace(return_sampling_mask=True)], output
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

        with_mask_response = self._post_generate(
            sampling_params, return_sampling_mask=True
        )
        self.assertEqual(with_mask_response.status_code, 200, with_mask_response.text)

        without_mask_response = self._post_generate(
            sampling_params, return_sampling_mask=False
        )
        self.assertEqual(
            without_mask_response.status_code, 200, without_mask_response.text
        )

        with_mask_output = with_mask_response.json()
        without_mask_output = without_mask_response.json()
        self.assertEqual(
            with_mask_output["output_ids"], without_mask_output["output_ids"]
        )
        self.assertEqual(with_mask_output["text"], without_mask_output["text"])


class TestSamplingMaskPytorch(TestSamplingMask):
    _sampling_backend = "pytorch"

    @classmethod
    def setUpClass(cls):
        cls._launch_server(("--sampling-backend", "pytorch"))


if __name__ == "__main__":
    unittest.main()
