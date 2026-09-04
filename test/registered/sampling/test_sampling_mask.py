import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
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
)
_INVALID_SAMPLING_MASK_ERROR = (
    "top_p-only sampling is valid but can return huge masks in the tail"
)


class TestSamplingMaskCapture(CustomTestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)

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

        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            top_ks=torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda"),
            top_ps=torch.full((batch_size,), top_p, device="cuda"),
            min_ps=torch.zeros(batch_size, device="cuda"),
            return_sampling_masks=[True] * batch_size,
        )
        with patch(
            "sglang.srt.layers.sampler.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="flashinfer")
            ),
        ):
            sampled, capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.zeros(batch_size, dtype=torch.int64, device="cuda"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

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
        top_k = 2
        top_p = 0.45
        requested_rows = [1, 3]
        probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]], device="cuda").repeat(
            batch_size, 1
        )
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            top_ks=torch.full((batch_size,), top_k, dtype=torch.int32, device="cuda"),
            top_ps=torch.full((batch_size,), top_p, device="cuda"),
            min_ps=torch.zeros(batch_size, device="cuda"),
            return_sampling_masks=[False, True, False, True],
        )
        top_k_renorm = sampler_module.top_k_renorm_prob
        top_p_renorm = sampler_module.top_p_renorm_prob
        with (
            patch(
                "sglang.srt.layers.sampler.get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(sampling_backend="flashinfer")
                ),
            ),
            patch(
                "sglang.srt.layers.sampler.top_k_renorm_prob",
                wraps=top_k_renorm,
            ) as top_k_mock,
            patch(
                "sglang.srt.layers.sampler.top_p_renorm_prob",
                wraps=top_p_renorm,
            ) as top_p_mock,
        ):
            sampled, capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.zeros(batch_size, dtype=torch.int64, device="cuda"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), requested_rows)
        self.assertEqual(tuple(capture.weights.shape), (len(requested_rows), 5))
        self.assertEqual(tuple(top_k_mock.call_args.args[0].shape), (2, 5))
        self.assertEqual(tuple(top_p_mock.call_args.args[0].shape), (2, 5))

        output = LogitsProcessorOutput(next_token_logits=None)
        self.sampler._attach_sampling_mask_to_output(
            output, sampling_info, sampled, capture
        )
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
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            top_ks=torch.full((batch_size,), 2, dtype=torch.int32, device="cuda"),
            top_ps=torch.full((batch_size,), 0.45, device="cuda"),
            min_ps=torch.zeros(batch_size, device="cuda"),
            return_sampling_masks=[False, True, False, True],
        )
        with patch(
            "sglang.srt.layers.sampler.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="pytorch")
            ),
        ):
            sampled, capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.zeros(batch_size, dtype=torch.int64, device="cuda"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), requested_rows)
        self.assertEqual(tuple(capture.weights.shape), (len(requested_rows), 5))
        self.assertEqual(tuple(capture.token_ids.shape), (len(requested_rows), 5))

        output = LogitsProcessorOutput(next_token_logits=None)
        self.sampler._attach_sampling_mask_to_output(
            output, sampling_info, sampled, capture
        )
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
    ):
        payload = {
            "text": "The capital of France is",
            "sampling_params": sampling_params,
            "return_sampling_mask": return_sampling_mask,
        }
        if return_logprob:
            payload["return_logprob"] = True
            payload["top_logprobs_num"] = top_logprobs_num
        return requests.post(self.base_url + "/generate", json=payload, timeout=60)

    def _generate_sampling_masks(self, sampling_params):
        response = self._post_generate(sampling_params)
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        meta_info = output["meta_info"]
        output_ids = output["output_ids"]
        sampling_masks = meta_info["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(meta_info["completion_tokens"], len(output_ids))
        self.assertEqual(
            meta_info["output_token_sampling_mask_length"], len(output_ids)
        )
        self.assertEqual(len(sampling_masks), len(output_ids))
        for output_id, sampling_mask in zip(output_ids, sampling_masks):
            self.assertIn(output_id, sampling_mask)
            self.assertEqual(len(sampling_mask), len(set(sampling_mask)))
        return sampling_masks

    def _assert_rejects_unbounded_sampling_mask(self, sampling_params):
        response = self._post_generate(sampling_params)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(_INVALID_SAMPLING_MASK_ERROR, response.text)


class TestSamplingMask(SamplingMaskTestMixin, CustomTestCase):
    _sampling_backend = "flashinfer"

    @classmethod
    def setUpClass(cls):
        cls._launch_server()

    def test_generate_returns_sampling_mask(self):
        top_p_sampling_masks = self._generate_sampling_masks(
            {
                "temperature": 1.0,
                "top_k": _TOP_K,
                "top_p": _TOP_P,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        for sampling_mask in top_p_sampling_masks:
            self.assertGreater(len(sampling_mask), 0)

        top_k_sampling_masks = self._generate_sampling_masks(
            {
                "temperature": 1.0,
                "top_k": _TOP_K,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        for sampling_mask in top_k_sampling_masks:
            self.assertGreaterEqual(len(sampling_mask), _TOP_K)

        top_k_top_p_one_sampling_masks = self._generate_sampling_masks(
            {
                "temperature": 1.0,
                "top_k": _TOP_K,
                "top_p": 1.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        for sampling_mask in top_k_top_p_one_sampling_masks:
            self.assertGreaterEqual(len(sampling_mask), _TOP_K)

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
            {
                "temperature": 1.0,
                "top_k": top_k,
                "top_p": top_p,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
            return_logprob=True,
            top_logprobs_num=_TOP_LOGPROBS_NUM,
        )
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        meta_info = output["meta_info"]
        output_ids = output["output_ids"]
        sampling_masks = meta_info["output_token_sampling_mask"]
        sampling_logprobs = meta_info["output_token_sampling_logprobs"]
        top_logprobs = meta_info["output_top_logprobs"]  # [logprob, id, text] per token

        self.assertEqual(len(sampling_masks), len(output_ids))
        self.assertEqual(len(sampling_logprobs), len(output_ids))
        self.assertEqual(len(top_logprobs), len(output_ids))

        for output_id, mask, mask_logprob, step_top_logprobs in zip(
            output_ids, sampling_masks, sampling_logprobs, top_logprobs
        ):
            probs = {
                int(tid): math.exp(logprob) for logprob, tid, _ in step_top_logprobs
            }

            mask_set = set(mask)

            self.assertIn(output_id, mask_set)
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
        meta_info = choice["meta_info"]
        sampling_masks = meta_info["output_token_sampling_mask"]
        sampling_logprobs = meta_info["output_token_sampling_logprobs"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(len(sampling_masks), len(output_ids))
        self.assertEqual(len(sampling_logprobs), len(output_ids))
        for output_id, sampling_mask in zip(output_ids, sampling_masks):
            self.assertIn(output_id, sampling_mask)

    def test_generate_rejects_unbounded_sampling_mask(self):
        self._assert_rejects_unbounded_sampling_mask(
            {
                "temperature": 1.0,
                "top_p": _TOP_P,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )
        self._assert_rejects_unbounded_sampling_mask(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            }
        )


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
            "temperature": 1.0,
            "top_k": _TOP_K,
            "top_p": 1.0,
            "sampling_seed": _SAMPLING_SEED,
            "max_new_tokens": _MAX_NEW_TOKENS,
            "ignore_eos": True,
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
