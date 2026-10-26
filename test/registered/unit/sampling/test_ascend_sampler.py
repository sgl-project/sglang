"""CPU-only tests for Ascend sampling dispatch and mask export contracts."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestAscendSamplerDispatch(unittest.TestCase):
    def test_top_k_dispatch_does_not_read_device_values_on_cpu(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        top_ks = torch.tensor([2], dtype=torch.int32)
        top_ps = torch.ones(1)
        min_ps = torch.zeros(1)
        positions = torch.zeros(1, dtype=torch.int64)

        for eligible in (True, False):
            with self.subTest(eligible=eligible):
                npu_top_k_top_p = MagicMock(return_value=logits)
                with (
                    patch.object(
                        sampler_module,
                        "torch_npu",
                        SimpleNamespace(npu_top_k_top_p=npu_top_k_top_p),
                        create=True,
                    ),
                    patch.object(
                        sampler_module.torch,
                        "all",
                        side_effect=AssertionError("device predicate was inspected"),
                    ),
                ):
                    result = (
                        sampler_module.top_k_top_p_min_p_sampling_from_logits_ascend(
                            logits.clone(),
                            top_ks.clone(),
                            top_ps,
                            min_ps,
                            False,
                            None,
                            positions,
                            npu_top_k_top_p_eligible=eligible,
                        )
                    )

                self.assertEqual(tuple(result.shape), (1,))
                self.assertEqual(npu_top_k_top_p.called, eligible)


@patch.object(sampler_module, "torch_npu", SimpleNamespace(), create=True)
class TestAscendSamplingMaskCapture(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)
        self.sampler.use_ascend_backend = True

    @staticmethod
    def _sample(logits, *, fused=False, seed=None, export=False):
        return sampler_module.top_k_top_p_min_p_sampling_from_logits_ascend(
            logits.clone(),
            torch.tensor([2], dtype=torch.int32),
            torch.ones(1),
            torch.zeros(1),
            False,
            seed,
            torch.zeros(1, dtype=torch.int64),
            npu_top_k_top_p_eligible=fused,
            return_filtered_probs=export,
        )

    def test_export_contract_dtype_and_buffer(self):
        for dtype, fused in product((torch.float32, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, fused=fused):
                logits = torch.tensor([[1.0, 3.0, 0.0, 2.0]], dtype=dtype)
                filtered = logits.masked_fill(logits < 2, -float("inf"))
                probs = logits.softmax(-1)
                with (
                    patch.object(
                        sampler_module,
                        "torch_npu",
                        SimpleNamespace(npu_top_k_top_p=lambda *_: filtered.clone()),
                        create=True,
                    ),
                    patch.object(
                        sampler_module.torch,
                        "softmax",
                        side_effect=[probs, probs.clone()],
                    ),
                ):
                    torch.manual_seed(1234)
                    token, weights = self._sample(logits, fused=fused, export=True)
                    torch.manual_seed(1234)
                    token_without_mask = self._sample(logits, fused=fused)
                self.assertIsInstance(token_without_mask, torch.Tensor)
                self.assertEqual(tuple(token_without_mask.shape), (1,))
                self.assertTrue(torch.equal(token, token_without_mask))
                self.assertEqual(weights.dtype, dtype)
                self.assertEqual(tuple(weights.shape), tuple(logits.shape))
                self.assertEqual((weights[0] > 0).nonzero().flatten().tolist(), [1, 3])
                self.assertGreater(float(weights[0, token.item()]), 0)
                if not fused:
                    self.assertEqual(weights.data_ptr(), probs.data_ptr())

    def test_seeded_export_preserves_draw(self):
        # Fix the hash source while executing the real seeded sampling logic.
        hashed = torch.full((1, 4), 2**31, dtype=torch.uint32)
        for dtype, fused in product((torch.float32, torch.bfloat16), (False, True)):
            with self.subTest(dtype=dtype, fused=fused):
                logits = torch.tensor([[1.0, 3.0, 0.0, 2.0]], dtype=dtype)
                filtered = logits.masked_fill(logits < 2, -float("inf"))
                with (
                    patch.object(
                        sampler_module,
                        "torch_npu",
                        SimpleNamespace(npu_top_k_top_p=lambda *_: filtered.clone()),
                        create=True,
                    ),
                    patch.object(sampler_module, "murmur_hash32", return_value=hashed),
                ):
                    seed = torch.tensor([1234])
                    token, weights = self._sample(
                        logits, fused=fused, seed=seed, export=True
                    )
                    without_mask = self._sample(logits, fused=fused, seed=seed)
                self.assertTrue(torch.equal(token, without_mask))
                self.assertEqual(weights.dtype, dtype)
                # With a constant hash the maximum wins, in vocabulary order.
                self.assertEqual(token.item(), 1)
                self.assertGreater(float(weights[0, token.item()]), 0)

    def test_capture_only_requested_rows(self):
        weights = torch.arange(20, dtype=torch.float32).view(4, 5)
        for rows in ([0, 1, 2, 3], [1, 3]):
            with self.subTest(rows=rows):
                info = SimpleNamespace(sampling_mask_batch_indices=torch.tensor(rows))
                capture = self.sampler._build_ascend_sampling_mask_capture(
                    weights, info
                )
                self.assertEqual(capture.batch_rows.tolist(), rows)
                self.assertEqual(tuple(capture.weights.shape), (len(rows), 5))
                self.assertTrue(torch.equal(capture.weights, weights[rows]))
                self.assertIsNone(capture.token_ids)

    def test_mask_disabled_does_not_construct_capture(self):
        info = SimpleNamespace(
            sampling_seed=None,
            temperatures=torch.ones(1, 1),
            top_ks=torch.tensor([2], dtype=torch.int32),
            top_ps=torch.ones(1),
            min_ps=torch.zeros(1),
            need_min_p_sampling=False,
            npu_top_k_top_p_eligible=False,
            sampling_mask_batch_indices=None,
        )
        with patch.object(
            self.sampler,
            "_build_ascend_sampling_mask_capture",
            side_effect=AssertionError("capture constructed without opt-in"),
        ):
            sampled, _, capture = self.sampler._forward_ascend_backend(
                torch.tensor([[1.0, 3.0, 0.0, 2.0]]),
                info,
                simple_sampling_case=False,
                return_logprob=False,
                positions=torch.zeros(1, dtype=torch.int64),
            )
        self.assertEqual(tuple(sampled.shape), (1,))
        self.assertIsNone(capture)


if __name__ == "__main__":
    unittest.main()
