"""CPU-only tests for Ascend sampling dispatch and mask export contracts."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestAscendSamplerDispatch(unittest.TestCase):
    @torch.random.fork_rng(devices=[])
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
                    result, filtered_probs = (
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
                self.assertIsNone(filtered_probs)
                self.assertEqual(npu_top_k_top_p.called, eligible)


@patch.object(sampler_module, "torch_npu", SimpleNamespace(), create=True)
class TestAscendSamplingMaskCapture(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)
        self.sampler.use_ascend_backend = True

    @staticmethod
    def _sample(logits, *, fused=False, export=False, seed=None):
        return sampler_module.top_k_top_p_min_p_sampling_from_logits_ascend(
            logits.clone(),
            torch.tensor([2], dtype=torch.int32),
            torch.ones(1),
            torch.zeros(1),
            False,
            None if seed is None else torch.tensor([seed]),
            torch.zeros(1, dtype=torch.int64),
            npu_top_k_top_p_eligible=fused,
            return_filtered_probs=export,
        )

    @torch.random.fork_rng(devices=[])
    def test_export_contract_dtype_and_buffer(self):
        for dtype, fused, seed in product(
            (torch.float32, torch.bfloat16), (False, True), (None, 1234)
        ):
            with self.subTest(dtype=dtype, fused=fused, seed=seed):
                logits = torch.tensor([[1.0, 3.0, 0.0, 2.0]], dtype=dtype)
                filtered = logits.masked_fill(logits < 2, -float("inf"))
                probs = logits.softmax(-1)
                with (
                    # Match NPU eager execution; Dynamo cannot trace these CPU mocks.
                    torch.compiler.set_stance("force_eager"),
                    patch.object(
                        sampler_module,
                        "torch_npu",
                        SimpleNamespace(npu_top_k_top_p=lambda *_: filtered.clone()),
                        create=True,
                    ),
                    # CPU cannot run the device hash kernel; keep its input fixed.
                    patch.object(
                        sampler_module,
                        "murmur_hash32",
                        return_value=torch.full((1, 4), 2**31, dtype=torch.uint32),
                    ),
                    patch.object(
                        sampler_module.torch,
                        "softmax",
                        side_effect=[probs, probs.clone()],
                    ),
                ):
                    torch.default_generator.manual_seed(1234)
                    token, weights = self._sample(
                        logits, fused=fused, export=True, seed=seed
                    )
                    torch.default_generator.manual_seed(1234)
                    token_without_mask, weights_without_mask = self._sample(
                        logits, fused=fused, seed=seed
                    )
                self.assertIsInstance(token_without_mask, torch.Tensor)
                self.assertIsNone(weights_without_mask)
                self.assertEqual(tuple(token_without_mask.shape), (1,))
                self.assertTrue(torch.equal(token, token_without_mask))
                self.assertEqual(weights.dtype, dtype)
                self.assertEqual(tuple(weights.shape), tuple(logits.shape))
                self.assertEqual((weights[0] > 0).nonzero().flatten().tolist(), [1, 3])
                self.assertGreater(float(weights[0, token.item()]), 0)
                if not fused:
                    self.assertEqual(weights.data_ptr(), probs.data_ptr())

    @torch.random.fork_rng(devices=[])
    def test_cutoff_ties_and_overflow(self):
        # Emulate fused top-k=2 retaining both tokens tied at the cutoff.
        logits = torch.tensor([[3.0, 2.0, 2.0, -float("inf")]])
        self.sampler.tp_sync_group = self.sampler.cp_sync_group = None
        with patch.object(
            sampler_module,
            "torch_npu",
            SimpleNamespace(npu_top_k_top_p=lambda *_: logits.clone()),
        ):
            token, weights = self._sample(logits, fused=True, export=True)
        self.assertEqual((weights[0] > 0).nonzero().flatten().tolist(), [0, 1, 2])
        capture = self.sampler._build_ascend_sampling_mask_capture(
            weights, SimpleNamespace(sampling_mask_batch_indices=torch.tensor([0]))
        )
        for cap, mode in product((3, 2), ("support", "selected")):
            with self.subTest(cap=cap, mode=mode):
                self.sampler.sampling_mask_max_tokens = cap
                packed = self.sampler._build_sampling_mask_output(
                    token, capture, torch.tensor([0]) if mode == "support" else None
                )
                output = LogitsProcessorOutput(
                    next_token_logits=None, sampling_mask_output=packed
                )
                SchedulerBatchResultProcessor.materialize_sampling_mask_output(
                    [
                        SimpleNamespace(
                            return_sampling_mask=True, sampling_logprobs_mode=mode
                        )
                    ],
                    output,
                )
                status = (
                    SamplingMaskStatus.OK if cap == 3 else SamplingMaskStatus.OVERFLOW
                )
                self.assertEqual(output.next_token_sampling_mask_status, [status])
                if cap == 3:
                    self.assertEqual(
                        set(output.next_token_sampling_mask_idx[0]), {0, 1, 2}
                    )
                else:
                    self.assertEqual(output.next_token_sampling_mask_idx, [None])
                    self.assertEqual(output.next_token_sampling_logprobs, [None])

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

    @torch.random.fork_rng(devices=[])
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
