"""Sampling metadata built on the host and copied to the device in one transfer."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.sampling.penaltylib import (
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

VOCAB_SIZE = 32


class _Batch:
    def __init__(self, reqs, device):
        self.reqs = reqs
        self.device = device


def _req(**sampling_params):
    return SimpleNamespace(
        sampling_params=SamplingParams(**sampling_params),
        eos_token_ids=None,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
        custom_logit_processor=None,
        return_sampling_mask=False,
    )


class _H2DCopies(TorchDispatchMode):
    """Record the element count of every host-to-device copy."""

    def __init__(self):
        super().__init__()
        self.numels = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default:
            source = args[0]
            target = kwargs.get("device")
            if source.device.type == "cpu" and target and target.type == "cuda":
                self.numels.append(source.numel())
        return func(*args, **kwargs)


class _SamplingMetadataTestBase(CustomTestCase):
    device = "cpu"

    def setUp(self):
        super().setUp()
        exec_context = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
            features=SimpleNamespace(enable_custom_logit_processor=True),
        )
        context_patch = patch(
            "sglang.srt.sampling.sampling_batch_info.get_exec",
            return_value=exec_context,
        )
        context_patch.start()
        self.addCleanup(context_patch.stop)

    def assert_device_tensor(self, actual, expected, dtype=None):
        self.assertEqual(actual.device, torch.device(self.device))
        if dtype is not None:
            self.assertEqual(actual.dtype, dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


class TestSamplingMetadataCPU(_SamplingMetadataTestBase):
    def test_min_tokens_pads_ragged_stop_sets(self):
        reqs = [
            _req(min_new_tokens=2, stop_token_ids=[3]),
            _req(min_new_tokens=0, stop_token_ids=[5]),
            _req(min_new_tokens=1),
        ]
        # Row 0 unions four stop sources and must drop the None entries.
        reqs[0].sampling_params.stop_token_ids.add(None)
        reqs[0].eos_token_ids = {2}
        reqs[0].tokenizer.additional_stop_token_ids = {4, None}
        reqs[0].tokenizer.eos_token_id = 1
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, _Batch(reqs, self.device), {BatchedMinNewTokensPenalizer}
        )
        self.assert_device_tensor(
            orch.penalizers[BatchedMinNewTokensPenalizer].min_new_tokens,
            torch.tensor([[2], [0], [1]], dtype=torch.int32),
            torch.int32,
        )
        for step in range(3):
            logits = torch.zeros(3, VOCAB_SIZE, device=self.device)
            orch.apply(logits)
            expected = torch.zeros(3, VOCAB_SIZE)
            if step < 2:
                expected[0, [1, 2, 3, 4]] = -torch.inf
            self.assert_device_tensor(logits, expected)
            orch.cumulate_output_tokens(
                torch.ones(3, dtype=torch.long, device=self.device)
            )

    def test_min_tokens_without_any_stop_tokens(self):
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE,
            _Batch([_req(min_new_tokens=1)], self.device),
            {BatchedMinNewTokensPenalizer},
        )
        logits = torch.zeros(1, VOCAB_SIZE, device=self.device)
        orch.apply(logits)
        self.assert_device_tensor(logits, torch.zeros(1, VOCAB_SIZE))

    def test_sparse_logit_bias_keeps_last_value_for_colliding_keys(self):
        reqs = [
            _req(logit_bias={"0": -100, "31": 100, "1": 2, "01": 3}),
            _req(),
            _req(logit_bias={}),
            _req(logit_bias={"0": 0, "2": -1.25}),
        ]
        info = SamplingBatchInfo.from_schedule_batch(
            _Batch(reqs, self.device), VOCAB_SIZE
        )
        expected = torch.zeros(len(reqs), VOCAB_SIZE)
        expected[0, 0], expected[0, 31], expected[0, 1] = -100, 100, 3
        expected[3, 2] = -1.25
        self.assert_device_tensor(info.logit_bias, expected, torch.float32)

    def test_logit_bias_is_none_without_any_bias(self):
        info = SamplingBatchInfo.from_schedule_batch(
            _Batch([_req(), _req()], self.device), VOCAB_SIZE
        )
        self.assertIsNone(info.logit_bias)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSamplingMetadataCUDA(_SamplingMetadataTestBase):
    device = "cuda:0"

    def test_stop_token_copy_count_does_not_grow_with_batch_size(self):
        copy_counts = {}
        for batch_size in (1, 16):
            reqs = [
                _req(min_new_tokens=2, stop_token_ids=[2, 3]) for _ in range(batch_size)
            ]
            with _H2DCopies() as copies:
                orch = BatchedPenalizerOrchestrator(
                    VOCAB_SIZE,
                    _Batch(reqs, self.device),
                    {BatchedMinNewTokensPenalizer},
                )
                logits = torch.zeros(batch_size, VOCAB_SIZE, device=self.device)
                orch.apply(logits)
            torch.cuda.synchronize()
            copy_counts[batch_size] = len(copies.numels)
            self.assertTrue(torch.isneginf(logits[:, 2:4]).all().item())
        self.assertEqual(copy_counts[16], copy_counts[1])

    def test_logit_bias_never_copies_a_dense_row(self):
        reqs = [_req(logit_bias={"1": 2, "31": -1}), _req(), _req(logit_bias={})]
        with _H2DCopies() as copies:
            info = SamplingBatchInfo.from_schedule_batch(
                _Batch(reqs, self.device), VOCAB_SIZE
            )
        torch.cuda.synchronize()
        self.assertLess(max(copies.numels), VOCAB_SIZE)
        expected = torch.zeros(len(reqs), VOCAB_SIZE)
        expected[0, 1], expected[0, 31] = 2, -1
        self.assert_device_tensor(info.logit_bias, expected, torch.float32)


if __name__ == "__main__":
    unittest.main()
