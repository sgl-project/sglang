"""Sampling metadata built on the host and copied to the device in one transfer."""

import random
import unittest
from types import SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.sampling.logit_bias import LogitBiasEntries, build_logit_bias
from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="base-a-jit-kernel-unit", runner_config="amd")

VOCAB_SIZE = 32


def _req(**sampling_params):
    params = SamplingParams(**sampling_params)
    return SimpleNamespace(
        penalty_index=None,
        output_ids=[],
        sampling_params=params,
        logit_bias_entries=(
            None
            if params.logit_bias is None
            else LogitBiasEntries.parse(params.logit_bias)
        ),
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


def _admitted_pool(reqs, device):
    pool = PenaltyStatusPool(VOCAB_SIZE, device)
    pool.alloc(reqs)
    pool.reset_history(reqs)
    return pool


def _step_penalties(reqs, device):
    return StepPenalties.build(reqs, _admitted_pool(reqs, device))


class _SamplingMetadataTestBase(CustomTestCase):
    device = "cpu"

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
        pool = _admitted_pool(reqs, self.device)
        penalty_indices = StepPenalties.build(reqs, pool).penalty_indices
        self.assert_device_tensor(
            pool.live_tables().min_new_tokens[penalty_indices],
            torch.tensor([2, 1], dtype=torch.int32),
            torch.int32,
        )
        for step in range(3):
            penalties = StepPenalties.build(reqs, pool)
            additive, scaling = penalties.logit_penalties()
            self.assertIsNone(scaling)
            expected = torch.zeros(3, VOCAB_SIZE)
            if step < 2:
                expected[0, [1, 2, 3, 4]] = -torch.inf
            self.assert_device_tensor(additive, expected)
            penalties.commit(torch.ones(3, dtype=torch.long, device=self.device))

    def test_min_tokens_without_any_stop_tokens(self):
        penalties = _step_penalties([_req(min_new_tokens=1)], self.device)
        additive, _ = penalties.logit_penalties()
        self.assert_device_tensor(additive, torch.zeros(1, VOCAB_SIZE))

    def test_sparse_logit_bias_keeps_last_value_for_colliding_keys(self):
        reqs = [
            _req(logit_bias={"0": -100, "31": 100, "1": 2, "01": 3}),
            _req(),
            _req(logit_bias={}),
            _req(logit_bias={"0": 0, "2": -1.25}),
        ]
        logit_bias = build_logit_bias(reqs, vocab_size=VOCAB_SIZE, device=self.device)
        expected = torch.zeros(len(reqs), VOCAB_SIZE)
        expected[0, 0], expected[0, 31], expected[0, 1] = -100, 100, 3
        expected[3, 2] = -1.25
        self.assert_device_tensor(logit_bias, expected, torch.float32)

    def test_logit_bias_matches_a_dense_build(self):
        rng = random.Random(0)
        for _ in range(100):
            reqs = []
            expected = torch.zeros(rng.randint(1, 8), VOCAB_SIZE)
            for row in range(len(expected)):
                if rng.random() < 0.3:
                    reqs.append(_req())
                    continue
                # Distinct ids, so no two keys scatter to one element; negative keys wrap.
                bias = {
                    rng.choice([str(key), f"0{key}", str(key - VOCAB_SIZE)]): (
                        rng.uniform(-100, 100)
                    )
                    for key in rng.sample(range(VOCAB_SIZE), rng.randint(0, 6))
                }
                reqs.append(_req(logit_bias=bias))
                for key, value in bias.items():
                    expected[row, int(key)] = value
            logit_bias = build_logit_bias(
                reqs, vocab_size=VOCAB_SIZE, device=self.device
            )
            if any(r.logit_bias_entries is not None for r in reqs):
                self.assert_device_tensor(logit_bias, expected, torch.float32)
            else:
                self.assertIsNone(logit_bias)
        with self.assertRaises(IndexError):
            build_logit_bias(
                [_req(logit_bias={str(VOCAB_SIZE): 1.0})],
                vocab_size=VOCAB_SIZE,
                device=self.device,
            )

    def test_logit_bias_is_none_without_any_bias(self):
        """Once its last biased request finished, a running batch used to keep a
        zero [bs, vocab] bias, which kept the bias-free fast paths off."""
        self.assertIsNone(
            build_logit_bias(
                [_req(), _req()], vocab_size=VOCAB_SIZE, device=self.device
            )
        )


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
                additive, _ = _step_penalties(reqs, self.device).logit_penalties()
            torch.cuda.synchronize()
            copy_counts[batch_size] = len(copies.numels)
            self.assertTrue(torch.isneginf(additive[:, 2:4]).all().item())
        self.assertEqual(copy_counts[16], copy_counts[1])

    def test_step_build_copies_only_the_indices(self):
        reqs = [
            _req(min_new_tokens=2, stop_token_ids=[2, 3], frequency_penalty=0.5),
            _req(),
            _req(repetition_penalty=1.2),
        ]
        pool = _admitted_pool(reqs, self.device)
        pool.live_tables()
        with _H2DCopies() as copies:
            StepPenalties.build(reqs, pool)
        torch.cuda.synchronize()
        self.assertEqual(copies.numels, [4])

    def test_logit_bias_never_copies_a_dense_row(self):
        reqs = [_req(logit_bias={"1": 2, "31": -1}), _req(), _req(logit_bias={})]
        with _H2DCopies() as copies:
            logit_bias = build_logit_bias(
                reqs, vocab_size=VOCAB_SIZE, device=self.device
            )
        torch.cuda.synchronize()
        self.assertEqual(len(copies.numels), 1)
        self.assertLess(max(copies.numels), VOCAB_SIZE)
        expected = torch.zeros(len(reqs), VOCAB_SIZE)
        expected[0, 1], expected[0, 31] = 2, -1
        self.assert_device_tensor(logit_bias, expected, torch.float32)


if __name__ == "__main__":
    unittest.main()
