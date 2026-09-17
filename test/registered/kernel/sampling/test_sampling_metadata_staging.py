"""Sampling metadata must retain its semantics across asynchronous H2D staging."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.layers.logprob_processor import (
    LogprobStage,
    get_token_ids_logprobs_raw,
)
from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor
from sglang.srt.sampling.penaltylib import (
    BatchedFrequencyPenalizer,
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedPresencePenalizer,
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

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
    def __init__(self):
        super().__init__()
        self.copies = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default:
            source = args[0]
            target = kwargs.get("device")
            if source.device.type == "cpu" and target and target.type == "cuda":
                self.copies.append(
                    (
                        source.numel(),
                        source.is_pinned(),
                        kwargs.get("non_blocking"),
                        target,
                    )
                )
        return func(*args, **kwargs)


class SamplingMetadataMixin:
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

    def test_heterogeneous_penalties_preserve_sign_and_repeated_token_counts(self):
        reqs = [
            _req(frequency_penalty=0.5, presence_penalty=0.25, repetition_penalty=2),
            _req(
                frequency_penalty=-0.5, presence_penalty=-0.25, repetition_penalty=0.5
            ),
            _req(),
        ]
        batch = _Batch(reqs, self.device)
        generated = [[1, 20, 3], [1, 22, 3]]
        original = torch.arange(-16, 16, dtype=torch.float32).repeat(3, 1)
        for cls, name in (
            (BatchedFrequencyPenalizer, "frequency"),
            (BatchedPresencePenalizer, "presence"),
            (BatchedRepetitionPenalizer, "repetition"),
        ):
            with self.subTest(penalty=name):
                orchestrator = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, {cls})
                for output_ids in generated:
                    orchestrator.cumulate_output_tokens(
                        torch.tensor(output_ids, device=self.device)
                    )
                logits = original.to(self.device).clone()
                orchestrator.apply(logits)
                expected = original.clone()
                for row, req in enumerate(reqs):
                    tokens = [step[row] for step in generated]
                    penalty = getattr(req.sampling_params, f"{name}_penalty")
                    for token in set(tokens):
                        if name == "frequency":
                            expected[row, token] -= penalty * tokens.count(token)
                        elif name == "presence":
                            expected[row, token] -= penalty
                        else:
                            value = expected[row, token]
                            expected[row, token] = (
                                value * penalty if value < 0 else value / penalty
                            )
                self.assert_device_tensor(logits, expected, torch.float32)

    def test_min_tokens_pads_stop_sets_and_handles_no_stop_tokens(self):
        reqs = [
            _req(min_new_tokens=2, stop_token_ids=[3]),
            _req(min_new_tokens=0, stop_token_ids=[5]),
            _req(min_new_tokens=1),
        ]
        reqs[0].sampling_params.stop_token_ids.add(None)
        reqs[0].eos_token_ids = {2}
        reqs[0].tokenizer.additional_stop_token_ids = {4, None}
        reqs[0].tokenizer.eos_token_id = 1
        batch = _Batch(reqs, self.device)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedMinNewTokensPenalizer}
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

        batch = _Batch([_req(min_new_tokens=1)], self.device)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedMinNewTokensPenalizer}
        )
        logits = torch.zeros(1, VOCAB_SIZE, device=self.device)
        orch.apply(logits)
        self.assert_device_tensor(logits, torch.zeros(1, VOCAB_SIZE))

    def test_sparse_bias_keeps_last_assignment_and_zero_rows(self):
        reqs = [
            _req(logit_bias={"0": -100, "31": 100, "1": 2, "01": 3}),
            _req(),
            _req(logit_bias={}),
            _req(logit_bias={"0": 0, "2": -1.25}),
        ]
        batch = _Batch(reqs, self.device)
        original_dtype = torch.get_default_dtype()
        try:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(dtype=dtype):
                    torch.set_default_dtype(dtype)
                    info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
                    expected = torch.zeros(len(reqs), VOCAB_SIZE)
                    for row, req in enumerate(reqs):
                        for key, value in (
                            req.sampling_params.logit_bias or {}
                        ).items():
                            expected[row, int(key)] = value
                    self.assert_device_tensor(info.logit_bias, expected, dtype)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_empty_batch_and_empty_bias_dict(self):
        batch = _Batch([], self.device)
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertEqual(info.temperatures.shape, (0, 1))
        self.assertIsNone(info.logit_bias)
        self.assertFalse(info.penalizer_orchestrator.is_required)
        self.assertFalse(info.has_custom_logit_processor)

        batch = _Batch([_req(logit_bias={})], self.device)
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assert_device_tensor(info.logit_bias, torch.zeros(1, VOCAB_SIZE))

    def test_custom_processor_masks_keep_request_groups(self):
        processor = DisallowedTokensLogitsProcessor.to_str()
        reqs = [_req(custom_params={"token_ids": [3]}) for _ in range(4)]
        reqs[0].custom_logit_processor = processor
        reqs[2].custom_logit_processor = processor
        reqs[3].custom_logit_processor = processor + " "
        batch = _Batch(reqs, self.device)
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertEqual(len(info.custom_logit_processor), 2)
        for serialized, rows in (
            (processor, [True, False, True, False]),
            (processor + " ", [False, False, False, True]),
        ):
            obj, mask = info.custom_logit_processor[hash(serialized)]
            self.assertIsInstance(obj, DisallowedTokensLogitsProcessor)
            self.assert_device_tensor(mask, torch.tensor(rows), torch.bool)
        self.assertEqual(
            info.custom_params, [r.sampling_params.custom_params for r in reqs]
        )

    def test_raw_logprob_indices_preserve_order_duplicates_and_skipped_rows(self):
        reference = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 4
        logprobs = reference.to(self.device)
        probes = [None, [], [3, 1, 3], [0]]
        for stage, lengths, expected in (
            (LogprobStage.DECODE, None, [[], [], [4.75, 4.25, 4.75], [6.0]]),
            (
                LogprobStage.PREFILL,
                [1, 0, 2, 1],
                [[], [], [[2.75, 2.25, 2.75], [4.75, 4.25, 4.75]], [[6.0]]],
            ),
        ):
            for no_copy in (False, True):
                with self.subTest(stage=stage, no_copy=no_copy):
                    vals, idxs = get_token_ids_logprobs_raw(
                        logprobs, probes, stage, lengths, no_copy_to_cpu=no_copy
                    )
                    for actual, wanted in zip(vals, expected):
                        if isinstance(actual, torch.Tensor):
                            self.assertEqual(actual.device, torch.device(self.device))
                            actual = actual.tolist()
                        self.assertEqual(actual, wanted)
                    self.assertEqual(
                        idxs,
                        [[], [], [3, 1, 3], [0]]
                        if stage == LogprobStage.DECODE
                        else [[], [], [[3, 1, 3], [3, 1, 3]], [[0]]],
                    )

    def test_raw_logprobs_empty_probe_sets(self):
        logprobs = torch.ones(2, 8, device=self.device)
        for no_copy in (False, True):
            vals, idxs = get_token_ids_logprobs_raw(
                logprobs,
                [[], None],
                LogprobStage.PREFILL,
                [2, 0],
                no_copy_to_cpu=no_copy,
            )
            first = vals[0].tolist() if no_copy else vals[0]
            self.assertEqual(first, [[], []])
            self.assertEqual(idxs, [[[], []], []])


class TestSamplingMetadataCPU(SamplingMetadataMixin, CustomTestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSamplingMetadataCUDA(SamplingMetadataMixin, CustomTestCase):
    device = "cuda:0"

    def assert_pinned_copies(self, copies, device):
        self.assertTrue(copies.copies)
        for _, pinned, non_blocking, target in copies.copies:
            self.assertTrue(pinned)
            self.assertTrue(non_blocking)
            self.assertEqual(target, device)

    def test_stop_padding_copy_count_does_not_grow_with_batch_size(self):
        devices = [0, 1] if torch.cuda.device_count() > 1 else [0]
        for device_index in devices:
            device = torch.device("cuda", device_index)
            stream = torch.cuda.Stream(device=device)
            for batch_size in (1, 16):
                reqs = [
                    _req(min_new_tokens=2, stop_token_ids=[2, 3])
                    for _ in range(batch_size)
                ]
                batch = _Batch(reqs, device)
                with self.subTest(device=device, batch_size=batch_size):
                    with torch.cuda.stream(stream), _H2DCopies() as copies:
                        orch = BatchedPenalizerOrchestrator(
                            VOCAB_SIZE, batch, {BatchedMinNewTokensPenalizer}
                        )
                        logits = torch.zeros(batch_size, VOCAB_SIZE, device=device)
                        orch.apply(logits)
                    stream.synchronize()
                    self.assertEqual(len(copies.copies), 2)
                    self.assert_pinned_copies(copies, device)
                    self.assertTrue(torch.isneginf(logits[:, 2:4]).all().item())

    def test_penalty_parameters_use_pinned_nonblocking_staging(self):
        device = torch.device(self.device)
        stream = torch.cuda.Stream(device=device)
        reqs = [
            _req(frequency_penalty=0.5, presence_penalty=0.25, repetition_penalty=2),
            _req(
                frequency_penalty=-0.5, presence_penalty=-0.25, repetition_penalty=0.5
            ),
            _req(),
        ]
        batch = _Batch(reqs, device)
        for cls, field, expected in (
            (BatchedFrequencyPenalizer, "frequency_penalties", [0.5, -0.5, 0]),
            (BatchedPresencePenalizer, "presence_penalties", [0.25, -0.25, 0]),
            (BatchedRepetitionPenalizer, "repetition_penalties", [2, 0.5, 1]),
        ):
            with self.subTest(penalizer=cls.__name__):
                with torch.cuda.stream(stream), _H2DCopies() as copies:
                    orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, {cls})
                stream.synchronize()
                self.assertEqual(len(copies.copies), 1)
                self.assertEqual(copies.copies[0][0], len(reqs))
                self.assert_pinned_copies(copies, device)
                self.assert_device_tensor(
                    getattr(orch.penalizers[cls], field),
                    torch.tensor(expected, dtype=torch.float32).view(-1, 1),
                    torch.float32,
                )

    def test_bias_masks_and_logprob_indices_on_non_default_stream(self):
        device = torch.device(self.device)
        stream = torch.cuda.Stream(device=device)
        processor = DisallowedTokensLogitsProcessor.to_str()
        reqs = [_req(logit_bias={"1": 2, "31": -1}), _req(), _req(logit_bias={})]
        reqs[0].custom_logit_processor = processor
        reqs[2].custom_logit_processor = processor
        batch = _Batch(reqs, device)
        logprobs = torch.arange(
            3 * VOCAB_SIZE, device=device, dtype=torch.float32
        ).view(3, -1)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), _H2DCopies() as copies:
            info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
            vals, idxs = get_token_ids_logprobs_raw(
                logprobs,
                [[31, 1, 31], None, [2]],
                LogprobStage.DECODE,
                no_copy_to_cpu=True,
            )
        stream.synchronize()
        self.assert_pinned_copies(copies, device)
        self.assertLess(max(numel for numel, *_ in copies.copies), VOCAB_SIZE)
        expected_bias = torch.zeros(3, VOCAB_SIZE)
        expected_bias[0, 1], expected_bias[0, 31] = 2, -1
        self.assert_device_tensor(info.logit_bias, expected_bias, torch.float32)
        self.assert_device_tensor(
            info.custom_logit_processor[hash(processor)][1],
            torch.tensor([True, False, True]),
            torch.bool,
        )
        self.assertEqual(
            [v.tolist() if isinstance(v, torch.Tensor) else v for v in vals],
            [[31, 1, 31], [], [66]],
        )
        self.assertEqual(idxs, [[31, 1, 31], [], [2]])


if __name__ == "__main__":
    unittest.main()
