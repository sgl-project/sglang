import dataclasses
import enum
import typing
import unittest

import torch

from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
    _BatchedPenalizer,
    _BatchLike,
)


@dataclasses.dataclass
class MockSamplingParams:
    frequency_penalty: float = 0.0
    min_new_tokens: int = 0
    stop_token_ids: typing.List[int] = None
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


@dataclasses.dataclass
class MockTokenizer:
    eos_token_id: int
    additional_stop_token_ids: typing.Optional[typing.List[int]] = None


@dataclasses.dataclass
class MockReq:
    origin_input_ids: typing.List[int]
    sampling_params: MockSamplingParams
    tokenizer: MockTokenizer


class StepType(enum.Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclasses.dataclass
class Step:
    type: StepType
    token_ids: typing.List[int]
    expected_tensors: typing.Dict[str, torch.Tensor]
    # assume initial logits are all 1
    expected_logits: torch.Tensor


@dataclasses.dataclass
class Subject:
    sampling_params: MockSamplingParams
    # first step must be input, which will be converted to Req
    steps: typing.List[Step]
    eos_token_id: int = -1

    def __post_init__(self):
        if self.steps[0].type != StepType.INPUT:
            raise ValueError("First step must be input")

        # each steps should have the same expected_tensors.keys()
        for i in range(1, len(self.steps)):
            if self.tensor_keys(i) != self.tensor_keys():
                raise ValueError(
                    f"Expected tensors keys must be the same for all steps. Got {self.steps[i].expected_tensors.keys()} for key={i} and {self.steps[0].expected_tensors.keys()}"
                )

    def tensor_keys(self, i: int = 0) -> typing.Set[str]:
        return set(self.steps[i].expected_tensors.keys())

    def to_req(self) -> MockReq:
        return MockReq(
            origin_input_ids=self.steps[0].token_ids,
            sampling_params=self.sampling_params,
            tokenizer=MockTokenizer(eos_token_id=self.eos_token_id),
        )


@dataclasses.dataclass
class Case:
    enabled: bool
    test_subjects: typing.List[Subject]

    def __post_init__(self):
        # each test_subjects.steps should have the same expected_tensors.keys()
        for i in range(1, len(self.test_subjects)):
            if self.tensor_keys(i) != self.tensor_keys():
                raise ValueError(
                    f"Expected tensors keys must be the same for all test_subjects. Got {self.test_subjects[i].tensor_keys()} for key={i} and {self.test_subjects[0].tensor_keys()}"
                )

    def tensor_keys(self, i: int = 0) -> typing.List[str]:
        return set(self.test_subjects[i].tensor_keys())


class BaseBatchedPenalizerTest(unittest.TestCase):
    Penalizer: typing.Type[_BatchedPenalizer]
    device = "cuda"
    vocab_size = 5

    enabled: Subject = None
    disabled: Subject = None

    def setUp(self):
        if self.__class__ == BaseBatchedPenalizerTest:
            self.skipTest("Base class for penalizer tests")

        self.create_test_subjects()
        self.create_test_cases()

    def tensor(self, data, **kwargs) -> torch.Tensor:
        """
        Shortcut to create a tensor with device=self.device.
        """
        return torch.tensor(data, **kwargs, device=self.device)

    def create_test_subjects(self) -> typing.List[Subject]:
        raise NotImplementedError()

    def create_test_cases(self):
        self.test_cases = [
            Case(enabled=True, test_subjects=[self.enabled]),
            Case(enabled=False, test_subjects=[self.disabled]),
            Case(enabled=True, test_subjects=[self.enabled, self.disabled]),
        ]

    def _create_penalizer(
        self, case: Case
    ) -> typing.Tuple[BatchedPenalizerOrchestrator, _BatchedPenalizer]:
        orchestrator = BatchedPenalizerOrchestrator(
            vocab_size=self.vocab_size,
            batch=_BatchLike(reqs=[subject.to_req() for subject in case.test_subjects]),
            device=self.device,
            Penalizers={self.Penalizer},
        )

        return orchestrator, orchestrator.penalizers[self.Penalizer]

    def test_is_required(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                _, penalizer = self._create_penalizer(case)
                self.assertEqual(case.enabled, penalizer.is_required())

    def test_prepare(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                orchestrator, penalizer = self._create_penalizer(case)
                self.assertEqual(case.enabled, penalizer.is_prepared())

                if case.enabled:
                    for key, tensor in {
                        key: torch.cat(
                            tensors=[
                                subject.steps[0].expected_tensors[key]
                                for subject in case.test_subjects
                            ],
                        )
                        for key in case.tensor_keys()
                    }.items():
                        torch.testing.assert_close(
                            actual=getattr(penalizer, key),
                            expected=tensor,
                            msg=f"key={key}\nactual={getattr(penalizer, key)}\nexpected={tensor}",
                        )

                original = torch.ones(
                    size=(len(case.test_subjects), self.vocab_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                actual = orchestrator.apply(original.clone())
                expected = torch.cat(
                    tensors=[
                        subject.steps[0].expected_logits
                        for subject in case.test_subjects
                    ],
                )
                if actual is None:
                    actual = original
                torch.testing.assert_close(
                    actual=actual,
                    expected=expected,
                    msg=f"logits\nactual={actual}\nexpected={expected}",
                )

    def test_teardown(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                _, penalizer = self._create_penalizer(case)
                penalizer.teardown()

                for key in case.test_subjects[0].steps[0].expected_tensors.keys():
                    self.assertIsNone(getattr(penalizer, key, None))

    def test_filter(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                orchestrator, penalizer = self._create_penalizer(case)

                indices_to_keep = [0]
                orchestrator.filter(indices_to_keep=indices_to_keep)

                filtered_subjects = [case.test_subjects[i] for i in indices_to_keep]

                if penalizer.is_required():
                    self.assertTrue(penalizer.is_prepared())
                    for key, tensor in {
                        key: torch.cat(
                            tensors=[
                                subject.steps[0].expected_tensors[key]
                                for subject in filtered_subjects
                            ],
                        )
                        for key in case.tensor_keys()
                    }.items():
                        torch.testing.assert_close(
                            actual=getattr(penalizer, key),
                            expected=tensor,
                            msg=f"key={key}\nactual={getattr(penalizer, key)}\nexpected={tensor}",
                        )

                actual_logits = orchestrator.apply(
                    torch.ones(
                        size=(len(filtered_subjects), self.vocab_size),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                if actual_logits is None:
                    continue
                filtered_expected_logits = torch.cat(
                    tensors=[
                        subject.steps[0].expected_logits
                        for subject in filtered_subjects
                    ],
                )
                torch.testing.assert_close(
                    actual=actual_logits,
                    expected=filtered_expected_logits,
                    msg=f"logits\nactual={actual_logits}\nexpected={filtered_expected_logits}",
                )

    def test_merge_enabled_with_disabled(self):
        enabled_test_case = self.test_cases[0]
        disabled_test_case = self.test_cases[1]

        orchestrator, penalizer = self._create_penalizer(enabled_test_case)
        theirs, _ = self._create_penalizer(disabled_test_case)

        orchestrator.merge(theirs)

        for key, tensor in {
            key: torch.cat(
                tensors=[
                    enabled_test_case.test_subjects[0].steps[0].expected_tensors[key],
                    disabled_test_case.test_subjects[0].steps[0].expected_tensors[key],
                ],
            )
            for key in enabled_test_case.tensor_keys()
        }.items():
            torch.testing.assert_close(
                actual=getattr(penalizer, key),
                expected=tensor,
                msg=f"key={key}\nactual={getattr(penalizer, key)}\nexpected={tensor}",
            )

    def test_cumulate_apply_repeat(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                orchestrator, penalizer = self._create_penalizer(case)

                max_step = max(len(subject.steps) for subject in case.test_subjects)
                for i in range(1, max_step):
                    orchestrator.filter(
                        indices_to_keep=[
                            j
                            for j, subject in enumerate(case.test_subjects)
                            if i < len(subject.steps)
                        ]
                    )

                    filtered_subjects = [
                        subject
                        for subject in case.test_subjects
                        if i < len(subject.steps)
                    ]

                    inputs: typing.List[typing.List[int]] = []
                    outputs: typing.List[typing.List[int]] = []
                    for subject in filtered_subjects:
                        step = subject.steps[i]
                        if step.type == StepType.INPUT:
                            inputs.append(step.token_ids)
                            outputs.append([])
                        else:
                            inputs.append([])
                            outputs.append(step.token_ids)

                    if any(inputs):
                        orchestrator.cumulate_input_tokens(inputs)

                    if any(outputs):
                        orchestrator.cumulate_output_tokens(outputs)

                    if penalizer.is_required():
                        self.assertTrue(penalizer.is_prepared())
                        for key, tensor in {
                            key: torch.cat(
                                tensors=[
                                    subject.steps[i].expected_tensors[key]
                                    for subject in filtered_subjects
                                ],
                            )
                            for key in case.tensor_keys()
                        }.items():
                            torch.testing.assert_close(
                                actual=getattr(penalizer, key),
                                expected=tensor,
                                msg=f"key={key}\nactual={getattr(penalizer, key)}\nexpected={tensor}",
                            )

                    original = torch.ones(
                        size=(len(filtered_subjects), self.vocab_size),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    actual_logits = orchestrator.apply(original.clone())
                    filtered_expected_logits = torch.cat(
                        tensors=[
                            subject.steps[i].expected_logits
                            for subject in filtered_subjects
                        ],
                    )
                    if actual_logits is None:
                        actual_logits = original
                    torch.testing.assert_close(
                        actual=actual_logits,
                        expected=filtered_expected_logits,
                        msg=f"logits\nactual={actual_logits}\nexpected={filtered_expected_logits}",
                    )
