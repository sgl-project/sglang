from __future__ import annotations

import abc
import weakref
from typing import TYPE_CHECKING, Optional, Set, Type

import torch

from sglang.srt.utils import get_compiler_backend, is_npu
from sglang.srt.utils.device import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

_is_npu = is_npu()


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type[_BatchedPenalizer]],
    ):
        self.vocab_size = vocab_size
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}

        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

    @property
    def batch(self) -> ScheduleBatch | None:
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        return self.batch.reqs

    def cumulate_output_tokens(
        self, output_ids: torch.Tensor, req_indices: torch.Tensor
    ):
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (torch.Tensor): The output tokens.
            req_indices (torch.Tensor): The batch row of each token.
        """
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(
                output_ids=output_ids, req_indices=req_indices
            )

    def apply(self, logits: torch.Tensor):
        """Apply all penalizers to the logits in-place."""
        for penalizer in self.penalizers.values():
            penalizer.apply(logits)

    def accumulate_additive_penalties(self, logits: torch.Tensor):
        """Apply only additive (non-multiplicative) penalizers."""
        for penalizer in self.penalizers.values():
            if not penalizer.is_multiplicative:
                penalizer.apply(logits)

    def accumulate_scaling_penalties(self) -> Optional[torch.Tensor]:
        """Accumulate all multiplicative penalty tensors into one, or None if none active."""
        result = None
        for penalizer in self.penalizers.values():
            if not penalizer._is_prepared or not penalizer.is_multiplicative:
                continue
            if result is None:
                result = penalizer.get_scaling_penalties().clone()
            else:
                result *= penalizer.get_scaling_penalties()
        return result

    def filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizers based on the indices to keep in the batch.

        Args:
            keep_indices (torch.Tensor): Tensor of indices to keep in the batch.
        """
        if not self.is_required:
            return

        if len(keep_indices) == 0:
            # No requests left in the batch, fully release orchestrator resources
            self.release()
            return

        is_required = False
        for penalizer in self.penalizers.values():
            tmp_is_required = penalizer.is_required()
            is_required |= tmp_is_required
            if tmp_is_required:
                penalizer.filter(keep_indices=keep_indices)
            else:
                penalizer.teardown()
        self.is_required = is_required

    # Resource management helpers
    def release(self) -> None:
        """Release all penalizers and break references so GC can reclaim promptly."""
        for penalizer in self.penalizers.values():
            penalizer.teardown()
        self.penalizers.clear()
        # Break reference to ScheduleBatch
        self._batch_ref = None
        self.is_required = False

    # Context manager support
    def __enter__(self) -> BatchedPenalizerOrchestrator:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def merge(self, their: BatchedPenalizerOrchestrator):
        """
        Merge the penalizers of another orchestrator into this one.

        Note that this function **must** be called _before_ self.batch.reqs is updated (filtered).
        Each unprepared penalizers would have to be prepared (creating tensors, etc.) first before merging.
        This step requires the original batch.reqs, before it gets merged with other batch.reqs.

        Args:
            their (BatchedPenalizerOrchestrator): The orchestrator to merge into this one.
        """
        if not self.is_required and not their.is_required:
            return

        self.is_required = True
        for penalizer, their_penalizer in their.penalizers.items():
            self.penalizers[penalizer].merge(their_penalizer)


class _BatchedPenalizer(abc.ABC):
    """
    An abstract class for a batched penalizer.
    """

    is_multiplicative: bool = False

    def __init__(self, orchestrator: BatchedPenalizerOrchestrator):
        self._orchestrator_ref: weakref.ReferenceType[BatchedPenalizerOrchestrator] = (
            weakref.ref(orchestrator)
        )
        self._is_prepared = False

    @property
    def orchestrator(self) -> BatchedPenalizerOrchestrator:
        orch: Optional[BatchedPenalizerOrchestrator] = self._orchestrator_ref()
        # This should never happen, but we need to handle it gracefully
        if orch is None:
            raise RuntimeError(
                "BatchedPenalizerOrchestrator has been garbage-collected"
            )
        return orch

    def is_prepared(self) -> bool:
        return self._is_prepared

    def is_required(self) -> bool:
        return self._is_required()

    def prepare(self):
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        if self._is_required():
            self.prepare()
            return True
        else:
            return False

    def teardown(self):
        self._teardown()
        self._is_prepared = False

    def cumulate_output_tokens(
        self, output_ids: torch.Tensor, req_indices: torch.Tensor
    ):
        if not self._is_prepared:
            return

        self._cumulate_output_tokens(output_ids=output_ids, req_indices=req_indices)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._is_prepared:
            return

        self._apply(logits=logits)

    def filter(self, keep_indices: torch.Tensor):
        if not self._is_prepared:
            return

        self._filter(keep_indices=keep_indices)

    def merge(self, their: _BatchedPenalizer):
        if not self._is_prepared and not their._is_prepared:
            return

        self.prepare()
        their.prepare()
        self._merge(their)

    @abc.abstractmethod
    def _is_required(self) -> bool:
        """
        Check if the penalizer is required to be prepared.
        """
        pass

    @abc.abstractmethod
    def _prepare(self):
        """
        Prepare the penalizer.
        Usually, this is where the penalizer initializes its tensors.
        """
        pass

    @abc.abstractmethod
    def _cumulate_output_tokens(
        self, output_ids: torch.Tensor, req_indices: torch.Tensor
    ):
        """
        Cumulate the output tokens; output_ids[j] belongs to batch row req_indices[j].
        Orchestrator will call this function to feed the output tokens to the penalizer.
        """
        pass

    @abc.abstractmethod
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the penalizer to the logits.
        Penalizers can modify the logits in-place if needed.
        """
        pass

    def get_scaling_penalties(self) -> torch.Tensor:
        """
        Return the accumulated scaling penalty tensor for multiplicative penalizers.
        Only meaningful when is_multiplicative is True. Subclasses should override.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizer (tensors or underlying data) based on the indices to keep in the batch.
        """
        pass

    @abc.abstractmethod
    def _merge(self, their: _BatchedPenalizer):
        """
        Merge the penalizer with another penalizer.
        """
        pass

    @abc.abstractmethod
    def _teardown(self):
        """
        Teardown the penalizer.
        """
        pass


class BatchedFrequencyPenalizer(_BatchedPenalizer):
    """
    Frequency penalizer penalizes tokens based on their frequency in the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.cumulated_frequency_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.frequency_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.frequency_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

    def _cumulate_output_tokens(
        self, output_ids: torch.Tensor, req_indices: torch.Tensor
    ):
        self.cumulated_frequency_penalties.index_put_(
            (req_indices, output_ids),
            self.frequency_penalties[req_indices, 0],
            accumulate=True,
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits.sub_(self.cumulated_frequency_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        self.frequency_penalties = self.frequency_penalties[keep_indices]
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            keep_indices
        ]

    def _merge(self, their: BatchedFrequencyPenalizer):
        self.frequency_penalties = torch.cat(
            [self.frequency_penalties, their.frequency_penalties], dim=0
        )
        self.cumulated_frequency_penalties = torch.cat(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("frequency_penalties", "cumulated_frequency_penalties"):
            if hasattr(self, name):
                delattr(self, name)


class BatchedPresencePenalizer(_BatchedPenalizer):
    """
    Presence penalizer penalizes tokens based on their presence in the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.presence_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.cumulated_presence_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.presence_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.presence_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

    def _cumulate_output_tokens(
        self, output_ids: torch.Tensor, req_indices: torch.Tensor
    ):
        self.cumulated_presence_penalties[req_indices, output_ids] = (
            self.presence_penalties[req_indices, 0]
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits.sub_(self.cumulated_presence_penalties)

    def _filter(self, keep_indices: torch.Tensor):
        self.presence_penalties = self.presence_penalties[keep_indices]
        self.cumulated_presence_penalties = self.cumulated_presence_penalties[
            keep_indices
        ]

    def _merge(self, their: BatchedPresencePenalizer):
        self.presence_penalties = torch.cat(
            [self.presence_penalties, their.presence_penalties], dim=0
        )
        self.cumulated_presence_penalties = torch.cat(
            [self.cumulated_presence_penalties, their.cumulated_presence_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("presence_penalties", "cumulated_presence_penalties"):
            if hasattr(self, name):
                delattr(self, name)


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer penalizes tokens based on their presence in the generated output.
    """

    is_multiplicative: bool = True

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )
        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

    def _cumulate_output_tokens(
        self, output_ids: torch.Tensor, req_indices: torch.Tensor
    ):
        self.cumulated_repetition_penalties[req_indices, output_ids] = (
            self.repetition_penalties[req_indices, 0]
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        apply_scaling_penalties(logits, self.cumulated_repetition_penalties)
        return logits

    def get_scaling_penalties(self) -> torch.Tensor:
        return self.cumulated_repetition_penalties

    def _filter(self, keep_indices: torch.Tensor):
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            keep_indices
        ]

    def _merge(self, their: BatchedRepetitionPenalizer):
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("repetition_penalties", "cumulated_repetition_penalties"):
            if hasattr(self, name):
                delattr(self, name)


class BatchedMinNewTokensPenalizer(_BatchedPenalizer):
    """
    Min new tokens penalizer penalizes tokens based on the length of the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.min_new_tokens > 0 for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.min_new_tokens = (
            torch.tensor(
                data=[
                    req.sampling_params.min_new_tokens
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.int32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

        vocab_size = self.orchestrator.vocab_size
        padded_stop_token_ids_cpu = torch.nn.utils.rnn.pad_sequence(
            sequences=[
                torch.tensor(
                    data=[
                        token_id
                        for token_id in (
                            (req.sampling_params.stop_token_ids or set())
                            | (req.eos_token_ids or set())
                            | (req.tokenizer.additional_stop_token_ids or set())
                            | (
                                {req.tokenizer.eos_token_id}
                                if req.tokenizer.eos_token_id is not None
                                else set()
                            )
                        )
                        # Out-of-vocab ids can never be sampled;
                        # scattering them would index outside the penalty tensor.
                        if token_id is not None and 0 <= token_id < vocab_size
                    ],
                    dtype=torch.int64,
                )
                for req in self.orchestrator.reqs()
            ],
            batch_first=True,
            padding_value=self.orchestrator.vocab_size,
        )
        if pin_memory:
            padded_stop_token_ids_cpu = padded_stop_token_ids_cpu.pin_memory()
        padded_stop_token_ids = padded_stop_token_ids_cpu.to(
            self.orchestrator.device, non_blocking=True
        )
        self.stop_token_penalties = torch.zeros(
            size=(len(self.orchestrator.reqs()), self.orchestrator.vocab_size + 1),
            dtype=torch.float32,
            device=self.orchestrator.device,
        ).scatter_add_(
            dim=1,
            index=padded_stop_token_ids,
            src=torch.full_like(
                input=padded_stop_token_ids,
                dtype=torch.float32,
                fill_value=float("-inf"),
                device=self.orchestrator.device,
            ),
        )[:, : self.orchestrator.vocab_size]

        self.len_output_tokens = torch.zeros(
            size=(len(self.orchestrator.reqs()), 1),
            dtype=torch.int32,
            device=self.orchestrator.device,
        )

    def _cumulate_output_tokens(
        self, output_ids: torch.Tensor, req_indices: torch.Tensor
    ):
        self.len_output_tokens.index_add_(
            0,
            req_indices,
            req_indices.new_ones((len(req_indices), 1), dtype=torch.int32),
        )

    def _apply(self, logits: torch.Tensor):
        # Boolean-mask indexing (logits[mask]) is data-dependent and forces a
        # device-to-host sync every decode step; torch.where is a plain
        # elementwise select with no sync (and no -inf*0=nan).
        mask = self.len_output_tokens < self.min_new_tokens
        logits.add_(torch.where(mask, self.stop_token_penalties, 0.0))

    def _filter(self, keep_indices: torch.Tensor):
        self.min_new_tokens = self.min_new_tokens[keep_indices]
        self.stop_token_penalties = self.stop_token_penalties[keep_indices]
        self.len_output_tokens = self.len_output_tokens[keep_indices]

    def _merge(self, their: BatchedMinNewTokensPenalizer):
        self.min_new_tokens = torch.cat(
            [self.min_new_tokens, their.min_new_tokens], dim=0
        )
        self.stop_token_penalties = torch.cat(
            [self.stop_token_penalties, their.stop_token_penalties], dim=0
        )
        self.len_output_tokens = torch.cat(
            [self.len_output_tokens, their.len_output_tokens], dim=0
        )

    # Explicit resource cleanup to aid GC and free CUDA memory promptly
    def _teardown(self) -> None:
        for name in ("min_new_tokens", "stop_token_penalties", "len_output_tokens"):
            if hasattr(self, name):
                delattr(self, name)
