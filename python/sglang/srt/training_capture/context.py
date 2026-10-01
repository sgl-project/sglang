"""Request-owned position ledger, independent of mutable scheduler batches."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from sglang.srt.training_capture.host_pool import HostSlot
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.kv_staging import KVStaging
from sglang.srt.training_capture.protocol import ContractError, SequenceInfo, aux_specs
from sglang.srt.training_capture.snapshot import (
    SnapshotMetadata,
    build_snapshot,
    prepare_snapshot_partition,
)
from sglang.srt.training_capture.teacher import TeacherRows
from sglang.srt.training_capture.topology import CapturePartition


class RequestCaptureContext:
    """A single generation's accepted tokens, teacher rows, KV and completion.

    Only the inference thread mutates a collecting context. After seal/abort it
    transfers ownership to the writer, which never reads the serving request.
    This initial codec uses ordinary, zero-based full-attention positions.
    """

    def __init__(
        self,
        *,
        slot: HostSlot | None,
        prompt_ids: tuple[int, ...],
        max_tokens: int,
        vocab_size: int,
        partition: CapturePartition | None = None,
    ):
        if not prompt_ids or len(prompt_ids) >= max_tokens:
            raise ContractError(
                "capture requires a nonempty prompt and response capacity"
            )
        if any(not 0 <= token < vocab_size for token in prompt_ids):
            raise ContractError("prompt token outside target vocabulary")
        self.partition = partition
        self.owns_aux = partition is None or partition.include_aux
        self.owns_kv = partition is None or bool(partition.heads)
        active = partition is None or partition.active
        if active != (slot is not None):
            raise ContractError("context storage differs from payload ownership")
        if partition is not None and active:
            expected = (
                set(aux_specs(max_tokens, max_tokens)) if self.owns_aux else set()
            )
            for heads in partition.heads:
                for component in ("k", "v"):
                    name = f"target_{component}.{heads.layer_id}"
                    expected.add(name)
                    value = slot.tensors.get(name)
                    if (
                        value is None
                        or value.ndim != 3
                        or value.shape[0] < max_tokens
                        or value.shape[1] != heads.end - heads.start
                    ):
                        raise ContractError(
                            "request buffers differ from local KV ownership"
                        )
            if not partition.active or set(slot.tensors) != expected:
                raise ContractError("request buffers differ from partition ownership")
        self.slot = slot
        self.prompt_length = len(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.pending_tokens = {}
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.kv_end = 0
        self.teacher_rows = 0
        self.last_event = None
        self.last_stream = None
        self.kv_staging = (
            KVStaging(slot.device_tensors, slot.tensors)
            if slot is not None and slot.device_tensors is not None
            else None
        )
        self.transfer_uncertain = False
        self.state = "COLLECTING"
        self.failure_reason = None
        self.sequence = None
        if self.owns_aux:
            self.slot.tensors["position_ids"][:max_tokens].copy_(
                torch.arange(max_tokens)
            )
            self.slot.tensors["loss_mask"][:max_tokens].zero_()
            self.slot.tensors["kv_valid"][:max_tokens].zero_()

    def _collecting(self):
        if self.state != "COLLECTING":
            raise ContractError("capture is no longer collecting")

    def _sealed(self):
        if self.state != "SEALED":
            raise ContractError("only a sealed request can prepare a snapshot")

    def _record_completion(self, device):
        if device.type == "cuda":
            try:
                stream = torch.cuda.current_stream(device)
                if self.last_event is not None and stream != self.last_stream:
                    stream.wait_event(self.last_event)
                event = torch.cuda.Event()
                event.record(stream)
                self.last_event = event
                self.last_stream = stream
            except Exception:
                self.transfer_uncertain = True
                raise

    def export_kv(
        self, exporter: SelectedLayerKVExporter, slots: torch.Tensor, *, end: int
    ):
        self._collecting()
        if not self.owns_kv:
            raise ContractError("aux-only context cannot export KV payloads")
        if not self.kv_end <= end <= self.max_tokens:
            raise ContractError("KV positions retracted or exceeded capture capacity")
        if end == self.kv_end:
            return
        start = self.kv_end
        try:
            if self.kv_staging is None:
                exporter.export(slots, self.slot.tensors, start, end)
            else:
                self.kv_staging.export(exporter, slots, start=start, end=end)
        finally:
            if self.kv_staging is not None and self.kv_staging.transfer_uncertain:
                self.transfer_uncertain = True
            # Also fence copies queued before a later layer raises.
            self._record_completion(exporter.device)
        self.kv_end = end

    def record_kv_progress(self, *, end: int):
        """A rank without KV payload records its locally computed prefix."""
        self._collecting()
        if self.owns_kv or not self.kv_end <= end <= self.max_tokens:
            raise ContractError(
                "KV owners must advance through completed export enqueue"
            )
        self.kv_end = end

    def record_positions(self, positions: torch.Tensor, *, start: int):
        self._collecting()
        if not self.owns_aux:
            raise ContractError("only the aux owner records position payloads")
        end = start + positions.numel()
        if positions.ndim != 1 or not 0 <= start < end <= self.kv_end:
            raise ContractError("forward positions do not cover the exported KV range")
        try:
            self.slot.tensors["position_ids"][start:end].copy_(
                positions, non_blocking=positions.is_cuda
            )
            if positions.is_cuda:
                positions.record_stream(torch.cuda.current_stream(positions.device))
        finally:
            self._record_completion(positions.device)

    def record_teacher(self, rows: TeacherRows, *, row: int, position: int):
        if position != self.kv_end:
            raise ContractError(
                "teacher prediction is not aligned with the computed KV prefix"
            )
        self.record_teacher_range(rows, row=row, position=position, count=1)

    def record_teacher_range(
        self, rows: TeacherRows, *, row: int, position: int, count: int
    ):
        self._collecting()
        if not self.owns_aux:
            raise ContractError("only the aux owner records teacher payloads")
        if (
            count < 1
            or position != self.prompt_length + self.teacher_rows
            or position + count > self.max_tokens
            or position + count - 1 > self.kv_end
            or row < 0
            or row + count > rows.logits.shape[0]
        ):
            raise ContractError("teacher range is missing, duplicated, or shifted")
        index = self.teacher_rows
        try:
            for name, source in (
                ("teacher_topk_ids", rows.token_ids),
                ("teacher_topk_logits", rows.logits),
                ("teacher_logsumexp", rows.logsumexp),
            ):
                self.slot.tensors[name][index : index + count].copy_(
                    source[row : row + count], non_blocking=source.is_cuda
                )
                if source.is_cuda:
                    source.record_stream(torch.cuda.current_stream(source.device))
        finally:
            self._record_completion(rows.logits.device)
        self.slot.tensors["logits_positions"][index : index + count].copy_(
            torch.arange(position, position + count)
        )
        self.teacher_rows += count

    def trim_terminal_prefix(self):
        """Discard lookahead/verify suffixes past the scheduler's output boundary."""
        self._collecting()
        n = len(self.token_ids)
        response_length = n - self.prompt_length
        if (
            response_length < 1
            or (self.owns_aux and response_length > self.teacher_rows)
            or self.kv_end < n - 1
        ):
            raise ContractError("terminal output has incomplete teacher or KV coverage")
        if self.owns_aux:
            self.teacher_rows = response_length
        self.kv_end = min(self.kv_end, n)

    def commit_token(self, *, position: int, token_id: int):
        self._collecting()
        if (
            position != len(self.token_ids)
            or position >= self.max_tokens
            or position > self.kv_end
            or (self.owns_aux and position >= self.prompt_length + self.teacher_rows)
        ):
            raise ContractError("accepted token has no matching teacher row")
        if not 0 <= token_id < self.vocab_size:
            raise ContractError("accepted token outside target vocabulary")
        if (
            position in self.pending_tokens
            and self.pending_tokens[position] != token_id
        ):
            raise ContractError("scheduler token differs from the observed target path")
        self.token_ids.append(token_id)
        self.pending_tokens.pop(position, None)

    def observe_tokens(self, *, position: int, tokens: list[int]):
        """Preserve the model's path until delayed scheduler results confirm it."""
        self._collecting()
        if position < 0:
            raise ContractError("observed token position must be nonnegative")
        for offset, token in enumerate(tokens):
            index = position + offset
            if index >= self.max_tokens:
                break
            if not 0 <= token < self.vocab_size:
                raise ContractError("observed token outside target vocabulary")
            if index < len(self.token_ids):
                if self.token_ids[index] != token:
                    raise ContractError("observed path differs from committed tokens")
            elif index in self.pending_tokens:
                if self.pending_tokens[index] != token:
                    raise ContractError("observed target path changed before commit")
            elif index == len(self.token_ids) + len(self.pending_tokens):
                self.pending_tokens[index] = token
            else:
                raise ContractError("observed target path contains a gap")

    def seal(self, stop_reason: str):
        self._collecting()
        n = len(self.token_ids)
        r = n - self.prompt_length
        if (
            not r
            or (self.owns_aux and r != self.teacher_rows)
            or self.kv_end not in (n - 1, n)
        ):
            raise ContractError("cannot seal an incomplete response or KV prefix")
        self._flush_kv()
        if self.owns_aux:
            self.slot.tensors["token_ids"][:n].copy_(
                torch.tensor(self.token_ids, dtype=torch.int32)
            )
            self.slot.tensors["loss_mask"][self.prompt_length : n] = 1
            self.slot.tensors["kv_valid"][: self.kv_end] = 1
        self.sequence = SequenceInfo(
            prompt_length=self.prompt_length,
            response_length=r,
            total_length=n,
            stop_reason=stop_reason,
        )
        self.state = "SEALED"

    def _flush_kv(self):
        staging = self.kv_staging
        if staging is None or staging.start == staging.end:
            return
        # Finalization may run outside the model's forward-stream context.
        with (
            torch.cuda.stream(staging.stream)
            if staging.stream is not None
            else nullcontext()
        ):
            try:
                staging.flush()
            finally:
                self._record_completion(staging.device)

    def abort(self, reason: str):
        if self.state in ("COLLECTING", "SEALED"):
            self.state = "FAILED"
            self.failure_reason = reason

    def wait_for_copies(self):
        if self.transfer_uncertain:
            raise ContractError("CUDA copy completion is uncertain")
        if self.last_event is not None:
            try:
                self.last_event.synchronize()
            except Exception:
                self.transfer_uncertain = True
                raise

    def snapshot(self, **metadata):
        if self.partition is not None:
            raise ContractError(
                "partitioned requests require coordinated snapshot assembly"
            )
        self._sealed()
        self.wait_for_copies()
        result = build_snapshot(
            SnapshotMetadata(sequence=self.sequence, **metadata),
            self.slot.tensors,
            valid_kv_tokens=self.kv_end,
        )
        self._sealed()
        return result

    def prepare_partition(self, **metadata):
        if self.partition is None or not self.partition.active:
            raise ContractError(
                "partition preparation requires an active partitioned request"
            )
        self._sealed()
        self.wait_for_copies()
        result = prepare_snapshot_partition(
            SnapshotMetadata(sequence=self.sequence, **metadata),
            self.slot.tensors,
            valid_kv_tokens=self.kv_end,
            partition=self.partition,
            token_ids=self.token_ids,
        )
        self._sealed()
        return result
