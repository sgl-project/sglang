"""Request-owned position ledger, independent of mutable scheduler batches."""

from __future__ import annotations

import torch
from sglang.srt.training_capture.host_pool import HostSlot
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.protocol import ContractError, SequenceInfo
from sglang.srt.training_capture.snapshot import SnapshotMetadata, build_snapshot
from sglang.srt.training_capture.teacher import TeacherRows


class RequestCaptureContext:
    """A single generation's accepted tokens, teacher rows, KV and completion.

    Only the inference thread mutates a collecting context. After seal/abort it
    transfers ownership to the writer, which never reads the serving request.
    This initial codec uses ordinary, zero-based full-attention positions.
    """

    def __init__(
        self,
        *,
        slot: HostSlot,
        prompt_ids: tuple[int, ...],
        max_tokens: int,
        vocab_size: int,
    ):
        if not prompt_ids or len(prompt_ids) >= max_tokens:
            raise ContractError(
                "capture requires a nonempty prompt and response capacity"
            )
        if any(not 0 <= token < vocab_size for token in prompt_ids):
            raise ContractError("prompt token outside target vocabulary")
        self.slot = slot
        self.prompt_length = len(prompt_ids)
        self.token_ids = list(prompt_ids)
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.kv_end = 0
        self.teacher_rows = 0
        self.last_event = None
        self.transfer_uncertain = False
        self.state = "COLLECTING"
        self.failure_reason = None
        self.sequence = None
        self.slot.tensors["position_ids"][:max_tokens].copy_(torch.arange(max_tokens))
        self.slot.tensors["loss_mask"][:max_tokens].zero_()
        self.slot.tensors["kv_valid"][:max_tokens].zero_()

    def _collecting(self):
        if self.state != "COLLECTING":
            raise ContractError("capture is no longer collecting")

    def _record_completion(self, device):
        if device.type == "cuda":
            try:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                self.last_event = event
            except Exception:
                self.transfer_uncertain = True
                raise

    def export_kv(
        self, exporter: SelectedLayerKVExporter, slots: torch.Tensor, *, end: int
    ):
        self._collecting()
        if not self.kv_end <= end <= self.max_tokens:
            raise ContractError("KV positions retracted or exceeded capture capacity")
        if end == self.kv_end:
            return
        start = self.kv_end
        try:
            exporter.export(slots, self.slot.tensors, start, end)
        finally:
            # Also fence copies queued before a later layer raises.
            self._record_completion(exporter.device)
        self.kv_end = end

    def record_positions(self, positions: torch.Tensor, *, start: int):
        self._collecting()
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
        self._collecting()
        if (
            position != self.prompt_length + self.teacher_rows
            or not position < self.max_tokens
        ):
            raise ContractError("teacher row is missing, duplicated, or shifted")
        if position != self.kv_end:
            raise ContractError(
                "teacher prediction is not aligned with the computed KV prefix"
            )
        index = self.teacher_rows
        try:
            for name, source in (
                ("teacher_topk_ids", rows.token_ids),
                ("teacher_topk_logits", rows.logits),
                ("teacher_logsumexp", rows.logsumexp),
            ):
                self.slot.tensors[name][index].copy_(
                    source[row], non_blocking=source.is_cuda
                )
                if source.is_cuda:
                    source.record_stream(torch.cuda.current_stream(source.device))
        finally:
            self._record_completion(rows.logits.device)
        self.slot.tensors["logits_positions"][index] = position
        self.teacher_rows += 1

    def commit_token(self, *, position: int, token_id: int):
        self._collecting()
        if (
            position != len(self.token_ids)
            or position >= self.prompt_length + self.teacher_rows
        ):
            raise ContractError("accepted token has no matching teacher row")
        if not 0 <= token_id < self.vocab_size:
            raise ContractError("accepted token outside target vocabulary")
        self.token_ids.append(token_id)

    def seal(self, stop_reason: str):
        self._collecting()
        n = len(self.token_ids)
        r = n - self.prompt_length
        if not r or r != self.teacher_rows or self.kv_end not in (n - 1, n):
            raise ContractError("cannot seal an incomplete response or KV prefix")
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

    def abort(self, reason: str):
        if self.state == "COLLECTING":
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
        if self.state != "SEALED":
            raise ContractError("only a sealed request can form a snapshot")
        self.wait_for_copies()
        return build_snapshot(
            SnapshotMetadata(sequence=self.sequence, **metadata),
            self.slot.tensors,
            valid_kv_tokens=self.kv_end,
        )
