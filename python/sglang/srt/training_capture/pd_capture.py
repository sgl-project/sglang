"""Prefill owns a bounded teacher handoff; decode owns the complete snapshot."""

from __future__ import annotations

from collections import Counter

import msgspec
import torch
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.training_capture.cohort_coordinator import CohortCaptureCoordinator
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.pd_protocol import (
    CaptureTransferContext,
    PrefillTeacherHandoff,
    contract_digest,
    decode_handoff,
    encode_handoff,
    prompt_digest,
    sampling_digest,
)
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.teacher import TeacherRows, capture_teacher


class PrefillCaptureState(msgspec.Struct):
    context: CaptureTransferContext
    epoch: int = 0
    teacher: TeacherRows | None = None
    handoff: bytes | None = None
    failed: bool = False


class PrefillCaptureCoordinator:
    """No Store client, capture lease or full Host KV arena on the P worker."""

    def __init__(self, *, config, teacher, kv):
        self.config, self.teacher, self.kv = config, teacher, kv
        self.contract_sha256 = contract_digest(teacher, kv)
        self.capture_mode = "pd_autoregressive"
        self.disabled_reason = None
        self.admission_paused = False
        self.abort_epoch = 0
        self.counters = Counter()

    _provenance = CaptureCoordinator._provenance

    def before_forward(self, reqs):
        for req in reqs:
            if req.training_capture_attempted:
                continue
            req.training_capture_attempted = True
            if (
                self.disabled_reason
                or self.admission_paused
                or req.disagg_kv_sender is None
                or req.bootstrap_host == FAKE_BOOTSTRAP_HOST
                or req.multimodal_inputs is not None
                or req.input_embeds is not None
                or req.positional_embed_overrides is not None
                or req.session is not None
                or req.lora_id is not None
                or req.custom_logit_processor is not None
            ):
                continue
            try:
                payload = req.disagg_kv_sender.get_training_capture_context()
                if payload is None:
                    continue
                context = decode_handoff(payload, CaptureTransferContext)
                if (
                    context.bootstrap_room != req.bootstrap_room
                    or context.contract_sha256 != self.contract_sha256
                    or context.prompt_length != len(req.origin_input_ids)
                    or context.prompt_sha256 != prompt_digest(req.origin_input_ids)
                    or context.sampling_sha256
                    != sampling_digest(self._provenance(req).sampling_config)
                ):
                    raise ContractError(
                        "prefill request does not match capture context"
                    )
                req.training_capture_pd = PrefillCaptureState(
                    context, epoch=self.abort_epoch
                )
                self.counters["pd_selected"] += 1
            except Exception:  # noqa: BLE001 - Exclude the sample on failure.
                self.counters["pd_context_rejected"] += 1

    def after_forward(
        self, batch, forward_batch, logits_output, *, can_run_cuda_graph=False
    ):
        if self.disabled_reason or logits_output is None:
            return
        offset = 0
        for row, req in enumerate(batch.reqs):
            count = forward_batch.extend_seq_lens_cpu[row]
            state = req.training_capture_pd
            if self._live_state(state) and int(batch.seq_lens_cpu[row]) == len(
                req.origin_input_ids
            ):
                try:
                    end = len(req.origin_input_ids)
                    if forward_batch.positions[
                        offset : offset + count
                    ].tolist() != list(range(end - count, end)):
                        raise ContractError("prefill positions are not canonical")
                    state.teacher = capture_teacher(
                        logits_output.next_token_logits,
                        self.teacher.vocab_size,
                        torch.tensor(
                            [row], device=logits_output.next_token_logits.device
                        ),
                    )
                    state.handoff = None
                    self.counters["pd_teacher_copied"] += 1
                except Exception:  # noqa: BLE001 - Never publish a partial handoff.
                    state.failed = True
                    self.counters["pd_teacher_failed"] += 1
            offset += count

    def _encode_handoff(self, state, output_token_id):
        if state.handoff is None:
            rows = state.teacher
            if rows is None:
                raise ContractError("prefill teacher is missing")
            handoff = PrefillTeacherHandoff(
                context=state.context,
                output_token_id=output_token_id,
                topk_ids=rows.token_ids[0].cpu().tolist(),
                topk_logits=rows.logits[0].cpu().tolist(),
                logsumexp=float(rows.logsumexp[0].cpu()),
            )
            handoff.teacher_rows(self.teacher.vocab_size)
            state.handoff = encode_handoff(handoff)
            state.teacher = None
        else:
            handoff = decode_handoff(state.handoff, PrefillTeacherHandoff)
            if handoff.output_token_id != output_token_id:
                raise ContractError("PP handoff does not match the committed token")
        return state.handoff

    def _live_state(self, state):
        return (
            state is not None
            and not state.failed
            and state.epoch == self.abort_epoch
            and not self.disabled_reason
        )

    def pack_pp_handoffs(self, batch, next_token_ids):
        """Attach bounded immutable rows to the existing PP output circulation."""
        payloads = [None] * len(batch.reqs)
        for row, req in enumerate(batch.reqs):
            state = req.training_capture_pd
            if not self._live_state(state) or int(batch.seq_lens_cpu[row]) != len(
                req.origin_input_ids
            ):
                continue
            try:
                payloads[row] = self._encode_handoff(state, int(next_token_ids[row]))
            except Exception:  # noqa: BLE001 - Teacher failure cannot stop PP output.
                state.failed = True
                self.counters["pd_pp_handoff_failed"] += 1
        return tuple(payloads) if any(payloads) else None

    def accept_pp_handoffs(self, batch, payloads):
        for row, req in enumerate(batch.reqs):
            state = req.training_capture_pd
            if not self._live_state(state) or int(batch.seq_lens_cpu[row]) != len(
                req.origin_input_ids
            ):
                continue
            try:
                if not isinstance(payloads, (tuple, list)) or len(payloads) != len(
                    batch.reqs
                ):
                    raise ContractError("missing or misaligned PP teacher rows")
                payload = payloads[row]
                handoff = decode_handoff(payload, PrefillTeacherHandoff)
                if handoff.context != state.context or (
                    state.handoff is not None and state.handoff != payload
                ):
                    raise ContractError("PP teacher belongs to another capture")
                handoff.teacher_rows(self.teacher.vocab_size)
                state.handoff, state.teacher = payload, None
                self.counters["pd_pp_handoff_received"] += 1
            except Exception:  # noqa: BLE001 - The final KV send omits a bad handoff.
                state.failed = True
                self.counters["pd_pp_handoff_failed"] += 1

    def finish_handoff(self, req):
        state = req.training_capture_pd
        req.training_capture_pd = None
        if not self._live_state(state):
            return None
        try:
            payload = self._encode_handoff(state, int(req.output_ids[0]))
            self.counters["pd_handoff_ready"] += 1
            return payload
        except Exception:  # noqa: BLE001 - Missing handoff is rejected by decode.
            self.counters["pd_handoff_failed"] += 1
            return None

    def after_result(self, ticket, *, requests=()):
        pass

    def on_idle(self):
        pass

    def disable(self, reason):
        self.disabled_reason = reason

    def control(self, action):
        if action not in ("pause", "resume", "abort"):
            raise ContractError("capture action must be pause, resume or abort")
        self.admission_paused = action != "resume"
        if action == "abort":
            self.abort_epoch += 1
        self.counters["control_" + action] += 1

    def close(self):
        self.disable("closed")
        return True

    def stats(self):
        return {
            "role": "prefill_teacher",
            "disabled_reason": self.disabled_reason,
            "admission_paused": self.admission_paused,
            "counters": dict(self.counters),
        }


class DecodeCaptureMixin:
    """Import the PD boundary using the coordinator's existing lease and owners."""

    def begin_pd_transfer(self, req):
        self.before_forward([req])
        record = req.training_capture_context
        if record is None:
            return None
        try:
            lease = record.lease
            context = CaptureTransferContext(
                capture_id=lease.capture_id,
                fencing_token=lease.fencing_token,
                dataset_id=lease.dataset_id,
                sample_id=lease.sample_id,
                generation_id=lease.generation_id,
                bootstrap_room=req.bootstrap_room,
                contract_sha256=contract_digest(self.teacher, self.kv),
                prompt_sha256=prompt_digest(req.origin_input_ids),
                prompt_length=len(req.origin_input_ids),
                sampling_sha256=sampling_digest(record.provenance.sampling_config),
            )
            req.training_capture_pd = context
            return encode_handoff(context)
        except Exception:  # noqa: BLE001 - Admission must not prevent transfer.
            self._fail_request(req, record, "pd_context_failed")
            return None

    def accept_pd_handoff(self, req, payload):
        record = req.training_capture_context
        if record is None:
            return
        try:
            handoff = decode_handoff(payload, PrefillTeacherHandoff)
            if (
                handoff.context != req.training_capture_pd
                or record.invalid_reason
                or len(req.output_ids) != 1
                or handoff.output_token_id != req.output_ids[0]
            ):
                raise ContractError(
                    "PD teacher does not belong to this capture attempt"
                )
            rows = handoff.teacher_rows(self.teacher.vocab_size)
            context = record.context
            slots = self.req_to_token.req_to_token[
                req.req_pool_idx, : context.prompt_length
            ].clone()
            if context.owns_kv:
                context.export_kv(self.exporter, slots, end=context.prompt_length)
            else:
                context.record_kv_progress(end=context.prompt_length)
            if context.owns_aux:
                context.record_positions(
                    torch.arange(context.prompt_length, device=slots.device), start=0
                )
                context.record_teacher(rows, row=0, position=context.prompt_length)
            context.commit_token(
                position=context.prompt_length, token_id=handoff.output_token_id
            )
            req.training_capture_pd = None
            self._count("pd_handoff_committed")
        except Exception:  # noqa: BLE001 - Generation can proceed without a sample.
            self._fail_request(req, record, "pd_handoff_failed")

    def _detach(self, req, record):
        req.training_capture_pd = None
        return super()._detach(req, record)


class DecodeCaptureCoordinator(DecodeCaptureMixin, CaptureCoordinator):
    pass


class CohortDecodeCaptureCoordinator(DecodeCaptureMixin, CohortCaptureCoordinator):
    pass
