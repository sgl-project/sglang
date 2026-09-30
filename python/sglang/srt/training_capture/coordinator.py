"""Nonblocking admission, request collection, lease renewal and Store writing."""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import queue
import random
import threading
import time
import uuid
from collections import Counter, deque
from typing import Any

import msgspec
import torch
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.training_capture.catalog import (
    CaptureLease,
    CatalogConflict,
    HTTPCaptureCatalog,
)
from sglang.srt.training_capture.config import CaptureConfig
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool, HostSlot
from sglang.srt.training_capture.identity import bind_contract
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.protocol import OWNER, ContractError, Provenance
from sglang.srt.training_capture.snapshot_writer import (
    PublicationJournal,
    SnapshotWriter,
)
from sglang.srt.training_capture.teacher import capture_teacher

logger = logging.getLogger(__name__)


class CaptureReservation(msgspec.Struct, eq=False):
    lease: CaptureLease
    slot: HostSlot
    deadline: float
    renew_at: float
    state: str = "available"
    invalid_reason: str | None = None
    context: RequestCaptureContext | None = None
    provenance: Provenance | None = None
    started: float = 0.0


class CaptureStep(msgspec.Struct, frozen=True):
    request: Any
    reservation: CaptureReservation
    prediction_position: int | None


class CaptureBatch(msgspec.Struct, frozen=True):
    steps: tuple[CaptureStep, ...]


class CaptureCoordinator:
    @classmethod
    def create(
        cls, *, config_path, model, model_config, tokenizer_path, pool, req_to_token
    ):
        if config_path is None:
            return None
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if not isinstance(pool, MHATokenToKVPool):
            raise ContractError("training capture requires a dense MHA/GQA KV pool")
        config = CaptureConfig.load(config_path)
        teacher, kv = bind_contract(
            config=config,
            model=model,
            model_config=model_config,
            tokenizer_path=tokenizer_path,
            pool=pool,
        )
        exporter = SelectedLayerKVExporter.from_pool(kv, pool)
        if exporter.device.type != "cuda":
            raise ContractError("serving capture currently requires CUDA")
        token = (
            os.environ[config.catalog_token_env] if config.catalog_token_env else None
        )
        catalog = HTTPCaptureCatalog(
            config.catalog_endpoint,
            bearer_token=token,
            timeout=config.http_timeout_seconds,
            attempts=config.http_attempts,
        )
        store = MooncakeSnapshotStore.connect(
            msgspec.to_builtins(config.store),
            replica_num=config.replica_num,
            max_receive_bytes=config.max_host_bytes,
        )
        try:
            coordinator = cls(
                config=config,
                teacher=teacher,
                kv=kv,
                exporter=exporter,
                req_to_token=req_to_token,
                store=store,
                catalog=catalog,
            )
        except Exception:
            store.close()
            raise
        atexit.register(coordinator.close)
        return coordinator

    def __init__(
        self,
        *,
        config,
        teacher,
        kv,
        exporter,
        req_to_token,
        store,
        catalog,
        pin_memory=True,
    ):
        self.config, self.teacher, self.kv = config, teacher, kv
        self.exporter, self.req_to_token = exporter, req_to_token
        self.store, self.catalog = store, catalog
        self.pool = HostBufferPool(
            kv=kv,
            max_tokens=config.max_sample_tokens,
            slots=config.max_inflight_samples,
            max_bytes=config.max_host_bytes,
            registrar=store,
            manifest_bytes=config.manifest_buffer_bytes,
            pin_memory=pin_memory,
        )
        try:
            self.journal = PublicationJournal(config.journal_directory)
        except Exception:
            self.pool.close()
            raise
        self.writer = SnapshotWriter(store, catalog, self.journal)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.writer_stop = threading.Event()
        self.available = deque()
        self.records: dict[str, CaptureReservation] = {}
        self.requests: dict[str, Any] = {}
        self.work = queue.Queue(maxsize=config.max_inflight_samples)
        self.counters = Counter()
        self.disabled_reason = None
        self.closed = False
        self.rng = random.Random(config.sample_seed)
        self.writer_thread = threading.Thread(
            target=self._writer_loop, name="training-snapshot-writer", daemon=True
        )
        self.lease_thread = threading.Thread(
            target=self._lease_loop, name="training-capture-leases", daemon=True
        )
        self.writer_thread.start()
        self.lease_thread.start()

    def _count(self, name):
        with self.lock:
            self.counters[name] += 1

    def stats(self):
        with self.lock:
            return {
                "counters": dict(self.counters),
                "disabled_reason": self.disabled_reason,
                "reservations": len(self.records),
                "states": dict(
                    Counter(record.state for record in self.records.values())
                ),
                "queued": self.work.qsize(),
                "host_pool": self.pool.stats(),
            }

    def _reserve(self):
        slot = self.pool.acquire()
        if slot is None:
            return
        sample_id, generation_id = uuid.uuid4().hex, uuid.uuid4().hex
        started = time.monotonic()
        try:
            lease = self.catalog.begin(
                {
                    "dataset_id": self.config.dataset_id,
                    "sample_id": sample_id,
                    "generation_id": generation_id,
                    "contract_id": self.config.contract_id,
                    "teacher": msgspec.to_builtins(self.teacher),
                    "kv": msgspec.to_builtins(self.kv),
                    "owners": [OWNER],
                    "reserved_bytes": slot.storage.numel(),
                    "lease_seconds": self.config.capture_lease_seconds,
                    "idempotency_key": f"begin-{sample_id}-{generation_id}",
                }
            )
            if (lease.dataset_id, lease.sample_id, lease.generation_id) != (
                self.config.dataset_id,
                sample_id,
                generation_id,
            ):
                raise CatalogConflict("admission lease changed the requested identity")
            record = CaptureReservation(
                lease,
                slot,
                started + lease.expires_in_seconds,
                started + lease.renew_after_seconds,
            )
            with self.lock:
                self.records[lease.capture_id] = record
                self.available.append(record)
        except Exception:
            self.pool.release(slot, transfer_complete=True)
            self._count("admission_catalog_error")

    def _lease_loop(self):
        while not self.stop.is_set():
            with self.lock:
                records = list(self.records.values())
                enabled = self.disabled_reason is None
            now = time.monotonic()
            for record in records:
                if record.state == "done":
                    continue
                if now >= record.deadline or (
                    record.started
                    and now - record.started > self.config.max_capture_seconds
                ):
                    record.invalid_reason = (
                        record.invalid_reason or "capture_lease_expired"
                    )
                if record.invalid_reason:
                    if record.state == "available":
                        self._queue_record(record)
                    continue
                if now >= record.renew_at:
                    started = time.monotonic()
                    try:
                        lease = self.catalog.heartbeat(record.lease)
                        with self.lock:
                            record.lease = lease
                            record.deadline = started + lease.expires_in_seconds
                            record.renew_at = started + lease.renew_after_seconds
                    except CatalogConflict:
                        record.invalid_reason = "capture_lease_rejected"
                        self._count("lease_rejected")
                    except Exception:
                        record.renew_at = time.monotonic() + 1
                        self._count("lease_renew_error")
            if enabled:
                self._reserve()
            self.stop.wait(0.1)

    def _queue_record(self, record):
        with self.lock:
            if record.state in ("queued", "writing", "pending_publication", "done"):
                return
            if record.state == "available" and record in self.available:
                self.available.remove(record)
            record.state = "queued"
            self.work.put_nowait(record)

    def _detach(self, req, record):
        self.requests.pop(record.lease.capture_id, None)
        req.training_capture_context = None
        req.training_capture_finalize = None
        self._queue_record(record)

    def _fail_request(self, req, record, reason):
        record.invalid_reason = reason
        record.context.abort(reason)
        self._count("failed_" + reason)
        self._detach(req, record)

    def _admit(self, req):
        req.training_capture_attempted = True
        self._count("considered")
        if self.disabled_reason:
            self._count("excluded_disabled")
            return
        if self.rng.random() >= self.config.sample_ratio:
            self._count("sampled_out")
            return
        if req.rid.startswith(HEALTH_CHECK_RID_PREFIX):
            self._count("excluded_health_check")
            return
        exclusions = {
            "finished": req.finished(),
            "retracted": req.is_retracted,
            "existing_response": bool(req.output_ids),
            "lora": bool(req.lora_id),
            "multimodal": bool(req.multimodal_inputs),
            "input_embeddings": req.input_embeds is not None,
            "custom_positions": req.positional_embed_overrides is not None,
            "session": req.session is not None,
            "custom_logits": req.custom_logit_processor is not None,
        }
        for reason, excluded in exclusions.items():
            if excluded:
                self._count("excluded_" + reason)
                return
        requested = req.sampling_params.max_new_tokens
        if (
            requested is None
            or requested < 1
            or len(req.origin_input_ids) + requested > self.config.max_sample_tokens
        ):
            self._count("excluded_length")
            return
        record = None
        with self.lock:
            while self.available:
                candidate = self.available.popleft()
                if (
                    candidate.state == "available"
                    and not candidate.invalid_reason
                    and time.monotonic() < candidate.deadline
                ):
                    candidate.state = "active"
                    candidate.started = time.monotonic()
                    record = candidate
                    break
        if record is None:
            self._count("admission_backpressure")
            return
        try:
            record.context = RequestCaptureContext(
                slot=record.slot,
                prompt_ids=tuple(req.origin_input_ids),
                max_tokens=self.config.max_sample_tokens,
                vocab_size=self.teacher.vocab_size,
            )
            params = req.sampling_params
            sampling = {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "min_p": params.min_p,
                "max_new_tokens": params.max_new_tokens,
                "min_new_tokens": params.min_new_tokens,
                "frequency_penalty": params.frequency_penalty,
                "presence_penalty": params.presence_penalty,
                "repetition_penalty": params.repetition_penalty,
                "ignore_eos": params.ignore_eos,
                "logit_bias": params.logit_bias,
                "stop_token_ids": sorted(params.stop_token_ids or []),
                "stop_strs": params.stop_strs,
                "stop_regex_strs": params.stop_regex_strs,
                "grammar": {
                    "json_schema": params.json_schema,
                    "regex": params.regex,
                    "ebnf": params.ebnf,
                    "structural_tag": params.structural_tag,
                },
            }
            record.provenance = Provenance(
                capture_mode="autoregressive",
                producer_revision=self.config.producer_revision,
                capture_config_sha256=self.config.fingerprint,
                sampling_config=sampling,
                trace_id=hashlib.sha256(req.rid.encode()).hexdigest(),
            )
            req.training_capture_context = record
            req.training_capture_finalize = self.on_release
            self.requests[record.lease.capture_id] = req
            self._count("admitted")
        except Exception:
            record.invalid_reason = "admission_invalid_request"
            self._queue_record(record)
            self._count("admission_invalid_request")

    def before_forward(self, reqs):
        for req in list(self.requests.values()):
            record = req.training_capture_context
            if record.invalid_reason or req.is_retracted:
                self._fail_request(
                    req, record, record.invalid_reason or "request_retracted"
                )
            elif req.finished():
                self.on_release(req)
        for req in reqs:
            if not req.training_capture_attempted:
                self._admit(req)

    def after_forward(
        self, batch, forward_batch, logits_output, *, can_run_cuda_graph=False
    ):
        steps = []
        teacher_indices = []
        teacher_records = []
        offset = 0
        for row, req in enumerate(batch.reqs):
            extend_len = (
                forward_batch.extend_seq_lens_cpu[row]
                if batch.forward_mode.is_extend()
                else 1
            )
            record = req.training_capture_context
            if record is not None:
                context = record.context
                try:
                    if record.invalid_reason:
                        raise ContractError(record.invalid_reason)
                    end = int(batch.seq_lens_cpu[row])
                    slots = self.req_to_token.req_to_token[
                        req.req_pool_idx, context.kv_end : end
                    ].clone()
                    context.export_kv(self.exporter, slots, end=end)
                    # Cached prefix positions are canonical for the validated codec;
                    # freshly computed positions are copied from the actual forward.
                    context.record_positions(
                        forward_batch.positions[offset : offset + extend_len],
                        start=end - extend_len,
                    )
                    prediction = end if end >= context.prompt_length else None
                    steps.append(CaptureStep(req, record, prediction))
                    if prediction is not None:
                        teacher_indices.append(row)
                        teacher_records.append((req, record, prediction))
                except Exception:
                    self._fail_request(req, record, "kv_export_failed")
            offset += extend_len
        if teacher_indices:
            try:
                indices = torch.tensor(
                    teacher_indices,
                    dtype=torch.long,
                    device=logits_output.next_token_logits.device,
                )
                rows = capture_teacher(
                    logits_output.next_token_logits, self.teacher.vocab_size, indices
                )
                for row, (req, record, position) in enumerate(teacher_records):
                    record.context.record_teacher(rows, row=row, position=position)
            except Exception:
                for req, record, _ in teacher_records:
                    if req.training_capture_context is record:
                        self._fail_request(req, record, "teacher_capture_failed")
        if steps:
            self._count(
                "cuda_graph_forwards" if can_run_cuda_graph else "eager_forwards"
            )
        return CaptureBatch(tuple(steps))

    def _commit(self, req, record):
        context = record.context
        committed = len(context.token_ids) - context.prompt_length
        for index in range(committed, len(req.output_ids)):
            context.commit_token(
                position=context.prompt_length + index,
                token_id=int(req.output_ids[index]),
            )

    def after_result(self, ticket: CaptureBatch | None):
        if ticket is None:
            return
        for step in ticket.steps:
            req, record = step.request, step.reservation
            if req.training_capture_context is not record:
                continue
            if req.finished():
                self.on_release(req)
            elif req.is_retracted or record.invalid_reason:
                self._fail_request(
                    req, record, record.invalid_reason or "request_retracted"
                )
            else:
                try:
                    self._commit(req, record)
                except Exception:
                    self._fail_request(req, record, "token_alignment_failed")

    def on_release(self, req):
        record = req.training_capture_context
        if record is None:
            return
        from sglang.srt.managers.schedule_batch import (
            FINISH_ABORT,
            FINISH_LENGTH,
            FINISH_MATCHED_TOKEN,
        )

        if (
            not req.finished()
            or req.is_retracted
            or isinstance(req.finished_reason, FINISH_ABORT)
            or record.invalid_reason
        ):
            self._fail_request(
                req, record, record.invalid_reason or "request_aborted_or_retracted"
            )
            return
        try:
            self._commit(req, record)
            if isinstance(req.finished_reason, FINISH_LENGTH):
                reason = "length"
            elif isinstance(req.finished_reason, FINISH_MATCHED_TOKEN):
                reason = (
                    "eos"
                    if req.output_ids[-1] in (req.eos_token_ids or set())
                    else "stop_token"
                )
            else:
                reason = "stop_string"
            record.context.seal(reason)
            self._count("sealed")
            self._detach(req, record)
        except Exception:
            self._fail_request(req, record, "sequence_seal_failed")

    def disable(self, reason):
        with self.lock:
            self.disabled_reason = reason
        for req in list(self.requests.values()):
            self._fail_request(req, req.training_capture_context, reason)

    def on_idle(self):
        if self.work.unfinished_tasks:
            # The idle scheduler polls in Python. Yield the GIL so the writer
            # can reacquire it between CPU tensor checks and hashing calls.
            time.sleep(0)

    def _retire(self, record, *, complete):
        with self.lock:
            record.state = "done"
            self.records.pop(record.lease.capture_id, None)
        self.pool.release(record.slot, transfer_complete=complete)

    def _recover_pending(self):
        self.writer.recover()
        with self.lock:
            pending = [
                r for r in self.records.values() if r.state == "pending_publication"
            ]
        for record in pending:
            self._retire(
                record,
                complete=record.slot.storage.data_ptr() not in self.store.quarantined,
            )
        with self.lock:
            if self.disabled_reason == "publication_pending":
                self.disabled_reason = None

    def _writer_loop(self):
        # Match the CUDA worker's CPU budget in this thread's OpenMP context.
        torch.set_num_threads(1)
        recovery_at = 0.0
        while not self.writer_stop.is_set() or not self.work.empty():
            if time.monotonic() >= recovery_at:
                try:
                    self._recover_pending()
                except Exception as error:
                    with self.lock:
                        self.disabled_reason = (
                            self.disabled_reason or "publication_pending"
                        )
                    logger.warning(
                        "Training publication recovery pending: %s",
                        type(error).__name__,
                    )
                recovery_at = time.monotonic() + 5
            try:
                record = self.work.get(timeout=0.1)
            except queue.Empty:
                continue
            record.state = "writing"
            self._count("writer_started")
            complete = True
            try:
                if record.context is not None:
                    record.context.wait_for_copies()
                    self._count("copies_completed")
                if (
                    record.invalid_reason
                    or record.context is None
                    or record.context.state != "SEALED"
                ):
                    self.catalog.fail(
                        record.lease, record.invalid_reason or "capture_failed"
                    )
                else:
                    manifest, tensors = record.context.snapshot(
                        dataset_id=record.lease.dataset_id,
                        sample_id=record.lease.sample_id,
                        generation_id=record.lease.generation_id,
                        teacher=self.teacher,
                        kv=self.kv,
                        provenance=record.provenance,
                        contract_id=self.config.contract_id,
                    )
                    self._count("snapshot_built")
                    self.writer.write(
                        manifest, tensors, record.slot.manifest_buffer, record.lease
                    )
                    self._count("ready")
                    logger.info(
                        "Training sample READY: sample_id=%s generation_id=%s",
                        record.lease.sample_id,
                        record.lease.generation_id,
                    )
            except Exception as error:
                complete = not isinstance(error, TransportError) and not (
                    record.context and record.context.transfer_uncertain
                )
                self._count("writer_failed_" + type(error).__name__)
                logger.warning(
                    "Training capture write failed: sample_id=%s error=%s",
                    record.lease.sample_id,
                    type(error).__name__,
                )
                logger.debug("Training capture write details", exc_info=True)
                try:
                    pending = self.journal.has_pending(record.lease.capture_id)
                except OSError:
                    # An inaccessible journal cannot disprove a prepared publish.
                    pending = True
                if pending:
                    record.state = "pending_publication"
                    with self.lock:
                        self.disabled_reason = (
                            self.disabled_reason or "publication_pending"
                        )
                    self.work.task_done()
                    continue
                try:
                    self.catalog.fail(record.lease, "writer_" + type(error).__name__)
                except Exception:
                    self._count("failure_report_error")
            self._retire(record, complete=complete)
            self.work.task_done()

    def close(self):
        if self.closed:
            return
        self.disable("producer_shutdown")
        self.stop.set()
        self.lease_thread.join(timeout=20)
        if self.lease_thread.is_alive():
            logger.error(
                "Capture lease thread did not stop; retaining registered buffers"
            )
            return
        with self.lock:
            spare = [r for r in self.records.values() if r.state == "available"]
        for record in spare:
            record.invalid_reason = "producer_shutdown"
            self._queue_record(record)
        self.writer_stop.set()
        self.writer_thread.join(timeout=20)
        if self.writer_thread.is_alive():
            logger.error("Capture writer did not stop; retaining registered buffers")
            return
        # Client close is the transport stop barrier, including quarantined slots.
        self.store.close()
        self.journal.close()
        self.closed = True
