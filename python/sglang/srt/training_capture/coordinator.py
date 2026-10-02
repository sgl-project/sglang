"""Nonblocking admission, request collection, lease renewal and Store writing."""

from __future__ import annotations

import atexit
import hashlib
import logging
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
from sglang.srt.training_capture.admission import CaptureAdmission
from sglang.srt.training_capture.catalog import CaptureLease, CatalogConflict
from sglang.srt.training_capture.cohort_service import CaptureHandle
from sglang.srt.training_capture.config import CaptureConfig
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostSlot
from sglang.srt.training_capture.identity import (
    assemble_target_contract,
    bind_rank_target_contract,
)
from sglang.srt.training_capture.mooncake_store import TransportError
from sglang.srt.training_capture.protocol import OWNER, ContractError, Provenance
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.snapshot_writer import SnapshotWriter
from sglang.srt.training_capture.startup import (
    coordinate_capture_activation,
    coordinate_resource_startup,
    coordinate_target_startup,
)
from sglang.srt.training_capture.teacher import (
    TeacherRows,
    capture_teacher,
    warmup_teacher_capture,
)

logger = logging.getLogger(__name__)


class CaptureReservation(msgspec.Struct, eq=False):
    lease: CaptureLease
    slot: HostSlot | None
    deadline: float
    renew_at: float
    state: str = "available"
    invalid_reason: str | None = None
    context: RequestCaptureContext | None = None
    provenance: Provenance | None = None
    started: float = 0.0
    queued_at: float = 0.0
    cohort_handle: CaptureHandle | None = None
    execution_sha256: str | None = None


class CaptureStep(msgspec.Struct, frozen=True):
    request: Any
    reservation: CaptureReservation
    prediction_position: int | None


class CaptureBatch(msgspec.Struct, frozen=True):
    steps: tuple[CaptureStep, ...]


class VerifyCaptureStep(msgspec.Struct, frozen=True):
    request: Any
    reservation: CaptureReservation
    batch_row: int
    prefix_end: int
    row_start: int
    num_rows: int


class VerifyCaptureBatch(msgspec.Struct, frozen=True):
    steps: tuple[VerifyCaptureStep, ...]
    teacher: TeacherRows | None
    input_tokens: torch.Tensor
    positions: torch.Tensor
    cache_locs: torch.Tensor


class CaptureCoordinator:
    # Single-rank capture samples locally; distributed capture installs a router.
    request_router = None

    @classmethod
    def create(
        cls,
        *,
        config_path,
        model,
        model_config,
        tokenizer_path,
        pool,
        req_to_token,
        enable_overlap=False,
        metrics_labels=None,
        startup_group=None,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=0,
    ):
        if config_path is None:
            return None
        from sglang.srt.runtime_context import get_disagg, get_spec
        from sglang.srt.training_capture.metrics import CaptureMetrics

        config = None
        distributed = tp_size > 1 or pp_size > 1
        if distributed and startup_group is None:
            raise ContractError(
                "distributed capture requires its startup process group"
            )

        def build_local():
            nonlocal config
            config = CaptureConfig.load(config_path)
            contract = bind_rank_target_contract(
                model_id=config.model_id,
                selected_layer_ids=config.selected_layer_ids,
                storage_chunk_tokens=config.storage_chunk_tokens,
                expected_weights_revision=config.expected_weights_revision,
                expected_tokenizer_revision=config.expected_tokenizer_revision,
                model=model,
                model_config=model_config,
                tokenizer_path=tokenizer_path,
                pool=pool,
                tp_rank=tp_rank,
                tp_size=tp_size,
                pp_rank=pp_rank,
                pp_size=pp_size,
                dp_rank=dp_rank,
            )
            warmup_teacher_capture(
                contract.teacher.vocab_size,
                getattr(pool, "device", "cpu"),
                topk_backend=config.teacher_topk_backend,
            )
            return contract

        topology = {"tp_size": tp_size, "pp_size": pp_size, "dp_rank": dp_rank}
        if startup_group is None:
            teacher, kv, layout = assemble_target_contract([build_local()], **topology)
        else:
            teacher, kv, layout = coordinate_target_startup(
                group=startup_group, build_local=build_local, **topology
            )
        if dp_rank != 0:
            raise ContractError("DP request capture is not yet connected")
        disaggregation_mode = get_disagg().disaggregation_mode
        coordinator_type = cls
        if disaggregation_mode == "prefill":
            from sglang.srt.training_capture.pd_capture import PrefillCaptureCoordinator

            coordinator = PrefillCaptureCoordinator(
                config=config, teacher=teacher, kv=kv
            )
            atexit.register(coordinator.close)
            return coordinator
        if disaggregation_mode == "decode":
            from sglang.srt.training_capture.pd_capture import DecodeCaptureCoordinator

            coordinator_type = DecodeCaptureCoordinator
        capture_mode = (
            "speculative_accepted_target_path"
            if get_spec().speculative_algorithm == "DSPARK"
            else "autoregressive"
        )
        if disaggregation_mode == "decode":
            capture_mode = "pd_" + capture_mode

        def prepare_local():
            if distributed:
                from sglang.srt.training_capture.cohort_startup import (
                    prepare_cohort_capture,
                )

                return prepare_cohort_capture(
                    startup_group=startup_group,
                    config=config,
                    teacher=teacher,
                    kv=kv,
                    layout=layout,
                    pool=pool,
                    req_to_token=req_to_token,
                    enable_overlap=enable_overlap,
                    capture_mode=capture_mode,
                    metrics_labels=metrics_labels,
                )
            resources = CaptureResources.prepare(
                config=config,
                kv=kv,
                partition=layout.partition(OWNER),
                source_pool=pool,
            )
            try:
                if resources.exporter.device.type != "cuda":
                    raise ContractError("serving capture currently requires CUDA")
                metrics = (
                    CaptureMetrics(metrics_labels)
                    if metrics_labels is not None
                    else None
                )
                return coordinator_type(
                    config=config,
                    teacher=teacher,
                    kv=kv,
                    req_to_token=req_to_token,
                    resources=resources,
                    enable_overlap=enable_overlap,
                    metrics=metrics,
                    capture_mode=capture_mode,
                    autostart=False,
                )
            except Exception:
                resources.close()
                raise

        if startup_group is None:
            coordinator = prepare_local()
        else:
            coordinator = coordinate_resource_startup(
                group=startup_group,
                build_policy=lambda: (
                    config.startup_policy,
                    teacher,
                    kv,
                    layout,
                    capture_mode,
                    enable_overlap,
                ),
                prepare_local=prepare_local,
            )
        if distributed:
            coordinate_capture_activation(group=startup_group, coordinator=coordinator)
        else:
            coordinator.activate()
        atexit.register(coordinator.close)
        return coordinator

    def __init__(
        self,
        *,
        config,
        teacher,
        kv,
        exporter=None,
        req_to_token,
        store=None,
        catalog=None,
        pin_memory=True,
        capture_mode="autoregressive",
        enable_overlap=False,
        metrics=None,
        resources=None,
        autostart=True,
    ):
        self.config, self.teacher, self.kv = config, teacher, kv
        self.capture_mode = capture_mode
        self.enable_overlap = enable_overlap
        self.resources = resources or CaptureResources.from_connected(
            config=config,
            kv=kv,
            exporter=exporter,
            store=store,
            catalog=catalog,
            pin_memory=pin_memory,
        )
        self.exporter, self.req_to_token = self.resources.exporter, req_to_token
        self.store, self.catalog = self.resources.store, self.resources.catalog
        self.pool, self.journal = self.resources.pool, self.resources.journal
        self.writer = SnapshotWriter(self.store, self.catalog, self.journal)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.lease_wake = threading.Event()
        self.reservation_retry_at = 0.0
        self.writer_stop = threading.Event()
        self.activation = threading.Event()
        self.available = deque()
        self.records: dict[str, CaptureReservation] = {}
        self.requests: dict[str, Any] = {}
        self.work = queue.Queue(maxsize=config.max_inflight_samples)
        self.counters = Counter()
        self.disabled_reason = None
        self.admission_paused = False
        self.closed = False
        self.rng = random.Random(config.sample_seed)
        self.admission = CaptureAdmission(config.sample_ratio, config.adaptive)
        self.metrics = metrics
        self.metrics_thread = (
            threading.Thread(
                target=self._after_activation,
                args=(self._metrics_loop,),
                name="training-capture-metrics",
                daemon=True,
            )
            if metrics is not None
            else None
        )
        self.writer_thread = threading.Thread(
            target=self._after_activation,
            args=(self._writer_loop,),
            name="training-snapshot-writer",
            daemon=True,
        )
        self.lease_thread = threading.Thread(
            target=self._after_activation,
            args=(self._lease_loop,),
            name="training-capture-leases",
            daemon=True,
        )
        try:
            self.writer_thread.start()
            self.lease_thread.start()
            if self.metrics_thread is not None:
                self.metrics_thread.start()
        except Exception:
            self.close()
            raise
        if autostart:
            self.activate()

    def _after_activation(self, callback):
        self.activation.wait()
        if not self.stop.is_set():
            callback()

    def activate(self):
        if self.stop.is_set() or self.closed:
            raise ContractError("cannot activate a stopped capture coordinator")
        self.activation.set()

    def _count(self, name):
        with self.lock:
            self.counters[name] += 1

    def stats(self):
        with self.lock:
            occupancy, writer_age = self._pressure(time.monotonic())
            return {
                "counters": dict(self.counters),
                "disabled_reason": self.disabled_reason,
                "admission_paused": self.admission_paused,
                "enable_overlap": self.enable_overlap,
                "reservations": len(self.records),
                "states": dict(
                    Counter(record.state for record in self.records.values())
                ),
                "queued": self.work.qsize(),
                "occupied_fraction": occupancy,
                "writer_age_seconds": writer_age,
                "stage_timings": self.writer.timings.stats(),
                "host_pool": self.pool.stats(),
                "admission": self.admission.stats(
                    time.monotonic(),
                    disabled=self.disabled_reason is not None or self.admission_paused,
                ),
            }

    def _pressure(self, now):
        busy = [r for r in self.records.values() if r.state != "available"]
        quarantined = self.pool.stats()["quarantined"]
        writer_age = max(
            (
                now - record.queued_at
                for record in busy
                if record.state in ("queued", "writing", "pending_publication")
            ),
            default=0.0,
        )
        return (
            min(1.0, (len(busy) + quarantined) / self.config.max_inflight_samples),
            max(0.0, writer_age),
        )

    def _metrics_loop(self):
        while not self.stop.is_set():
            try:
                self.metrics.update(self.stats())
            except Exception:
                logger.exception("Training capture metrics update failed")
            self.stop.wait(1.0)

    def _admission_ratio(self):
        with self.lock:
            if self.admission_paused or self.disabled_reason is not None:
                return 0.0
            if self.admission.config is None:
                return self.config.sample_ratio
            now = time.monotonic()
            occupancy, writer_age = self._pressure(now)
            return self.admission.observe(
                now,
                occupancy=occupancy,
                writer_age_seconds=writer_age,
            )

    def _admission_failure(self, reason):
        with self.lock:
            self.admission.failure(time.monotonic(), reason)

    def _reserve(self):
        slot = self.pool.acquire()
        if slot is None:
            return False
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
            return True
        except Exception:
            self.pool.release(slot, transfer_complete=True)
            self._count("admission_catalog_error")
            self._admission_failure("catalog_error")
            self.reservation_retry_at = time.monotonic() + 0.1
            return False

    def _lease_loop(self):
        while not self.stop.is_set():
            # Clear before inspecting the pool so a concurrent release cannot
            # be lost between a failed acquire and the maintenance wait.
            self.lease_wake.clear()
            with self.lock:
                records = list(self.records.values())
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
                        self._admission_failure("lease_rejected")
                    except Exception:
                        record.renew_at = time.monotonic() + 1
                        self._count("lease_renew_error")
                        self._admission_failure("catalog_error")
            self._admission_ratio()
            with self.lock:
                can_reserve = (
                    self.disabled_reason is None
                    and not self.admission_paused
                    and time.monotonic()
                    >= max(self.admission.pause_until, self.reservation_retry_at)
                )
            if not self.stop.is_set() and can_reserve and self._reserve():
                # Recheck leases and admission between successful reservations;
                # only an idle, paused or failed refill needs the polling delay.
                continue
            if not self.stop.is_set():
                self.lease_wake.wait(0.1)

    def _queue_record(self, record):
        with self.lock:
            if record.state in ("queued", "writing", "pending_publication", "done"):
                return
            if record.state == "available" and record in self.available:
                self.available.remove(record)
            record.state = "queued"
            record.queued_at = time.monotonic()
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
        if self.admission_paused:
            self._count("excluded_paused")
            return
        ratio = self._admission_ratio()
        draw = self.rng.random()
        if draw >= ratio:
            self._count("sampled_out")
            if draw < self.config.sample_ratio:
                self._count("adaptive_sampled_out")
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
            record.provenance = self._provenance(req)
            req.training_capture_context = record
            req.training_capture_finalize = self.on_release
            self.requests[record.lease.capture_id] = req
            self._count("admitted")
        except Exception:
            record.invalid_reason = "admission_invalid_request"
            self._queue_record(record)
            self._count("admission_invalid_request")

    def _provenance(self, req, *, config_sha256=None):
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
        return Provenance(
            capture_mode=self.capture_mode,
            producer_revision=self.config.producer_revision,
            capture_config_sha256=config_sha256 or self.config.fingerprint,
            sampling_config=sampling,
            trace_id=hashlib.sha256(req.rid.encode()).hexdigest(),
        )

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
                    if context.owns_kv:
                        slots = self.req_to_token.req_to_token[
                            req.req_pool_idx, context.kv_end : end
                        ].clone()
                        context.export_kv(self.exporter, slots, end=end)
                    else:
                        context.record_kv_progress(end=end)
                    # Cached prefix positions are canonical for the validated codec;
                    # freshly computed positions are copied from the actual forward.
                    if context.owns_aux:
                        context.record_positions(
                            forward_batch.positions[offset : offset + extend_len],
                            start=end - extend_len,
                        )
                    prediction = end if end >= context.prompt_length else None
                    if self.enable_overlap and end == context.max_tokens:
                        # This lookahead forwarded the last possible output token;
                        # its prediction cannot belong to the bounded sample.
                        prediction = None
                    steps.append(CaptureStep(req, record, prediction))
                    if prediction is not None and context.owns_aux:
                        teacher_indices.append(row)
                        teacher_records.append((req, record, prediction))
                except Exception:
                    self._fail_request(req, record, "kv_export_failed")
            offset += extend_len
        if teacher_indices:
            try:
                rows = capture_teacher(
                    logits_output.next_token_logits,
                    self.teacher.vocab_size,
                    teacher_indices,
                    topk_backend=self.config.teacher_topk_backend,
                )
                for row, (req, record, position) in enumerate(teacher_records):
                    record.context.record_teacher(rows, row=row, position=position)
            except Exception:
                for req, record, _ in teacher_records:
                    if req.training_capture_context is record:
                        self._fail_request(req, record, "teacher_capture_failed")
        if steps:
            if self.enable_overlap:
                self._count("overlap_forwards")
            self._count(
                "cuda_graph_forwards" if can_run_cuda_graph else "eager_forwards"
            )
        return CaptureBatch(tuple(steps))

    def after_verify_forward(
        self,
        batch,
        forward_batch,
        logits_output,
        *,
        width,
        can_run_cuda_graph,
        verify_lens=None,
    ):
        """Own compact raw rows before grammar, penalties or rejection sampling."""
        steps = tuple(
            VerifyCaptureStep(
                req,
                req.training_capture_context,
                row,
                int(batch.seq_lens_cpu[row]),
                0,
                width,
            )
            for row, req in enumerate(batch.reqs)
            if req.training_capture_context is not None
            and int(batch.seq_lens_cpu[row])
            < req.training_capture_context.context.max_tokens
        )
        if not steps:
            return None
        try:
            owns_aux = steps[0].reservation.context.owns_aux
            if verify_lens is None:
                lengths = [width] * len(batch.reqs)
            else:
                if (
                    verify_lens.ndim != 1
                    or verify_lens.numel() != len(batch.reqs)
                    or verify_lens.dtype not in (torch.int32, torch.int64)
                ):
                    raise ContractError("invalid compact verify lengths")
                lengths = verify_lens.cpu().tolist()
            offsets = [0]
            for count in lengths:
                if not 1 <= count <= width:
                    raise ContractError("verify length exceeds its proposal window")
                offsets.append(offsets[-1] + count)
            rows = forward_batch.input_ids.numel()
            if (
                width < 1
                or offsets[-1] > rows
                or (verify_lens is None and rows != offsets[-1])
                or forward_batch.positions.numel() != rows
                or forward_batch.out_cache_loc.numel() != rows
                or (owns_aux and logits_output.next_token_logits.shape[0] < offsets[-1])
                or (
                    owns_aux
                    and verify_lens is None
                    and logits_output.next_token_logits.shape[0] != rows
                )
            ):
                raise ContractError("verify rows do not cover the request layout")
            # Compact row offsets include unselected requests. Captured offsets
            # include only selected requests and exclude graph padding entirely.
            mapped_steps, selected_indices = [], []
            for step in steps:
                count = lengths[step.batch_row]
                mapped_steps.append(
                    msgspec.structs.replace(
                        step, row_start=len(selected_indices), num_rows=count
                    )
                )
                selected_indices.extend(
                    range(offsets[step.batch_row], offsets[step.batch_row + 1])
                )
            steps = tuple(mapped_steps)
            indices = torch.tensor(
                selected_indices,
                dtype=torch.long,
                device=forward_batch.input_ids.device,
            )
            ticket = VerifyCaptureBatch(
                steps=steps,
                teacher=(
                    capture_teacher(
                        logits_output.next_token_logits,
                        self.teacher.vocab_size,
                        indices,
                        topk_backend=self.config.teacher_topk_backend,
                    )
                    if owns_aux
                    else None
                ),
                input_tokens=forward_batch.input_ids.index_select(0, indices),
                positions=forward_batch.positions.index_select(0, indices),
                cache_locs=forward_batch.out_cache_loc.index_select(0, indices),
            )
            self._count("speculative_verify_forwards")
            if self.enable_overlap:
                self._count("overlap_forwards")
            self._count(
                "cuda_graph_forwards" if can_run_cuda_graph else "eager_forwards"
            )
            return ticket
        except Exception:  # noqa: BLE001 - Capture failure must not stop serving.
            for step in steps:
                self._fail_request(
                    step.request, step.reservation, "verify_capture_failed"
                )
            return None

    def after_verify_accept(self, ticket, *, commit_lens, out_tokens):
        if ticket is None:
            return None
        steps = []
        try:
            counts = commit_lens.cpu().tolist()
            outputs = out_tokens.cpu().tolist()
            inputs = ticket.input_tokens.cpu().tolist()
            positions = ticket.positions.cpu().tolist()
        except Exception:  # noqa: BLE001 - Capture failure must not stop serving.
            for step in ticket.steps:
                self._fail_request(
                    step.request, step.reservation, "verify_commit_failed"
                )
            return None
        for step in ticket.steps:
            req, record, start = step.request, step.reservation, step.prefix_end
            if req.training_capture_context is not record:
                continue
            context = record.context
            try:
                count = counts[step.batch_row]
                row_start, row_end = step.row_start, step.row_start + step.num_rows
                input_row = inputs[row_start:row_end]
                observed_end = len(context.token_ids) + len(context.pending_tokens)
                first_overlap_anchor = (
                    self.enable_overlap
                    and start == context.prompt_length
                    and observed_end == start
                )
                if (
                    record.invalid_reason
                    or not 1 <= count <= step.num_rows
                    or start != context.kv_end
                    or (observed_end != start + 1 and not first_overlap_anchor)
                    or positions[row_start:row_end]
                    != list(range(start, start + step.num_rows))
                    or outputs[step.batch_row][: count - 1] != input_row[1:count]
                ):
                    raise ContractError(
                        "verify commit does not extend the recorded token path"
                    )
                kv_count = min(count, context.max_tokens - start)
                teacher_count = min(count, context.max_tokens - start - 1)
                context.observe_tokens(
                    position=start,
                    tokens=[input_row[0]] + outputs[step.batch_row][:count],
                )
                if context.owns_kv:
                    context.export_kv(
                        self.exporter,
                        ticket.cache_locs[row_start : row_start + kv_count],
                        end=start + kv_count,
                    )
                else:
                    context.record_kv_progress(end=start + kv_count)
                if context.owns_aux:
                    context.record_positions(
                        ticket.positions[row_start : row_start + kv_count], start=start
                    )
                if teacher_count and context.owns_aux:
                    context.record_teacher_range(
                        ticket.teacher,
                        row=row_start,
                        position=start + 1,
                        count=teacher_count,
                    )
                steps.append(CaptureStep(req, record, None))
                self._count("speculative_commits_copied")
            except Exception:  # noqa: BLE001 - Capture failure must not stop serving.
                self._fail_request(req, record, "verify_commit_failed")
        return CaptureBatch(tuple(steps))

    def _commit(self, req, record):
        context = record.context
        committed = len(context.token_ids) - context.prompt_length
        outputs = req.output_ids_through_stop
        if len(outputs) < committed:
            raise ContractError("scheduler output regressed behind committed capture")
        for index in range(committed, len(outputs)):
            context.commit_token(
                position=context.prompt_length + index,
                token_id=int(outputs[index]),
            )

    def after_result(self, ticket: CaptureBatch | None, *, requests=()):
        if self.admission.latency is not None:
            from sglang.srt.training_capture.latency import request_latency

            observed_at = time.perf_counter()
            with self.lock:
                now = time.monotonic()
                for req in requests:
                    if not req.rid.startswith(HEALTH_CHECK_RID_PREFIX):
                        ttft, tpot = request_latency(req, observed_at)
                        self.admission.latency.observe(now, ttft=ttft, tpot=tpot)
            self._admission_ratio()
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
            if self.enable_overlap or record.provenance.capture_mode in (
                "speculative_accepted_target_path",
                "pd_speculative_accepted_target_path",
            ):
                record.context.trim_terminal_prefix()
            if isinstance(req.finished_reason, FINISH_LENGTH):
                reason = "length"
            elif isinstance(req.finished_reason, FINISH_MATCHED_TOKEN):
                reason = (
                    "eos"
                    if record.context.token_ids[-1] in (req.eos_token_ids or set())
                    else "stop_token"
                )
            else:
                reason = "stop_string"
            record.context.seal(reason)
            self._count("sealed")
            self._detach(req, record)
        except Exception:
            self._fail_request(req, record, "sequence_seal_failed")

    def _set_admission_paused(self, paused):
        with self.lock:
            self.admission_paused = paused
        self.lease_wake.set()

    def control(self, action):
        """Scheduler-thread control; Store and D2H cleanup remain asynchronous."""
        if action not in ("pause", "resume", "abort"):
            raise ContractError("capture action must be pause, resume or abort")
        self._set_admission_paused(action != "resume")
        if action == "abort":
            for req in list(self.requests.values()):
                self._fail_request(
                    req, req.training_capture_context, "operator_aborted"
                )
        self._count("control_" + action)

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
        self.lease_wake.set()

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
                self.lease_wake.set()

    def _check_publication(self, record):
        # Full payload validation may outlive cancellation or the local lease.
        with self.lock:
            if record.invalid_reason or record.context.state != "SEALED":
                raise ContractError("capture invalidated before publication")
            now = time.monotonic()
            if now >= record.deadline or (
                record.started
                and now - record.started > self.config.max_capture_seconds
            ):
                raise ContractError("capture expired before publication")

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
            self.writer.timings.observe(
                "queue_wait", max(0.0, time.monotonic() - record.queued_at)
            )
            self._count("writer_started")
            complete = True
            try:
                if record.context is not None:
                    self.writer.timings.call(
                        "copy_wait", record.context.wait_for_copies
                    )
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
                    manifest, tensors = self.writer.timings.call(
                        "snapshot_build",
                        record.context.prepare_snapshot,
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
                        manifest,
                        tensors,
                        record.slot.manifest_buffer,
                        record.lease,
                        check_current=lambda record=record: self._check_publication(
                            record
                        ),
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
                self._admission_failure("writer_error")
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
        self.lease_wake.set()
        self.activation.set()
        if self.metrics_thread is not None and self.metrics_thread.ident is not None:
            self.metrics_thread.join(timeout=5)
        if self.lease_thread.ident is not None:
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
        if self.writer_thread.ident is not None:
            self.writer_thread.join(timeout=20)
        if self.writer_thread.is_alive():
            logger.error("Capture writer did not stop; retaining registered buffers")
            return
        # Client close is the transport stop barrier, including quarantined slots.
        self.resources.close()
        self.closed = True
