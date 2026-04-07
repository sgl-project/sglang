import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spectre.spectre_protocol import (
    SpectreAction,
    SpectreRequest,
    SpecType,
)
from sglang.srt.speculative.spectre.spectre_protocol import (
    is_health_check_req as _is_health_check,
)
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj

logger = logging.getLogger(__name__)


def _spectre_now_us() -> float:
    return time.time() * 1e6


class DraftCircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 30,
        cooldown_rounds: int = 100,
        tp_rank: int = 0,
    ):
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.failure_threshold = failure_threshold
        self.cooldown_rounds = cooldown_rounds
        self.rounds_in_open = 0
        self.tp_rank = tp_rank

    def should_send(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            self.rounds_in_open += 1
            if self.rounds_in_open >= self.cooldown_rounds:
                self.state = self.HALF_OPEN
                if self.tp_rank == 0:
                    logger.info(
                        "\033[34m [CircuitBreaker] OPEN -> HALF_OPEN, probing draft... \033[0m"
                    )
                return True
            return False
        if self.state == self.HALF_OPEN:
            return True
        return False

    def record_success(self):
        if self.state != self.CLOSED and self.tp_rank == 0:
            logger.info(
                f"\033[34m [CircuitBreaker] {self.state} -> CLOSED, draft recovered \033[0m"
            )
        self.consecutive_failures = 0
        self.state = self.CLOSED
        self.rounds_in_open = 0

    def record_failure(self):
        self.consecutive_failures += 1
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            self.rounds_in_open = 0
            if self.tp_rank == 0:
                logger.info(
                    "\033[34m [CircuitBreaker] HALF_OPEN -> OPEN, probe failed \033[0m"
                )
        elif self.consecutive_failures >= self.failure_threshold:
            if self.state != self.OPEN and self.tp_rank == 0:
                logger.info(
                    f"\033[34m [CircuitBreaker] CLOSED -> OPEN after "
                    f"{self.consecutive_failures} consecutive timeouts, "
                    f"cooldown {self.cooldown_rounds} rounds \033[0m"
                )
            self.state = self.OPEN
            self.rounds_in_open = 0


class SchedulerSpectreTargetMixin:
    def _init_draft_recv_infra(self):
        self._recv_timeout_s = (
            float(os.environ.get("SPECTRE_RECV_TIMEOUT_MS", "200")) / 1000.0
        )
        self._msg_buffer: List[SpectreRequest] = []
        self._msg_lock = threading.Lock()
        self._data_ready = threading.Event()
        self._bg_running = True
        self._spectre_flush_at_us = 0.0
        self._accept_reject_messages = True

        failure_threshold = int(os.environ.get("SPECTRE_FAILURE_THRESHOLD", "30"))
        cooldown_rounds = int(os.environ.get("SPECTRE_COOLDOWN_ROUNDS", "100"))
        self.draft_circuit_breaker = DraftCircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_rounds=cooldown_rounds,
            tp_rank=self.tp_rank,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            self._bg_recv_thread = threading.Thread(
                target=self._bg_recv_loop, daemon=True, name="draft_recv_bg"
            )
            self._bg_recv_thread.start()

    def _bg_recv_loop(self):
        while self._bg_running:
            try:
                if (
                    hasattr(self, "zmq_communicator")
                    and self.zmq_communicator is not None
                ):
                    msgs = self.zmq_communicator.recv_all_objs()
                    if msgs:
                        with self._msg_lock:
                            self._msg_buffer.extend(msgs)
                        self._data_ready.set()
                    else:
                        time.sleep(0.0005)
                else:
                    time.sleep(0.0005)
            except Exception as e:
                logger.error(f"\033[34m [Target][BgRecv] Error: {e} \033[0m")
                time.sleep(0.0005)

    def _drain_msg_buffer(self) -> List[SpectreRequest]:
        with self._msg_lock:
            msgs = list(self._msg_buffer)
            self._msg_buffer.clear()
        self._data_ready.clear()
        return msgs

    def reset_spectre_target_state(self) -> None:
        self._spectre_flush_at_us = _spectre_now_us()
        self._accept_reject_messages = False

        if hasattr(self, "req_to_draft_token"):
            self.req_to_draft_token.clear()

        if hasattr(self, "_msg_buffer"):
            if hasattr(self, "_msg_lock"):
                with self._msg_lock:
                    self._msg_buffer.clear()
            else:
                self._msg_buffer.clear()

        if hasattr(self, "_data_ready"):
            self._data_ready.clear()

        if hasattr(self, "draft_circuit_breaker"):
            self.draft_circuit_breaker.state = DraftCircuitBreaker.CLOSED
            self.draft_circuit_breaker.consecutive_failures = 0
            self.draft_circuit_breaker.rounds_in_open = 0

        if hasattr(self, "is_rejected"):
            self.is_rejected = False
        if hasattr(self, "rejected_forward_ct"):
            self.rejected_forward_ct = 0

    def _should_store_draft_message(self, msg: SpectreRequest) -> bool:
        rid_cache = getattr(self, "req_to_draft_token", {}).get(msg.request_id)
        if rid_cache is None or msg.spec_cnt not in rid_cache:
            return False

        flush_at_us = getattr(self, "_spectre_flush_at_us", 0.0)
        if (
            flush_at_us > 0.0
            and msg.target_send_time is not None
            and msg.target_send_time >= 0.0
            and msg.target_send_time < flush_at_us
        ):
            return False

        return True

    @DynamicGradMode()
    def event_loop_normal_spectre_target(self):
        self.req_to_draft_token: Dict[
            str, Dict[int, Optional[Tuple[List[int], List[float]]]]
        ] = defaultdict(dict)
        self.is_rejected: bool = False
        self.rejected_forward_ct: int = 0

        self._init_draft_recv_infra()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                if self._is_self_high_overhead_target(batch):
                    batch.draft_num_tokens = 1
                    batch.recv_draft_fn = None
                elif not self.draft_circuit_breaker.should_send():
                    batch.draft_num_tokens = 1
                    batch.recv_draft_fn = None
                else:
                    draft_num_tokens = self._decide_speculative_num_draft_tokens(batch)
                    self.send_batch_draft_requests(batch, draft_num_tokens)
                    batch.draft_num_tokens = self._decide_verify_num_draft_tokens(batch)
                    batch.recv_draft_fn = self.recv_drafts_for_batch
                    batch.retry_fn = self.retry_drafts_for_reqs
                    batch.retry_fail_ratio = self.server_args.spectre_retry_fail_ratio
                    batch.retry_min_count = self.server_args.spectre_retry_min_count

                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    def _collect_draft_messages(
        self,
        pending_rids: Set[str],
        pending_spec_cnts: Dict[str, int],
        timeout_s: float,
    ) -> List[SpectreRequest]:
        all_messages: List[SpectreRequest] = []
        deadline = time.perf_counter() + timeout_s

        while pending_rids:
            msgs = self._drain_msg_buffer()
            if msgs:
                all_messages.extend(msgs)
                for msg in msgs:
                    if msg.action != SpectreAction.DRAFT:
                        continue
                    if msg.request_id not in pending_rids:
                        continue
                    expected_sc = pending_spec_cnts.get(msg.request_id)
                    if expected_sc is None or msg.spec_cnt == expected_sc:
                        pending_rids.discard(msg.request_id)
                if not pending_rids:
                    break

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                if pending_rids and self.tp_rank == 0:
                    logger.info(
                        f"\033[32m [Target] Recv timeout, {len(pending_rids)} rids still pending: "
                        f"{list(pending_rids)} \033[0m"
                    )
                break
            self._data_ready.wait(timeout=remaining)

        return all_messages

    def _tp_broadcast_messages(
        self, messages: Optional[List[SpectreRequest]]
    ) -> List[SpectreRequest]:
        if self.tp_size > 1:
            return broadcast_pyobj(
                messages if messages else [],
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return messages or []

    def _store_messages(self, messages: List[SpectreRequest]) -> bool:
        has_draft = False
        for msg in messages:
            assert isinstance(
                msg, SpectreRequest
            ), f"Expected SpectreRequest, got {type(msg)}"
            if msg.action == SpectreAction.REJECT:
                if getattr(self, "_accept_reject_messages", True):
                    self.process_reject_action()
                    if self.tp_rank == 0:
                        logger.info(
                            "\033[32m [Target] Received REJECT from draft \033[0m"
                        )
            elif msg.action == SpectreAction.DRAFT:
                if not self._should_store_draft_message(msg):
                    continue
                self.req_to_draft_token[msg.request_id][msg.spec_cnt] = (
                    msg.draft_token_ids,
                    msg.draft_logprobs,
                )
                has_draft = True
                self._accept_reject_messages = True
        return has_draft

    def _build_result_from_cache(
        self, reqs: List[Req]
    ) -> Dict[str, Tuple[List[int], List[float]]]:
        result = {}
        for req in reqs:
            if _is_health_check(req):
                continue
            rid, sc = req.rid, req.spec_cnt
            rid_cache = self.req_to_draft_token.get(rid)
            if rid_cache is None:
                continue

            entry = rid_cache.get(sc)
            if entry is not None:
                result[rid] = entry
                del rid_cache[sc]

            stale_keys = [k for k in list(rid_cache.keys()) if k < sc]
            for k in stale_keys:
                del rid_cache[k]

        return result

    def _get_extend_decode_req_ids(self, batch: ScheduleBatch) -> Set[int]:
        decoding_reqs = getattr(batch, "decoding_reqs", None)
        if not decoding_reqs:
            return set()
        return {id(req) for req in decoding_reqs if not _is_health_check(req)}

    def _get_reqs_waiting_for_drafts(self, batch: ScheduleBatch) -> List[Req]:
        forward_mode = getattr(batch, "forward_mode", None)
        is_extend_batch = (
            forward_mode is not None and forward_mode.is_extend()
        ) or getattr(batch, "is_extend_in_batch", False)
        if not is_extend_batch:
            return [req for req in batch.reqs if not _is_health_check(req)]

        decoding_req_ids = self._get_extend_decode_req_ids(batch)
        return [
            req
            for req in batch.reqs
            if not _is_health_check(req)
            and (id(req) in decoding_req_ids or getattr(req, "is_chunked", 0) <= 0)
        ]

    def recv_drafts_for_batch(self, batch: ScheduleBatch) -> dict:
        reqs_waiting_for_drafts = self._get_reqs_waiting_for_drafts(batch)

        if self.tp_size == 1 or self.tp_rank == 0:
            pending_rids = {
                req.rid
                for req in reqs_waiting_for_drafts
                if req.spec_cnt in self.req_to_draft_token.get(req.rid, {})
                and self.req_to_draft_token[req.rid][req.spec_cnt] is None
            }
            messages = (
                self._collect_draft_messages(
                    pending_rids=pending_rids,
                    pending_spec_cnts={
                        req.rid: req.spec_cnt
                        for req in reqs_waiting_for_drafts
                        if req.spec_cnt in self.req_to_draft_token.get(req.rid, {})
                        and self.req_to_draft_token[req.rid][req.spec_cnt] is None
                    },
                    timeout_s=self._recv_timeout_s,
                )
                if pending_rids
                else []
            )
        else:
            messages = None

        messages = self._tp_broadcast_messages(messages)

        self._store_messages(messages)

        result = self._build_result_from_cache(reqs_waiting_for_drafts)
        if reqs_waiting_for_drafts:
            if result:
                self.draft_circuit_breaker.record_success()
            else:
                self.draft_circuit_breaker.record_failure()
        return result

    def retry_drafts_for_reqs(self, failed_reqs: List[Req]) -> dict:
        if not failed_reqs:
            return {}

        num_draft_tokens = self.server_args.speculative_num_steps + 1

        for req in failed_reqs:
            if _is_health_check(req):
                continue
            self.req_to_draft_token[req.rid][req.spec_cnt] = None

        if self.tp_size == 1 or self.tp_rank == 0:
            self._send_retry_requests(failed_reqs, max(1, num_draft_tokens - 1))

        if self.tp_size == 1 or self.tp_rank == 0:
            pending_rids = {req.rid for req in failed_reqs if not _is_health_check(req)}
            pending_spec_cnts = {
                req.rid: req.spec_cnt
                for req in failed_reqs
                if not _is_health_check(req)
            }
            messages = self._collect_draft_messages(
                pending_rids=pending_rids,
                pending_spec_cnts=pending_spec_cnts,
                timeout_s=self._recv_timeout_s * 0.5,
            )
        else:
            messages = None

        messages = self._tp_broadcast_messages(messages)

        self._store_messages(messages)
        result = self._build_result_from_cache(failed_reqs)
        return result

    def send_batch_draft_requests(
        self, batch: ScheduleBatch, speculative_num_draft_tokens: int
    ) -> None:
        if (
            self.is_rejected
            and self.server_args.spectre_reject_interval > 0
            and (
                (self.forward_ct - self.rejected_forward_ct + 1)
                % self.server_args.spectre_reject_interval
                != 0
            )
        ):
            return

        self.is_rejected = False

        reqs_to_send: List[Req] = []

        for req in batch.reqs:
            if _is_health_check(req):
                continue
            rid_cache = self.req_to_draft_token[req.rid]
            if req.spec_cnt in rid_cache:
                continue
            rid_cache[req.spec_cnt] = None
            reqs_to_send.append(req)

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                draft_reqs = []
                is_half_open = (
                    self.draft_circuit_breaker.state == DraftCircuitBreaker.HALF_OPEN
                )
                for req in reqs_to_send:
                    needs_full_context = req.spec_cnt == 0 or is_half_open
                    draft_reqs.append(
                        SpectreRequest(
                            request_id=req.rid,
                            spec_cnt=req.spec_cnt,
                            action=SpectreAction.DRAFT,
                            spec_type=SpecType.DRAFT_REQUEST,
                            input_ids=(
                                req.origin_input_ids if needs_full_context else None
                            ),
                            output_ids=req.output_ids,
                            draft_token_ids=req.cur_drafts,
                            num_draft_tokens=speculative_num_draft_tokens,
                            sampling_params=(
                                req.sampling_params if needs_full_context else None
                            ),
                            grammar=None,
                        )
                    )
                self._zmq_send(draft_reqs)

    def _send_retry_requests(
        self, failed_reqs: List[Req], num_draft_tokens: int
    ) -> None:
        if not (
            hasattr(self, "zmq_communicator") and self.zmq_communicator is not None
        ):
            return
        retry_send_time = time.perf_counter()
        reqs_to_send = [
            SpectreRequest(
                request_id=req.rid,
                spec_cnt=req.spec_cnt,
                action=SpectreAction.DRAFT,
                spec_type=SpecType.DRAFT_REQUEST,
                output_ids=req.output_ids,
                draft_token_ids=[],
                num_draft_tokens=num_draft_tokens,
                target_send_time=retry_send_time,
            )
            for req in failed_reqs
            if not _is_health_check(req)
        ]
        if reqs_to_send:
            self._zmq_send(reqs_to_send)

    def _zmq_send(self, reqs: List[SpectreRequest]) -> None:
        all_drafts_identity = self.zmq_communicator.get_all_drafts_identity()
        if not all_drafts_identity:
            logger.warning(
                "\033[32m [Target] No draft available, check draft status! \033[0m"
            )
            return
        self.zmq_communicator.send_objs(reqs, all_drafts_identity[0])

    def notify_draft_request_finished_or_aborted(
        self, req: Req, action: SpectreAction
    ) -> None:
        if _is_health_check(req):
            return

        msg = SpectreRequest(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=action,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[],
            output_ids=[],
            draft_token_ids=[],
            num_draft_tokens=0,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                self._zmq_send([msg])

        try:
            if req.rid in self.req_to_draft_token:
                del self.req_to_draft_token[req.rid]
        except Exception as e:
            if self.tp_rank == 0:
                logger.error(
                    f"\033[34m [Target][Notify] Failed to cleanup req_to_draft_token "
                    f"for {req.rid}: {e} \033[0m"
                )

    def _is_self_high_overhead_target(self, batch: ScheduleBatch) -> bool:
        current_bsz = max(batch.batch_size(), self.running_batch.batch_size())
        if current_bsz > self.server_args.spectre_max_batch_size:
            batch.is_high_overhead = True
            return True
        batch.is_high_overhead = False
        return False

    def _decide_speculative_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        return self.server_args.speculative_num_steps + 1

    def process_reject_action(self) -> None:
        self.is_rejected = True
        self.rejected_forward_ct = self.forward_ct

    def _decide_verify_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        if batch.forward_mode == ForwardMode.EXTEND:
            return self.server_args.speculative_num_draft_tokens

        if self.is_rejected:
            if self.tp_rank == 0:
                logger.info("\033[34m [Target] draft_num_tokens=1 (rejected) \033[0m")
            return 1

        no_draft_reqs = sum(
            1 for req in batch.reqs if not _is_health_check(req) and not req.cur_drafts
        )
        bs = batch.batch_size()
        if bs > 0 and no_draft_reqs / bs > self.server_args.spectre_no_draft_ratio:
            return 1
        return self.server_args.speculative_num_draft_tokens
