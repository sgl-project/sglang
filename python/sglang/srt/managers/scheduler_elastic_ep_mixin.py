from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, List

import torch

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    can_recover_ranks,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import EmbeddingBatchResult, Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


class SchedulerElasticEPMixin:
    def _handle_elastic_ep_result_boundary(
        self: Scheduler,
        result: GenerationBatchResult | EmbeddingBatchResult,
    ) -> bool:
        """Handle the elastic EP result boundary before processing outputs.

        Returns False when fault/recovery retracts in-flight work and the
        caller should skip the rest of the current scheduler iteration.
        """
        elastic_ep_state = ElasticEPStateManager.instance()
        assert elastic_ep_state is not None
        if result.copy_done is not None:
            result.copy_done.synchronize()
        if elastic_ep_state.commit_active_snapshot(
            self.tp_group.active_ranks_cpu, self.tp_cpu_group
        ):
            self._publish_active_ranks_from_committed_snapshot()
            if elastic_ep_state.is_stale_snapshot():
                self._retract_all_and_rebalance_on_rank_fault()
                return False
            if self._maybe_recover_ep_ranks_from_cpu_snapshot():
                return False

        return True

    def _publish_active_ranks_from_committed_snapshot(self: Scheduler):
        elastic_ep_state = ElasticEPStateManager.instance()
        assert elastic_ep_state is not None

        self.elastic_ep_status_publisher.publish_committed_active_ranks(
            elastic_ep_state.committed_active_ranks_cpu
        )

    def _get_elastic_ep_ranks_to_recover_from_cpu_snapshot(
        self: Scheduler,
    ) -> List[int]:
        elastic_ep_state = ElasticEPStateManager.instance()
        assert elastic_ep_state is not None

        return (
            torch.nonzero(
                elastic_ep_state.committed_active_ranks_cpu == 0, as_tuple=False
            )
            .flatten()
            .tolist()
        )

    def _retract_inflight_batches_for_elastic_ep(
        self: Scheduler,
        *,
        abort_message: str,
        err_type: str,
    ) -> tuple[int, int]:
        if self.enable_overlap:
            self.result_queue.clear()
        ElasticEPStateManager.instance().clear_pending_snapshots()
        torch.cuda.synchronize()

        max_retraction = envs.SGLANG_ELASTIC_EP_MAX_RETRACTION.get()
        abort_reason = FINISH_ABORT(
            message=abort_message.format(max_retraction=max_retraction),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            err_type=err_type,
        )

        # In decode, cur_batch IS running_batch (same object). For a freshly
        # launched prefill, cur_batch is a separate batch whose reqs are not
        # yet merged into running_batch — drain both in that case.
        batches = [self.running_batch]
        if self.cur_batch is not None and self.cur_batch is not self.running_batch:
            batches.append(self.cur_batch)

        requeued_count = 0
        aborted_count = 0
        for batch in batches:
            for idx, req in enumerate(batch.reqs):
                batch.release_req(idx, 0, self.server_args)
                if req.retraction_count > max_retraction:
                    req.finished_reason = abort_reason
                    self.ipc_channels.send_to_tokenizer.send_output(
                        AbortReq(
                            finished_reason=abort_reason.to_json(),
                            rid=req.rid,
                            http_worker_ipc=req.http_worker_ipc,
                        ),
                        req,
                    )
                    aborted_count += 1
                else:
                    self._add_request_to_queue(req, is_retracted=True)
                    requeued_count += 1

        self.running_batch.filter_batch(keep_indices=[])
        self.last_batch = self.cur_batch = None
        # Clear stale pointer to a chunked-prefill req we just released; the
        # next iteration's stash_chunked_request would otherwise dereference
        # its now-None req_pool_idx.
        self.chunked_req = None
        return requeued_count, aborted_count

    def _retract_all_and_rebalance_on_rank_fault(self: Scheduler):
        """Rank fault: drain GPU, retract in-flight decode reqs, rebalance."""
        requeued_count, aborted_count = self._retract_inflight_batches_for_elastic_ep(
            abort_message=(
                "Elastic EP rank fault; aborted after {max_retraction} retractions."
            ),
            err_type="ElasticEPRankFault",
        )

        eplb_manager = self.tp_worker.model_runner.eplb_manager
        if eplb_manager is not None:
            gen = eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
        ElasticEPStateManager.instance().mark_snapshot_handled()
        logger.info(
            "Elastic EP rank fault handled. requeued=%s aborted=%s",
            requeued_count,
            aborted_count,
        )

    def _maybe_recover_ep_ranks_from_cpu_snapshot(self: Scheduler) -> bool:
        ranks_to_recover = self._get_elastic_ep_ranks_to_recover_from_cpu_snapshot()
        if not ranks_to_recover or not can_recover_ranks(ranks_to_recover):
            return False

        requeued_count, aborted_count = self._retract_inflight_batches_for_elastic_ep(
            abort_message=(
                "Elastic EP rank recovery; aborted after "
                "{max_retraction} retractions."
            ),
            err_type="ElasticEPRankRecovery",
        )
        self.tp_worker.model_runner.recover_ep_ranks_after_retract(ranks_to_recover)
        self._publish_active_ranks_from_committed_snapshot()
        logger.info(
            "Elastic EP rank recovery handled. ranks=%s requeued=%s aborted=%s",
            ranks_to_recover,
            requeued_count,
            aborted_count,
        )
        return True
