from __future__ import annotations

import logging
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List

import torch.cuda
import torch.distributed as dist
from torch import nn

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    format_expert_location_layout,
    format_expert_location_layout_diff,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_context, get_exec, get_model, get_parallel

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class EPLBManager:
    # A coupled placement policy (the L2 expression names its L1 half, e.g.
    # ultraep) re-plans each layer from the batch it is about to route, every
    # `eplb_rebalance_num_iterations` representative batches, instead of the
    # periodic whole-model rebalance.
    refreshes_per_layer = False
    _forward_batch = None

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        get_model: Callable[[], nn.Module],
        get_expert_location_updater: Callable[[], ExpertLocationUpdater],
        get_expert_backup_client: Callable[[], Any],
        get_weight_updater: Callable[[], Any],
        moe_load_balancer=None,
    ):
        super().__init__()
        # These collaborators are set on ModelRunner AFTER EPLBManager is
        # constructed (model load, expert_backup_client, weight_updater), so
        # they are read through getters at rebalance time, not captured here.
        self._moe_load_balancer = moe_load_balancer
        self._model_config = model_config
        self._get_model = get_model
        self._get_expert_location_updater = get_expert_location_updater
        self._get_expert_backup_client = get_expert_backup_client
        self._get_weight_updater = get_weight_updater
        self._rebalance_layers_per_chunk = (
            get_exec().moe.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = get_exec().moe.eplb_rebalance_num_iterations
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False
        self.refreshes_per_layer = (
            moe_load_balancer is not None
            and moe_load_balancer.placement_policy is not None
        )
        self._refresh_gate = None
        if self.refreshes_per_layer:
            from evenmoe.policies.l1 import RefreshGate

            self._refresh_gate = RefreshGate(self._rebalance_num_iterations)

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            get_exec().moe.eplb_rebalance_num_iterations
            >= get_exec().moe.expert_distribution_recorder_buffer_size
        ), (
            "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"
        )

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._main_generator = self._entrypoint()

    def on_forward_pass_start(self, forward_batch):
        self._forward_batch = forward_batch

    def on_forward_pass_end(self):
        self._forward_batch = None
        if not self.refreshes_per_layer:
            next(self._main_generator)

    def refresh_layer(self, layer_id, logical_topk_ids):
        """Re-plan one layer from its current TopK and commit weights + placement
        before that layer dispatches. Every EP rank must take the same branch:
        the gate, the gather and the weight move are collectives."""
        # TopK does not see the batch; ModelRunner.forward published it.
        forward_batch = self._forward_batch
        if (
            forward_batch is None
            or self._rebalance_disabled_reason is not None
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        mode = forward_batch.global_forward_mode or forward_batch.forward_mode
        # Decode-only passes (idle DP ranks included) do not count as batches.
        if (
            mode.is_decode() or mode.is_idle()
        ) and not forward_batch.is_extend_in_batch:
            return
        if not self._refresh_gate.is_due(layer_id):
            return
        counts = forward_batch.original_global_num_tokens_cpu
        if counts is None:
            counts = forward_batch.global_num_tokens_cpu
        num_tokens = (
            sum(counts)
            if counts is not None
            else forward_batch.global_num_token_non_padded_cpu
        )
        if not self._refresh_gate.should_refresh(
            layer_id, representative=num_tokens >= 512
        ):
            return

        from evenmoe.adapters.sglang import (
            commit_placement,
            to_placement_request,
            to_sglang_maps,
        )
        from evenmoe.kernels.ops.counting import count_logical_experts

        live = get_global_expert_location_metadata()
        local_count = count_logical_experts(
            logical_topk_ids, live.num_logical_experts, dtype=torch.int32
        )
        per_rank_count = local_count.new_empty((live.ep_size, live.num_logical_experts))
        dist.all_gather_into_tensor(
            per_rank_count,
            local_count.contiguous(),
            group=get_parallel().moe_ep_group.device_group,
        )
        plan = self._moe_load_balancer.plan_placement(
            to_placement_request(per_rank_count[:, None, :], context=get_context())
        )
        maps = to_sglang_maps(plan)
        layer_metadata = ExpertLocationMetadata._init_raw(
            ep_size=live.ep_size,
            physical_to_logical_map=maps.physical_to_logical_map,
            logical_to_all_physical_map=maps.logical_to_all_physical_map,
            mlb_routing_metadata={
                layer_id: value for value in maps.routing_metadata.values()
            },
        )
        self._get_expert_location_updater().update_layer(
            self._get_model().routed_experts_weights_of_layer[layer_id],
            layer_metadata,
            layer_id,
            nnodes=get_parallel().nnodes,
            rank=get_parallel().tp_rank,
        )
        commit_placement(self._moe_load_balancer, get_context(), [layer_id])

    def reset_generator(self):
        self._main_generator = self._entrypoint()

    def disable_rebalance(self, reason: str):
        self._rebalance_disabled_reason = reason
        self._rebalance_disabled_logged = False
        self.reset_generator()

    def enable_rebalance(self):
        self._rebalance_disabled_reason = None
        self._rebalance_disabled_logged = False
        self.reset_generator()

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        if self._rebalance_disabled_reason is not None:
            if not self._rebalance_disabled_logged:
                logger.debug(
                    "[EPLBManager] rebalance disabled: %s",
                    self._rebalance_disabled_reason,
                )
                self._rebalance_disabled_logged = True
            return

        elastic_state = ElasticEPStateManager.instance()
        is_post_scale_rebalance = elastic_state is not None and elastic_state.has_scaled
        # A failed later scale leaves the previously committed world serving.
        if is_post_scale_rebalance and (
            elastic_state.pending_ep_size is not None
            or elastic_state.scale_phase not in ("serving_expanded", "failed")
        ):
            return

        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        dump_record_output = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )
        logical_count = dump_record_output["logical_count"]
        average_utilization_rate_over_window = dump_record_output[
            "average_utilization_rate_over_window"
        ]

        # Check whether rebalancing is needed
        if not self._check_rebalance_needed(average_utilization_rate_over_window):
            return

        expert_location_metadata = self._compute_expert_location_metadata(
            logical_count,
            broadcast_over_world=is_post_scale_rebalance,
        )

        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            init_lplb_solvers,
        )

        on_placement_committed = None
        if self._moe_load_balancer is not None:
            from evenmoe.adapters.sglang import commit_placement

            on_placement_committed = partial(
                commit_placement, self._moe_load_balancer, get_context()
            )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        all_update_layer_ids = [
            layer_id for chunk in update_layer_ids_chunks for layer_id in chunk
        ]
        self._log_rebalance_layout_before_update(
            expert_location_metadata,
            update_layer_ids=all_update_layer_ids,
        )
        for chunk_layer_ids in update_layer_ids_chunks:
            if len(update_layer_ids_chunks) > 1:
                yield
            update_expert_location_with_recovery(
                on_placement_committed=on_placement_committed,
                expert_location_updater=self._get_expert_location_updater(),
                model=self._get_model(),
                new_expert_location_metadata=expert_location_metadata,
                update_layer_ids=chunk_layer_ids,
                tp_rank=(
                    self._elastic_global_rank()
                    if is_post_scale_rebalance
                    else get_parallel().tp_rank
                ),
                use_flat_topology=is_post_scale_rebalance,
                expert_backup_client=self._get_expert_backup_client(),
                update_weights_from_disk_callable=self._get_weight_updater().update_weights_from_disk,
                ep_dispatch_algorithm=get_exec().moe.ep_dispatch_algorithm,
                init_lplb_solvers_callable=lambda: init_lplb_solvers(
                    model_config=self._model_config
                ),
            )
            if is_post_scale_rebalance:
                # P2P waits only synchronize participating peers. Ranks without
                # moves must also install this chunk before NIXL resumes.
                dist.barrier()

        self._log_rebalance_layout_after_update(update_layer_ids=all_update_layer_ids)

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _compute_expert_location_metadata(
        self, logical_count, *, broadcast_over_world: bool
    ) -> ExpertLocationMetadata:
        if not broadcast_over_world:
            return ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
                moe_load_balancer=self._moe_load_balancer,
            )

        current_metadata = get_global_expert_location_metadata()
        assert current_metadata is not None
        # One owner prevents process-local launch topology from influencing
        # the mapping chosen for the expanded world.
        if dist.get_rank() == 0:
            computed_metadata = ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
                moe_load_balancer=self._moe_load_balancer,
                # Arbitrary append topologies may not preserve node divisibility.
                use_flat_topology=True,
            )
            physical_to_logical_map = (
                computed_metadata.physical_to_logical_map.contiguous()
            )
        else:
            physical_to_logical_map = torch.empty_like(
                current_metadata.physical_to_logical_map
            )

        dist.broadcast(physical_to_logical_map, src=0)
        return ExpertLocationMetadata.init_by_mapping(
            self._model_config,
            physical_to_logical_map,
            moe_ep_rank=self._elastic_global_rank(),
        )

    def _elastic_global_rank(self) -> int:
        return get_parallel().tp_rank + get_parallel().ep_join_rank_offset

    def _check_rebalance_needed(self, average_utilization_rate_over_window):
        if average_utilization_rate_over_window is None:
            return True

        if (
            average_utilization_rate_over_window
            > get_exec().moe.eplb_min_rebalancing_utilization_threshold
        ):
            logger.info(
                f"[EPLBManager] Skipped ep rebalancing: current GPU utilization {average_utilization_rate_over_window:.2f} > minimum rebalance threshold {get_exec().moe.eplb_min_rebalancing_utilization_threshold:.2f}"
            )
            return False

        return True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._get_model().routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _should_log_expert_location_metadata(self) -> bool:
        return (
            get_parallel().tp_rank == 0
            and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get()
        )

    def _log_rebalance_layout_before_update(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if not self._should_log_expert_location_metadata():
            return

        old_expert_location_metadata = get_global_expert_location_metadata()
        logger.info(
            "[EPLBManager] rebalance layout before:\n%s",
            format_expert_location_layout(
                old_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout target:\n%s",
            format_expert_location_layout(
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )
        logger.info(
            "[EPLBManager] rebalance layout diff:\n%s",
            format_expert_location_layout_diff(
                old_expert_location_metadata,
                new_expert_location_metadata,
                layer_ids=update_layer_ids,
            ),
        )

    def _log_rebalance_layout_after_update(self, update_layer_ids: List[int]):
        if not self._should_log_expert_location_metadata():
            return

        logger.info(
            "[EPLBManager] rebalance layout after:\n%s",
            format_expert_location_layout(
                get_global_expert_location_metadata(),
                layer_ids=update_layer_ids,
            ),
        )


def update_expert_location_with_recovery(
    *,
    expert_location_updater: ExpertLocationUpdater,
    model: nn.Module,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    tp_rank: int,
    use_flat_topology: bool = False,
    expert_backup_client,
    update_weights_from_disk_callable,
    ep_dispatch_algorithm: str,
    init_lplb_solvers_callable,
    on_placement_committed=None,
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=get_parallel().nnodes,
        rank=tp_rank,
        use_flat_topology=use_flat_topology,
    )

    if len(p2p_missing_logical_experts) > 0:
        # Load the missing expert weights from disk
        if callable(getattr(model, "generate_weight_name_filter", None)):
            # Filter and load only missing expert weights
            weight_name_filter = model.generate_weight_name_filter(
                p2p_missing_logical_experts
            )
        else:
            # Do a full reload from disk/DRAM
            logger.info(
                "[Elastic EP] Model does not implement generate_weight_name_filter. "
                "Performing full weight reload."
            )
            weight_name_filter = None

        if expert_backup_client is not None and expert_backup_client.use_backup:
            # Load the missing weights from the DRAM backup
            expert_backup_client.update_weights(weight_name_filter)
        else:
            # Load the missing weights from disk
            update_weights_from_disk_callable(
                get_model().model_path,
                get_model().load_format,
                weight_name_filter=weight_name_filter,
            )

    if on_placement_committed is not None:
        on_placement_committed(update_layer_ids)

    # Re-init LPLB solvers after expert location update
    if ep_dispatch_algorithm == "lp":
        init_lplb_solvers_callable()


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
