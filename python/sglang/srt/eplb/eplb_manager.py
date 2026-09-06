from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, List

import torch.cuda
import torch.distributed as dist
from torch import nn

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
    format_expert_location_layout,
    format_expert_location_layout_diff,
    get_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_exec, get_model, get_parallel

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        ps: Any,
        get_model: Callable[[], nn.Module],
        get_expert_location_updater: Callable[[], ExpertLocationUpdater],
        get_expert_backup_client: Callable[[], Any],
        get_weight_updater: Callable[[], Any],
    ):
        super().__init__()
        # These collaborators are set on ModelRunner AFTER EPLBManager is
        # constructed (model load, expert_backup_client, weight_updater), so
        # they are read through getters at rebalance time, not captured here.
        self._model_config = model_config
        self._ps = ps
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

    def on_forward_pass_end(self):
        next(self._main_generator)

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
            or elastic_state.scale_phase
            not in ("serving_expanded", "serving_shrunk", "failed")
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
                expert_location_updater=self._get_expert_location_updater(),
                model=self._get_model(),
                new_expert_location_metadata=expert_location_metadata,
                update_layer_ids=chunk_layer_ids,
                nnodes=get_parallel().nnodes,
                tp_rank=(
                    self._elastic_global_rank()
                    if is_post_scale_rebalance
                    else self._ps.tp_rank
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

    def reshuffle_for_scale(self, cohort_size: int) -> bool:
        """Rebalance experts onto the post-scale cohort before serving resumes.

        The recorder must still be the pre-scale one: its counts make the layout
        load-aware. Every collective below runs on every rank -- a per-rank bail strands
        peers in the P2P wait -- so pre-checks abort by cohort vote at ``cohort_size``
        and the update loop keeps barrier participation on a local failure."""
        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return False
        # num_local_physical_experts asserts unless ep_size divides the width, and an
        # offset joiner pins ep_size, leaving its own view indivisible after a later
        # shrink -- rank-local, hence the vote.
        healthy = (
            metadata.num_physical_experts >= metadata.num_logical_experts
            and metadata.num_physical_experts % metadata.ep_size == 0
        )
        if not healthy:
            logger.warning(
                "[EPLBManager] skip scale reshuffle: %d physical, %d logical, ep_size=%d",
                metadata.num_physical_experts,
                metadata.num_logical_experts,
                metadata.ep_size,
            )
        if not self._cohort_accepts(healthy, cohort_size, "dims"):
            return False

        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            init_lplb_solvers,
        )

        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]

        # One owner proposes so process-local topology cannot skew the layout.
        is_owner = dist.get_rank() == 0
        current_p2l = metadata.physical_to_logical_map
        proposed = None
        if is_owner:
            try:
                proposed = self._scaled_p2l(metadata, logical_count)
            except Exception:
                logger.warning(
                    "[EPLBManager] scale reshuffle layout failed", exc_info=True
                )
                proposed = None
        # Only the owner holds a verdict; peers abstain as yes so unanimity is its own.
        if not self._cohort_accepts(
            proposed is not None or not is_owner, cohort_size, "layout"
        ):
            return False

        p2l = proposed if is_owner else torch.empty_like(current_p2l)
        dist.broadcast(p2l, src=0)
        new_metadata = None
        try:
            new_metadata = ExpertLocationMetadata.init_by_mapping(
                self._model_config,
                p2l,
                moe_ep_rank=self._elastic_global_rank(),
            )
        except Exception:
            logger.warning(
                "[EPLBManager] scale reshuffle metadata failed", exc_info=True
            )
        if not self._cohort_accepts(new_metadata is not None, cohort_size, "metadata"):
            return False

        # Raising mid-loop would strand peers in the chunk barrier, so a local
        # failure keeps participating and is surfaced once the loop is done.
        failed = False
        for chunk_layer_ids in self._compute_update_layer_ids_chunks():
            if not failed:
                try:
                    update_expert_location_with_recovery(
                        expert_location_updater=self._get_expert_location_updater(),
                        model=self._get_model(),
                        new_expert_location_metadata=new_metadata,
                        update_layer_ids=chunk_layer_ids,
                        nnodes=get_parallel().nnodes,
                        tp_rank=self._elastic_global_rank(),
                        # rank // gpus_per_node stops tracking physical placement once a
                        # masked shrink leaves a partial node.
                        use_flat_topology=True,
                        expert_backup_client=self._get_expert_backup_client(),
                        update_weights_from_disk_callable=self._get_weight_updater().update_weights_from_disk,
                        ep_dispatch_algorithm=get_exec().moe.ep_dispatch_algorithm,
                        init_lplb_solvers_callable=lambda: init_lplb_solvers(
                            model_config=self._model_config
                        ),
                    )
                except Exception:
                    logger.warning(
                        "[EPLBManager] scale reshuffle update failed", exc_info=True
                    )
                    failed = True
            # Ranks without moves must install the chunk too (see rebalance()).
            dist.barrier()
        if failed:
            raise RuntimeError("scale reshuffle update failed; layout may diverge")
        logger.info("[EPLBManager] scale reshuffle done ep_size=%d", metadata.ep_size)
        return True

    @staticmethod
    def _cohort_accepts(ok: bool, cohort_size: int, tag: str) -> bool:
        """Make a local verdict cohort-wide so aborts stay collective."""
        from sglang.srt.elastic_ep.elastic_ep import cohort_vote_via_store

        if cohort_vote_via_store(ok, cohort_size, tag=f"reshuffle_{tag}"):
            return True
        logger.warning("[EPLBManager] scale reshuffle aborted cohort-wide (%s)", tag)
        return False

    def _scaled_p2l(self, metadata, logical_count) -> torch.Tensor:
        """Propose a layout sized from the installed metadata.

        Not via init_by_eplb/_init_common: without --elastic-ep-initial-size those size
        to the launch cohort, a pre-shrink width. Peers shape their receive buffer from
        the installed map, so a mismatch leaves logical experts with no replica
        downstream; the shape check turns that into a caught failure here."""
        from sglang.srt.eplb import eplb_algorithms

        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if logical_count.dim() == 2:
            logical_count = logical_count.unsqueeze(0)
        current_p2l = metadata.physical_to_logical_map
        num_groups = ModelConfigForExpertLocation.from_model_config(
            self._model_config
        ).num_groups
        p2l = eplb_algorithms.rebalance_experts(
            tokens_per_expert=logical_count.to(current_p2l.device),
            num_physical_experts=metadata.num_physical_experts,
            num_local_physical_experts=metadata.num_local_physical_experts,
            num_groups=num_groups,
            # An arbitrary survivor set need not divide by node.
            num_nodes=1,
            algorithm=eplb_algorithms.compute_algorithm(
                raw_algorithm=get_exec().moe.eplb_algorithm,
                num_groups=num_groups,
                num_nodes=1,
            ),
        )[0].to(current_p2l.device)
        if p2l.shape != current_p2l.shape:
            raise ValueError(f"layout {tuple(p2l.shape)} != {tuple(current_p2l.shape)}")
        return p2l.contiguous()

    def _compute_expert_location_metadata(
        self, logical_count, *, broadcast_over_world: bool
    ) -> ExpertLocationMetadata:
        if not broadcast_over_world:
            return ExpertLocationMetadata.init_by_eplb(
                self._model_config,
                logical_count,
            )

        current_metadata = get_global_expert_location_metadata()
        assert current_metadata is not None
        # One owner prevents process-local launch topology from influencing
        # the mapping chosen for the expanded world.
        if dist.get_rank() == 0:
            try:
                physical_to_logical_map = self._scaled_p2l(
                    current_metadata, logical_count
                )
            except Exception:
                # Peers are already in the broadcast below: hand them the installed map
                # so this cycle no-ops instead of diverging, and the next one retries.
                logger.warning(
                    "[EPLBManager] post-scale layout failed; skipping", exc_info=True
                )
                physical_to_logical_map = (
                    current_metadata.physical_to_logical_map.contiguous()
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
        return self._ps.tp_rank + get_parallel().ep_join_rank_offset

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
        return self._ps.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get()

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


def reload_missing_expert_weights(
    missing_logical_experts,
    *,
    model: nn.Module,
    expert_backup_client,
    update_weights_from_disk_callable,
) -> None:
    """Reload from backup/disk the logicals whose physical slot does not hold them.

    The p2p transfer reads the installed map as ground truth for what a slot holds, so
    any logical it could not source has to come from outside the cohort.
    """
    if len(missing_logical_experts) == 0:
        return

    if callable(getattr(model, "generate_weight_name_filter", None)):
        # Filter and load only missing expert weights
        weight_name_filter = model.generate_weight_name_filter(missing_logical_experts)
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


def update_expert_location_with_recovery(
    *,
    expert_location_updater: ExpertLocationUpdater,
    model: nn.Module,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    nnodes: int,
    tp_rank: int,
    use_flat_topology: bool = False,
    expert_backup_client,
    update_weights_from_disk_callable,
    ep_dispatch_algorithm: str,
    init_lplb_solvers_callable,
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=nnodes,
        rank=tp_rank,
        use_flat_topology=use_flat_topology,
    )

    reload_missing_expert_weights(
        p2p_missing_logical_experts,
        model=model,
        expert_backup_client=expert_backup_client,
        update_weights_from_disk_callable=update_weights_from_disk_callable,
    )

    # Re-init LPLB solvers after expert location update
    if ep_dispatch_algorithm == "lp":
        init_lplb_solvers_callable()


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
