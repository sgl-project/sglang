from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence

import torch.cuda
import torch.distributed as dist
from torch import nn

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.elastic_ep.errors import ElasticLayoutFatal
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


class ExpertLayoutDivergence(ElasticLayoutFatal):
    """A reshuffle stopped with the cohort's expert layouts possibly disagreeing.

    Separate from a clean abort, which returns False having installed nothing. Once a
    chunk is in, a rank that falls back holds a different map than its peers and routes
    tokens to the wrong expert, returning plausible garbage instead of failing. Fatal,
    like the orphan reload in _load_missing_expert_weights, and for the same reason.
    """


class EPLBManager:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
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
        # Not while the width is moving. dump_record all reduces the logical counts over
        # WORLD and broadcasts the utilization rate from src 0, and a rank that is still
        # arriving or already gone posts neither. The has_scaled arm below only covers a
        # cohort that has scaled once already, so a server's first resize would walk
        # straight into both collectives.
        if ElasticEPStateManager.is_scale_pending():
            return
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
        # Deferred, as in reshuffle_for_scale: raising mid-loop skips the chunk barrier
        # below and strands every peer in it until the vote times out.
        failure = None
        for chunk_layer_ids in update_layer_ids_chunks:
            if len(update_layer_ids_chunks) > 1:
                yield
            if failure is None:
                try:
                    update_expert_location_with_recovery(
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
                except Exception as exc:
                    logger.warning(
                        "[EPLBManager] rebalance update failed", exc_info=True
                    )
                    failure = exc
            if is_post_scale_rebalance:
                # P2P waits only synchronize participating peers. Ranks without
                # moves must also install this chunk before NIXL resumes.
                self._cohort_barrier("rebalance_chunk")

        if failure is not None:
            raise failure

        self._log_rebalance_layout_after_update(update_layer_ids=all_update_layer_ids)

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def snapshot_logical_count(self):
        """Take the recorder's logical counts while the pre-scale map is installed.

        The accumulator converts physical to logical through the metadata object a
        shrink rewrites in place, so a dump taken after the truncation scatters
        pre-shrink counts through the post-shrink map and credits donor traffic to the
        orphan logicals the shrink just moved. ``dump_record`` resets the recorder, and
        the shrink re-initializes it afterwards, so taking it early loses no counts.
        """
        try:
            return get_global_expert_distribution_recorder().dump_record(
                output_mode="object"
            )["logical_count"]
        except Exception:
            # Load-awareness is an optimization; the reshuffle re-dumps if this is None.
            logger.warning(
                "[EPLBManager] pre-scale count snapshot failed", exc_info=True
            )
            return None

    def reshuffle_for_scale(self, cohort_size: int, logical_count=None) -> bool:
        """Rebalance experts onto the post-scale cohort before serving resumes.

        Needs the pre-scale recorder's counts to stay load-aware, so ``logical_count``
        should be the snapshot taken before the metadata was truncated. Pre-checks abort
        by cohort vote because a per-rank bail strands peers in the P2P wait below."""
        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return False
        # num_local_physical_experts asserts unless ep_size divides the width, and an
        # offset joiner pins ep_size, so a shrink can leave one rank indivisible: vote.
        healthy = metadata.num_physical_experts % metadata.ep_size == 0
        if not healthy:
            logger.warning(
                "[EPLBManager] skip scale reshuffle: %d physical, ep_size=%d",
                metadata.num_physical_experts,
                metadata.ep_size,
            )
        if not self._cohort_accepts(healthy, cohort_size, "dims"):
            return False

        from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
            init_lplb_solvers,
        )

        if logical_count is None:
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
        # A scale cohort is dense: a shrink retires the top slots and a grow fills them,
        # so here the count is also the id range. The fault path takes live ids instead.
        self._share_p2l(p2l, is_owner=is_owner, cohort_ranks=range(cohort_size))
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

        for chunk_layer_ids in self._compute_update_layer_ids_chunks():
            failed = False
            try:
                update_expert_location_with_recovery(
                    expert_location_updater=self._get_expert_location_updater(),
                    model=self._get_model(),
                    new_expert_location_metadata=new_metadata,
                    update_layer_ids=chunk_layer_ids,
                    tp_rank=self._elastic_global_rank(),
                    # rank // gpus_per_node mistracks a masked partial-node shrink.
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
            # Voted per chunk, not once after the loop: a rank that gives up keeps
            # attending the rendezvous but stops posting transfers, so peers carrying
            # on into the next chunk block in the p2p wait against a rank that will
            # never post its side. Deciding here stops the cohort before those ops go
            # out. Still a rendezvous for ranks with no moves in this chunk, which
            # have to install it too (see rebalance()).
            if not self._cohort_accepts(not failed, cohort_size, "chunk"):
                # The first chunk is no cleaner than the rest. update() installs the
                # chunk's metadata after the p2p copy and before the weights reload, so
                # by the time a vote fails the ranks that passed already hold it, and
                # the one that failed holds either the old map or the new map over slots
                # it could not reload. Either way the cohort has stopped agreeing. Only
                # the two votes above this loop leave every rank on the map it came in
                # with, and those are the only clean aborts.
                raise ExpertLayoutDivergence(
                    "scale reshuffle stopped on a failed chunk vote; layout may diverge"
                )
        return True

    @staticmethod
    def _cohort_barrier(tag: str, cohort_size: Optional[int] = None) -> None:
        """Barrier over the store, scoped to the ranks still running.

        Not dist.barrier(): that is WORLD, which still counts ranks that retired and
        exited, so every survivor would block until the barrier timed out."""
        from sglang.srt.elastic_ep.elastic_ep import (
            cohort_vote_via_store,
            live_cohort_size,
        )

        if cohort_size is None:
            cohort_size = live_cohort_size()
        # Unconditional: this is a rendezvous, not a verdict. Callers surface failure
        # themselves, after every rank has crossed.
        cohort_vote_via_store(True, cohort_size, tag=f"eplb_{tag}")

    @staticmethod
    def _share_p2l(
        p2l: torch.Tensor,
        *,
        is_owner: bool,
        cohort_ranks: Optional[Sequence[int]] = None,
    ) -> None:
        """Hand rank 0's layout to the cohort, scoped to the ranks still running.

        Not dist.broadcast on the default group: that is WORLD, which keeps counting
        retirees after a shrink commits, so the owner would wait on exited ranks."""
        from sglang.srt.elastic_ep.elastic_ep import (
            live_cohort_ranks,
            share_expert_map_via_store,
        )

        if cohort_ranks is None:
            cohort_ranks = live_cohort_ranks()
        if not share_expert_map_via_store(
            p2l,
            is_src=is_owner,
            cohort_ranks=cohort_ranks,
            group_rank=dist.get_rank(),
        ):
            # No store configured: the old path, as in expert_location.py.
            dist.broadcast(p2l, src=0)

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

        Not init_by_eplb/_init_common: absent --elastic-ep-initial-size they size to the
        launch cohort, a pre-shrink width; the shape check catches peer mismatch."""
        from sglang.srt.eplb import eplb_algorithms

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
                # Peers already await the broadcast; no-op with the installed map.
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

        self._share_p2l(physical_to_logical_map, is_owner=dist.get_rank() == 0)
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


def reload_missing_expert_weights(
    missing_logical_experts,
    *,
    model: nn.Module,
    expert_backup_client,
    update_weights_from_disk_callable,
) -> None:
    """Reload logicals whose slot does not hold them from backup or disk: the p2p
    transfer trusts the installed map and cannot source them from the cohort."""
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
        success, message = update_weights_from_disk_callable(
            get_model().model_path,
            get_model().load_format,
            weight_name_filter=weight_name_filter,
        )
        if not success:
            # Discarding this verdict leaves the slot holding the expert it had
            # before, which the freshly installed map says is a different one, so the
            # rank would serve the wrong expert instead of failing. The backup path
            # above raises for the same reason.
            raise RuntimeError(
                f"[Elastic EP] failed to reload {len(missing_logical_experts)} "
                f"missing expert(s) from disk: {message}"
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
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=get_parallel().nnodes,
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
