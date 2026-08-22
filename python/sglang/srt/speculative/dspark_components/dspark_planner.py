from __future__ import annotations

import logging
import time
from typing import Optional, Union

import msgspec
import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
    compute_sort_survival,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_LAG,
    FutureMap,
    ResolvedConfidence,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_disagg, get_parallel, get_schedule, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
    _interp_clamped,
    build_capture_derived_sps_table,
    build_uninitialized_sps_table,
    is_uninitialized_sps_table,
    load_sps_table_from_path,
)
from sglang.srt.speculative.dspark_components.dspark_sts import (
    load_sts_calibration_from_path,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
    round_up_grid,
)
from sglang.srt.utils.common import require_mlp_tp_gather
from sglang.srt.utils.invariants import (
    Bucket,
    InClosedRange,
    Invariant,
    IsTrue,
    expect,
    resolve_level,
)

logger = logging.getLogger(__name__)

# Bounds on the capture-time cost sweep. Spend it on PROBE DENSITY rather than on repeats: the
# lookup is a floor step function, so a sparse grid becomes wide plateaus and the chosen budget
# quantises to the end of whichever plateau it lands in, while a replay is reproducible enough that
# a median over five samples is already tight (measured spread is well under 1% of the median). A
# coarser grid also errs toward the cheaper neighbour, i.e. toward wider budgets and so toward
# today's verify-all, which is the safe direction to be wrong in.
#
# The counts are fixed rather than adaptive so that every rank performs exactly the same replays:
# captured graphs carry the model's collectives, and a rank that replays a different number of times
# hangs its peers.

# A derived curve only earns the right to trim when it predicts a win worth having. How much a
# deployment can win was read directly, by forcing each budget fraction at a fixed load through
# /set_internal_state -- that measures the objective rather than modelling it. On a Qwen3-4B target
# at concurrency 48 every fraction from 0.3 to 1.0 lands within noise of verifying everything, so
# trimming there is a small net loss; on Qwen3-14B the same sweep has a clear interior peak.
#
# The threshold separates "the model sees a real win" from "the model sees rounding". It is a regime
# separator rather than a tuned constant: swept over 0.05-0.35, the deployments that gain move by
# less than the run-to-run spread between 0.05 and 0.20, and only start giving the gain back beyond
# that. Raising it is the safe direction -- behaviour moves monotonically towards verify-all, so a
# threshold set too high forfeits gain but cannot introduce a regression.
CAPTURE_DERIVED_SPS_MIN_GAIN = 0.05
CAPTURE_DERIVED_SPS_MAX_PROBES = 40
CAPTURE_DERIVED_SPS_WARMUP = 2
CAPTURE_DERIVED_SPS_REPS = 5


def _rows_to_probes(*, rows: list[list[float]]) -> list[tuple[int, float, float]]:
    """Read back (shape, seconds, spread) rows, dropping the ones nobody measured."""
    return [
        (int(shape), seconds, spread)
        for shape, seconds, spread in rows
        if shape >= 1.0 and seconds > 0.0
    ]


# DSpark confidence is a per-token score that must stay in [0, 1].
_CONFIDENCE = Invariant(
    "dspark.planner.confidence", Bucket.GUARD, InClosedRange(0.0, 1.0)
)
# Scheduled verify lengths must not exceed the per-step token budget.
_VERIFY_LEN_BUDGET = Invariant("dspark.verify_len_budget", Bucket.GUARD, IsTrue())


class VerifyWindow(msgspec.Struct, frozen=True):
    positions_2d: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_cache_loc_2d: torch.Tensor


class DSparkVerifyPlanner:
    def __init__(
        self,
        *,
        draft_model,
        gamma: int,
        model_runner,
        device,
        tp_rank: int,
        server_args: ServerArgs,
        verify_num_draft_tokens: int,
    ) -> None:
        self.draft_model = draft_model
        self.gamma = gamma
        self.model_runner = model_runner
        self.device = device
        self.tp_rank = tp_rank
        self.server_args = server_args
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self._align_verify_tokens_to_graph_tier = (
            server_args.speculative_dspark_align_verify_tokens_to_graph_tier
        )

        self._confidence_head = getattr(self.draft_model, "confidence_head", None)

        sts_path = server_args.speculative_dspark_confidence_sts_path
        if sts_path and self._confidence_head is not None:
            calibration = load_sts_calibration_from_path(sts_path)
            sts_temperatures = torch.tensor(
                calibration.temperatures, dtype=torch.float32, device=device
            )
            if envs.SGLANG_DSPARK_STS_COLLECT_PATH.get() and not bool(
                torch.all(sts_temperatures == 1.0)
            ):
                raise ValueError(
                    "DSpark STS data collection (SGLANG_DSPARK_STS_COLLECT_PATH) "
                    "requires identity temperatures, but a non-identity calibration "
                    f"was loaded from {sts_path}. Collect pre-calibration logits with "
                    "no table (omit --speculative-dspark-confidence-sts-path)."
                )
            if sts_temperatures.numel() != self.gamma:
                raise ValueError(
                    "DSpark STS calibration was fit for gamma="
                    f"{sts_temperatures.numel()} but the runtime gamma is "
                    f"{self.gamma}; refit the table for gamma={self.gamma} or omit "
                    "--speculative-dspark-confidence-sts-path."
                )
            self._confidence_head.sts_temperatures = sts_temperatures
            if tp_rank == 0:
                logger.info(
                    "DSpark STS calibration loaded from %s (gamma=%d); per-position "
                    "temperatures applied to confidence-head survival.",
                    sts_path,
                    self.gamma,
                )
        elif sts_path and self._confidence_head is None:
            if tp_rank == 0:
                logger.warning(
                    "DSpark STS calibration path given but no confidence head present "
                    "(static mode / head-less checkpoint); ignoring %s.",
                    sts_path,
                )

        self._ragged_verify_mode = read_ragged_verify_mode()
        self._schedule_cfg = DSparkScheduleConfig(gamma=self.gamma)
        self._budget_planner: Optional[HostConfidenceBudgetPlanner] = None
        self._dynamic_graph_tier = False
        self._dp_tier_gather_enabled = False
        self._is_verify_all = True
        self._uniform_layout_cache: dict = {}
        # Set only when this planner installs a table it measured itself. The full-window fast path
        # below keys off it so that a deployment which did not opt in -- including one running a
        # profiled table from disk -- keeps exactly the scheduling path it has today.
        self._derived_sps_installed = False
        # Assigned before the mode branch: install_capture_derived_sps_table() is called
        # unconditionally after capture, including in static mode where the branch below never runs.
        self._derive_sps_at_capture = False
        if self._ragged_verify_mode is not RaggedVerifyMode.STATIC:
            if self._confidence_head is None:
                raise ValueError(
                    f"DSpark ragged-verify mode {self._ragged_verify_mode.value!r} "
                    f"schedules per-request verify lengths from the draft confidence "
                    f"head, but this DSpark draft checkpoint has no confidence head -- "
                    f"the checkpoint is wrong/incomplete (it ships no "
                    f"enable_confidence_head + trained confidence_head weights). Use a "
                    f"draft checkpoint that includes the confidence head, or run "
                    f"SGLANG_RAGGED_VERIFY_MODE=static."
                )
            sps_table = build_sps_cost_table(
                server_args=self.server_args,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
            )
            self._is_verify_all = (
                self._ragged_verify_mode is RaggedVerifyMode.COMPACT
                and is_uninitialized_sps_table(sps_table)
            )
            # A cost model can be measured from the verify graphs the compact path already
            # captures, which happens after this constructor runs. Decide here, install later.
            #
            # --speculative-dspark-sps-table-path wins, and does so by construction rather than by
            # an explicit branch: a table loaded from disk is not uninitialized, so
            # is_uninitialized_sps_table is False, _is_verify_all is False, and the derivation is
            # never armed. An operator who pinned a curve keeps exactly that curve.
            #
            # Compact only, via _is_verify_all above, and deliberately: cap-accept runs its target
            # forward at full uniform width and its budget only caps acceptance afterwards, so a
            # cost curve there could lose accepted tokens without saving any compute.
            # Excluded when a simulated acceptance length is forcing a verify-all schedule: the
            # validation below keys off is_verify_all, so deriving a table would silently
            # invalidate a check that has already passed.
            self._derive_sps_at_capture = (
                self._is_verify_all
                and envs.SGLANG_DSPARK_ENABLE_CAPTURE_DERIVED_SPS.get()
                and not simulate_acc_len_needs_verify_all()
            )
            relay_lag_steps = (
                0
                if get_schedule().disable_overlap_schedule
                else CONFIDENCE_RELAY_RING_LAG
            )
            self._budget_planner = HostConfidenceBudgetPlanner(
                sps_table=sps_table,
                cfg=self._schedule_cfg,
                model_runner=self.model_runner,
                relay_lag_steps=relay_lag_steps,
            )
            self._dynamic_graph_tier = not is_dp_attention_enabled()
            self._dp_tier_gather_enabled = (
                self._ragged_verify_mode is RaggedVerifyMode.COMPACT
                and is_dp_attention_enabled()
                and get_parallel().attn_tp_size == 1
                and get_parallel().attn_cp_size == 1
                and require_mlp_tp_gather(self.server_args)
                and not get_schedule().disable_overlap_schedule
                and not get_spec().speculative_skip_dp_mlp_sync
                and get_disagg().disaggregation_mode == "null"
                and get_parallel().pp_size == 1
                and not envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
            )
            if tp_rank == 0:
                sps_table_source = (
                    get_spec().speculative_dspark_sps_table_path or "uninitialized"
                )
                logger.info(
                    "DSpark ragged-verify scheduler enabled (mode=%s, lag=%d, "
                    "relay_lag=%d, sps_table=%s, graph_tier=%s).",
                    self._ragged_verify_mode.value,
                    self._budget_planner.lag_steps,
                    relay_lag_steps,
                    sps_table_source,
                    (
                        "dynamic"
                        if self._dynamic_graph_tier
                        else (
                            "dp-gathered" if self._dp_tier_gather_enabled else "pinned"
                        )
                    ),
                )
                if (
                    isinstance(sps_table, SpsCostTable)
                    and is_uninitialized_sps_table(sps_table)
                    and not self._derive_sps_at_capture
                ):
                    logger.warning(
                        "DSpark SPS table is uninitialized (flat): the verify "
                        "budget degenerates to verify-all (zero scheduling gain). "
                        "Pass a profiled --speculative-dspark-sps-table-path."
                    )

    def install_capture_derived_sps_table(self, *, draft_model_runner) -> None:
        """Replace the uninitialized cost model with one measured from the captured graphs.

        Runs after CUDA graph capture, because that is the earliest point the measurement exists:
        this planner is constructed during worker init, which happens before capture.

        Without a profiled table the cost model is a single probe reporting the same steps-per-sec
        for every batch size, i.e. "a step costs the same no matter how wide it is". Under that
        model the budget objective is strictly increasing and argmax always lands on verify-all, so
        the planner cannot trim however good its confidence estimates are. The compact path already
        captures one verify graph per token tier, so replaying those tiers yields a real cost curve
        over exactly the axis the planner looks up.

        Opt-in via SGLANG_DSPARK_ENABLE_CAPTURE_DERIVED_SPS: whether trimming pays is a property of
        the deployment, not of the cost model. Where the throughput-versus-budget curve is flat,
        acting on a predicted gain costs accepted tokens for a time saving that does not materialise,
        so a deployment should measure before enabling this.

        Every failure leaves the uninitialized table in place: this is a startup optimisation and must never
        be able to stop the engine from starting.
        """
        if not self._derive_sps_at_capture:
            return
        self._derive_sps_at_capture = False

        if not ragged_verify_graphs_are_replayable(model_runner=self.model_runner):
            # The captured graphs will never be replayed on this backend, so timing them would price
            # a path the engine does not take. Keep the uninitialized table: verify-all, exactly as today.
            if self.tp_rank == 0:
                runner = self.model_runner.decode_cuda_graph_runner
                reason = (
                    "decode CUDA graphs are disabled"
                    if runner is None
                    else f"{type(runner.attn_backend).__name__} does not replay ragged verify graphs"
                )
                logger.info(
                    "DSpark capture-derived SPS table skipped: %s, so replay time does not "
                    "describe a verify step. The verify budget stays at verify-all.",
                    reason,
                )
            return

        started = time.perf_counter()
        try:
            verify_probes, draft_probes = self._measure_step_cost_probes(
                draft_model_runner=draft_model_runner
            )
        except Exception:
            # A startup optimisation must never stop the engine starting. The collective inside is
            # deliberately outside this guard -- see _measure_step_cost_probes.
            logger.warning(
                "DSpark capture-time SPS derivation failed; keeping the uninitialized table.",
                exc_info=True,
            )
            verify_probes, draft_probes = [], []
        elapsed = time.perf_counter() - started

        sps_table = build_capture_derived_sps_table(
            verify_probes=verify_probes, draft_probes=draft_probes
        )
        if sps_table is None:
            if self.tp_rank == 0:
                logger.warning(
                    "DSpark SPS table is uninitialized (flat) and could not be derived from the "
                    "captured verify graphs (%d usable probes): the verify budget degenerates to "
                    "verify-all (zero scheduling gain). Pass a profiled "
                    "--speculative-dspark-sps-table-path.",
                    len(verify_probes),
                )
            return

        self._budget_planner.sps_table = sps_table
        self._schedule_cfg.min_predicted_gain = CAPTURE_DERIVED_SPS_MIN_GAIN
        self._derived_sps_installed = True
        # The verify-all fast path short-circuits schedule_layout before the budget is consulted,
        # and its cached uniform layouts were built for that path, so both must go with the table.
        self._is_verify_all = False
        self._uniform_layout_cache = {}
        if self.tp_rank == 0:
            logger.info(
                "DSpark SPS table derived from %d verify tiers and %d draft batch sizes in %.2f s; "
                "the verify budget is now scheduled. Verify cost (tokens: ms) %s. Draft cost "
                "(requests: ms) %s. Unset SGLANG_DSPARK_ENABLE_CAPTURE_DERIVED_SPS to restore "
                "verify-all.",
                len(sps_table.m_probes),
                len(sps_table.bs_probes),
                elapsed,
                ", ".join(
                    f"{tokens}: {seconds * 1000.0:.2f}"
                    for tokens, seconds in zip(
                        sps_table.m_probes, sps_table.theta_seconds
                    )
                ),
                ", ".join(
                    f"{reqs}: {seconds * 1000.0:.2f}"
                    for reqs, seconds in zip(
                        sps_table.bs_probes, sps_table.alpha_seconds
                    )
                ),
            )

    def _measure_step_cost_probes(
        self, *, draft_model_runner
    ) -> tuple[list[tuple[int, float, float]], list[tuple[int, float, float]]]:
        """Measure both axes of the step cost, replicated identically across ranks.

        A verify graph replay covers the target forward, whose cost tracks the token count. The step
        the planner prices also runs the draft model, whose cost tracks the request count and is
        already spent by the time a budget is chosen. Both ladders are captured, so both terms are
        measurable here, and they are returned separately -- see build_capture_derived_sps_table for
        why collapsing them onto one axis is what makes the curve wrong.

        The result is broadcast from rank 0: the chosen budget selects which captured graph a step
        replays, so ranks planning against different curves would replay different shapes and their
        in-graph collectives would not match.

        Only the local measurement may fail. The broadcast is unconditional, because a rank that
        skipped it would leave its peers blocked in the collective for the distributed timeout --
        a worse failure than the one the measurement was guarded against. A rank that measured
        nothing contributes zeros and takes rank 0's curves; if rank 0 is the one that failed, every
        rank ends up with zeros and keeps the uninitialized table together.
        """
        tiers = ragged_capture_num_tokens(model_runner=self.model_runner)
        if not tiers:
            return [], []

        # Shape depends only on the tier list and a constant -- never on what this rank measured --
        # which is what keeps the collective below deadlock-free.
        costs = torch.zeros(
            (len(tiers) + CAPTURE_DERIVED_SPS_MAX_PROBES, 3),
            dtype=torch.float64,
            device=self.device,
        )
        try:
            self._fill_local_step_costs(
                costs=costs, tiers=tiers, draft_model_runner=draft_model_runner
            )
        except Exception:
            logger.warning(
                "DSpark capture-time cost measurement failed on this rank; falling back to "
                "whatever rank 0 measured.",
                exc_info=True,
            )

        broadcast_group, group_size = verify_lens_broadcast_group(
            tp_size=get_parallel().tp_size
        )
        if group_size > 1:
            broadcast_group.broadcast(costs, src=0)

        rows = costs.tolist()
        return _rows_to_probes(rows=rows[: len(tiers)]), _rows_to_probes(
            rows=rows[len(tiers) :]
        )

    def _fill_local_step_costs(
        self, *, costs: torch.Tensor, tiers: list[int], draft_model_runner
    ) -> None:
        """Time this rank's captured graphs into `costs` as (shape, seconds, spread) rows.

        Verify tiers take the first `len(tiers)` rows, keyed by total verify tokens; draft batch
        sizes take the rest, keyed by request count. Rows this rank could not measure stay zero and
        are dropped after the broadcast.

        The fixed shape is what makes the broadcast deadlock-free: it does not depend on how many
        probes this rank managed to measure.
        """
        measure_kwargs = dict(
            max_probes=CAPTURE_DERIVED_SPS_MAX_PROBES,
            warmup=CAPTURE_DERIVED_SPS_WARMUP,
            reps=CAPTURE_DERIVED_SPS_REPS,
        )
        verify_measured = (
            self.model_runner.decode_cuda_graph_runner.measure_captured_replay_seconds(
                **measure_kwargs
            )
        )
        tier_index = {tier: i for i, tier in enumerate(tiers)}
        for tier, seconds, spread in verify_measured:
            self._write_cost_row(
                costs=costs,
                row=tier_index[tier],
                shape=tier,
                seconds=seconds,
                spread=spread,
            )

        draft_runner = draft_model_runner.decode_cuda_graph_runner
        if draft_runner is None:
            return
        draft_measured = draft_runner.measure_captured_replay_seconds(**measure_kwargs)
        for offset, (batch_size, seconds, spread) in enumerate(
            draft_measured[:CAPTURE_DERIVED_SPS_MAX_PROBES]
        ):
            self._write_cost_row(
                costs=costs,
                row=len(tiers) + offset,
                shape=batch_size,
                seconds=seconds,
                spread=spread,
            )

    @staticmethod
    def _write_cost_row(
        *, costs: torch.Tensor, row: int, shape: int, seconds: float, spread: float
    ) -> None:
        costs[row, 0] = float(shape)
        costs[row, 1] = seconds
        costs[row, 2] = spread

    @property
    def carries_confidence(self) -> bool:
        return self._confidence_head is not None

    @property
    def last_confidence_raw(self) -> Optional[torch.Tensor]:
        if self._confidence_head is None:
            return None
        return self._confidence_head._last_confidence_raw

    @property
    def schedules_verify_budget(self) -> bool:
        return self._budget_planner is not None

    @property
    def is_compact_mode(self) -> bool:
        return self._ragged_verify_mode is RaggedVerifyMode.COMPACT

    @property
    def is_verify_all(self) -> bool:
        return self._is_verify_all

    @property
    def mode_value(self) -> str:
        return self._ragged_verify_mode.value

    @property
    def lag_steps(self) -> Optional[int]:
        if self._budget_planner is None:
            return None
        return self._budget_planner.lag_steps

    def take_budget_decision(self) -> Optional[VerifyBudgetDecision]:
        if self._budget_planner is None:
            return None
        return self._budget_planner.take_last_decision()

    def should_run_compact(self, *, layout: Optional[RaggedVerifyLayout]) -> bool:
        return (
            self._ragged_verify_mode is RaggedVerifyMode.COMPACT and layout is not None
        )

    def compute_confidence_tensor(
        self,
        *,
        draft_hidden: Optional[torch.Tensor],
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        confidence_tap: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self._confidence_head is None:
            return None
        compute_confidence_hook = getattr(self.draft_model, "compute_confidence", None)
        if compute_confidence_hook is not None:
            assert (
                confidence_tap is not None
            ), "dsv4 compute_confidence needs the compute_base_logits tap"
            with torch.inference_mode():
                return compute_confidence_hook(
                    anchor_tokens=anchor_tokens,
                    sampled_tokens=draft_tokens,
                    x_post_hc=confidence_tap,
                )
        assert draft_hidden is not None
        return compute_confidence(
            draft_hidden=draft_hidden,
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            confidence_head=self._confidence_head,
            markov_head=self.draft_model.markov_head,
            gamma=self.gamma,
        )

    def prepare_verify_budget(
        self, batch: ScheduleBatch, future_map: FutureMap
    ) -> None:
        draft_input = batch.spec_info
        if self._budget_planner is None:
            return
        if draft_input is None:
            local_tier_num_tokens = 0 if batch.batch_size() == 0 else -1
            self._maybe_gather_dp_verify_tier(
                batch=batch, local_tier_num_tokens=local_tier_num_tokens
            )
            return
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._budget_planner.note_non_decode_step()
            self._maybe_gather_dp_verify_tier(batch=batch, local_tier_num_tokens=0)
            return
        resolved = future_map.resolve_confidence_cpu(batch)
        draft_input.verify_token_budget = self._budget_from_resolved(
            resolved=resolved, req_pool_indices_cpu=batch.req_pool_indices_cpu
        )
        batch.spec_verify_tier_num_tokens = local_verify_tier_num_tokens(
            bs=batch.batch_size(),
            verify_token_budget=draft_input.verify_token_budget,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            min_verify_len=self._schedule_cfg.min_verify_len,
        )
        self._maybe_gather_dp_verify_tier(
            batch=batch, local_tier_num_tokens=batch.spec_verify_tier_num_tokens
        )

    def _maybe_gather_dp_verify_tier(
        self, *, batch: ScheduleBatch, local_tier_num_tokens: int
    ) -> None:
        if not self._dp_tier_gather_enabled:
            return
        if batch.is_extend_in_batch:
            batch.global_spec_verify_tier_num_tokens = None
            return
        cpu_group = get_tp_group().cpu_group
        local_tensor = torch.tensor([local_tier_num_tokens], dtype=torch.int64)
        gathered = torch.empty(
            (torch.distributed.get_world_size(group=cpu_group),), dtype=torch.int64
        )
        torch.distributed.all_gather_into_tensor(
            gathered, local_tensor, group=cpu_group
        )
        batch.global_spec_verify_tier_num_tokens = gathered.tolist()

    def note_non_decode_step(self) -> None:
        if self._budget_planner is not None:
            self._budget_planner.note_non_decode_step()

    def set_forced_budget_frac(self, frac) -> None:
        if self._budget_planner is not None:
            self._budget_planner.forced_budget_frac = frac

    def compute_budget_sync(
        self,
        *,
        confidence: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> Optional[int]:
        del prefix_lens
        if self._budget_planner is None:
            return None
        req_pool_indices_cpu = req_pool_indices.to("cpu").to(torch.int64)
        generation = self.model_runner.req_to_token_pool.req_generation[
            req_pool_indices_cpu
        ].clone()
        resolved = ResolvedConfidence(
            confidence=confidence.to("cpu"),
            generation=generation,
        )
        return self._budget_from_resolved(
            resolved=resolved, req_pool_indices_cpu=req_pool_indices_cpu
        )

    def resolve_verify_token_budget(
        self,
        *,
        draft_input: DFlashDraftInputV2,
        confidence: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> Optional[int]:
        """Per-step verify-token budget: under overlap it was precomputed into
        the draft input by prepare_verify_budget; otherwise compute it now."""
        if not self.schedules_verify_budget or confidence is None:
            return None
        if not get_schedule().disable_overlap_schedule:
            return draft_input.verify_token_budget
        return self.compute_budget_sync(
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
        )

    def confidence_budget_prepare(self):
        if not self.schedules_verify_budget:
            return None
        return self.prepare_verify_budget

    def _budget_from_resolved(
        self,
        *,
        resolved: Optional[ResolvedConfidence],
        req_pool_indices_cpu: torch.Tensor,
    ) -> Optional[int]:
        if resolved is None:
            self._budget_planner.note_non_decode_step()
            return None
        current_generation = self.model_runner.req_to_token_pool.req_generation[
            req_pool_indices_cpu.to(torch.int64)
        ]
        return int(
            self._budget_planner.compute_budget(
                confidence=resolved.confidence,
                generation=resolved.generation,
                current_generation=current_generation,
                req_pool_indices_cpu=req_pool_indices_cpu,
            )
        )

    def _budget_covers_full_window(self, *, bs: int, budget: Optional[int]) -> bool:
        """Would this budget let every request verify its whole window?

        A real cost table makes `_is_verify_all` False for the engine's lifetime, but the planner
        still returns a full-width budget whenever trimming does not pay -- on a deployment with no
        headroom, that is every step. Without this the engine would pay the per-step schedule and its
        host<->device round-trips to arrive at the layout the cache already holds, which is a
        measured regression on models that gain nothing from trimming.

        The test is deliberately the *sufficient* one: at this budget the top-k can fill every
        window, so the uniform layout is what verify-all serves today. Anything less falls through
        to the scheduler.
        """
        if budget is None:
            return False
        cfg = self._schedule_cfg
        widest = cfg.resolved_max_verify_len() - cfg.min_verify_len
        return budget >= bs * widest

    def schedule_layout(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        device: torch.device,
        confidence: Optional[torch.Tensor],
        budget: Optional[int],
        global_num_reqs: Optional[int] = None,
        dp_tier_num_tokens: Optional[int] = None,
    ) -> Optional[RaggedVerifyLayout]:
        if self._ragged_verify_mode is RaggedVerifyMode.STATIC:
            return None
        serves_full_window = self._is_verify_all or (
            self._derived_sps_installed
            and self._budget_covers_full_window(
                bs=int(req_pool_indices.shape[0]), budget=budget
            )
        )
        if serves_full_window and self._ragged_verify_mode is RaggedVerifyMode.COMPACT:
            # Verify-all: the uniform layout (or None, past the captured grid)
            # is constant per (bs, tier); serve it from cache instead of paying
            # the per-step schedule and its host<->device round-trips.
            key = (int(req_pool_indices.shape[0]), global_num_reqs)
            if key not in self._uniform_layout_cache:
                self._uniform_layout_cache[key] = uniform_ragged_layout(
                    bs=key[0],
                    device=device,
                    verify_num_draft_tokens=self.verify_num_draft_tokens,
                    ragged_verify_mode=self._ragged_verify_mode,
                    model_runner=self.model_runner,
                    tier_num_reqs=global_num_reqs,
                )
            return self._uniform_layout_cache[key]
        verify_lens = self._schedule_verify_lens(
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=self._budget_aligned_to_graph_tier(
                req_pool_indices=req_pool_indices,
                budget=budget,
                global_num_reqs=global_num_reqs,
                dp_tier_num_tokens=dp_tier_num_tokens,
            ),
        )
        if verify_lens is None:
            assert dp_tier_num_tokens is None, (
                "dp tier agreement present but local verify lens are None; "
                "the gathered hint and the local budget diverged"
            )
            if self._ragged_verify_mode is RaggedVerifyMode.COMPACT:
                return uniform_ragged_layout(
                    bs=len(req_pool_indices),
                    device=device,
                    verify_num_draft_tokens=self.verify_num_draft_tokens,
                    ragged_verify_mode=self._ragged_verify_mode,
                    model_runner=self.model_runner,
                    tier_num_reqs=global_num_reqs,
                )
            return None
        bs = int(verify_lens.shape[0])
        tier_num_reqs = bs if global_num_reqs is None else global_num_reqs
        if dp_tier_num_tokens is not None:
            assert global_num_reqs is not None, (
                "dp tier agreement requires the dp-global request count; "
                "keying the tier off the local bs diverges across ranks"
            )
            tier_num_tokens = dp_tier_num_tokens
        elif self._dynamic_graph_tier and budget is not None:
            tier_num_tokens = local_verify_tier_num_tokens(
                bs=tier_num_reqs,
                verify_token_budget=budget,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                min_verify_len=self._schedule_cfg.min_verify_len,
            )
        else:
            tier_num_tokens = None
        if ragged_layout_exceeds_captured_grid(
            num_reqs=tier_num_reqs,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            tier_tokens_hint=tier_num_tokens,
        ):
            return None
        graph_num_tokens_floor = verify_layout_graph_num_tokens_floor(
            num_reqs=tier_num_reqs,
            ragged_verify_mode=self._ragged_verify_mode,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            tier_num_tokens=tier_num_tokens,
        )
        capture_num_tokens = ragged_capture_num_tokens(model_runner=self.model_runner)
        if graph_num_tokens_floor > 0 and capture_num_tokens is not None:
            graph_num_tokens = round_up_grid(graph_num_tokens_floor, capture_num_tokens)
            return RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
            )
        verify_lens_cpu = verify_lens.to("cpu").tolist()
        grid = verify_layout_grid(
            verify_lens_cpu=verify_lens_cpu,
            ragged_verify_mode=self._ragged_verify_mode,
            model_runner=self.model_runner,
        )
        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=device,
            grid=grid,
            graph_num_tokens_floor=graph_num_tokens_floor,
        )

    def _budget_aligned_to_graph_tier(
        self,
        *,
        req_pool_indices: torch.Tensor,
        budget: Optional[int],
        global_num_reqs: Optional[int],
        dp_tier_num_tokens: Optional[int],
    ) -> Optional[int]:
        # Flag off (default): returns budget unchanged, so the schedule below is
        # byte-for-byte the original. On: ceils role 1's verify-token total up to the
        # padded graph tier graph_num_tokens = round_up(dp-max tier, captured token
        # bucket), which folds in the cuda-graph bucket round-up (H1) and the dp
        # cross-rank max (H2); role 2 (the single top-k) then admits that many real
        # draft tokens. graph_num_tokens is derived from the same (request count,
        # gathered dp tier, original budget) inputs the layout below uses, so the two
        # agree by construction -- this only feeds the larger budget into the top-k,
        # it does not touch the layout's own tier computation.
        if not self._align_verify_tokens_to_graph_tier or budget is None:
            return budget
        tier_num_reqs = (
            int(req_pool_indices.shape[0])
            if global_num_reqs is None
            else global_num_reqs
        )
        if dp_tier_num_tokens is not None:
            tier_num_tokens = dp_tier_num_tokens
        elif self._dynamic_graph_tier:
            tier_num_tokens = local_verify_tier_num_tokens(
                bs=tier_num_reqs,
                verify_token_budget=budget,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                min_verify_len=self._schedule_cfg.min_verify_len,
            )
        else:
            tier_num_tokens = None
        graph_num_tokens_floor = verify_layout_graph_num_tokens_floor(
            num_reqs=tier_num_reqs,
            ragged_verify_mode=self._ragged_verify_mode,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            tier_num_tokens=tier_num_tokens,
        )
        capture_num_tokens = ragged_capture_num_tokens(model_runner=self.model_runner)
        if graph_num_tokens_floor <= 0 or capture_num_tokens is None:
            return budget
        graph_num_tokens = round_up_grid(graph_num_tokens_floor, capture_num_tokens)
        return graph_tier_fill_budget(
            graph_num_tokens=graph_num_tokens,
            bs=int(req_pool_indices.shape[0]),
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            min_verify_len=self._schedule_cfg.min_verify_len,
        )

    def _schedule_verify_lens(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        device: torch.device,
        confidence: Optional[torch.Tensor],
        budget: Optional[int],
    ) -> Optional[torch.Tensor]:
        if self._budget_planner is None or confidence is None or budget is None:
            return None
        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=confidence,
            budget=budget,
            cfg=self._schedule_cfg,
        ).to(device=device, dtype=torch.int32)

        if resolve_level() >= InvariantCheckLevel.WARN:
            verify_lens_64 = verify_lens.to(torch.int64)
            effective_floor = max(self._schedule_cfg.min_verify_len, 1)
            expect(
                _VERIFY_LEN_BUDGET,
                (verify_lens_64 - effective_floor).sum() <= budget,
                msg=f"budget={budget}",
            )

        if envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER.get():
            self._log_verify_lens_decision(
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                budget=budget,
                sort_survival=compute_sort_survival(confidence),
                verify_lens=verify_lens,
            )

        broadcast_group, group_size = verify_lens_broadcast_group(
            tp_size=get_parallel().tp_size
        )
        if group_size > 1:
            broadcast_group.broadcast(verify_lens, src=0)

        return verify_lens

    def _log_verify_lens_decision(
        self,
        *,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        budget: int,
        sort_survival: torch.Tensor,
        verify_lens: torch.Tensor,
    ) -> None:
        cfg = self._schedule_cfg
        max_len = cfg.resolved_max_verify_len()
        req_ids = req_pool_indices.tolist()
        prefixes = prefix_lens.tolist()
        lens = verify_lens.tolist()
        sort_rows = sort_survival.to(torch.float32).tolist()
        logger.info(
            "[DSPARK-CPS] num_reqs=%d budget=%d gamma=%d verify_len_range=[%d,%d]",
            len(req_ids),
            budget,
            cfg.gamma,
            cfg.min_verify_len,
            max_len,
        )
        for row in range(len(req_ids)):
            survival_str = "[" + ", ".join(f"{p:.3f}" for p in sort_rows[row]) + "]"
            logger.info(
                "[DSPARK-CPS]   req=%d prefix=%d verify_len=%d sort_survival=%s",
                int(req_ids[row]),
                int(prefixes[row]),
                int(lens[row]),
                survival_str,
            )


def local_verify_tier_num_tokens(
    *,
    bs: int,
    verify_token_budget: Optional[int],
    verify_num_draft_tokens: int,
    min_verify_len: int,
) -> int:
    if verify_token_budget is None:
        return -1
    floor_tokens = bs * max(min_verify_len, 1)
    return min(floor_tokens + verify_token_budget, bs * verify_num_draft_tokens)


def graph_tier_fill_budget(
    *,
    graph_num_tokens: int,
    bs: int,
    verify_num_draft_tokens: int,
    min_verify_len: int,
) -> int:
    # top-k budget (tokens above the per-request floor) that makes the scheduled
    # total reach the padded graph tier, capped at bs * verify_num_draft_tokens
    # since a request cannot verify more than its proposed drafts. Inverse of
    # local_verify_tier_num_tokens: total = floor_tokens + budget.
    fill_total = min(graph_num_tokens, bs * verify_num_draft_tokens)
    floor_tokens = bs * max(min_verify_len, 1)
    return max(0, fill_total - floor_tokens)


def dp_global_verify_tier_num_tokens(
    *,
    global_tier_num_tokens: Optional[list[int]],
) -> Optional[int]:
    if global_tier_num_tokens is None:
        return None
    if any(tier_num_tokens < 0 for tier_num_tokens in global_tier_num_tokens):
        return None
    max_tier_num_tokens = max(global_tier_num_tokens, default=0)
    return max_tier_num_tokens if max_tier_num_tokens > 0 else None


def idle_ragged_layout(
    *,
    tier_num_reqs: int,
    dp_tier_num_tokens: Optional[int],
    device: torch.device,
    verify_num_draft_tokens: int,
    model_runner,
) -> Optional[RaggedVerifyLayout]:
    if ragged_capture_num_tokens(model_runner=model_runner) is None:
        dp_tier_num_tokens = None
    if dp_tier_num_tokens is None:
        return uniform_ragged_layout(
            bs=tier_num_reqs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            ragged_verify_mode=RaggedVerifyMode.COMPACT,
            model_runner=model_runner,
        )
    if ragged_layout_exceeds_captured_grid(
        num_reqs=tier_num_reqs,
        verify_num_draft_tokens=verify_num_draft_tokens,
        model_runner=model_runner,
        tier_tokens_hint=dp_tier_num_tokens,
    ):
        return None
    verify_lens_cpu = [1] * tier_num_reqs
    grid = verify_layout_grid(
        verify_lens_cpu=verify_lens_cpu,
        ragged_verify_mode=RaggedVerifyMode.COMPACT,
        model_runner=model_runner,
    )
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_cpu,
        device=device,
        grid=grid,
        graph_num_tokens_floor=dp_tier_num_tokens,
    )


def uniform_ragged_layout(
    *,
    bs: int,
    device: torch.device,
    verify_num_draft_tokens: int,
    ragged_verify_mode: RaggedVerifyMode,
    model_runner,
    tier_num_reqs: Optional[int] = None,
) -> Optional[RaggedVerifyLayout]:
    tier_num_reqs = bs if tier_num_reqs is None else tier_num_reqs
    if ragged_layout_exceeds_captured_grid(
        num_reqs=tier_num_reqs,
        verify_num_draft_tokens=verify_num_draft_tokens,
        model_runner=model_runner,
    ):
        return None
    verify_lens_cpu = [verify_num_draft_tokens] * bs
    grid = verify_layout_grid(
        verify_lens_cpu=verify_lens_cpu,
        ragged_verify_mode=ragged_verify_mode,
        model_runner=model_runner,
    )
    graph_num_tokens_floor = verify_layout_graph_num_tokens_floor(
        num_reqs=tier_num_reqs,
        ragged_verify_mode=ragged_verify_mode,
        verify_num_draft_tokens=verify_num_draft_tokens,
        model_runner=model_runner,
    )
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens_cpu,
        device=device,
        grid=grid,
        graph_num_tokens_floor=graph_num_tokens_floor,
    )


def verify_lens_broadcast_group(*, tp_size: int) -> tuple:
    if is_dp_attention_enabled():
        return get_parallel().attn_tp_group, get_parallel().attn_tp_size
    return get_tp_group(), tp_size


def verify_layout_grid(
    *,
    verify_lens_cpu: list[int],
    ragged_verify_mode: RaggedVerifyMode,
    model_runner,
) -> list[int]:
    total = sum(verify_lens_cpu)
    if ragged_verify_mode is not RaggedVerifyMode.COMPACT:
        return [total]
    capture_num_tokens = ragged_capture_num_tokens(model_runner=model_runner)
    if capture_num_tokens is None:
        return [total]
    return capture_num_tokens


def verify_layout_graph_num_tokens_floor(
    *,
    num_reqs: int,
    ragged_verify_mode: RaggedVerifyMode,
    verify_num_draft_tokens: int,
    model_runner,
    tier_num_tokens: Optional[int] = None,
) -> int:
    if (
        ragged_verify_mode is not RaggedVerifyMode.COMPACT
        or ragged_capture_num_tokens(model_runner=model_runner) is None
    ):
        return 0
    if tier_num_tokens is not None:
        return min(tier_num_tokens, num_reqs * verify_num_draft_tokens)
    return num_reqs * verify_num_draft_tokens


def ragged_capture_num_tokens(*, model_runner) -> Optional[list[int]]:
    runner = model_runner.decode_cuda_graph_runner
    if runner is None or not runner.ragged_verify_mode:
        return None
    return runner.capture_num_tokens


def ragged_capture_max_slots(*, model_runner) -> Optional[int]:
    runner = model_runner.decode_cuda_graph_runner
    if runner is None or not runner.ragged_verify_mode:
        return None
    return runner.max_bs


def ragged_verify_graphs_are_replayable(*, model_runner) -> bool:
    """Will a verify step ever replay one of the captured ragged graphs?

    Capture records a verify graph per token tier regardless, but the runner's admission test starts
    at `attn_backend.supports_ragged_verify_graph`, and a backend that does not implement the ragged
    metadata path fails it on every step -- hybrid linear-attention targets, for instance, whose GDN
    backend leaves the base class default of False.

    On such a target the captured graphs are recorded and never replayed, so a cost model derived by
    timing them describes a path the engine does not take. Deriving one there is worse than useless:
    it spends the measurement, and then hands the planner a curve with no bearing on the step it is
    pricing.
    """
    runner = model_runner.decode_cuda_graph_runner
    if runner is None or not runner.ragged_verify_mode:
        return False
    return bool(runner.attn_backend.supports_ragged_verify_graph)


def ragged_layout_exceeds_captured_grid(
    *,
    num_reqs: int,
    verify_num_draft_tokens: int,
    model_runner,
    tier_tokens_hint: Optional[int] = None,
) -> bool:
    capture_num_tokens = ragged_capture_num_tokens(model_runner=model_runner)
    if capture_num_tokens is None:
        return False
    max_slots = ragged_capture_max_slots(model_runner=model_runner)
    if max_slots is not None and num_reqs > max_slots:
        return True
    tier_tokens = (
        tier_tokens_hint
        if tier_tokens_hint is not None
        else num_reqs * verify_num_draft_tokens
    )
    return tier_tokens > capture_num_tokens[-1]


def alloc_verify_window(
    *,
    batch: ScheduleBatch,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    block_pos_offsets: torch.Tensor,
    model_runner,
) -> VerifyWindow:
    prefix_lens = batch.seq_lens
    verify_w = verify_num_draft_tokens
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets
    verify_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_w,
        batch_size=bs,
        draft_token_num=verify_w,
        device=device,
    )
    verify_cache_loc_2d = verify_cache_loc.view(bs, verify_w)
    return VerifyWindow(
        positions_2d=positions_2d,
        verify_cache_loc=verify_cache_loc,
        verify_cache_loc_2d=verify_cache_loc_2d,
    )


def apply_logits_adjustments_strided(
    *,
    next_token_logits: torch.Tensor,
    sampling_info,
    verify_num_draft_tokens: int,
) -> None:
    if sampling_info is None:
        return
    apply_dflash_verify_logits_adjustments(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=verify_num_draft_tokens,
    )


def build_markov_embed_stack(
    *,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    markov_head,
    gamma: int,
) -> torch.Tensor:
    prev_seq = torch.cat(
        [anchor_tokens.view(-1, 1), draft_tokens[:, : gamma - 1]], dim=1
    )
    return markov_head.get_prev_embeddings(prev_seq)


def compute_confidence(
    *,
    draft_hidden: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    confidence_head,
    markov_head,
    gamma: int,
) -> torch.Tensor:
    assert confidence_head is not None
    if confidence_head.with_markov:
        markov_embed_stack = build_markov_embed_stack(
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            markov_head=markov_head,
            gamma=gamma,
        )
    else:
        markov_embed_stack = None
    confidence_raw = confidence_head(draft_hidden, markov_embed_stack)
    confidence = confidence_head.apply_sts(confidence_raw)
    expect(_CONFIDENCE, confidence)
    return confidence


class DSparkScheduleConfig(msgspec.Struct):
    gamma: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6
    # Smallest predicted throughput gain, relative to verifying everything, that justifies trimming
    # at all. Zero keeps the historical behaviour of always taking the argmax.
    min_predicted_gain: float = 0.0

    def resolved_max_verify_len(self) -> int:
        return self.max_verify_len or (self.gamma + 1)

    def validate(self) -> None:
        max_len = self.resolved_max_verify_len()
        if self.gamma < 1:
            raise ValueError(f"DSpark gamma must be >= 1, got {self.gamma}.")
        if not (0 <= self.min_verify_len <= max_len <= self.gamma + 1):
            raise ValueError(
                "DSpark verify-len config must satisfy 0 <= min <= max <= gamma+1, "
                f"got min={self.min_verify_len}, max={max_len}, gamma={self.gamma}."
            )
        if self.survival_eps < 0:
            raise ValueError(f"survival_eps must be >= 0, got {self.survival_eps}.")


class VerifyBudgetDecision(msgspec.Struct):
    budget: int
    predicted_step_seconds: Optional[float] = None
    predicted_theta: Optional[float] = None


def simulate_acc_len_needs_verify_all() -> bool:
    """True when SGLANG_SIMULATE_ACC_LEN pins the schedule to verify-all.

    A constant simulated correct_len > 0 can exceed a trimmed request's verify budget and break the
    cutoff accounting, so DSparkWorkerV2 refuses that combination at construction. Deriving a cost
    table happens afterwards and would flip the planner off verify-all, so both the admission check
    and the derivation read this one predicate rather than each carrying a copy.
    """
    simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
    return simulate_acc_len > 0.0 and simulate_acc_len != 1.0


def compute_verify_token_budget(
    *,
    history_survival_probs: torch.Tensor,
    sps_table: Union[SpsCostTable, SpsAdditiveCostTable],
    cfg: DSparkScheduleConfig,
) -> VerifyBudgetDecision:
    num_requests = history_survival_probs.shape[0]
    max_len = cfg.resolved_max_verify_len()

    candidates = history_survival_probs[:, :max_len].flatten()
    candidates = candidates[candidates >= cfg.survival_eps].to(torch.float64)
    candidates_sorted = torch.sort(candidates, descending=True).values
    prefix_sum = torch.cumsum(candidates_sorted, dim=0)

    tau_star = num_requests + torch.cat(
        [torch.zeros(1, dtype=torch.float64), prefix_sum]
    )
    if isinstance(sps_table, SpsAdditiveCostTable):
        step_time = _additive_step_time_tensor(
            table=sps_table,
            num_requests=int(num_requests),
            num_budgets=int(tau_star.numel()),
        )
        theta = tau_star / step_time
        idx = _argmax_worth_trimming(theta=theta, min_gain=cfg.min_predicted_gain)
        predicted_step_seconds = float(step_time[idx])
    else:
        batch_tokens = num_requests + torch.arange(tau_star.numel(), dtype=torch.int64)
        sps = _lookup_sps_tensor(sps_table=sps_table, batch_tokens=batch_tokens)
        theta = tau_star * sps
        idx = _argmax_worth_trimming(theta=theta, min_gain=cfg.min_predicted_gain)
        sps_at_idx = float(sps[idx])
        predicted_step_seconds = 1.0 / sps_at_idx if sps_at_idx > 0 else None
    return VerifyBudgetDecision(
        budget=idx,
        predicted_step_seconds=predicted_step_seconds,
        predicted_theta=float(theta[idx]),
    )


def _argmax_worth_trimming(*, theta: torch.Tensor, min_gain: float) -> int:
    """Best budget, unless the predicted win over verifying everything is too small to be worth it.

    Trimming is not free even when the cost model likes it: fewer drafted tokens are verified, so a
    step that would have committed them has to earn the loss back in reduced step time. Where the
    cost curve is nearly flat -- small models, low load -- the predicted win shrinks to noise while
    the accepted-token loss stays real, and acting on it is a measured throughput regression.

    Verify-all is the last index because tau_star is strictly increasing, so it is also the baseline
    to beat. `min_gain` of zero reproduces a plain argmax.
    """
    idx = int(torch.argmax(theta))
    if min_gain <= 0.0:
        return idx
    verify_all = int(theta.numel()) - 1
    baseline = float(theta[verify_all])
    if baseline <= 0.0 or float(theta[idx]) < baseline * (1.0 + min_gain):
        return verify_all
    return idx


def _lookup_sps_tensor(
    *, sps_table: SpsCostTable, batch_tokens: torch.Tensor
) -> torch.Tensor:
    probes = torch.tensor(sps_table.sample_batch_tokens, dtype=torch.int64)
    sps = torch.tensor(sps_table.sample_steps_per_sec, dtype=torch.float64)
    idx = torch.bucketize(batch_tokens, probes, right=True) - 1
    idx = idx.clamp_(0, probes.numel() - 1)
    return sps[idx]


def _additive_step_time_tensor(
    *, table: SpsAdditiveCostTable, num_requests: int, num_budgets: int
) -> torch.Tensor:
    floor = table.bias_seconds + _interp_clamped(
        table.bs_probes, table.alpha_seconds, float(num_requests)
    )
    m_probes = torch.tensor(table.m_probes, dtype=torch.float64)
    theta_vals = torch.tensor(table.theta_seconds, dtype=torch.float64)
    m = (num_requests + torch.arange(num_budgets, dtype=torch.float64)).clamp_(
        min=float(table.m_probes[0]), max=float(table.m_probes[-1])
    )
    hi = torch.bucketize(m, m_probes, right=True).clamp_(1, m_probes.numel() - 1)
    lo = hi - 1
    span = (m_probes[hi] - m_probes[lo]).clamp_(min=1e-9)
    frac = (m - m_probes[lo]) / span
    theta_at_m = theta_vals[lo] + frac * (theta_vals[hi] - theta_vals[lo])
    return floor + theta_at_m


class HostConfidenceBudgetPlanner:

    def __init__(
        self,
        *,
        sps_table: Union[SpsCostTable, SpsAdditiveCostTable],
        cfg: DSparkScheduleConfig,
        model_runner,
        relay_lag_steps: int = 1,
    ) -> None:
        cfg.validate()
        self.sps_table = sps_table
        self.cfg = cfg
        self._model_runner = model_runner
        self.forced_budget_frac: Optional[float] = None
        self.last_decision: Optional[VerifyBudgetDecision] = None
        self.lag_steps = max(
            int(envs.SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS.get()), 1
        )
        self.carry_steps = max(self.lag_steps - int(relay_lag_steps), 0)
        self._carry_confidence: Optional[torch.Tensor] = None
        self._carry_generation: Optional[torch.Tensor] = None
        self._carry_pos = 0

    def compute_budget(
        self,
        *,
        confidence: torch.Tensor,
        generation: torch.Tensor,
        current_generation: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> int:
        lagged_confidence, lagged_generation = self._shift_to_lag(
            confidence=confidence,
            generation=generation,
            req_pool_indices_cpu=req_pool_indices_cpu,
        )
        survival = self._two_steps_prior_survival(
            lagged_confidence=lagged_confidence,
            lagged_generation=lagged_generation,
            current_generation=current_generation,
        )
        forced_frac = self.forced_budget_frac
        if forced_frac is not None:
            full_budget = int(survival[:, : self.cfg.resolved_max_verify_len()].numel())
            forced_budget = max(0, int(float(forced_frac) * full_budget))
            self.last_decision = VerifyBudgetDecision(budget=forced_budget)
            return forced_budget
        decision = compute_verify_token_budget(
            history_survival_probs=survival,
            sps_table=self.sps_table,
            cfg=self.cfg,
        )
        self.last_decision = decision
        return decision.budget

    def take_last_decision(self) -> Optional[VerifyBudgetDecision]:
        decision = self.last_decision
        self.last_decision = None
        return decision

    def note_non_decode_step(self) -> None:
        self.last_decision = None

    def _shift_to_lag(
        self,
        *,
        confidence: torch.Tensor,
        generation: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.carry_steps == 0:
            return confidence, generation
        self._ensure_carry(gamma=confidence.shape[-1])
        slot = self._carry_pos % self.carry_steps
        rows = req_pool_indices_cpu.to(torch.int64)
        lagged_confidence = self._carry_confidence[slot, rows].clone()
        lagged_generation = self._carry_generation[slot, rows].clone()
        self._carry_confidence[slot, rows] = confidence.to(torch.float32)
        self._carry_generation[slot, rows] = generation.to(torch.int64)
        self._carry_pos += 1
        return lagged_confidence, lagged_generation

    def _two_steps_prior_survival(
        self,
        *,
        lagged_confidence: torch.Tensor,
        lagged_generation: torch.Tensor,
        current_generation: torch.Tensor,
    ) -> torch.Tensor:
        k_survival = torch.cumprod(lagged_confidence.to(torch.float32), dim=1)
        current_gen = current_generation.to(torch.int64)
        fresh = (
            (current_gen >= 1) & (lagged_generation.to(torch.int64) == current_gen)
        ).view(-1, 1)
        return torch.where(fresh, k_survival, torch.ones_like(k_survival))

    def _ensure_carry(self, *, gamma: int) -> None:
        if self._carry_confidence is not None:
            return
        req_pool_size = int(self._model_runner.req_to_token_pool.req_to_token.shape[0])
        self._carry_confidence = torch.zeros(
            (self.carry_steps, req_pool_size, gamma), dtype=torch.float32
        )
        self._carry_generation = torch.zeros(
            (self.carry_steps, req_pool_size),
            dtype=torch.int64,
        )


def build_sps_cost_table(
    *,
    server_args: ServerArgs,
    verify_num_draft_tokens: int,
) -> Union[SpsCostTable, SpsAdditiveCostTable]:
    sps_table_path = server_args.speculative_dspark_sps_table_path
    if sps_table_path:
        return load_sps_table_from_path(sps_table_path)
    max_batch_tokens = max(
        1,
        int(get_schedule().max_running_requests or 1) * verify_num_draft_tokens,
    )
    return build_uninitialized_sps_table(max_batch_tokens=max_batch_tokens)
