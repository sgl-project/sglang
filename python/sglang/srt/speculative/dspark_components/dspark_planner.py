from __future__ import annotations

import logging
from typing import Optional, Union

import msgspec
import torch

from sglang.kernels.ops.memory.req_to_token_pool import AssignExtendCacheLocs
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_LAG,
    FutureMap,
    ResolvedConfidence,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
    _interp_clamped,
    build_uninitialized_sps_table,
    is_uninitialized_sps_table,
    load_sps_table_from_path,
)
from sglang.srt.speculative.dspark_components.dspark_sts import (
    load_sts_calibration_from_path,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_schedule import (
    ScheduleVerifyLensTopk,
    compute_sort_survival,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
    round_up_grid,
)
from sglang.srt.utils.async_probe import (
    maybe_assert_async,
    maybe_detect_in_closed_range,
)
from sglang.srt.utils.common import require_mlp_tp_gather

logger = logging.getLogger(__name__)


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
            self._require_prep_in_cuda_graph()
            sps_table = build_sps_cost_table(
                server_args=self.server_args,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
            )
            self._is_verify_all = (
                self._ragged_verify_mode is RaggedVerifyMode.COMPACT
                and is_uninitialized_sps_table(sps_table)
            )
            relay_lag_steps = (
                0
                if self.server_args.disable_overlap_schedule
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
                and not self.server_args.disable_overlap_schedule
                and not self.server_args.speculative_skip_dp_mlp_sync
                and self.server_args.disaggregation_mode == "null"
                and self.server_args.pp_size == 1
                and not envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
            )
            if tp_rank == 0:
                sps_table_source = (
                    self.server_args.speculative_dspark_sps_table_path
                    or "uninitialized"
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
                if isinstance(sps_table, SpsCostTable) and is_uninitialized_sps_table(
                    sps_table
                ):
                    logger.warning(
                        "DSpark SPS table is uninitialized (flat): the verify "
                        "budget degenerates to verify-all (zero scheduling gain). "
                        "Pass a profiled --speculative-dspark-sps-table-path."
                    )

    def _require_prep_in_cuda_graph(self) -> None:
        if not envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            raise ValueError(
                f"DSpark ragged-verify mode {self._ragged_verify_mode.value!r} "
                f"requires SGLANG_PREP_IN_CUDA_GRAPH=1 (the captured-graph prepare "
                f"path). It is currently disabled, which would put per-step "
                f"verify_lens_cpu host reads on the critical path. Set "
                f"SGLANG_PREP_IN_CUDA_GRAPH=1 or run SGLANG_RAGGED_VERIFY_MODE=static."
            )

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
        if not self.server_args.disable_overlap_schedule:
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

        if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
            verify_lens_64 = verify_lens.to(torch.int64)
            effective_floor = max(self._schedule_cfg.min_verify_len, 1)
            maybe_assert_async(
                (verify_lens_64 - effective_floor).sum() <= budget,
                f"DSpark verify-len budget violated (budget={budget})",
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
            tp_size=self.server_args.tp_size
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
    verify_cache_loc = AssignExtendCacheLocs.execute(
        model_runner.req_to_token_pool.req_to_token,
        req_pool_indices=batch.req_pool_indices,
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
    maybe_detect_in_closed_range(confidence, 0.0, 1.0, "DSpark confidence")
    return confidence


class DSparkScheduleConfig(msgspec.Struct):
    gamma: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

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
        idx = int(torch.argmax(theta))
        predicted_step_seconds = float(step_time[idx])
    else:
        batch_tokens = num_requests + torch.arange(tau_star.numel(), dtype=torch.int64)
        sps = _lookup_sps_tensor(sps_table=sps_table, batch_tokens=batch_tokens)
        theta = tau_star * sps
        idx = int(torch.argmax(theta))
        sps_at_idx = float(sps[idx])
        predicted_step_seconds = 1.0 / sps_at_idx if sps_at_idx > 0 else None
    return VerifyBudgetDecision(
        budget=idx,
        predicted_step_seconds=predicted_step_seconds,
        predicted_theta=float(theta[idx]),
    )


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
        sps_table: SpsCostTable,
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
        int(server_args.max_running_requests or 1) * verify_num_draft_tokens,
    )
    return build_uninitialized_sps_table(max_batch_tokens=max_batch_tokens)
