import logging
from typing import Optional

import torch

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
from sglang.srt.speculative.dspark_components.dspark_confidence import (
    compute_confidence,
)
from sglang.srt.speculative.dspark_components.dspark_scheduler import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
    VerifyBudgetDecision,
    build_sps_cost_table,
)
from sglang.srt.speculative.dspark_components.dspark_sps_table import (
    SpsCostTable,
    is_uninitialized_sps_table,
)
from sglang.srt.speculative.dspark_components.dspark_sts import (
    load_sts_calibration_from_path,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    graph_tier_fill_budget,
    local_verify_tier_num_tokens,
    ragged_capture_num_tokens,
    ragged_layout_exceeds_captured_grid,
    uniform_ragged_layout,
    verify_layout_graph_num_tokens_floor,
    verify_layout_grid,
    verify_lens_broadcast_group,
)
from sglang.srt.speculative.dspark_components.kernels.schedule_verify_lens_topk import (
    ScheduleVerifyLensTopk,
    compute_sort_survival,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
    read_ragged_verify_mode,
    round_up_grid,
)
from sglang.srt.utils.async_probe import maybe_assert_async
from sglang.srt.utils.common import require_mlp_tp_gather

logger = logging.getLogger(__name__)


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
