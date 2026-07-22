from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from array import array
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.self_benchmark_decode import (
    SyntheticDecodeBatchBuilder,
)
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.observability.forward_pass_metrics import ForwardPassMetrics
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)

SELF_BENCHMARK_REQ_PREFIX = "__sgl_bench_"
SELF_BENCHMARK_DUMMY_TOKEN_ID = 0


@dataclass
class SelfBenchmarkConfig:
    mode: str
    prefill_isl_granularity: int = 16
    prefill_kv_read_granularity: int = 1
    decode_length_granularity: int = 6
    decode_batch_size_granularity: int = 6
    warmup_iterations: int = 5
    output_path: str = "/tmp/benchmark_results.json"


@dataclass
class BenchmarkPoint:
    point_type: str
    isl: int = 0
    kv_read_tokens: int = 0
    context_length: int = 0
    batch_size: int = 0


@dataclass
class BenchmarkPointResult:
    point: BenchmarkPoint
    fpms: list = field(default_factory=list)


@dataclass
class SkippedBenchmarkPoint:
    point: BenchmarkPoint
    reason: str


class BenchmarkPhase(Enum):
    WARMUP = "warmup"
    SWEEP = "sweep"
    DONE = "done"


class SelfBenchmark:
    """Scheduler-local self benchmark.

    Multi-rank lockstep assumption: the benchmark is constructed and advanced on
    every scheduler rank, and FPM is forced on for all ranks during the sweep
    (see metrics_reporter._init_fpm) so observe_forward_pass fires everywhere and
    every rank advances WARMUP->SWEEP->DONE together. The grid (_build_grid) and
    synthetic allocations are deterministic functions of the server args and the
    homogeneous per-rank limits (max_total_num_tokens, page_size, max_req_len,
    etc.), so each iteration runs an identical synthetic batch in the same
    collective across ranks. There is intentionally no cross-rank barrier: any
    per-rank, data-dependent skip in maybe_schedule_next / observe_forward_pass
    would desync the collectives, so all skip/advance decisions here must depend
    only on homogeneous state.
    """

    MAX_AXIS_GRANULARITY = 1024
    MAX_GRID_POINTS = 4096

    @classmethod
    def create_if_enabled(cls, scheduler: Scheduler) -> Optional[SelfBenchmark]:
        """Validate runtime compatibility and create an enabled benchmark."""
        if scheduler.server_args.benchmark_mode is None:
            return None

        if scheduler.ps.pp_size > 1:
            raise ValueError(
                "--benchmark-mode is not supported with pipeline parallelism"
            )
        if scheduler.enable_pdmux:
            raise ValueError("--benchmark-mode is not supported with PD multiplexing")
        if scheduler.enable_overlap_mlx:
            raise ValueError("--benchmark-mode is not supported with MLX overlap")
        if scheduler.dllm_config is not None:
            raise ValueError("--benchmark-mode is not supported with diffusion LLMs")
        if hasattr(scheduler.token_to_kv_pool_allocator, "c4_attn_allocator"):
            raise ValueError(
                "--benchmark-mode is not supported with DeepSeek V4 on NPU"
            )
        if not scheduler.is_generation:
            # Non-generation (embedding/reward) models would leak synthetic
            # prefill outputs to the tokenizer (suppress_output is only honored
            # on the generation streaming path).
            raise ValueError("--benchmark-mode is only supported for generative models")
        if not scheduler.spec_algorithm.is_none():
            # The synthetic decode path is incompatible with speculative
            # decoding, and the synthetic prefill path is not guarded for it.
            raise ValueError(
                "--benchmark-mode is not supported with speculative decoding"
            )
        if scheduler.model_config.is_encoder_decoder:
            raise ValueError(
                "--benchmark-mode is not supported with encoder-decoder models"
            )
        if scheduler.model_config.is_multimodal:
            # Synthetic requests carry no multimodal inputs.
            raise ValueError("--benchmark-mode is not supported with multimodal models")
        if scheduler.enable_lora:
            raise ValueError("--benchmark-mode is not supported with LoRA")
        if scheduler.server_args.load_format == "dummy":
            # Dummy weights produce meaningless benchmark results.
            raise ValueError(
                "--benchmark-mode is not supported with dummy weights "
                "(--load-format dummy)"
            )

        return cls(scheduler)

    @classmethod
    def validate_args(cls, server_args: ServerArgs) -> None:
        """Validate self-benchmark-specific server arguments."""
        if server_args.benchmark_mode is None:
            return

        # Non-positive values collapse an axis to one point, while very large
        # values can exhaust host memory before the event loop starts.
        for name in (
            "benchmark_prefill_granularity",
            "benchmark_prefill_kv_read_granularity",
            "benchmark_decode_length_granularity",
            "benchmark_decode_batch_granularity",
        ):
            value = getattr(server_args, name)
            flag = f"--{name.replace('_', '-')}"
            if value < 1:
                raise ValueError(f"{flag} must be >= 1 when --benchmark-mode is set.")
            if value > cls.MAX_AXIS_GRANULARITY:
                raise ValueError(
                    f"{flag} must be <= {cls.MAX_AXIS_GRANULARITY} "
                    "when --benchmark-mode is set."
                )

        grid_points = {
            "prefill": server_args.benchmark_prefill_granularity
            * server_args.benchmark_prefill_kv_read_granularity,
            "decode": server_args.benchmark_decode_length_granularity
            * server_args.benchmark_decode_batch_granularity,
        }
        requested_grid_points = (
            sum(grid_points.values())
            if server_args.benchmark_mode == "agg"
            else grid_points[server_args.benchmark_mode]
        )
        if requested_grid_points > cls.MAX_GRID_POINTS:
            raise ValueError(
                f"--benchmark-mode {server_args.benchmark_mode} requests "
                f"{requested_grid_points} grid points; the maximum is "
                f"{cls.MAX_GRID_POINTS}."
            )

        if server_args.benchmark_warmup_iterations < 0:
            raise ValueError(
                "--benchmark-warmup-iterations must be >= 0 when "
                "--benchmark-mode is set."
            )

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self._decode_batch_builder = SyntheticDecodeBatchBuilder(scheduler)
        self.config = SelfBenchmarkConfig(
            mode=scheduler.server_args.benchmark_mode,
            prefill_isl_granularity=scheduler.server_args.benchmark_prefill_granularity,
            prefill_kv_read_granularity=(
                scheduler.server_args.benchmark_prefill_kv_read_granularity
            ),
            decode_length_granularity=(
                scheduler.server_args.benchmark_decode_length_granularity
            ),
            decode_batch_size_granularity=(
                scheduler.server_args.benchmark_decode_batch_granularity
            ),
            warmup_iterations=scheduler.server_args.benchmark_warmup_iterations,
            output_path=scheduler.server_args.benchmark_output_path,
        )
        self.phase = BenchmarkPhase.WARMUP
        self._grid: list[BenchmarkPoint] = []
        self._results: list[BenchmarkPointResult] = []
        self._skipped_points: list[SkippedBenchmarkPoint] = []
        self._current: Optional[BenchmarkPointResult] = None
        self._active_reqs: list[Req] = []
        self._seq = 0
        self._warmup_remaining = max(0, self.config.warmup_iterations)
        self._grid_index = 0
        # Keep output writing keyed to whether THIS rank is a real FPM rank, not
        # to the (possibly benchmark-forced) enable_fpm flag. Otherwise every TP
        # rank forced into FPM for the sweep would write a redundant JSON file.
        # _fpm_is_real_rank is set in metrics_reporter._init_fpm; fall back to
        # enable_fpm for fake schedulers in tests that don't set it.
        self._write_results = bool(
            getattr(
                scheduler,
                "_fpm_is_real_rank",
                getattr(scheduler, "enable_fpm", False),
            )
        )
        # Original per-rank enable_fpm, restored when the sweep finishes so a
        # benchmark-forced rank stops publishing afterwards.
        self._restore_enable_fpm = bool(getattr(scheduler, "enable_fpm", False)) and (
            not bool(getattr(scheduler, "_fpm_benchmark_forced", False))
        )
        self._run_id = self._make_run_id()
        self._identity = self._build_output_identity()
        self._output_path = self._rank_output_path(self.config.output_path)
        if self._write_results:
            self._invalidate_output()
        self._pending_seed_point: Optional[BenchmarkPoint] = None
        self._pending_seed_extra_key: Optional[str] = None
        self._build_grid()
        if self._warmup_remaining == 0:
            self.phase = BenchmarkPhase.SWEEP
        logger.info("Self-benchmark enabled: %s", self.config)

    @property
    def active(self) -> bool:
        return self.phase != BenchmarkPhase.DONE

    def maybe_schedule_next(self) -> None:
        if not self.active:
            return

        # Synthetic requests are owned by normal scheduler/result processing after
        # injection. Record a completed current point as soon as its tracked
        # requests finish, but do not inject more work or advertise readiness
        # until all scheduler-owned state drains.
        if self._current is not None:
            # Disaggregated prefill can mark a synthetic request finished after
            # observe_forward_pass() has already checked it. Re-check here so a
            # point completed by post-forward transfer processing can advance.
            if self._current_point_finished():
                self._save_current_point()
            else:
                return
        if self._has_inflight_work():
            return

        if self.phase == BenchmarkPhase.WARMUP:
            if self._inject_warmup() == 0:
                self.phase = BenchmarkPhase.SWEEP
            return

        if self._pending_seed_point is not None:
            self._inject_pending_seeded_prefill()
            return

        if self._grid_index >= len(self._grid):
            self._finish()
            return

        point = self._grid[self._grid_index]
        if point.point_type == "prefill" and point.kv_read_tokens > 0:
            injected = self._inject_prefill_seed(point)
            if injected > 0:
                return
            logger.warning("Skipping benchmark point with no valid seed: %s", point)
            self._skip_grid_point(point, "seed_injection_failed")
            return

        self._current = BenchmarkPointResult(point=point)
        self._active_reqs = []
        if point.point_type == "prefill":
            injected = self._inject_prefill(point=point)
        else:
            injected = self._inject_synthetic_decode(
                context_length=point.context_length, batch_size=point.batch_size
            )

        if injected == 0:
            logger.warning("Skipping benchmark point with no valid requests: %s", point)
            self._skip_grid_point(point, "request_injection_failed")

    def observe_forward_pass(
        self, batch: ScheduleBatch, fpm: Optional[ForwardPassMetrics]
    ) -> None:
        if not self.active:
            return
        if getattr(batch.forward_mode, "is_prebuilt", lambda: False)():
            return

        point_type = self._scheduled_point_type(batch, fpm)
        if point_type is None:
            return

        if self.phase == BenchmarkPhase.WARMUP:
            self._warmup_remaining -= 1
            if self._warmup_remaining <= 0:
                self.phase = BenchmarkPhase.SWEEP
                self._active_reqs = []
            return

        if self._current is None:
            return

        current_type = self._current.point.point_type
        if current_type == "prefill" and point_type == "decode":
            if self._current_point_finished():
                self._save_current_point()
            return
        elif point_type != current_type:
            return

        if fpm is not None:
            self._current.fpms.append(msgspec.to_builtins(fpm))
        if not self._current_point_finished():
            return
        self._save_current_point()

    def _current_point_finished(self) -> bool:
        if not self._active_reqs:
            return True
        return all(req.finished() for req in self._active_reqs)

    def _save_current_point(self) -> None:
        if self._current is None:
            return
        if not self._current.fpms:
            point = self._current.point
            logger.warning("Skipping benchmark point with no metrics: %s", point)
            self._skip_grid_point(point, "no_forward_pass_metrics")
            return
        self._results.append(self._current)
        self._advance_grid_point()

    def _skip_grid_point(self, point: BenchmarkPoint, reason: str) -> None:
        self._skipped_points.append(SkippedBenchmarkPoint(point=point, reason=reason))
        self._advance_grid_point()

    def _advance_grid_point(self) -> None:
        self._current = None
        self._active_reqs = []
        self._grid_index += 1

    def _build_grid(self) -> None:
        mode = self.config.mode
        disaggregation_mode = self.scheduler.disaggregation_mode

        if disaggregation_mode == DisaggregationMode.PREFILL:
            if self._supports_prefill_points() and mode in ("prefill", "agg"):
                self._build_prefill_grid()
            else:
                logger.warning(
                    "Skipping decode self-benchmark grid on disaggregated prefill worker"
                )
            logger.info("Self-benchmark grid: %d point(s)", len(self._grid))
            return

        if disaggregation_mode == DisaggregationMode.DECODE:
            if self._supports_decode_points() and mode in ("decode", "agg"):
                self._build_decode_grid()
            else:
                logger.warning(
                    "Skipping prefill self-benchmark grid on disaggregated decode worker"
                )
            logger.info("Self-benchmark grid: %d point(s)", len(self._grid))
            return

        if mode in ("prefill", "agg"):
            self._build_prefill_grid()
        if mode in ("decode", "agg"):
            self._build_decode_grid()
        logger.info("Self-benchmark grid: %d point(s)", len(self._grid))

    def _supports_prefill_points(self) -> bool:
        return self.scheduler.disaggregation_mode != DisaggregationMode.DECODE

    def _supports_decode_points(self) -> bool:
        return self.scheduler.disaggregation_mode != DisaggregationMode.PREFILL

    def _build_prefill_grid(self) -> None:
        n = max(1, self.config.prefill_isl_granularity)
        max_isl = self._max_prefill_isl()
        if max_isl < 1:
            return
        min_isl = min(10, max_isl)
        for isl in np.unique(np.linspace(min_isl, max_isl, n, dtype=int)):
            isl = int(isl)
            for kv_read_tokens in self._prefill_kv_read_points(isl):
                self._grid.append(
                    BenchmarkPoint(
                        point_type="prefill",
                        isl=isl,
                        kv_read_tokens=kv_read_tokens,
                    )
                )

    def _prefill_kv_read_points(self, isl: int) -> list[int]:
        n = max(1, self.config.prefill_kv_read_granularity)
        if n == 1:
            return [0]
        max_kv_read_tokens = self._align_prefill_kv_read_tokens(isl - 1)
        if max_kv_read_tokens < 1:
            return [0]
        raw_points = np.unique(np.linspace(0, max_kv_read_tokens, n, dtype=int))
        points = {
            self._align_prefill_kv_read_tokens(int(kv_read_tokens))
            for kv_read_tokens in raw_points
        }
        return sorted(points)

    def _align_prefill_kv_read_tokens(self, kv_read_tokens: int) -> int:
        page_size = max(1, self.scheduler.page_size)
        kv_read_tokens = max(0, kv_read_tokens)
        return kv_read_tokens // page_size * page_size

    def _build_decode_grid(self) -> None:
        n_len = max(1, self.config.decode_length_granularity)
        n_bs = max(1, self.config.decode_batch_size_granularity)
        max_ctx = self._max_decode_context_len()
        if max_ctx < 1:
            return
        ctx_lens = np.unique(np.linspace(1, max_ctx, n_len, dtype=int))
        for ctx_len_raw in ctx_lens:
            ctx_len = int(ctx_len_raw)
            max_bs = self._max_batch_size_for_context(ctx_len)
            if max_bs < 1:
                continue
            for bs in np.unique(np.linspace(1, max_bs, n_bs, dtype=int)):
                self._grid.append(
                    BenchmarkPoint(
                        point_type="decode",
                        context_length=ctx_len,
                        batch_size=int(bs),
                    )
                )

    def _max_prefill_isl(self) -> int:
        return max(
            0,
            min(
                self._max_valid_input_len(),
                self._max_prefill_forward_tokens(),
                self.scheduler.max_total_num_tokens - 2,
            ),
        )

    def _max_valid_input_len(self) -> int:
        # validate_input_length rejects requests with len >= max_req_input_len.
        return max(0, self.scheduler.max_req_input_len - 1)

    def _max_prefill_forward_tokens(self) -> int:
        # max_total_num_tokens is KV capacity, not transient forward/logits
        # headroom. Keep optional startup benchmarking within the scheduler's
        # normal prefill-forward token budget.
        return max(0, getattr(self.scheduler, "max_prefill_tokens", 0))

    def _max_decode_context_len(self) -> int:
        page_size = max(1, self.scheduler.page_size)
        max_total_budget = self.scheduler.max_total_num_tokens - page_size - 2
        max_total_for_one_decode = (max_total_budget // page_size) * page_size
        max_req_for_one_decode = self.scheduler.max_req_len - 2
        return max(
            0,
            min(
                self._max_valid_input_len(),
                max_req_for_one_decode,
                max_total_for_one_decode,
            ),
        )

    def _max_batch_size_for_context(self, context_length: int) -> int:
        max_running = max(1, getattr(self.scheduler, "max_running_requests", 1))
        max_tokens = max(1, getattr(self.scheduler, "max_total_num_tokens", 1))
        page_size = max(1, self.scheduler.page_size)
        paged_context = ((context_length + page_size - 1) // page_size) * page_size
        tokens_per_req = paged_context + page_size
        token_capped = max(1, max_tokens // max(1, tokens_per_req))
        return min(max_running, token_capped, self._max_decode_forward_batch_size())

    def _max_decode_forward_batch_size(self) -> int:
        """Return the configured decode-forward batch ceiling.

        KV capacity and request slots do not account for transient full-vocabulary
        logits. The decode graph limit is SGLang's existing memory-tuned forward
        ceiling; keeping startup diagnostics within it avoids eager-only batches
        whose logits can exceed the remaining device headroom.
        """
        max_bs = self.scheduler.server_args.cuda_graph_config.decode.max_bs
        if max_bs is None:
            raise RuntimeError(
                "Decode CUDA graph max batch size must be resolved before "
                "self-benchmark initialization"
            )
        return max(1, int(max_bs))

    def _inject_warmup(self) -> int:
        if self._supports_decode_points() and self._should_use_decode_warmup():
            return self._inject_synthetic_decode(
                context_length=min(256, self._max_decode_context_len()),
                batch_size=1,
            )
        if self._supports_prefill_points() and self.config.mode in ("prefill", "agg"):
            return self._inject_prefill(
                point=BenchmarkPoint(
                    point_type="prefill",
                    isl=min(256, self._max_prefill_isl()),
                )
            )
        return 0

    def _should_use_decode_warmup(self) -> bool:
        return self.config.mode == "decode" or (
            self.config.mode == "agg"
            and self.scheduler.disaggregation_mode == DisaggregationMode.DECODE
        )

    def _inject_prefill(
        self, point: BenchmarkPoint, extra_key: Optional[str] = None
    ) -> int:
        # Chunked prefill requests need a decode step to reach the normal request
        # finished/release path. The benchmark still records only prefill FPMs.
        return self._inject_requests(
            prompt_len=point.isl,
            max_tokens=1,
            n=1,
            extra_key=extra_key,
            track_active=True,
        )

    def _inject_prefill_seed(self, point: BenchmarkPoint) -> int:
        if point.kv_read_tokens <= 0:
            return 0
        extra_key = self._seed_extra_key()
        injected = self._inject_requests(
            prompt_len=point.kv_read_tokens,
            max_tokens=0,
            n=1,
            extra_key=extra_key,
            track_active=False,
        )
        if injected == 0:
            return 0
        self._pending_seed_point = point
        self._pending_seed_extra_key = extra_key
        return injected

    def _inject_pending_seeded_prefill(self) -> None:
        point = self._pending_seed_point
        extra_key = self._pending_seed_extra_key
        self._pending_seed_point = None
        self._pending_seed_extra_key = None
        if point is None or extra_key is None:
            return

        actual_kv_read_tokens = self._cached_kv_read_tokens_for_point(point, extra_key)
        if actual_kv_read_tokens != point.kv_read_tokens:
            logger.warning(
                "Skipping benchmark point after seed cache validation failed: "
                "point=%s expected_kv_read_tokens=%d actual_kv_read_tokens=%d",
                point,
                point.kv_read_tokens,
                actual_kv_read_tokens,
            )
            self._skip_grid_point(point, "seed_cache_validation_failed")
            return

        self._current = BenchmarkPointResult(point=point)
        self._active_reqs = []
        injected = self._inject_prefill(point=point, extra_key=extra_key)
        if injected == 0:
            logger.warning("Skipping benchmark point with no valid requests: %s", point)
            self._skip_grid_point(point, "request_injection_failed")

    def _inject_synthetic_decode(self, context_length: int, batch_size: int) -> int:
        if not self._synthetic_decode_supported():
            return 0

        max_context = self._max_decode_context_len()
        if max_context < 1:
            return 0
        context_length = max(1, min(context_length, max_context))
        batch_size = max(1, batch_size)

        if not self.scheduler.running_batch.is_empty():
            return 0

        reqs = []
        for _ in range(batch_size):
            req = self._new_synthetic_req(prompt_len=context_length, max_tokens=2)
            self.scheduler.init_req_max_new_tokens(req)
            if req.sampling_params.max_new_tokens < 2:
                logger.warning(
                    "Skipping decode benchmark request after max_new_tokens clamp: "
                    "rid=%s context_length=%d max_new_tokens=%d",
                    req.rid,
                    context_length,
                    req.sampling_params.max_new_tokens,
                )
                continue
            error_msg = validate_input_length(
                req,
                self.scheduler.max_req_input_len,
                self.scheduler.server_args.allow_auto_truncate,
            )
            if error_msg:
                logger.warning("Skipping invalid benchmark request: %s", error_msg)
                continue
            req.skip_radix_cache_insert = True
            # Model the normal post-prefill boundary: prefill has produced one
            # output token, but decode has not yet allocated KV for that token.
            req.output_ids.append(SELF_BENCHMARK_DUMMY_TOKEN_ID)
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.kv_committed_len = context_length
            req.kv_allocated_len = context_length
            req.already_computed = context_length
            reqs.append(req)

        if not reqs:
            return 0

        try:
            batch = self._decode_batch_builder.build(reqs, context_length)
        except Exception:
            # A rank-local skip would let successful peers publish running_batch
            # and enter model collectives alone. Clean up what we can, then let the
            # scheduler wrapper fail startup and tear down every rank.
            try:
                self._decode_batch_builder.cleanup(reqs)
            except Exception:
                logger.exception("Failed to clean up a synthetic decode batch")
            raise

        self.scheduler.running_batch = batch
        self._active_reqs = reqs
        return len(reqs)

    def _inject_requests(
        self,
        prompt_len: int,
        max_tokens: int,
        n: int,
        extra_key: Optional[str] = None,
        track_active: bool = True,
    ) -> int:
        max_prompt_len = self._max_valid_input_len()
        if max_prompt_len < 1:
            return 0
        prompt_len = max(1, min(prompt_len, max_prompt_len))
        injected = 0
        for _ in range(n):
            req = self._new_synthetic_req(
                prompt_len=prompt_len,
                max_tokens=max_tokens,
                extra_key=extra_key,
            )
            self.scheduler.init_req_max_new_tokens(req)
            if max_tokens > 0 and req.sampling_params.max_new_tokens < max_tokens:
                logger.warning(
                    "Skipping benchmark request after max_new_tokens clamp: "
                    "rid=%s requested=%d actual=%d",
                    req.rid,
                    max_tokens,
                    req.sampling_params.max_new_tokens,
                )
                continue
            error_msg = validate_input_length(
                req,
                self.scheduler.max_req_input_len,
                self.scheduler.server_args.allow_auto_truncate,
            )
            if error_msg:
                logger.warning("Skipping invalid benchmark request: %s", error_msg)
                continue
            # Synthetic requests use FAKE_BOOTSTRAP_HOST to avoid real disagg
            # transfer, which makes Req default to skipping cache insert. Chunked
            # prefill still needs unfinished-request cache bookkeeping so that
            # prefix_indices advance between chunks.
            req.skip_radix_cache_insert = False
            self.scheduler._add_request_to_queue(req)
            if track_active:
                self._active_reqs.append(req)
            injected += 1
        return injected

    def _new_synthetic_req(
        self, prompt_len: int, max_tokens: int, extra_key: Optional[str] = None
    ) -> Req:
        rid = f"{SELF_BENCHMARK_REQ_PREFIX}{self._seq}"
        self._seq += 1
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", [SELF_BENCHMARK_DUMMY_TOKEN_ID] * prompt_len),
            sampling_params=SamplingParams(
                max_new_tokens=max_tokens,
                stop=[],
                stop_regex=[],
                temperature=0.0,
                ignore_eos=True,
            ),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_port=self.scheduler.server_args.disaggregation_bootstrap_port,
            bootstrap_room=self._seq,
            disagg_mode=self.scheduler.disaggregation_mode,
            vocab_size=self.scheduler.model_config.vocab_size,
            metrics_collector=None,
            extra_key=extra_key or rid,
        )
        # Synthetic requests bypass Scheduler.handle_generate_request, which
        # disables input logprob computation when return_logprob is false.
        req.logprob_start_len = -1
        req.tokenizer = self.scheduler.tokenizer
        req.suppress_output = True
        return req

    def _cached_kv_read_tokens_for_point(
        self, point: BenchmarkPoint, extra_key: str
    ) -> int:
        if envs.SGLANG_RADIX_FORCE_MISS.get():
            return 0
        if getattr(self.scheduler.tree_cache, "disable", True):
            return 0
        token_ids = array("q", [SELF_BENCHMARK_DUMMY_TOKEN_ID] * point.isl)
        max_prefix_len = max(point.isl - 1, 0)
        match_result = self.scheduler.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids=token_ids[:max_prefix_len],
                    extra_key=extra_key,
                )
            )
        )
        return len(match_result.device_indices)

    def _seed_extra_key(self) -> str:
        return f"{SELF_BENCHMARK_REQ_PREFIX}kv_seed_{self._grid_index}"

    def _synthetic_decode_supported(self) -> bool:
        if not self.scheduler.is_generation:
            return False
        if self.scheduler.model_config.is_encoder_decoder:
            logger.warning(
                "Synthetic decode self-benchmark does not support encoder-decoder models"
            )
            return False
        if not self.scheduler.spec_algorithm.is_none():
            logger.warning(
                "Synthetic decode self-benchmark does not support speculative decoding"
            )
            return False
        return True

    def _has_inflight_work(self) -> bool:
        result_queue = getattr(self.scheduler, "result_queue", None)
        if result_queue:
            return True
        if getattr(self.scheduler, "chunked_req", None) is not None:
            return True
        if getattr(self.scheduler, "waiting_queue", None):
            return True
        # Requests waiting on grammar compilation are not yet in waiting_queue.
        grammar_manager = getattr(self.scheduler, "grammar_manager", None)
        if grammar_manager is not None and getattr(
            grammar_manager, "grammar_queue", None
        ):
            return True
        for queue_name in (
            "disagg_prefill_bootstrap_queue",
            "disagg_prefill_inflight_queue",
            "disagg_decode_prealloc_queue",
            "disagg_decode_transfer_queue",
        ):
            queue_owner = getattr(self.scheduler, queue_name, None)
            if queue_owner is None:
                continue
            if isinstance(queue_owner, list):
                if queue_owner:
                    return True
                continue
            queue = getattr(queue_owner, "queue", None)
            if queue:
                return True
            # The decode prealloc queue also holds retracted and not-yet-resolved
            # requests outside its main `queue`.
            if getattr(queue_owner, "retracted_queue", None):
                return True
            if getattr(queue_owner, "pending_reqs", None):
                return True
        running = getattr(self.scheduler, "running_batch", None)
        if running is not None and not running.is_empty():
            return True
        return False

    def _scheduled_point_type(
        self, batch: ScheduleBatch, fpm: Optional[ForwardPassMetrics]
    ) -> Optional[str]:
        if fpm is not None:
            scheduled = fpm.scheduled_requests
            if scheduled.num_decode_requests > 0:
                return "decode"
            if scheduled.num_prefill_requests > 0:
                return "prefill"
            return None
        if batch.forward_mode.is_decode():
            return "decode"
        if batch.forward_mode.is_extend():
            return "prefill"
        return None

    def _finish(self) -> None:
        if self.phase == BenchmarkPhase.DONE:
            return
        if self._write_results:
            self._write_output()
        self.phase = BenchmarkPhase.DONE
        self._restore_fpm_state()
        on_finish = getattr(self.scheduler, "on_self_benchmark_finished", None)
        if on_finish is not None:
            on_finish()
        logger.info("Self-benchmark completed")

    def _restore_fpm_state(self) -> None:
        """Restore FPM to its prior per-rank state after the sweep.

        FPM was turned on for every rank for the sweep's duration (so each rank
        advances WARMUP->SWEEP->DONE in lockstep). On ranks where FPM was
        benchmark-forced, tear the forced publisher/timer down and disable FPM;
        on the real FPM rank, leave production publishing intact.
        """
        reporter = getattr(self.scheduler, "metrics_reporter", None)
        # Shut down the forced publisher BEFORE flipping enable_fpm: the forced
        # teardown does not depend on enable_fpm, but flipping it first would
        # leave the publisher thread running on benchmark-forced ranks.
        if reporter is not None and hasattr(reporter, "shutdown_benchmark_forced_fpm"):
            reporter.shutdown_benchmark_forced_fpm()
        if hasattr(self.scheduler, "enable_fpm"):
            self.scheduler.enable_fpm = self._restore_enable_fpm

    def _write_output(self) -> None:
        expected_points = len(self._grid)
        completed_points = len(self._results)
        skipped_count = len(self._skipped_points)
        output = {
            "schema_version": 1,
            "scope": "local_diagnostics",
            "status": "complete",
            "valid": completed_points == expected_points and skipped_count == 0,
            "run_id": self._run_id,
            "completed_at_unix": time.time(),
            "identity": self._identity,
            "output_path": self._output_path,
            "config": asdict(self.config),
            "limits": {
                "max_num_scheduled_tokens": getattr(
                    self.scheduler, "max_prefill_tokens", None
                ),
                "max_num_running_reqs": getattr(
                    self.scheduler, "max_running_requests", None
                ),
                "max_model_len": getattr(self.scheduler, "max_req_len", None),
                "block_size": getattr(self.scheduler, "page_size", None),
                "num_gpu_blocks": getattr(self.scheduler, "max_total_num_tokens", None),
                "max_decode_forward_batch_size": (
                    self._max_decode_forward_batch_size()
                ),
            },
            "coverage": {
                "expected_points": expected_points,
                "completed_points": completed_points,
                "skipped_points": skipped_count,
            },
            "results": [
                {"point": asdict(result.point), "fpms": result.fpms}
                for result in self._results
            ],
            "skipped_points": [
                asdict(skipped_point) for skipped_point in self._skipped_points
            ],
        }
        self._atomic_write_json(self._output_path, output)
        logger.info(
            "Self-benchmark results written to %s (%d/%d point(s), %d skipped)",
            self._output_path,
            completed_points,
            expected_points,
            skipped_count,
        )

    def _invalidate_output(self) -> None:
        output_dir = os.path.dirname(self._output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        try:
            os.unlink(self._output_path)
        except FileNotFoundError:
            pass
        output = {
            "schema_version": 1,
            "scope": "local_diagnostics",
            "status": "running",
            "valid": False,
            "run_id": self._run_id,
            "started_at_unix": time.time(),
            "identity": self._identity,
            "output_path": self._output_path,
            "message": "Self-benchmark is running; previous results are invalid.",
        }
        self._atomic_write_json(self._output_path, output)

    def _atomic_write_json(self, output_path: str, output: dict) -> None:
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(output_path)
        fd, tmp = tempfile.mkstemp(
            prefix=f".{basename}.{os.getpid()}.",
            suffix=".tmp",
            dir=output_dir,
            text=True,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(output, f, indent=2)
                f.write("\n")
            os.replace(tmp, output_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _make_run_id(self) -> str:
        return getattr(self.scheduler, "instance_id", None) or uuid.uuid4().hex[:12]

    def _build_output_identity(self) -> dict:
        server_args = self.scheduler.server_args
        ps = self.scheduler.ps
        return {
            "model_path": getattr(server_args, "model_path", None),
            "served_model_name": getattr(server_args, "served_model_name", None),
            "benchmark_mode": self.config.mode,
            "disaggregation_mode": self._role_name(),
            "node_rank": getattr(server_args, "node_rank", None),
            "nnodes": getattr(server_args, "nnodes", None),
            "dp_rank": self._rank_value("dp_rank", default=0),
            "dp_size": getattr(ps, "dp_size", None),
            "tp_rank": self._rank_value("tp_rank", default=0),
            "tp_size": getattr(ps, "tp_size", None),
            "attn_tp_rank": self._rank_value("attn_tp_rank", default=0),
            "attn_tp_size": getattr(ps, "attn_tp_size", None),
            "attn_cp_rank": self._rank_value("attn_cp_rank", default=0),
            "attn_cp_size": getattr(ps, "attn_cp_size", None),
            "pid": os.getpid(),
        }

    def _rank_output_path(self, base_path: str) -> str:
        # The consumer addresses rank files by DP rank only: dp_rank 0 writes the
        # caller-assigned base path, dp_rank N writes the "_dpN" sibling. We keep
        # the filename to exactly that contract and carry the full
        # role/rank/run identity inside the file (see _build_output_identity), so
        # a consumer validates provenance from contents rather than parsing a
        # brittle path suffix. Co-located workers (e.g. disagg prefill/decode)
        # are kept distinct by the caller assigning a unique base path per
        # worker, not by namespacing the filename here.
        dp_rank = self._rank_value("dp_rank", default=0)
        if dp_rank == 0:
            return base_path
        stem, ext = os.path.splitext(base_path)
        return f"{stem}_dp{dp_rank}{ext}"

    def _role_name(self) -> str:
        role = getattr(self.scheduler, "disaggregation_mode", DisaggregationMode.NULL)
        return str(getattr(role, "value", role))

    def _rank_value(self, name: str, *, default: int) -> int:
        value = getattr(self.scheduler.ps, name, default)
        return default if value is None else value
