from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from array import array
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence

import msgspec

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler_components.benchmark_points import (
    BenchmarkPoints,
    DecodePointCandidate,
    PrefillPointCandidate,
    load_benchmark_points_file,
)
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


def _balanced_partition(
    total: int, count: int, *, unit: int = 1, minimum_units: int = 0
) -> list[int]:
    if count < 1 or unit < 1:
        raise ValueError("count and unit must be positive")
    if total < 0 or total % unit != 0:
        raise ValueError("total must be a non-negative multiple of unit")
    total_units = total // unit
    required_units = count * minimum_units
    if total_units < required_units:
        raise ValueError("total is too small for the requested minimum")
    quotient, remainder = divmod(total_units - required_units, count)
    return [
        (minimum_units + quotient + int(index < remainder)) * unit
        for index in range(count)
    ]


def _powers_of_two_up_to(limit: int) -> list[int]:
    values: list[int] = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def _uniformly_limit_axis(values: Sequence[int], max_samples: int) -> list[int]:
    if max_samples < 2:
        raise ValueError("uniform axis sample limits must be at least 2")
    if len(values) <= max_samples:
        return list(values)
    last_index = len(values) - 1
    intervals = max_samples - 1
    return [
        values[(sample * last_index + intervals // 2) // intervals]
        for sample in range(max_samples)
    ]


def _limit_cudagraph_axis(
    values: Sequence[int], capture_sizes: Sequence[int], max_samples: int
) -> list[int]:
    candidates = list(values)
    if len(candidates) <= max_samples:
        return candidates
    captures = [int(size) for size in capture_sizes if int(size) >= 1]
    if not captures:
        return _uniformly_limit_axis(candidates, max_samples)
    max_capture = max(captures)
    graph_points = [value for value in candidates if value <= max_capture]
    eager_tail = [value for value in candidates if value > max_capture]
    protect_tail = bool(eager_tail) and len(eager_tail) * 5 <= len(candidates)
    graph_budget = max_samples - len(eager_tail)
    if not protect_tail or not graph_points or graph_budget < 1:
        return _uniformly_limit_axis(candidates, max_samples)
    limited_graph = (
        [graph_points[0]]
        if graph_budget == 1
        else _uniformly_limit_axis(graph_points, graph_budget)
    )
    return limited_graph + eager_tail


def _cudagraph_axis_points(capture_sizes: Sequence[int], limit: int) -> list[int]:
    if limit < 1:
        return []
    configured = sorted({int(size) for size in capture_sizes if int(size) >= 1})
    if not configured:
        return sorted(set(_powers_of_two_up_to(limit) + [limit]))
    points: set[int] = set()
    for capture_size in (size for size in configured if size <= limit):
        points.add(capture_size)
        if capture_size < limit:
            points.add(capture_size + 1)
    if configured[-1] <= limit:
        value = configured[-1] * 2
        while value < limit:
            points.add(value)
            value *= 2
    points.add(limit)
    return sorted(points)


@dataclass
class SelfBenchmarkConfig:
    mode: str
    prefill_isl_granularity: int = 16
    prefill_kv_read_granularity: int = 1
    prefill_batch_size_granularity: int = 3
    decode_length_granularity: int = 6
    decode_batch_size_granularity: int = 6
    warmup_iterations: int = 5
    output_path: str = "/tmp/benchmark_results.json"
    points_file: Optional[str] = None


@dataclass
class BenchmarkPoint:
    point_type: str
    benchmark_id: int = 0
    total_prefill_tokens: int = 0
    total_kv_read_tokens: int = 0
    batch_size: int = 0
    expected_cudagraph_mode: str = "NONE"
    expected_capture_size: Optional[int] = None
    padding_tokens: Optional[int] = None
    sample_reasons: list[str] = field(default_factory=list)

    # Compatibility coordinates retained for existing result consumers.
    isl: int = 0
    kv_read_tokens: int = 0
    context_length: int = 0

    def __post_init__(self) -> None:
        if self.point_type == "prefill":
            if self.batch_size == 0:
                self.batch_size = 1
            if self.total_kv_read_tokens == 0 and self.kv_read_tokens:
                self.total_kv_read_tokens = self.kv_read_tokens
            if self.total_prefill_tokens == 0 and self.isl:
                self.total_prefill_tokens = max(
                    self.batch_size, self.isl - self.total_kv_read_tokens
                )
            if self.isl == 0 and self.batch_size == 1:
                self.isl = self.total_prefill_tokens + self.total_kv_read_tokens
            if self.kv_read_tokens == 0 and self.batch_size == 1:
                self.kv_read_tokens = self.total_kv_read_tokens
        elif self.point_type == "decode":
            if self.batch_size == 0:
                self.batch_size = 1
            if self.total_kv_read_tokens == 0 and self.context_length:
                self.total_kv_read_tokens = (self.context_length + 1) * self.batch_size
            if (
                self.context_length == 0
                and self.total_kv_read_tokens % self.batch_size == 0
            ):
                self.context_length = max(
                    0, self.total_kv_read_tokens // self.batch_size - 1
                )


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
            if server_args.benchmark_points_file is not None:
                raise ValueError("--benchmark-points-file requires --benchmark-mode")
            return

        explicit_points = None
        if server_args.benchmark_points_file is not None:
            explicit_points = load_benchmark_points_file(
                server_args.benchmark_points_file
            )

        if explicit_points is None:
            # Non-positive values collapse an axis to one point, while very large
            # values can exhaust host memory before the event loop starts.
            for name in (
                "benchmark_prefill_granularity",
                "benchmark_prefill_kv_read_granularity",
                "benchmark_prefill_batch_granularity",
                "benchmark_decode_length_granularity",
                "benchmark_decode_batch_granularity",
            ):
                value = getattr(server_args, name)
                flag = f"--{name.replace('_', '-')}"
                if value < 1:
                    raise ValueError(
                        f"{flag} must be >= 1 when --benchmark-mode is set."
                    )
                if value > cls.MAX_AXIS_GRANULARITY:
                    raise ValueError(
                        f"{flag} must be <= {cls.MAX_AXIS_GRANULARITY} "
                        "when --benchmark-mode is set."
                    )

            grid_points = {
                "prefill": server_args.benchmark_prefill_granularity
                * server_args.benchmark_prefill_kv_read_granularity
                * server_args.benchmark_prefill_batch_granularity,
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
            prefill_batch_size_granularity=(
                scheduler.server_args.benchmark_prefill_batch_granularity
            ),
            decode_length_granularity=(
                scheduler.server_args.benchmark_decode_length_granularity
            ),
            decode_batch_size_granularity=(
                scheduler.server_args.benchmark_decode_batch_granularity
            ),
            warmup_iterations=scheduler.server_args.benchmark_warmup_iterations,
            output_path=scheduler.server_args.benchmark_output_path,
            points_file=scheduler.server_args.benchmark_points_file,
        )
        self._explicit_points = (
            load_benchmark_points_file(self.config.points_file)
            if self.config.points_file is not None
            else None
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
        self._pending_seed_extra_keys: Optional[list[str]] = None
        self._max_generated_prefill_tokens_cache: Optional[int] = None
        self._max_feasible_decode_batch_size_cache: Optional[int] = None
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
        if point.point_type == "prefill" and point.total_kv_read_tokens > 0:
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
                context_lengths=self._decode_context_lengths(point)
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
        point = self._current.point
        if not self._current.fpms:
            logger.warning("Skipping benchmark point with no metrics: %s", point)
            self._skip_grid_point(point, "no_forward_pass_metrics")
            return
        if point.benchmark_id > 0 and len(self._current.fpms) != 1:
            self._skip_grid_point(point, "expected_exactly_one_forward_pass_metric")
            return
        if point.benchmark_id > 0:
            failure = self._fpm_validation_failure(point, self._current.fpms[0])
            if failure is not None:
                self._skip_grid_point(point, failure)
                return
        self._results.append(self._current)
        self._advance_grid_point()

    @staticmethod
    def _fpm_validation_failure(point: BenchmarkPoint, fpm: dict) -> Optional[str]:
        scheduled = fpm.get("scheduled_requests", {})
        if point.point_type == "prefill":
            expected = {
                "num_prefill_requests": point.batch_size,
                "sum_prefill_tokens": point.total_prefill_tokens,
                "sum_prefill_kv_tokens": point.total_kv_read_tokens,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
            }
        else:
            expected = {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": point.batch_size,
                "sum_decode_kv_tokens": point.total_kv_read_tokens,
            }
        actual = {name: scheduled.get(name, 0) for name in expected}
        if actual != expected:
            return f"measured_shape_mismatch expected={expected} actual={actual}"
        return None

    def _skip_grid_point(self, point: BenchmarkPoint, reason: str) -> None:
        if "explicit" in point.sample_reasons:
            raise RuntimeError(
                f"explicit benchmark point benchmark_id={point.benchmark_id} "
                f"failed at runtime: {reason}"
            )
        self._skipped_points.append(SkippedBenchmarkPoint(point=point, reason=reason))
        self._advance_grid_point()

    def _advance_grid_point(self) -> None:
        self._current = None
        self._active_reqs = []
        self._grid_index += 1

    def _build_grid(self) -> None:
        mode = self.config.mode
        prefill_enabled = self._supports_prefill_points() and mode in ("prefill", "agg")
        decode_enabled = self._supports_decode_points() and mode in ("decode", "agg")

        if self._explicit_points is not None:
            self._build_explicit_grid(
                self._explicit_points,
                prefill_enabled=prefill_enabled,
                decode_enabled=decode_enabled,
            )
        else:
            if prefill_enabled:
                self._build_prefill_grid()
            if decode_enabled:
                self._build_decode_grid()

        for benchmark_id, point in enumerate(self._grid, start=1):
            point.benchmark_id = benchmark_id
        payload = json.dumps(
            [asdict(point) for point in self._grid],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._grid_digest = hashlib.sha256(payload).hexdigest()
        logger.info(
            "Self-benchmark grid: %d point(s), digest=%s",
            len(self._grid),
            self._grid_digest,
        )

    def _build_explicit_grid(
        self,
        points: BenchmarkPoints,
        *,
        prefill_enabled: bool,
        decode_enabled: bool,
    ) -> None:
        if prefill_enabled:
            self._grid.extend(
                self._materialize_prefill_candidate(candidate, f"prefill[{index}]")
                for index, candidate in enumerate(points.prefill)
            )
        if decode_enabled:
            self._grid.extend(
                self._materialize_decode_candidate(candidate, f"decode[{index}]")
                for index, candidate in enumerate(points.decode)
            )

    def _materialize_prefill_candidate(
        self, candidate: PrefillPointCandidate, path: str
    ) -> BenchmarkPoint:
        point = self._prefill_point(
            candidate.total_prefill_tokens,
            candidate.total_kv_read_tokens,
            candidate.batch_size,
            sample_reasons=["explicit"],
        )
        if not self._prefill_point_feasible(point):
            self._raise_explicit_infeasible(path, candidate)
        return point

    def _materialize_decode_candidate(
        self, candidate: DecodePointCandidate, path: str
    ) -> BenchmarkPoint:
        point = self._decode_point(
            candidate.total_kv_read_tokens,
            candidate.batch_size,
            sample_reasons=["explicit"],
        )
        if not self._decode_point_feasible(point):
            self._raise_explicit_infeasible(path, candidate)
        return point

    def _raise_explicit_infeasible(self, path: str, candidate: object) -> None:
        raise ValueError(
            f"{path}: explicit benchmark point is infeasible: "
            f"point={candidate.model_dump()} limits={self._benchmark_limits()}"
        )

    def _supports_prefill_points(self) -> bool:
        return self.scheduler.disaggregation_mode != DisaggregationMode.DECODE

    def _supports_decode_points(self) -> bool:
        return self.scheduler.disaggregation_mode != DisaggregationMode.PREFILL

    def _build_prefill_grid(self) -> None:
        max_tokens = self._max_generated_prefill_tokens()
        capture_sizes = self._prefill_capture_sizes()
        totals = _limit_cudagraph_axis(
            _cudagraph_axis_points(capture_sizes, max_tokens),
            capture_sizes,
            max(2, self.config.prefill_isl_granularity),
        )
        points: list[BenchmarkPoint] = []
        for total_prefill_tokens in totals:
            for batch_size in self._prefill_batch_sizes(total_prefill_tokens):
                for total_kv_read_tokens in self._prefill_kv_read_points(
                    total_prefill_tokens, batch_size
                ):
                    point = self._prefill_point(
                        total_prefill_tokens,
                        total_kv_read_tokens,
                        batch_size,
                    )
                    if self._prefill_point_feasible(point):
                        points.append(point)
        self._grid.extend(
            sorted(
                points,
                key=lambda point: (
                    point.total_prefill_tokens,
                    point.batch_size,
                    point.total_kv_read_tokens,
                ),
                reverse=True,
            )
        )

    def _prefill_point(
        self,
        total_prefill_tokens: int,
        total_kv_read_tokens: int,
        batch_size: int,
        *,
        sample_reasons: Optional[list[str]] = None,
    ) -> BenchmarkPoint:
        capture_size, padding_tokens, reasons = self._cudagraph_metadata(
            total_prefill_tokens,
            self._prefill_capture_sizes(),
            self._max_generated_prefill_tokens(),
        )
        return BenchmarkPoint(
            point_type="prefill",
            total_prefill_tokens=total_prefill_tokens,
            total_kv_read_tokens=total_kv_read_tokens,
            batch_size=batch_size,
            expected_cudagraph_mode=(
                self._prefill_cudagraph_mode() if capture_size is not None else "NONE"
            ),
            expected_capture_size=capture_size,
            padding_tokens=padding_tokens,
            sample_reasons=[*(sample_reasons or []), *reasons],
        )

    def _prefill_batch_sizes(self, total_prefill_tokens: int) -> list[int]:
        upper_bound = min(
            total_prefill_tokens,
            self._available_req_slots(),
            max(1, getattr(self.scheduler, "max_running_requests", 1)),
        )
        legal = [
            batch_size
            for batch_size in range(1, upper_bound + 1)
            if self._prefill_point_feasible(
                self._prefill_point(total_prefill_tokens, 0, batch_size)
            )
        ]
        if not legal:
            return []
        maximum = legal[-1]
        presets = _powers_of_two_up_to(maximum) + [maximum]
        return sorted(set(presets))[: self.config.prefill_batch_size_granularity]

    def _prefill_kv_read_points(
        self, total_prefill_tokens: int, batch_size: int
    ) -> list[int]:
        if self.config.prefill_kv_read_granularity == 1:
            return [0]
        page_size = max(1, self.scheduler.page_size)
        max_total = self._max_prefill_kv_read_tokens(total_prefill_tokens, batch_size)
        max_blocks = max_total // page_size
        if max_blocks < batch_size:
            return [0]
        block_totals = [batch_size]
        block_totals.extend(
            value for value in _powers_of_two_up_to(max_blocks) if value >= batch_size
        )
        block_totals.append(max_blocks)
        candidates = [0, *(value * page_size for value in sorted(set(block_totals)))]
        return _uniformly_limit_axis(
            candidates,
            max(2, self.config.prefill_kv_read_granularity),
        )

    def _max_prefill_kv_read_tokens(
        self, total_prefill_tokens: int, batch_size: int
    ) -> int:
        page_size = max(1, self.scheduler.page_size)
        high = max(0, self._available_kv_tokens() // page_size)
        low = batch_size
        best = 0
        while low <= high:
            mid = (low + high) // 2
            point = self._prefill_point(
                total_prefill_tokens, mid * page_size, batch_size
            )
            if self._prefill_point_feasible(point):
                best = mid * page_size
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _prefill_lengths(
        self, point: BenchmarkPoint
    ) -> tuple[list[int], list[int], list[int]]:
        new_lengths = _balanced_partition(
            point.total_prefill_tokens,
            point.batch_size,
            minimum_units=1,
        )
        if point.total_kv_read_tokens == 0:
            kv_lengths = [0] * point.batch_size
        else:
            kv_lengths = _balanced_partition(
                point.total_kv_read_tokens,
                point.batch_size,
                unit=max(1, self.scheduler.page_size),
                minimum_units=1,
            )
        prompt_lengths = [
            new_tokens + kv_tokens
            for new_tokens, kv_tokens in zip(new_lengths, kv_lengths)
        ]
        return new_lengths, kv_lengths, prompt_lengths

    def _prefill_point_feasible(self, point: BenchmarkPoint) -> bool:
        if (
            point.total_prefill_tokens < point.batch_size
            or point.total_prefill_tokens > self._max_prefill_forward_tokens()
            or point.batch_size < 1
            or point.batch_size > self._available_req_slots()
            or point.batch_size > getattr(self.scheduler, "max_running_requests", 1)
        ):
            return False
        try:
            new_lengths, kv_lengths, prompt_lengths = self._prefill_lengths(point)
        except ValueError:
            return False
        page_size = max(1, self.scheduler.page_size)
        scheduled_new_lengths = [
            ((length + page_size - 1) // page_size) * page_size
            for length in new_lengths
        ]
        # PrefillAdder accounts the page-aligned token total against one shared
        # chunk budget. A point that changes shape during admission cannot yield
        # the single exact FPM required by self-benchmarking.
        if sum(scheduled_new_lengths) != point.total_prefill_tokens:
            return False
        chunk_size = getattr(self.scheduler.server_args, "chunked_prefill_size", None)
        if chunk_size is not None and chunk_size > 0:
            if sum(scheduled_new_lengths) > chunk_size:
                return False
        if any(
            prompt_len > self._max_valid_input_len() for prompt_len in prompt_lengths
        ):
            return False
        required = sum(
            ((prompt_len + 1 + page_size - 1) // page_size) * page_size
            for prompt_len in prompt_lengths
        )
        seed_required = sum(
            ((kv_len + page_size - 1) // page_size) * page_size for kv_len in kv_lengths
        )
        return max(required, seed_required) <= self._available_kv_tokens()

    def _build_decode_grid(self) -> None:
        max_batch_size = self._max_feasible_decode_batch_size()
        if max_batch_size < 1:
            return
        capture_sizes = self._decode_capture_sizes()
        batch_sizes = _limit_cudagraph_axis(
            _cudagraph_axis_points(capture_sizes, max_batch_size),
            capture_sizes,
            max(2, self.config.decode_batch_size_granularity),
        )
        for batch_size in batch_sizes:
            max_total = self._max_decode_kv_read_tokens(batch_size)
            minimum_total = batch_size * 2
            if max_total < minimum_total:
                continue
            totals = [minimum_total]
            totals.extend(
                value
                for value in _powers_of_two_up_to(max_total)
                if value >= minimum_total
            )
            totals.append(max_total)
            for total_kv_read_tokens in _uniformly_limit_axis(
                sorted(set(totals)),
                max(2, self.config.decode_length_granularity),
            ):
                point = self._decode_point(total_kv_read_tokens, batch_size)
                if self._decode_point_feasible(point):
                    self._grid.append(point)

    def _decode_point(
        self,
        total_kv_read_tokens: int,
        batch_size: int,
        *,
        sample_reasons: Optional[list[str]] = None,
    ) -> BenchmarkPoint:
        capture_size, padding_tokens, reasons = self._cudagraph_metadata(
            batch_size,
            self._decode_capture_sizes(),
            self._max_feasible_decode_batch_size(),
        )
        return BenchmarkPoint(
            point_type="decode",
            total_kv_read_tokens=total_kv_read_tokens,
            batch_size=batch_size,
            expected_cudagraph_mode=(
                self._decode_cudagraph_mode() if capture_size is not None else "NONE"
            ),
            expected_capture_size=capture_size,
            padding_tokens=padding_tokens,
            sample_reasons=[*(sample_reasons or []), *reasons],
        )

    def _decode_context_lengths(self, point: BenchmarkPoint) -> list[int]:
        # SGLang's decode FPM includes the current decode input token in each
        # request's KV-read length. Preload one fewer token per request so the
        # measured sum equals the canonical total_kv_read_tokens coordinate.
        measured_lengths = _balanced_partition(
            point.total_kv_read_tokens,
            point.batch_size,
            minimum_units=2,
        )
        return [length - 1 for length in measured_lengths]

    def _decode_point_feasible(self, point: BenchmarkPoint) -> bool:
        if (
            point.batch_size < 1
            or point.batch_size > self._available_req_slots()
            or point.batch_size > getattr(self.scheduler, "max_running_requests", 1)
            or point.batch_size > self._max_decode_forward_batch_size()
        ):
            return False
        try:
            context_lengths = self._decode_context_lengths(point)
        except ValueError:
            return False
        if any(
            context_length > self._max_decode_context_len()
            for context_length in context_lengths
        ):
            return False
        page_size = max(1, self.scheduler.page_size)
        required = sum(
            ((context_length + 1 + page_size - 1) // page_size) * page_size
            for context_length in context_lengths
        )
        return required <= self._available_kv_tokens()

    def _max_decode_kv_read_tokens(self, batch_size: int) -> int:
        low = batch_size * 2
        high = batch_size * (self._max_decode_context_len() + 1)
        best = 0
        while low <= high:
            mid = (low + high) // 2
            if self._decode_point_feasible(self._decode_point(mid, batch_size)):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _available_req_slots(self) -> int:
        pool = getattr(self.scheduler, "req_to_token_pool", None)
        available = getattr(pool, "available_size", None)
        if callable(available):
            return max(0, int(available()))
        return max(1, getattr(self.scheduler, "max_running_requests", 1))

    def _available_kv_tokens(self) -> int:
        allocator = getattr(self.scheduler, "token_to_kv_pool_allocator", None)
        available = getattr(allocator, "available_size", None)
        if callable(available):
            return max(0, int(available()))
        return max(0, getattr(self.scheduler, "max_total_num_tokens", 0))

    def _prefill_capture_sizes(self) -> list[int]:
        config = getattr(self.scheduler.server_args.cuda_graph_config, "prefill", None)
        return list(getattr(config, "bs", None) or [])

    def _decode_capture_sizes(self) -> list[int]:
        config = self.scheduler.server_args.cuda_graph_config.decode
        return list(getattr(config, "bs", None) or [])

    def _prefill_cudagraph_mode(self) -> str:
        config = getattr(self.scheduler.server_args.cuda_graph_config, "prefill", None)
        backend = getattr(config, "backend", None)
        return str(getattr(backend, "value", backend) or "NONE").upper()

    def _decode_cudagraph_mode(self) -> str:
        config = self.scheduler.server_args.cuda_graph_config.decode
        backend = getattr(config, "backend", None)
        return str(getattr(backend, "value", backend) or "FULL").upper()

    @staticmethod
    def _cudagraph_metadata(
        num_tokens: int, capture_sizes: Sequence[int], axis_limit: int
    ) -> tuple[Optional[int], Optional[int], list[str]]:
        captures = sorted({int(size) for size in capture_sizes if int(size) > 0})
        capture_size = next((size for size in captures if size >= num_tokens), None)
        reasons: list[str] = []
        if num_tokens in captures:
            reasons.append("capture")
        if num_tokens > 1 and num_tokens - 1 in captures:
            reasons.append("post_capture")
        if not captures:
            reasons.append("cudagraph_disabled")
            if num_tokens != axis_limit:
                reasons.append("geometric_axis")
        elif num_tokens > captures[-1]:
            reasons.append("eager_tail")
            if num_tokens != axis_limit:
                reasons.append("geometric_tail")
        if num_tokens == axis_limit:
            reasons.append("engine_limit")
        padding_tokens = capture_size - num_tokens if capture_size is not None else None
        return capture_size, padding_tokens, reasons

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
        limit = max(0, getattr(self.scheduler, "max_prefill_tokens", 0))
        chunk_size = getattr(self.scheduler.server_args, "chunked_prefill_size", None)
        if chunk_size is not None and chunk_size > 0:
            limit = min(limit, int(chunk_size))
        return limit

    def _max_generated_prefill_tokens(self) -> int:
        """Return the largest schedulable generated-grid prefill total.

        The forward-token ceiling can be larger than any legal request shape,
        for example when the per-request input limit is smaller than an even
        partition of that ceiling. Keep the generated axis endpoint feasible so
        limiting the axis never discards every large prefill sample.
        """
        if self._max_generated_prefill_tokens_cache is not None:
            return self._max_generated_prefill_tokens_cache

        upper_bound = min(
            self._max_prefill_forward_tokens(),
            self._available_req_slots() * self._max_valid_input_len(),
            max(1, getattr(self.scheduler, "max_running_requests", 1))
            * self._max_valid_input_len(),
            max(0, self._available_kv_tokens() - max(1, self.scheduler.page_size)),
        )
        maximum = 0
        for total_prefill_tokens in range(upper_bound, 0, -1):
            max_batch_size = min(
                total_prefill_tokens,
                self._available_req_slots(),
                max(1, getattr(self.scheduler, "max_running_requests", 1)),
            )
            for batch_size in range(1, max_batch_size + 1):
                candidate = BenchmarkPoint(
                    point_type="prefill",
                    total_prefill_tokens=total_prefill_tokens,
                    total_kv_read_tokens=0,
                    batch_size=batch_size,
                )
                if self._prefill_point_feasible(candidate):
                    maximum = total_prefill_tokens
                    break
            if maximum:
                break

        self._max_generated_prefill_tokens_cache = maximum
        return maximum

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

    def _max_feasible_decode_batch_size(self) -> int:
        if self._max_feasible_decode_batch_size_cache is not None:
            return self._max_feasible_decode_batch_size_cache

        upper_bound = min(
            self._available_req_slots(),
            max(1, getattr(self.scheduler, "max_running_requests", 1)),
            self._max_decode_forward_batch_size(),
        )
        maximum = 0
        for batch_size in range(upper_bound, 0, -1):
            candidate = BenchmarkPoint(
                point_type="decode",
                total_kv_read_tokens=batch_size * 2,
                batch_size=batch_size,
            )
            if self._decode_point_feasible(candidate):
                maximum = batch_size
                break
        self._max_feasible_decode_batch_size_cache = maximum
        return maximum

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
        self, point: BenchmarkPoint, extra_keys: Optional[Sequence[str]] = None
    ) -> int:
        _, _, prompt_lengths = self._prefill_lengths(point)
        return self._inject_requests(
            prompt_lens=prompt_lengths,
            max_tokens=1,
            extra_keys=extra_keys,
            track_active=True,
        )

    def _inject_prefill_seed(self, point: BenchmarkPoint) -> int:
        if point.total_kv_read_tokens <= 0:
            return 0
        _, kv_lengths, _ = self._prefill_lengths(point)
        extra_keys = [self._seed_extra_key(index) for index in range(point.batch_size)]
        injected = self._inject_requests(
            prompt_lens=kv_lengths,
            max_tokens=0,
            extra_keys=extra_keys,
            track_active=False,
        )
        if injected != point.batch_size:
            return 0
        self._pending_seed_point = point
        self._pending_seed_extra_keys = extra_keys
        return injected

    def _inject_pending_seeded_prefill(self) -> None:
        point = self._pending_seed_point
        extra_keys = self._pending_seed_extra_keys
        self._pending_seed_point = None
        self._pending_seed_extra_keys = None
        if point is None or extra_keys is None:
            return

        _, kv_lengths, prompt_lengths = self._prefill_lengths(point)
        actual_kv_read_tokens = [
            self._cached_kv_read_tokens(prompt_len, extra_key)
            for prompt_len, extra_key in zip(prompt_lengths, extra_keys)
        ]
        if actual_kv_read_tokens != kv_lengths:
            logger.warning(
                "Skipping benchmark point after seed cache validation failed: "
                "point=%s expected_kv_read_tokens=%s actual_kv_read_tokens=%s",
                point,
                kv_lengths,
                actual_kv_read_tokens,
            )
            self._skip_grid_point(point, "seed_cache_validation_failed")
            return

        self._current = BenchmarkPointResult(point=point)
        self._active_reqs = []
        injected = self._inject_prefill(point=point, extra_keys=extra_keys)
        if injected != point.batch_size:
            logger.warning("Skipping benchmark point with no valid requests: %s", point)
            self._skip_grid_point(point, "request_injection_failed")

    def _inject_synthetic_decode(
        self,
        context_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        *,
        context_lengths: Optional[Sequence[int]] = None,
    ) -> int:
        if not self._synthetic_decode_supported():
            return 0

        if context_lengths is None:
            if context_length is None or batch_size is None:
                raise ValueError("decode injection requires context lengths")
            max_context = self._max_decode_context_len()
            if max_context < 1:
                return 0
            context_lengths = [max(1, min(context_length, max_context))] * max(
                1, batch_size
            )
        else:
            context_lengths = list(context_lengths)
        if not context_lengths or not self.scheduler.running_batch.is_empty():
            return 0

        reqs = []
        for request_context_length in context_lengths:
            req = self._new_synthetic_req(
                prompt_len=request_context_length, max_tokens=2
            )
            self.scheduler.init_req_max_new_tokens(req)
            if req.sampling_params.max_new_tokens < 2:
                logger.warning(
                    "Skipping decode benchmark request after max_new_tokens clamp: "
                    "rid=%s context_length=%d max_new_tokens=%d",
                    req.rid,
                    request_context_length,
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
            req.output_ids.append(SELF_BENCHMARK_DUMMY_TOKEN_ID)
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.kv_committed_len = request_context_length
            req.kv = ReqKvInfo(
                kv_allocated_len=request_context_length,
                swa_evicted_seqlen=0,
            )
            req.already_computed = request_context_length
            reqs.append(req)

        if len(reqs) != len(context_lengths):
            return 0

        try:
            batch = self._decode_batch_builder.build(reqs, context_lengths)
        except Exception:
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
        prompt_lens: Optional[Sequence[int]] = None,
        max_tokens: int = 0,
        extra_keys: Optional[Sequence[str]] = None,
        track_active: bool = True,
        *,
        prompt_len: Optional[int] = None,
        n: Optional[int] = None,
        extra_key: Optional[str] = None,
    ) -> int:
        # Keep the scalar form for callers/tests from the first implementation.
        if prompt_lens is None:
            if prompt_len is None:
                raise ValueError("prompt_lens is required")
            prompt_lens = [prompt_len] * (n or 1)
        prompt_lens = list(prompt_lens)
        if extra_keys is None and extra_key is not None:
            extra_keys = [extra_key] * len(prompt_lens)
        if extra_keys is not None and len(extra_keys) != len(prompt_lens):
            raise ValueError("extra_keys must match prompt_lens")

        max_prompt_len = self._max_valid_input_len()
        if max_prompt_len < 1:
            return 0
        requests: list[Req] = []
        for index, requested_prompt_len in enumerate(prompt_lens):
            if requested_prompt_len < 1 or requested_prompt_len > max_prompt_len:
                return 0
            req = self._new_synthetic_req(
                prompt_len=requested_prompt_len,
                max_tokens=max_tokens,
                extra_key=extra_keys[index] if extra_keys is not None else None,
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
                return 0
            error_msg = validate_input_length(
                req,
                self.scheduler.max_req_input_len,
                self.scheduler.server_args.allow_auto_truncate,
            )
            if error_msg:
                logger.warning("Skipping invalid benchmark request: %s", error_msg)
                return 0
            req.skip_radix_cache_insert = False
            requests.append(req)

        for req in requests:
            self.scheduler._add_request_to_queue(req)
        if track_active:
            self._active_reqs.extend(requests)
        return len(requests)

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

    def _cached_kv_read_tokens(self, prompt_len: int, extra_key: str) -> int:
        if envs.SGLANG_RADIX_FORCE_MISS.get():
            return 0
        if getattr(self.scheduler.tree_cache, "disable", True):
            return 0
        token_ids = array("q", [SELF_BENCHMARK_DUMMY_TOKEN_ID] * prompt_len)
        match_result = self.scheduler.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids=token_ids[: max(prompt_len - 1, 0)],
                    extra_key=extra_key,
                )
            )
        )
        return len(match_result.device_indices)

    def _seed_extra_key(self, request_index: int = 0) -> str:
        return (
            f"{SELF_BENCHMARK_REQ_PREFIX}kv_seed_" f"{self._grid_index}_{request_index}"
        )

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

    def _benchmark_limits(self) -> dict:
        return {
            "max_num_scheduled_tokens": getattr(
                self.scheduler, "max_prefill_tokens", None
            ),
            "max_num_running_reqs": getattr(
                self.scheduler, "max_running_requests", None
            ),
            "max_model_len": getattr(self.scheduler, "max_req_len", None),
            "block_size": getattr(self.scheduler, "page_size", None),
            "kv_capacity_tokens": getattr(self.scheduler, "max_total_num_tokens", None),
            "available_kv_tokens": self._available_kv_tokens(),
            "available_request_slots": self._available_req_slots(),
            "max_decode_forward_batch_size": (self._max_decode_forward_batch_size()),
        }

    def _write_output(self) -> None:
        expected_points = len(self._grid)
        completed_points = len(self._results)
        skipped_count = len(self._skipped_points)
        output = {
            "schema_version": 2,
            "scope": "local_diagnostics",
            "status": "complete",
            "valid": completed_points == expected_points and skipped_count == 0,
            "usable": completed_points > 0 or expected_points == 0,
            "grid_digest": self._grid_digest,
            "run_id": self._run_id,
            "completed_at_unix": time.time(),
            "identity": self._identity,
            "output_path": self._output_path,
            "config": asdict(self.config),
            "limits": self._benchmark_limits(),
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
