from __future__ import annotations

import json
import logging
import os
from array import array
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode, FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_req_slots,
    alloc_token_slots,
    release_kv_cache,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.observability.forward_pass_metrics import ForwardPassMetrics


logger = logging.getLogger(__name__)

SELF_BENCHMARK_REQ_PREFIX = "__sgl_bench_"


@dataclass
class SelfBenchmarkConfig:
    mode: str
    prefill_isl_granularity: int = 16
    decode_length_granularity: int = 6
    decode_batch_size_granularity: int = 6
    warmup_iterations: int = 5
    output_path: str = "/tmp/benchmark_results.json"
    timeout: int = 300


@dataclass
class BenchmarkPoint:
    point_type: str
    isl: int = 0
    context_length: int = 0
    batch_size: int = 0


@dataclass
class BenchmarkPointResult:
    point: BenchmarkPoint
    fpms: list = field(default_factory=list)


class BenchmarkPhase(Enum):
    WARMUP = "warmup"
    SWEEP = "sweep"
    DONE = "done"


class SelfBenchmark:
    """Scheduler-local self benchmark."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.config = SelfBenchmarkConfig(
            mode=scheduler.server_args.benchmark_mode,
            prefill_isl_granularity=scheduler.server_args.benchmark_prefill_granularity,
            decode_length_granularity=(
                scheduler.server_args.benchmark_decode_length_granularity
            ),
            decode_batch_size_granularity=(
                scheduler.server_args.benchmark_decode_batch_granularity
            ),
            warmup_iterations=scheduler.server_args.benchmark_warmup_iterations,
            output_path=scheduler.server_args.benchmark_output_path,
            timeout=scheduler.server_args.benchmark_timeout,
        )
        self.phase = BenchmarkPhase.WARMUP
        self._grid: list[BenchmarkPoint] = []
        self._results: list[BenchmarkPointResult] = []
        self._current: Optional[BenchmarkPointResult] = None
        self._seq = 0
        self._warmup_remaining = max(0, self.config.warmup_iterations)
        self._grid_index = 0
        self._write_results = bool(getattr(scheduler, "enable_fpm", False))
        self._build_grid()
        if self._warmup_remaining == 0:
            self.phase = BenchmarkPhase.SWEEP
        logger.info("Self-benchmark enabled: %s", self.config)

    @property
    def active(self) -> bool:
        return self.phase != BenchmarkPhase.DONE

    def maybe_schedule_next(self) -> None:
        if not self.active or self._current is not None or self._has_inflight_work():
            return

        if self.phase == BenchmarkPhase.WARMUP:
            if self._inject_warmup() == 0:
                self.phase = BenchmarkPhase.SWEEP
            return

        if self._grid_index >= len(self._grid):
            self._finish()
            return

        point = self._grid[self._grid_index]
        self._current = BenchmarkPointResult(point=point)
        if point.point_type == "prefill":
            injected = self._inject_prefill(prompt_len=point.isl, max_tokens=0)
        else:
            injected = self._inject_decode(
                context_length=point.context_length, batch_size=point.batch_size
            )

        if injected == 0:
            logger.warning("Skipping benchmark point with no valid requests: %s", point)
            self._current = None
            self._grid_index += 1

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
            return

        if self._current is None:
            return

        current_type = self._current.point.point_type
        if current_type == "decode" and point_type == "prefill":
            return
        if point_type != current_type:
            return

        if fpm is not None:
            self._current.fpms.append(msgspec.to_builtins(fpm))
        self._results.append(self._current)
        self._current = None
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
            self._grid.append(BenchmarkPoint(point_type="prefill", isl=int(isl)))

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
        chunked_prefill_size = getattr(
            self.scheduler.server_args, "chunked_prefill_size", None
        )
        if chunked_prefill_size is None or chunked_prefill_size <= 0:
            chunked_prefill_size = self._max_valid_input_len()
        return max(
            0,
            min(
                self._max_valid_input_len(),
                chunked_prefill_size,
                self.scheduler.max_total_num_tokens - 2,
            ),
        )

    def _max_valid_input_len(self) -> int:
        # validate_input_length rejects requests with len >= max_req_input_len.
        return max(0, self.scheduler.max_req_input_len - 1)

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
        return min(max_running, token_capped)

    def _inject_warmup(self) -> int:
        if self._supports_decode_points() and self._should_use_decode_warmup():
            return self._inject_decode(
                context_length=min(256, self._max_decode_context_len()),
                batch_size=1,
            )
        if self._supports_prefill_points() and self.config.mode in ("prefill", "agg"):
            return self._inject_prefill(
                prompt_len=min(256, self._max_prefill_isl()), max_tokens=0
            )
        return 0

    def _should_use_decode_warmup(self) -> bool:
        return self.config.mode == "decode" or (
            self.config.mode == "agg"
            and self.scheduler.disaggregation_mode == DisaggregationMode.DECODE
        )

    def _inject_prefill(self, prompt_len: int, max_tokens: int) -> int:
        return self._inject_requests(prompt_len=prompt_len, max_tokens=max_tokens, n=1)

    def _inject_decode(self, context_length: int, batch_size: int) -> int:
        return self._inject_synthetic_decode(
            context_length=context_length,
            batch_size=batch_size,
        )

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
            req = self._new_synthetic_req(prompt_len=context_length, max_tokens=1)
            self.scheduler.init_req_max_new_tokens(req)
            if req.sampling_params.max_new_tokens < 1:
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
            req.fill_ids = req.origin_input_ids
            req.kv_committed_len = context_length
            req.kv_allocated_len = context_length
            req.already_computed = context_length
            reqs.append(req)

        if not reqs:
            return 0

        try:
            batch = self._build_synthetic_decode_batch(reqs, context_length)
        except RuntimeError as exc:
            logger.warning(
                "Skipping decode benchmark point due to synthetic KV allocation "
                "failure: %s",
                exc,
            )
            self._free_synthetic_decode_reqs(reqs)
            return 0

        self.scheduler.running_batch = batch
        return len(reqs)

    def _inject_requests(self, prompt_len: int, max_tokens: int, n: int) -> int:
        max_prompt_len = self._max_valid_input_len()
        if max_prompt_len < 1:
            return 0
        prompt_len = max(1, min(prompt_len, max_prompt_len))
        injected = 0
        for _ in range(n):
            req = self._new_synthetic_req(prompt_len=prompt_len, max_tokens=max_tokens)
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
            self.scheduler._add_request_to_queue(req)
            injected += 1
        return injected

    def _new_synthetic_req(self, prompt_len: int, max_tokens: int) -> Req:
        token_id = self._dummy_token_id()
        rid = f"{SELF_BENCHMARK_REQ_PREFIX}{self._seq}"
        self._seq += 1
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", [token_id] * prompt_len),
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
            extra_key=rid,
        )
        req.tokenizer = self.scheduler.tokenizer
        req.suppress_output = True
        return req

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

    def _build_synthetic_decode_batch(
        self, reqs: list[Req], context_length: int
    ) -> ScheduleBatch:
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
            tree_cache=self.scheduler.tree_cache,
            model_config=self.scheduler.model_config,
            enable_overlap=self.scheduler.enable_overlap,
            spec_algorithm=self.scheduler.spec_algorithm,
            dllm_config=self.scheduler.dllm_config,
        )
        if getattr(self.scheduler, "enable_hisparse", False):
            batch.hisparse_coordinator = self.scheduler.hisparse_coordinator

        self._place_synthetic_context_cache(batch, context_length)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
        )

        last_tokens = torch.tensor(
            [req.origin_input_ids[-1] for req in reqs],
            dtype=torch.int64,
            device=batch.device,
        )
        self.scheduler.future_map.stash(batch.req_pool_indices, last_tokens)
        batch.input_ids = None
        return batch

    def _place_synthetic_context_cache(
        self, batch: ScheduleBatch, context_length: int
    ) -> None:
        reqs = batch.reqs
        req_pool_indices = alloc_req_slots(
            batch.req_to_token_pool, reqs, batch.tree_cache
        )
        req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        batch.req_pool_indices_cpu = req_pool_indices_cpu
        batch.req_pool_indices = req_pool_indices_cpu.to(
            batch.device, non_blocking=True
        )

        seq_lens_cpu = torch.full((len(reqs),), context_length, dtype=torch.int64)
        batch.seq_lens_cpu = seq_lens_cpu
        batch.seq_lens = seq_lens_cpu.to(batch.device, non_blocking=True)
        batch.orig_seq_lens = torch.full(
            (len(reqs),), context_length, dtype=torch.int32, device=batch.device
        )
        batch.seq_lens_sum = context_length * len(reqs)

        total_context_tokens = context_length * len(reqs)
        if batch.tree_cache.page_size == 1:
            context_locs = alloc_token_slots(batch.tree_cache, total_context_tokens)
        else:
            prefix_lens_cpu = torch.zeros((len(reqs),), dtype=torch.int64)
            prefix_lens = prefix_lens_cpu.to(batch.device, non_blocking=True)
            last_loc = torch.full(
                (len(reqs),), -1, dtype=torch.int64, device=batch.device
            )
            context_locs = alloc_paged_token_slots_extend(
                tree_cache=batch.tree_cache,
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=total_context_tokens,
            )

        for i, req in enumerate(reqs):
            start = i * context_length
            end = start + context_length
            batch.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, context_length)),
                context_locs[start:end].to(torch.int32),
            )
            req.synthetic_benchmark_kv_placed = True

    def _free_synthetic_decode_reqs(self, reqs: list[Req]) -> None:
        for req in reqs:
            if getattr(req, "synthetic_benchmark_kv_placed", False):
                release_kv_cache(req, self.scheduler.tree_cache, is_insert=False)
            elif req.req_pool_idx is not None:
                self.scheduler.req_to_token_pool.free(req)

    def _dummy_token_id(self) -> int:
        vocab_size = getattr(self.scheduler.model_config, "vocab_size", None) or 1
        return 0 if vocab_size > 0 else 0

    def _has_inflight_work(self) -> bool:
        result_queue = getattr(self.scheduler, "result_queue", None)
        if result_queue:
            return True
        if getattr(self.scheduler, "waiting_queue", None):
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
            queue = getattr(queue_owner, "queue", None)
            if queue:
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
        if self._write_results:
            self._write_output()
        self.phase = BenchmarkPhase.DONE
        logger.info("Self-benchmark completed")

    def _write_output(self) -> None:
        output_path = self._rank_output_path(self.config.output_path)
        output = {
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
            },
            "results": [
                {"point": asdict(result.point), "fpms": result.fpms}
                for result in self._results
            ],
        }
        tmp = output_path + ".tmp"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(output, f, indent=2)
        os.replace(tmp, output_path)
        logger.info(
            "Self-benchmark results written to %s (%d point(s))",
            output_path,
            len(self._results),
        )

    def _rank_output_path(self, base_path: str) -> str:
        dp_rank = (
            self.scheduler.ps.dp_rank if self.scheduler.ps.dp_rank is not None else 0
        )
        if dp_rank == 0:
            return base_path
        stem, ext = os.path.splitext(base_path)
        return f"{stem}_dp{dp_rank}{ext}"
