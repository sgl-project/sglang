"""Real-path profiling for throughput-aware speculative decoding.

Prefill is untimed. Each measured decode cycle runs the same draft, verify,
and draft-extend path used by online serving, including CUDA graph selection.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
from array import array
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_schedule
from sglang.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@runtime_checkable
class ProfilableSpecWorker(Protocol):
    """Worker interface required by :class:`SpecProfilingSession`."""

    req_to_token_pool: object
    token_to_kv_pool_allocator: object
    speculative_algorithm: object
    device: str

    @property
    def model_config(self) -> object: ...

    def forward_batch_generation(self, batch: ScheduleBatch) -> object: ...


class SpecProfilingSession:
    """Measure one ``(batch_size, num_steps, seq_len)`` profile point."""

    def __init__(
        self,
        worker: ProfilableSpecWorker,
        tree_cache,
        batch_size: int,
        num_steps: int,
        seq_len: int,
        n_warmup: int,
        n_measure: int,
    ) -> None:
        self._worker = worker
        self._tree_cache = tree_cache
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._seq_len = seq_len
        self._n_warmup = n_warmup
        self._n_measure = n_measure
        self._device_mod = torch.get_device_module(worker.device)
        self._forward_iter = 0

    def measure(self) -> float:
        """Return median decode latency in milliseconds."""
        reqs = self._build_reqs()
        primary_error = None
        try:
            batch = self._run_prefill(reqs)
            for _ in range(self._n_warmup):
                self._run_decode(batch)
            self._device_mod.synchronize()
            return statistics.median(self._measure_decode_steps(batch))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self._teardown(reqs)
            except Exception:
                if primary_error is None:
                    raise
                logger.exception(
                    "Failed to clean up profiling requests while handling an "
                    "earlier profiling error"
                )

    def _build_reqs(self) -> list[Req]:
        model_config = self._worker.model_config
        vocab_size = getattr(model_config, "vocab_size", 32000)
        sampling_params = SamplingParams(
            temperature=1.0,
            max_new_tokens=self._n_warmup + self._n_measure + 8,
            ignore_eos=True,
        )
        sampling_params.normalize(None)

        reqs = []
        rng = np.random.default_rng(0)
        for index in range(self._batch_size):
            token_ids = rng.integers(
                1, max(2, vocab_size), size=self._seq_len, dtype=np.int64
            )
            req = Req(
                rid=(
                    f"spec_profile_s{self._num_steps}_" f"b{self._batch_size}_{index}"
                ),
                origin_input_text="",
                origin_input_ids=array("q", token_ids.tolist()),
                sampling_params=sampling_params,
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.init_next_round_input(self._tree_cache)
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )
            reqs.append(req)
        return reqs

    def _build_batch(self, reqs: list[Req]) -> ScheduleBatch:
        if (
            self._worker.req_to_token_pool is None
            or self._worker.token_to_kv_pool_allocator is None
        ):
            raise RuntimeError(
                "Startup profiling requires initialized target memory pools"
            )
        return ScheduleBatch.init_new(
            reqs,
            self._worker.req_to_token_pool,
            self._worker.token_to_kv_pool_allocator,
            self._tree_cache,
            self._worker.model_config,
            False,
            self._worker.speculative_algorithm,
        )

    def _run_prefill(self, reqs: list[Req]) -> ScheduleBatch:
        max_prefill_tokens = int(get_schedule().max_prefill_tokens or (2**31 - 1))
        reqs_per_batch = max_prefill_tokens // self._seq_len
        if reqs_per_batch < 1:
            raise ValueError(
                "throughput-aware profile seq_len exceeds max_prefill_tokens: "
                f"{self._seq_len} > {max_prefill_tokens}"
            )

        merged_batch = None
        for start in range(0, len(reqs), reqs_per_batch):
            batch = self._build_batch(reqs[start : start + reqs_per_batch])
            batch.prepare_for_extend()
            if (
                batch.input_ids is None
                and getattr(batch, "prefill_input_ids_cpu", None) is not None
            ):
                batch.input_ids = batch.prefill_input_ids_cpu.to(
                    self._worker.device, non_blocking=True
                )
                batch.prefill_input_ids_cpu = None
            self._run_forward_isolated(batch)
            if merged_batch is None:
                merged_batch = batch
            else:
                merged_batch.merge_batch(batch)

        assert merged_batch is not None
        return merged_batch

    def _run_decode(self, batch: ScheduleBatch):
        # The online scheduler sets this after the rank-consistent graph
        # eligibility vote. Synthetic profile batches are identical on each TP
        # rank, so the successful vote is represented directly here.
        batch.can_run_decode_cuda_graph = True
        batch.prepare_for_decode()
        result = self._run_forward_isolated(batch)
        if not result.can_run_cuda_graph:
            raise RuntimeError(
                "Throughput-aware profiling did not run on a CUDA graph: "
                f"batch_size={self._batch_size}, num_steps={self._num_steps}"
            )
        return result

    def _measure_decode_steps(self, batch: ScheduleBatch) -> list[float]:
        events = []
        for _ in range(self._n_measure):
            start = self._device_mod.Event(enable_timing=True)
            end = self._device_mod.Event(enable_timing=True)
            start.record()
            self._run_decode(batch)
            end.record()
            events.append((start, end))
        self._device_mod.synchronize()
        return [start.elapsed_time(end) for start, end in events]

    def _run_forward_isolated(self, batch: ScheduleBatch):
        """Mirror scheduler isolation and post-forward carry-over."""
        self._forward_iter += 1
        batch.forward_iter = self._forward_iter
        snapshot = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            batch.sampling_info = sampling_info.copy_for_forward()
        try:
            result = self._worker.forward_batch_generation(batch)
        finally:
            for name, value in snapshot.items():
                setattr(batch, name, value)

        batch.spec_info = result.next_draft_input
        if result.new_seq_lens is not None:
            batch.seq_lens = result.new_seq_lens
            batch.seq_lens_cpu = result.new_seq_lens.to("cpu")
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
                req.kv.kv_committed_len = int(seq_len)
        batch.input_ids = None
        return result

    def _teardown(self, reqs: list[Req]) -> None:
        errors = []
        for req in reqs:
            try:
                release_kv_cache(req, self._tree_cache, is_insert=False)
            except Exception as exc:
                errors.append(exc)
                logger.exception("Failed to free profiling request %s", req.rid)
        if errors:
            raise RuntimeError(
                f"Failed to free {len(errors)} profiling request(s)"
            ) from errors[0]
