"""Real-path profiling: measure one (batch_size, num_steps, seq_len) decode cost.

Prefill is untimed; only decode cycles (draft + verify + draft_extend) are timed.
Forward isolation mirrors ``Scheduler._forward_isolation`` + carry-over.
"""

from __future__ import annotations

import dataclasses
import logging
from array import array as _array
from typing import List, Protocol, runtime_checkable

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
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
    """Profile one (batch_size, num_steps, seq_len) point; caller must activate STS first."""

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

    def measure(self) -> float:
        """Return average decode latency (ms). Always frees resources in finally."""
        reqs, batch = self._build_batch()
        avg_ms = float("nan")
        try:
            self._run_prefill(batch)
            for _ in range(self._n_warmup):
                self._run_decode(batch)
            self._device_mod.synchronize()
            avg_ms = self._run_decode_timed(batch)
        finally:
            self._teardown(reqs)
        return avg_ms

    def _build_batch(self) -> tuple[List[Req], ScheduleBatch]:
        model_config = self._worker.model_config
        vocab_size = getattr(model_config, "vocab_size", 32000)
        max_new = self._n_warmup + self._n_measure + 8
        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=max_new,
            ignore_eos=True,
        )
        sampling_params.normalize(None)

        reqs: List[Req] = []
        for i in range(self._batch_size):
            tok = np.random.randint(
                1, max(2, vocab_size), size=self._seq_len, dtype=np.int64
            )
            input_ids = _array("q", tok.tobytes())
            req = Req(
                rid=f"spec_profile_s{self._num_steps}_b{self._batch_size}_{i}",
                origin_input_text="",
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.fill_len = len(req.full_untruncated_fill_ids)
            req.logprob_start_len = -1
            req.set_extend_input_len(req.fill_len - len(req.prefix_indices))
            reqs.append(req)

        batch = ScheduleBatch.init_new(
            reqs,
            self._worker.req_to_token_pool,
            self._worker.token_to_kv_pool_allocator,
            self._tree_cache,
            model_config,
            False,
            self._worker.speculative_algorithm,
        )
        return reqs, batch

    def _run_prefill(self, batch: ScheduleBatch) -> None:
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

    def _run_decode(self, batch: ScheduleBatch) -> None:
        batch.prepare_for_decode()
        self._run_forward_isolated(batch)

    def _run_decode_timed(self, batch: ScheduleBatch) -> float:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(self._n_measure):
            self._run_decode(batch)
        end_evt.record()
        self._device_mod.synchronize()
        return start_evt.elapsed_time(end_evt) / max(1, self._n_measure)

    def _run_forward_isolated(self, batch: ScheduleBatch) -> None:
        """Mirror scheduler forward isolation; branch on batch.is_spec_v2."""
        is_v2 = batch.is_spec_v2
        snapshot = (
            {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
            if is_v2
            else None
        )
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            batch.sampling_info = sampling_info.copy_for_forward()
        try:
            result = self._worker.forward_batch_generation(batch)
        finally:
            if is_v2:
                for name, value in snapshot.items():
                    setattr(batch, name, value)
            else:
                batch.sampling_info = sampling_info

        self._apply_carry_over(batch, result)

    def _apply_carry_over(self, batch: ScheduleBatch, result) -> None:
        """Mirror scheduler post-forward carry-over (spec_v2 vs spec_v1)."""
        if batch.is_spec_v2:
            batch.spec_info = result.next_draft_input
            if result.new_seq_lens is not None:
                batch.seq_lens = result.new_seq_lens
                if batch.seq_lens_cpu is not None:
                    batch.seq_lens_cpu = result.new_seq_lens.to("cpu")
                    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            batch.input_ids = None
        else:
            if result.next_token_ids is not None:
                batch.input_ids = result.next_token_ids.to(torch.int64)

    def _teardown(self, reqs: List[Req]) -> None:
        pool = self._worker.req_to_token_pool
        kv_alloc = self._worker.token_to_kv_pool_allocator

        for req in reqs:
            if getattr(req, "req_pool_idx", None) is None:
                continue
            # free_mamba_cache needs req_pool_idx; call before pool.free().
            if (
                getattr(req, "mamba_pool_idx", None) is not None
                and hasattr(pool, "free_mamba_cache")
            ):
                pool.free_mamba_cache(req)
            end = max(int(getattr(req, "kv_allocated_len", 0)), int(req.fill_len))
            kv_indices = pool.req_to_token[req.req_pool_idx, :end]
            kv_alloc.free(kv_indices)
            pool.free(req)
