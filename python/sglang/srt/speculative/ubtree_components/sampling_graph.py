# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Full-vocabulary target sampling and graph-safe UBTree path commits."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
)
from sglang.srt.speculative.ubtree_components.epilogue import (
    UBTreeVerifyEpilogue,
    UBTreeVerifyResult,
)
from sglang.srt.utils.async_probe import sanitize_nan_logits

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def _sample_target_tokens(
    logits: torch.Tensor,
    sampling_info,
    node_count: int,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Use the ordinary FlashInfer AR sampler for every verification node.

    Keep the full vocabulary, including probability ties at the top-k cutoff.
    Joint top-k/top-p filtering differs from applying top-p after top-k
    renormalization. The min-p branch follows AR's separate renormalization
    path instead.
    """
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

    # UBTree admission currently restricts serving to one running request.
    probs = torch.softmax(logits / sampling_info.temperatures, dim=-1)
    top_ks = torch.repeat_interleave(sampling_info.top_ks, node_count, dim=0)
    top_ps = torch.repeat_interleave(sampling_info.top_ps, node_count, dim=0)
    if sampling_info.need_min_p_sampling:
        probs = top_k_renorm_prob(probs, top_ks)
        probs = top_p_renorm_prob(probs, top_ps)
        min_ps = torch.repeat_interleave(sampling_info.min_ps, node_count, dim=0)
        target_predict = min_p_sampling_from_probs(
            probs, min_ps, seed=seed, offset=offset
        )
    else:
        target_predict = top_k_top_p_sampling_from_probs(
            probs, top_ks, top_ps, filter_apply_order="joint", seed=seed, offset=offset
        )
    # FlashInfer returns int32; the tree-walk kernel consumes int64 tokens.
    return target_predict.to(torch.int64).view(1, node_count)


@dataclass
class _CapturedSamplingGraph:
    graph: torch.cuda.CUDAGraph
    # Retain the external model outputs and target Graph metadata whose
    # addresses the tail captured, as well as its private sampling buffers.
    inputs: tuple
    sampling_info: SimpleNamespace
    target_predict: torch.Tensor
    # A later prefill may replace the materializer's current workspace. Keep
    # this tail's captured storage alive independently of the target Graph.
    draft_kv_workspaces: tuple[torch.Tensor, ...]


class UBTreeSamplingGraphRunner:
    """Sample, verify, compact, and commit without recapturing the target.

    With model CUDA Graphs disabled, the same sampling and epilogue functions
    run eagerly. Sampling supports the unseeded FlashInfer backend; request
    seeds and deterministic sampling are rejected before admission.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        # Lazy capture can run after a different device becomes current.
        # Keep streams and generators on the worker's concrete CUDA device.
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._captured: dict[tuple, _CapturedSamplingGraph] = {}
        # Follow SGLang's process seed while owning this worker's RNG stream.
        # Each captured tail registers the generator so its Philox offset
        # advances on every replay, rather than repeating capture-time coins.
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(torch.initial_seed())
        self._rng_seed = torch.empty((1,), dtype=torch.int64, device=self.device)
        self._rng_offset = torch.zeros_like(self._rng_seed)

    @staticmethod
    def _stage_inputs(destination, source) -> None:
        # Always refresh the small parameter tensors. Their objects can be
        # replaced or updated in place between requests and overlap batches.
        for name in ("temperatures", "top_ks", "top_ps", "min_ps"):
            getattr(destination, name).copy_(getattr(source, name))

    def _run_tail(
        self, *, logits, hidden, sampling_info, epilogue, metadata
    ) -> torch.Tensor:
        # Some verification kernels launch on the current CUDA device rather
        # than selecting it from their tensors. This also covers eager tails.
        with torch.cuda.device(self.device):
            self._rng_seed.random_(generator=self._generator)
            target_predict = _sample_target_tokens(
                logits,
                sampling_info,
                epilogue.node_count,
                self._rng_seed,
                self._rng_offset,
            )
            req_pool_indices, seq_lens, out_cache_loc, mamba_track_indices = metadata
            epilogue.run_from_target_predict(
                target_predict=target_predict,
                hidden=hidden,
                bs=1,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                mamba_track_indices=mamba_track_indices,
                broadcast_outputs=True,
            )
            return target_predict

    def _capture(
        self, *, logits, hidden, sampling_info, epilogue, metadata
    ) -> _CapturedSamplingGraph:
        graph_sampling_info = SimpleNamespace(
            temperatures=torch.empty_like(sampling_info.temperatures),
            top_ks=torch.empty_like(sampling_info.top_ks),
            top_ps=torch.empty_like(sampling_info.top_ps),
            min_ps=torch.empty_like(sampling_info.min_ps),
            need_min_p_sampling=bool(sampling_info.need_min_p_sampling),
        )
        self._stage_inputs(graph_sampling_info, sampling_info)

        def graph_body():
            return self._run_tail(
                logits=logits,
                hidden=hidden,
                sampling_info=graph_sampling_info,
                epilogue=epilogue,
                metadata=metadata,
            )

        # Warmup and capture must not commit a path to persistent KV/KDA state.
        # The target Graph's greedy epilogue uses this same device gate.
        epilogue.set_armed(False)
        current_stream = torch.cuda.current_stream(self.device)
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(self._generator)
        # Use the same distributed capture context as the draft Graph. It
        # enables graph-safe TP communicators for sampled-path broadcasts and
        # the draft cache writer's collectives.
        with graph_capture(
            stream=get_or_create_global_graph_capture_stream()
        ) as capture_context:
            capture_stream = capture_context.stream
            for _ in range(3):
                graph_body()
            with torch.cuda.graph(graph, stream=capture_stream):
                target_predict = graph_body()
        current_stream.wait_stream(capture_stream)
        return _CapturedSamplingGraph(
            graph=graph,
            inputs=(logits, hidden, *metadata),
            sampling_info=graph_sampling_info,
            target_predict=target_predict,
            draft_kv_workspaces=epilogue.draft_kv_workspace_tensors(),
        )

    def run(
        self,
        *,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        epilogue: UBTreeVerifyEpilogue,
        forward_batch: ForwardBatch,
        can_run_cuda_graph: bool,
    ) -> UBTreeVerifyResult:
        if forward_batch.batch_size != 1 or sampling_info.is_all_greedy:
            raise ValueError("UBTREE sampling tail requires one non-greedy request")
        if sampling_info.sampling_seed is not None:
            raise ValueError("UBTREE does not support deterministic sampling yet")

        logits = logits[: epilogue.node_count]
        hidden = hidden[: epilogue.node_count]
        # Keep probes and NaN sanitization outside capture, as in the ordinary
        # sampler. The tail reads logits without modifying their temperatures.
        sanitize_nan_logits(logits, "UBTree verify: target model logits")
        if not can_run_cuda_graph:
            epilogue.set_armed(True)
            self._run_tail(
                logits=logits,
                hidden=hidden,
                sampling_info=sampling_info,
                epilogue=epilogue,
                metadata=(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.out_cache_loc,
                    forward_batch.mamba_track_indices,
                ),
            )
            return epilogue.read_result(1)

        metadata = epilogue.graph_metadata(1)
        inputs = (logits, hidden, *metadata)
        # Pointer identity matters: a recaptured model or changed track buffer
        # must not reuse a tail that still reads previous allocations.
        key = (
            id(epilogue),
            bool(sampling_info.need_min_p_sampling),
            *(
                (tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype)
                if tensor is not None
                else None
                for tensor in inputs
            ),
        )
        captured = self._captured.get(key)
        if captured is None:
            captured = self._capture(
                logits=logits,
                hidden=hidden,
                sampling_info=sampling_info,
                epilogue=epilogue,
                metadata=metadata,
            )
            self._captured[key] = captured
        self._stage_inputs(captured.sampling_info, sampling_info)
        epilogue.set_armed(True)
        captured.graph.replay()
        return epilogue.read_result(1)


__all__ = ["UBTreeSamplingGraphRunner"]
