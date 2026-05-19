# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ZMQ client proxy for TpModelWorker — used by the scheduler (CPU side).

``TpWorkerClient`` implements the same interface as ``TpModelWorker`` /
``BaseTpWorker`` but routes all heavy GPU calls through a ZMQ PAIR socket to
``TpWorkerServer`` running in the GPU process/thread.

Two modes of operation
----------------------
Phase 1 (same OS process, separate threads — current default):
    ``direct_worker_ref`` is set to the actual ``TpModelWorker`` instance.
    - Attributes that cannot be pickled (memory pools, lora_manager,
      hisparse_coordinator, etc.) are read directly from the worker object.
    - Forward passes and weight-update operations go over ZMQ so that the
      communication layer is exercised and tested.

Phase 2 (separate OS processes — future work):
    ``direct_worker_ref=None``.
    - Everything goes over ZMQ.
    - Memory-pool sharing requires CUDA IPC (torch multiprocessing).
    - HiSparse coordinator and LoRA manager would need special IPC handling.

Protocol (see tp_worker_server.py for the server side):
  Handshake  client → b"hello" | server → pickle(dict)
  Request    client → [METHOD: bytes, PAYLOAD: pickle]
  Response   server → [STATUS: b"ok"|b"error", RESULT: pickle]
"""

import logging
import pickle
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import zmq

from sglang.srt.managers.tp_worker import BaseTpWorker
from sglang.srt.managers.tp_worker_server import M_DECODE_STEP

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

STATUS_OK = b"ok"
STATUS_ERROR = b"error"


# ---------------------------------------------------------------------------
# Proxy for model_runner attributes
# ---------------------------------------------------------------------------

class ModelRunnerProxy:
    """Lightweight namespace populated from the handshake payload.

    Provides attribute-level access to the cached model-runner state without
    holding a reference to the actual GPU object.  Dynamic metrics
    (``weight_load_mem_usage``, ``graph_mem_usage``) start at 0.0 and are
    refreshed on demand via ``TpWorkerClient.refresh_mem_usage()``.
    """

    weight_load_mem_usage: float = 0.0
    graph_mem_usage: float = 0.0

    def __init__(self, attrs: dict) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class TpWorkerClient(BaseTpWorker):
    """ZMQ PAIR client that implements the BaseTpWorker interface.

    Parameters
    ----------
    ipc_name:
        ZMQ address the server is listening on (e.g. ``ipc:///tmp/tp.ipc``).
    direct_worker_ref:
        The actual ``BaseTpWorker`` instance when running in the same process.
        Used to access non-serialisable GPU objects (memory pools, etc.)
        without going through ZMQ.  Set to ``None`` for cross-process mode.
    """

    def __init__(
        self,
        ipc_name: str,
        direct_worker_ref: Optional["BaseTpWorker"] = None,
    ) -> None:
        ctx = zmq.Context()
        self._socket = ctx.socket(zmq.PAIR)
        self._socket.connect(ipc_name)

        # Phase-1: keep a direct reference to the GPU worker for non-serialisable objects
        self._direct = direct_worker_ref

        # Trigger handshake: send "hello", receive init data
        self._socket.send(b"hello")
        handshake: dict = pickle.loads(self._socket.recv())
        self._init_from_handshake(handshake)
        logger.info("TpWorkerClient connected to %s and received handshake", ipc_name)

    # ------------------------------------------------------------------
    # Handshake initialisation
    # ------------------------------------------------------------------

    def _init_from_handshake(self, hs: dict) -> None:
        self._worker_info: tuple = hs["worker_info"]
        self._is_hybrid_swa: bool = hs.get("is_hybrid_swa", False)
        self._sliding_window_size: Optional[int] = hs.get("sliding_window_size")
        self._tokens_per_layer_info: tuple = hs.get("tokens_per_layer_info", (None, None))
        self._pad_input_ids_func = hs.get("pad_input_ids_func")
        self._is_dllm: bool = hs.get("is_dllm", False)
        # model_config: same as scheduler's own model_config; stored for completeness
        self.model_config = hs.get("model_config")
        # Proxy for model-runner attributes that *are* serialisable
        self._model_runner_proxy = ModelRunnerProxy(hs.get("model_runner_attrs", {}))

        # Memory pool comes from the handshake payload (CUDA tensors pickled
        # via torch's CUDA IPC support when cross-process, or plain object
        # references when same-process because pickle just copies the reference).
        self._memory_pool: tuple = hs.get("memory_pool", (None, None))

    # ------------------------------------------------------------------
    # ZMQ RPC helper
    # ------------------------------------------------------------------

    def _rpc(self, method: bytes, **kwargs) -> Any:
        payload = pickle.dumps(kwargs)
        self._socket.send_multipart([method, payload])
        status, result_bytes = self._socket.recv_multipart()
        result = pickle.loads(result_bytes)
        if status == STATUS_ERROR:
            raise RuntimeError(f"TpWorkerServer raised an error for {method!r}: {result}")
        return result

    # ------------------------------------------------------------------
    # BaseTpWorker abstract interface
    # ------------------------------------------------------------------

    def forward_batch_generation(
        self,
        model_worker_batch: "ModelWorkerBatch",
        forward_batch=None,
        pp_proxy_tensors=None,
        is_verify: bool = False,
        skip_attn_backend_init: bool = False,
    ) -> "GenerationBatchResult":
        return self._rpc(
            b"forward_batch_generation",
            batch=model_worker_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            is_verify=is_verify,
            skip_attn_backend_init=skip_attn_backend_init,
        )

    @property
    def model_runner(self) -> "ModelRunner":
        # Phase 1: return the real model_runner so non-serialisable objects
        # (lora_manager, hisparse_coordinator, etc.) are accessible directly.
        if self._direct is not None:
            return self._direct.model_runner
        # Phase 2 (cross-process): return the proxy populated from the handshake.
        return self._model_runner_proxy  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # BaseTpWorker concrete helpers (delegated or cached)
    # ------------------------------------------------------------------

    @property
    def is_hybrid_swa(self) -> bool:
        if self._direct is not None:
            return self._direct.is_hybrid_swa
        return self._is_hybrid_swa

    @property
    def sliding_window_size(self) -> Optional[int]:
        if self._direct is not None:
            return self._direct.sliding_window_size
        return self._sliding_window_size

    def get_tokens_per_layer_info(self) -> tuple:
        if self._direct is not None:
            return self._direct.get_tokens_per_layer_info()
        return self._tokens_per_layer_info

    def get_pad_input_ids_func(self):
        if self._direct is not None:
            return self._direct.get_pad_input_ids_func()
        return self._pad_input_ids_func

    def get_memory_pool(self) -> Tuple["ReqToTokenPool", "BaseTokenToKVPoolAllocator"]:
        # Phase 1: direct reference avoids CUDA IPC complexity.
        if self._direct is not None:
            return self._direct.get_memory_pool()
        # Phase 2: memory_pool tensors were pickled via torch CUDA IPC.
        return self._memory_pool

    # ------------------------------------------------------------------
    # TpModelWorker-specific (not in BaseTpWorker abstract interface)
    # ------------------------------------------------------------------

    def get_worker_info(self) -> tuple:
        # worker_info contains the forward_stream (a CUDA Stream object).
        # Return the original tuple from the actual worker so the stream object
        # is the same Python reference as the one used by model_runner.
        if self._direct is not None:
            return self._direct.get_worker_info()
        return self._worker_info

    def is_dllm(self) -> bool:
        if self._direct is not None and hasattr(self._direct, "is_dllm"):
            return self._direct.is_dllm()
        return self._is_dllm

    def forward_batch_embedding(self, model_worker_batch: "ModelWorkerBatch"):
        return self._rpc(b"forward_batch_embedding", batch=model_worker_batch)

    def forward_batch_split_prefill(self, batch: "ScheduleBatch") -> "GenerationBatchResult":
        # ScheduleBatch may hold a cached ForwardBatch (split_forward_batch)
        # that is a GPU object — for Phase 1 (same process) this works via
        # pickle's reference semantics; cross-process would need extra work.
        return self._rpc(b"forward_batch_split_prefill", batch=batch)

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def load_lora_adapter(self, recv_req):
        return self._rpc(b"load_lora_adapter", req=recv_req)

    def unload_lora_adapter(self, recv_req):
        return self._rpc(b"unload_lora_adapter", req=recv_req)

    def load_lora_adapter_from_tensors(self, recv_req):
        return self._rpc(b"load_lora_adapter_from_tensors", req=recv_req)

    # ------------------------------------------------------------------
    # Weight update operations
    # ------------------------------------------------------------------

    def update_weights_from_disk(self, recv_req):
        return self._rpc(b"update_weights_from_disk", req=recv_req)

    def init_weights_update_group(self, recv_req):
        return self._rpc(b"init_weights_update_group", req=recv_req)

    def destroy_weights_update_group(self, recv_req):
        return self._rpc(b"destroy_weights_update_group", req=recv_req)

    def init_weights_send_group_for_remote_instance(self, recv_req):
        return self._rpc(b"init_weights_send_group_for_remote_instance", req=recv_req)

    def send_weights_to_remote_instance(self, recv_req):
        return self._rpc(b"send_weights_to_remote_instance", req=recv_req)

    def update_weights_from_distributed(self, recv_req):
        return self._rpc(b"update_weights_from_distributed", req=recv_req)

    def update_weights_from_tensor(self, recv_req):
        return self._rpc(b"update_weights_from_tensor", req=recv_req)

    def update_weights_from_ipc(self, recv_req):
        return self._rpc(b"update_weights_from_ipc", req=recv_req)

    def get_weights_by_name(self, recv_req):
        return self._rpc(b"get_weights_by_name", req=recv_req)

    # ------------------------------------------------------------------
    # HiCache and HiSparse registration
    # ------------------------------------------------------------------

    def register_hicache_layer_transfer_counter(self, counter) -> None:
        if self._direct is not None:
            # Phase 1: direct call (counter object may not be serialisable)
            self._direct.register_hicache_layer_transfer_counter(counter)
        else:
            self._rpc(b"register_hicache_layer_transfer_counter", counter=counter)

    def register_hisparse_coordinator(self, coordinator) -> None:
        if self._direct is not None:
            self._direct.register_hisparse_coordinator(coordinator)
        else:
            self._rpc(b"register_hisparse_coordinator", coordinator=coordinator)

    def set_hicache_consumer(self, consumer_index: int) -> None:
        if self._direct is not None and hasattr(self._direct, "set_hicache_consumer"):
            self._direct.set_hicache_consumer(consumer_index)

    # ------------------------------------------------------------------
    # Dynamic metrics (fetched via RPC, not cached at init)
    # ------------------------------------------------------------------

    def refresh_mem_usage(self) -> dict:
        """Fetch current GPU memory-usage metrics and update the proxy."""
        usage = self._rpc(b"get_mem_usage")
        self._model_runner_proxy.weight_load_mem_usage = usage.get(
            "weight_load_mem_usage", 0.0
        )
        self._model_runner_proxy.graph_mem_usage = usage.get("graph_mem_usage", 0.0)
        return usage

    # ------------------------------------------------------------------
    # Low-level async helpers used by TpWorkerClientGroup
    # ------------------------------------------------------------------

    def _rpc_send(self, method: bytes, **kwargs) -> None:
        """Enqueue an RPC request without waiting for the reply."""
        payload = pickle.dumps(kwargs)
        self._socket.send_multipart([method, payload])

    def _rpc_recv(self) -> Any:
        """Receive and return the reply from the most recent _rpc_send."""
        status, result_bytes = self._socket.recv_multipart()
        result = pickle.loads(result_bytes)
        if status == STATUS_ERROR:
            raise RuntimeError(f"TpWorkerServer raised an error: {result}")
        return result


# ---------------------------------------------------------------------------
# Multi-worker fan-out proxy (used when 1 scheduler drives N TP workers)
# ---------------------------------------------------------------------------

class TpWorkerClientGroup(BaseTpWorker):
    """Wraps *N* ``TpWorkerClient`` objects (one per TP rank) and fans out
    forward-pass calls to all of them simultaneously.

    Used in the Rust-path scheduler where a single scheduler process manages
    all TP workers instead of having one scheduler process per TP rank.

    Forward-pass protocol
    ---------------------
    1. Enqueue the request on every client socket (non-blocking ZMQ send).
    2. All TP-worker servers receive their message and start the forward pass.
       They synchronise with each other internally via NCCL all-reduce.
    3. Receive and discard replies from workers 1..N-1 (we only need rank 0's
       result, but we must drain all sockets to keep the PAIR protocol in sync).
    4. Return rank-0's result.

    All other calls (info queries, weight updates, LoRA, …) are forwarded
    to every worker and the rank-0 result is returned.
    """

    def __init__(self, clients: List[TpWorkerClient]) -> None:
        assert len(clients) >= 1, "TpWorkerClientGroup needs at least one client"
        self._clients = clients
        self._lead = clients[0]
        # Decode fast-path state (M_DECODE_STEP).
        # _decode_fast_valid is True once the server(s) have a cached GPU decode
        # batch from a full M_FORWARD_GENERATION with decode mode.
        self._decode_fast_valid: bool = False
        # Tuple fingerprint of req_pool_indices for O(batch) equality without
        # repeated .cpu().long()/torch.equal() per step.
        self._last_req_pool_fp: Optional[Tuple[int, ...]] = None

        # Perf counters — logged periodically. Decode-step path is the hot loop.
        self._decode_fast_hits: int = 0
        self._decode_fast_misses: int = 0
        self._decode_full_calls: int = 0
        # Wall-clock send-to-recv timing (microseconds), aggregated per window.
        self._rt_count: int = 0
        self._rt_send_sum_us: float = 0.0
        self._rt_recv_sum_us: float = 0.0
        self._rt_max_us: float = 0.0
        self._rt_window: int = 100  # log every N forward calls
        # rt_min initialized to a large value so the first sample replaces it
        self._rt_min_us: float = float("inf")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fanout_forward(self, method: bytes, **kwargs) -> Any:
        """Send *method* to all workers and return rank-0's result."""
        # Phase 1: enqueue requests on all sockets (ZMQ send is non-blocking).
        t0 = time.perf_counter()
        for client in self._clients:
            client._rpc_send(method, **kwargs)
        t_after_send = time.perf_counter()
        # Phase 2: collect replies; return rank-0's, discard the rest.
        result = self._lead._rpc_recv()
        for client in self._clients[1:]:
            client._rpc_recv()
        t_after_recv = time.perf_counter()
        send_us = (t_after_send - t0) * 1e6
        recv_us = (t_after_recv - t_after_send) * 1e6
        dt_us = send_us + recv_us
        self._rt_count += 1
        self._rt_send_sum_us += send_us
        self._rt_recv_sum_us += recv_us
        if dt_us > self._rt_max_us:
            self._rt_max_us = dt_us
        if dt_us < self._rt_min_us:
            self._rt_min_us = dt_us
        if self._rt_count >= self._rt_window:
            total = self._decode_fast_hits + self._decode_fast_misses + self._decode_full_calls
            hit_rate = (
                self._decode_fast_hits / max(1, total) * 100.0
            )
            logger.info(
                "TpWorkerClient[%d ops] us send=%.0f recv=%.0f min=%.0f max=%.0f "
                "| decode_fast hits=%d misses=%d full=%d (hit_rate=%.1f%%)",
                self._rt_count,
                self._rt_send_sum_us / self._rt_count,
                self._rt_recv_sum_us / self._rt_count,
                self._rt_min_us,
                self._rt_max_us,
                self._decode_fast_hits,
                self._decode_fast_misses,
                self._decode_full_calls,
                hit_rate,
            )
            self._rt_count = 0
            self._rt_send_sum_us = 0.0
            self._rt_recv_sum_us = 0.0
            self._rt_max_us = 0.0
            self._rt_min_us = float("inf")
            self._decode_fast_hits = 0
            self._decode_fast_misses = 0
            self._decode_full_calls = 0
        return result

    def _fanout_rpc(self, method: bytes, **kwargs) -> Any:
        """Send *method* sequentially to all workers; return rank-0's result."""
        result = None
        for i, client in enumerate(self._clients):
            r = client._rpc(method, **kwargs)
            if i == 0:
                result = r
        return result

    # ------------------------------------------------------------------
    # Async send/recv split — used by the scheduler's send-ahead overlap
    # pattern. ``_fanout_forward`` = ``_fanout_send_only`` + ``_fanout_recv``.
    # ------------------------------------------------------------------

    def _fanout_send_only(self, method: bytes, **kwargs) -> float:
        """Non-blocking send of *method* to all workers.

        Returns the perf_counter timestamp at which the send completed, used
        by ``_fanout_recv`` to compute the send-to-recv aggregator slice.
        """
        t0 = time.perf_counter()
        for client in self._clients:
            client._rpc_send(method, **kwargs)
        return t0

    def _fanout_recv(self, send_t0: float) -> Any:
        """Wait for rank-0's reply and drain the rest.

        Pair with ``_fanout_send_only``. ``send_t0`` is the timestamp returned
        by the matching send, used to keep the send/recv aggregator consistent
        with the synchronous ``_fanout_forward`` path.
        """
        t_after_send = time.perf_counter()
        result = self._lead._rpc_recv()
        for client in self._clients[1:]:
            client._rpc_recv()
        t_after_recv = time.perf_counter()
        send_us = (t_after_send - send_t0) * 1e6
        recv_us = (t_after_recv - t_after_send) * 1e6
        dt_us = send_us + recv_us
        self._rt_count += 1
        self._rt_send_sum_us += send_us
        self._rt_recv_sum_us += recv_us
        if dt_us > self._rt_max_us:
            self._rt_max_us = dt_us
        if dt_us < self._rt_min_us:
            self._rt_min_us = dt_us
        if self._rt_count >= self._rt_window:
            total = self._decode_fast_hits + self._decode_fast_misses + self._decode_full_calls
            hit_rate = self._decode_fast_hits / max(1, total) * 100.0
            logger.info(
                "TpWorkerClient[%d ops] us send=%.0f recv=%.0f min=%.0f max=%.0f "
                "| decode_fast hits=%d misses=%d full=%d (hit_rate=%.1f%%)",
                self._rt_count,
                self._rt_send_sum_us / self._rt_count,
                self._rt_recv_sum_us / self._rt_count,
                self._rt_min_us,
                self._rt_max_us,
                self._decode_fast_hits,
                self._decode_fast_misses,
                self._decode_full_calls,
                hit_rate,
            )
            self._rt_count = 0
            self._rt_send_sum_us = 0.0
            self._rt_recv_sum_us = 0.0
            self._rt_max_us = 0.0
            self._rt_min_us = float("inf")
            self._decode_fast_hits = 0
            self._decode_fast_misses = 0
            self._decode_full_calls = 0
        return result

    # ------------------------------------------------------------------
    # Decode fast-path helpers
    # ------------------------------------------------------------------

    def _req_pool_fingerprint(self, batch: "ModelWorkerBatch") -> Tuple[int, ...]:
        """Cheap fingerprint of req_pool_indices for change-detection.

        Under CpuScheduler req_pool_indices is already a CPU tensor — tolist()
        is a fast contiguous read.  Tuple equality is then O(batch) ints.
        Avoids the per-step .cpu().long() + torch.equal() pair.
        """
        rpi = batch.req_pool_indices
        if torch.is_tensor(rpi):
            return tuple(rpi.tolist())
        return tuple(rpi)

    def _batch_composition_unchanged_fp(self, fp: Tuple[int, ...]) -> bool:
        """True if the given fingerprint matches the cached one."""
        return self._last_req_pool_fp is not None and fp == self._last_req_pool_fp

    def _maybe_rehydrate_decode_reply(self, slim):
        """Rehydrate the worker's slim reply dict (decode OR prefill) into a
        ``GenerationBatchResult``.  Returns *slim* unchanged if it's already
        a full result object.
        """
        if isinstance(slim, dict) and slim.get("_slim_decode"):
            from sglang.srt.managers.utils import GenerationBatchResult
            return GenerationBatchResult(
                next_token_ids=slim["next_token_ids"],
                deferred_alloc=slim.get("deferred_alloc"),
                accept_lens=slim.get("accept_lens"),
                can_run_cuda_graph=slim.get("can_run_cuda_graph", False),
                num_accepted_drafts=slim.get("num_accepted_drafts", 0),
                num_accepted_drafts_per_req_cpu=slim.get("num_accepted_drafts_per_req_cpu"),
                logits_output=slim.get("logits_output"),
                routed_experts_output=slim.get("routed_experts_output"),
                expert_distribution_metrics=slim.get("expert_distribution_metrics"),
                next_draft_input=slim.get("next_draft_input"),
            )
        return slim

    def _build_decode_control(
        self,
        batch: "ModelWorkerBatch",
        *,
        input_slot: Optional[int] = None,
        output_slot: Optional[int] = None,
    ) -> dict:
        """Extract the delta fields needed for a decode_step control message.

        Only fields that change between consecutive decode steps are included.
        Stable fields (req_pool_indices, lora_ids, forward_mode, etc.) remain
        on the GPU worker from the cached batch.

        FutureMap wire contract — for 2-ahead pipelined decode driven from a
        Rust scheduler:
        - ``input_slot``: the worker resolves input_ids from its GPU future
          slots at this slot id, ignoring the payload's input_ids field.
          Use when the previous batch's sampled tokens haven't been
          recv'd yet.
        - ``output_slot``: the worker stores this step's sampled tokens at
          this slot id so the *next* step can refer to them by slot.
        Active Python event loops never pass these; the Rust scheduler will.
        """
        si = batch.sampling_info
        # Include sampling_info only when penalties or grammars are active;
        # otherwise the GPU-cached version is still valid.
        needs_si = si is not None and (
            getattr(getattr(si, "penalizer_orchestrator", None), "is_required", False)
            or bool(getattr(si, "grammars", None))
            or bool(batch.has_grammar)
        )
        # Skip indices_to_free unless there's something to free — saves a few
        # bytes off the wire and avoids a no-op .to(device) on the GPU side.
        itf = batch.indices_to_free
        if itf is not None and torch.is_tensor(itf) and itf.numel() == 0:
            itf = None
        control = {
            # When input_slot is set the worker ignores this and reads from
            # its future-slots dict instead. Sending None here saves the
            # pickle of a tensor that's about to be discarded.
            "input_ids": None if input_slot is not None else batch.input_ids,
            "seq_lens": batch.seq_lens,
            "seq_lens_cpu": batch.seq_lens_cpu,
            "orig_seq_lens": batch.orig_seq_lens,
            "seq_lens_sum": batch.seq_lens_sum,
            "indices_to_free": itf,
            "sampling_info": si if needs_si else None,
            # Mamba state (None for non-Mamba models — zero overhead).
            "mamba_track_indices": batch.mamba_track_indices,
            "mamba_track_mask": batch.mamba_track_mask,
            "mamba_track_seqlens": batch.mamba_track_seqlens,
            # DP-attention token counts (None for standard single-GPU).
            "global_num_tokens": batch.global_num_tokens,
            "global_num_tokens_for_logprob": batch.global_num_tokens_for_logprob,
        }
        # FutureMap fields; omitted when None so the wire stays small and
        # the worker's payload.get(...) returns None by default.
        if input_slot is not None:
            control["input_slot"] = input_slot
        if output_slot is not None:
            control["output_slot"] = output_slot
        return control

    # ------------------------------------------------------------------
    # BaseTpWorker — forward passes
    # ------------------------------------------------------------------

    def forward_batch_generation(
        self,
        model_worker_batch: "ModelWorkerBatch",
        forward_batch=None,
        pp_proxy_tensors=None,
        is_verify: bool = False,
        skip_attn_backend_init: bool = False,
    ) -> "GenerationBatchResult":
        # Decode fast path: send only the small delta control message when the
        # GPU workers already have a valid cached batch from the previous step.
        if model_worker_batch.forward_mode.is_decode():
            fp = self._req_pool_fingerprint(model_worker_batch)
            if self._decode_fast_valid and self._batch_composition_unchanged_fp(fp):
                control = self._build_decode_control(model_worker_batch)
                slim = self._fanout_forward(M_DECODE_STEP, **control)
                self._last_req_pool_fp = fp
                self._decode_fast_hits += 1
                return self._maybe_rehydrate_decode_reply(slim)
            # Fast path eligible (is_decode) but composition changed — miss.
            self._decode_fast_misses += 1
            full_method = b"forward_batch_generation"
            result = self._fanout_forward(
                full_method,
                batch=model_worker_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                is_verify=is_verify,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            self._decode_fast_valid = True
            self._last_req_pool_fp = fp
            return self._maybe_rehydrate_decode_reply(result)

        # Non-decode forward (extend / idle / mixed): full path.
        self._decode_full_calls += 1
        result = self._fanout_forward(
            b"forward_batch_generation",
            batch=model_worker_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            is_verify=is_verify,
            skip_attn_backend_init=skip_attn_backend_init,
        )
        # Non-decode (extend / idle) invalidates the GPU decode cache.
        self._decode_fast_valid = False
        self._last_req_pool_fp = None
        return self._maybe_rehydrate_decode_reply(result)

    def forward_batch_embedding(self, model_worker_batch: "ModelWorkerBatch"):
        return self._fanout_forward(b"forward_batch_embedding", batch=model_worker_batch)

    # ------------------------------------------------------------------
    # Split forward — send-only / wait pair used by the scheduler's
    # pipelined event loop. Equivalent to ``forward_batch_generation``
    # but lets the caller do other work between send and recv.
    # ------------------------------------------------------------------

    def forward_batch_generation_start(
        self,
        model_worker_batch: "ModelWorkerBatch",
        pp_proxy_tensors=None,
        is_verify: bool = False,
        skip_attn_backend_init: bool = False,
        input_slot: Optional[int] = None,
        output_slot: Optional[int] = None,
    ):
        """Send the forward request and return a handle for the matching wait.

        ``input_slot`` / ``output_slot`` (decode fast path only) plumb the
        FutureMap contract — see ``_build_decode_control``. Pass either or
        both when driving the 2-ahead pipeline; leave both ``None`` for the
        normal 1-ahead overlap (current Python default).

        Returns a tuple ``(send_t0, is_decode_fast, fingerprint, kind)``.
        """
        if model_worker_batch.forward_mode.is_decode():
            fp = self._req_pool_fingerprint(model_worker_batch)
            if self._decode_fast_valid and self._batch_composition_unchanged_fp(fp):
                control = self._build_decode_control(
                    model_worker_batch,
                    input_slot=input_slot,
                    output_slot=output_slot,
                )
                send_t0 = self._fanout_send_only(M_DECODE_STEP, **control)
                return (send_t0, True, fp, "fast")
            send_t0 = self._fanout_send_only(
                b"forward_batch_generation",
                batch=model_worker_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                is_verify=is_verify,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            return (send_t0, True, fp, "miss")
        # Non-decode: full path
        send_t0 = self._fanout_send_only(
            b"forward_batch_generation",
            batch=model_worker_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            is_verify=is_verify,
            skip_attn_backend_init=skip_attn_backend_init,
        )
        return (send_t0, False, None, "full")

    def forward_batch_generation_wait(self, handle) -> "GenerationBatchResult":
        """Wait for a previously-issued ``forward_batch_generation_start``."""
        send_t0, is_decode, fp, kind = handle
        result = self._fanout_recv(send_t0)
        # Update fast-path tracking and counters to mirror
        # ``forward_batch_generation`` semantics.
        if kind == "fast":
            self._decode_fast_hits += 1
            self._last_req_pool_fp = fp
        elif kind == "miss":
            self._decode_fast_misses += 1
            self._decode_fast_valid = True
            self._last_req_pool_fp = fp
        else:  # kind == "full" — non-decode forward
            self._decode_full_calls += 1
            self._decode_fast_valid = False
            self._last_req_pool_fp = None
        return self._maybe_rehydrate_decode_reply(result)

    def forward_batch_split_prefill(self, batch: "ScheduleBatch") -> "GenerationBatchResult":
        return self._fanout_forward(b"forward_batch_split_prefill", batch=batch)

    # ------------------------------------------------------------------
    # BaseTpWorker — info queries (delegate to rank-0)
    # ------------------------------------------------------------------

    @property
    def model_runner(self) -> "ModelRunner":
        return self._lead.model_runner

    @property
    def is_hybrid_swa(self) -> bool:
        return self._lead.is_hybrid_swa

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self._lead.sliding_window_size

    def get_tokens_per_layer_info(self) -> tuple:
        return self._lead.get_tokens_per_layer_info()

    def get_pad_input_ids_func(self):
        return self._lead.get_pad_input_ids_func()

    def get_memory_pool(self) -> Tuple["ReqToTokenPool", "BaseTokenToKVPoolAllocator"]:
        return self._lead.get_memory_pool()

    def get_worker_info(self) -> tuple:
        return self._lead.get_worker_info()

    def is_dllm(self) -> bool:
        return self._lead.is_dllm()

    @property
    def model_config(self):
        return self._lead.model_config

    # ------------------------------------------------------------------
    # BaseTpWorker — management ops (fan out to all, return rank-0)
    # ------------------------------------------------------------------

    def load_lora_adapter(self, recv_req):
        return self._fanout_rpc(b"load_lora_adapter", req=recv_req)

    def unload_lora_adapter(self, recv_req):
        return self._fanout_rpc(b"unload_lora_adapter", req=recv_req)

    def load_lora_adapter_from_tensors(self, recv_req):
        return self._fanout_rpc(b"load_lora_adapter_from_tensors", req=recv_req)

    def update_weights_from_disk(self, recv_req):
        return self._fanout_rpc(b"update_weights_from_disk", req=recv_req)

    def init_weights_update_group(self, recv_req):
        return self._fanout_rpc(b"init_weights_update_group", req=recv_req)

    def destroy_weights_update_group(self, recv_req):
        return self._fanout_rpc(b"destroy_weights_update_group", req=recv_req)

    def init_weights_send_group_for_remote_instance(self, recv_req):
        return self._fanout_rpc(b"init_weights_send_group_for_remote_instance", req=recv_req)

    def send_weights_to_remote_instance(self, recv_req):
        return self._fanout_rpc(b"send_weights_to_remote_instance", req=recv_req)

    def update_weights_from_distributed(self, recv_req):
        return self._fanout_rpc(b"update_weights_from_distributed", req=recv_req)

    def update_weights_from_tensor(self, recv_req):
        return self._fanout_rpc(b"update_weights_from_tensor", req=recv_req)

    def update_weights_from_ipc(self, recv_req):
        return self._fanout_rpc(b"update_weights_from_ipc", req=recv_req)

    def get_weights_by_name(self, recv_req):
        return self._lead._rpc(b"get_weights_by_name", req=recv_req)

    def register_hicache_layer_transfer_counter(self, counter) -> None:
        for client in self._clients:
            client.register_hicache_layer_transfer_counter(counter)

    def register_hisparse_coordinator(self, coordinator) -> None:
        for client in self._clients:
            client.register_hisparse_coordinator(coordinator)

    def refresh_mem_usage(self) -> dict:
        return self._lead.refresh_mem_usage()
