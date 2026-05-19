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
"""ZMQ server that wraps TpModelWorker and serves forward-pass requests.

Architecture (Rust/subprocess path):
    launch_rust_server
        ├── run_gpu_worker_process (subprocess, "sglang::gpu_worker")
        │       TpModelWorker ← TpWorkerServer  (ZMQ PAIR, binds)
        │
        └── run_scheduler_process (subprocess, "sglang::scheduler")
                Scheduler → TpWorkerClient      (ZMQ PAIR, connects)

Architecture (non-Rust / thread path):
    run_scheduler_process
        └── Scheduler.init_tp_model_worker
                TpModelWorker ← TpWorkerServer  (daemon thread, ZMQ PAIR inproc)
                Scheduler → TpWorkerClient      (ZMQ PAIR, connects)

Protocol (all frames are raw bytes):
  Handshake  client→ b"hello"            server→ pickle(dict)
  Request    client→ [METHOD, pickle(kwargs)]
  Response   server→ [b"ok"|b"error", pickle(result)]

Rank-specific IPC names
-----------------------
When multiple TP / PP ranks are in play every (tp_rank, pp_rank) pair needs
its own socket.  Call ``tp_worker_ipc_for_rank`` to derive the per-rank name
from the shared base stored in ``PortArgs.tp_worker_ipc_name``.
"""

import logging
import os
import pickle
import threading
from typing import TYPE_CHECKING, Any, Optional

import torch
import zmq
import time

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank-specific IPC name helper
# ---------------------------------------------------------------------------

def tp_worker_ipc_for_rank(base_ipc: str, tp_rank: int, pp_rank: int) -> str:
    """Derive the per-rank ZMQ PAIR address from the shared base IPC name.

    Example: ``ipc:///tmp/tmpXYZ`` → ``ipc:///tmp/tmpXYZ_0_0`` for rank 0/0.
    The GPU worker subprocess binds this address; the scheduler connects to it.
    """
    return f"{base_ipc}_{tp_rank}_{pp_rank}"

# ---------------------------------------------------------------------------
# RPC method name constants (bytes to avoid repeated encoding)
# ---------------------------------------------------------------------------
M_FORWARD_GENERATION = b"forward_batch_generation"
M_FORWARD_EMBEDDING = b"forward_batch_embedding"
M_FORWARD_SPLIT_PREFILL = b"forward_batch_split_prefill"
M_DECODE_STEP = b"decode_step"  # fast-path: delta-only control message
M_LOAD_LORA = b"load_lora_adapter"
M_UNLOAD_LORA = b"unload_lora_adapter"
M_LOAD_LORA_FROM_TENSORS = b"load_lora_adapter_from_tensors"
M_UPDATE_WEIGHTS_FROM_DISK = b"update_weights_from_disk"
M_INIT_WEIGHTS_UPDATE_GROUP = b"init_weights_update_group"
M_DESTROY_WEIGHTS_UPDATE_GROUP = b"destroy_weights_update_group"
M_INIT_WEIGHTS_SEND_GROUP = b"init_weights_send_group_for_remote_instance"
M_SEND_WEIGHTS_TO_REMOTE = b"send_weights_to_remote_instance"
M_UPDATE_WEIGHTS_FROM_DISTRIBUTED = b"update_weights_from_distributed"
M_UPDATE_WEIGHTS_FROM_TENSOR = b"update_weights_from_tensor"
M_UPDATE_WEIGHTS_FROM_IPC = b"update_weights_from_ipc"
M_GET_WEIGHTS_BY_NAME = b"get_weights_by_name"
M_GET_MEM_USAGE = b"get_mem_usage"
M_REGISTER_HICACHE = b"register_hicache_layer_transfer_counter"
M_REGISTER_HISPARSE = b"register_hisparse_coordinator"

STATUS_OK = b"ok"
STATUS_OK_MSGPACK = b"ok-mp"  # Slim reply encoded with msgspec.msgpack — Rust-decodable
STATUS_ERROR = b"error"


class TpWorkerServer:
    """ZMQ PAIR server that wraps a BaseTpWorker and serves RPC requests.

    Typical usage (same-process threading, Phase 1)::

        worker = TpModelWorker(...)
        server = TpWorkerServer(worker, ipc_name="ipc:///tmp/tp_worker.ipc")
        server.run_in_thread()
        # Scheduler then creates TpWorkerClient(ipc_name, direct_worker_ref=worker)
    """

    def __init__(self, worker: "BaseTpWorker", ipc_name: str) -> None:
        self.worker = worker
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.PAIR)
        self.socket.bind(ipc_name)
        logger.info("TpWorkerServer bound to %s", ipc_name)
        # Cached GPU-resident batch for the decode fast path (M_DECODE_STEP).
        # Set after every decode M_FORWARD_GENERATION; cleared on non-decode.
        self._decode_cache: Optional["ModelWorkerBatch"] = None
        # FutureMap-style GPU-resident future slots — the wire contract for
        # cross-process 2-ahead pipelined decode. Keyed by an integer slot id
        # the scheduler chooses; value is the GPU ``next_token_ids`` tensor
        # sampled at that step.
        #
        # When the scheduler sends ``input_slot`` in a decode control message,
        # the worker reads ``input_ids`` from this map instead of expecting
        # them in the payload — so the scheduler can send batch N+1 without
        # waiting for batch N's reply.
        #
        # The active Python event loop (``event_loop_pipelined``) is 1-ahead
        # and never sets these fields, so the map stays empty in practice.
        # The Rust scheduler (in progress) is expected to set them and drive
        # 2-ahead pipelining from across the IPC boundary.
        #
        # All GPU reads / writes here ride the default CUDA stream, so the
        # natural in-order semantics give us correct "store-then-read"
        # ordering between consecutive decode steps without explicit syncs.
        self._future_tokens: dict = {}
        # When the dispatch path builds a slim reply, it also caches the typed
        # msgspec.Struct here so run_loop's send step can encode it without
        # repeating the conversion. None outside the per-step window.
        self._typed_slim_for_reply = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _build_handshake(self) -> dict:
        """Collect init-time data to send to the client in a single pickle.

        Objects that are GPU-process-local (CUDA streams, coordinators that
        hold device pointers) are replaced with ``None`` — the scheduler side
        uses ``direct_worker_ref`` for same-process access to these, and the
        cross-process path (Rust) never touches them directly.
        """
        w = self.worker
        mr = w.model_runner

        # Gather cacheable model-runner attributes; skip any that fail.
        def _safe_get(obj, attr):
            try:
                return getattr(obj, attr, None)
            except Exception:
                return None

        model_runner_attrs = {
            "lora_manager": _safe_get(mr, "lora_manager"),
            "token_table": _safe_get(mr, "token_table"),
            "linear_attn_model_spec": _safe_get(mr, "linear_attn_model_spec"),
            "hybrid_gdn_config": _safe_get(mr, "hybrid_gdn_config"),
            "mamba2_config": _safe_get(mr, "mamba2_config"),
            # CUDA Stream and coordinator objects cannot be pickled across
            # process boundaries — set to None for the cross-process case.
            "forward_stream": None,
            "hisparse_coordinator": None,
        }

        # worker_info[8] is forward_stream (a CUDA Stream); replace with None
        # so the tuple can be safely pickled for cross-process transport.
        raw_info = list(w.get_worker_info())
        raw_info[8] = None  # forward_stream
        worker_info = tuple(raw_info)

        return {
            "worker_info": worker_info,
            # "is_hybrid_swa": w.is_hybrid_swa,
            # "sliding_window_size": w.sliding_window_size,
            # "tokens_per_layer_info": w.get_tokens_per_layer_info(),
            # "pad_input_ids_func": w.get_pad_input_ids_func(),
            # "memory_pool": w.get_memory_pool(),
            "model_config": getattr(w, "model_config", None),
            "is_dllm": w.is_dllm() if hasattr(w, "is_dllm") else False,
            "model_runner_attrs": model_runner_attrs,
        }

    @staticmethod
    def _decode_step_control_to_payload(typed, ipc_to_tensor):
        """Convert a typed ``DecodeStepControl`` msgspec.Struct back to the
        dict shape ``_apply_decode_control`` expects.

        Tensors come in as ``TensorIPC`` (raw bytes + shape + dtype) and
        get materialized into CPU torch.Tensors. The legacy pickle path
        already produces CPU tensors at this point, so downstream code
        sees an identical payload.
        """
        import pickle as _pickle
        out = {
            "seq_lens": ipc_to_tensor(typed.seq_lens),
            "seq_lens_cpu": ipc_to_tensor(typed.seq_lens_cpu),
            "orig_seq_lens": ipc_to_tensor(typed.orig_seq_lens),
            "seq_lens_sum": typed.seq_lens_sum,
            "input_ids": ipc_to_tensor(typed.input_ids),
            "indices_to_free": ipc_to_tensor(typed.indices_to_free),
            "mamba_track_indices": ipc_to_tensor(typed.mamba_track_indices),
            "mamba_track_mask": ipc_to_tensor(typed.mamba_track_mask),
            "mamba_track_seqlens": ipc_to_tensor(typed.mamba_track_seqlens),
            "global_num_tokens": typed.global_num_tokens,
            "global_num_tokens_for_logprob": typed.global_num_tokens_for_logprob,
            "sampling_info": (
                _pickle.loads(typed.sampling_info_pickle)
                if typed.sampling_info_pickle is not None
                else None
            ),
        }
        if typed.input_slot is not None:
            out["input_slot"] = typed.input_slot
        if typed.output_slot is not None:
            out["output_slot"] = typed.output_slot
        return out

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def run_loop(self) -> None:
        """Blocking event loop.  Must be called from the GPU process/thread."""
        # Wait for the client's "hello" before sending the (potentially large)
        # handshake payload so no messages are lost during socket setup.
        self.socket.recv()
        handshake = self._build_handshake()
        self.socket.send(pickle.dumps(handshake))
        logger.info("TpWorkerServer handshake sent")

        # Phase 0 instrumentation: aggregate run_loop sub-step times for the
        # decode-step path so we can see where the per-step time actually goes.
        # Separate counters for decode (M_DECODE_STEP) and full (M_FORWARD_GENERATION)
        # so prefill regressions don't get hidden behind decode aggregates.
        _agg = {
            M_DECODE_STEP: {"count": 0, "recv": 0.0, "unpkl": 0.0, "disp": 0.0, "repkl": 0.0, "send": 0.0, "win": 100},
            M_FORWARD_GENERATION: {"count": 0, "recv": 0.0, "unpkl": 0.0, "disp": 0.0, "repkl": 0.0, "send": 0.0, "win": 5},
        }

        while True:
            try:
                _t0 = time.perf_counter()
                parts = self.socket.recv_multipart()
                _t1 = time.perf_counter()
            except zmq.ZMQError as exc:
                logger.error("TpWorkerServer ZMQ error: %s", exc)
                break

            if len(parts) < 1:
                continue

            method = parts[0]
            # Detect typed msgpack control for M_DECODE_STEP: a length-3
            # frame [method, b"mp", msgpack_bytes] indicates the
            # Rust-decodable wire format. Fall back to pickle otherwise.
            if (
                len(parts) >= 3
                and parts[1] == b"mp"
                and method == M_DECODE_STEP
            ):
                import msgspec
                from sglang.srt.managers.io_struct.msgpack_struct import (
                    DecodeStepControl,
                )
                from sglang.srt.managers.io_struct import ipc_to_tensor
                typed_ctrl = msgspec.msgpack.decode(parts[2], type=DecodeStepControl)
                payload = self._decode_step_control_to_payload(typed_ctrl, ipc_to_tensor)
            else:
                payload: dict = pickle.loads(parts[1]) if len(parts) > 1 else {}
            _t2 = time.perf_counter()

            try:
                result = self._dispatch(method, payload)
                _t3 = time.perf_counter()
                # Slim msgpack reply for the hot decode/forward path —
                # Rust-decodable. Other methods (handshake / LoRA / weight
                # updates / etc.) keep pickle for back-compat. Gated by env
                # so a deployment can opt out if msgspec encoding ever shows
                # a regression on its specific workload.
                use_mp = (
                    os.environ.get("SGLANG_MSGPACK_WORKER_REPLY", "1") == "1"
                    and isinstance(result, dict)
                    and result.get("_slim_decode") is True
                )
                if use_mp:
                    import msgspec
                    typed_obj = self._typed_slim_for_reply
                    self._typed_slim_for_reply = None
                    if typed_obj is None:
                        # Fallback if _typed_slim_for_reply wasn't populated
                        # (shouldn't happen on M_DECODE_STEP / M_FORWARD_GENERATION
                        # paths after this commit, but defensive).
                        reply = pickle.dumps(result)
                        self.socket.send_multipart([STATUS_OK, reply])
                    else:
                        reply = msgspec.msgpack.encode(typed_obj)
                        self.socket.send_multipart([STATUS_OK_MSGPACK, reply])
                    _t4 = time.perf_counter()
                    _t5 = _t4
                else:
                    reply = pickle.dumps(result)
                    _t4 = time.perf_counter()
                    self.socket.send_multipart([STATUS_OK, reply])
                    _t5 = time.perf_counter()
            except SystemExit:
                self.socket.send_multipart([STATUS_OK, pickle.dumps(None)])
                break
            except Exception as exc:
                logger.exception("TpWorkerServer error in method %r", method)
                self.socket.send_multipart([STATUS_ERROR, pickle.dumps(str(exc))])
                continue

            # Aggregate per-step per-method
            if method in _agg:
                a = _agg[method]
                a["count"] += 1
                a["recv"] += _t1 - _t0
                a["unpkl"] += _t2 - _t1
                a["disp"] += _t3 - _t2
                a["repkl"] += _t4 - _t3
                a["send"] += _t5 - _t4
                if a["count"] >= a["win"]:
                    n = a["count"]
                    label = "decode" if method == M_DECODE_STEP else "full"
                    logger.info(
                        "TpWorkerServer run_loop[%s %d steps] avg ms: recv_wait=%.2f "
                        "unpickle=%.2f dispatch=%.2f repickle=%.2f send=%.2f total=%.2f",
                        label, n,
                        a["recv"] / n * 1000,
                        a["unpkl"] / n * 1000,
                        a["disp"] / n * 1000,
                        a["repkl"] / n * 1000,
                        a["send"] / n * 1000,
                        (a["recv"] + a["unpkl"] + a["disp"] + a["repkl"] + a["send"]) / n * 1000,
                    )
                    win = a["win"]
                    for k in a:
                        if k == "win":
                            continue
                        a[k] = 0 if k == "count" else 0.0
                    a["win"] = win

    def run_in_thread(self) -> threading.Thread:
        """Start the event loop in a background daemon thread and return it."""
        t = threading.Thread(target=self.run_loop, daemon=True, name="TpWorkerServer")
        t.start()
        return t

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _prepare_batch(self, batch: "ModelWorkerBatch") -> None:
        """Sync CPU-scheduler state into the GPU worker before a forward pass.

        When the scheduler is CPU-only, batch tensors arrive on CPU.
        This method:
        1. Frees any KV slot indices queued by the CPU scheduler.
        2. Copies the req_to_token snapshot rows (prefix hits) into the
           worker's GPU tensor.
        3. Moves all remaining batch tensors to the worker's device.
        4. If out_cache_loc is None (deferred allocation), runs
           alloc_extend_kernel / alloc_decode_kernel on the GPU allocator
           and writes the result into batch.out_cache_loc + GPU req_to_token.
           The result is stashed as batch._deferred_alloc for the caller to
           attach to GenerationBatchResult.
        """
        if batch is None:
            return

        mr = self.worker.model_runner
        device = mr.device

        # 1. Process freed slot indices from completed/evicted requests
        if batch.indices_to_free is not None and batch.indices_to_free.numel() > 0:
            mr.token_to_kv_pool_allocator.free(batch.indices_to_free.to(device))

        # 2. Sync prefix-hit req_to_token rows (still CPU at this point)
        if batch.req_to_token_cpu is not None:
            req_pool_indices_cpu = (
                batch.req_pool_indices.cpu().long()
                if torch.is_tensor(batch.req_pool_indices)
                else torch.tensor(batch.req_pool_indices, dtype=torch.long)
            )
            mr.req_to_token_pool.req_to_token[
                req_pool_indices_cpu
            ] = batch.req_to_token_cpu.to(device)
            batch.req_to_token_cpu = None

        # 3. Move all batch tensors to GPU
        _move_model_worker_batch_to_device(batch, device)

        # 4. Deferred KV allocation (tensors are now on GPU)
        if batch.out_cache_loc is None and not batch.forward_mode.is_idle():
            batch._deferred_alloc = self._allocate_kv_deferred(batch, mr, device)
        else:
            batch._deferred_alloc = None

    def _allocate_kv_deferred(
        self,
        batch: "ModelWorkerBatch",
        mr,
        device: str,
    ) -> Optional[dict]:
        """Allocate KV slots on the GPU using Triton kernels.

        Called when out_cache_loc is None (the CPU scheduler deferred allocation
        to us).  After allocation:
          - batch.out_cache_loc is set to the GPU int64 tensor.
          - GPU req_to_token is updated with the new slot indices.
          - Returns a dict with CPU copies of the indices so the CPU scheduler
            can update its req_to_token_pool and radix tree.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        allocator = mr.token_to_kv_pool_allocator
        req_to_token_gpu = mr.req_to_token_pool.req_to_token

        req_pool_indices_gpu = batch.req_pool_indices.long()  # on device
        seq_lens_gpu = batch.seq_lens                         # on device (int32)
        seq_lens_cpu = batch.seq_lens_cpu                     # CPU (already +1 for decode)

        if batch.forward_mode.is_extend():
            # extend_prefix_lens is a Python list sent in the batch
            prefix_lens_list = batch.extend_prefix_lens or [0] * len(seq_lens_cpu)
            prefix_lens_cpu = torch.tensor(prefix_lens_list, dtype=torch.int64)
            prefix_lens_gpu = prefix_lens_cpu.to(device)
            extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu

            # last_loc: the last slot of the cached prefix for each request
            last_loc = torch.where(
                prefix_lens_gpu > 0,
                req_to_token_gpu[
                    req_pool_indices_gpu,
                    (prefix_lens_gpu - 1).clamp(min=0),
                ].to(torch.int64),
                torch.full_like(prefix_lens_gpu, -1, dtype=torch.int64),
            )

            out_cache_loc = allocator.alloc_extend(
                prefix_lens=prefix_lens_gpu,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens_gpu.to(torch.int64),
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=batch.extend_num_tokens,
            )
            if out_cache_loc is None:
                raise RuntimeError(
                    "GPU deferred KV allocation OOM during extend. "
                    "Try lowering --max-running-requests."
                )

            # Write new extend slots into GPU req_to_token.
            # Vectorized: build (row, col) index pairs once and do a single
            # fancy-indexed scatter instead of a per-request Python loop with
            # GPU sync points. Significant at prefill batches > a few reqs.
            bs = len(prefix_lens_list)
            if bs == 1:
                pre = prefix_lens_list[0]
                ext = int(extend_lens_cpu[0])
                req_to_token_gpu[
                    req_pool_indices_gpu[0], pre : pre + ext
                ] = out_cache_loc[:ext].to(torch.int32)
            else:
                extend_lens_gpu = extend_lens_cpu.to(device)
                # Row index per scattered token: repeat each req's pool index by
                # its extend length.
                row_idx = torch.repeat_interleave(req_pool_indices_gpu, extend_lens_gpu)
                # Column index per scattered token:
                #   prefix_lens[req] + (offset within that req's extend range)
                # offset = global_index - cumulative_extend_start_of_that_req.
                cumulative_extend = torch.cumsum(extend_lens_gpu, dim=0)
                slice_starts = torch.cat(
                    [torch.zeros(1, dtype=torch.int64, device=device), cumulative_extend[:-1]]
                )
                total_tokens = int(cumulative_extend[-1])
                token_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)
                slice_start_per_token = torch.repeat_interleave(slice_starts, extend_lens_gpu)
                token_offset = token_indices - slice_start_per_token
                col_idx = (
                    torch.repeat_interleave(prefix_lens_gpu, extend_lens_gpu)
                    + token_offset
                )
                req_to_token_gpu[row_idx, col_idx] = out_cache_loc.to(torch.int32)

            batch.out_cache_loc = out_cache_loc

            return {
                "mode": "extend",
                "req_pool_indices": req_pool_indices_gpu,
                "prefix_lens": prefix_lens_cpu,
                "extend_lens": extend_lens_cpu,
                "out_cache_loc": out_cache_loc,
                "free_pages_remaining": len(allocator.free_pages),
            }

        elif batch.forward_mode.is_decode():
            # seq_lens is already +1 (incremented in prepare_for_decode before
            # get_model_worker_batch is called).
            # last_loc is at position seq_lens-2 (the last allocated slot).
            last_loc = req_to_token_gpu[
                req_pool_indices_gpu,
                (seq_lens_gpu - 2).long().clamp(min=0),
            ].to(torch.int64)

            out_cache_loc = allocator.alloc_decode(
                seq_lens=seq_lens_gpu,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
            )
            if out_cache_loc is None:
                raise RuntimeError(
                    "GPU deferred KV allocation OOM during decode. "
                    "Try lowering --max-running-requests."
                )

            # Write new decode slot at position seq_lens-1
            req_to_token_gpu[
                req_pool_indices_gpu,
                (seq_lens_gpu - 1).long(),
            ] = out_cache_loc.to(torch.int32)

            batch.out_cache_loc = out_cache_loc

            return {
                "mode": "decode",
                "req_pool_indices": req_pool_indices_gpu,
                "seq_lens_minus1": (seq_lens_cpu - 1),
                "out_cache_loc": out_cache_loc,
                "free_pages_remaining": len(allocator.free_pages),
            }

        return None

    # ------------------------------------------------------------------
    # Decode fast-path helpers
    # ------------------------------------------------------------------

    def _apply_decode_control(self, payload: dict, device: str) -> "ModelWorkerBatch":
        """Apply a delta control message to the cached decode batch.

        Only the fields that change between consecutive decode steps are updated:
        input_ids, seq_lens, seq_lens_cpu, orig_seq_lens, seq_lens_sum,
        indices_to_free, and (optionally) sampling_info/mamba state.

        All stable fields — req_pool_indices, forward_mode, lora_ids, etc. —
        are left as-is from the previous step.  The GPU req_to_token is also
        left untouched: it is already correct because _allocate_kv_deferred
        wrote the new decode slot at the end of the last step.
        """
        batch = self._decode_cache
        mr = self.worker.model_runner

        # 1. Free any KV slots completed since the last step.
        indices_to_free = payload.get("indices_to_free")
        if (
            indices_to_free is not None
            and isinstance(indices_to_free, torch.Tensor)
            and indices_to_free.numel() > 0
        ):
            mr.token_to_kv_pool_allocator.free(indices_to_free.to(device))

        # 2. Update per-step delta fields (move to GPU device).
        # ``input_slot`` is the 2-ahead pipeline contract (active when a Rust
        # scheduler drives the loop): scheduler sends N+1 before recv'ing N,
        # references N's output_slot here, and we resolve input_ids from the
        # GPU buffer the previous step's sampling wrote. Same default CUDA
        # stream ⇒ correct store/read ordering. Each slot is consumed once.
        input_slot = payload.get("input_slot")
        if input_slot is not None:
            batch.input_ids = self._future_tokens.pop(input_slot)
        else:
            batch.input_ids = payload["input_ids"].to(device, non_blocking=True)
        batch.seq_lens = payload["seq_lens"].to(device, non_blocking=True)
        batch.seq_lens_cpu = payload["seq_lens_cpu"]          # stays on CPU
        batch.orig_seq_lens = payload["orig_seq_lens"].to(device, non_blocking=True)
        batch.seq_lens_sum = payload["seq_lens_sum"]
        batch.out_cache_loc = None                            # reset for deferred alloc
        batch.indices_to_free = None

        # 3. Update sampling_info when penalties / grammars changed.
        si = payload.get("sampling_info")
        if si is not None:
            batch.sampling_info = si.to(device)

        # 4. Update Mamba state if present.
        if payload.get("mamba_track_indices") is not None:
            batch.mamba_track_indices = payload["mamba_track_indices"].to(
                device, non_blocking=True
            )
        if payload.get("mamba_track_mask") is not None:
            batch.mamba_track_mask = payload["mamba_track_mask"].to(
                device, non_blocking=True
            )
        if payload.get("mamba_track_seqlens") is not None:
            batch.mamba_track_seqlens = payload["mamba_track_seqlens"].to(
                device, non_blocking=True
            )

        # 5. Update DP-attention token counts if present.
        batch.global_num_tokens = payload.get("global_num_tokens", batch.global_num_tokens)
        batch.global_num_tokens_for_logprob = payload.get(
            "global_num_tokens_for_logprob", batch.global_num_tokens_for_logprob
        )

        return batch

    def _dispatch(self, method: bytes, payload: dict) -> Any:  # noqa: C901
        w = self.worker

        if method == M_DECODE_STEP:
            if self._decode_cache is None:
                raise RuntimeError(
                    "decode_step received but no cached decode batch exists; "
                    "a full forward_batch_generation must precede the fast path."
                )
            device = w.model_runner.device
            mr = w.model_runner
            start_time = time.perf_counter()
            batch = self._apply_decode_control(payload, device)
            decode_control_time = time.perf_counter()
            # Deferred KV allocation only — skip full _prepare_batch because:
            # (a) req_to_token on GPU is already correct from the previous step's
            #     _allocate_kv_deferred write, and
            # (b) all stable batch tensors (req_pool_indices, sampling params, etc.)
            #     are already on GPU from the cached batch.
            if not batch.forward_mode.is_idle():
                batch._deferred_alloc = self._allocate_kv_deferred(batch, mr, device)
            else:
                batch._deferred_alloc = None
            allocate_kv_time = time.perf_counter()
            result = w.forward_batch_generation(batch)
            forward_batch_time = time.perf_counter()
            result.deferred_alloc = batch._deferred_alloc
            # FutureMap contract: if the scheduler asked us to publish this
            # step's sampled tokens at a slot, capture the GPU reference
            # before D2H. Keeping it in self._future_tokens prevents the
            # caching allocator from reclaiming the GPU tensor between
            # steps. The CPU copy below builds a separate tensor that goes
            # back to the scheduler on the wire; both coexist.
            output_slot = payload.get("output_slot")
            if output_slot is not None and result.next_token_ids is not None:
                self._future_tokens[output_slot] = result.next_token_ids
            _drop_ipc_unsafe_logits(result.logits_output, batch)
            _move_generation_result_to_cpu(result)
            slim_result = _build_slim_reply(result, batch)
            # Build the typed msgspec version too for the run_loop msgpack path.
            try:
                self._typed_slim_for_reply = _build_typed_slim_reply(result, batch)
            except Exception:
                # If typed build fails (unforeseen field), fall back to pickle.
                self._typed_slim_for_reply = None
            move_result_to_cpu_time = time.perf_counter()
            # Aggregate timings every N steps so we can see worker-side breakdown
            # without flooding the log.
            self._decode_step_count = getattr(self, "_decode_step_count", 0) + 1
            self._decode_step_control_sum = getattr(self, "_decode_step_control_sum", 0.0) + (decode_control_time - start_time)
            self._decode_step_alloc_sum = getattr(self, "_decode_step_alloc_sum", 0.0) + (allocate_kv_time - decode_control_time)
            self._decode_step_forward_sum = getattr(self, "_decode_step_forward_sum", 0.0) + (forward_batch_time - allocate_kv_time)
            self._decode_step_move_sum = getattr(self, "_decode_step_move_sum", 0.0) + (move_result_to_cpu_time - forward_batch_time)
            if self._decode_step_count >= 100:
                logger.info(
                    "TpWorkerServer decode_step[%d steps] avg ms: control=%.2f alloc_kv=%.2f forward=%.2f move_to_cpu=%.2f total=%.2f",
                    self._decode_step_count,
                    self._decode_step_control_sum / self._decode_step_count * 1000,
                    self._decode_step_alloc_sum / self._decode_step_count * 1000,
                    self._decode_step_forward_sum / self._decode_step_count * 1000,
                    self._decode_step_move_sum / self._decode_step_count * 1000,
                    (self._decode_step_control_sum + self._decode_step_alloc_sum +
                     self._decode_step_forward_sum + self._decode_step_move_sum) / self._decode_step_count * 1000,
                )
                self._decode_step_count = 0
                self._decode_step_control_sum = 0.0
                self._decode_step_alloc_sum = 0.0
                self._decode_step_forward_sum = 0.0
                self._decode_step_move_sum = 0.0
            return slim_result

        if method == M_FORWARD_GENERATION:
            batch = payload["batch"]
            start_time = time.perf_counter()
            self._prepare_batch(batch)
            prepare_time = time.perf_counter()
            result = w.forward_batch_generation(
                batch,
                pp_proxy_tensors=payload.get("pp_proxy_tensors"),
                is_verify=payload.get("is_verify", False),
                skip_attn_backend_init=payload.get("skip_attn_backend_init", False),
            )
            forward_batch_time = time.perf_counter()
            result.deferred_alloc = getattr(batch, "_deferred_alloc", None)
            # Cache the GPU-resident batch for subsequent decode fast-path steps.
            # Invalidate on non-decode (extend/idle) since req_pool_indices change.
            if batch.forward_mode.is_decode():
                self._decode_cache = batch
            else:
                self._decode_cache = None
            _drop_ipc_unsafe_logits(result.logits_output, batch)
            _move_generation_result_to_cpu(result)
            slim = _build_slim_reply(result, batch)
            try:
                self._typed_slim_for_reply = _build_typed_slim_reply(result, batch)
            except Exception:
                self._typed_slim_for_reply = None
            move_result_to_cpu_time = time.perf_counter()
            logger.debug(
                "TpWorkerServer forward_batch_generation times: "
                "prepare=%.3f sec, forward=%.3f sec, move_to_cpu=%.3f sec",
                prepare_time - start_time,
                forward_batch_time - prepare_time,
                move_result_to_cpu_time - forward_batch_time,
            )
            return slim

        if method == M_FORWARD_EMBEDDING:
            batch = payload["batch"]
            self._prepare_batch(batch)
            return w.forward_batch_embedding(batch)

        if method == M_FORWARD_SPLIT_PREFILL:
            batch = payload["batch"]
            self._prepare_batch(batch)
            result = w.forward_batch_split_prefill(batch)
            result.deferred_alloc = getattr(batch, "_deferred_alloc", None)
            _move_generation_result_to_cpu(result)
            return result

        if method == M_LOAD_LORA:
            return w.load_lora_adapter(payload["req"])

        if method == M_UNLOAD_LORA:
            return w.unload_lora_adapter(payload["req"])

        if method == M_LOAD_LORA_FROM_TENSORS:
            return w.load_lora_adapter_from_tensors(payload["req"])

        if method == M_UPDATE_WEIGHTS_FROM_DISK:
            return w.update_weights_from_disk(payload["req"])

        if method == M_INIT_WEIGHTS_UPDATE_GROUP:
            return w.init_weights_update_group(payload["req"])

        if method == M_DESTROY_WEIGHTS_UPDATE_GROUP:
            return w.destroy_weights_update_group(payload["req"])

        if method == M_INIT_WEIGHTS_SEND_GROUP:
            return w.init_weights_send_group_for_remote_instance(payload["req"])

        if method == M_SEND_WEIGHTS_TO_REMOTE:
            return w.send_weights_to_remote_instance(payload["req"])

        if method == M_UPDATE_WEIGHTS_FROM_DISTRIBUTED:
            return w.update_weights_from_distributed(payload["req"])

        if method == M_UPDATE_WEIGHTS_FROM_TENSOR:
            return w.update_weights_from_tensor(payload["req"])

        if method == M_UPDATE_WEIGHTS_FROM_IPC:
            return w.update_weights_from_ipc(payload["req"])

        if method == M_GET_WEIGHTS_BY_NAME:
            return w.get_weights_by_name(payload["req"])

        if method == M_GET_MEM_USAGE:
            mr = w.model_runner
            return {
                "weight_load_mem_usage": getattr(mr, "weight_load_mem_usage", 0.0),
                "graph_mem_usage": getattr(mr, "graph_mem_usage", 0.0),
            }

        if method == M_REGISTER_HICACHE:
            w.register_hicache_layer_transfer_counter(payload["counter"])
            return None

        if method == M_REGISTER_HISPARSE:
            w.register_hisparse_coordinator(payload["coordinator"])
            return None

        raise ValueError(f"TpWorkerServer: unknown RPC method {method!r}")


# ---------------------------------------------------------------------------
# Helpers: prepare a GenerationBatchResult for IPC return to the scheduler.
# ---------------------------------------------------------------------------

def _drop_ipc_unsafe_logits(lo, batch) -> None:
    """Null out fields on LogitsProcessorOutput the scheduler never reads.

    Each of these is a potentially large GPU tensor that the worker has
    finished consuming (sampling, etc.). Leaving them non-None forces them
    to ride pickle/ZMQ to the scheduler for no purpose — at batch=200,
    vocab=152k, that's ~155 MB *per step*.

    - next_token_logits: [batch, vocab_size], from sampling — never read.
    - full_logits: [batch, seq_len, vocab_size], dLLM only — never read.
    - mm_input_embeds: multimodal patches — never read in this path.
    - hidden_states: only used by spec / return_hidden_states; drop otherwise.
    """
    if lo is None:
        return
    lo.next_token_logits = None
    lo.full_logits = None
    lo.mm_input_embeds = None
    if not (
        getattr(batch, "is_spec_v2", False)
        or getattr(batch, "return_hidden_states", False)
    ):
        lo.hidden_states = None


def _build_typed_slim_reply(result, batch):
    """Build a typed ``DecodeForwardReplySlim`` from the post-D2H result.

    Returns the msgspec.Struct (the run_loop encodes it once via
    msgspec.msgpack). Falls back to ``None`` for callers that want the
    legacy dict path — e.g. when the result has unrepresentable fields
    we haven't ported yet (multimodal, certain spec paths).
    """
    import pickle as _pickle
    from sglang.srt.managers.io_struct.msgpack_struct import (
        DecodeForwardReplySlim,
        DeferredAllocIPC,
    )
    from sglang.srt.managers.io_struct import tensor_to_ipc

    da_obj = result.deferred_alloc
    da_ipc = None
    if da_obj is not None:
        # ``deferred_alloc`` is a dict from _allocate_kv_deferred; the
        # tensors inside it have been D2H'd by _move_generation_result_to_cpu.
        da_ipc = DeferredAllocIPC(
            mode=da_obj["mode"],
            req_pool_indices=tensor_to_ipc(da_obj["req_pool_indices"]),
            out_cache_loc=tensor_to_ipc(da_obj["out_cache_loc"]),
            seq_lens_minus1=tensor_to_ipc(da_obj.get("seq_lens_minus1")),
            prefix_lens=tensor_to_ipc(da_obj.get("prefix_lens")),
            extend_lens=tensor_to_ipc(da_obj.get("extend_lens")),
            free_pages_remaining=da_obj.get("free_pages_remaining", 0),
        )

    return DecodeForwardReplySlim(
        next_token_ids=tensor_to_ipc(result.next_token_ids),
        deferred_alloc=da_ipc,
        accept_lens=tensor_to_ipc(result.accept_lens),
        can_run_cuda_graph=bool(result.can_run_cuda_graph),
        num_accepted_drafts=int(result.num_accepted_drafts or 0),
        num_accepted_drafts_per_req_cpu=getattr(
            result, "num_accepted_drafts_per_req_cpu", None
        ),
        # Rare paths: keep as opaque blobs so the Rust scheduler can pass
        # them through unchanged. Cheap when None (the common case).
        logits_output_pickle=(
            _pickle.dumps(result.logits_output)
            if (result.logits_output is not None and getattr(batch, "return_logprob", False))
            else None
        ),
        routed_experts_output_pickle=(
            _pickle.dumps(result.routed_experts_output)
            if result.routed_experts_output is not None
            else None
        ),
        expert_distribution_metrics_pickle=(
            _pickle.dumps(result.expert_distribution_metrics)
            if result.expert_distribution_metrics is not None
            else None
        ),
        next_draft_input_pickle=(
            _pickle.dumps(result.next_draft_input)
            if getattr(result, "next_draft_input", None) is not None
            else None
        ),
    )


def _build_slim_reply(result, batch) -> dict:
    """Pack only the fields the scheduler consumes downstream.

    The scheduler's process_batch_result_decode / _prefill / overlap-handlers
    read: next_token_ids, deferred_alloc, accept_lens, can_run_cuda_graph,
    num_accepted_drafts, num_accepted_drafts_per_req_cpu, next_draft_input
    (spec), logits_output (return_logprob), routed_experts_output (MoE),
    expert_distribution_metrics. The remaining ~10 Optional dataclass fields
    on GenerationBatchResult are dead weight for the wire.
    """
    slim = {
        "_slim_decode": True,
        "next_token_ids": result.next_token_ids,
        "deferred_alloc": result.deferred_alloc,
        "accept_lens": result.accept_lens,
        "can_run_cuda_graph": result.can_run_cuda_graph,
        "num_accepted_drafts": result.num_accepted_drafts,
    }
    if getattr(result, "num_accepted_drafts_per_req_cpu", None) is not None:
        slim["num_accepted_drafts_per_req_cpu"] = result.num_accepted_drafts_per_req_cpu
    if (
        result.logits_output is not None
        and getattr(batch, "return_logprob", False)
    ):
        slim["logits_output"] = result.logits_output
    if result.routed_experts_output is not None:
        slim["routed_experts_output"] = result.routed_experts_output
    if result.expert_distribution_metrics is not None:
        slim["expert_distribution_metrics"] = result.expert_distribution_metrics
    if getattr(result, "next_draft_input", None) is not None:
        slim["next_draft_input"] = result.next_draft_input
    return slim


# ---------------------------------------------------------------------------
# Helper: migrate a CPU-scheduler ModelWorkerBatch to the GPU worker's device
# ---------------------------------------------------------------------------

def _move_model_worker_batch_to_device(
    batch: "ModelWorkerBatch",
    device: str,
) -> None:
    """Move all tensor fields of *batch* to *device* in-place.

    Called by TpWorkerServer when the scheduler is CPU-only and sends batches
    whose tensors are on CPU.  Applies two things:
    1. Sync req_to_token: copies the CPU snapshot rows into the worker's GPU
       req_to_token tensor so attention kernels see up-to-date KV locations.
    2. Device placement: moves all standard tensor fields to *device*.
    """
    if batch is None:
        return

    # Nothing to do if already on the right device.
    if batch.req_pool_indices is not None and hasattr(batch.req_pool_indices, "device"):
        if str(batch.req_pool_indices.device) == device:
            return

    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        return x

    # Core index tensors
    batch.input_ids = _t(batch.input_ids)
    batch.req_pool_indices = _t(batch.req_pool_indices)
    batch.seq_lens = _t(batch.seq_lens)
    batch.out_cache_loc = _t(batch.out_cache_loc)
    batch.orig_seq_lens = _t(batch.orig_seq_lens)

    # Encoder tensors
    batch.encoder_lens = _t(batch.encoder_lens)
    batch.encoder_out_cache_loc = _t(batch.encoder_out_cache_loc)

    # Mamba state tensors
    batch.mamba_track_indices = _t(batch.mamba_track_indices)
    batch.mamba_track_mask = _t(batch.mamba_track_mask)
    batch.mamba_track_seqlens = _t(batch.mamba_track_seqlens)

    # Misc optional tensors
    batch.extend_input_logprob_token_ids = _t(batch.extend_input_logprob_token_ids)
    batch.input_embeds = _t(batch.input_embeds)
    batch.replace_embeds = _t(batch.replace_embeds)
    batch.ne_token_table = _t(batch.ne_token_table)
    batch.token_type_ids = _t(batch.token_type_ids)

    # Sampling info: move all sampling tensors to device
    if batch.sampling_info is not None:
        batch.sampling_info = batch.sampling_info.to(device)

    # indices_to_free is processed in _prepare_batch before this function runs;
    # it doesn't need device movement (already consumed by the time we get here).


# ---------------------------------------------------------------------------
# Helper: move GPU tensors in GenerationBatchResult to CPU before pickling
# ---------------------------------------------------------------------------

def _move_generation_result_to_cpu(result: "GenerationBatchResult") -> None:
    """In-place copy of all GPU tensors in *result* to CPU.

    Called by the server before pickling the result so the client can
    deserialise it without needing GPU access.

    Uses non_blocking=True for all D2H copies and issues a single
    cudaStreamSynchronize at the end, avoiding one sync per tensor.
    """
    if result is None:
        return

    _needs_sync = False

    def _cpu_nb(t):
        nonlocal _needs_sync
        if t is None or not torch.is_tensor(t) or not t.is_cuda:
            return t
        _needs_sync = True
        return t.to("cpu", non_blocking=True)

    # deferred_alloc may contain GPU tensors (req_pool_indices, out_cache_loc).
    # Always move to CPU so the scheduler process can read them without
    # initializing a CUDA context (and so .item() calls in update_cache_from_scheduler
    # don't trigger per-request GPU syncs).
    if result.deferred_alloc is not None:
        result.deferred_alloc = {
            k: _cpu_nb(v) for k, v in result.deferred_alloc.items()
        }

    result.next_token_ids = _cpu_nb(result.next_token_ids)
    result.accept_lens = _cpu_nb(result.accept_lens)

    lo = result.logits_output
    if lo is not None:
        lo.next_token_logprobs = _cpu_nb(getattr(lo, "next_token_logprobs", None))
        lo.input_token_logprobs = _cpu_nb(getattr(lo, "input_token_logprobs", None))
        lo.hidden_states = _cpu_nb(getattr(lo, "hidden_states", None))

        val = getattr(lo, "next_token_top_logprobs_val", None)
        if val is not None:
            lo.next_token_top_logprobs_val = [
                _cpu_nb(v) if torch.is_tensor(v) else v for v in val
            ]
        idx = getattr(lo, "next_token_top_logprobs_idx", None)
        if idx is not None:
            lo.next_token_top_logprobs_idx = [
                _cpu_nb(x) if torch.is_tensor(x) else x for x in idx
            ]
        val2 = getattr(lo, "next_token_token_ids_logprobs_val", None)
        if val2 is not None:
            lo.next_token_token_ids_logprobs_val = [
                _cpu_nb(v) if torch.is_tensor(v) else v for v in val2
            ]

    # Single synchronize flushes all pending non-blocking copies above.
    if _needs_sync:
        torch.cuda.current_stream().synchronize()

    # These methods manage their own synchronization internally.
    if result.routed_experts_output is not None:
        result.routed_experts_output.copy_to_cpu()

    if result.expert_distribution_metrics is not None:
        result.expert_distribution_metrics.copy_to_cpu()


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

def run_gpu_worker_process(
    server_args: "ServerArgs",
    port_args: "PortArgs",
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
) -> None:
    """Entry point for the dedicated GPU-worker subprocess.

    Launched by ``launch_rust_server`` (one subprocess per (tp_rank, pp_rank)).
    Creates the TpModelWorker, binds the ZMQ PAIR socket, and blocks in the
    server event loop until the process is killed.

    The process title is set to ``sglang::gpu_worker`` so it is easy to
    identify in ``ps``/``top`` output.
    """
    import setproctitle

    from sglang.srt.managers.scheduler import configure_scheduler_process
    from sglang.srt.managers.tp_worker import TpModelWorker

    # ---- process housekeeping ----
    configure_scheduler_process(
        server_args, gpu_id, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
        pp_rank, dp_rank,
    )
    # Override process title set by configure_scheduler_process.
    prefix = f"_TP{tp_rank}_PP{pp_rank}" if (
        server_args.tp_size > 1 or server_args.pp_size > 1
    ) else ""
    setproctitle.setproctitle(f"sglang::gpu_worker{prefix}")

    # ---- build the GPU worker ----
    worker = TpModelWorker(
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        moe_ep_rank=moe_ep_rank,
        pp_rank=pp_rank,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        dp_rank=dp_rank,
        nccl_port=port_args.nccl_port,
    )

    # ---- bind rank-specific ZMQ PAIR socket and run forever ----
    ipc_name = tp_worker_ipc_for_rank(port_args.tp_worker_ipc_name, tp_rank, pp_rank)
    server = TpWorkerServer(worker, ipc_name)
    logger.info("GPU worker (tp=%d pp=%d) serving on %s", tp_rank, pp_rank, ipc_name)
    server.run_loop()  # blocks until process is terminated
