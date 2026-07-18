from __future__ import annotations

import os
import random
import logging
import threading
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    overload,
)

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import get_dsa_index_topk
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.environ import envs
from sglang.srt.utils import is_hip, is_npu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.base.conn import KVArgs, StateType
    from sglang.srt.disaggregation.common.conn import (
        CommonKVBootstrapServer,
        CommonKVManager,
        CommonKVReceiver,
        CommonKVSender,
    )
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"
_IS_HIP = is_hip()


def get_dsa_seed_metadata_dim(hf_config) -> int:
    """Return the model-defined PD seed width, independent of local spec mode."""
    if not getattr(hf_config, "index_share_for_mtp_iteration", False):
        return 0
    return get_dsa_index_topk(hf_config)


def is_dsv4_c128_online_enabled() -> bool:
    """Return whether DSV4 C128 uses request-scoped online state."""
    return not _IS_HIP and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


def get_dsv4_c128_state_indices(
    req_pool_idx: int,
    seq_len: int,
    *,
    online: bool,
    ring_size: int,
) -> np.ndarray:
    """Return the PD transfer row/page indices for DSV4 C128 state."""
    if seq_len == 0 or seq_len % 128 == 0:
        return np.empty((0,), dtype=np.int32)
    if online:
        return np.array([int(req_pool_idx)], dtype=np.int32)

    assert ring_size % 128 == 0, f"C128 ring_size must be 128-aligned, got {ring_size}"
    pages_per_req = ring_size // 128
    page = int(req_pool_idx) * pages_per_req + ((seq_len - 1) % ring_size) // 128
    return np.array([page], dtype=np.int32)


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"


#########################
# Synchronization
#########################


def _get_failure_prob() -> float:
    try:
        return float(envs.SGLANG_TEST_DISAGG_FAILURE_PROB.get())
    except Exception:
        # fallback to legacy env var
        return float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", "0"))


def _poll_with_failure_injection(pollers) -> List[int]:
    if (failure_prob := _get_failure_prob()) > 0:
        return [
            int(KVPoll.Failed) if random.random() < failure_prob else int(poller.poll())
            for poller in pollers
        ]
    return [int(poller.poll()) for poller in pollers]


def _is_fake_transfer(req: Req, server_args: ServerArgs) -> bool:
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and server_args.disaggregation_transfer_backend == "fake"
    )


def _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args) -> None:
    """Downgrade Success → Transferring for requests whose metadata hasn't landed.

    Mutates `polls` in-place. Called before all-reduce so that MIN across TP
    ranks naturally prevents any rank from committing before all ranks are ready.
    """
    for i, poll_val in enumerate(polls):
        if poll_val == int(KVPoll.Success):
            decode_req = decode_reqs[i]
            if _is_fake_transfer(decode_req.req, server_args):
                continue
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)


def poll_and_all_reduce(
    pollers,
    gloo_group: dist.ProcessGroup,
    decode_reqs=None,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    # at a certain prob, the poll is failed to simulate failure
    polls = _poll_with_failure_injection(pollers)

    # Apply metadata gate on the decode requests to downgrade Success → Transferring for requests whose metadata hasn't landed.
    if (
        decode_reqs is not None
        and metadata_buffers is not None
        and server_args is not None
    ):
        _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args)
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    # First sync across attn-tp ranks so all TP participants for a given (dp, cp)
    # shard observe the same status transitions.
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    # Then sync across attn-cp ranks, so all TPxCP participants in one DP shard
    # converge to the same global status.
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()


def poll_and_all_reduce_with_staging(
    decode_reqs,
    staging_handler,
    gloo_group: dist.ProcessGroup,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    """Staging-aware polling: advance scatter, demote incomplete transfers, all_reduce."""
    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)

    # allow test injection of failure probability at runtime
    receivers = [dr.kv_receiver for dr in decode_reqs]
    raw_polls = _poll_with_failure_injection(receivers)
    for i, decode_req in enumerate(decode_reqs):
        if raw_polls[i] == int(KVPoll.Success):
            if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
                decode_req
            ):
                raw_polls[i] = int(KVPoll.Transferring)
    # Apply metadata gate on the decode requests to downgrade Success → Transferring for requests whose metadata hasn't landed.
    if metadata_buffers is not None and server_args is not None:
        _apply_metadata_gate(raw_polls, decode_reqs, metadata_buffers, server_args)
    poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=gloo_group)
    return poll_tensor.tolist()


#########################
# Metadata Buffers
#########################


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self) -> Optional[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)


class DSparkHiddenRowPool:
    """Compact row pool for DSpark PD hidden-state transfer."""

    def __init__(
        self,
        size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: str = "cpu",
    ):
        self.size = max(0, int(size))
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.device = device
        self.buffer = torch.zeros(
            (self.size, self.hidden_size), dtype=dtype, device=device
        )
        self._free_intervals = [(0, self.size - 1)] if self.size else []
        self._free_count = self.size
        self.lock = threading.Lock()

    def available_size(self) -> int:
        with self.lock:
            return self._free_count

    def alloc(self, n: int) -> Optional[List[int]]:
        n = int(n)
        if n <= 0:
            return []
        with self.lock:
            if n > self._free_count:
                return None

            for interval_idx, (start, end) in enumerate(self._free_intervals):
                if end - start + 1 < n:
                    continue
                allocated_end = start + n - 1
                if allocated_end == end:
                    self._free_intervals.pop(interval_idx)
                else:
                    self._free_intervals[interval_idx] = (allocated_end + 1, end)
                self._free_count -= n
                return list(range(start, allocated_end + 1))

            # Preserve the previous fallback behavior when fragmentation leaves
            # no contiguous run: consume the lowest free rows across intervals.
            remaining = n
            indices = []
            updated_intervals = []
            for start, end in self._free_intervals:
                if remaining == 0:
                    updated_intervals.append((start, end))
                    continue
                take = min(remaining, end - start + 1)
                indices.extend(range(start, start + take))
                remaining -= take
                if start + take <= end:
                    updated_intervals.append((start + take, end))
            self._free_intervals = updated_intervals
            self._free_count -= n
            return indices

    def free(self, indices: Optional[List[int]]) -> None:
        if not indices:
            return
        with self.lock:
            candidates = sorted(
                int(idx) for idx in indices if 0 <= int(idx) < self.size
            )
            if not candidates:
                return

            interval_idx = 0
            last_candidate = None
            to_free = []
            for idx in candidates:
                if idx == last_candidate:
                    continue
                last_candidate = idx
                while (
                    interval_idx < len(self._free_intervals)
                    and self._free_intervals[interval_idx][1] < idx
                ):
                    interval_idx += 1
                if (
                    interval_idx < len(self._free_intervals)
                    and self._free_intervals[interval_idx][0] <= idx
                ):
                    continue
                to_free.append(idx)
            if not to_free:
                return

            freed_intervals = []
            start = end = to_free[0]
            for idx in to_free[1:]:
                if idx == end + 1:
                    end = idx
                else:
                    freed_intervals.append((start, end))
                    start = end = idx
            freed_intervals.append((start, end))

            merged = []
            existing_idx = freed_idx = 0
            while (
                existing_idx < len(self._free_intervals)
                or freed_idx < len(freed_intervals)
            ):
                if (
                    freed_idx == len(freed_intervals)
                    or (
                        existing_idx < len(self._free_intervals)
                        and self._free_intervals[existing_idx][0]
                        < freed_intervals[freed_idx][0]
                    )
                ):
                    interval = self._free_intervals[existing_idx]
                    existing_idx += 1
                else:
                    interval = freed_intervals[freed_idx]
                    freed_idx += 1
                if merged and interval[0] <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
                else:
                    merged.append(interval)
            self._free_intervals = merged
            self._free_count += len(to_free)

    def write(self, indices: List[int], hidden: torch.Tensor) -> None:
        if not indices:
            return
        if hidden.shape[0] != len(indices):
            raise ValueError(
                "DSpark hidden row count mismatch: "
                f"hidden={hidden.shape[0]}, indices={len(indices)}"
            )
        if hidden.shape[-1] > self.hidden_size:
            raise ValueError(
                "DSpark hidden width exceeds row pool width: "
                f"hidden={hidden.shape[-1]}, pool={self.hidden_size}"
            )
        hidden = hidden.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_width = hidden.shape[-1]
        first = int(indices[0])
        contiguous = all(int(idx) == first + i for i, idx in enumerate(indices))
        if contiguous:
            dst = self.buffer[first : first + len(indices)]
            if hidden_width < self.hidden_size:
                dst.zero_()
            dst[:, :hidden_width].copy_(hidden)
            return

        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if hidden_width < self.hidden_size:
            self.buffer[index_tensor, :] = 0
        self.buffer[index_tensor, :hidden_width] = hidden

    def read(self, indices: List[int]) -> torch.Tensor:
        if not indices:
            return torch.empty(
                (0, self.hidden_size), dtype=self.dtype, device=self.device
            )
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self.buffer[index_tensor].clone()

    def read_view(self, indices: List[int]) -> torch.Tensor:
        if not indices:
            return torch.empty(
                (0, self.hidden_size), dtype=self.dtype, device=self.device
            )
        first = int(indices[0])
        contiguous = all(int(idx) == first + i for i, idx in enumerate(indices))
        if contiguous:
            return self.buffer[first : first + len(indices)]
        return self.read(indices)

    def get_state_buf_infos(self):
        if self.size <= 0:
            return [], [], []
        return [self.buffer.data_ptr()], [self.buffer.nbytes], [self.buffer[0].nbytes]


@dataclass
class DSparkHiddenTransferPlan:
    row_count: int
    item_len: int
    row_chunks: List[Dict[str, Any]]

    @classmethod
    def build(cls, row_count: int, item_len: int) -> "DSparkHiddenTransferPlan":
        row_count = int(row_count)
        item_len = int(item_len)
        if row_count <= 0:
            return cls(row_count=0, item_len=item_len, row_chunks=[])
        return cls(
            row_count=row_count,
            item_len=item_len,
            row_chunks=[{"row_start": 0, "row_len": row_count}],
        )

    def to_dynamic_dst(self, ptr: int = 0) -> Dict[str, Any]:
        return {
            "ptr": int(ptr),
            "nbytes": int(self.row_count * self.item_len),
            "item_len": int(self.item_len),
            "row_count": int(self.row_count),
            "row_chunks": [dict(chunk) for chunk in self.row_chunks],
        }

    @staticmethod
    def trim_dynamic_dst(
        dynamic_dst: Dict[str, Any],
        *,
        offset: int,
        new_row_count: int,
        old_row_count: int,
    ) -> Dict[str, Any]:
        new_dynamic_dst = dict(dynamic_dst)
        item_len = int(new_dynamic_dst.get("item_len", 0))
        offset = int(offset)
        new_row_count = int(new_row_count)
        old_row_count = int(old_row_count)
        old_chunks = [dict(chunk) for chunk in new_dynamic_dst.get("row_chunks") or []]

        new_dynamic_dst["row_count"] = new_row_count
        new_dynamic_dst["nbytes"] = int(new_row_count * item_len)

        if old_chunks and "ptr" in old_chunks[0]:
            new_chunks = []
            for old_chunk in old_chunks:
                chunk_start = int(old_chunk.get("row_start", 0))
                chunk_len = int(old_chunk.get("row_len", 0))
                chunk_end = chunk_start + chunk_len
                overlap_start = max(chunk_start, offset)
                overlap_end = min(chunk_end, old_row_count)
                if overlap_end <= overlap_start:
                    continue
                new_chunks.append(
                    {
                        "row_start": int(overlap_start - offset),
                        "row_len": int(overlap_end - overlap_start),
                        "ptr": int(old_chunk["ptr"])
                        + int(overlap_start - chunk_start) * item_len,
                        "nbytes": int((overlap_end - overlap_start) * item_len),
                    }
                )
            new_dynamic_dst["row_chunks"] = new_chunks
            new_dynamic_dst["ptr"] = int(new_chunks[0]["ptr"]) if new_chunks else 0
            return new_dynamic_dst

        if item_len > 0:
            new_dynamic_dst["ptr"] = int(new_dynamic_dst.get("ptr", 0)) + offset * item_len
        plan = DSparkHiddenTransferPlan.build(new_row_count, item_len)
        new_dynamic_dst["row_chunks"] = plan.row_chunks
        return new_dynamic_dst


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        max_sampling_mask_tokens: Optional[int] = None,
        custom_mem_pool: torch.cuda.MemPool = None,
        output_dsa_topk_indices_dim: int = 0,
        dspark_prefill_tail_len: int = 0,
        dspark_hidden_pool_size: int = 0,
        dspark_hidden_size: int = 0,
        dspark_hidden_device: str = "cpu",
    ):
        self.custom_mem_pool = custom_mem_pool
        self.output_dsa_topk_indices_dim = output_dsa_topk_indices_dim
        self.dspark_prefill_tail_len = max(0, int(dspark_prefill_tail_len))
        self.dspark_hidden_pool: Optional[DSparkHiddenRowPool] = None
        if dspark_hidden_pool_size > 0 and dspark_hidden_size > 0:
            self.dspark_hidden_pool = DSparkHiddenRowPool(
                dspark_hidden_pool_size,
                dspark_hidden_size,
                hidden_states_dtype,
                device=dspark_hidden_device,
            )
        if max_sampling_mask_tokens is None:
            max_sampling_mask_tokens = (
                envs.SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS.get()
            )
        self.enable_sampling_mask = max_sampling_mask_tokens > 0
        bootstrap_room_dtype = torch.uint64
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
            # TODO: Fix me when npu backend supports torch.uint64
            bootstrap_room_dtype = torch.int64
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        elif envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "INTRA_NODE_NVLINK":
            device = "cuda"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            self.output_token_sampling_mask_len = None
            self.output_token_sampling_mask_idx = None
            self.output_token_sampling_logprobs = None
            if self.enable_sampling_mask:
                self.output_token_sampling_mask_len = torch.zeros(
                    (size, 16), dtype=torch.int32, device=device
                )
                self.output_token_sampling_mask_idx = torch.zeros(
                    (size, max_sampling_mask_tokens), dtype=torch.int32, device=device
                )
                self.output_token_sampling_logprobs = torch.zeros(
                    (size, 16), dtype=torch.float32, device=device
                )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )
            if self.output_dsa_topk_indices_dim > 0:
                self.output_dsa_topk_indices = torch.full(
                    (size, self.output_dsa_topk_indices_dim),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                self.output_dsa_topk_indices = None
            self.output_dspark_prefill_tail_hidden_states = None
            self.output_dspark_prefill_tail_valid_mask = None
            if self.dspark_prefill_tail_len > 0:
                self.output_dspark_prefill_tail_hidden_states = torch.zeros(
                    (size, self.dspark_prefill_tail_len, hidden_size),
                    dtype=hidden_states_dtype,
                    device=device,
                )
                self.output_dspark_prefill_tail_valid_mask = torch.zeros(
                    (size, self.dspark_prefill_tail_len),
                    dtype=torch.bool,
                    device=device,
                )
            # Request validation: store bootstrap_room to detect metadata corruption
            self.bootstrap_room = torch.zeros(
                (size, 8), dtype=bootstrap_room_dtype, device=device
            )

    def get_buf_infos(self):
        bufs = [
            self.output_ids,
            self.cached_tokens,
            self.output_token_logprobs_val,
            self.output_token_logprobs_idx,
            self.output_top_logprobs_val,
            self.output_top_logprobs_idx,
        ]
        if self.enable_sampling_mask:
            bufs.extend(
                [
                    self.output_token_sampling_mask_len,
                    self.output_token_sampling_mask_idx,
                    self.output_token_sampling_logprobs,
                ]
            )
        bufs.extend(
            [
                self.output_topk_p,
                self.output_topk_index,
                self.output_hidden_states,
            ]
        )
        if self.output_dsa_topk_indices is not None:
            bufs.append(self.output_dsa_topk_indices)
        bufs.append(self.bootstrap_room)
        if self.output_dspark_prefill_tail_hidden_states is not None:
            bufs.extend(
                [
                    self.output_dspark_prefill_tail_hidden_states,
                    self.output_dspark_prefill_tail_valid_mask,
                ]
            )
        ptrs = [buf.data_ptr() for buf in bufs]
        data_lens = [buf.nbytes for buf in bufs]
        item_lens = [buf[0].nbytes for buf in bufs]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        sampling_mask_len = None
        sampling_mask_idx = None
        sampling_logprobs = None
        if self.enable_sampling_mask:
            sampling_mask_len = self.output_token_sampling_mask_len[idx].clone()
            sampling_mask_idx = self.output_token_sampling_mask_idx[idx].clone()
            sampling_logprobs = self.output_token_sampling_logprobs[idx].clone()
        ret = (
            self.output_ids[idx].clone(),
            self.cached_tokens[idx].clone(),
            self.output_token_logprobs_val[idx].clone(),
            self.output_token_logprobs_idx[idx].clone(),
            self.output_top_logprobs_val[idx].clone(),
            self.output_top_logprobs_idx[idx].clone(),
            sampling_mask_len,
            sampling_mask_idx,
            sampling_logprobs,
            self.output_topk_p[idx].clone(),
            self.output_topk_index[idx].clone(),
            self.output_hidden_states[idx].clone(),
            (
                self.output_dsa_topk_indices[idx].clone()
                if self.output_dsa_topk_indices is not None
                else None
            ),
            self.bootstrap_room[idx].clone(),
        )
        if self.output_dspark_prefill_tail_hidden_states is not None:
            ret += (
                self.output_dspark_prefill_tail_hidden_states[idx].clone(),
                self.output_dspark_prefill_tail_valid_mask[idx].clone(),
            )
        return ret

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        # The cached_tokens buffer is (size, 16); slots 0-3 hold cached token
        # counts and slots 4-6 are reused for multimodal prompt token counts
        # (slots 7-15 remain spare). This avoids adding new RDMA buffers.
        # Slot map: 0=cached 1=device 2=host 3=storage 4=image 5=audio 6=video.
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        self.cached_tokens[req.metadata_buffer_index][1] = req.cached_tokens_device
        self.cached_tokens[req.metadata_buffer_index][2] = req.cached_tokens_host
        self.cached_tokens[req.metadata_buffer_index][3] = req.cached_tokens_storage

        # Compute multimodal prompt token counts on the prefill node so decode
        # can report them in usage.
        if req.multimodal_inputs:
            image_t, audio_t, video_t = req.multimodal_inputs.compute_mm_token_counts()
        else:
            image_t = audio_t = video_t = 0
        self.cached_tokens[req.metadata_buffer_index][4] = image_t
        self.cached_tokens[req.metadata_buffer_index][5] = audio_t
        self.cached_tokens[req.metadata_buffer_index][6] = video_t
        if req.return_logprob:
            if req.logprob.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.logprob.output_token_logprobs_val[0]
                )
            if req.logprob.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.logprob.output_token_logprobs_idx[0]
                )

            if req.logprob.output_top_logprobs_val:  # not none or empty list
                top_logprobs_len = len(req.logprob.output_top_logprobs_val[0])
                max_top_logprobs_len = self.output_top_logprobs_val.shape[1]
                if top_logprobs_len > max_top_logprobs_len:
                    raise RuntimeError(
                        f"top_logprobs_num {top_logprobs_len} exceeds "
                        f"disaggregation metadata capacity {max_top_logprobs_len}. "
                        "Lower top_logprobs_num or increase the metadata buffer."
                    )
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.logprob.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.logprob.output_top_logprobs_val[0],
                    dtype=torch.float32,
                    device="cpu",
                )
            if req.logprob.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.logprob.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.logprob.output_top_logprobs_idx[0],
                    dtype=torch.int32,
                    device="cpu",
                )
        if req.return_sampling_mask:
            if not self.enable_sampling_mask:
                raise RuntimeError(
                    "return_sampling_mask with disaggregation requires "
                    "SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS > 0."
                )
            # Sentinel -1: the decode side records None for this handoff token.
            self.output_token_sampling_mask_len[req.metadata_buffer_index][0] = -1
            sampling_masks = req.output_token_sampling_mask
            sampling_logprobs = req.output_token_sampling_logprobs
            if sampling_masks:
                sampling_mask = sampling_masks[0]
                sampling_logprob = sampling_logprobs[0] if sampling_logprobs else None
                if sampling_mask is not None and sampling_logprob is not None:
                    mask_len = len(sampling_mask)
                    max_mask_len = self.output_token_sampling_mask_idx.shape[1]
                    if mask_len > max_mask_len:
                        raise RuntimeError(
                            f"Sampling mask length {mask_len} exceeds disaggregation "
                            f"metadata capacity {max_mask_len}. Increase "
                            "SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS."
                        )
                    self.output_token_sampling_mask_len[req.metadata_buffer_index][
                        0
                    ] = mask_len
                    if mask_len:
                        self.output_token_sampling_mask_idx[
                            req.metadata_buffer_index, :mask_len
                        ].copy_(
                            torch.tensor(
                                sampling_mask,
                                dtype=torch.int32,
                                device=self.output_token_sampling_mask_idx.device,
                            )
                        )
                    self.output_token_sampling_logprobs[req.metadata_buffer_index][
                        0
                    ] = float(sampling_logprob)
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )
            if self.output_dsa_topk_indices is not None:
                dsa_topk_indices = req.output_dsa_topk_indices
                if dsa_topk_indices is not None:
                    self.output_dsa_topk_indices[req.metadata_buffer_index].copy_(
                        dsa_topk_indices
                    )
                else:
                    self.output_dsa_topk_indices[req.metadata_buffer_index].fill_(-1)
        # Store bootstrap_room for validation on decode side
        self.bootstrap_room[req.metadata_buffer_index, 0] = (
            req.bootstrap_room if req.bootstrap_room is not None else 0
        )
        if self.output_dspark_prefill_tail_hidden_states is not None:
            self.output_dspark_prefill_tail_hidden_states[
                req.metadata_buffer_index
            ].zero_()
            self.output_dspark_prefill_tail_valid_mask[
                req.metadata_buffer_index
            ].zero_()
            tail_hidden = getattr(req, "prefill_tail_hidden_states_tensor", None)
            tail_mask = getattr(req, "prefill_tail_valid_mask", None)
            if tail_hidden is not None and tail_mask is not None:
                tail_len = min(
                    int(tail_hidden.shape[0]),
                    int(self.output_dspark_prefill_tail_hidden_states.shape[1]),
                )
                if tail_len > 0:
                    self.output_dspark_prefill_tail_hidden_states[
                        req.metadata_buffer_index, :tail_len
                    ].copy_(tail_hidden[:tail_len].to(self.output_hidden_states.device))
                    self.output_dspark_prefill_tail_valid_mask[
                        req.metadata_buffer_index, :tail_len
                    ].copy_(tail_mask[:tail_len].to(self.output_hidden_states.device))

    def ensure_dspark_hidden_pool(
        self,
        *,
        size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: str = "cpu",
    ) -> DSparkHiddenRowPool:
        if self.dspark_hidden_pool is None:
            self.dspark_hidden_pool = DSparkHiddenRowPool(
                size=size,
                hidden_size=hidden_size,
                dtype=dtype,
                device=device,
            )
        elif self.dspark_hidden_pool.hidden_size != int(hidden_size):
            raise ValueError(
                "DSpark hidden pool hidden_size mismatch: "
                f"existing={self.dspark_hidden_pool.hidden_size}, "
                f"requested={hidden_size}"
            )
        return self.dspark_hidden_pool

    def get_dspark_hidden_state_buf_infos(self):
        if self.dspark_hidden_pool is None:
            return [], [], []
        return self.dspark_hidden_pool.get_state_buf_infos()


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.KVARGS]
) -> Type[KVArgs]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.MANAGER]
) -> Type[CommonKVManager]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.SENDER]
) -> Type[CommonKVSender]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.RECEIVER]
) -> Type[CommonKVReceiver]: ...
@overload
def get_kv_class(
    transfer_backend: TransferBackend, class_type: Literal[KVClassType.BOOTSTRAP_SERVER]
) -> Type[CommonKVBootstrapServer]: ...


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.MORI:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mori import (
            MoriKVBootstrapServer,
            MoriKVManager,
            MoriKVReceiver,
            MoriKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MoriKVManager,
            KVClassType.SENDER: MoriKVSender,
            KVClassType.RECEIVER: (MoriKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MoriKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import (
            FakeKVManager,
            FakeKVReceiver,
            FakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: FakeKVManager,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def _get_cp_rank_page_bounds(
    total_pages: int, cp_rank: int, cp_size: int
) -> Tuple[int, int]:
    base = total_pages // cp_size
    rem = total_pages % cp_size
    local_start = cp_rank * base + min(cp_rank, rem)
    n_pages = base + (1 if cp_rank < rem else 0)
    return local_start, local_start + n_pages


def page_indices_to_cp_rank_page_indices(
    page_indices: np.ndarray,
    total_pages: int,
    cp_rank: int,
    cp_size: int,
) -> np.ndarray:
    """
    Filter page_indices (which are *global* page ids in the KV pool) to those
    belonging to the given CP rank for this request.

    For a single request, its pages occupy a contiguous global range
    [first_page, first_page + total_pages). We first compute the local
    split [0, total_pages) across cp_size ranks, then shift that local
    range by first_page back into the global page id space and take
    the intersection with page_indices.

    Returns:
        Subset of page_indices that fall in this rank's global
        [start_page, end_page) slice for the given CP rank.
    """
    if cp_size <= 1:
        return page_indices

    if page_indices.size == 0:
        return np.asarray(page_indices)

    first_page = int(page_indices.min())
    base = total_pages // cp_size
    rem = total_pages % cp_size

    if rem == 0:
        local_start = cp_rank * base
        local_end = local_start + base
    else:
        local_start = cp_rank * base + min(cp_rank, rem)
        n_pages = base + (1 if cp_rank < rem else 0)
        local_end = local_start + n_pages

    # Map back to global page ids.
    start_page = first_page + local_start
    end_page = first_page + local_end

    mask = (page_indices >= start_page) & (page_indices < end_page)
    return np.asarray(page_indices)[mask]


def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager,
    kv_indices: np.ndarray,
    index_slice: slice,
    total_pages: Optional[int] = None,
) -> Tuple[np.ndarray, slice]:
    """Filters kv_indices and index_slice for the current CP rank."""
    if total_pages is None:
        total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    if cp_size <= 1:
        return kv_indices, index_slice

    rank_start, rank_end = _get_cp_rank_page_bounds(total_pages, cp_rank, cp_size)
    chunk_start = index_slice.start if index_slice.start is not None else 0
    chunk_end = index_slice.stop if index_slice.stop is not None else total_pages
    first_pos = max(rank_start, chunk_start) - chunk_start
    last_pos = min(rank_end, chunk_end) - chunk_start

    if last_pos <= first_pos:
        new_kv_indices = kv_indices[:0]
        new_index_slice = slice(chunk_start, chunk_start)
    else:
        new_kv_indices = kv_indices[first_pos:last_pos]
        new_index_slice = slice(
            chunk_start + first_pos,
            chunk_start + last_pos,
        )
    return new_kv_indices, new_index_slice


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, (MLATokenToKVPool, DeepSeekV4TokenToKVPool))


def compute_mamba_state_slice_blocks(
    src_dim: int,
    dst_dim: int,
    src_attn_tp_size: int,
    dst_attn_tp_size: int,
    dst_tp_rank_in_group: int,
    local_tp_rank_in_group: int,
    conv_shard_groups: Optional[List[int]] = None,
) -> List[Tuple[int, int, int]]:
    """Blocks to copy one mamba state item across differing attn-TP sizes.

    Returns ``(src_dim_start, dst_dim_start, num_dims)`` triples in units of the
    sliceable (3rd) dimension. Single-axis states (temporal_state, or when
    ``conv_shard_groups`` is None) return one contiguous block -- byte-identical to
    the legacy behavior.

    GDN conv_state is ``cat([query | key | value])`` where each sub-block (full
    dims == ``conv_shard_groups``, e.g. ``[key_dim, key_dim, value_dim]``) is
    head-sharded INDEPENDENTLY across attn-TP. In the SCATTER direction
    (1 prefill rank -> several decode ranks) a single contiguous slice straddles
    the q/k/v boundaries and delivers wrong channels. The AGGREGATION direction
    (several prefill ranks -> 1 decode rank) has the symmetric problem: a single
    contiguous write interleaves the sub-blocks by writer. Both directions emit one
    block per sub-block for conv_state; temporal_state and non-GDN states (when
    ``conv_shard_groups`` is None) keep the single contiguous slice.
    """
    use_subdims = (
        conv_shard_groups is not None
        and sum(conv_shard_groups) == src_dim * src_attn_tp_size
    )

    if src_attn_tp_size > dst_attn_tp_size:
        # Aggregation: several prefill ranks each write their shard into one decode slot.
        writers_per_decode = src_attn_tp_size // dst_attn_tp_size
        local_writer_idx = local_tp_rank_in_group % writers_per_decode
        if not use_subdims:
            return [(0, local_writer_idx * src_dim, src_dim)]
        # conv_state: a plain contiguous write would interleave the sub-blocks by
        # writer ([q0,k0,v0,q1,k1,v1,...]); place this writer's shard of each
        # independently head-sharded sub-block at its grouped offset so the decode
        # buffer is [q0,q1,...,k0,k1,...,v0,v1,...].
        blocks: List[Tuple[int, int, int]] = []
        src_off = 0
        dst_off = 0
        for full_sd in conv_shard_groups:
            src_sub = full_sd // src_attn_tp_size
            dst_sub = full_sd // dst_attn_tp_size
            blocks.append((src_off, dst_off + local_writer_idx * src_sub, src_sub))
            src_off += src_sub
            dst_off += dst_sub
        return blocks

    # Scatter: 1 prefill rank feeds several decode ranks.
    if not use_subdims:
        src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
        return [(src_dim_start, 0, dst_dim)]

    # conv_state: gather the decode rank's [q | k | v] shard from the three
    # independently head-sharded sub-blocks of the src tensor. dst is contiguous.
    blocks: List[Tuple[int, int, int]] = []
    src_off = 0
    dst_off = 0
    for full_sd in conv_shard_groups:
        src_sub = full_sd // src_attn_tp_size  # this prefill rank's shard of sub-block
        dst_sub = full_sd // dst_attn_tp_size  # this decode rank's shard of sub-block
        src_start = src_off + (dst_tp_rank_in_group * dst_sub) % src_sub
        blocks.append((src_start, dst_off, dst_sub))
        src_off += src_sub
        dst_off += dst_sub
    return blocks


def append_state_component(
    kv_args: KVArgs,
    state_type: StateType,
    data_ptrs: List[int],
    data_lens: List[int],
    item_lens: List[int],
    dim_per_tensor: Optional[List[int]] = None,
    conv_shard_groups: Optional[List[Optional[List[int]]]] = None,
) -> None:
    """Append one state component. Caller orders state_types consistently
    on prefill and decode sides."""
    kv_args.state_types.append(state_type)
    kv_args.state_data_ptrs.append(data_ptrs)
    kv_args.state_data_lens.append(data_lens)
    kv_args.state_item_lens.append(item_lens)
    kv_args.state_dim_per_tensor.append(dim_per_tensor or [])
    kv_args.state_conv_shard_groups.append(conv_shard_groups or [])


def setup_state_kv_args(
    kv_args: KVArgs,
    token_to_kv_pool,
    draft_token_to_kv_pool=None,
    total_kv_layers: int = None,
    req_to_token_pool=None,
    dspark_hidden_pool: Optional[DSparkHiddenRowPool] = None,
) -> None:
    """Populate ``kv_args`` state-buffer fields from the given pool.
    Shared by prefill and decode bootstrap paths so the state_type dispatch
    lives in one place.
    """
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        HybridLinearKVPool,
        MiniMaxSparseKVPool,
    )

    kv_args.state_types = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []
    kv_args.is_hybrid_mla_backend = False
    kv_args.state_conv_shard_groups = []

    if isinstance(token_to_kv_pool, MiniMaxSparseKVPool):
        if token_to_kv_pool.index_kv_pool is not None:
            raise NotImplementedError(
                "PD disaggregation for MiniMax sparse layers with index value "
                "(index_kv_pool) is not yet supported; only K-only sparse layers are."
            )
        if token_to_kv_pool.index_k_pool is not None:
            dp, dl, il = token_to_kv_pool.get_index_k_state_buf_infos()
            append_state_component(kv_args, StateType.MINIMAX_INDEX_K, dp, dl, il)
    elif hasattr(token_to_kv_pool, "get_state_buf_infos"):
        data_ptrs, data_lens, item_lens = token_to_kv_pool.get_state_buf_infos()

        # DeepSeekV4TokenToKVPool inherits BaseSWAKVPool; its heterogeneous
        # state list is described per-entry via get_state_buf_infos.
        if isinstance(token_to_kv_pool, BaseSWAKVPool):
            append_state_component(
                kv_args, StateType.SWA, data_ptrs, data_lens, item_lens
            )
            # unified_kv: the SWA ring lives in the unified buffers (no separate
            # swa_kv_pool) and is addressed per-row, so ship it as SWA_RING.
            if getattr(token_to_kv_pool, "_unified_kv", False) and hasattr(
                token_to_kv_pool, "get_unified_swa_ring_buf_infos"
            ):
                ring_ptrs, ring_lens, ring_item_lens = (
                    token_to_kv_pool.get_unified_swa_ring_buf_infos()
                )
                if ring_ptrs:
                    append_state_component(
                        kv_args,
                        StateType.SWA_RING,
                        ring_ptrs,
                        ring_lens,
                        ring_item_lens,
                    )
            if hasattr(token_to_kv_pool, "get_c128_state_buf_infos"):
                c128_ptrs, c128_lens, c128_item_lens = (
                    token_to_kv_pool.get_c128_state_buf_infos()
                )
                if c128_ptrs:
                    append_state_component(
                        kv_args,
                        StateType.C128_STATE,
                        c128_ptrs,
                        c128_lens,
                        c128_item_lens,
                    )
        elif isinstance(token_to_kv_pool, HybridLinearKVPool):
            dim = (
                token_to_kv_pool.get_state_dim_per_tensor()
                if hasattr(token_to_kv_pool, "get_state_dim_per_tensor")
                else None
            )
            kv_args.is_hybrid_mla_backend = is_mla_backend(
                token_to_kv_pool.full_kv_pool
            )
            conv_shard_groups = (
                token_to_kv_pool.get_state_conv_shard_groups()
                if hasattr(token_to_kv_pool, "get_state_conv_shard_groups")
                else None
            )
            append_state_component(
                kv_args,
                StateType.MAMBA,
                data_ptrs,
                data_lens,
                item_lens,
                dim,
                conv_shard_groups,
            )
        elif isinstance(token_to_kv_pool, (DSATokenToKVPool, NPUMLATokenToKVPool)):
            if draft_token_to_kv_pool is not None and isinstance(
                draft_token_to_kv_pool, DSATokenToKVPool
            ):
                (
                    draft_data_ptrs,
                    draft_data_lens,
                    draft_item_lens,
                ) = draft_token_to_kv_pool.get_state_buf_infos()
                data_ptrs = data_ptrs + draft_data_ptrs
                data_lens = data_lens + draft_data_lens
                item_lens = item_lens + draft_item_lens
            if isinstance(token_to_kv_pool, NPUMLATokenToKVPool):
                kv_args.kv_buf_groups = (
                    len(kv_args.kv_data_ptrs) // token_to_kv_pool.layer_num
                )
                kv_args.total_kv_layers = total_kv_layers
            else:
                append_state_component(
                    kv_args, StateType.DSA, data_ptrs, data_lens, item_lens
                )

    if dspark_hidden_pool is not None:
        data_ptrs, data_lens, item_lens = dspark_hidden_pool.get_state_buf_infos()
        if data_ptrs:
            append_state_component(
                kv_args,
                StateType.DSPARK_HIDDEN,
                data_ptrs,
                data_lens,
                item_lens,
            )

    # DSV4 NextN shares the target allocator, so target and draft use the same
    # local SWA indices. Keep draft buffers in a separate positional component
    # to avoid mixing them into the target's heterogeneous state layout, while
    # reusing the existing SWA transport dispatch. NPU has a different paged
    # state layout and is intentionally left unchanged.
    if (
        not is_npu()
        and isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        and isinstance(draft_token_to_kv_pool, DeepSeekV4TokenToKVPool)
    ):
        if not draft_token_to_kv_pool.compression_ratios or not all(
            ratio == 0 for ratio in draft_token_to_kv_pool.compression_ratios
        ):
            raise RuntimeError(
                "DSV4 draft state transfer expects SWA-only NextN layers"
            )
        if token_to_kv_pool._unified_kv != draft_token_to_kv_pool._unified_kv:
            raise RuntimeError(
                "DSV4 target and draft pools must use the same unified-KV mode"
            )

        if token_to_kv_pool._unified_kv:
            target_geometry = (
                token_to_kv_pool.unified_swa_window,
                token_to_kv_pool.unified_swa_ring_size,
                token_to_kv_pool.unified_swa_pages,
            )
            draft_geometry = (
                draft_token_to_kv_pool.unified_swa_window,
                draft_token_to_kv_pool.unified_swa_ring_size,
                draft_token_to_kv_pool.unified_swa_pages,
            )
            if target_geometry != draft_geometry:
                raise RuntimeError(
                    "DSV4 target and draft pools must share SWA ring geometry: "
                    f"target={target_geometry}, draft={draft_geometry}"
                )
            draft_ptrs, draft_lens, draft_item_lens = (
                draft_token_to_kv_pool.get_unified_swa_ring_buf_infos()
            )
            draft_state_type = StateType.SWA_RING
        else:
            if (
                token_to_kv_pool.full_to_swa_index_mapping
                is not draft_token_to_kv_pool.full_to_swa_index_mapping
            ):
                raise RuntimeError(
                    "DSV4 target and draft pools must share the SWA index mapping"
                )
            target_geometry = (
                token_to_kv_pool.page_size,
                token_to_kv_pool.sliding_window,
            )
            draft_geometry = (
                draft_token_to_kv_pool.page_size,
                draft_token_to_kv_pool.sliding_window,
            )
            if target_geometry != draft_geometry:
                raise RuntimeError(
                    "DSV4 target and draft pools must share paged SWA geometry: "
                    f"target={target_geometry}, draft={draft_geometry}"
                )
            draft_ptrs, draft_lens, draft_item_lens = (
                draft_token_to_kv_pool.get_state_buf_infos()
            )
            draft_state_type = StateType.SWA

        if draft_ptrs:
            append_state_component(
                kv_args,
                draft_state_type,
                draft_ptrs,
                draft_lens,
                draft_item_lens,
            )

    if (
        StateType.MAMBA not in kv_args.state_types
        and req_to_token_pool is not None
        and hasattr(req_to_token_pool, "get_state_buf_infos")
    ):
        data_ptrs, data_lens, item_lens = req_to_token_pool.get_state_buf_infos()
        if data_ptrs:
            dim = (
                req_to_token_pool.get_state_dim_per_tensor()
                if hasattr(req_to_token_pool, "get_state_dim_per_tensor")
                else None
            )
            conv_shard_groups = (
                req_to_token_pool.get_state_conv_shard_groups()
                if hasattr(req_to_token_pool, "get_state_conv_shard_groups")
                else None
            )
            append_state_component(
                kv_args,
                StateType.MAMBA,
                data_ptrs,
                data_lens,
                item_lens,
                dim,
                conv_shard_groups,
            )


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.logprob.input_token_logprobs_val = []
        req.logprob.input_token_logprobs_idx = []
        req.logprob.input_top_logprobs_val = []
        req.logprob.input_top_logprobs_idx = []
        req.logprob.input_token_ids_logprobs_val = []
        req.logprob.input_token_ids_logprobs_idx = []


def is_aborted(req: Req) -> bool:
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    return isinstance(req.to_finish, FINISH_ABORT) or isinstance(
        req.finished_reason, FINISH_ABORT
    )
