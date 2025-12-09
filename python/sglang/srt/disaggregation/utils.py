from __future__ import annotations

import dataclasses
import logging
import os
import random
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Type

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler, Scheduler

#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


def poll_and_all_reduce(pollers, gloo_group):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


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


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
    ):
        self.custom_mem_pool = custom_mem_pool
        device = "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
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

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.cached_tokens[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.output_topk_p[idx],
            self.output_topk_index[idx],
            self.output_hidden_states[idx],
        )

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )
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


@dataclasses.dataclass
class ReqMetadataView:
    """
    Lightweight read-only snapshot of Req for metadata buffer operations.
    Used by transfer worker to avoid race conditions with main thread.
    This view contains only the fields needed by set_buf().
    """
    metadata_buffer_index: int
    output_ids: List[int]
    cached_tokens: int
    return_logprob: bool
    output_token_logprobs_val: Optional[List[float]] = None
    output_token_logprobs_idx: Optional[List[int]] = None
    output_top_logprobs_val: Optional[List[List[float]]] = None
    output_top_logprobs_idx: Optional[List[List[int]]] = None
    hidden_states_tensor: Optional[torch.Tensor] = None
    output_topk_p: Optional[torch.Tensor] = None
    output_topk_index: Optional[torch.Tensor] = None
    
    @classmethod
    def from_req(cls, req: "Req") -> "ReqMetadataView":
        """
        Create a snapshot view from a Req object.
        This creates copies of mutable data to avoid race conditions.
        """
        return cls(
            metadata_buffer_index=req.metadata_buffer_index,
            output_ids=req.output_ids[:] if req.output_ids else [],  # 浅拷贝列表
            cached_tokens=req.cached_tokens,
            return_logprob=req.return_logprob,
            output_token_logprobs_val=(
                req.output_token_logprobs_val[:] 
                if req.output_token_logprobs_val else None
            ),
            output_token_logprobs_idx=(
                req.output_token_logprobs_idx[:] 
                if req.output_token_logprobs_idx else None
            ),
            output_top_logprobs_val=(
                req.output_top_logprobs_val[:] 
                if req.output_top_logprobs_val else None
            ),
            output_top_logprobs_idx=(
                req.output_top_logprobs_idx[:] 
                if req.output_top_logprobs_idx else None
            ),
            hidden_states_tensor=(
                req.hidden_states_tensor.clone() 
                if req.hidden_states_tensor is not None else None
            ),
            output_topk_p=(
                req.output_topk_p.clone() 
                if req.output_topk_p is not None else None
            ),
            output_topk_index=(
                req.output_topk_index.clone() 
                if req.output_topk_index is not None else None
            ),
        )
    
    def to_req_like(self, mutable: bool = False):
        """
        Convert to a minimal Req-like object for set_buf compatibility.
        
        Args:
            mutable: If True, the returned object directly references the view's
                attributes, so modifications will update the view. If False,
                creates a read-only copy.
        """
        class MinimalReq:
            """Minimal Req-like object for set_buf compatibility."""
            def __init__(self, view: ReqMetadataView, mutable: bool = False):
                self.metadata_buffer_index = view.metadata_buffer_index
                self.cached_tokens = view.cached_tokens
                self.return_logprob = view.return_logprob
                self.hidden_states_tensor = view.hidden_states_tensor
                self.output_topk_p = view.output_topk_p
                self.output_topk_index = view.output_topk_index
                
                if mutable:
                    # Direct references - modifications update the view
                    self.output_ids = view.output_ids
                    self.output_token_logprobs_val = view.output_token_logprobs_val
                    self.output_token_logprobs_idx = view.output_token_logprobs_idx
                    self.output_top_logprobs_val = view.output_top_logprobs_val
                    self.output_top_logprobs_idx = view.output_top_logprobs_idx
                else:
                    # Copies - modifications don't affect the view
                    self.output_ids = view.output_ids[:] if view.output_ids else []
                    self.output_token_logprobs_val = (
                        view.output_token_logprobs_val[:] 
                        if view.output_token_logprobs_val else None
                    )
                    self.output_token_logprobs_idx = (
                        view.output_token_logprobs_idx[:] 
                        if view.output_token_logprobs_idx else None
                    )
                    self.output_top_logprobs_val = (
                        view.output_top_logprobs_val[:] 
                        if view.output_top_logprobs_val else None
                    )
                    self.output_top_logprobs_idx = (
                        view.output_top_logprobs_idx[:] 
                        if view.output_top_logprobs_idx else None
                    )
        
        return MinimalReq(self, mutable)


def process_logprobs_for_request(
    scheduler: "Scheduler",
    req: "Req",
    req_idx: int,
    logits_output,
    extend_input_len_per_req: list[int],
    extend_logprob_start_len_per_req: list[int],
    logprob_pt: int,
    next_token_ids: Optional[list[int]] = None,
    is_last_prefill_chunk: bool = False,
) -> int:
    """
    Process logprobs for a single request during prefill.
    
    This function handles both complete prefill (when prefill is finished) and
    chunked prefill (when prefill is still ongoing).
    
    Args:
        scheduler: The scheduler instance with logprob processing methods
        req: The request to process logprobs for
        req_idx: Index of the request in the batch
        logits_output: Logits processor output containing logprob data
        extend_input_len_per_req: List of input lengths per request
        extend_logprob_start_len_per_req: List of logprob start lengths per request
        logprob_pt: Current logprob pointer position
        next_token_ids: List of next token IDs (required for complete prefill)
        is_last_prefill_chunk: Whether this is the last prefill chunk (True for complete prefill)
    
    Returns:
        Updated logprob_pt after processing
    """
    if not req.return_logprob:
        return logprob_pt
    
    assert extend_logprob_start_len_per_req is not None
    assert extend_input_len_per_req is not None
    
    extend_logprob_start_len = extend_logprob_start_len_per_req[req_idx]
    extend_input_len = extend_input_len_per_req[req_idx]
    num_input_logprobs = extend_input_len - extend_logprob_start_len
    
    if is_last_prefill_chunk:
        # Complete prefill: use add_logprob_return_values which handles both
        # output and input logprobs
        assert next_token_ids is not None, "next_token_ids required for complete prefill"
        scheduler.add_logprob_return_values(
            req_idx,
            req,
            logprob_pt,
            next_token_ids,
            num_input_logprobs,
            logits_output,
        )
        logprob_pt += num_input_logprobs
    else:
        # Chunked prefill: only process input logprobs if there are any
        if extend_logprob_start_len < extend_input_len:
            scheduler.add_input_logprob_return_values(
                req_idx,
                req,
                logits_output,
                logprob_pt,
                num_input_logprobs,
                last_prefill_chunk=False,
            )
            logprob_pt += num_input_logprobs
    
    return logprob_pt


class TransferContext:
    """
    Multiple requests can share the same batch and result through this object.
    Thread-safe: resolve() can be called multiple times but executes only once.
    """

    def __init__(
        self,
        batch: "ScheduleBatch",
        result: "GenerationBatchResult",
        metadata_buffers: MetadataBuffers,
        scheduler: Optional["Scheduler"] = None,
    ) -> None:
        self.batch = batch
        self.result = result
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self._resolved = False
    
    def resolve(self) -> None:
        """
        Synchronize CUDA and populate request metadata buffers.
        This method creates read-only snapshots of reqs to avoid race conditions
        with the main thread. It does NOT modify the original req objects.
        Safe to call multiple times - only executes once.
        """
        if self._resolved:
            return
        
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            self.result.logits_output,
            self.result.next_token_ids,
            self.result.extend_input_len_per_req,
            self.result.extend_logprob_start_len_per_req,
            self.result.copy_done,
        )
        if copy_done is not None:
            copy_done.synchronize()
        
        # Convert to list if tensor for processing
        next_token_ids_list:List[int] = (
            next_token_ids.tolist()
            if isinstance(next_token_ids, torch.Tensor)
            else next_token_ids
        )
        
        # Prepare logprobs data if needed
        if self.batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        
        # Create read-only snapshots of reqs and populate metadata buffers
        # This avoids race conditions with the main thread which modifies reqs
        logprob_pt = 0
        for i, (req, next_token_id) in enumerate(
            zip(self.batch.reqs, next_token_ids_list, strict=True)
        ):
            if req.is_chunked <= 0:
                # Create a snapshot of the req with updated data for set_buf
                # We need to include the next_token_id in output_ids for set_buf
                snapshot = ReqMetadataView.from_req(req)
                
                snapshot.output_ids = [next_token_id]
                
                # Update snapshot with spec info if available
                # This is safe because we're modifying the snapshot, not the original req
                if self.batch.spec_info is not None:
                    snapshot.output_topk_p = self.batch.spec_info.topk_p[i]
                    snapshot.output_topk_index = self.batch.spec_info.topk_index[i]
                    snapshot.hidden_states_tensor = (
                        self.batch.spec_info.hidden_states[i].cpu().clone()
                    )
                else:
                    snapshot.hidden_states_tensor = None
                
                # This updates the snapshot's logprobs fields without modifying the original req
                # Use mutable=True so modifications to minimal_req update the snapshot
                minimal_req_for_logprobs = snapshot.to_req_like(mutable=True)
                logprob_pt = process_logprobs_for_request(
                    self.scheduler,
                    minimal_req_for_logprobs,
                    i,
                    logits_output,
                    extend_input_len_per_req,
                    extend_logprob_start_len_per_req,
                    logprob_pt,
                    next_token_ids=[next_token_id],
                    is_last_prefill_chunk=False,
                )
                
                # Use snapshot to populate metadata buffer (read-only operation)
                minimal_req = snapshot.to_req_like(mutable=False)
                self.metadata_buffers.set_buf(minimal_req)
            else:
                # Chunked prefill: skip logprobs processing here
                # The main thread will process logprobs when the req is scheduled again.
                # set_buf will be called only when the last chunk is processed, at which
                # point the main thread has already filled all logprobs.
                pass
        
        self._resolved = True


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


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
        from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


#########################
# KV Pages
#########################


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []
