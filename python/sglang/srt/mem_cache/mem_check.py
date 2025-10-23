from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.utils import rank0_log

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.radix_cache import TreeNode


def used_pages_from_req(reqs: list[torch.Tensor], page_size: int) -> torch.Tensor:
    used_tokens = torch.cat(reqs)
    used_pages_from_req = used_tokens.floor_divide_(page_size).unique()
    return used_pages_from_req


def used_pages_from_pool(free_pages: torch.Tensor, pool_size: int) -> torch.Tensor:
    all_pages = torch.arange(
        1, pool_size + 1, dtype=free_pages.dtype, device=free_pages.device
    )
    not_present = torch.isin(all_pages, free_pages, invert=True, assume_unique=True)
    used_pages = all_pages[not_present].to(torch.int32)
    return used_pages


def from_req_to_token_pool(scheduler: Scheduler, req: Req, length: int) -> torch.Tensor:
    return scheduler.req_to_token_pool.req_to_token[
        req.req_pool_idx,
        :length,
    ]


def from_tree_cache(scheduler: Scheduler) -> torch.Tensor:
    tree_cache = scheduler.tree_cache
    if tree_cache.disable:
        return torch.tensor([], dtype=torch.int32, device="cuda")
    values = []

    def _dfs_helper(node: TreeNode):
        if node.lock_ref == 0:
            values.append(node.value)
        for _, child in node.children.items():
            _dfs_helper(child)

    _dfs_helper(tree_cache.root_node)
    if values:
        return torch.cat(values).to("cuda")
    else:
        return torch.tensor([], dtype=torch.int32, device="cuda")


def get_req_from_batches(scheduler: Scheduler):
    cur_batch_reqs = (
        {r.rid: r for r in scheduler.cur_batch.reqs} if scheduler.cur_batch else {}
    )
    running_batch_reqs = (
        {r.rid: r for r in scheduler.running_batch.reqs}
        if scheduler.running_batch
        else {}
    )
    last_batch_reqs = (
        {r.rid: r for r in scheduler.last_batch.reqs} if scheduler.last_batch else {}
    )
    return cur_batch_reqs, running_batch_reqs, last_batch_reqs


def is_extend(batch: ScheduleBatch) -> bool:
    return batch.forward_mode.is_extend() or batch.forward_mode.is_fake_extend()


def decode_queue_check(self: Scheduler):
    requests: dict[str, torch.Tensor] = {}  # rid -> used tokens
    for req in self.waiting_queue:
        requests[req.rid] = from_req_to_token_pool(self, req, req.seqlen - 1)
    for req in self.disagg_decode_transfer_queue.queue:
        requests[req.req.rid] = from_req_to_token_pool(self, req.req, req.req.seqlen)
    return requests


def prefill_queue_check(self: Scheduler):
    requests: dict[str, torch.Tensor] = {}  # rid -> used tokens
    for req in self.disagg_prefill_inflight_queue:
        requests[req.rid] = from_req_to_token_pool(self, req, req.seqlen - 1)
    return requests


def memory_check(self: Scheduler):
    if self.swa:
        rank0_log("SWA is enabled, skipping memory check")
        return
    if self.enable_hierarchical_cache:
        rank0_log("Hierarchical cache is enabled, skipping memory check")
        return

    self.token_to_kv_pool_allocator.merge_and_sort_free()

    cur_batch_reqs, running_batch_reqs, last_batch_reqs = get_req_from_batches(self)

    requests: dict[str, torch.Tensor] = {}  # rid -> used tokens

    # Active requests

    # only overlap scheduler cares about last batch
    if self.enable_overlap and self.last_batch:
        tmp_batch: ScheduleBatch = self.result_queue[0][0]
        # Last batch can be modified in-place as it will become running batch in the next iteration.
        # We use batch info from tmp_batch. Note that the requests are copied to tmp_batch by reference.
        for i, req in enumerate(tmp_batch.reqs):
            rid = req.rid
            if tmp_batch.forward_mode.is_extend():
                requests[rid] = from_req_to_token_pool(self, req, len(req.fill_ids))
            # elif tmp_batch.forward_mode.is_fake_extend():
            elif (
                tmp_batch.forward_mode.is_extend()
                and self.disaggregation_mode == DisaggregationMode.DECODE
            ):
                requests[rid] = from_req_to_token_pool(self, req, req.seqlen - 1)
            elif req.finished() or req.is_retracted:
                if (req.seqlen - 1) % self.page_size == 0:
                    # Have one extra token
                    requests[rid] = tmp_batch.out_cache_loc[i : i + 1]
            else:
                # this request is running
                requests[rid] = from_req_to_token_pool(self, req, req.seqlen)

    # current batch
    for rid, req in cur_batch_reqs.items():
        assert not req.finished(), "current batch should not contain finished requests."
        if rid in last_batch_reqs and self.enable_overlap:
            # this request is scheduled again while it is running.
            # The last token of the previous iteration has not been appended so the #token is beyond seqlen.
            # if not tmp_batch.forward_mode.is_fake_extend():
            if not (
                tmp_batch.forward_mode.is_extend()
                and self.disaggregation_mode == DisaggregationMode.DECODE
            ):
                # for fake extend, the last token has been generated, so the last batch is not running.
                requests[rid] = from_req_to_token_pool(self, req, req.seqlen + 1)
                continue
        if self.cur_batch.forward_mode.is_extend():
            requests[rid] = from_req_to_token_pool(self, req, len(req.fill_ids))
        # elif self.cur_batch.forward_mode.is_fake_extend():
        elif (
            self.cur_batch.forward_mode.is_extend()
            and self.disaggregation_mode == DisaggregationMode.DECODE
        ):
            requests[rid] = from_req_to_token_pool(self, req, req.seqlen - 1)
        elif self.cur_batch.forward_mode.is_decode():
            if self.spec_algorithm.is_eagle():
                assert not self.enable_overlap, "Spec decode does not support overlap"
                requests[rid] = from_req_to_token_pool(self, req, req.seqlen - 1)
            else:
                requests[rid] = from_req_to_token_pool(self, req, req.seqlen)
        else:
            raise NotImplementedError(
                f"Unsupported forward mode: {self.cur_batch.forward_mode.name}"
            )

    # running batch
    for rid, req in running_batch_reqs.items():
        if rid in requests:
            continue
        if req.finished():
            assert is_extend(
                self.cur_batch
            ), "Only extend can contain finished requests."
            continue
        assert is_extend(self.cur_batch), "Current batch must be extend."
        if self.enable_overlap:
            assert is_extend(self.last_batch), "Last batch must be extend."
        requests[rid] = from_req_to_token_pool(self, req, req.seqlen - 1)

    # radix cache
    requests["tree_cache"] = from_tree_cache(self)

    # Pending requests
    if self.disaggregation_mode == DisaggregationMode.DECODE:
        requests.update(decode_queue_check(self))
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        requests.update(prefill_queue_check(self))

    # for dp attention
    if len(requests) == 0:
        assert self.cur_batch.forward_mode.is_idle(), "Current batch should be idle"
        return

    # Actual memory usage
    if self.page_size == 1:
        pool_size = self.token_to_kv_pool_allocator.size
    else:
        pool_size = self.token_to_kv_pool_allocator.num_pages

    from_req = used_pages_from_req(list(requests.values()), self.page_size)
    from_pool = used_pages_from_pool(
        self.token_to_kv_pool_allocator.free_pages, pool_size
    )
    memory_leak = (
        from_req.numel() != from_pool.numel() or torch.any(from_req != from_pool).item()
    )
    # print error info if memory leak
    if memory_leak:
        logger.error(f"Used pages counted from req: {from_req.numel()} pages")
        logger.error(f"Used pages counted from pool: {from_pool.numel()} pages")
        data_to_dump = {
            "waiting_queue": self.waiting_queue,
            "running_batch": repr(self.running_batch),
            "last_batch": repr(self.last_batch),
            "current_batch": repr(self.cur_batch),
            "pages_from_req": from_req.cpu(),
            "pages_from_pool": from_pool.cpu(),
            "token_usage": {r: t.cpu() for r, t in requests.items()},
        }
        # Dump requests

        crash_dump_folder = "/root/debug/"

        if crash_dump_folder and self.attn_tp_rank == 0:
            object_name = (
                f'memory_check_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'
            )
            filename = os.path.join(
                crash_dump_folder,
                object_name,
            )
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(data_to_dump, f)
            logger.error(f"Dumped memory check dump to {filename}")
        raise RuntimeError(f"Memory check failed")
