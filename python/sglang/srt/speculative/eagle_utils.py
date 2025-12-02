import math
import os
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num


@dataclass
class HiddenStateDumpPayload:
    reqs: List[Req]
    aux_hidden_states: torch.Tensor
    last_hidden_states: torch.Tensor
    accept_length_per_req_cpu: List[int]


class HiddenStateDumper:
    def __init__(self, server_args: ServerArgs, tp_rank: int = 0, tp_size: int = 1):
        self.server_args = server_args
        self.dump_path: str = server_args.speculative_eagle_hidden_states_dump_path
        self.tp_rank: int = tp_rank
        self.tp_size: int = tp_size
        self.dump_stream = torch.cuda.Stream()
        self.buffer_pool = FlatBufferPool(
            available_size=server_args.speculative_eagle_dump_buffer_pool_size
        )
        self.dump_executor = futures.ProcessPoolExecutor(
            max_workers=server_args.speculative_eagle_dump_worker_num,
        )
        self.dump_worker_idx: int = -1
        self.payloads: HiddenStateDumpPayload = None
        self.dump_tokens = {}

        os.makedirs(
            server_args.speculative_eagle_hidden_states_dump_path, exist_ok=True
        )

    def prepare_payload(
        self,
        reqs: List[Req],
        hidden_states: torch.Tensor,
        last_hidden_states: torch.Tensor,
        accept_length_per_req_cpu: List[int],
    ):
        self.payloads = HiddenStateDumpPayload(
            reqs=reqs,
            aux_hidden_states=hidden_states,
            last_hidden_states=last_hidden_states,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
        )

    def process_dump_payload(
        self,
    ):
        if self.payloads is None:
            return

        reqs = self.payloads.reqs
        aux_hidden_states = self.payloads.aux_hidden_states
        last_hidden_states = self.payloads.last_hidden_states
        accept_length_per_req_cpu = self.payloads.accept_length_per_req_cpu
        with torch.cuda.stream(self.dump_stream):
            aux_hidden_states = aux_hidden_states.cpu()
            last_hidden_states = last_hidden_states.cpu()

        accept_len_offset = 0
        for i, req in enumerate(reqs):
            if req.rid not in self.dump_tokens:
                self.dump_tokens[req.rid] = 0
                self.append_req_hidden_states(
                    req,
                    req.hidden_states_for_dump.cpu(),
                    req.last_hidden_states_for_dump.cpu(),
                )
            accept_len = accept_length_per_req_cpu[i] + 1  # +1 for a bonus token
            self.append_req_hidden_states(
                req,
                aux_hidden_states[accept_len_offset : accept_len_offset + accept_len],
                last_hidden_states[accept_len_offset : accept_len_offset + accept_len],
            )
            accept_len_offset += accept_len

            self._dump_if_needed(req)

            if req.finished():
                self.buffer_pool.release_buffer(f"{req.rid}_hs")
                self.buffer_pool.release_buffer(f"{req.rid}_lhs")
                self.dump_tokens.pop(req.rid)

        assert accept_len_offset == aux_hidden_states.shape[0]
        self.payloads = None

    def append_req_hidden_states(
        self,
        req: Req,
        aux_hidden_states: torch.Tensor,
        last_hidden_states: torch.Tensor,
    ):
        num_new_tokens = aux_hidden_states.shape[0]
        H = aux_hidden_states.shape[1]
        H_last = last_hidden_states.shape[1]
        dump_tokens = self.dump_tokens[req.rid]

        hs_buf = self.buffer_pool.get_buffer(
            f"{req.rid}_hs",
            aux_hidden_states.device,
            aux_hidden_states.dtype,
            dump_tokens * H + aux_hidden_states.numel(),
        )
        hs_buf[dump_tokens * H : (dump_tokens + num_new_tokens) * H] = (
            aux_hidden_states.view(-1)
        )
        lhs_buf = self.buffer_pool.get_buffer(
            f"{req.rid}_lhs",
            last_hidden_states.device,
            last_hidden_states.dtype,
            dump_tokens * H_last + last_hidden_states.numel(),
        )
        lhs_buf[dump_tokens * H_last : (dump_tokens + num_new_tokens) * H_last] = (
            last_hidden_states.view(-1)
        )
        dump_tokens += num_new_tokens
        req.hidden_states_for_dump = hs_buf[: dump_tokens * H].view(dump_tokens, H)
        req.last_hidden_states_for_dump = lhs_buf[: dump_tokens * H_last].view(
            dump_tokens, H_last
        )
        self.dump_tokens[req.rid] = dump_tokens

    def _dump_if_needed(self, req: Req):
        if not req.finished():
            return

        self.dump_worker_idx = (self.dump_worker_idx + 1) % self.tp_size
        if self.dump_worker_idx != self.tp_rank:
            return

        assert (
            self.server_args.speculative_eagle_hidden_states_dump_path is not None
        ), "speculative_eagle_hidden_states_dump_path must be set"

        if self.server_args.speculative_eagle_dump_accept_rate_threshold < 1.0:
            acceptance_rate = req.spec_accepted_tokens / (
                req.spec_verify_ct * self.server_args.speculative_num_draft_tokens
            )
            # Skip dump if acceptance rate is higher than threshold
            if (
                acceptance_rate
                >= self.server_args.speculative_eagle_dump_accept_rate_threshold
            ):
                return

        dump_path = os.path.join(
            self.server_args.speculative_eagle_hidden_states_dump_path,
            f"{req.rid}_data.ckpt",
        )

        self.dump_executor.submit(
            dump_hidden_states,
            dump_path,
            req.last_hidden_states_for_dump[: req.seqlen - 1],
            req.hidden_states_for_dump[: req.seqlen - 1],
            req.origin_input_ids,
            req.output_ids,
        )


def dump_hidden_states(
    dump_path: str,
    last_hidden_states: torch.Tensor,
    aux_hidden_states: torch.Tensor,
    origin_input_ids: List[int],
    output_ids: List[int],
):
    input_ids = torch.tensor(origin_input_ids + output_ids[:-1], dtype=torch.long).view(
        -1
    )
    loss_mask = torch.zeros_like(input_ids)
    loss_mask[len(origin_input_ids) :] = 1
    save_dict = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_state": last_hidden_states,
        "aux_hidden_state": aux_hidden_states,
    }
    torch.save(save_dict, dump_path)


class FlatBufferPool:
    def __init__(self, available_size: int = 256):
        self.pool = dict()
        self.available_buffers = deque(maxlen=available_size)

    def get_buffer(
        self,
        key: str,
        device: torch.device,
        dtype: torch.dtype,
        needed_elems: int,
    ):
        return self._ensure_buf(key, device, dtype, needed_elems)

    def release_buffer(self, key: str):
        if key in self.pool:
            self.available_buffers.append(self.pool[key])
            del self.pool[key]

    def _ensure_buf(
        self,
        key: str,
        device: torch.device,
        dtype: torch.dtype,
        needed_elems: int,
        growth: float = 2.0,
        min_elem: int = 2 * 1024 * 1024,
    ):
        """
        Reusable expandable flat buffer (1D). Grows geometrically.
        """

        buf = self.pool[key] if key in self.pool else None
        if buf is None:
            buf = self.available_buffers.popleft() if self.available_buffers else None
            if buf is not None:
                self.pool[key] = buf

        need_new = (
            buf is None
            or buf.device != device
            or buf.dtype != dtype
            or buf.numel() < needed_elems
        )

        if need_new:
            # geometric growth to reduce realloc frequency
            cap = max(needed_elems, min_elem)
            if buf is not None and buf.numel() < needed_elems:
                cap = max(int(buf.numel() * growth), cap)
            new_buf = torch.empty(cap, device=device, dtype=dtype)
            if buf is not None and buf.dtype == dtype and buf.device == device:
                new_buf[: buf.numel()].copy_(buf)
            self.pool[key] = new_buf
            return new_buf

        return buf
