from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.pool_patcher.adapters.mha import attach_mha
from sglang.srt.kv_canary.pool_patcher.api import register_pool_attacher
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.model_executor.forward_batch_info import ForwardMode

DEFAULT_DEVICE: torch.device = torch.device("cuda")


@dataclass
class FakeMHAPool:
    layer_num: int
    k_buffer: List[torch.Tensor]
    v_buffer: List[torch.Tensor]
    page_size: int = 1

    def get_contiguous_buf_infos(self):
        ptrs = [b.data_ptr() for b in self.k_buffer] + [
            b.data_ptr() for b in self.v_buffer
        ]
        lens = [b.nbytes for b in self.k_buffer] + [b.nbytes for b in self.v_buffer]
        item_lens = [b[0].nbytes * self.page_size for b in self.k_buffer] + [
            b[0].nbytes * self.page_size for b in self.v_buffer
        ]
        return ptrs, lens, item_lens


def make_mha_pool(
    device: torch.device = DEFAULT_DEVICE,
    *,
    num_slots: int = 16,
    dim: int = 8,
    layer_num: int = 2,
) -> FakeMHAPool:
    k_layers = [
        torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
        for _ in range(layer_num)
    ]
    v_layers = [
        torch.zeros(num_slots, dim, dtype=torch.float16, device=device)
        for _ in range(layer_num)
    ]
    return FakeMHAPool(layer_num=layer_num, k_buffer=k_layers, v_buffer=v_layers)


def make_base_config() -> CanaryConfig:
    return CanaryConfig(
        mode=CanaryMode.RAISE,
        ring_capacity=1024,
    )


def make_req_to_token_pool(
    device: torch.device = DEFAULT_DEVICE,
    *,
    max_reqs: int = 8,
    max_seq_len: int = 32,
) -> SimpleNamespace:
    req_to_token = torch.zeros(max_reqs, max_seq_len, dtype=torch.int32, device=device)
    return SimpleNamespace(
        req_to_token=req_to_token, size=max_reqs, max_context_len=max_seq_len
    )


def make_forward_batch(
    device: torch.device = DEFAULT_DEVICE,
    *,
    bs: int = 2,
    seq_lens_list: tuple[int, ...] = (3, 4),
    req_pool_indices: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    seq_lens_sum: Optional[int] = None,
    extend_prefix_lens: Optional[torch.Tensor] = None,
    extend_prefix_lens_cpu: Optional[list] = None,
    extend_seq_lens: Optional[torch.Tensor] = None,
    extend_seq_lens_cpu: Optional[list] = None,
    is_extend: bool = False,
    is_target_verify: bool = False,
    is_draft_extend_v2: bool = False,
    spec_info: Optional[object] = None,
    input_ids: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    out_cache_loc: Optional[torch.Tensor] = None,
    num_token_non_padded_cpu: Optional[int] = None,
) -> SimpleNamespace:
    seq_lens_default = list(seq_lens_list[:bs])
    if req_pool_indices is None:
        req_pool_indices = torch.tensor([1, 2][:bs], dtype=torch.int64, device=device)
    if seq_lens is None:
        seq_lens = torch.tensor(seq_lens_default, dtype=torch.int32, device=device)
    if seq_lens_sum is None:
        seq_lens_sum = int(sum(seq_lens_default))
    if input_ids is None:
        input_ids = torch.zeros(bs, dtype=torch.int32, device=device)
    if positions is None:
        # Default to decode-canonical: positions = seq_lens - 1 (one-token-per-req decode write
        # at the post-bump tail). Plan input derives decode prefix_lens from positions
        # directly, so the default must keep parity.
        positions = (seq_lens.to(torch.int64) - 1).clamp(min=0).to(torch.int32)
    if out_cache_loc is None:
        out_cache_loc = torch.zeros(bs, dtype=torch.int32, device=device)

    if is_extend:
        forward_mode = ForwardMode.EXTEND
    elif is_target_verify:
        forward_mode = ForwardMode.TARGET_VERIFY
    elif is_draft_extend_v2:
        forward_mode = ForwardMode.DRAFT_EXTEND_V2
    else:
        forward_mode = ForwardMode.DECODE
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=bs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_sum=seq_lens_sum,
        extend_prefix_lens=extend_prefix_lens,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        spec_info=spec_info,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        num_token_non_padded_cpu=num_token_non_padded_cpu,
        req_all_ids_flat=None,
        req_all_ids_lens=None,
    )


def make_buffer_group(
    *,
    device: torch.device = DEFAULT_DEVICE,
    kind: PoolKind = PoolKind.FULL,
    has_v: bool = True,
    swa_index_lut: Optional[torch.Tensor] = None,
    num_slots: int = 4,
    kv_token_id_vs_position_offset: int = 0,
) -> CanaryBufferGroup:
    def _zero() -> torch.Tensor:
        return torch.zeros(
            num_slots, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device
        )

    return CanaryBufferGroup(
        kind=kind,
        k_head=_zero(),
        k_tail=_zero(),
        v_head=_zero() if has_v else None,
        v_tail=_zero() if has_v else None,
        swa_index_lut=swa_index_lut,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )


def make_radix_cache(
    slot_lists: List[List[int]], device: torch.device = DEFAULT_DEVICE
):
    cache = RadixCache.__new__(RadixCache)
    cache.device = device
    cache.page_size = 1
    cache.disable = False

    root = TreeNode()
    root.value = torch.tensor(
        slot_lists[0] if slot_lists else [], dtype=torch.int32, device=device
    )
    cache.root_node = root

    current = root
    for child_slots in slot_lists[1:]:
        child = TreeNode()
        child.value = torch.tensor(child_slots, dtype=torch.int32, device=device)
        child.parent = current
        current.children[child.id] = child
        current = child

    return cache


register_pool_attacher(FakeMHAPool, attach_mha)
