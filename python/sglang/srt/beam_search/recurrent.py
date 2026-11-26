from __future__ import annotations

import logging
from math import ceil
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.ops.attention.fla.beam_replay_layout import (
    GDN_BEAM_REPLAY_WINDOW,
    records_per_state_slot,
)
from sglang.srt.beam_search.fork import plan_state_forks
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_memory, get_parallel
from sglang.srt.utils import is_cuda
from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.beam_search.batch_tail import BeamTail
    from sglang.srt.beam_search.beam_group import BeamGroup
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class GdnBeamReplayInputs(msgspec.Struct, frozen=True):
    group_ids: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    window_rows: torch.Tensor | None = None


class GdnBeamReplayWindow:
    """A bounded tape with a dense checkpoint per live ancestor."""

    def __init__(
        self,
        checkpoints: torch.Tensor,
        arena: torch.Tensor,
        records_per_slot: int,
        num_record_steps: int = GDN_BEAM_REPLAY_WINDOW,
    ):
        self.checkpoints = checkpoints.to(dtype=torch.int32)
        width = checkpoints.numel()
        self.rows = torch.arange(width, device=checkpoints.device)
        self.anchor_rows = torch.zeros_like(self.rows)
        records = torch.arange(num_record_steps * width, device=checkpoints.device)
        self.records = (
            arena[records // records_per_slot].to(torch.int32) * records_per_slot
            + (records % records_per_slot).to(torch.int32)
        ).view(num_record_steps, width)
        self.history: torch.Tensor | None = None

    @staticmethod
    def depth(num_generated: int) -> int:
        return (num_generated - 1) % (GDN_BEAM_REPLAY_WINDOW + 1)

    def rows_for_decode(self, group: BeamGroup) -> torch.Tensor:
        depth = self.depth(group.num_generated)
        state_slots = self.checkpoints[self.anchor_rows]
        padding = state_slots.new_full(
            (state_slots.numel(), GDN_BEAM_REPLAY_WINDOW - depth), -1
        )
        records = (
            self.records[depth]
            if depth < GDN_BEAM_REPLAY_WINDOW
            and not group.terminal_state_is_read_only()
            else torch.full_like(state_slots, -1)
        )
        return torch.cat(
            (
                state_slots[:, None],
                records[:, None],
                self.history.transpose(0, 1)
                if self.history is not None
                else padding[:, :0],
                padding,
            ),
            dim=1,
        )

    def advance(
        self, group: BeamGroup, parents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        depth = self.depth(group.num_generated)
        if depth == 0:
            self.history = None
        else:
            selected = self.records[depth - 1, parents][None, :]
            self.history = (
                selected
                if self.history is None
                else torch.cat((self.history[:, parents], selected), dim=0)
            )
        anchors = self.anchor_rows[parents]
        if depth == GDN_BEAM_REPLAY_WINDOW and not group.terminal_state_is_read_only():
            # A fold writes dense state in-place. Give each child a private
            # checkpoint before the kernel reads or writes any of those slots.
            self.anchor_rows, src, dst = plan_state_forks(
                slots=self.rows, parents=anchors
            )
            return self.checkpoints[src], self.checkpoints[dst]
        self.anchor_rows = anchors
        return None


def gdn_beam_replay_record_steps(max_new_tokens: int) -> int:
    return min(GDN_BEAM_REPLAY_WINDOW, max_new_tokens - 2)


def gdn_beam_replay_arena_size(
    width: int, max_new_tokens: int, state: StateStorage
) -> tuple[int, int]:
    temporal = state.pool.mamba_cache.temporal
    v, k = temporal.shape[-2:]
    per_slot = records_per_state_slot(v, k, temporal.element_size())
    return ceil(
        gdn_beam_replay_record_steps(max_new_tokens) * width / per_slot
    ), per_slot


def resolve_beam_state_optimizations(
    *, model_runner: ModelRunner, state: StateStorage | None, static_storage: bool
) -> tuple[bool, bool]:
    requested_replay = envs.SGLANG_ENABLE_BEAM_GDN_RANK_ONE_REPLAY.get()
    if not (envs.SGLANG_ENABLE_BEAM_TERMINAL_READ_ONLY_STATE.get() or requested_replay):
        return False, False
    read_only = (
        state is not None
        and is_cuda()
        and static_storage
        and get_memory().disable_radix_cache
        and not get_exec().mamba.enable_mamba_extra_buffer
        and not get_exec().overlap.enable_two_batch_overlap
        and get_parallel().attn_cp_size == 1
        and get_parallel().attn_dcp_size == 1
        and _supports_terminal_read_only_state(state=state, model_runner=model_runner)
    )
    if read_only:
        logger.info("Enabled terminal read-only beam state.")
    else:
        logger.warning(
            "Terminal read-only beam state is unsupported by the current "
            "attention backend or cache configuration; using private state."
        )

    replay = (
        requested_replay
        and read_only
        and state is not None
        and _supports_gdn_beam_replay(backend=model_runner.attn_backend, state=state)
    )
    if replay:
        logger.info("Enabled rank-one GDN beam replay.")
    elif requested_replay:
        logger.warning(
            "Rank-one beam replay requires packed Triton GDN and a compatible "
            "static state cache; using the standard state path."
        )
    return read_only, replay


def _linear_decode_backend(backend: AttentionBackend) -> AttentionBackend:
    from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    if isinstance(backend, HybridAttnBackend):
        backend = backend.decode_backend
    if isinstance(backend, HybridLinearAttnBackend):
        backend = backend.linear_attn_backend
    return backend


def _supports_terminal_read_only_state(
    *, state: StateStorage, model_runner: ModelRunner
) -> bool:
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        Mamba2AttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
    from sglang.srt.layers.attention.mamba.mamba1 import MambaMixer1

    if (
        state.own_state is not None
        or state.short_conv_pool.enabled
        or state.ngram_pool.enabled
        or state.pool.replayssm_write_pos is not None
    ):
        return False

    linear = _linear_decode_backend(model_runner.attn_backend)
    if type(linear) is GDNAttnBackend:
        return linear.supports_terminal_read_only_state
    if type(linear) is not Mamba2AttnBackend or get_exec().mamba.mamba_backend not in (
        None,
        "triton",
    ):
        return False
    # Mamba-1 shares the Mamba2 backend but does not support masked writeback.
    mixers = [
        module
        for module in model_runner.model.modules()
        if isinstance(module, (MambaMixer1, MambaMixer2))
    ]
    return bool(mixers) and all(type(mixer) is MambaMixer2 for mixer in mixers)


def _supports_gdn_beam_replay(
    *, backend: AttentionBackend, state: StateStorage
) -> bool:
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

    if type(_linear_decode_backend(backend)) is not GDNAttnBackend:
        return False
    temporal = state.pool.mamba_cache.temporal
    v, k = temporal.shape[-2:]
    factor = 4 // temporal.element_size()
    return (
        temporal.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and temporal.stride(-1) == 1
        and temporal.stride(-2) == k
        and temporal.storage_offset() % factor == 0
        and k % factor == 0
        and all(stride % factor == 0 for stride in temporal.stride()[:-1])
        and records_per_state_slot(v, k, temporal.element_size()) >= 2
    )


def build_gdn_beam_replay_inputs(
    *, tail: BeamTail, num_rows: int, device: torch.device
) -> GdnBeamReplayInputs | None:
    inline = [entry for entry in tail.entries if entry.group.use_gdn_beam_replay]
    windowed = [
        entry
        for entry in tail.entries
        if entry.group.gdn_replay_window is not None
        and not entry.group.launch_complete()
    ]
    if not inline and not windowed:
        return None
    group_ids = slots = window_rows = None
    if inline:
        host_ids = [-1] * num_rows
        inline_slots = []
        for group_id, entry in enumerate(inline):
            kv = entry.group.leader.kv
            assert kv.mamba_pool_idx is not None and kv.beam_mamba_slots is not None
            inline_slots.extend((kv.mamba_pool_idx.view(1), kv.beam_mamba_slots[:1]))
            host_ids[entry.leader_idx] = group_id
            start = tail.num_base_rows + entry.start
            end = tail.num_base_rows + entry.end
            host_ids[start:end] = [group_id] * (end - start)
        group_ids = async_h2d(host_ids, dtype=torch.int32, device=device)
        slots = torch.cat(inline_slots)
    if windowed:
        ids = []
        metadata = []
        for entry in windowed:
            group = entry.group
            ids.append(entry.leader_idx)
            ids.extend(
                range(tail.num_base_rows + entry.start, tail.num_base_rows + entry.end)
            )
            metadata.append(group.gdn_replay_window.rows_for_decode(group))
        window_rows = torch.full(
            (num_rows, GDN_BEAM_REPLAY_WINDOW + 2), -1, device=device, dtype=torch.int32
        )
        window_rows.index_copy_(
            0, async_h2d(ids, dtype=torch.int64, device=device), torch.cat(metadata)
        )
    return GdnBeamReplayInputs(group_ids, slots, window_rows)


def build_read_only_state_mask(
    *, tail: BeamTail, num_rows: int, device: torch.device
) -> torch.Tensor | None:
    terminal = [
        entry for entry in tail.entries if entry.group.terminal_state_is_read_only()
    ]
    if not terminal:
        return None
    mask = [False] * num_rows
    for entry in terminal:
        mask[entry.leader_idx] = True
        start = tail.num_base_rows + entry.start
        end = tail.num_base_rows + entry.end
        mask[start:end] = [True] * (end - start)
    return async_h2d(mask, dtype=torch.bool, device=device)
