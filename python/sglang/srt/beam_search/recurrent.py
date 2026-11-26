from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_memory, get_parallel
from sglang.srt.utils import is_cuda
from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.beam_search.batch_tail import BeamTail
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class GdnBeamReplayInputs(msgspec.Struct, frozen=True):
    group_ids: torch.Tensor
    slots: torch.Tensor


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
        and (v * k) // factor >= 2 * (v + k + 1)
    )


def build_gdn_beam_replay_inputs(
    *, tail: BeamTail, num_rows: int, device: torch.device
) -> GdnBeamReplayInputs | None:
    replay = [entry for entry in tail.entries if entry.group.use_gdn_beam_replay]
    if not replay:
        return None
    group_ids = [-1] * num_rows
    slots = []
    for group_id, entry in enumerate(replay):
        kv = entry.group.leader.kv
        assert kv.mamba_pool_idx is not None and kv.beam_mamba_slots is not None
        slots.extend((kv.mamba_pool_idx.view(1), kv.beam_mamba_slots[:1]))
        group_ids[entry.leader_idx] = group_id
        start = tail.num_base_rows + entry.start
        end = tail.num_base_rows + entry.end
        group_ids[start:end] = [group_id] * (end - start)
    return GdnBeamReplayInputs(
        group_ids=async_h2d(group_ids, dtype=torch.int32, device=device),
        slots=torch.cat(slots),
    )


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
