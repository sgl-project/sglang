from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.distributed import get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.utils import init_cublas

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedAttentionBackendStr:
    prefill: str
    decode: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AttentionBackends:
    attn_backend: AttentionBackend
    decode_attn_backend: Optional[AttentionBackend]
    decode_attn_backend_group: list[AttentionBackend]
    prefill_attention_backend_str: str
    decode_attention_backend_str: str


def configure_aux_hidden_state_capture(
    *,
    model,
    eagle_use_aux_hidden_state: bool,
    eagle_aux_hidden_state_layer_ids,
    dflash_use_aux_hidden_state: bool,
    dflash_target_layer_ids,
) -> None:
    if eagle_use_aux_hidden_state:
        model.set_eagle3_layers_to_capture(eagle_aux_hidden_state_layer_ids)
    if dflash_use_aux_hidden_state:
        if not hasattr(model, "set_dflash_layers_to_capture"):
            raise ValueError(
                f"Model {model.__class__.__name__} does not implement "
                "set_dflash_layers_to_capture, which is required for DFLASH."
            )
        model.set_dflash_layers_to_capture(dflash_target_layer_ids)


def build_attention_backends(*, model_runner: ModelRunner) -> AttentionBackends:
    # Takes the whole ModelRunner by design (the documented leaf exception to
    # "pass narrow params, not the god object"): the backend constructor contract
    # is ``ATTENTION_BACKENDS[str](model_runner)``, and backends retain a live
    # reference for forward-time reads of mutable runner state (KV pools).
    server_args = model_runner.server_args

    # device-specific pre-init. TODO: platform interface (separate PR).
    if model_runner.device in ("cuda", "musa"):
        init_cublas()

    resolved = _resolve_attention_backend_strs(
        server_args=server_args, is_draft_worker=model_runner.is_draft_worker
    )

    if server_args.enable_pdmux:
        attn_backend = _build_resolved_backend(
            model_runner=model_runner, resolved=resolved, init_new_workspace=True
        )
        decode_attn_backend_group = [
            _build_resolved_backend(
                model_runner=model_runner, resolved=resolved, init_new_workspace=False
            )
            for _ in range(server_args.sm_group_num)
        ]
        decode_attn_backend = decode_attn_backend_group[0]
    elif server_args.enable_two_batch_overlap and not model_runner.is_draft_worker:
        attn_backend = TboAttnBackend.init_new(
            lambda: _build_resolved_backend(
                model_runner=model_runner, resolved=resolved, init_new_workspace=False
            )
        )
        decode_attn_backend = None
        decode_attn_backend_group = []
    else:
        attn_backend = _build_resolved_backend(
            model_runner=model_runner, resolved=resolved, init_new_workspace=False
        )
        decode_attn_backend = None
        decode_attn_backend_group = []

    # device-specific post-init. TODO: platform interface (separate PR).
    if (
        model_runner.device == "npu"
        and envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0
        and not model_runner.is_draft_worker
    ):
        # lazy init for zbal with mix mode (before graph capture when enable_cuda_graph)
        from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

        lazy_init_zbal_gva_mem(
            model_runner.device,
            model_runner.gpu_id,
            get_world_group().rank_in_group,
            get_world_group().world_size,
            get_world_group().cpu_group,
        )

    # Record resolved per-mode backends on the backend for model dispatch.
    attn_backend.prefill_attention_backend_str = resolved.prefill
    attn_backend.decode_attention_backend_str = resolved.decode

    return AttentionBackends(
        attn_backend=attn_backend,
        decode_attn_backend=decode_attn_backend,
        decode_attn_backend_group=decode_attn_backend_group,
        prefill_attention_backend_str=resolved.prefill,
        decode_attention_backend_str=resolved.decode,
    )


def get_attention_backend(
    *, model_runner: ModelRunner, init_new_workspace: bool = False
) -> AttentionBackend:
    resolved = _resolve_attention_backend_strs(
        server_args=model_runner.server_args,
        is_draft_worker=model_runner.is_draft_worker,
    )
    return _build_resolved_backend(
        model_runner=model_runner,
        resolved=resolved,
        init_new_workspace=init_new_workspace,
    )


def _resolve_attention_backend_strs(
    *, server_args: ServerArgs, is_draft_worker: bool
) -> ResolvedAttentionBackendStr:
    draft_attn_backend = server_args.speculative_draft_attention_backend
    if is_draft_worker and draft_attn_backend:
        logger.warning(f"Overriding draft attention backend to {draft_attn_backend}.")
        # Single backend for all draft modes (no prefill/decode split).
        return ResolvedAttentionBackendStr(
            prefill=draft_attn_backend, decode=draft_attn_backend
        )
    prefill, decode = server_args.get_attention_backends()
    return ResolvedAttentionBackendStr(prefill=prefill, decode=decode)


def _build_resolved_backend(
    *,
    model_runner: ModelRunner,
    resolved: ResolvedAttentionBackendStr,
    init_new_workspace: bool,
) -> AttentionBackend:
    if resolved.decode != resolved.prefill:
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        attn_backend = HybridAttnBackend(
            model_runner,
            decode_backend=_build_backend_from_str(
                model_runner=model_runner,
                backend_str=resolved.decode,
                init_new_workspace=init_new_workspace,
            ),
            prefill_backend=_build_backend_from_str(
                model_runner=model_runner,
                backend_str=resolved.prefill,
                init_new_workspace=init_new_workspace,
            ),
        )
        logger.info(
            f"Using hybrid attention backend for decode and prefill: "
            f"decode_backend={resolved.decode}, "
            f"prefill_backend={resolved.prefill}."
        )
        logger.warning(
            "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
            "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
        )
    else:
        # Non-hybrid uses the raw server_args.attention_backend (may be "auto"),
        # not resolved.prefill -- the two can differ under auto resolution.
        attn_backend = _build_backend_from_str(
            model_runner=model_runner,
            backend_str=model_runner.server_args.attention_backend,
            init_new_workspace=init_new_workspace,
        )
    return attn_backend


def _build_backend_from_str(
    *, model_runner: ModelRunner, backend_str: str, init_new_workspace: bool
) -> AttentionBackend:
    if backend_str not in ATTENTION_BACKENDS:
        raise ValueError(f"Invalid attention backend: {backend_str}")
    # The one unavoidable runner write (read-only-god-object rule exception): the
    # registry builds the backend as ATTENTION_BACKENDS[str](runner) and reads
    # init_new_workspace off the runner (attention_registry.py), so the single-arg
    # contract leaves no other way to pass it. Removing it means reworking the
    # registry / flashinfer / eagle save-restore -- a separate PR.
    model_runner.init_new_workspace = init_new_workspace
    full_attention_backend = ATTENTION_BACKENDS[backend_str](model_runner)
    return attn_backend_wrapper(model_runner, full_attention_backend)
