from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec

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


class ResolvedAttentionBackendStr(msgspec.Struct, frozen=True, kw_only=True):
    prefill: str
    decode: str
    is_draft_override: bool = False


class AttentionBackends(msgspec.Struct, frozen=True, kw_only=True):
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
    is_dspark: bool,
) -> None:
    """Configure auxiliary hidden state capture for speculative decoding.

    Must be called before CUDA graph capture so the captured graphs
    include aux hidden state output paths.
    """
    if eagle_use_aux_hidden_state:
        model.set_eagle3_layers_to_capture(eagle_aux_hidden_state_layer_ids)
    if dflash_use_aux_hidden_state:
        if is_dspark and hasattr(model, "set_dspark_layers_to_capture"):
            model.set_dspark_layers_to_capture(dflash_target_layer_ids)
        elif hasattr(model, "set_dflash_layers_to_capture"):
            model.set_dflash_layers_to_capture(dflash_target_layer_ids)
        else:
            raise ValueError(
                f"Model {model.__class__.__name__} implements neither "
                "set_dspark_layers_to_capture nor set_dflash_layers_to_capture, "
                "one of which is required for DFLASH/DSPARK."
            )


def build_attention_backends(*, model_runner: ModelRunner) -> AttentionBackends:
    """Init attention kernel backend."""
    server_args = model_runner.server_args

    # TODO: Refactor device-specific init branches into platform interface (separate PR).
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
                model_runner=model_runner,
                resolved=resolved,
                init_new_workspace=False,
            )
            for _ in range(server_args.sm_group_num)
        ]
        decode_attn_backend = decode_attn_backend_group[0]
    elif server_args.enable_two_batch_overlap and not model_runner.is_draft_worker:
        attn_backend = TboAttnBackend.init_new(
            lambda: _build_resolved_backend(
                model_runner=model_runner,
                resolved=resolved,
                init_new_workspace=False,
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
    """Init attention kernel backend."""
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
            prefill=draft_attn_backend,
            decode=draft_attn_backend,
            is_draft_override=True,
        )
    prefill, decode = server_args.get_attention_backends()
    return ResolvedAttentionBackendStr(prefill=prefill, decode=decode)


def _build_resolved_backend(
    *,
    model_runner: ModelRunner,
    resolved: ResolvedAttentionBackendStr,
    init_new_workspace: bool,
) -> AttentionBackend:
    if resolved.is_draft_override:
        attn_backend = _build_backend_from_str(
            model_runner=model_runner,
            backend_str=resolved.prefill,
            init_new_workspace=init_new_workspace,
        )
    elif resolved.decode != resolved.prefill:
        from sglang.srt.layers.attention.hybrid_attn_backend import (
            HybridAttnBackend,
        )

        # Compose the two full-attention backends first, then apply model-level
        # wrappers once.  Wrapping each child independently duplicates the
        # linear/sparse side backend for hybrid models (for example, two GDN
        # dispatchers for Qwen3.5 when prefill and decode use different MHA
        # backends), duplicating initialization and associated state while only
        # one side backend can be active in a forward pass.
        attn_backend = attn_backend_wrapper(
            model_runner,
            HybridAttnBackend(
                model_runner=model_runner,
                decode_backend=_build_full_attention_backend_from_str(
                    model_runner=model_runner,
                    backend_str=resolved.decode,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=_build_full_attention_backend_from_str(
                    model_runner=model_runner,
                    backend_str=resolved.prefill,
                    init_new_workspace=init_new_workspace,
                ),
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
        attn_backend = _build_backend_from_str(
            model_runner=model_runner,
            backend_str=model_runner.server_args.attention_backend,
            init_new_workspace=init_new_workspace,
        )
    return attn_backend


def _build_backend_from_str(
    *, model_runner: ModelRunner, backend_str: str, init_new_workspace: bool
) -> AttentionBackend:
    return attn_backend_wrapper(
        model_runner,
        _build_full_attention_backend_from_str(
            model_runner=model_runner,
            backend_str=backend_str,
            init_new_workspace=init_new_workspace,
        ),
    )


def _build_full_attention_backend_from_str(
    *, model_runner: ModelRunner, backend_str: str, init_new_workspace: bool
) -> AttentionBackend:
    if backend_str not in ATTENTION_BACKENDS:
        raise ValueError(f"Invalid attention backend: {backend_str}")
    model_runner.init_new_workspace = init_new_workspace
    return ATTENTION_BACKENDS[backend_str](model_runner)
