from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
    get_spec,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

_MUSE_LAYER_OUTPUT_DRAFT_ARCHITECTURES = frozenset(
    {"DFlash2DraftModel", "MuseGlimmerAssistantModel"}
)


def _map_muse_target_layer_ids(*, target_hf_config, draft_hf_config, layer_ids):
    architectures = getattr(draft_hf_config, "architectures", None) or []
    uses_layer_outputs = getattr(
        target_hf_config, "model_type", None
    ) == "muse_glimmer" and bool(
        _MUSE_LAYER_OUTPUT_DRAFT_ARCHITECTURES.intersection(architectures)
    )
    return [i + 1 for i in layer_ids] if uses_layer_outputs else layer_ids


class SpecAuxHiddenStateConfig(msgspec.Struct, kw_only=True):
    eagle_use_aux_hidden_state: bool = False
    eagle_draft_num_layers: Optional[int] = None
    # Draft layers whose KV cache uses the target SWA pool capacity.
    eagle_draft_swa_num_layers: Optional[int] = None
    eagle_aux_hidden_state_layer_ids: Any = None
    # EAGLE draft KV head geometry, recorded so the unified fused-draft-KV
    # path can build the draft region and price the fused entry at TARGET boot
    # — before the draft worker exists. `total_kv_heads` is the UNDIVIDED
    # count: this config resolves before the attention-TP group is
    # initialized, so consumers apply attn_tp themselves. None when no EAGLE
    # draft config was loaded.
    eagle_draft_total_kv_heads: Optional[int] = None
    eagle_draft_head_dim: Optional[int] = None
    eagle_draft_v_head_dim: Optional[int] = None
    dflash_use_aux_hidden_state: bool = False
    dflash_draft_num_layers: Optional[int] = None
    dflash_target_layer_ids: Any = None
    # DFLASH draft KV bytes/token; None when unresolved.
    dflash_draft_cell_size_per_token: int | None = None


def resolve_spec_aux_hidden_state_config(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> SpecAuxHiddenStateConfig:
    config = SpecAuxHiddenStateConfig()
    _resolve_eagle_aux_hidden_state(
        config=config,
        server_args=server_args,
        model_config=model_config,
        spec_algorithm=spec_algorithm,
        is_draft_worker=is_draft_worker,
    )
    _resolve_dflash_aux_hidden_state(
        config=config,
        server_args=server_args,
        model_config=model_config,
        spec_algorithm=spec_algorithm,
        is_draft_worker=is_draft_worker,
    )
    return config


def _resolve_eagle_aux_hidden_state(
    *,
    config: SpecAuxHiddenStateConfig,
    server_args: ServerArgs,
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> None:
    if (spec_algorithm.is_eagle() or spec_algorithm.is_standalone()) and (
        not is_draft_worker
    ):
        # Load draft config to get layer count for KV cache sizing.
        # A path-less NEXTN run (the MTP head ships INSIDE the target
        # checkpoint) is the same code path: `from_server_args` falls back to
        # the target path when model_path is None, and `is_draft_model=True`
        # is what makes ModelConfig fill in `num_nextn_predict_layers` at all
        # -- every assignment of that field is guarded by `is_draft_model`, so
        # reading it off the TARGET's own config always answers None and the
        # geometry silently never resolves (eval_568: Qwen NEXTN fell back to
        # a private draft pool instead of fusing).
        draft_path = get_spec().speculative_draft_model_path
        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=draft_path,
            model_revision=get_spec().speculative_draft_model_revision,
            is_draft_model=True,
        )
        num_nextn_predict_layers = draft_model_config.num_nextn_predict_layers
        if num_nextn_predict_layers is not None:
            config.eagle_draft_num_layers = int(num_nextn_predict_layers)
        elif draft_path is None:
            # No draft path AND no MTP head: there is no draft geometry to
            # record. Leave the aux unset so the pool falls back to a private
            # draft pool rather than fusing a region sized like the whole
            # target.
            return
        else:
            config.eagle_draft_num_layers = int(
                max(
                    draft_model_config.num_hidden_layers,
                    draft_model_config.num_attention_layers,
                )
            )
        # TOTAL kv heads, NOT the per-GPU count: this resolver runs before
        # `init_torch_distributed`, so the attention-TP group does not exist
        # yet and any `get_parallel()` read asserts. Consumers divide by
        # attn_tp themselves, at pool-build time (see
        # KVCacheConfigurator.fused_draft_kv_region).
        config.eagle_draft_total_kv_heads = int(
            draft_model_config.get_total_num_kv_heads()
        )
        config.eagle_draft_head_dim = int(draft_model_config.head_dim)
        config.eagle_draft_v_head_dim = int(draft_model_config.v_head_dim)

        if (
            draft_model_config.is_hybrid_swa
            and not draft_model_config.is_deepseek_v4_arch
        ):
            config.eagle_draft_swa_num_layers = len(
                draft_model_config.swa_attention_layer_ids
            )

        if spec_algorithm.is_eagle3():
            config.eagle_use_aux_hidden_state = True
            try:
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                config.eagle_use_aux_hidden_state = eagle_config.get(
                    "use_aux_hidden_state", True
                )
                config.eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                config.eagle_aux_hidden_state_layer_ids = None


def _resolve_dflash_aux_hidden_state(
    *,
    config: SpecAuxHiddenStateConfig,
    server_args: ServerArgs,
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> None:
    if spec_algorithm.is_dflash_family() and not is_draft_worker:
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

        # Select target layers to capture for building draft context features.
        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(get_spec().speculative_draft_model_path),
            model_revision=get_spec().speculative_draft_model_revision,
            is_draft_model=True,
        )
        dflash_draft_config = parse_dflash_draft_config(
            draft_hf_config=draft_model_config.hf_config
        )
        draft_num_layers = dflash_draft_config.require_num_layers()
        trained_target_layers = dflash_draft_config.num_target_layers

        target_num_layers = getattr(
            model_config.hf_text_config, "num_hidden_layers", None
        )
        if target_num_layers is None:
            raise ValueError(
                "Block-draft-with-target-kv spec requires target num_hidden_layers "
                f"in config. Got target={target_num_layers}."
            )
        target_num_layers = int(target_num_layers)

        if (
            trained_target_layers is not None
            and trained_target_layers != target_num_layers
        ):
            logger.warning(
                "Draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                "selecting capture layers based on the runtime target model.",
                trained_target_layers,
                target_num_layers,
            )

        target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
            target_num_layers=int(target_num_layers),
            draft_num_layers=int(draft_num_layers),
        )

        # These Muse drafts use HF layer-output ids, while the Muse target captures
        # before each layer. Legacy Muse drafts already store layer-input ids.
        target_layer_ids = _map_muse_target_layer_ids(
            target_hf_config=model_config.hf_config,
            draft_hf_config=draft_model_config.hf_config,
            layer_ids=target_layer_ids,
        )

        if spec_algorithm.is_dspark():
            from sglang.srt.speculative.dspark_components.dspark_config import (
                parse_dspark_draft_config,
            )

            dspark_draft_config = parse_dspark_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            if not dspark_draft_config.require_markov():
                raise ValueError(
                    "DSPARK requires markov_rank > 0 in the draft config, "
                    f"got markov_rank={dspark_draft_config.markov_rank}."
                )
            if dspark_draft_config.target_layer_ids is not None:
                target_layer_ids = list(dspark_draft_config.target_layer_ids)

        config.dflash_use_aux_hidden_state = True
        config.dflash_draft_num_layers = int(draft_num_layers)
        config.dflash_target_layer_ids = target_layer_ids
        config.dflash_draft_cell_size_per_token = _resolve_dflash_draft_cell_size(
            draft_model_config=draft_model_config,
            draft_num_layers=int(draft_num_layers),
        )


def _resolve_dflash_draft_cell_size(
    *,
    draft_model_config: ModelConfig,
    draft_num_layers: int,
) -> int | None:
    """Bytes/token the DFLASH draft KV pool will cost the target's pool budget.

    Resolved from the draft's own attention geometry and the KV dtype the draft
    worker will actually resolve. Returns None if anything is unresolvable,
    leaving callers on layer-count scaling.
    """
    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
    from sglang.srt.speculative.dflash_utils import dflash_draft_cell_size_per_token

    try:
        _, draft_kv_cache_dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype=get_model().kv_cache_dtype,
            speculative_draft_kv_cache_dtype=(
                get_spec().speculative_draft_kv_cache_dtype
            ),
            model=None,
            model_dtype=draft_model_config.dtype,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend=(
                get_spec().speculative_draft_attention_backend
            ),
        )
        return dflash_draft_cell_size_per_token(
            draft_model_config=draft_model_config,
            draft_num_layers=draft_num_layers,
            draft_kv_cache_dtype=draft_kv_cache_dtype,
            tp_size=get_parallel().tp_size,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not resolve DFLASH draft KV bytes/token (%s); falling back to "
            "layer-count scaling for the KV pool budget.",
            e,
        )
        return None
