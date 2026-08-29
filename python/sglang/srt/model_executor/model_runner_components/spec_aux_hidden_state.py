from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.runtime_context import (
    get_disagg,
    get_model,
    get_parallel,
    get_schedule,
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
    dflash_use_aux_hidden_state: bool = False
    dflash_draft_num_layers: Optional[int] = None
    dflash_target_layer_ids: Any = None
    # DFLASH draft KV bytes/token; None when unresolved.
    dflash_draft_cell_size_per_token: int | None = None
    # Fixed per-rank draft KV allocation, including the pool sentinel page.
    dflash_draft_fixed_bytes: int = 0


def resolve_spec_aux_hidden_state_config(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
    attn_dp_size: int,
) -> SpecAuxHiddenStateConfig:
    config = SpecAuxHiddenStateConfig()
    _resolve_eagle_aux_hidden_state(
        config=config,
        server_args=server_args,
        spec_algorithm=spec_algorithm,
        is_draft_worker=is_draft_worker,
    )
    _resolve_dflash_aux_hidden_state(
        config=config,
        server_args=server_args,
        model_config=model_config,
        spec_algorithm=spec_algorithm,
        is_draft_worker=is_draft_worker,
        attn_dp_size=attn_dp_size,
    )
    return config


def _resolve_eagle_aux_hidden_state(
    *,
    config: SpecAuxHiddenStateConfig,
    server_args: ServerArgs,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> None:
    if (
        (spec_algorithm.is_eagle() or spec_algorithm.is_standalone())
        and not is_draft_worker
        and get_spec().speculative_draft_model_path
    ):
        # Load draft config to get layer count for KV cache sizing
        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=get_spec().speculative_draft_model_path,
            model_revision=get_spec().speculative_draft_model_revision,
            is_draft_model=True,
        )
        num_nextn_predict_layers = draft_model_config.num_nextn_predict_layers
        if num_nextn_predict_layers is not None:
            config.eagle_draft_num_layers = int(num_nextn_predict_layers)
        else:
            config.eagle_draft_num_layers = int(
                max(
                    draft_model_config.num_hidden_layers,
                    draft_model_config.num_attention_layers,
                )
            )

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
    attn_dp_size: int,
) -> None:
    if spec_algorithm.is_dflash_family() and not is_draft_worker:
        if _compact_dflash_enabled() and not spec_algorithm.is_dflash():
            raise RuntimeError(
                "Compact DFlash cache supports the DFLASH algorithm only, "
                f"got speculative_algorithm={spec_algorithm}"
            )
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
        legacy_draft_cell_size = _resolve_dflash_draft_cell_size(
            draft_model_config=draft_model_config,
            draft_num_layers=int(draft_num_layers),
        )
        config.dflash_draft_cell_size_per_token = legacy_draft_cell_size
        if _compact_dflash_enabled():
            config.dflash_draft_cell_size_per_token = _compact_dflash_linear_budget(
                legacy_draft_cell_size
            )
            config.dflash_draft_fixed_bytes = _compact_dflash_fixed_bytes(
                legacy_draft_cell_size,
                owner_count=_compact_dflash_owner_count(attn_dp_size=attn_dp_size),
                window_size=get_spec().speculative_draft_window_size,
                block_size=get_spec().speculative_num_draft_tokens,
                page_size=get_schedule().page_size,
            )


def _compact_dflash_owner_count(*, attn_dp_size: int) -> int:
    """Return request-owner rows for this precomputed attention-DP shard."""
    requested = get_schedule().max_running_requests
    if requested is None:
        raise RuntimeError(
            "Compact DFlash physical mode requires an explicit resolved "
            "max_running_requests"
        )
    requested = int(requested)
    attn_dp_size = int(attn_dp_size)
    if requested <= 0 or attn_dp_size <= 0 or requested % attn_dp_size:
        raise RuntimeError(
            "Compact DFlash owner budget requires max_running_requests to be "
            "evenly sharded by attention DP: "
            f"max_running_requests={requested}, attn_dp_size={attn_dp_size}"
        )
    owner_count = requested // attn_dp_size
    if get_disagg().disaggregation_mode == "decode":
        extra_slots = int(get_disagg().disaggregation_decode_extra_slots or 0)
        if extra_slots < 0:
            raise RuntimeError(
                "Compact DFlash owner budget requires non-negative decode "
                f"extra slots, got disaggregation_decode_extra_slots={extra_slots}"
            )
        owner_count += extra_slots
    return owner_count


def _compact_dflash_linear_budget(legacy_bytes_per_token: int | None) -> int | None:
    """Replace the legacy capacity-linear term only in compact mode."""
    return 0 if _compact_dflash_enabled() else legacy_bytes_per_token


def _compact_dflash_enabled() -> bool:
    return bool(getattr(get_spec(), "speculative_dflash_compact_cache", False))


def _compact_dflash_fixed_bytes(
    legacy_bytes_per_token: int | None,
    *,
    owner_count: int | None,
    window_size: int | None,
    block_size: int | None,
    page_size: int,
) -> int:
    """Exact per-rank bytes allocated by the compact draft pool."""
    if not _compact_dflash_enabled():
        return 0
    values = {
        "legacy_bytes_per_token": legacy_bytes_per_token,
        "owner_count": owner_count,
        "window_size": window_size,
        "block_size": block_size,
        "page_size": page_size,
    }
    if any(value is None or int(value) <= 0 for value in values.values()):
        raise RuntimeError(
            "Compact DFlash fixed-pool budget requires resolved positive geometry: "
            f"{values}"
        )
    if int(page_size) != 1:
        raise RuntimeError(
            "Compact DFlash physical mode requires page_size=1; larger pages "
            "can make one attention window address aliased modulo-ring rows"
        )
    from sglang.srt.speculative.dflash_compact_physical_layout import (
        CompactDFlashPhysicalLayout,
    )

    layout = CompactDFlashPhysicalLayout.build(
        owner_count=int(owner_count),
        window_size=int(window_size),
        block_size=int(block_size),
        page_size=int(page_size),
    )
    # TokenToKVPool allocates one sentinel page in addition to usable rows.
    return (layout.physical_tokens + int(page_size)) * int(legacy_bytes_per_token)


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
