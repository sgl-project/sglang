from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.configs.model_config import ModelConfig

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class SpecAuxHiddenStateConfig(msgspec.Struct, kw_only=True):
    eagle_use_aux_hidden_state: bool = False
    eagle_draft_num_layers: Optional[int] = None
    eagle_aux_hidden_state_layer_ids: Any = None
    dflash_use_aux_hidden_state: bool = False
    dflash_draft_num_layers: Optional[int] = None
    dflash_target_layer_ids: Any = None
    # Actual bytes/token of the DFLASH draft KV pool, used by the target's
    # pool configurator to reserve draft-pool memory accurately. None falls
    # back to the layer-ratio heuristic (fine when draft and target share
    # per-layer KV cost, e.g. EAGLE-style drafters). The raw KV-shape fields
    # below feed the lazy computation (the parallel state is not initialized
    # when this config is built, so the tp-dependent cell size is computed
    # later in the pool configurator).
    dflash_draft_total_num_kv_heads: Optional[int] = None
    dflash_draft_head_dim: Optional[int] = None
    dflash_draft_v_head_dim: Optional[int] = None
    dflash_draft_kv_element_size: Optional[int] = None


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
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> None:
    if (
        (spec_algorithm.is_eagle() or spec_algorithm.is_standalone())
        and not is_draft_worker
        and server_args.speculative_draft_model_path
    ):
        # Load draft config to get layer count for KV cache sizing
        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.speculative_draft_model_path,
            model_revision=server_args.speculative_draft_model_revision,
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
            model_path=(server_args.speculative_draft_model_path),
            model_revision=server_args.speculative_draft_model_revision,
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
        _populate_dflash_draft_kv_shape(
            config=config,
            server_args=server_args,
            draft_model_config=draft_model_config,
        )


def _populate_dflash_draft_kv_shape(
    *,
    config: SpecAuxHiddenStateConfig,
    server_args: ServerArgs,
    draft_model_config: ModelConfig,
) -> None:
    """Store the draft's raw KV shape so the pool configurator can compute the
    draft-pool bytes/token lazily (once the parallel state is initialized).

    None on any field falls back to the layer-ratio heuristic.
    """
    try:
        from sglang.srt.speculative.dflash_utils import (
            resolve_dflash_draft_kv_element_size,
        )

        config.dflash_draft_total_num_kv_heads = int(
            draft_model_config.get_total_num_kv_heads()
        )
        config.dflash_draft_head_dim = int(draft_model_config.head_dim)
        config.dflash_draft_v_head_dim = int(draft_model_config.v_head_dim)
        config.dflash_draft_kv_element_size = resolve_dflash_draft_kv_element_size(
            draft_model_dtype=draft_model_config.dtype,
            server_args_kv_cache_dtype=server_args.kv_cache_dtype,
            speculative_draft_attention_backend=(
                server_args.speculative_draft_attention_backend
            ),
            speculative_draft_kv_cache_dtype=(
                server_args.speculative_draft_kv_cache_dtype
            ),
        )
    except Exception:
        logger.warning(
            "Could not resolve DFLASH draft KV shape; falling back to the "
            "layer-ratio draft-pool memory reservation.",
            exc_info=True,
        )
