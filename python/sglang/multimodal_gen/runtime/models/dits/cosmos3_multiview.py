# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV transformer.

Same weights and layer stack as ``Cosmos3OmniTransformer``. The only change
inside the network is the GEN cross-attention: when the pipeline hands over a
``MultiviewLayout``, each layer runs the block-sparse multiview attention from
``cosmos3_multiview_attention`` instead of dense attention over ``[UND | GEN]``.
The transformer also owns the request-local mask and packing-buffer caches so
36 layers and every denoising step reuse one block map and one set of padded
q/k/v buffers.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.cosmos3video import Cosmos3VideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_BACKBONE_TYPE,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_attention import (
    MultiviewAttentionContext,
    MultiviewLayout,
    padded_multiview_flex_attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3CrossAttention,
    Cosmos3OmniTransformer,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Cosmos3MultiviewCrossAttention(Cosmos3CrossAttention):
    """GEN cross-attention that runs the sparse multiview kernel for a layout."""

    def _forward_multiview(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        multiview_layout: Any,
    ) -> torch.Tensor:
        if not isinstance(multiview_layout, MultiviewAttentionContext):
            raise TypeError(
                "Cosmos3 multiview cross-attention expected MultiviewAttentionContext, "
                f"got {type(multiview_layout).__name__}."
            )
        return padded_multiview_flex_attention(q, k, v, k_und, v_und, multiview_layout)


class Cosmos3MultiviewTransformer(Cosmos3OmniTransformer):
    """Cosmos3 Nano weights with request-local multiview block-mask caching."""

    _cross_attention_cls = Cosmos3MultiviewCrossAttention

    def __init__(
        self,
        config: Cosmos3VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        backbone_type = (
            hf_config.get("backbone_type") if isinstance(hf_config, dict) else None
        )
        if backbone_type != COSMOS3_MULTIVIEW_BACKBONE_TYPE:
            raise ValueError(
                "Cosmos3MultiviewTransformer requires transformer/config.json "
                f"backbone_type={COSMOS3_MULTIVIEW_BACKBONE_TYPE!r}, got {backbone_type!r}."
            )
        super().__init__(config, hf_config, quant_config)
        self._multiview_mask_cache: dict[tuple[Any, ...], Any] = {}
        # Padded q/k/v packing buffers, keyed by shape/dtype/device. Held on
        # the transformer rather than a per-forward context so the packed
        # tensors are allocated and zeroed once per request, not once per layer.
        self._multiview_buffer_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def reset_cache(self, cache_key: str | None = None) -> None:
        super().reset_cache(cache_key)
        if cache_key is None:
            self._multiview_mask_cache.clear()
            self._multiview_buffer_cache.clear()

    def forward(
        self,
        *args,
        multiview_layout: MultiviewLayout | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if multiview_layout is None:
            return super().forward(*args, **kwargs)
        control_latents = kwargs.get("control_latents")
        if isinstance(control_latents, torch.Tensor):
            control_count = 1
        elif control_latents is None:
            control_count = 0
        else:
            control_count = len(control_latents)
        if control_count != 1:
            raise ValueError(
                "Cosmos3 multiview v1 requires exactly one packed WSM control item, "
                f"got {control_count}."
            )
        if (
            kwargs.get("action_latents") is not None
            or kwargs.get("sound_latents") is not None
        ):
            raise ValueError(
                "Cosmos3 multiview v1 cannot be combined with action or sound streams."
            )
        if get_sp_world_size() > 1:
            raise ValueError(
                "Cosmos3 multiview v1 does not support sequence parallelism."
            )
        context = MultiviewAttentionContext(
            multiview_layout,
            self._multiview_mask_cache,
            self._multiview_buffer_cache,
        )
        return super().forward(*args, multiview_layout=context, **kwargs)


EntryClass = Cosmos3MultiviewTransformer
