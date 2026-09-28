"""GLM-5.2 (glm_moe_dsa) family identity, topk carry and stage allocators."""

from __future__ import annotations

from typing import Any

from ..contracts import AFDError, AFDRole
from ..metadata import DSAMetadataGuard
from .base import AFDDecoderAdapter, AFDStage


def matches_glm_moe_dsa(model: Any) -> bool:
    """Return whether a model is a DSA GLM checkpoint with a usable topk plan.

    Unlike the Qwen entry this does not pin a layer count, because the same
    decoder shape is served by the production checkpoint and by the reduced
    configs used to exercise the path.
    """

    try:
        config = model.model.config
        index_topk = config.index_topk
        pattern = getattr(config, "index_topk_pattern", None)
        freq = getattr(config, "index_topk_freq", 1)
    except (AttributeError, TypeError):
        return False
    if type(model).__name__ != "GlmMoeDsaForCausalLM":
        return False
    if not isinstance(index_topk, int) or index_topk < 1:
        return False
    if pattern is not None:
        return all(item in ("S", "I") for item in pattern)
    return freq is None or (isinstance(freq, int) and freq > 0)


class _StageState:
    """Address-stable per-stage scratch for one DSA traversal of all layers.

    A model-side bump allocator is sized for exactly one traversal, and AFD
    traverses every layer once per stage, so each stage owns its own storage.
    The storage is allocated once and rewound per step rather than rebuilt,
    because captured role graphs hold its address.
    """

    def __init__(self, *, num_layers: int, device: Any) -> None:
        import torch

        from sglang.srt.utils.common import BumpAllocator

        self.zero_allocator = BumpAllocator(
            buffer_size=num_layers * 2,
            dtype=torch.float32,
            device=device,
        )
        self.topk_indices: Any = None
        self.topk_step: int = -1

    def begin_step(self) -> None:
        self.zero_allocator.reset()


class Glm5AFDAdapter(AFDDecoderAdapter):
    """GLM-5.2 adapter: DSA metadata, a shared-topk carry, stage allocators.

    GLM-5.2 shares one indexer's top-k across a run of DSA layers, and the AFD
    loop unpacks only (hidden, residual), so the indices ride caller-owned
    per-stage state instead of a third return value. Stages hold disjoint
    requests, so the carry cannot be global.

    Within a whole-role graph, the owner and sharing layers capture the same
    stage-local storage. Eager steps stamp this carry so a sharing layer cannot
    consume indices from a previous step; a failed role replay aborts the step.
    """

    guard_class = DSAMetadataGuard
    family_error = "AFD_GLM_MOE_DSA_REQUIRED"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stage_states: dict[int, _StageState] = {}
        self._step_id = 0

    def matches_family(self) -> bool:
        return matches_glm_moe_dsa(self.model)

    def split_step(self, **kwargs: Any) -> list[AFDStage]:
        self._step_id += 1
        device = kwargs["hidden_states"].device
        for index in range(kwargs["stages"]):
            state = self._stage_states.get(index)
            if state is None:
                # Allocated here, never in local_compute: local_compute runs
                # inside graph capture, where a fresh buffer would land in the
                # graph pool and its zero-fill would become a captured kernel.
                state = _StageState(num_layers=self.num_layers, device=device)
                self._stage_states[index] = state
            state.begin_step()
        return super().split_step(**kwargs)

    def local_compute(
        self,
        *,
        layer: int,
        stage: AFDStage,
        hidden_states: Any,
        residual: Any,
        positions: Any = None,
    ) -> tuple[Any, ...]:
        if self.role != AFDRole.ATTENTION:
            return super().local_compute(
                layer=layer,
                stage=stage,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
            )
        state = self._stage_states.get(stage.index)
        if state is None:
            raise AFDError(
                "AFD_GLM_DSA_STAGE_STATE_MISSING",
                f"stage={stage.index}",
            )
        decoder_layer = self.inner.layers[layer]
        attention = decoder_layer.self_attn
        consumes = bool(getattr(attention, "skip_topk", None))
        produces = bool(getattr(attention, "next_skip_topk", None))
        if consumes and state.topk_step != self._step_id:
            raise AFDError(
                "AFD_GLM_DSA_TOPK_CARRY_STALE",
                f"layer={layer} stage={stage.index} "
                f"carry_step={state.topk_step} step={self._step_id}",
            )
        hidden, residual, produced_topk = decoder_layer.forward_attention_for_afd(
            positions=stage.positions if positions is None else positions,
            hidden_states=hidden_states,
            forward_batch=stage.forward_batch,
            residual=residual,
            zero_allocator=state.zero_allocator,
            prev_topk_indices=state.topk_indices if consumes else None,
        )
        if consumes or produces:
            state.topk_indices = produced_topk
            state.topk_step = self._step_id
        return hidden, residual
