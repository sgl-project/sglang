"""Model-side AFD surface shared by every decoder-layer family.

A family contributes only what differs: which attention and MLP classes to
build, and how one layer's local compute is spelled. Role resolution, the step
delegation and the role-aware weight plan live here. Deliberately torch-free so
the no-GPU contract suite can load it.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.runtime_context import get_server_args

from .config import AFDExecutionMode, execution_mode_from_server_args


def afd_execution_mode() -> AFDExecutionMode:
    """Resolve the role from process-local runtime context, not a global."""

    return execution_mode_from_server_args(get_server_args())


def afd_model_forward(
    *,
    model: Any,
    pipeline: Any,
    input_ids: Any,
    positions: Any,
    forward_batch: Any,
    input_embeds: Any,
    pp_proxy_tensors: Any,
) -> Any:
    """Embed, run the stage-wise AFD pipeline, then apply the final norm."""

    if (
        not model.pp_group.is_first_rank
        or not model.pp_group.is_last_rank
        or pp_proxy_tensors is not None
    ):
        raise RuntimeError("AFD_MODEL_PIPELINE_PARALLEL_UNSUPPORTED")
    if forward_batch.forward_mode.is_draft_extend_v2():
        raise RuntimeError("AFD_MODEL_DRAFT_MUST_BE_LOCAL")
    if forward_batch.can_run_tbo:
        raise RuntimeError("AFD_MODEL_TBO_UNSUPPORTED")
    hidden_states = (
        model.embed_tokens(input_ids) if input_embeds is None else input_embeds
    )
    hidden_states, residual = pipeline.execute(
        hidden_states=hidden_states,
        residual=None,
        positions=positions,
        forward_batch=forward_batch,
    )
    if hidden_states.shape[0]:
        hidden_states = (
            model.norm(hidden_states)
            if residual is None
            else model.norm(hidden_states, residual)[0]
        )
    return hidden_states


def afd_stacked_params_mapping(
    *,
    attention: tuple,
    dense_mlp: tuple,
) -> tuple:
    """Return the stacked-weight plan for the weights this role owns."""

    mode = afd_execution_mode()
    if mode == AFDExecutionMode.FFN:
        return ()
    if mode == AFDExecutionMode.ATTENTION:
        return tuple(attention)
    return tuple(attention) + tuple(dense_mlp)


def afd_owns_expert_weights() -> bool:
    """The attention role never materializes experts."""

    return afd_execution_mode() != AFDExecutionMode.ATTENTION
