from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

from sglang.srt.configs.draft_parallel import (
    DraftParallelLayout,
    DraftParallelSpec,
    DraftTPExecution,
)
from sglang.srt.distributed.parallel_state import (
    get_attn_tp_group,
    get_tp_group,
    patch_tensor_parallel_group,
)
from sglang.srt.layers import dp_attention
from sglang.srt.layers.cp import base as cp_state
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig


def resolve_draft_parallel_layout(
    model_config: ModelConfig, *, enable_dp_attention: bool
) -> DraftParallelLayout:
    """Use the registered model's declaration, with the legacy DFlash default."""
    from sglang.srt.model_loader.utils import get_model_architecture

    model_cls, _ = get_model_architecture(model_config)
    spec = getattr(
        model_cls,
        "draft_parallel_spec",
        DraftParallelSpec(DraftTPExecution.SINGLE_GROUP),
    )
    owns_attention = spec.tp_execution is DraftTPExecution.SINGLE_GROUP
    group = (
        get_attn_tp_group()
        if owns_attention and enable_dp_attention
        else get_tp_group()
    )
    return DraftParallelLayout(
        tp_group=group,
        attn_tp_size=(
            group.world_size if owns_attention else get_attn_tp_group().world_size
        ),
        owns_attention=owns_attention,
    )


@contextmanager
def draft_parallel_context(
    layout: DraftParallelLayout, server_args: ServerArgs | None = None
):
    """Give a dense draft its own TP view; restore target state even on error.

    This fork passes no target DP token arrays into the dense draft forward.
    Target DP ranks and buffer metadata stay intact outside this scope.
    """
    names = (
        "_ENABLE_DP_ATTENTION_FLAG",
        "_ATTN_DP_RANK",
        "_ATTN_DP_SIZE",
        "_LOCAL_ATTN_DP_RANK",
        "_LOCAL_ATTN_DP_SIZE",
        "_DP_MAX_LEN_WITH_IDLE",
    )
    tp_context = (
        patch_tensor_parallel_group(layout.tp_group, owns_attention=True)
        if layout.owns_attention
        else nullcontext()
    )
    with tp_context:
        previous = {name: getattr(dp_attention, name) for name in names}
        previous_args = get_global_server_args() if server_args is not None else None
        previous_cp_strategy = cp_state._STRATEGY
        try:
            if server_args is not None:
                set_global_server_args_for_scheduler(server_args)
            if layout.owns_attention:
                cp_state._STRATEGY = None
                dp_attention._ENABLE_DP_ATTENTION_FLAG = False
                dp_attention._ATTN_DP_RANK = dp_attention._LOCAL_ATTN_DP_RANK = 0
                dp_attention._ATTN_DP_SIZE = dp_attention._LOCAL_ATTN_DP_SIZE = 1
                dp_attention._DP_MAX_LEN_WITH_IDLE = False
            with (
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                if layout.owns_attention:
                    # A dense draft has no expert dispatch, even when the target
                    # leaves its speculative A2A override unset under EP.
                    moe_utils.MOE_A2A_BACKEND = moe_utils.MoeA2ABackend.NONE
                yield
        finally:
            cp_state._STRATEGY = previous_cp_strategy
            if server_args is not None:
                set_global_server_args_for_scheduler(previous_args)
            for name, value in previous.items():
                setattr(dp_attention, name, value)
