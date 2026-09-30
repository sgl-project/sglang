"""Adaptive verify width for DFLASH: the draft always proposes its full block and
the target verifies a batch-size-dependent prefix of it."""

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Optional

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    with_phase,
)
from sglang.srt.runtime_context import get_context, get_exec, get_spec
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.utils.common import get_available_gpu_memory, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def build_dflash_verify_state(
    *,
    target_model_runner: "ModelRunner",
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    cuda_graph_bs: Optional[list[int]],
) -> SpecRuntimeState:
    """Target attention backend and verify graphs for one verify width.

    ``cuda_graph_bs`` lists the decode graph batch sizes that route to this width;
    ``None`` means decode CUDA graphs are disabled.
    """
    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(
        target_model_runner.device, target_model_runner.gpu_id
    )
    with _override_verify_width(
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        cuda_graph_bs=cuda_graph_bs,
    ):
        backup_init = target_model_runner.init_new_workspace
        try:
            attn_backend = target_model_runner._get_attention_backend(
                init_new_workspace=True
            )
        finally:
            target_model_runner.init_new_workspace = backup_init
        graph_runner = None
        if cuda_graph_bs and not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        ):
            graph_runner = target_model_runner._decode_cuda_graph_runner_cls()(
                target_model_runner,
                attn_backend=attn_backend,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )
    after_mem = get_available_gpu_memory(
        target_model_runner.device, target_model_runner.gpu_id
    )
    log_info_on_rank0(
        logger,
        f"Built DFLASH verify state num_draft_tokens={speculative_num_draft_tokens}: "
        f"cuda_graph_bs={cuda_graph_bs}, "
        f"elapsed={time.perf_counter() - tic:.2f}s, "
        f"mem={(before_mem - after_mem):.2f}GB",
    )
    return SpecRuntimeState(
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=attn_backend,
        target_graph_runner=graph_runner,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


@contextlib.contextmanager
def _override_verify_width(
    *, speculative_num_draft_tokens: int, cuda_graph_bs: Optional[list[int]]
):
    graph = get_exec().graph
    backup = (get_spec().speculative_num_draft_tokens, graph.cuda_graph_config)
    get_context().override(
        "adaptive_spec.capture_override",
        speculative_num_draft_tokens=speculative_num_draft_tokens,
    )
    if cuda_graph_bs:
        # Capture only the graph batch sizes that route to this width.
        get_context().override(
            "adaptive_spec.capture_override",
            cuda_graph_config=with_phase(
                graph.cuda_graph_config, Phase.DECODE, bs=cuda_graph_bs
            ),
        )
    try:
        yield
    finally:
        get_context().override(
            "adaptive_spec.capture_restore",
            speculative_num_draft_tokens=backup[0],
            cuda_graph_config=backup[1],
        )
