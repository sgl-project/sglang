from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.perturb.config import PerturbConfig
from sglang.srt.kv_canary.pool_patcher.api import attach_canary_buffers
from sglang.srt.kv_canary.pool_patcher.utils import wrap_method
from sglang.srt.kv_canary.runner.canary_manager import CanaryManager
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def install_canary(
    *,
    server_args: ServerArgs,
    model_runner: ModelRunner,
    token_oracle_manager: Optional[TokenOracleManager] = None,
) -> Optional[CanaryManager]:
    config = CanaryConfig.from_env(server_args)
    if config.mode is CanaryMode.NONE:
        return None

    assert not check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE), (
        "kv-canary: piecewise cuda graph is not supported by the current "
        "SingleForwardManager design; set --cuda-graph-backend-prefill=disabled "
        "(or =breakable) when canary is enabled"
    )

    perturb_config = PerturbConfig.from_env()
    device = torch.device(model_runner.device)
    # EAGLE draft worker pools rotate input_ids so slot ``p`` stores K/V for the token at position ``p+1``;
    # target pools have no such shift. Threaded into the plan-side expected-token gather kernel.
    kv_token_id_vs_position_offset = 1 if model_runner.is_draft_worker else 0
    buffer_groups = attach_canary_buffers(
        pool=model_runner.token_to_kv_pool,
        config=config,
        device=device,
        kv_token_id_vs_position_offset=kv_token_id_vs_position_offset,
    )
    allocator = model_runner.token_to_kv_pool_allocator
    swa_allocator = (
        allocator if isinstance(allocator, SWATokenToKVPoolAllocator) else None
    )
    launch_capacities = CanaryLaunchCapacities.from_args(
        server_args=model_runner.server_args,
        req_to_token_pool_size=model_runner.req_to_token_pool.size,
        max_seq_len_per_req=model_runner.req_to_token_pool.req_to_token.shape[1],
        pool_slot_count=model_runner.max_total_num_tokens,
    )
    swa_window_size = model_runner.sliding_window_size or 0
    speculative_num_steps = int(server_args.speculative_num_steps or 1)
    manager = CanaryManager(
        config=config,
        perturb_config=perturb_config,
        buffer_groups=buffer_groups,
        device=device,
        req_to_token_pool=model_runner.req_to_token_pool,
        launch_capacities=launch_capacities,
        swa_window_size=swa_window_size,
        token_oracle_manager=token_oracle_manager,
        swa_allocator=swa_allocator,
        speculative_num_steps=speculative_num_steps,
        is_eagle_draft_decode=model_runner.is_draft_worker,
    )

    _patch_model_forward(model_runner=model_runner, manager=manager)

    # Single-line summary of every knob that controls canary behavior at boot time.
    # Disaggregation mode is included so PD logs are unambiguous about which side this is.
    logger.info(
        "install_canary: disaggregation_mode=%s config=%s perturb_config=%s "
        "launch_capacities=%s n_buffer_groups=%d buffer_group_kinds=%s "
        "swa_window_size=%d speculative_num_steps=%d",
        server_args.disaggregation_mode,
        config,
        perturb_config,
        launch_capacities,
        len(buffer_groups),
        [g.kind.name for g in buffer_groups],
        swa_window_size,
        speculative_num_steps,
    )
    return manager


def _patch_model_forward(*, model_runner: ModelRunner, manager: CanaryManager) -> None:
    def _with_canary_bracketing(original: Callable, *args: Any, **kwargs: Any) -> Any:
        with manager.model_forward_bracket_scope() as should_bracket:
            if not should_bracket:
                # Nested model.forward calls share the active SingleForwardManager.
                # Only the outermost call may run kv-canary pre/post ops; otherwise
                # the phase checker sees a second pre-op before the first post-op.
                return original(*args, **kwargs)

            forward_batch = _extract_forward_batch(args, kwargs)
            assert (
                forward_batch is not None
            ), "kv-canary: patched model.forward called without a ForwardBatch"

            canary_pre_ops_output = manager.pre_ops_maybe_inside_graph(forward_batch)
            output = original(*args, **kwargs)
            manager.post_ops_maybe_inside_graph(forward_batch, canary_pre_ops_output)
            return output

    wrap_method(model_runner.model, "forward", wrapper=_with_canary_bracketing)


def _extract_forward_batch(args, kwargs) -> Optional[ForwardBatch]:
    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
