from __future__ import annotations

import threading
from dataclasses import dataclass, field, fields
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    _is_stream_capturing,
)
from sglang.srt.utils import is_sm120_supported, is_sm121

MAX_LOCAL_EXPERTS = 512
NVFP4_MOE_SM120_MAX_TOKENS = 16


@dataclass
class Nvfp4MoeWorkspace:
    """Preallocated scratch shared by one SGLang MoE layer.

    Eager launches are ordered across caller streams. CUDA graphs that capture
    this workspace must also replay in stream order; SGLang's graph runner uses
    one capture stream and replays one selected graph inline per model step.
    """

    x_q: torch.Tensor
    x_scale: torch.Tensor
    fc1: torch.Tensor
    fc1_split: torch.Tensor
    act_q: torch.Tensor
    act_scale: torch.Tensor
    fc2: torch.Tensor
    pair_experts: torch.Tensor
    group_rows: torch.Tensor
    group_pairs: torch.Tensor
    expert_counts: torch.Tensor
    group_experts: torch.Tensor
    group_offsets: torch.Tensor
    num_groups: torch.Tensor
    max_tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    graph_capture_supported: Optional[bool] = None
    _completion_event: Optional[torch.cuda.Event] = field(
        init=False, default=None, repr=False
    )
    _last_stream: Optional[int] = field(init=False, default=None, repr=False)
    _launch_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._launch_lock = threading.Lock()

    @classmethod
    def allocate(
        cls,
        *,
        max_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
        graph_capture_supported: Optional[bool] = None,
    ) -> Nvfp4MoeWorkspace:
        if max_tokens <= 0 or max_tokens > NVFP4_MOE_SM120_MAX_TOKENS:
            raise ValueError(
                "max_tokens must be in "
                f"[1, {NVFP4_MOE_SM120_MAX_TOKENS}], got {max_tokens}"
            )
        if hidden_size % 256 or intermediate_size % 64:
            raise ValueError(
                "hidden_size must be divisible by 256 and intermediate_size by 64"
            )
        pairs = max_tokens * top_k
        return cls(
            x_q=torch.empty(
                max_tokens, hidden_size // 2, dtype=torch.uint8, device=device
            ),
            x_scale=torch.empty(
                max_tokens, hidden_size // 16, dtype=torch.uint8, device=device
            ),
            fc1=torch.empty(
                pairs, 2 * intermediate_size, dtype=torch.float32, device=device
            ),
            fc1_split=torch.empty(
                pairs, 2 * intermediate_size, dtype=torch.float32, device=device
            ),
            act_q=torch.empty(
                pairs, intermediate_size // 2, dtype=torch.uint8, device=device
            ),
            act_scale=torch.empty(
                pairs, intermediate_size // 16, dtype=torch.uint8, device=device
            ),
            fc2=torch.empty(pairs, hidden_size, dtype=torch.float32, device=device),
            pair_experts=torch.empty(pairs, dtype=torch.int32, device=device),
            group_rows=torch.empty(pairs, dtype=torch.int32, device=device),
            group_pairs=torch.empty(
                MAX_LOCAL_EXPERTS, pairs, dtype=torch.int32, device=device
            ),
            expert_counts=torch.empty(
                MAX_LOCAL_EXPERTS, dtype=torch.int32, device=device
            ),
            group_experts=torch.empty(pairs, dtype=torch.int32, device=device),
            group_offsets=torch.empty(pairs, dtype=torch.int32, device=device),
            num_groups=torch.empty(1, dtype=torch.int32, device=device),
            max_tokens=max_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            graph_capture_supported=graph_capture_supported,
        )

    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(
            getattr(self, field.name).data_ptr()
            for field in fields(self)
            if isinstance(getattr(self, field.name), torch.Tensor)
        )

    def _bind_current_stream(
        self, stream: torch.cuda.Stream, *, capturing: bool = False
    ) -> None:
        stream_id = stream.cuda_stream
        if stream_id != self._last_stream:
            # SGLang captures a model on one Python thread after synchronizing
            # its warmup stream. The eager path makes a stream change wait for
            # the previous launch; capture uses the serialized setup handoff.
            if not capturing and self._completion_event is not None:
                stream.wait_event(self._completion_event)
            self._last_stream = stream_id

    def _record_completion(self, stream: torch.cuda.Stream) -> None:
        if self._completion_event is None:
            # An external event remains a valid synchronization point when its
            # record is captured as an explicit CUDA graph node.
            self._completion_event = torch.cuda.Event(
                enable_timing=False, external=True
            )
        self._completion_event.record(stream)


@cache_once
def _jit_nvfp4_moe_module(hidden_size: int, intermediate_size: int, top_k: int):
    if not is_sm120_supported() or is_sm121():
        raise RuntimeError(
            "nvfp4_moe_sm120 requires an SM120 GPU; SM121 (GB10) is refused "
            "until its cubins are validated"
        )
    args = make_cpp_args(hidden_size, intermediate_size, top_k)
    return load_jit(
        "nvfp4_moe_sm120",
        *args,
        cuda_files=["moe/nvfp4_moe_sm120.cuh"],
        cuda_wrappers=[
            ("nvfp4_moe_sm120", f"Nvfp4MoeKernel<{args}>::run"),
            (
                "nvfp4_moe_sm120_graph_capture_supported",
                f"Nvfp4MoeKernel<{args}>::graph_capture_supported",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _nvfp4_moe_graph_capture_supported(
    hidden_size: int, intermediate_size: int, top_k: int
) -> bool:
    module = _jit_nvfp4_moe_module(hidden_size, intermediate_size, top_k)
    return bool(module.nvfp4_moe_sm120_graph_capture_supported())


def nvfp4_moe_sm120_enabled() -> bool:
    return envs.SGLANG_NVFP4_MOE_SM120.get()


def nvfp4_moe_sm120(
    *,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    input_scale_1: torch.Tensor,
    input_scale_2: torch.Tensor,
    g1_alpha: torch.Tensor,
    g1_alpha_up: torch.Tensor,
    g2_alpha: torch.Tensor,
    global_routed_experts: int,
    local_routed_experts: int,
    local_expert_start: int,
    output: torch.Tensor,
    workspace: Nvfp4MoeWorkspace,
) -> bool:
    if x.shape[0] > workspace.max_tokens:
        raise ValueError(
            f"workspace holds {workspace.max_tokens} tokens, got {x.shape[0]}"
        )
    if (
        workspace.top_k != topk_ids.shape[1]
        or workspace.hidden_size != x.shape[1]
        or workspace.intermediate_size != w2_weight.shape[2] * 2
    ):
        raise ValueError("workspace does not match the MoE shape")
    capturing = _is_stream_capturing(torch.cuda.current_stream(x.device))
    if capturing and workspace.graph_capture_supported is not True:
        return False

    module = _jit_nvfp4_moe_module(
        workspace.hidden_size, workspace.intermediate_size, workspace.top_k
    )
    if workspace.graph_capture_supported is None:
        workspace.graph_capture_supported = _nvfp4_moe_graph_capture_supported(
            workspace.hidden_size,
            workspace.intermediate_size,
            workspace.top_k,
        )

    stream = torch.cuda.current_stream(x.device)
    with workspace._launch_lock:
        workspace._bind_current_stream(stream, capturing=capturing)
        launched = module.nvfp4_moe_sm120(
            x,
            topk_ids,
            topk_weights,
            w13_weight,
            w2_weight,
            w13_scale,
            w2_scale,
            input_scale_1,
            input_scale_2,
            g1_alpha,
            g1_alpha_up,
            g2_alpha,
            global_routed_experts,
            local_routed_experts,
            local_expert_start,
            workspace.x_q,
            workspace.x_scale,
            workspace.fc1,
            workspace.fc1_split,
            workspace.act_q,
            workspace.act_scale,
            workspace.fc2,
            output,
            workspace.pair_experts,
            workspace.group_rows,
            workspace.group_pairs,
            workspace.expert_counts,
            workspace.group_experts,
            workspace.group_offsets,
            workspace.num_groups,
        )
        if launched:
            workspace._record_completion(stream)
        return bool(launched)


def prepare_nvfp4_moe_sm120(
    *,
    max_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> Nvfp4MoeWorkspace:
    graph_capture_supported = _nvfp4_moe_graph_capture_supported(
        hidden_size, intermediate_size, top_k
    )
    return Nvfp4MoeWorkspace.allocate(
        max_tokens=max_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        graph_capture_supported=graph_capture_supported,
    )


__all__ = [
    "NVFP4_MOE_SM120_MAX_TOKENS",
    "Nvfp4MoeWorkspace",
    "nvfp4_moe_sm120",
    "nvfp4_moe_sm120_enabled",
    "prepare_nvfp4_moe_sm120",
]
