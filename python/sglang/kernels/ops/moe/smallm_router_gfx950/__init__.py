"""gfx950 one-launch Qwen3.5 MoE router for 1..8 tokens: gate GEMV (routed logits rounded to bf16, as today) +
softmax top-10 + renormalize + fused shared-expert slot, written as the [M, 11] int32 ids / fp32 weights that
smallm_moe_gfx950 consumes. SGLANG_ROCM_SMALLM_ROUTER=0 disables it; a build/load failure disables it too."""

import ctypes
import os
import subprocess
import tempfile

import torch

from sglang.kernels.ops.moe.smallm_moe_gfx950 import _check, _hip_lib, _hipcc, _Kernel
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smallm_router.hip")
MAX_TOK = 8  # MAX_TOK in smallm_router.hip; from 9 tokens the unfused path is faster
_kernels: dict = {}
_ws: dict = {}
_disabled = False


class Args(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        (n, ctypes.c_void_p) for n in ("x", "wg", "ws", "ids", "wts", "logits", "cnt")
    ] + [
        ("tok", ctypes.c_int32),
        ("x_stride", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("pad", ctypes.c_int32),
    ]


def smallm_router_supported(x: torch.Tensor) -> bool:
    return (
        not _disabled
        and get_bool_env_var("SGLANG_ROCM_SMALLM_ROUTER", "true")
        and is_gfx95_supported()
        and 1 <= x.shape[0] <= MAX_TOK
        and x.dtype == torch.bfloat16
        and x.shape[1] == 4096
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    )


def smallm_router(x, gate_weight, shared_gate_weight, scale=1.0):
    """(topk_weights [M, 11] fp32, topk_ids [M, 11] int32), or None if unavailable (build failure, first call
    under graph capture)."""
    global _disabled
    ws = _ws.get(x.device)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ws = _ws[x.device] = (
            torch.empty(MAX_TOK * 520, dtype=torch.float32, device=x.device),
            torch.zeros(64 * 17, dtype=torch.int32, device=x.device),
        )
    if not _kernels:
        co = os.path.join(tempfile.mkdtemp(prefix="sglang_smallm_router_"), "r.co")
        cmd = [_hipcc(), "--genco", "--offload-arch=gfx950", "-O3", "-o", co, _SRC]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            mod = ctypes.c_void_p()
            _check(_hip_lib().hipModuleLoad(ctypes.byref(mod), co.encode()), "load")
            _kernels.update({r: _Kernel(mod, f"smallm_router_r{r}") for r in (1, 2)})
        except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[smallm_router] disabled: {e}", flush=True)
            _disabled = True
            return None
    tok = x.shape[0]
    ids = torch.empty(tok, 11, dtype=torch.int32, device=x.device)
    wts = torch.empty(tok, 11, dtype=torch.float32, device=x.device)
    ptrs = (x, gate_weight, shared_gate_weight, ids, wts, *ws)
    args = Args(*(t.data_ptr() for t in ptrs), tok, x.stride(0), float(scale), 0)
    rows = 2 if tok <= 2 else 1
    stream = torch.cuda.current_stream().cuda_stream
    _kernels[rows].launch(512 // rows + 1, args, stream)
    return wts, ids


def qwen_moe_topk_output(block, hidden_states):
    """StandardTopKOutput for Qwen2MoeSparseMoeBlock with fused shared expert, or None (unfused path)."""
    from sglang.srt.environ import envs
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.moe import get_moe_runner_backend
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput,
        _eplb_remap_enabled,
        capture_routed_experts_if_allowed,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        is_in_tc_piecewise_cuda_graph,
    )
    from sglang.srt.runtime_context import get_parallel

    ok = getattr(block, "_smallm_router_ok", None)
    if ok is None:
        cfg, gw = block.topk.topk_config, block.gate.weight
        sw, backend = block.shared_expert_gate.weight, get_moe_runner_backend()
        ok = block._smallm_router_ok = (
            block.num_fused_shared_experts == 1
            and get_parallel().moe_ep_size == 1
            and (backend.is_auto() or backend.is_aiter())
            and cfg.top_k == 10
            and cfg.renormalize
            and type(gw.data) is torch.Tensor
            and gw.dtype == sw.dtype == torch.bfloat16
            and gw.shape == (512, 4096)
            and sw.shape == (1, 4096)
            and gw.is_contiguous()
            and sw.is_contiguous()
            and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
            and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
            and not _eplb_remap_enabled()
        )
    if (
        not ok
        or not smallm_router_supported(hidden_states)
        or is_in_tc_piecewise_cuda_graph()
    ):
        return None
    gw, sw = block.gate.weight, block.shared_expert_gate.weight
    out = smallm_router(hidden_states, gw, sw, block._shared_expert_scale())
    if out is None:
        return None
    routed = out[1][:, :10]
    capture_routed_experts_if_allowed(block.topk.topk_config, block.layer_id, routed)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=routed)
    return StandardTopKOutput(out[0], out[1], None)
