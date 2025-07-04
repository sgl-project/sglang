from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

IS_TRITON_KERNELS_AVAILABLE = False
try:
    import os
    import sys
    llm_root = os.getenv('SGL_ROOT')
    if llm_root:
        # On CI, we use SGL_ROOT to locate the 3rdparty directory.
        triton_path = os.path.join(llm_root, '3rdparty', 'triton', 'python',
                                   'triton_kernels')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        triton_path = os.path.join(current_dir, '..', '..', '..', '..', '..', '..',
                                   '3rdparty', 'triton', 'python',
                                   'triton_kernels')
    triton_path = os.path.abspath(triton_path)
    if os.path.exists(triton_path) and triton_path not in sys.path:
        sys.path.insert(0, triton_path)
    import triton
    import triton_kernels
    import triton_kernels.swiglu
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, matmul_ogs
    from triton_kernels.routing import routing
    IS_TRITON_KERNELS_AVAILABLE = True
except ImportError:
    print("triton_kernels not available")

def fused_experts_oai(
    hidden_states: torch.Tensor,  # (num_tokens, hidden_dim)
    w13: torch.Tensor,  # (num_experts, hidden_dim, intermediate_dim * 2)
    w2: torch.Tensor,  # (num_experts, intermediate_dim, hidden_dim)
    expert_logits: torch.Tensor,  # (num_tokens, num_experts)
    top_k: int,
    inplace: bool,
    activation: str,  # "swiglu"
    apply_router_weight_on_input: bool,
    no_combine: bool,
    routed_scaling_factor: Optional[float],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    swiglu_alpha: torch.Tensor,
    swiglu_beta: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:

    gemm1_weights = w13
    gemm2_weights = w2

    num_experts = expert_logits.shape[1]
    if num_experts > 1:
        rdata, gather_indx, scatter_indx = routing(expert_logits, top_k)
    else:
        rdata, gather_indx, scatter_indx = None, None, None

    pc1 = PrecisionConfig(flex_ctx=FlexCtx(),
                        allow_tf32=False,
                        out_dtype=dtype)
    gemm1_output = matmul_ogs(
        hidden_states,
        gemm1_weights,
        w1_bias,
        rdata,
        gather_indx=gather_indx,
        precision_config=pc1)

    pcs = triton_kernels.swiglu.PrecisionConfig(limit=None)

    act_out = triton_kernels.swiglu.swiglu(
        gemm1_output,
        alpha=swiglu_alpha,
        beta=swiglu_beta,
        precision_config=pcs,
        routing_data=rdata)

    pc2 = PrecisionConfig(flex_ctx=FlexCtx(),
                            allow_tf32=False,
                            out_dtype=dtype)

    gemm2_output = matmul_ogs(
        act_out,
        gemm2_weights,
        w2_bias,
        rdata,
        scatter_indx=scatter_indx,
        precision_config=pc2,
        gammas=rdata.gate_scal if rdata else None)
    return gemm2_output
