#!/usr/bin/env python3
"""Patch dsa_indexer.py + dsa/utils.py: enable DSA graph split-op dispatch on HIP.

Seven issues fixed:
1. Graph functions defined inside `if _is_cuda:` → change to `if _is_cuda or _is_hip:`
2. logits_head_gate_graph uses torch.mm with FP8 input → cast to bfloat16 first
3. aiter 3-tuple (fp8, scale, bf16) passed to graph function → extract bf16
4. is_graph_dsa_split_op_surface checks is_cuda() → add is_hip() support
5. split-op dispatch uses x.shape/x.device but x may be aiter 3-tuple → extract tensor
6. pcg_dsa_indexer_prefill_split asserts _is_cuda → allow _is_hip
7. graph_dispatch_fn called with tuple x → extract tensor before calling
"""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

with open(FILE, "r") as f:
    content = f.read()

# Patch 1: HIP graph functions
old = """if _is_cuda:
    from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_first_quant
    from sglang.jit_kernel.dsv32 import (
        fused_k_indexer_norm_rope,
        fused_k_indexer_norm_rope_store,
    )

    def _scale_head_gate_graph_fake_impl("""

new = """if _is_cuda or _is_hip:
    if _is_cuda:
        from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_first_quant
        from sglang.jit_kernel.dsv32 import (
            fused_k_indexer_norm_rope,
            fused_k_indexer_norm_rope_store,
        )

    def _scale_head_gate_graph_fake_impl("""

if old in content:
    content = content.replace(old, new)
    print("PATCHED: dsa_indexer.py graph functions now available on HIP")
else:
    if "if _is_cuda or _is_hip:" in content:
        print("dsa_indexer.py: already patched (graph functions on HIP)")
    else:
        print("WARNING: dsa_indexer.py pattern not found for graph function fix")
        sys.exit(1)

# Patch 2: FP8 dtype fix
old_mm = """        out = torch.mm(x, weight.t(), out_dtype=torch.float32)"""
new_mm = """        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
        out = torch.mm(x_bf16, weight.t(), out_dtype=torch.float32)"""
if old_mm in content:
    content = content.replace(old_mm, new_mm)
    print("PATCHED: dsa_indexer.py logits_head_gate_graph FP8 dtype fix applied")
else:
    if "x_bf16 = x.to(torch.bfloat16)" in content:
        print("dsa_indexer.py: already patched (FP8 dtype fix present)")
    else:
        print("WARNING: dsa_indexer.py FP8 dtype pattern not found")

# Patch 3: aiter 3-tuple extraction for logits_head_gate_graph
old_call = """                    weights = logits_head_gate_graph(
                        x_for_gate,
                        self.weights_proj.weight,
                        self.n_heads**-0.5,
                        self.softmax_scale,
                        q_scale,
                    )"""
new_call = """                    x_gate = x_for_gate
                    if isinstance(x_gate, tuple) and len(x_gate) == 3:
                        x_gate = x_gate[2]
                    elif isinstance(x_gate, tuple) and len(x_gate) == 2:
                        x_gate = x_gate[0].to(torch.bfloat16)
                    weights = logits_head_gate_graph(
                        x_gate,
                        self.weights_proj.weight,
                        self.n_heads**-0.5,
                        self.softmax_scale,
                        q_scale,
                    )"""
if old_call in content:
    content = content.replace(old_call, new_call)
    print("PATCHED: dsa_indexer.py aiter 3-tuple extraction for graph path")
else:
    if "x_gate = x_for_gate" in content:
        print("dsa_indexer.py: already patched (aiter tuple extraction)")
    else:
        print("WARNING: dsa_indexer.py aiter tuple pattern not found")

# Patch 5: Extract tensor from aiter 3-tuple in split-op dispatch
old_split = """            if weights_proj_lora:
                raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
            x_tensor = x[2] if isinstance(x, tuple) and len(x) == 3 else x
            if return_indices:
                topk_result = torch.full(
                    (x_tensor.shape[0], self.index_topk),
                    -1,
                    device=x_tensor.device,
                    dtype=torch.int32,
                )"""
new_split = """            if weights_proj_lora:
                raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
            x_tensor = x[2] if isinstance(x, tuple) and len(x) == 3 else x
            if return_indices:
                topk_result = torch.full(
                    (x_tensor.shape[0], self.index_topk),
                    -1,
                    device=x_tensor.device,
                    dtype=torch.int32,
                )"""
# Already patched, skip
if "x_tensor = x[2]" in content:
    print("dsa_indexer.py: already patched (split-op tuple extraction)")
else:
    old_split_v0 = """            if weights_proj_lora:
                raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
            if return_indices:
                topk_result = torch.full(
                    (x.shape[0], self.index_topk),
                    -1,
                    device=x.device,
                    dtype=torch.int32,
                )"""
    if old_split_v0 in content:
        content = content.replace(old_split_v0, new_split)
        print("PATCHED: dsa_indexer.py split-op dispatch aiter tuple extraction")
    else:
        print("WARNING: dsa_indexer.py split-op pattern not found")

# Patch 7: Extract tensor before calling graph_dispatch_fn
old_dispatch = """            graph_dispatch_fn(
                layer_id=layer_id,
                x=x,
                q_lora=q_lora,
                positions=positions,
                topk_result=topk_result,
            )"""
new_dispatch = """            x_dispatch = x_tensor if 'x_tensor' in dir() else (x[2] if isinstance(x, tuple) and len(x) == 3 else x)
            graph_dispatch_fn(
                layer_id=layer_id,
                x=x_dispatch,
                q_lora=q_lora,
                positions=positions,
                topk_result=topk_result,
            )"""
if old_dispatch in content:
    content = content.replace(old_dispatch, new_dispatch)
    print("PATCHED: dsa_indexer.py graph_dispatch_fn tensor extraction before call")
else:
    if "x_dispatch" in content:
        print("dsa_indexer.py: already patched (dispatch tensor extraction)")
    else:
        print("WARNING: dsa_indexer.py dispatch pattern not found")

# Patch 6: Fix assert _is_cuda in pcg_dsa_indexer_prefill_split
old_assert = '''    assert _is_cuda, "Internal error: DSA graph dispatch is only supported on CUDA"'''
new_assert = '''    assert _is_cuda or _is_hip, "Internal error: DSA graph dispatch is only supported on CUDA"'''
if old_assert in content:
    content = content.replace(old_assert, new_assert)
    print("PATCHED: dsa_indexer.py pcg_dsa_indexer_prefill_split HIP support")
else:
    if "_is_cuda or _is_hip" in content:
        print("dsa_indexer.py: already patched (split-op HIP support)")
    else:
        print("WARNING: dsa_indexer.py split-op assert pattern not found")

with open(FILE, "w") as f:
    f.write(content)

# ---- Patch dsa/utils.py ----
FILE2 = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/utils.py"
with open(FILE2, "r") as f:
    content2 = f.read()
old_check = """    return (
        is_cuda()
        and (is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph())
        and forward_batch.forward_mode.is_extend_without_speculative()
    )"""
new_check = """    return (
        (is_cuda() or is_hip())
        and (is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph())
        and forward_batch.forward_mode.is_extend_without_speculative()
    )"""
if old_check in content2:
    content2 = content2.replace(old_check, new_check)
    if "is_hip" not in content2:
        old_import = "from sglang.srt.utils import is_cuda"
        new_import = "from sglang.srt.utils import is_cuda, is_hip"
        if old_import in content2:
            content2 = content2.replace(old_import, new_import)
        else:
            for line in content2.split("\n"):
                if "is_cuda" in line and "import" in line:
                    content2 = content2.replace(line, line.replace("is_cuda", "is_cuda, is_hip"))
                    break
    with open(FILE2, "w") as f:
        f.write(content2)
    print("PATCHED: dsa/utils.py is_graph_dsa_split_op_surface now supports HIP")
else:
    if "is_cuda() or is_hip()" in content2:
        print("dsa/utils.py: already patched (HIP support in split-op surface)")
    else:
        print("WARNING: dsa/utils.py pattern not found")
