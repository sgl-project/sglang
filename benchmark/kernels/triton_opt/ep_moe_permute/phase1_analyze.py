"""Phase 1: Understand the deepep_permute_triton_kernel.

AST analysis, PTX extraction, instruction annotation, roofline, occupancy.
"""
import ast
import os
import re
import shutil
from pathlib import Path

import torch
import triton
import triton.language as tl

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_gpu_3"

# Clear cache to get fresh PTX
cache_dir = Path("/tmp/triton_cache_gpu_3")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

# ---- 1.1 AST Analysis ----
print("=" * 60)
print("PHASE 1.1: AST Analysis")
print("=" * 60)

kernel_file = Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt" / "layers" / "moe" / "ep_moe" / "kernels.py"
source = kernel_file.read_text()
tree = ast.parse(source)

TARGET = "deepep_permute_triton_kernel"
func = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == TARGET:
        func = node
        break

assert func is not None, f"Could not find {TARGET}"

# Parameters
print(f"\nFunction: {func.name}")
print(f"Line range: {func.lineno}-{func.end_lineno}")
print(f"\nParameters:")
for arg in func.args.args:
    annotation = ast.dump(arg.annotation) if arg.annotation else "None"
    is_constexpr = "tl.constexpr" in annotation
    print(f"  {arg.arg}: {'constexpr' if is_constexpr else 'runtime'}")

# Memory ops
loads = [n for n in ast.walk(func) if isinstance(n, ast.Call) and
         isinstance(n.func, ast.Attribute) and n.func.attr == 'load']
stores = [n for n in ast.walk(func) if isinstance(n, ast.Call) and
          isinstance(n.func, ast.Attribute) and n.func.attr == 'store']
print(f"\nMemory ops: {len(loads)} loads, {len(stores)} stores")

# Dtype casts
casts = [n for n in ast.walk(func) if isinstance(n, ast.Call) and
         isinstance(n.func, ast.Attribute) and n.func.attr == 'to']
print(f"Dtype casts (.to()): {len(casts)}")

# Loops
for_loops = [n for n in ast.walk(func) if isinstance(n, (ast.For,))]
print(f"Loops (for/tl.range): {len(for_loops)}")

# Arithmetic
arith_ops = [n for n in ast.walk(func) if isinstance(n, (ast.BinOp, ast.Compare))]
print(f"Arithmetic/Compare ops: {len(arith_ops)}")

print("\n--- Kernel Structure ---")
print("Outer loop: tl.range(0, hidden_size, BLOCK_SIZE) -- iterates over chunks of hidden dim")
print("Inner loop: range(topk) -- iterates over each expert assignment")
print("Per iteration: load src block, store to each valid dst")


# ---- 1.2 Compile and Extract PTX ----
print("\n" + "=" * 60)
print("PHASE 1.2: Compile and Extract PTX")
print("=" * 60)

@triton.jit
def deepep_permute_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    src_ptr = input_ptr + src_idx * hidden_size

    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)

        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


# Representative: num_tokens=1024, hidden_size=4096, topk=6
device = torch.device("cuda:0")
num_tokens = 1024
hidden_size = 4096
topk = 6

input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
gateup_input = torch.empty(num_tokens * topk, hidden_size, device=device, dtype=torch.bfloat16)
src2dst = torch.arange(num_tokens * topk, device=device, dtype=torch.int64).reshape(num_tokens, topk)
topk_ids = torch.randint(0, 64, (num_tokens, topk), device=device, dtype=torch.int64)

grid = (num_tokens,)
deepep_permute_triton_kernel[grid](
    input_tensor, gateup_input, src2dst, topk_ids, None,
    topk, hidden_size, BLOCK_SIZE=512,
)
torch.cuda.synchronize()

# Find PTX
ptx_files = sorted(cache_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
if ptx_files:
    ptx = ptx_files[0].read_text()
    print(f"PTX file: {ptx_files[0]}")
    print(f"PTX length: {len(ptx)} chars")

    # Save baseline PTX
    out_dir = Path(__file__).parent
    (out_dir / "ptx_baseline.txt").write_text(ptx)
    print(f"Saved to {out_dir / 'ptx_baseline.txt'}")
else:
    print("No PTX files found!")
    ptx = ""


# ---- 1.3 Annotate PTX ----
print("\n" + "=" * 60)
print("PHASE 1.3: PTX Instruction Analysis")
print("=" * 60)

def count_ptx(ptx_text, pattern):
    return len(re.findall(pattern, ptx_text))

categories = {
    "ld.global": r"ld\.global",
    "st.global": r"st\.global",
    "ld.shared": r"ld\.shared",
    "st.shared": r"st\.shared",
    "ld.param": r"ld\.param",
    "mul.f32 / mul.wide": r"mul\.(f32|wide)",
    "add.f32 / add.s64": r"add\.(f32|s64|s32)",
    "fma.rn.f32": r"fma\.rn\.f32",
    "cvt.*": r"cvt\.",
    "setp.*": r"setp\.",
    "mov.*": r"\tmov\.",
    "cp.async": r"cp\.async",
    "bar.sync": r"bar\.sync",
    "bra / branch": r"\tbra\b",
}

for name, pat in categories.items():
    c = count_ptx(ptx, pat)
    print(f"  {name:25s}: {c}")

# ---- 1.4 Roofline ----
print("\n" + "=" * 60)
print("PHASE 1.4: Roofline Analysis")
print("=" * 60)

# Per token:
# Loads: hidden_size elements of bf16 (2 bytes each) = hidden_size * 2 bytes
# Stores: topk * hidden_size elements of bf16 = topk * hidden_size * 2 bytes
# Also loads topk scalar src2dst indices (8 bytes each) per outer iteration
# hidden_size / BLOCK_SIZE outer iterations

for hs in [2048, 4096, 7168]:
    for tk in [2, 6, 8]:
        bytes_loaded = hs * 2 + tk * 8 * (hs // 512)  # data + index loads per outer iter
        bytes_stored = tk * hs * 2
        total_bytes = bytes_loaded + bytes_stored
        # FP ops: essentially none (just a dtype cast, which is a cvt not FP math)
        flops = 0  # pure memory copy kernel
        ai = flops / total_bytes if total_bytes > 0 else 0
        print(f"  hidden={hs}, topk={tk}: {total_bytes/1024:.1f} KB/token, AI={ai:.4f} -> MEMORY BOUND")

print("\nConclusion: Kernel is purely memory-bound (scatter copy with optional dtype cast)")

# ---- 1.5 Occupancy ----
print("\n" + "=" * 60)
print("PHASE 1.5: Occupancy Check")
print("=" * 60)

props = torch.cuda.get_device_properties(0)
print(f"Device: {props.name}")
print(f"SM count: {props.multi_processor_count}")
print(f"Max threads/SM: {props.max_threads_per_multi_processor}")

for num_warps in [1, 2, 4, 8]:
    threads_per_block = num_warps * 32
    blocks_per_sm = props.max_threads_per_multi_processor // threads_per_block
    for nt in [64, 256, 1024, 4096]:
        total_blocks = nt  # grid = (num_tokens,)
        active_sms = min(total_blocks, props.multi_processor_count * blocks_per_sm)
        waves = total_blocks / (props.multi_processor_count * blocks_per_sm)
        if nt == 1024:  # representative
            print(f"  num_warps={num_warps}, num_tokens={nt}: "
                  f"{threads_per_block} threads/block, {blocks_per_sm} blocks/SM, "
                  f"{waves:.2f} waves")

print("\nDone with Phase 1 analysis.")
