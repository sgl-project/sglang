"""Phase 1.1: AST Analysis of _fwd_kernel in prefill_attention.py"""
import ast
from pathlib import Path

SOURCE_FILE = Path(__file__).resolve().parents[4] / "python/sglang/srt/layers/attention/triton_ops/prefill_attention.py"
source = SOURCE_FILE.read_text()
tree = ast.parse(source)

# Find _fwd_kernel function
func = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_fwd_kernel"][0]

# 1. Parameters: constexpr vs runtime
print("=" * 60)
print("PARAMETER ANALYSIS")
print("=" * 60)
constexpr_params = []
runtime_params = []
for arg in func.args.args:
    name = arg.arg
    if arg.annotation:
        ann_src = ast.unparse(arg.annotation)
        if "constexpr" in ann_src:
            constexpr_params.append(name)
        else:
            runtime_params.append(name)
    else:
        runtime_params.append(name)

print(f"Constexpr params ({len(constexpr_params)}): {constexpr_params}")
print(f"Runtime params ({len(runtime_params)}): {runtime_params}")

# 2. Memory ops: count tl.load / tl.store
loads = []
stores = []
for node in ast.walk(func):
    if isinstance(node, ast.Call):
        func_name = ast.unparse(node.func) if hasattr(node, 'func') else ""
        if "tl.load" in func_name:
            loads.append(ast.unparse(node))
        elif "tl.store" in func_name:
            stores.append(ast.unparse(node))

print(f"\n{'=' * 60}")
print(f"MEMORY OPS")
print(f"{'=' * 60}")
print(f"tl.load calls: {len(loads)}")
for i, l in enumerate(loads):
    print(f"  [{i}] {l[:120]}")
print(f"tl.store calls: {len(stores)}")
for i, s in enumerate(stores):
    print(f"  [{i}] {s[:120]}")

# 3. Dtype casts (.to() calls)
casts = []
for node in ast.walk(func):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            casts.append(ast.unparse(node))

print(f"\n{'=' * 60}")
print(f"DTYPE CASTS (.to())")
print(f"{'=' * 60}")
print(f"Cast count: {len(casts)}")
for c in casts:
    print(f"  {c}")

# 4. Loop structure
print(f"\n{'=' * 60}")
print(f"LOOP STRUCTURE")
print(f"{'=' * 60}")
for node in ast.walk(func):
    if isinstance(node, ast.For):
        target = ast.unparse(node.target)
        iter_src = ast.unparse(node.iter)
        print(f"  for {target} in {iter_src}")
        # Count ops inside loop
        inner_loads = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call) and "tl.load" in ast.unparse(n.func))
        inner_stores = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call) and "tl.store" in ast.unparse(n.func))
        inner_dots = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call) and "tl.dot" in ast.unparse(n.func))
        inner_exps = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call) and "tl.exp" in ast.unparse(n.func))
        print(f"    Inner tl.load: {inner_loads}")
        print(f"    Inner tl.store: {inner_stores}")
        print(f"    Inner tl.dot: {inner_dots}")
        print(f"    Inner tl.exp: {inner_exps}")

# 5. Arithmetic: tl.exp, tl.dot, divisions
dots = []
exps = []
divs = []
for node in ast.walk(func):
    if isinstance(node, ast.Call):
        fn = ast.unparse(node.func) if hasattr(node, 'func') else ""
        if "tl.dot" in fn:
            dots.append(ast.unparse(node))
        elif "tl.exp" in fn:
            exps.append(ast.unparse(node))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        divs.append(ast.unparse(node))

print(f"\n{'=' * 60}")
print(f"ARITHMETIC OPS")
print(f"{'=' * 60}")
print(f"tl.dot calls: {len(dots)}")
for d in dots:
    print(f"  {d}")
print(f"tl.exp calls: {len(exps)}")
for e in exps:
    print(f"  {e}")
print(f"Division ops: {len(divs)}")
for d in divs:
    print(f"  {d}")

# 6. Summary
print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"{'=' * 60}")
print(f"Kernel: _fwd_kernel (flash attention for prefill)")
print(f"Grid: (batch, head, cdiv(max_len, BLOCK_M))")
print(f"Inner loop: iterates over K/V blocks of size BLOCK_N")
print(f"  - 2x tl.load (K, V) per iteration")
print(f"  - 2x tl.dot (q@k, p@v) per iteration - COMPUTE BOUND")
print(f"  - 2x tl.exp (softmax numerics) per iteration")
print(f"  - 1x division (p_scale = beta / l_i_new)")
print(f"  - 1x .to() cast (p to v.dtype)")
print(f"Outer: 1x tl.load for Q, 1x tl.store for output")
