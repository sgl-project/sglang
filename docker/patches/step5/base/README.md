# Step5 FlashInfer TRT-LLM MoE

Current Step5 supports `--moe-runner-backend flashinfer_trtllm`, which performs
routing inside FlashInfer. `flashinfer_trtllm_routed` remains available for
routing in SGLang and passing expert IDs/FP32 weights to FlashInfer. These
patches also provide the baseline for the Step5 performance patches.

Both paths require the post-SiLU clamp and BF16 FC1 patches. Internal routing
additionally requires patches 06/07. The main Dockerfiles do not apply these
patches, and an unmodified FlashInfer package does not provide this integration.
The setup below combines the patch instructions and the TP4 launch configuration
with real MTP.

## Patch responsibilities

| Patches | Purpose | Used by |
| --- | --- | --- |
| Base 01 | Post-SiLU clamp: `min(silu(gate), limit) * clamp(up, -limit, limit)` | Both MoE paths |
| Base 02 | Optional FC1/activation diagnostic exports; serving leaves them disabled | Required patch context for 03 |
| Base 03 | BF16 FC1 output using NoRoute GEMM plus input/scale permutation | Both MoE paths |
| Base 06/07 | `SigmoidBiasOrderedRenorm` and matching Python enum | Internal routing |
| Step5 01/02/03 | MoE activation/copy, routing prefix and TP4 RMSNorm optimizations | Optional performance extension |

Internal routing uses biased sigmoid to select experts, unbiased sigmoid for
FP32 weights, ordered renormalization and one application of the routing scale.
Step5 uses BF16 router operands with FP32 GEMM output and FP32 bias/weights.
The model passes `silu_clamp_mode="post"` and `fc1_output_dtype="bf16"` to the
FlashInfer adapter. The clamp limit comes from the checkpoint configuration.

BF16 FC1 avoids an additional FP8 FC1-output rounding boundary. The activation
product is computed in FP32 and quantized directly to FP8 without a separate
BF16 product-rounding step; this is not a bitwise-equivalence claim.

Step5 preserves checkpoint BF16 expert layers and runs them on TRTLLM BF16
with the same post-SiLU clamp. This requires Step5 patch 04 and its capability
marker; no BF16 backend environment variable is needed.
`SGLANG_STEP5_QUANTIZE_BF16_MOE=1` explicitly opts into FP8 conversion.

## Requirements

- The original `flashinfer-python==0.7.1` wheel matching the manifests,
  as pinned in `python/pyproject.toml`. The version string alone is
  insufficient; any other package version needs a separately validated patch
  port.
- A dedicated environment or image, `patch`, and a CUDA toolkit for JIT.
  The referenced B300 runs used CUDA 13.
- For the in-place setup below, **no `flashinfer-jit-cache` AOT package**.
  Build the image with `INSTALL_FLASHINFER_JIT_CACHE=0`; prebuilt AOT modules
  otherwise override the patched sources. A fresh JIT cache alone does not
  prevent that override.
- A Step5 checkpoint with the configuration/architecture names expected by
  this branch.

From the SGLang repository root, install SGLang in the intended environment:

```bash
pip install -e "python"
```

## Apply the patches

Run from the repository root in the dedicated environment, never on a live
server. This applies the functional baseline and optional Step5 performance
patches in order.

```bash
set -eu
base_patch_dir="$(pwd)/docker/patches/step5/base"
step5_patch_dir="$(pwd)/docker/patches/step5"
flashinfer_dir="$(python -c 'import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent)')"
python - <<'PY'
import importlib.util

if importlib.util.find_spec("flashinfer_jit_cache") is not None:
    raise SystemExit("Use an environment without flashinfer-jit-cache for in-place patching")
PY

# 1. Functional baseline. The base manifest covers the eight files patched by 01/02/03.
python "$base_patch_dir/verify.py" "$flashinfer_dir" before
for patch_name in 01-flashinfer-post-silu 02-flashinfer-diagnostics 03-flashinfer-bf16-fc1; do
  patch --batch --forward --fuzz=0 -d "$flashinfer_dir" -p1 < "$base_patch_dir/$patch_name.patch"
done
python "$base_patch_dir/verify.py" "$flashinfer_dir" after

# 2. Internal routing. Run the preceding after check before 06 modifies its files.
# These patches are not covered by the base manifest.
for patch_name in 06-flashinfer-sigmoid-bias-ordered 07-flashinfer-tllm-enums; do
  patch --batch --forward --fuzz=0 -d "$flashinfer_dir" -p1 < "$base_patch_dir/$patch_name.patch"
done

# 3. Optional performance patches. Check their exact input/output hashes.
check_step5_sources() {
  python - "$flashinfer_dir" "$step5_patch_dir/manifest.json" "$1" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

package = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
stage = sys.argv[3]
bad = [
    name
    for name, entry in manifest["files"].items()
    if not (package / name).is_file()
    or hashlib.sha256((package / name).read_bytes()).hexdigest() != entry[stage]
]
if bad:
    raise SystemExit(f"Step5 {stage} mismatch: {bad}")
print(f"Step5 {stage} sources verified")
PY
}
check_step5_sources before
for patch_name in 01-flashinfer-moe 02-flashinfer-routing-prefix 03-flashinfer-tp4-rmsnorm-order 04-flashinfer-bf16-swiglu-step; do
  patch --batch --forward --fuzz=0 -d "$flashinfer_dir" -p1 < "$step5_patch_dir/$patch_name.patch"
done
check_step5_sources after
```

If a hash check or patch fails, discard the partially modified environment and
resolve the baseline mismatch. Do not force hunks or relax `--fuzz=0`.
For the functional internal-routing baseline alone, stop after step 2; the
Step5 performance extension is not required to select `flashinfer_trtllm`.

An alternative for environments retaining AOT is the isolated overlay builder
in [`../README.md`](../README.md). It requires a matching parent AOT
package and a package already patched with base 01/02/03/06/07; it applies only
the Step5 performance patches and replaces the affected AOT modules. Do not
combine that builder's AOT requirements with the in-place setup above.

## Compile and check

Choose a fresh absolute `FLASHINFER_WORKSPACE_BASE` after patching, then reuse
it for subsequent launches. Persist it and the SGLang cache directory across
container restarts. The first call compiles/autotunes kernels and is not a
steady-state measurement.

```bash
export FLASHINFER_WORKSPACE_BASE=/path/to/flashinfer-step5-cache
export FLASHINFER_JIT_DEBUG=0
```

After the complete patch chain, optionally prebuild and check the modules on
one B300 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.jit.comm import gen_trtllm_mnnvl_comm_module
from flashinfer.tllm_enums import RoutingMethodType

assert int(RoutingMethodType.SigmoidBiasOrderedRenorm) == 11
moe = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
try:
    moe.trtllm_fp8_block_scale_moe()
except TypeError as error:
    assert "Expected 46 but got 0 arguments" in str(error), error
else:
    raise AssertionError("Expected the patched 46-argument MoE ABI")
comm = gen_trtllm_mnnvl_comm_module().build_and_load()
assert comm.step5_tp4_rmsnorm_order_version() == 2
print("FlashInfer Step5 patch modules loaded")
PY
```

This checks the enum and compiled ABI/capability, not model accuracy. The
patched 0.6.18 entry point took 42 arguments; 0.7.0.post1 adds its own
arguments (including `valid_hidden_size` and `valid_intermediate_size`) ahead
of the patched `silu_clamp_mode`, `diag_return_fc1`, `diag_return_activation`
and `fc1_output_dtype`. The comm
capability is provided by Step5 patch 03. With the full performance patch set,
the TP4 launch below should initialize the eligible MNNVL fusion band
on all four ranks; if its capability checks fail, SGLang retains the original
communication path.

## Launch Step5: 4x B300, TP4/EP1, real MTP

Use the paths for the patched package environment and the Step5 checkpoint.
This command follows PR #1's BF16-KV configuration. It does not enable the
later optional FP8 FULL KV feature.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLASHINFER_WORKSPACE_BASE=/path/to/flashinfer-step5-cache
export FLASHINFER_JIT_DEBUG=0
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE

python -m sglang.launch_server \
  --model-path /path/to/step5-ckpt --tp 4 --ep-size 1 \
  --mem-fraction-static 0.9 --swa-full-tokens-ratio 0.1 \
  --attention-backend fa4 --page-size 64 \
  --reasoning-parser step5 --moe-runner-backend flashinfer_trtllm \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-multi-layer-eagle --max-running-requests 64 \
  --chunked-prefill-size 16384 \
  --flashinfer-allreduce-fusion-backend trtllm --port 31131
```

Memory fraction 0.9 is the referenced configuration, not a capacity guarantee
for every environment; lower it if graph/workspace allocation does not fit.
To use SGLang external routing, replace only the MoE backend with
`flashinfer_trtllm_routed`; patches 01/02/03 remain required.

## FlashInfer 0.7.1 port

The current patch files and manifests target the official 0.7.1 wheel. They
build on the 0.7.0.post1 port below. The BF16-FC1 tactic-query constructors
retain upstream's per-channel and per-token scaling flags and append the
Step5 FC1 output selector. All eight base/Step5 patches were regenerated
against 0.7.1 and replayed on a clean wheel with `--fuzz=0`; both manifest
stages match. The SGLang dependency remains pinned to upstream's 0.7.1.

This validates source application and hashes only. CUDA compilation,
numerical parity, TP4/TP8 collectives and serving tests have not yet been
run for 0.7.1.
Main Dockerfiles still require the explicit patch/build workflow described
above; updating these files alone does not install patched FlashInfer.

## FlashInfer 0.7.0.post1 port

The patches were ported from their 0.6.18 versions. FlashInfer 0.7.0.post1
does not contain any of these changes. `gemm1_clamp_limit` is a pre-SiLU clamp
and is not equivalent to `silu_clamp_mode="post"`; the `fc1_output_dtype` names
in 0.7 belong to the `moe_ep` CuTe megamoe kernels, not to the TRT-LLM fused MoE
path. Kernel logic is unchanged; the port follows the 0.7 source layout:

| Patch | 0.7.0.post1 difference |
| --- | --- |
| Base 01 | Also threads `silu_clamp_mode` through the new `fused_moe/backends/trtllm/sm100_runner.py`; the launcher argument is appended after the new `valid_*_size` arguments |
| Base 02 | Additional hunks in `fused_moe/shared/inputs.py` and `sm100_runner.py`; launcher hunks moved |
| Base 03 | Threads `fc1_output_dtype` through `sm100_runner.py`; kernel, runner and launcher changes unchanged |
| Base 06 | Routing dispatch moved to `trtllm_backend/trtllm_fused_moe_routing_runner.cu` |
| Base 07 | Unchanged |
| Step5 01 | Unchanged apart from context whitespace |
| Step5 02 | Target renamed to `trtllm_fused_moe_routing_custom.cuh` |
| Step5 03 | Capability export hunk now carries context; body unchanged |

Validation of the port on one GB300 GPU (SM103, CUDA 13):

- A clean 0.7.0.post1 install applies the full chain with `--fuzz=0`, without
  offsets, and passes both manifest checks before and after.
- `fused_moe_trtllm_sm100` and `trtllm_mnnvl_comm` compile; the module check
  above passes.
- `SigmoidBiasOrderedRenorm` routing matches a reference implementation
  (max abs error 1.2e-7). BF16 FC1 with the post-clamp activation is bit-exact
  with the FP8 block-scale reference over 1,572,864 elements and selects the
  Step5 warp activation kernel. E4M3 FC1 with diagnostics matches with max abs
  error 1.1e-3. EP windows (offset/local 0/32, 0/16, 16/16) and the routed
  entry point match the FP32 reference within 6.2e-3.
- SGLang fusion, layer-boundary and Step5 router unit tests: 68 passed with 50
  subtests.

Not yet validated on 0.7.0.post1: numerical TP4 MNNVL RMSNorm order (Step5 03;
only its capability export is checked), the Step5 AOT overlay builder
(`../step5/build.py`; only its ABI assertion was updated), and end-to-end
serving accuracy and throughput.
