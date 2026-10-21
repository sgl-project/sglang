# Ascend LIM native source and standalone build

This directory supplies the native source dependency of
`sglang.kernels.ops.attention.npu_lim`. It includes the LIM host tiling,
Ascend C device kernel, aclnn API, upstream torch adapter, and a standalone
torch registration. Installing vLLM is not required.

The vendor snapshot is from
[vllm-ascend PR #16640](https://github.com/vllm-project/vllm-ascend/pull/16640),
commit `3326f350c0de30a5170b86ecfce851322e3216e6`. Files under `vendor/` are
byte-for-byte copies with a size/SHA256 inventory in `upstream.json`.
The pinned upstream archive provides its CANN build infrastructure and common
headers. The build stages that infrastructure outside this checkout, verifies
the archive and operator sources, and builds only LIM.

## Scope and validation status

The registered symbol is
`_C_ascend::npu_fused_lightning_indexer_manage`, with 19 tensor arguments
and seven in-place tensor aliases. It produces TopK/cache-management metadata
and miss plans. It does not copy the main KV payload or perform SFA.
Copy-SFA is a separate upstream operator and is not included in this build.
BF16/FP16 are supported by the native ABI; the SGLang integration currently
requires BF16 KV, Q1, page 128, K=2048 and the other restrictions documented
in [LIM_VALIDATION.txt](../../../benchmark/hisparse/LIM_VALIDATION.txt).
This build targets Ascend A3 (`ascend910_93`).

The 151/217 measurements used a previously built external probe library with
SHA256 `5fabea4cb210d04e8c72c1e01f2b3fdc2866ab9bc2f35f56f07c5a775596be0a`.
This source delivery uses the upstream revision referenced by those experiments.
The new standalone build entry has passed source staging and static checks;
it has **not** been compiled or executed on Ascend. At delivery time the 217
SSH connection was unavailable, so the original native build directory could
not be re-audited. No binary identity or new full-model result is claimed.
Rebuild and rerun the native/framework checks before substituting this library
for the historically measured probe.

## Build in an Ascend development environment

Use Linux with CANN 9.1 development headers/compiler, a compatible C++ compiler,
CMake, Ninja, PyTorch 2.10 and torch_npu 2.10. The pinned CANN build also needs
its third-party build dependencies; provide their existing cache using
`--cann-third-party` when required. The historical container used
Python 3.12.13, torch 2.10.0 and torch_npu 2.10.0.post6.

Source your CANN environment first so `ASCEND_HOME_PATH` is set, then run from
the SGLang repository root:

```bash
python 3rdparty/ascend/lim/build.py --work-dir /tmp/sglang-lim --jobs 8
source /tmp/sglang-lim/env.sh
```

The build compiles the CANN package with
`--ops=fused_lightning_indexer_manage --soc=ascend910_93`, installs it only
under the work directory, builds `sglang_npu_lim.so`, and checks torch schema
registration. It does not install a package globally or launch a model.
Outputs include `install/opp/`, `install/lib/sglang_npu_lim.so`, `env.sh`, and
`binding-build.json` with the resulting library hash and runtime versions.
The generated environment uses the actual install location, without the
historical test machine's paths.

For an offline source archive, download the fixed `archive_url` in
`upstream.json`, verify its `archive_sha256`, and pass it with `--archive`.
The CANN build's own dependency cache must also be available for offline
compilation. Source staging alone works without torch, CANN or an NPU:

```bash
python 3rdparty/ascend/lim/build.py --work-dir /tmp/sglang-lim \
  --archive /path/to/upstream.tar.gz --prepare-only
```

To compile just the torch binding against an existing LIM-enabled CANN vendor:

```bash
python 3rdparty/ascend/lim/build.py --work-dir /tmp/sglang-lim \
  --vendor-path /path/to/vendors/custom_transformer
source /tmp/sglang-lim/env.sh
```

That vendor must expose `op_api/lib/libcust_opapi.so` with the matching LIM
API. Use a fresh work directory when changing revisions. Only load one torch
library defining this LIM symbol in a process.

## Use with SGLang

Source the generated `env.sh` before starting the server; it sets the custom
CANN vendor path and `SGLANG_NPU_LIM_LIBRARY`. Then configure the existing
experimental integration:

```bash
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
export SGLANG_NPU_SPARSE_KV_INDEXER=lim
export SGLANG_NPU_SPARSE_KV_CACHE_SIZE=4096
```

Schema registration alone does not validate NPU execution. Run the existing
`benchmark/hisparse/check_lim_graph_transitions.py` and
`benchmark/hisparse/bench_lim_framework.py` with their documented Ascend/host
helper setup, then the model clients in `benchmark/hisparse/`. Capture the new
library hash alongside the results. The earlier model validation found
long-generation numerical divergence and substantial no-offload latency
regression; bundling the sources does not resolve those limitations.

## Source notices

Upstream copyright and license headers are retained in `vendor/`. The upstream
repository license is included in `licenses/Apache-2.0.txt`; files declaring
CANN Open Software License Agreement Version 2.0 retain that declaration and
its text in `licenses/CANN-2.0.txt`. The aclnn torch adapter retains its BSD
notices; the torch_npu license is included in `licenses/torch_npu-BSD.txt`.
Per-file notices apply to the corresponding upstream components. New build
glue and registration code follow the SGLang repository license.
