# Optional DeepGEMM histogram producer

[opt_in_histogram.patch](opt_in_histogram.patch) adds an optional `histogram`
argument to `deep_gemm.fp8_fp4_paged_mqa_logits` in the official
[sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM) main commit
`1853080cf74589228fcd5dc333197c2ca6df49cf`. The existing paged kernel takes it
as a `kWithHistogram` template variant; no entry kernel, runtime class or API is
added. It is an independent companion patch, with four files changed
(+130 / −18). Its SHA256 is
`4397806ac79b5d03e69761832cc7b55d1ae1035f078460309e47091e57c17548`.
The original DeepGEMM license and source attribution remain applicable.

SGLang's default dependency remains `sgl-deep-gemm==0.2.0`. This companion
does not modify that dependency, install a replacement package, or enable
a default runtime route. The explicit LiteTopK fused API loads the private
build under a distinct module alias; the existing `deep_gemm` import remains
unchanged. No global `PYTHONPATH` or `sys.modules["deep_gemm"]` replacement
is needed.

Use a fresh external work directory. Set `LITETOPK_DG_PATCH` to the absolute
path of this patch before changing directories:

```sh
LITETOPK_DG_PATCH=/absolute/path/to/deepgemm/opt_in_histogram.patch
LITETOPK_DG_WORK=/absolute/path/to/private/deepgemm-work
git clone https://github.com/sgl-project/DeepGEMM.git "$LITETOPK_DG_WORK/source"
git -C "$LITETOPK_DG_WORK/source" checkout 1853080cf74589228fcd5dc333197c2ca6df49cf
git -C "$LITETOPK_DG_WORK/source" submodule update --init --recursive
git -C "$LITETOPK_DG_WORK/source" apply --check "$LITETOPK_DG_PATCH"
git -C "$LITETOPK_DG_WORK/source" apply "$LITETOPK_DG_PATCH"
```

The pinned submodules are CUTLASS
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` and fmt
`553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`. Verify these with
`git submodule status` in the checkout. The patch has been checked against
this exact main commit; it is not a patch for the v0.2.0 release.

Build the private package with [build.py](build.py). Run with an already
configured CUDA 13/PyTorch development environment and the compiler used
by the qualification. `--source-dir` is the prepared checkout above;
`--output-dir` must be fresh or empty:

```sh
CUDA_VISIBLE_DEVICES= CXX=/usr/bin/g++-13 python3 \
  /absolute/path/to/litetopk_decode/deepgemm/build.py \
  --source-dir "$LITETOPK_DG_WORK/source" \
  --output-dir "$LITETOPK_DG_WORK/package"
```

When omitted, `--output-dir` defaults to
`litetopk_decode/build/fused/deepgemm`. The output contains
`deep_gemm/_C.so`, the package's Python files and device headers, and
`manifest.json`. Pass that output directory as the fused API's
`deepgemm_package` argument. The build itself never imports the new package,
executes a GPU kernel, installs a distribution or changes import paths.
Device kernels compile through DeepGEMM's JIT on first use.

The direct C++ build retains the qualified build's compiler flags, headers
and link libraries. It validates the four patched source hashes, the
unchanged package `__init__.py`, key unchanged source/dependency hashes, and
the companion patch. When Git is
available with intact checkout metadata, it also verifies all three HEADs.
Without Git it records `git_verified: null`: commit and submodule pins are
declared provenance backed by checked file hashes, not a verified Git HEAD.
The manifest records project/dependency input hashes, compiler command,
library SHA256 and all staged package hashes. The fused loader verifies
the staged files before private import. Different compiler or library
versions still require requalification; identical source does not imply
an identical rebuilt binary.

Upstream's own wheel builder is an alternative for a separately managed
environment, but its output lacks the manifest required by the private
fused loader. It is not the primary reproduction path:

```sh
CUDA_VISIBLE_DEVICES= DG_FORCE_BUILD=1 MAX_JOBS=1 \
  python3 -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$LITETOPK_DG_WORK/wheels" "$LITETOPK_DG_WORK/source"
```

This creates a wheel without installing it. Do not install it over the
default SGLang environment to use the private fused loader.

Call the native API with the keyword:
`fp8_fp4_paged_mqa_logits((q, None), kv_cache, weights, context_lens,
block_table, schedule_meta, max_context_len, clean_logits=False,
indices=None, histogram=histogram)`. Omitting it keeps the native call and
template instance. The histogram variant supports SM100 FP8 E4M3 queries
`[B, 1, 32, 128]`, FP32 weights and logits, and native packed FP8 KV pages of
size 32/64/128; other inputs and `clean_logits=True` are rejected before
launch. Native requirements for valid lengths, page IDs and schedule contents
still apply.

`histogram` is contiguous CUDA `int32[B, 1024]` and must be **zero at entry**
for a single invocation's counts. The producer accumulates into it without
launching a reset; the caller or selector resets it before reuse and graph
replay. Only live, non-NaN scores are counted, so a NaN score leaves the row's
total below its length. Bins follow `hybrid1024-fp16rn16-unit-overflow-v1`:
exact FP16-RN bins below magnitude 16, then unit-width bins that saturate at
223. FP32 logits retain the native reduction semantics.

`indices=None` enables the Q1 tile; any supplied indices tensor retains Q4
grouping. Actual B can differ between separate calls with matching tensor
shapes and schedule. This API does not provide a device-active-batch scalar;
its same-graph qualification updates tensor contents at fixed B.

DeepGEMM keys its JIT cache by the generated kernel source, the compiler flags
(which include the package's include path) and an include hash. That hash is a
process-wide static shared by every DeepGEMM build loaded into one process, so
it can come from another build's headers. Point `DG_JIT_CACHE_DIR` at a
directory used only by this build (each DeepGEMM library reads it at its first
JIT compile) and clear that directory when rebuilding the package in place.

For this patch, qualification used real GLM-5.2 1M fixtures at B1/131072,
B2/786432, B8/524288, B32/262144 and B128/1048321. Histogram-path scores were
bit-identical to the pristine main native path and to the qualified r34
producer; histograms matched r34 and an independent CPU model of the mapping,
also with `indices` (Q4 tile). Native-path scores matched pristine main over
every written column. A NaN planted in a live score was left out of the
counts; one planted past the live length changed nothing. Host rejections
raised before launch without touching the histogram. For 16 audited native
kernel instances, machine code, registers, shared memory and spills matched
pristine main under the audited compiler; the 13 paged instances differ only
in 8 more bytes of parameter space for the appended pointer, which the native
path does not read. This is not an exhaustive compiler/template guarantee or
a claim that main is identical to the SGLang-pinned v0.2.0 release.
