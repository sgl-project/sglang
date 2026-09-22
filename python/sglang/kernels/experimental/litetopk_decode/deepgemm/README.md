# Optional DeepGEMM histogram producer

[opt_in_histogram.patch](opt_in_histogram.patch) adds
`deep_gemm.fp8_paged_mqa_logits_with_histogram` to the official
[sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM) main commit
`1853080cf74589228fcd5dc333197c2ca6df49cf`. It is an independent companion
patch, with five files changed (+236 / −15). Its SHA256 is
`8f5aab43b32e9c28552e9a752972a826473cac1104bf23ce9b2888a2849f5103`.
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
and link libraries. It validates the five patched source hashes, key
unchanged source/dependency hashes, and the companion patch. When Git is
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

The new API accepts the native paged inputs and schedule, followed by
`max_context_len`, `histogram`, and `diagnostics`; its keyword arguments are
`clean_logits=False` and `indices=None`. Supported inputs are SM100 FP8
E4M3 queries `[B, 1, 32, 128]`, FP32 weights, native packed FP8 KV pages of
size 32/64/128, and a maximum length in `1..1048576`. Native requirements
for valid lengths, page IDs and schedule contents still apply.

`histogram` is contiguous CUDA `int32[B, 1024]` and must be **zero at entry**
for a single invocation's counts. The producer accumulates into it without
launching a reset; the caller or selector resets it before reuse and graph
replay. `diagnostics` is contiguous CUDA `int32[deep_gemm.get_num_sms()]`,
overwritten on every call; bit `0x4` reports a live NaN. Only live, non-NaN
scores contribute to the FP16-rank histogram; FP32 logits retain the native
reduction semantics. `clean_logits=True` is rejected by this opt-in API.

`indices=None` enables the Q1 tile; any supplied indices tensor retains Q4
grouping. Actual B can differ between separate calls with matching tensor
shapes and schedule. This API does not provide a device-active-batch scalar;
its same-graph qualification updates tensor contents at fixed B.

For the exact source patch, the qualification covered 76 records at
B1/3/65/97/128, 10 API-contract records (four positive and six host rejects),
and two Compute Sanitizer suites of 20 and 10 records with zero errors.
Three clean-main/patched-main original-API smoke cases had matching live
score hashes. Full machine instructions and resources matched clean main
for 16 representative original kernel instances under the audited compiler;
this is not an exhaustive compiler/template guarantee or a claim that main
is identical to the SGLang-pinned v0.2.0 release. Header changes can invalidate
the JIT cache and cause a cold original-path call to recompile.
