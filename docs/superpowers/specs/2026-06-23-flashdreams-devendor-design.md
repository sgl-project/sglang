# Design: De-vendor FlashDreams CUDA Sources via sync_thirdparty

**Date:** 2026-06-23
**Branch:** `feat/omnidreams-p0-p4b-optimizations`

## Problem

`native/omnidreams_singleview/src/` contains **64 CUDA/C++ files (36K lines)** vendored from
NVIDIA's FlashDreams reference implementation. These are committed directly into the sglang repo,
which is problematic:

1. **Repo bloat**: 36K lines of model-specific CUDA in a Python-focused project.
2. **Upstream tracking**: No mechanism to track FlashDreams updates — manual copy-paste required.
3. **Contributor confusion**: Other models (Wan, FLUX, HunyuanVideo) have zero native extensions.
   Having 36K lines of NVIDIA-copyrighted CUDA under `python/sglang/` is misleading.
4. **Diff noise**: Any upstream update produces a massive diff that's hard to review.

## Current State

The repo already has a third-party sync mechanism:

```
native/omnidreams_singleview/
├── thirdparty_sources.json          # Manifest: CUTLASS + cuDNN-frontend (git SHA pinned)
├── tools/sync_thirdparty.py         # Clone, checkout, delete_paths, patches, overlays
├── 3rdparty/                        # Synced at build time (NOT committed)
│   ├── cutlass/
│   └── cudnn-frontend/
├── src/                             # ← 64 CUDA files, CURRENTLY COMMITTED (the problem)
│   ├── dit_streaming/kernels/*.cu
│   ├── dit_streaming/pyext/*.cu
│   ├── vae_streaming/*.cu
│   └── ...
└── python/                          # ← 3 Python shims, CURRENTLY COMMITTED
    ├── optimized_dit.py
    ├── cosmos_weights.py
    └── cosmos_fp8_utils.py
```

`sync_thirdparty.py` does a full git clone + `delete_paths` + `patches` + `overlays`. It does NOT
support selective checkout (sparse-checkout or `include_paths`).

The sglang-owned code that should NOT move:
- `native/singleview_loader.py` (JIT build loader, 586 lines)
- `native/acceleration.py` (auto/disabled/required policy)
- `native/__init__.py`
- `runtime/models/dits/omnidreams_fp8.py` (FP8 dispatch wrapper)
- `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py` (pipeline stages)
- All test files, config files, pipeline code

## Proposed Approach

### 1. Add `include_paths` support to `sync_thirdparty.py`

Add a new optional field `include_paths` to the `SourceSpec` dataclass and manifest schema.
When present, use **git sparse-checkout** to only fetch the specified paths instead of cloning
the entire repo.

Implementation:
- `_clone_source()`: after `git clone --filter=blob:none --no-checkout`, run:
  ```
  git sparse-checkout init --cone
  git sparse-checkout set <include_paths...>
  ```
- `_source_from_mapping()`: parse new `include_paths` field (list of relative paths).
- `_source_hash()`: include `include_paths` in the hash for cache invalidation.
- Bump `SCHEMA_VERSION` to 2 (backward-compatible: v1 manifests without `include_paths` still work).

### 2. Add FlashDreams as a managed source

Update `thirdparty_sources.json`:

```json
{
  "name": "flashdreams",
  "repo": "https://github.com/NVIDIA/flashdreams.git",
  "commit": "<pinned SHA from local fork>",
  "directory": "flashdreams",
  "include_paths": [
    "src/dit_streaming/",
    "src/vae_streaming/",
    "src/native_common/",
    "src/native_primitives.cpp",
    "src/native_primitives.h",
    "src/native_primitives_cuda.cu",
    "src/omnidreams_singleview_ext.cpp",
    "python/optimized_dit.py",
    "python/cosmos_weights.py",
    "python/cosmos_fp8_utils.py"
  ],
  "patches": [
    {
      "path": "patches/flashdreams/sglang-adapt.patch",
      "strip": 1
    }
  ]
}
```

### 3. Create sglang adaptation patch

The vendored Python shims have 5 FlashDreams-specific import stubs that need to be adapted for
sglang's module layout. Create `patches/flashdreams/sglang-adapt.patch` to handle these
differences. This patch is applied by `sync_thirdparty.py` after checkout.

Known differences (from `optimized_dit.py`):
- `from flashdreams.core.experimental.kvcache import ...` → sglang's `omnidreams_kvcache.py`
- `from flashdreams.recipes.cosmos.weights import ...` → sglang's `cosmos_weights.py`
- `from flashdreams.recipes.cosmos.fp8_utils import ...` → sglang's `cosmos_fp8_utils.py`

### 4. Update loader paths

`singleview_loader.py` currently compiles sources from:
```
ROOT / "src" / "dit_streaming" / ...
```

After de-vendoring, sources will be at:
```
ROOT / "3rdparty" / "flashdreams" / "src" / "dit_streaming" / ...
```

Update `singleview_loader.py`'s source discovery to look in the new location. The Python shims
(`optimized_dit.py` etc.) need to be importable — add the synced path to `sys.path` or use
a namespace package.

### 5. Remove vendored sources from git

```bash
git rm -r python/sglang/multimodal_gen/native/omnidreams_singleview/src/
git rm python/sglang/multimodal_gen/native/omnidreams_singleview/python/optimized_dit.py
git rm python/sglang/multimodal_gen/native/omnidreams_singleview/python/cosmos_weights.py
git rm python/sglang/multimodal_gen/native/omnidreams_singleview/python/cosmos_fp8_utils.py
```

Update `.gitignore` to exclude `3rdparty/flashdreams/`.

## Migration Plan

| Step | What | Risk |
|------|------|------|
| 1 | Add `include_paths` to `sync_thirdparty.py` | Low — additive, no existing behavior changed |
| 2 | Create `patches/flashdreams/sglang-adapt.patch` | Medium — need to capture all 5 import stubs |
| 3 | Update `thirdparty_sources.json` | Low — manifest change only |
| 4 | Update `singleview_loader.py` source paths | Medium — must not break JIT compilation |
| 5 | Update Python shim imports | Medium — `optimized_dit.py` import path changes |
| 6 | `git rm` vendored sources | Low — mechanical |
| 7 | Verify: `sync_thirdparty.py sync --source flashdreams` | Test on rtx6kd |
| 8 | Verify: JIT compile + E2E sage3_fp8 | Full regression |

## Impact Analysis

- **Repo size**: -36K lines from git history (future commits), ~36K lines removed from working tree
- **Build time**: Adds ~30s for FlashDreams git clone on first sync (subsequent syncs are fast —
  sparse-checkout + fetch only)
- **Offline builds**: `sync_thirdparty.py sync` requires network. Already the case for CUTLASS.
  For air-gapped environments, the synced `3rdparty/flashdreams/` directory can be pre-staged.
- **CI**: No change — CI already runs `sync_thirdparty.py sync` before native extension builds.
- **Upstream updates**: Change `commit` in `thirdparty_sources.json` + regenerate patch. One-line
  change vs 36K-line diff.

## Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| **Git submodule** | Precise version pinning | sglang has no submodule precedent; CI complexity |
| **Move to `3rdparty/`** | Signals third-party | Still in repo, same bloat |
| **Keep as-is** | Zero effort | 36K lines of model-specific CUDA in Python project |
| **Separate package** | Cleanest separation | Overkill for one model; release coordination |

## Design Decisions

1. **`include_paths` semantics**: Supports both directory prefixes (ending with `/`) and exact
   file paths. Directory prefixes use `git sparse-checkout set` which handles subtree matching.
   Example: `"src/dit_streaming/"` includes everything under that directory.

2. **Python shim import path**: Use option (b) — keep a thin re-export shim in the original
   location (`native/omnidreams_singleview/python/optimized_dit.py`) that re-exports from
   `3rdparty/flashdreams/python/optimized_dit.py`. This avoids changing any import sites and
   keeps the shim as a stable API boundary. The shim is ~3 lines per file:
   ```python
   # Re-export from synced FlashDreams source.
   from sglang.multimodal_gen.native.omnidreams_singleview._flashdreams_imports import *
   ```
   The actual import redirection happens in `singleview_loader.py` which adds
   `3rdparty/flashdreams/python/` to `sys.path` at sync time.

## Open Questions

1. **FlashDreams repo structure**: Need to verify the exact directory layout on the public
   `github.com/NVIDIA/flashdreams` repo matches our local fork. The `include_paths` depend on this.
2. **Patch maintenance**: The `sglang-adapt.patch` needs to be regenerated whenever FlashDreams
   changes the import paths. How often does FlashDreams update?
