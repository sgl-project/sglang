# Relocate the AOT Kernel Project under `sglang.kernels`

**Date:** 2026-07-28

**Status:** Approved for implementation

## Context

SGLang's unified kernel namespace now lives under `python/sglang/kernels/`.
JIT implementations are already grouped in `kernels/jit`, while the
independently built AOT CUDA/C++ project still lives at the repository root as
`sgl-kernel/`. This separates two implementation backends that are presented
through the same `sglang.kernels` operator layer and makes kernel ownership,
navigation, and future cleanup less coherent.

The AOT project is not just source code. It is an independently versioned and
published wheel with CUDA, ROCm, CPU, MUSA, and Metal build variants, dedicated
tests and benchmarks, release automation, Docker build contexts, and CI change
detection. The relocation must preserve those behaviors.

## Goals

- Move the complete `sgl-kernel/` project to
  `python/sglang/kernels/aot/`.
- Make `aot/`, `jit/`, and `ops/` the top-level kernel categories under
  `python/sglang/kernels/`.
- Keep the AOT project independently buildable and publishable.
- Update every repository-path consumer, including CI, release automation,
  platform scripts, ownership, labels, and developer documentation.
- Preserve existing CI selection behavior: an AOT-only change should run AOT
  jobs without accidentally becoming a JIT or general Python-package change.
- Ensure the main `sglang` wheel and source distribution do not absorb the
  nested AOT project.

## Non-goals and Compatibility Contract

This change does not rename or merge public artifacts. It preserves:

- the PyPI distribution name `sglang-kernel`;
- the Python import namespace `sgl_kernel`;
- existing C++/CUDA extension names, Torch operator namespaces, ABI, and API;
- wheel versions, wheel file naming, and platform-specific build behavior;
- AOT source layout relative to the AOT project root;
- runtime imports and kernel dispatch behavior.

No compatibility symlink or duplicate top-level `sgl-kernel/` directory will
remain. External scripts that build directly from a repository checkout must
adopt the new source path, but installed-package consumers are unaffected.

## Selected Structure

The complete project moves without internal source reclassification:

```text
python/sglang/kernels/
├── aot/
│   ├── benchmark/
│   ├── cmake/
│   ├── csrc/
│   ├── include/
│   ├── python/sgl_kernel/
│   ├── tests/
│   ├── CMakeLists.txt
│   ├── Dockerfile
│   ├── Makefile
│   ├── pyproject.toml
│   └── pyproject_{cpu,musa,rocm}.toml
├── jit/
└── ops/
```

Keeping the AOT tree intact minimizes functional risk and preserves all
relative paths used by CMake, scikit-build-core, setuptools, tests, and
benchmarks.

## Packaging Isolation

Nesting the AOT project under the `sglang` source tree introduces two packaging
hazards:

1. setuptools package discovery may see nested Python directories below
   `sglang.kernels.aot`;
2. the main package's broad `kernels/**/*` package-data rule may copy AOT
   sources into the `sglang` wheel.

The main `python/pyproject.toml` and, if required by the validated setuptools
behavior, its source-manifest configuration will explicitly exclude
`sglang.kernels.aot*` from package discovery and exclude
`kernels/aot/**` from `sglang` package data. The exact patterns must be
validated against locally built wheel and source archives, not accepted from
configuration inspection alone.

The AOT pyproject files continue to package `python/sgl_kernel` relative to the
new AOT root. Their distribution names and versions remain unchanged. Project
URLs that identify a repository path will point to the new location.

## Repository-path Migration

The migration will distinguish repository paths from stable names.

Repository-path references will be updated in:

- GitHub Actions build, test, nightly, and release workflows;
- AOT change detection and runner selection;
- artifact upload/download paths;
- Docker build contexts and source-copy commands;
- CUDA, ROCm, CPU, MUSA, NPU, XPU, MLX, and Metal scripts where applicable;
- kernel version bump, synchronization, and wheel-index tooling;
- `CODEOWNERS`, labeler configuration, and `.gitignore`;
- tests, developer skills, and documentation that instruct users to enter or
  reference the source directory.

Stable identifiers will not be mechanically renamed. Examples include:

- the `sglang-kernel` distribution requirement;
- `import sgl_kernel`;
- workflow and job display names;
- Docker-internal work directories such as `/sgl-kernel`;
- cache keys and cache directory names;
- labels and environment variables whose meaning is the AOT component rather
  than a checkout path.

Every remaining literal `sgl-kernel` occurrence after the move will be
classified rather than removed through a global replacement.

## CI Change-detection Semantics

The AOT filter will change from `sgl-kernel/**` to
`python/sglang/kernels/aot/**`, retaining the existing exclusions for
documentation and legal files.

Because the destination is currently covered by broad main-package and JIT
kernel globs, those filters will gain explicit AOT exclusions. The intended
matrix is:

| Changed path | AOT jobs | JIT jobs | General Python jobs |
| --- | --- | --- | --- |
| `kernels/aot/**` | yes | no | no |
| `kernels/jit/**` | no | yes | as currently defined |
| `kernels/ops/**` | no | yes | as currently defined |
| shared workflow/build logic | according to its existing component contract | according to its existing component contract | according to its existing component contract |

The filter order and negative-pattern syntax will be checked against the
semantics supported by `dorny/paths-filter`.

## Migration Mechanics

The implementation will:

1. move the tracked AOT project as one tree;
2. add the main-wheel packaging exclusions;
3. update build, test, release, platform, ownership, and documentation paths;
4. avoid source-level kernel changes in the same PR;
5. preserve file contents wherever a path edit is not required.

Git similarity detection should present the bulk of the PR as renames. A
manifest comparison will verify that every tracked file from the old tree has
one destination and that no AOT file is lost.

## Validation

Validation is layered so path coverage is checked before expensive GPU work:

1. Compare pre- and post-move file manifests and counts.
2. Scan for old path forms and manually classify remaining `sgl-kernel`
   literals.
3. Parse and lint modified workflow, TOML, Python, shell, and CMake files.
4. Build the main `sglang` wheel and source distribution, then inspect both
   archives to prove that no `sglang/kernels/aot/` source is included.
5. Build AOT package metadata and platform-independent artifacts from the new
   project root.
6. Build the CUDA AOT wheel on an H200 environment, install it, verify
   `import sgl_kernel`, and run the applicable AOT unit tests.
7. Inspect the resulting wheel name and metadata to confirm that
   `sglang-kernel` and its version are unchanged.
8. Review the final diff for unintended source or API changes and inspect PR
   change detection/check selection after publication.

Platform workflows that cannot be executed locally will receive static path
validation and rely on their existing CI runners.

## Risks and Mitigations

- **Main package size regression:** explicit discovery, package-data, and
  source-manifest exclusions where needed, followed by wheel and source-archive
  inspection.
- **AOT CI silently skipped:** update the dedicated filter and confirm PR check
  selection.
- **Unnecessary CI fanout:** exclude `aot/**` from broad JIT and main-package
  filters.
- **Broken release automation:** enumerate repository-path consumers and
  validate version and wheel scripts from the new working directory.
- **Platform-specific stale paths:** scan all workflows and scripts, including
  non-CUDA variants, rather than only the primary CUDA jobs.
- **Accidental public rename:** compare wheel metadata and run import smoke
  tests.

## Rollback

The PR is a path-only structural change plus path-consumer updates. If a
blocking integration issue appears before merge, the branch can be reverted as
one change without data migration or published-artifact changes. No transition
state or compatibility shim needs separate cleanup.
