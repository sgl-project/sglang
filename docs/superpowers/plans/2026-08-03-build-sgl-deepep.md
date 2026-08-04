# Standalone DeepEP Wheel Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and fully validate a non-Docker script that installs DeepEP build dependencies and produces a pip-installable wheel on x86_64/aarch64 with CUDA 12/13.

**Architecture:** A single sourceable Bash script contains small platform-selection and CUDA-patching functions plus a strict `main` orchestration path. Pytest invokes the sourceable functions for deterministic local coverage; real package installation, compilation, wheel installation, and SGLang model validation run in four RX devbox image/platform cells.

**Tech Stack:** Bash, apt/dpkg, GDRCopy v2.5.1, Python setuptools/wheel, PyTorch CUDA extensions, pytest, RX devboxes.

## Global Constraints

- The script is `scripts/build_sgl_deepep.sh [OUTPUT_DIR]` and must not start Docker.
- The default output directory is `$PWD/dist`.
- Clone only `https://github.com/sgl-project/DeepEP.git`.
- x86_64 with CUDA 12/13 uses `sgl-deepep-x86`.
- aarch64 with CUDA 13 uses `sgl-deepep-arm`.
- aarch64 with CUDA 12 uses `sgl-deepep-cu12-arm`.
- Every build uses `TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3'`.
- CUDA 13 builds add `${CUDA_HOME}/include/cccl` to `setup.py`.
- The script builds a wheel but does not install the newly built wheel.
- End-to-end acceptance requires `TestDSV4FlashFP4B200Balanced` to pass in all four `(cu12, cu13) x (x86, arm)` cells.

---

### Task 1: Platform and CUDA source helpers

**Files:**
- Create: `test/srt/test_build_sgl_deepep_script.py`
- Create: `scripts/build_sgl_deepep.sh`

**Interfaces:**
- Produces: `select_deepep_branch ARCH CUDA_MAJOR`, printing one branch or returning nonzero.
- Produces: `parse_cuda_major NVCC_OUTPUT`, printing `12` or `13` or returning nonzero.
- Produces: `patch_cuda13_cccl SETUP_PY CUDA_HOME`, adding exactly one CCCL include line.

- [ ] **Step 1: Write failing helper tests**

Create parametrized pytest coverage for these exact mappings:

```python
@pytest.mark.parametrize(
    ("arch", "cuda", "branch"),
    [
        ("x86_64", "12", "sgl-deepep-x86"),
        ("x86_64", "13", "sgl-deepep-x86"),
        ("aarch64", "12", "sgl-deepep-cu12-arm"),
        ("aarch64", "13", "sgl-deepep-arm"),
    ],
)
def test_select_deepep_branch(arch, cuda, branch):
    result = call_bash_function("select_deepep_branch", arch, cuda)
    assert result.stdout.strip() == branch
```

Also test unsupported architecture/CUDA, CUDA release strings for 12.9 and
13.0, CCCL insertion, duplicate insertion avoidance, and missing CCCL failure.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest -q test/srt/test_build_sgl_deepep_script.py`

Expected: FAIL because `scripts/build_sgl_deepep.sh` or its functions do not exist.

- [ ] **Step 3: Implement minimal sourceable helpers**

Use `set -euo pipefail`, define the three interfaces above, and guard `main`:

```bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
```

Use a checked Python edit for CCCL insertion so a changed upstream `setup.py`
fails explicitly rather than silently producing an invalid CUDA 13 build.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest -q test/srt/test_build_sgl_deepep_script.py`

Expected: all helper tests pass.

### Task 2: Dependency installation and wheel orchestration

**Files:**
- Modify: `test/srt/test_build_sgl_deepep_script.py`
- Modify: `scripts/build_sgl_deepep.sh`

**Interfaces:**
- Consumes: the helpers from Task 1.
- Produces: `install_apt_packages PACKAGE...`, with installed-package fallback.
- Produces: `install_gdrcopy`, installing packages built from GDRCopy v2.5.1.
- Produces: `build_deepep SOURCE_DIR OUTPUT_DIR`, producing one `deep_ep-*.whl`.
- Produces: `main [OUTPUT_DIR]`, orchestrating validation, cleanup, install, clone, patch, and build.

- [ ] **Step 1: Write failing orchestration tests**

Add a real minimal setuptools project whose `setup.py` rejects any
`TORCH_CUDA_ARCH_LIST` other than `9.0;10.0;10.3`. Source the Bash script and
call `build_deepep` with the current Python interpreter; assert that exactly one
installable `deep_ep-*.whl` appears in the requested output directory. Add
separate tests for zero/multiple-wheel rejection and build-failure propagation.

System dependency and GDRCopy behavior is intentionally exercised on the real
CUDA devboxes instead of asserting calls made to fake package managers.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest -q test/srt/test_build_sgl_deepep_script.py`

Expected: the new orchestration tests fail because the functions are absent.

- [ ] **Step 3: Implement dependency and build flow**

Copy the package groups and fallback semantics from
`scripts/ci/cuda/ci_install_deepep.sh`, pin GDRCopy to v2.5.1, use a `mktemp -d`
workspace with a cleanup trap, uninstall existing `deep_ep`, clone with
`--branch "$DEEPEP_BRANCH" --depth 1`, and invoke:

```bash
TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' MAX_JOBS="$MAX_JOBS" \
    "$PYTHON_BIN" setup.py bdist_wheel -d "$OUTPUT_DIR"
```

Keep the current PyTorch/SGLang installation intact and print the final absolute
wheel path.

- [ ] **Step 4: Run local verification**

Run:

```bash
bash -n scripts/build_sgl_deepep.sh
python3 -m pytest -q test/srt/test_build_sgl_deepep_script.py
git diff --check
```

Run ShellCheck too if it is installed. Expected: every available check exits 0.

- [ ] **Step 5: Commit implementation**

Stage only the script, tests, spec, and plan. Commit with:

```bash
git commit -m "build: add standalone DeepEP wheel builder"
```

### Task 3: Four-cell RX build and model validation

**Files:**
- Modify only if a devbox failure produces a reproducible script bug.

**Interfaces:**
- Consumes: `scripts/build_sgl_deepep.sh` and the two requested SGLang images.
- Produces: logs and one passing result for each matrix cell.

- [ ] **Step 1: Acquire two devboxes**

Acquire one 4x B200 devbox and one 4x GB300 devbox with a sufficient TTL. Start
with `lmsysorg/sglang:latest-cu129`; do not release either devbox after testing.

- [ ] **Step 2: Validate x86_64 + CUDA 12**

Sync the current SGLang branch, uninstall `deep_ep`, remove prior source/GDRCopy
build artifacts, run the builder, install the only generated wheel, verify
`import deep_ep`, and run:

```bash
python3 -m unittest \
  test.registered.models_e2e.test_deepseek_v4_flash_fp4_b200.TestDSV4FlashFP4B200Balanced
```

- [ ] **Step 3: Validate aarch64 + CUDA 12**

Repeat Step 2 on the GB300 devbox. Record architecture, `nvcc --version`, branch
and wheel filename in the log.

- [ ] **Step 4: Reprovision both boxes to CUDA 13**

Reprovision both existing boxes with `lmsysorg/sglang:latest-cu130`, wait until
ready, update the SGLang checkout to the tested branch, and reinstall repository
dependencies before the GB300 job as required by the GB300 journal.

- [ ] **Step 5: Validate x86_64 + CUDA 13 and aarch64 + CUDA 13**

Repeat the clean/build/install/import/test procedure on both boxes. Confirm the
build logs show the CCCL path and the expected branch in each cell.

- [ ] **Step 6: Debug failures with red-green coverage**

For each script defect, add a local failing pytest that reproduces it, observe
RED, patch the script, observe GREEN, resync, and rerun the failing matrix cell.
Do not accept a cell based only on wheel construction or import success.

### Task 4: Final verification and draft PR

**Files:**
- Verify the final intended diff only.

**Interfaces:**
- Consumes: passing local checks and four passing devbox cells.
- Produces: a pushed branch and draft PR against `sgl-project/sglang:main`.

- [ ] **Step 1: Run fresh final checks**

Run local Bash syntax, pytest, available ShellCheck/pre-commit checks, and
`git diff --check`. Re-read the design requirements against the final script and
matrix logs.

- [ ] **Step 2: Confirm PR scope and commit fixes**

Inspect `git status -sb`, `git diff`, and the commit range against `origin/main`.
Stage no unrelated files. Commit any post-devbox fixes tersely.

- [ ] **Step 3: Push and open a draft PR**

Push `codex/build-sgl-deepep` to the authenticated fork if direct origin push is
not appropriate, then open a draft PR targeting `sgl-project/sglang:main`. The
body must list the branch-selection matrix, dependency behavior, wheel output,
all four exact image/platform test results, and local checks.
