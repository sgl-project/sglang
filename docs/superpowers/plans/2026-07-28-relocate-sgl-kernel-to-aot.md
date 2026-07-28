# Relocate `sgl-kernel` to `sglang.kernels.aot` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the independently built `sglang-kernel` AOT project from the repository root to `python/sglang/kernels/aot/` without changing its published package, imports, operators, or CI coverage.

**Architecture:** Preserve the entire AOT project as a nested, independently buildable project and update only repository-path consumers. Explicitly exclude the nested project from the main `sglang` package discovery, wheel data, and source distribution, then validate both packaging systems independently.

**Tech Stack:** Git, setuptools/setuptools-scm, scikit-build-core, CMake, CUDA, Docker, GitHub Actions, Python, Bash, TOML, YAML

---

## File Structure

- Move: `sgl-kernel/` → `python/sglang/kernels/aot/`
- Modify: `python/pyproject.toml` — exclude the AOT project from main-package discovery and wheel data.
- Create: `python/MANIFEST.in` — prune the AOT project from the main `sglang` source distribution.
- Modify: `.github/workflows/_pr-test-check-changes.yml` — classify AOT changes without JIT/main-package fanout.
- Modify: `.github/workflows/_pr-test-sgl-kernel-build.yml` — build and upload from the new AOT root.
- Modify: `.github/workflows/pr-test-sgl-kernel.yml` — download artifacts and run AOT tests from the new root.
- Modify: `.github/workflows/pr-test-jit-kernel.yml` — consume freshly built AOT wheels from the new root.
- Modify: `.github/workflows/release-whl-kernel.yml` — release all AOT variants from the new root.
- Modify: platform workflows under `.github/workflows/` — update only checkout paths, build contexts, artifacts, and AOT filters.
- Modify: scripts under `scripts/ci/`, `scripts/release/`, `scripts/ci_monitor/`, and `scripts/update_kernel_whl_index.py` — resolve the new checkout path.
- Modify: `docker/arm64.Dockerfile`, `docker/rocm.Dockerfile`, `docker/xeon.Dockerfile` — build the AOT project at the new checkout path.
- Modify: `.github/CODEOWNERS`, `.github/labeler.yml`, `.gitignore` — preserve ownership, labeling, and generated-file exclusions.
- Modify: `.claude/skills/add-jit-kernel/SKILL.md`, `.claude/skills/add-sgl-kernel/SKILL.md`, relevant `docs_new/` pages, and `python/sglang/srt/hardware_backend/mlx/aot.py` — update source-tree instructions and links only.

### Task 1: Move the AOT project and isolate main-package artifacts

**Files:**
- Move: `sgl-kernel/` → `python/sglang/kernels/aot/`
- Modify: `python/pyproject.toml`
- Create: `python/MANIFEST.in`
- Modify: `python/sglang/kernels/aot/pyproject.toml`
- Modify: `python/sglang/kernels/aot/pyproject_cpu.toml`
- Modify: `python/sglang/kernels/aot/pyproject_musa.toml`
- Modify: `python/sglang/kernels/aot/pyproject_rocm.toml`
- Modify: `python/sglang/kernels/aot/README.md`

- [ ] **Step 1: Record the source manifest**

Run:

```bash
git ls-files sgl-kernel | sed 's#^sgl-kernel/##' | sort > /tmp/sgl-kernel-before.txt
wc -l /tmp/sgl-kernel-before.txt
```

Expected: `316` tracked files.

- [ ] **Step 2: Move the complete project**

Run:

```bash
git mv sgl-kernel python/sglang/kernels/aot
git ls-files python/sglang/kernels/aot | sed 's#^python/sglang/kernels/aot/##' | sort > /tmp/sgl-kernel-after.txt
diff -u /tmp/sgl-kernel-before.txt /tmp/sgl-kernel-after.txt
```

Expected: `diff` exits successfully with no output.

- [ ] **Step 3: Exclude AOT packages and data from the main wheel**

Add the following entry to the existing
`[tool.setuptools.packages.find].exclude` list in `python/pyproject.toml`:

```toml
  "sglang.kernels.aot*",
```

Add:

```toml
[tool.setuptools.exclude-package-data]
"sglang" = [
  "kernels/aot/*",
  "kernels/aot/**/*",
]
```

Add the same `sglang/kernels/aot*` path exclusion to
`[tool.wheel].exclude`.

- [ ] **Step 4: Exclude AOT sources from the main source distribution**

Create `python/MANIFEST.in` with:

```text
prune sglang/kernels/aot
```

- [ ] **Step 5: Update repository URLs inside AOT package metadata**

Change each repository-tree URL:

```text
https://github.com/sgl-project/sglang/tree/main/sgl-kernel
```

to:

```text
https://github.com/sgl-project/sglang/tree/main/python/sglang/kernels/aot
```

Do not change distribution names, `wheel.packages`, versions, imports, CMake
targets, extension names, or Docker-internal `/sgl-kernel` paths.

- [ ] **Step 6: Commit the move and packaging boundary**

Run:

```bash
git add python/pyproject.toml python/MANIFEST.in python/sglang/kernels/aot
git diff --cached --check
git commit -m "refactor: move sgl-kernel under kernels aot"
```

Expected: one commit whose bulk is detected as renames.

### Task 2: Preserve CI change detection and primary CUDA jobs

**Files:**
- Modify: `.github/workflows/_pr-test-check-changes.yml`
- Modify: `.github/workflows/_pr-test-sgl-kernel-build.yml`
- Modify: `.github/workflows/pr-test-sgl-kernel.yml`
- Modify: `.github/workflows/pr-test-jit-kernel.yml`
- Modify: `.github/workflows/pr-test-multimodal-gen.yml`
- Modify: `.github/workflows/pr-test-stage.yml`
- Modify: `.github/workflows/pr-test.yml`
- Modify: `.github/workflows/pr-test-extra.yml`
- Modify: `.github/workflows/lint.yml`

- [ ] **Step 1: Point the dedicated AOT filter at the new project**

Use this AOT source pattern everywhere a component filter currently uses
`sgl-kernel/**`:

```yaml
- "python/sglang/kernels/aot/**/!(*.md|THIRDPARTYNOTICES.txt|LICENSE)"
```

- [ ] **Step 2: Prevent broad main and JIT filters from claiming AOT-only changes**

Append this negative pattern after each broad
`python/sglang/kernels/**` pattern used for JIT detection:

```yaml
- "!python/sglang/kernels/aot/**"
```

Append the same negative pattern after the broad `python/sglang/**` main-package
pattern. Preserve existing workflow-trigger paths for shared CI files.

- [ ] **Step 3: Update CUDA wheel build and artifact paths**

Use:

```yaml
run: |
  cd python/sglang/kernels/aot
  ./build.sh "${{ matrix.python-version }}" "${{ matrix.cuda-version }}"
```

and:

```yaml
path: python/sglang/kernels/aot/dist/*
```

for build uploads. Update cleanup, download, listing, and pytest paths in
consumer workflows to the same AOT root.

- [ ] **Step 4: Keep component names stable**

Leave workflow names, job IDs such as `sgl-kernel-build-wheels`, inputs such as
`sgl_kernel`, artifact names, and user-facing check names unchanged. They name
the published component, not the old checkout directory.

- [ ] **Step 5: Commit the primary CI migration**

Run:

```bash
git add .github/workflows/_pr-test-check-changes.yml \
  .github/workflows/_pr-test-sgl-kernel-build.yml \
  .github/workflows/pr-test-sgl-kernel.yml \
  .github/workflows/pr-test-jit-kernel.yml \
  .github/workflows/pr-test-multimodal-gen.yml \
  .github/workflows/_pr-test-stage.yml \
  .github/workflows/pr-test.yml \
  .github/workflows/pr-test-extra.yml \
  .github/workflows/lint.yml
git diff --cached --check
git commit -m "ci: update AOT kernel build paths"
```

Expected: AOT checkout paths change while stable component names remain.

### Task 3: Update platform and release workflows

**Files:**
- Modify: `.github/workflows/pr-test-amd.yml`
- Modify: `.github/workflows/pr-test-amd-rocm720.yml`
- Modify: `.github/workflows/pr-test-amd-extra.yml`
- Modify: `.github/workflows/pr-test-arm64.yml`
- Modify: `.github/workflows/pr-test-mlx.yml`
- Modify: `.github/workflows/pr-test-musa.yml`
- Modify: `.github/workflows/pr-test-xeon.yml`
- Modify: `.github/workflows/pr-test-xpu.yml`
- Modify: `.github/workflows/nightly-test-musa.yml`
- Modify: `.github/workflows/diffusion-ci-gt-gen.yml`
- Modify: `.github/workflows/nightly-72-gpu-gb200.yml`
- Modify: `.github/workflows/release-whl-kernel.yml`

- [ ] **Step 1: Update platform-specific AOT filters and test roots**

Change repository checkout paths to:

```text
python/sglang/kernels/aot
```

and container checkout paths to:

```text
/sglang-checkout/python/sglang/kernels/aot
```

Keep package uninstall commands such as `pip uninstall sgl-kernel` unchanged.

- [ ] **Step 2: Update release build, version, and artifact paths**

Use:

```text
python/sglang/kernels/aot/python/sgl_kernel/version.py
python/sglang/kernels/aot/dist/
```

for release version reads and artifacts. Use
`working-directory: python/sglang/kernels/aot` for AOT-local commands. Preserve
Docker-internal `/sgl-kernel` paths and release tag formats.

- [ ] **Step 3: Validate workflow syntax**

Run:

```bash
python - <<'PY'
from pathlib import Path
import yaml

for path in sorted(Path(".github/workflows").glob("*.yml")):
    yaml.safe_load(path.read_text())
print("workflow YAML parsed")
PY
```

Expected: `workflow YAML parsed`.

- [ ] **Step 4: Commit platform and release workflow paths**

Run:

```bash
git add .github/workflows
git diff --cached --check
git commit -m "ci: relocate platform AOT kernel paths"
```

Expected: platform paths change without package/API renames.

### Task 4: Update build, release, Docker, and automation consumers

**Files:**
- Modify: `scripts/ci/amd/amd_ci_install_dependency.sh`
- Modify: `scripts/ci/cuda/ci_install_dependency.sh`
- Modify: `scripts/ci/musa/musa_install_dependency.sh`
- Modify: `scripts/ci/musa/rename_wheels_musa.sh`
- Modify: `scripts/ci/npu/npu_log_print.sh`
- Modify: `scripts/ci/utils/slash_command_handler.py`
- Modify: `scripts/ci_monitor/ci_auto_bisect.py`
- Modify: `scripts/release/README.md`
- Modify: `scripts/release/bump_kernel_version.py`
- Modify: `scripts/release/bump_kernel_version_to_sglang.py`
- Modify: `scripts/release/check_kernel_version_to_sglang.py`
- Modify: `scripts/release/commit_and_pr_kernel_to_sglang.sh`
- Modify: `scripts/update_kernel_whl_index.py`
- Modify: `docker/arm64.Dockerfile`
- Modify: `docker/rocm.Dockerfile`
- Modify: `docker/xeon.Dockerfile`

- [ ] **Step 1: Update scripts that resolve paths from the repository root**

Define or use this relative root consistently:

```text
python/sglang/kernels/aot
```

For PR file detection, replace:

```python
f.filename.startswith("sgl-kernel/")
```

with:

```python
f.filename.startswith("python/sglang/kernels/aot/")
```

Do not change package-index names, dependency strings, or external
`sgl-kernel-{npu,xpu}` repository names.

- [ ] **Step 2: Update Docker checkout navigation**

From repository-root Docker contexts, use:

```dockerfile
cd python/sglang/kernels/aot
```

Preserve `/sgl-kernel` as the internal work directory in the AOT project's own
Dockerfile and in standalone wheel-builder containers.

- [ ] **Step 3: Run script syntax and version-tool smoke tests**

Run:

```bash
bash -n scripts/ci/amd/amd_ci_install_dependency.sh
bash -n scripts/ci/cuda/ci_install_dependency.sh
bash -n scripts/ci/musa/musa_install_dependency.sh
bash -n scripts/release/commit_and_pr_kernel_to_sglang.sh
python -m compileall -q scripts/release scripts/ci/utils scripts/ci_monitor
python scripts/release/check_kernel_version_to_sglang.py
```

Expected: shell parsing and Python compilation succeed; the version check reads
the nested AOT pyproject rather than reporting it missing.

- [ ] **Step 4: Commit automation path changes**

Run:

```bash
git add scripts docker/arm64.Dockerfile docker/rocm.Dockerfile docker/xeon.Dockerfile
git diff --cached --check
git commit -m "build: update AOT kernel source paths"
```

Expected: build consumers resolve the nested project.

### Task 5: Update ownership and source-tree documentation

**Files:**
- Modify: `.github/CODEOWNERS`
- Modify: `.github/labeler.yml`
- Modify: `.gitignore`
- Modify: `.claude/skills/add-jit-kernel/SKILL.md`
- Modify: `.claude/skills/add-sgl-kernel/SKILL.md`
- Modify: `.claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md`
- Modify: `.claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py`
- Modify: `docs_new/docs/developer_guide/contribution_guide.mdx`
- Modify: `docs_new/docs/hardware-platforms/apple_metal.mdx`
- Modify: `python/sglang/srt/hardware_backend/mlx/aot.py`
- Modify: `test/registered/cpu/test_spec_kernels.py`
- Modify: `test/registered/kernels/ops/attention/test_hadamard_jit.py`
- Modify: `test/registered/kernels/ops/moe/test_renorm.py`

- [ ] **Step 1: Preserve ownership, labels, and generated-file ignores**

Change only old repository-root patterns to:

```text
python/sglang/kernels/aot/
```

Keep the existing owner teams and label names.

- [ ] **Step 2: Update source-checkout instructions and test paths**

Replace commands and links that identify the old checkout directory with the
new AOT root. Keep explanatory references to the `sglang-kernel` package and
`sgl_kernel` module unchanged.

- [ ] **Step 3: Scan and classify residual literals**

Run:

```bash
rg -n --hidden --glob '!.git/**' --glob '!docs/superpowers/**' 'sgl-kernel'
```

Expected residuals are stable distribution names, component/job names,
cache/container-internal names, or external repositories. No residual may
refer to the removed top-level checkout path.

- [ ] **Step 4: Commit metadata and documentation paths**

Run:

```bash
git add .github/CODEOWNERS .github/labeler.yml .gitignore .claude \
  docs_new/docs/developer_guide/contribution_guide.mdx \
  docs_new/docs/hardware-platforms/apple_metal.mdx \
  python/sglang/srt/hardware_backend/mlx/aot.py \
  test/registered/cpu/test_spec_kernels.py \
  test/registered/kernels/ops/attention/test_hadamard_jit.py \
  test/registered/kernels/ops/moe/test_renorm.py
git diff --cached --check
git commit -m "docs: update AOT kernel source location"
```

Expected: no public package or import rename.

### Task 6: Validate packaging and path completeness locally

**Files:**
- Test: `python/pyproject.toml`
- Test: `python/MANIFEST.in`
- Test: `python/sglang/kernels/aot/pyproject.toml`
- Test: all modified workflows and scripts

- [ ] **Step 1: Verify the move manifest and source integrity**

Run:

```bash
git ls-tree -r --name-only origin/main:sgl-kernel | sort > /tmp/sgl-kernel-origin.txt
git ls-files python/sglang/kernels/aot | sed 's#^python/sglang/kernels/aot/##' | sort > /tmp/sgl-kernel-branch.txt
diff -u /tmp/sgl-kernel-origin.txt /tmp/sgl-kernel-branch.txt
```

Expected: only deliberately added or removed path-local metadata is explained;
all original AOT source, tests, benchmarks, and legal files are present.

- [ ] **Step 2: Build and inspect the main package archives**

Run from `python/`:

```bash
python -m build --wheel --sdist
python - <<'PY'
from pathlib import Path
import tarfile
import zipfile

bad = []
for archive in Path("dist").iterdir():
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as tf:
            names = tf.getnames()
    else:
        continue
    hits = [name for name in names if "sglang/kernels/aot/" in name]
    bad.extend((archive.name, name) for name in hits)
assert not bad, bad
print("main package excludes kernels/aot")
PY
```

Expected: `main package excludes kernels/aot`.

- [ ] **Step 3: Validate AOT metadata from its new root**

Run from `python/sglang/kernels/aot/`:

```bash
python -m build --sdist
python - <<'PY'
from pathlib import Path
import tarfile

archive = next(Path("dist").glob("sglang_kernel-0.4.5.tar.gz"))
with tarfile.open(archive) as tf:
    names = tf.getnames()
assert any(name.endswith("/python/sgl_kernel/__init__.py") for name in names)
print(archive.name)
PY
```

Expected: `sglang_kernel-0.4.5.tar.gz`.

- [ ] **Step 4: Run repository checks**

Run:

```bash
git diff origin/main...HEAD --check
pre-commit run --from-ref origin/main --to-ref HEAD
```

Expected: all applicable checks pass.

### Task 7: Validate CUDA on H200 and publish the PR

**Files:**
- Test: `python/sglang/kernels/aot/`
- Publish: branch `bbuf/relocate-sgl-kernel-to-aot`

- [ ] **Step 1: Reproduce the branch on the H200 checkout**

Fetch the branch in the prepared H200 SGLang environment and check out the
exact local commit SHA. Confirm `nvidia-smi` reports H200 GPUs before starting
the build.

- [ ] **Step 2: Build and install the CUDA wheel from the new root**

Run:

```bash
cd python/sglang/kernels/aot
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps dist/*.whl
```

Expected: wheel build and installation succeed without referring to the old
repository path.

- [ ] **Step 3: Verify metadata, imports, and representative AOT tests**

Run:

```bash
python - <<'PY'
from importlib.metadata import distribution
import sgl_kernel

dist = distribution("sglang-kernel")
assert dist.version == "0.4.5"
print(dist.metadata["Name"], dist.version, sgl_kernel.__file__)
PY
pytest -q tests/test_activation.py tests/test_norm.py tests/test_topk.py
```

Expected: metadata reports `sglang-kernel 0.4.5`; import and selected H200 tests
pass.

- [ ] **Step 4: Push and create a draft PR**

Run:

```bash
git status --short
git push -u origin bbuf/relocate-sgl-kernel-to-aot
```

Create the PR with:

- title: `[Kernel] Move sgl-kernel under sglang.kernels.aot`
- base: `main`
- head: `bbuf/relocate-sgl-kernel-to-aot`
- state: draft
- body: summarize the physical relocation, compatibility guarantees,
  packaging isolation, CI/release path updates, and local/H200 validation.

- [ ] **Step 5: Inspect checks and labels**

Confirm that GitHub recognizes the change as AOT-kernel work, that the expected
wheel build/tests are selected, and that no unrelated label or check was added
because of stale path filters.
