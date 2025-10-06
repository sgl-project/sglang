# Release Scripts

This directory contains scripts to automate version bumping for SGLang releases.

## Scripts

### `bump_sglang_version.py`
Updates SGLang version across all relevant files following the pattern from [PR #10468](https://github.com/sgl-project/sglang/pull/10468).

**Usage:**
```bash
python scripts/release/bump_sglang_version.py 0.5.3rc0
```

**Files updated:**
- `Makefile`
- `benchmark/deepseek_v3/README.md`
- `docker/Dockerfile.rocm`
- `docs/get_started/install.md`
- `docs/platforms/amd_gpu.md`
- `docs/platforms/ascend_npu.md`
- `python/pyproject.toml`
- `python/pyproject_other.toml`
- `python/sglang/version.py`

### `bump_kernel_version.py`
Updates sgl-kernel version across all relevant files following the pattern from [PR #10732](https://github.com/sgl-project/sglang/pull/10732).

**Usage:**
```bash
python scripts/release/bump_kernel_version.py 0.3.12
```

**Files updated:**
- `sgl-kernel/pyproject.toml`
- `sgl-kernel/pyproject_cpu.toml`
- `sgl-kernel/pyproject_rocm.toml`
- `sgl-kernel/python/sgl_kernel/version.py`

## Manual Testing Instructions

### Test SGLang Version Bump

1. **Run the script:**
   ```bash
   python scripts/release/bump_sglang_version.py 0.5.4rc0
   ```

2. **Verify changes with git diff:**
   ```bash
   git diff
   ```

3. **Check specific files contain the new version:**
   ```bash
   grep -r "0.5.4rc0" python/sglang/version.py
   grep -r "0.5.4rc0" python/pyproject.toml
   grep -r "0.5.4rc0" docs/get_started/install.md
   ```

4. **Reset changes (if testing):**
   ```bash
   git checkout .
   ```

### Test Kernel Version Bump

1. **Run the script:**
   ```bash
   python scripts/release/bump_kernel_version.py 0.3.13
   ```

2. **Verify changes with git diff:**
   ```bash
   git diff
   ```

3. **Check specific files contain the new version:**
   ```bash
   grep -r "0.3.13" sgl-kernel/python/sgl_kernel/version.py
   grep -r "0.3.13" sgl-kernel/pyproject.toml
   ```

4. **Reset changes (if testing):**
   ```bash
   git checkout .
   ```

## Version Format Validation

- **SGLang versions:** `X.Y.Z` or `X.Y.ZrcN` (e.g., `0.5.3` or `0.5.3rc0`)
- **Kernel versions:** `X.Y.Z` (e.g., `0.3.12`)

The scripts will validate the version format and exit with an error if invalid.
