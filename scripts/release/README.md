# Release Scripts

This directory contains scripts to automate version bumping for SGLang releases.

## Scripts

### `upload_zip_to_whl.sh`

Uploads a local ZIP file to a new
[`sgl-project/whl`](https://github.com/sgl-project/whl) GitHub Release and
prints its direct download URL. The ZIP bytes remain in the Release rather than
the Git tree. The script adds the direct link to the flat
[`others/index.html`](https://docs.sglang.io/whl/others/) catalog on the
`gh-pages` branch and adds an `others/` entry to the root index. Existing
package-specific PEP 503 indexes are unchanged.

Prerequisites:

- Install [GitHub CLI](https://cli.github.com/).
- Authenticate once with `gh auth login`.
- Use a GitHub account with write access to `sgl-project/whl`.
- Install Git and Python 3.
- Keep each ZIP file smaller than 2 GiB.

**Usage:**

```bash
scripts/release/upload_zip_to_whl.sh <zip-path> <version> [release-tag] [release-title]
```

For example:

```bash
scripts/release/upload_zip_to_whl.sh ~/Downloads/model-cache.zip v1.2.0
```

This creates the tag and release `zip-v1.2.0`, preserves the filename
`model-cache.zip`, and prints output similar to:

```text
Release: https://github.com/sgl-project/whl/releases/tag/zip-v1.2.0
Asset:   https://github.com/sgl-project/whl/releases/download/zip-v1.2.0/model-cache.zip
SHA256:  <checksum>
Index:   https://docs.sglang.io/whl/others/
wget https://github.com/sgl-project/whl/releases/download/zip-v1.2.0/model-cache.zip
```

The optional third and fourth arguments override the default `zip-<version>` tag
and `ZIP <version>: <filename>` title:

```bash
scripts/release/upload_zip_to_whl.sh archive.zip 20260726 \
  special-build-20260726 "Special build 20260726"
```

The version and tag may contain ASCII letters, digits, `.`, `_`, and `-`. Every
upload must use a new tag. The script never overwrites or deletes a Release,
asset, tag, or index entry. ZIP filenames may contain spaces but not backslashes
or control characters.

The root and `others` index changes are pushed in one commit. If another process
updates `gh-pages` concurrently, the script reclones the latest branch and
retries up to three times.

If Release creation succeeds but index publication fails, rerun the exact same
command. The script resumes only when the existing Release's filename, byte
size, GitHub asset digest, and SHA256 marker match the local file; any mismatch
is treated as a version conflict. After three failed index pushes, use the
`Release` URL printed by the script to inspect the uploaded asset.

### `bump_sglang_version.py`
Updates SGLang version across all relevant files following the pattern from [PR #10468](https://github.com/sgl-project/sglang/pull/10468).

**Usage:**
```bash
python scripts/release/bump_sglang_version.py 0.5.3rc0
```

**Files updated:**
- `Makefile`
- `benchmark/deepseek_v3/README.md`
- `docker/rocm.Dockerfile`
- `docs/docs/get-started/install.mdx`
- `docs/docs/hardware-platforms/amd_gpu.mdx`
- `docs/docs/hardware-platforms/ascend-npus/ascend_npu.mdx`
- `python/pyproject.toml`
- `python/pyproject_other.toml`
- `python/pyproject_npu.toml`
- `python/sglang/version.py`

### `bump_docs_install_version.py`
Bumps the release version pinned in the Mintlify install docs — both the `git clone -b v<version> ...sglang.git` "install from source" line and the version-pinned `lmsysorg/sglang:v<version>` Docker example. Mutable tags (`latest`, `dev`) are intentionally left untouched. Driven automatically on release-tag push by [`.github/workflows/bot-bump-docs-version.yml`](../../.github/workflows/bot-bump-docs-version.yml), which opens a PR with the change.

**Usage:**
```bash
python scripts/release/bump_docs_install_version.py 0.5.13
```

**Files updated:**
- `docs/docs/get-started/install.mdx` (Method 2: From source; Method 3: pinned Docker image)
- `docs/docs/hardware-platforms/amd_gpu.mdx` (Install from Source)

### `bump_kernel_version.py`
Updates the `sglang-kernel` release version across all relevant files following the pattern from [PR #10732](https://github.com/sgl-project/sglang/pull/10732).

**Usage:**
```bash
python scripts/release/bump_kernel_version.py 0.4.0
```

**Files updated:**
- `python/sglang/kernels/aot/pyproject.toml`
- `python/sglang/kernels/aot/pyproject_cpu.toml`
- `python/sglang/kernels/aot/pyproject_rocm.toml`
- `python/sglang/kernels/aot/pyproject_musa.toml`
- `python/sglang/kernels/aot/python/sgl_kernel/version.py`

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
   grep -r "0.5.4rc0" docs/docs/get-started/install.mdx
   ```

4. **Reset changes (if testing):**
   ```bash
   git checkout .
   ```

### Test Kernel Version Bump

1. **Run the script:**
   ```bash
   python scripts/release/bump_kernel_version.py 0.4.0
   ```

2. **Verify changes with git diff:**
   ```bash
   git diff
   ```

3. **Check specific files contain the new version:**
   ```bash
   grep -r "0.4.0" python/sglang/kernels/aot/python/sgl_kernel/version.py
   grep -r "0.4.0" python/sglang/kernels/aot/pyproject.toml
   ```

4. **Reset changes (if testing):**
   ```bash
   git checkout .
   ```

## Version Format Validation

- **SGLang versions:** `X.Y.Z` or `X.Y.ZrcN` (e.g., `0.5.3` or `0.5.3rc0`)
- **Kernel versions:** `X.Y.Z` (e.g., `0.4.0`)

The scripts will validate the version format and exit with an error if invalid.
