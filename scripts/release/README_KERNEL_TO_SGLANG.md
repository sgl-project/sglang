# Bot Bump Kernel Version to SGLang

This workflow automatically syncs the `sgl-kernel` version from `sgl-kernel/pyproject.toml` to SGLang files.

## Workflow: `bot-bump-kernel-version-to-sglang.yml`

### Purpose

When the sgl-kernel version is updated in `sgl-kernel/pyproject.toml`, this workflow ensures that all SGLang files that reference the kernel version are updated accordingly.

### Triggers

1. **Manual trigger**: Can be manually triggered via GitHub Actions UI
2. **Scheduled**: Runs daily at 00:00 UTC to check for version mismatches

### What it does

1. Reads the kernel version from `sgl-kernel/pyproject.toml` (line 11)
2. Checks if it matches the versions in:
   - `python/pyproject.toml` - dependency specification (`"sgl-kernel==x.x.x"`)
   - `python/sglang/srt/entrypoints/engine.py` - version check in `assert_pkg_version()`
   - `docker/Dockerfile` - Docker build argument (`ARG SGL_KERNEL_VERSION=x.x.x`)
3. If any mismatch is detected:
   - Creates a new branch
   - Updates all three files
   - Creates a PR with the changes

### Scripts

- `check_kernel_version_to_sglang.py` - Checks if versions are in sync
- `bump_kernel_version_to_sglang.py` - Updates all SGLang files with kernel version
- `commit_and_pr_kernel_to_sglang.sh` - Commits changes and creates PR

### Manual Usage

To manually check if versions are in sync:

```bash
python scripts/release/check_kernel_version_to_sglang.py
```

To manually bump the version:

```bash
python scripts/release/bump_kernel_version_to_sglang.py
```

### Example

When `sgl-kernel/pyproject.toml` has version `0.3.17` but SGLang files have `0.3.16`, the workflow will:

1. Detect the mismatch
2. Create a branch like `bot/bump-kernel-version-to-sglang-0.3.17-a1b2`
3. Update all three files:
   ```diff
   # python/pyproject.toml
   -  "sgl-kernel==0.3.16",
   +  "sgl-kernel==0.3.17",

   # python/sglang/srt/entrypoints/engine.py
   -            "0.3.16",
   +            "0.3.17",

   # docker/Dockerfile
   -ARG SGL_KERNEL_VERSION=0.3.16
   +ARG SGL_KERNEL_VERSION=0.3.17
   ```
4. Create a PR titled "chore: bump sgl-kernel version to 0.3.17"

## Difference from `bot-bump-kernel-version.yml`

- **`bot-bump-kernel-version.yml`**: Manually bump kernel version in sgl-kernel itself (requires manual input)
- **`bot-bump-kernel-version-to-sglang.yml`**: Automatically sync kernel version FROM sgl-kernel TO SGLang files (no manual input needed)
