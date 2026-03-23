# rename_wheels — server / home follow-up testing

**Branch for this work:** `fix/20953-kernel-wheel-repack`

Use this checklist on a Linux box or GPU server when you want coverage beyond what a Windows laptop + fake wheels can do.

## Already covered locally (no server required)

- `bash sgl-kernel/test_rename_wheels.sh` — core scenarios + 10× idempotency per case.
- `bash sgl-kernel/test_rename_wheels_extra.sh` — multi-wheel, aarch64 **fake** wheel, odd version strings (`.post1`, `.dev…`, `0.4.1.post2`), optional real wheel via `REAL_WHEEL=/path/to/repacked.whl`.
- Real ~327 MB wheel from GitHub releases: rename `+cu130` out of the **filename** into `dist/`, run `rename_wheels.sh` multiple times; expect `Skipping … already suffixed` and stable `manylinux2014` (no `manymanylinux`).

**Requirements:** bash, Python with `pip install wheel`, and on Windows/Git Bash a real Python (e.g. conda) on `PATH` — see comments at the top of each script.

---

## Run on a server / at home (recommended extras)

### 1. uv + local index (matches reviewer flow)

After repacking, serve wheels and install:

```bash
mkdir -p whl/cu130/sglang-kernel
cp sgl-kernel/dist/*.whl whl/cu130/sglang-kernel/
cd whl   # parent of cu130/
python3 -m http.server 9000
```

In another shell:

```bash
uv pip install sglang-kernel --index-url http://HOST:9000/whl/cu130/
```

Expect: `Installed … sglang-kernel==0.4.0+cu130` (version must match filename + METADATA).

### 2. Post-install import (Linux + GPU stack)

On a machine with a matching CUDA stack:

```bash
python3 -c "import sgl_kernel; print('ok')"
```

This checks **runtime** (e.g. `libnvrtc`, SM variant). Fixing #20953 only guarantees **pip metadata consistency**; import can still fail for unrelated env reasons.

### 3. Same as CI: Docker CUDA image + real `/usr/local/cuda-*`

Build or use `lmsysorg/sglang:…-cu130` (or your pipeline image), run the kernel build that ends with `rename_wheels.sh` **without** `SGL_KERNEL_CUDA_SUFFIX_OVERRIDE`, and confirm `detect_cuda_suffix()` picks the right `+cu*`.

### 4. ROCm

In an image with `/opt/rocm-*` (or `SGL_KERNEL_ROCM_SUFFIX_OVERRIDE`), run:

```bash
bash 3rdparty/amd/wheel/sgl-kernel/rename_wheels_rocm.sh
```

on real ROCm-built wheels in `dist/`.

### 5. Real aarch64 (optional)

If you ship `linux_aarch64` / `manylinux2014_aarch64` wheels, repeat rename + install on **ARM64** Linux.

### 6. MUSA (optional)

If your team maintains `scripts/ci/musa/rename_wheels_musa.sh`, run it against **real MUSA CI artifacts**, not only bash smoke on fake wheels.

---

## What stays in upstream CI

Full matrix (every CUDA/ROCm/Python combo), release signing, and long GPU jobs are expected to run in project CI — not required for every local PR iteration.

---

## Issue context

- [#20953](https://github.com/sgl-project/sglang/issues/20953) — filename local version (`+cu130`) vs `METADATA` / `WHEEL` mismatch.
- Follow-up: idempotency — second run must not corrupt `manylinux` in the **filename** or **WHEEL** `Tag:` lines.
