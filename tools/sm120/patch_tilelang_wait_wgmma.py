#!/usr/bin/env python3
"""
Patch the installed tilelang CUDA template so DSA attention kernels compile on
SM120 (RTX PRO 6000 / Blackwell desktop).

Background:
  With `tl.disable_wgmma` set, tilelang correctly emits the MMA path
  (`tl::mma_sync`) for SM120, BUT the warp-specialized DSA attention kernel still
  emits a `tl::wait_wgmma<N>()` warp-group sync. That symbol is only defined in
  the Hopper template (gemm_sm90.h); `gemm_sm120.h` (used for arch >= 1200) does
  not define it -> nvcc fails with: namespace "tl" has no member "wait_wgmma".

  On the MMA path mma.sync is synchronous, so the warp-group wait is a no-op.
  This script adds a no-op `tl::wait_wgmma` to the installed `gemm_sm120.h`.
  tilelang JIT-compiles kernels with nvcc using these header templates at
  runtime, so no tilelang rebuild is needed -- the next kernel compile picks it
  up.

Idempotent: safe to run repeatedly. Run inside the serving image / on the box.

    python tools/sm120/patch_tilelang_wait_wgmma.py
"""
import os
import sys

SNIPPET = (
    "// [SM120 patch] No Hopper WGMMA on sm_120; mma.sync is synchronous, so the\n"
    "// warp-group wait tilelang still emits is a no-op. Define it so kernels that\n"
    "// mix MMA gemms with a wg_wait compile on sm_120.\n"
    "template <int num_mma> TL_DEVICE void wait_wgmma() {}\n"
)


def main() -> int:
    try:
        import tilelang
    except Exception as e:  # noqa: BLE001
        print("Could not import tilelang:", repr(e))
        return 1

    base = os.path.dirname(tilelang.__file__)
    header = os.path.join(base, "src", "tl_templates", "cuda", "gemm_sm120.h")
    if not os.path.exists(header):
        print("gemm_sm120.h not found at:", header)
        # Some installs ship templates under a different root; try to find it.
        for root, _dirs, files in os.walk(base):
            if "gemm_sm120.h" in files:
                header = os.path.join(root, "gemm_sm120.h")
                print("found instead:", header)
                break
        else:
            return 1

    src = open(header).read()
    if "wait_wgmma" in src:
        print("already patched (wait_wgmma present):", header)
        return 0

    anchor = "namespace tl {"
    if anchor not in src:
        print("unexpected gemm_sm120.h format; could not find 'namespace tl {'")
        return 1

    patched = src.replace(anchor, anchor + "\n" + SNIPPET, 1)
    # Back up once.
    bak = header + ".orig"
    if not os.path.exists(bak):
        open(bak, "w").write(src)
    open(header, "w").write(patched)
    print("patched:", header)
    print("backup :", bak)
    return 0


if __name__ == "__main__":
    sys.exit(main())
