r"""Manual ROCm regression for kvcache-ai/Mooncake#4657 through SGLang's wrapper.

Run on an idle AMD GPU with RDMA NICs and a HIP dma-buf-enabled Mooncake build:

    python test/manual/kv_transfer/test_mooncake_hip_memory_accounting.py \
        --ib-device ionic_0
    python test/manual/kv_transfer/test_mooncake_hip_memory_accounting.py \
        --ib-device ionic_0,ionic_1

Uses 60% of the initially available VRAM, so the registered allocation exceeds
the remaining headroom. Requires at least 2 GiB free. On affected drivers the
amdgpu peer-memory client is required for the workaround. Run in a fresh process
for each NIC configuration. This is a registration/lifetime check; run the PD
accuracy suite separately to validate cross-node transfers and generation.

The parent forces RDMA, dma-buf and automatic probing before importing HIP or
Mooncake in the child. It requires a conclusive native probe log, so a build or
kernel that silently falls back to peer-memory registration cannot pass.
"""

import argparse
import os
import subprocess
import sys

MIB = 1 << 20
GIB = 1 << 30
# Match Mooncake's native regression tolerance for KFD granularity/runtime use.
SLACK = 64 * MIB


def run_worker(args):
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("This regression requires a ROCm PyTorch build and AMD GPU")

    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )

    torch.cuda.set_device(args.gpu)
    engine = MooncakeTransferEngine(
        hostname="127.0.0.1", gpu_id=args.gpu, ib_device=args.ib_device
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * GIB:
        raise RuntimeError("Need at least 2 GiB free on an idle GPU")

    # Align both sub-buffer sizes to 2 MiB and retain their shared allocation.
    size = int(free * args.allocation_fraction) // (4 * MIB) * (4 * MIB)
    buffer = torch.empty(size, dtype=torch.uint8, device=f"cuda:{args.gpu}")
    ptrs = [buffer.data_ptr(), buffer.data_ptr() + size // 2]
    lengths = [size // 2, size // 2]
    before, _ = torch.cuda.mem_get_info()
    if size <= before + SLACK or before < 256 * MIB:
        raise RuntimeError("Allocation must exceed remaining headroom with probe space")
    print(
        f"allocation={size / GIB:.3f} GiB, free_before={before / GIB:.3f} GiB",
        flush=True,
    )

    def check_free(stage):
        torch.cuda.synchronize()
        now, _ = torch.cuda.mem_get_info()
        print(f"{stage}: free={now / GIB:.3f} GiB", flush=True)
        if abs(now - before) > SLACK:
            raise AssertionError(
                f"{stage}: free VRAM changed by {(now - before) / MIB:.1f} MiB "
                "while the underlying GPU allocation was still alive"
            )

    # Track every attempted region so a partially failed batch is also cleaned up.
    pending = list(ptrs)
    try:
        rc = engine.batch_register(ptrs, lengths)
        if rc != 0:
            raise RuntimeError(f"Mooncake batch registration failed: {rc}")
        check_free("both sub-buffers registered")

        # Exercise a new device allocation and a kernel launch while the large
        # buffer remains pinned.
        scratch = torch.empty(64 * MIB, dtype=torch.uint8, device=f"cuda:{args.gpu}")
        scratch.fill_(7)
        torch.cuda.synchronize()
        if scratch[0].item() != 7 or scratch[-1].item() != 7:
            raise AssertionError("Post-registration kernel result is incorrect")
        del scratch
        torch.cuda.empty_cache()

        # Kernel loading may retain runtime memory. Compare deregistration with
        # this new baseline instead of attributing those allocations to the MRs.
        before, _ = torch.cuda.mem_get_info()

        # The allocation's anchor must survive until the last sub-buffer goes.
        for ptr in ptrs:
            engine.deregister(ptr)
            pending.remove(ptr)
            check_free(f"{len(pending)} sub-buffers still registered")
    finally:
        # Preserve the original failure, but try to release all attempted MRs
        # before Python releases the tensor's underlying allocation.
        for ptr in pending:
            try:
                engine.deregister(ptr)
            except Exception as exc:
                print(f"Cleanup failed for {ptr:#x}: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ib-device", required=True, help="NIC or comma-separated NICs"
    )
    parser.add_argument("--gpu", type=int, default=0, help="Visible GPU index")
    parser.add_argument("--allocation-fraction", type=float, default=0.6)
    parser.add_argument(
        "--timeout", type=int, default=180, help="Worker timeout in seconds"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0.5 < args.allocation_fraction <= 0.8:
        parser.error("--allocation-fraction must be greater than 0.5 and at most 0.8")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.worker:
        run_worker(args)
        return 0

    env = os.environ.copy()
    env.update(
        MOONCAKE_PROTOCOL="rdma",
        MC_DISABLE_HIP="1",
        MC_FORCE_TCP="0",
        MOONCAKE_DISABLE_HIP_DMABUF="0",
        MC_HIP_KFD_PIN_FIX="1",
        GLOG_logtostderr="1",
        GLOG_minloglevel="0",
        # Use ordinary HIP allocations with stable allocation-base semantics.
        PYTORCH_ALLOC_CONF="backend:native,expandable_segments:False",
    )
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    try:
        result = subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--worker"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.output:
            print(exc.output.decode(errors="replace"))
        print("FAIL: hardware regression timed out", file=sys.stderr)
        return 1
    print(result.stdout, end="")
    if result.returncode:
        return 1
    probe_lines = [
        line for line in result.stdout.splitlines() if "KFD pin probe:" in line
    ]
    if not any(
        "holding one peer-memory pin per registered allocation" in line
        or "the driver counts it once, no anchor needed" in line
        for line in probe_lines
    ):
        print(
            "FAIL: no conclusive HIP dma-buf probe; check the build/driver",
            file=sys.stderr,
        )
        return 1
    if "amdgpu peer-memory pin unavailable" in result.stdout:
        print("FAIL: required KFD pin anchor was unavailable", file=sys.stderr)
        return 1
    print("PASS: HIP dma-buf accounting, sub-buffer lifetime and kernel launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
