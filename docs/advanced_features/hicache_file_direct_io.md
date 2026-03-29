# HiCacheFile Direct I/O Guide

This document explains how to enable and validate `direct_io` for the `HiCacheFile` storage backend.

## Overview

`HiCacheFile` now supports an optional Direct I/O mode (Linux `O_DIRECT`) through backend extra config.

When enabled, file read/write operations in `HiCacheFile` use `os.open(..., O_DIRECT)` with `os.readv`/`os.writev`.

For high-speed NVMe (especially PCIe Gen5 and above), bypassing page cache is often a better default for sustained write/read bandwidth, and it can also reduce page-cache DRAM pressure so CPU memory can be reserved for other workloads.

### Why use Direct I/O?

- Bypass page cache in high-throughput storage scenarios.
- Reduce page-cache memory pressure so host memory can be used for other tasks.
- Provide an explicit storage mode option for fast NVMe-based deployments.

## Important Behavior

- `direct_io` is **opt-in** and disabled by default.
- This implementation is intentionally strict:
  - no automatic fallback to buffered I/O
  - no proactive alignment handling in code
  - no pre-check that rewrites behavior
- If your filesystem/kernel/buffer does not satisfy Direct I/O requirements, operations fail and logs will tell you to disable `direct_io`.

## Configuration

Pass settings through `--hicache-storage-backend-extra-config`.

Example:

```json
{
  "hicache_storage_pass_prefix_keys": true,
  "direct_io": true,
  "direct_io_alignment": 4096
}
```

Fields:

- `direct_io` (bool): enable/disable Direct I/O for `HiCacheFile`.
- `direct_io_alignment` (int): alignment value for deployment documentation and operator reference (default `4096`).

## Launch Example

```bash
python -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-ratio 1.2 \
  --page-size 64 \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"hicache_storage_pass_prefix_keys": true, "direct_io": true, "direct_io_alignment": 4096}'
```

Recommended cache directory:

```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/nvme0n1/nfsrdma/testhicache
mkdir -p "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"
```

## Validation Checklist

1. Confirm server starts with `direct_io=true`.
2. Send traffic that triggers cache write/read.
3. Check server logs:
   - On failure, you should see `HiCacheFile direct_io read/write failed ...`.
4. Optional syscall-level validation:

```bash
sudo strace -f -e trace=openat -p <SERVER_PID> 2>&1 | rg "O_DIRECT|hicache"
```

If `O_DIRECT` appears in open flags for cache files, direct path is active.

## Benchmark Recommendations

For fair comparison between buffered and direct I/O:

- Keep model, prompt mix, request rate, and concurrency identical.
- Use same cache directory and same storage device.
- Run A/B:
  - A: `"direct_io": false`
  - B: `"direct_io": true`
- Compare at least:
  - throughput
  - p50/p95 latency
  - host memory/page-cache pressure

## Troubleshooting

### `direct_io` fails with `EINVAL`/`ENOTSUP`/similar

Likely reasons:

- filesystem/mount does not support `O_DIRECT`
- buffer/size alignment constraints are not satisfied
- environment/kernel restrictions

Action:

- disable direct I/O in config:

```json
{
  "hicache_storage_pass_prefix_keys": true,
  "direct_io": false
}
```

### Server works with buffered mode but fails with direct mode

This is expected in environments where direct path is unsupported or constrained.
Use buffered mode for compatibility.

## Operational Notes

- Prefer enabling `direct_io` only after validating your target filesystem and workload.
- Keep rollback simple by toggling `direct_io` to `false`.
- Treat `direct_io` as a deployment tuning knob, not a universal default.
