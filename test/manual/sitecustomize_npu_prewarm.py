"""H3 discriminator + mitigation: pre-map the activation arena before serving.

Copy (or symlink) this file as ``sitecustomize.py`` inside an empty directory
and prepend that directory to PYTHONPATH when launching the prefill server on
A5. Python auto-imports sitecustomize in every process of the server; this
module wraps ``torch.npu.set_device`` so the prewarm runs once per process,
on the device sglang selects, before any model load or forward:

  mkdir -p /tmp/prewarm && cp test/manual/sitecustomize_npu_prewarm.py \
      /tmp/prewarm/sitecustomize.py
  PYTHONPATH=/tmp/prewarm ./scripts/start_prefill_host_rdma.sh

Mechanism under test (docs/a5_pd_mf.md H3): every recorded crash has the
faulting hc_pre x block born in a caching-allocator segment grown milliseconds
earlier, inside the first big-T forward of a cold arena. If the fault is a
lazy-page-mapping race on fresh segment pages (driver maps on first touch
while AIV kernels are already in flight), forcing the arena to grow
single-threaded at startup — allocate PREWARM_BYTES in grow-sized chunks,
zero each (maps the pages), free (the caching allocator retains the segment,
so serving reuses already-mapped blocks instead of growing fresh ones) —
should stop the crash. If the crash persists unchanged, H3 loses weight and
H1/H2 (in-package tiling cross-write / sibling-op OOB) gain.

 verdict: crash gone   -> H3 effectively confirmed + a usable workaround
 verdict: crash stays  -> H3 demoted; run the H1/H2 discriminators next
"""

import os

# Match the growth burst sizes seen in the traces (192 MB segments, 16/64 MB
# tensor blocks) and stay small enough not to eat the KV budget.
PREWARM_BYTES = int(os.environ.get("PREWARM_BYTES", 6 * 1024**3))
CHUNK_BYTES = 192 * 1024**2

_installed = False


def _install():
    global _installed
    if _installed or os.environ.get("SGLANG_DEBUG_SKIP_PREWARM"):
        return
    _installed = True
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        return

    orig_set_device = torch.npu.set_device
    done = []

    def set_device_and_prewarm(device):
        orig_set_device(device)
        if done:
            return
        done.append(True)
        idx = device if isinstance(device, int) else int(device)
        total = torch.npu.get_device_properties(idx).total_memory
        budget = min(PREWARM_BYTES, max(0, int(total * 0.06)))
        n_chunks = budget // CHUNK_BYTES
        print(
            f"[npu.prewarm] device {idx}: mapping {budget >> 20} MB in "
            f"{n_chunks} x {CHUNK_BYTES >> 20} MB chunks + one small-block pass",
            flush=True,
        )
        blocks = []
        try:
            # Segment-sized chunks: forces arena growth in the same size
            # class the crash-time growth uses.
            for _ in range(n_chunks):
                t = torch.empty(CHUNK_BYTES, dtype=torch.uint8, device="npu")
                t.zero_()  # first touch: maps the pages now, not mid-forward
                blocks.append(t)
            # Small-block pass: fragments the fresh segments so the 16/32/64
            # MB activation blocks have recycled homes, as in steady state.
            for size in (64, 32, 16, 8, 4, 2, 1):
                for _ in range(4):
                    t = torch.empty(size * 1024**2, dtype=torch.uint8, device="npu")
                    t.zero_()
                    blocks.append(t)
            # Free but do NOT empty_cache: the allocator keeps the segments
            # mapped for serving to reuse.
            del blocks
        except RuntimeError as exc:
            print(f"[npu.prewarm] stopped early: {exc}", flush=True)
        torch.npu.synchronize()
        print("[npu.prewarm] done", flush=True)

    torch.npu.set_device = set_device_and_prewarm


_install()
