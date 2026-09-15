"""Check that EIC is correctly wired up behind sglang.

Exercises the same client sglang uses (EICKVClient) against the real cluster
named by remote-eic.yaml: write, read-back, existence, overwrite, delete, and
eviction. Every key is prefixed with a unique run id and deleted at the end.

    python scripts/eic_integration_check.py
    python scripts/eic_integration_check.py --page-bytes $((512*1024)) --flood-gib 8
"""

import argparse
import os
import sys
import time
import traceback
import uuid

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import eic

from sglang.srt.mem_cache.eic_memory_pool import EICKVClient, get_eic_config_file_path

FAILURES = []
PROBE = None


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


class Probe:
    def __init__(self, page_bytes, dtype=torch.bfloat16):
        numel = page_bytes // dtype.itemsize
        assert numel > 0, "page-bytes smaller than one element"
        self.shape = (numel,)
        self.dtype = dtype
        self.page_bytes = numel * dtype.itemsize
        self.run_id = uuid.uuid4().hex[:12]
        self.written = set()
        self.client = EICKVClient(dtype, self.shape, None, device="cpu")

    def key(self, i):
        return f"eic_integration_check/{self.run_id}/{i}"

    def page(self, seed):
        # Deterministic content so a read-back can be compared bit for bit.
        g = torch.Generator().manual_seed(seed)
        return torch.randint(-128, 127, self.shape, generator=g, dtype=torch.int16).to(
            self.dtype
        )

    def write(self, keys, pages):
        # Pages are built by the caller so that generating them stays out of
        # any throughput measurement.
        ok = self.client.set(keys, pages)
        self.written.update(keys)
        return ok

    def read(self, keys):
        return self.client.batch_get(keys)

    def delete(self, keys):
        keys_vec = eic.StringVector()
        for k in keys:
            keys_vec.append(k)
        option = eic.DelOption()
        option.ns = self.client.eic_namespace
        return self.client.connection.mdel(keys_vec, option)[0]

    def cleanup(self):
        keys = sorted(self.written)
        for i in range(0, len(keys), 256):
            self.delete(keys[i : i + 256])
        print(f"cleaned up {len(keys)} keys")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--page-bytes", type=int, default=1 << 20)
    p.add_argument("--pages", type=int, default=32, help="pages per probe batch")
    p.add_argument(
        "--flood-gib",
        type=float,
        default=0,
        help="write this much fresh data to force eviction, then recheck the probe keys",
    )
    args = p.parse_args()
    if args.pages < 2:
        p.error("--pages must be at least 2, or the mixed-batch check is vacuous")

    cfg = get_eic_config_file_path()
    if not check("config file present", os.path.exists(cfg), cfg):
        return 1

    global PROBE
    t0 = time.perf_counter()
    probe = PROBE = Probe(args.page_bytes)
    check(
        "client init",
        True,
        f"ns={probe.client.eic_namespace or '<default>'} "
        f"page={probe.page_bytes}B  {time.perf_counter() - t0:.1f}s",
    )

    n = args.pages
    keys = [probe.key(i) for i in range(n)]

    check("absent key reports miss", not any(probe.client.exists_batch(keys)))

    pages = [probe.page(i) for i in range(n)]
    t = time.perf_counter()
    wrote = probe.write(keys, pages)
    dt = time.perf_counter() - t
    check(
        "write",
        wrote,
        f"{n} pages  {n * probe.page_bytes / dt / 2**30:.2f} GiB/s",
    )

    check("exists after write", all(probe.client.exists_batch(keys)))

    t = time.perf_counter()
    vals, mask = probe.read(keys)
    dt = time.perf_counter() - t
    check(
        "read",
        all(mask),
        f"{n} pages  {n * probe.page_bytes / dt / 2**30:.2f} GiB/s",
    )
    if vals is not None:
        bad = [i for i in range(n) if not torch.equal(vals[i].cpu(), probe.page(i))]
        check("read-back is byte-identical", not bad, f"mismatched={bad[:4]}")

    # A batch mixing live and never-written keys must report per-key truth, not
    # collapse to all-fail.
    ghosts = [probe.key(f"ghost{i}") for i in range(n // 2)]
    _, mixed = probe.read(keys[: n // 2] + ghosts)
    check(
        "mixed batch reports per-key hits",
        mixed[: n // 2] == [True] * (n // 2)
        and mixed[n // 2 :] == [False] * len(ghosts),
        f"hits={sum(mixed)}/{len(mixed)}",
    )
    if hasattr(probe.client, "missing_keys"):
        check(
            "a read miss suppresses the next exists",
            not any(probe.client.exists_batch(ghosts)),
        )

    # Overwrite: the same key must come back with the new content.
    probe.write(keys[:1], [probe.page(10**6)])
    vals, mask = probe.read(keys[:1])
    check(
        "overwrite wins",
        mask[0] and torch.equal(vals[0].cpu(), probe.page(10**6)),
    )

    # Delete: exists and get must agree that it is gone. Ask exists first --
    # a read marks the key missing locally, which would answer for the server.
    victim = keys[:4]
    probe.delete(victim)
    gone = not any(probe.client.exists_batch(victim))
    _, mask = probe.read(victim)
    check("delete removes the value", gone and not any(mask))

    if args.flood_gib > 0:
        flood_pages = int(args.flood_gib * 2**30 / probe.page_bytes)
        t = time.perf_counter()
        gen = 0.0
        for i in range(0, flood_pages, n):
            batch = min(n, flood_pages - i)
            fkeys = [probe.key(f"flood{i + j}") for j in range(batch)]
            g = time.perf_counter()
            fpages = [probe.page(10**7 + i + j) for j in range(batch)]
            gen += time.perf_counter() - g
            probe.write(fkeys, fpages)
        t = time.perf_counter() - t - gen
        print(
            f"flooded {flood_pages} pages ({args.flood_gib} GiB) "
            f"in {t:.1f}s  {flood_pages * probe.page_bytes / t / 2**30:.2f} GiB/s"
        )

        survivors = keys[4:]
        exists = probe.client.exists_batch(survivors)
        _, mask = probe.read(survivors)
        # Eviction is allowed to take anything; what is not allowed is exists
        # claiming a hit the read cannot deliver.
        phantoms = [k for k, e, m in zip(survivors, exists, mask) if e and not m]
        check(
            "eviction keeps exists and get consistent",
            not phantoms,
            f"survived={sum(mask)}/{len(survivors)} phantom={len(phantoms)}",
        )

    return 1 if FAILURES else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        # Runs on the failure paths too: keys live in a shared namespace, and
        # EICKVClient's write thread is not a daemon and blocks on its queue
        # forever, so a normal exit would hang joining it.
        if PROBE is not None:
            PROBE.cleanup()
        print(
            "\nall checks passed"
            if code == 0
            else f"\n{len(FAILURES) or 'aborted'} failed"
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
