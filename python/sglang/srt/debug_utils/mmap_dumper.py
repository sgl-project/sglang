from __future__ import annotations

import json
import mmap
import os
from typing import Any, Optional

import torch


_META_CAPACITY = 64 * 1024


class MmapDumper:
    def __init__(self, dump_dir: Optional[str] = None) -> None:
        self._dump_dir = dump_dir
        self._pid = os.getpid()
        self._scalars: dict = {}
        self._tensor_meta: dict = {}
        self._tensor_mmaps: dict = {}
        self._meta_mmap = None
        if dump_dir:
            self._activate(dump_dir)

    def set_dir(self, dump_dir: str) -> None:
        if self._dump_dir == dump_dir:
            return
        self._activate(dump_dir)

    def is_active(self) -> bool:
        return self._dump_dir is not None and self._meta_mmap is not None

    def dump(self, items: dict) -> None:
        if not self.is_active():
            return
        import time

        t0 = time.perf_counter()
        for name, value in items.items():
            if isinstance(value, torch.Tensor):
                self._dump_tensor(name, value)
            else:
                self._scalars[name] = _jsonify(value)
        self._flush_meta()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[MmapDumper pid={self._pid}] dumped {len(items)} items "
            f"in {elapsed_ms:.2f} ms",
            flush=True,
        )

    def _activate(self, dump_dir: str) -> None:
        os.makedirs(dump_dir, exist_ok=True)
        self._dump_dir = dump_dir
        path = os.path.join(dump_dir, f"pid{self._pid}_meta.json.mmap")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(fd, _META_CAPACITY)
        self._meta_mmap = mmap.mmap(
            fd, _META_CAPACITY, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )

    def _dump_tensor(self, name: str, tensor: torch.Tensor) -> None:
        import numpy as np

        cpu_tensor = tensor.detach().cpu().contiguous()
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
        alloc_bytes = max(nbytes, 1)

        entry = self._tensor_mmaps.get(name)
        bin_path = os.path.join(self._dump_dir, f"pid{self._pid}_{name}.bin")
        if entry is None or entry["capacity"] < alloc_bytes:
            if entry is not None:
                entry["mmap"].close()
                os.close(entry["fd"])
            capacity = max(alloc_bytes * 2, 4096)
            fd = os.open(bin_path, os.O_RDWR | os.O_CREAT, 0o644)
            os.ftruncate(fd, capacity)
            mm = mmap.mmap(
                fd, capacity, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
            )
            entry = {"fd": fd, "mmap": mm, "capacity": capacity}
            self._tensor_mmaps[name] = entry

        if nbytes > 0:
            src = cpu_tensor.numpy().reshape(-1).view(np.uint8)
            dst = np.frombuffer(entry["mmap"], dtype=np.uint8, count=nbytes)
            np.copyto(dst, src)

        self._tensor_meta[name] = {
            "shape": list(cpu_tensor.shape),
            "stride": list(cpu_tensor.stride()),
            "dtype": str(cpu_tensor.dtype),
            "nbytes": nbytes,
            "bin_filename": os.path.basename(bin_path),
        }

    def _flush_meta(self) -> None:
        meta = {"pid": self._pid, "scalars": self._scalars, "tensors": self._tensor_meta}
        payload = json.dumps(meta).encode("utf-8")
        n = len(payload)
        assert n + 4 <= _META_CAPACITY, f"mmap dumper meta too big: {n}"
        self._meta_mmap[0:4] = n.to_bytes(4, "little")
        self._meta_mmap[4 : 4 + n] = payload


def _jsonify(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return repr(value)


_TORCH_DTYPE_TO_TORCH = {
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
    "torch.bool": torch.bool,
}


def read_dump(dump_dir: str, pid: int) -> dict:
    """
    Load a dump produced by `MmapDumper`.

    Returns:
        {
          "scalars": {name: jsonable_value, ...},
          "tensors": {name: torch.Tensor (cpu), ...},
        }
    """
    meta_path = os.path.join(dump_dir, f"pid{pid}_meta.json.mmap")
    with open(meta_path, "rb") as f:
        n = int.from_bytes(f.read(4), "little")
        meta = json.loads(f.read(n).decode("utf-8"))

    tensors = {}
    for name, info in meta["tensors"].items():
        torch_dtype = _TORCH_DTYPE_TO_TORCH[info["dtype"]]
        if info["nbytes"] == 0:
            tensors[name] = torch.empty(info["shape"], dtype=torch_dtype)
            continue
        bin_path = os.path.join(dump_dir, info["bin_filename"])
        elem_size = torch.empty((), dtype=torch_dtype).element_size()
        n_elem = info["nbytes"] // elem_size
        with open(bin_path, "rb") as f:
            buf = f.read(info["nbytes"])
        flat = torch.frombuffer(bytearray(buf), dtype=torch_dtype, count=n_elem)
        tensors[name] = flat.reshape(info["shape"])
    return {"scalars": meta["scalars"], "tensors": tensors}


def list_dump_pids(dump_dir: str) -> list:
    """Return all pids that have a dump in `dump_dir`."""
    pids = []
    for fn in os.listdir(dump_dir):
        if fn.startswith("pid") and fn.endswith("_meta.json.mmap"):
            pids.append(int(fn[len("pid") : -len("_meta.json.mmap")]))
    return sorted(pids)


def _tester() -> None:
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="mmap_dumper_test_")
    print(f"[tester] dir = {tmp_dir}")

    # ----- Test 1: scalars -----
    d = MmapDumper(tmp_dir)
    assert d.is_active()
    d.dump({"a": 1, "b": True, "c": "hello", "d": None, "e": [1, 2, 3]})
    out = read_dump(tmp_dir, os.getpid())
    assert out["scalars"] == {"a": 1, "b": True, "c": "hello", "d": None, "e": [1, 2, 3]}, out
    assert out["tensors"] == {}
    print("[tester] T1 scalars OK")

    # ----- Test 2: tensors of different dtypes -----
    t_i32 = torch.arange(10, dtype=torch.int32)
    t_i64 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    t_f32 = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    d.dump({"t_i32": t_i32, "t_i64": t_i64, "t_f32": t_f32})
    out = read_dump(tmp_dir, os.getpid())
    assert torch.equal(out["tensors"]["t_i32"], t_i32)
    assert torch.equal(out["tensors"]["t_i64"], t_i64)
    assert torch.equal(out["tensors"]["t_f32"], t_f32)
    print("[tester] T2 tensors OK")

    # ----- Test 3: re-dump with smaller tensor (capacity reuse) -----
    d.dump({"t_i32": torch.arange(3, dtype=torch.int32)})
    out = read_dump(tmp_dir, os.getpid())
    assert torch.equal(out["tensors"]["t_i32"], torch.arange(3, dtype=torch.int32))
    print("[tester] T3 shrink OK")

    # ----- Test 4: re-dump with larger tensor (mmap grow) -----
    big = torch.arange(10000, dtype=torch.int64)
    d.dump({"t_i32": big})  # different dtype, much larger
    out = read_dump(tmp_dir, os.getpid())
    assert torch.equal(out["tensors"]["t_i32"], big)
    print("[tester] T4 grow OK")

    # ----- Test 5: scalars + tensors mixed, multiple flushes -----
    d.dump({"counter": 1})
    d.dump({"counter": 2, "x": torch.zeros(5, dtype=torch.int32)})
    out = read_dump(tmp_dir, os.getpid())
    assert out["scalars"]["counter"] == 2, out
    assert torch.equal(out["tensors"]["x"], torch.zeros(5, dtype=torch.int32))
    print("[tester] T5 mixed flushes OK")

    # ----- Test 6: empty tensor -----
    d.dump({"empty": torch.zeros(0, dtype=torch.int32)})
    out = read_dump(tmp_dir, os.getpid())
    assert out["tensors"]["empty"].shape == (0,)
    print("[tester] T6 empty OK")

    # ----- Test 7: inactive dumper is a no-op -----
    d2 = MmapDumper(None)
    assert not d2.is_active()
    d2.dump({"foo": 1})  # should not crash
    print("[tester] T7 inactive no-op OK")

    # ----- Test 8: auto mkdir for missing nested dir -----
    nested = os.path.join(tmp_dir, "a", "b", "c")
    d3 = MmapDumper(nested)
    assert os.path.isdir(nested)
    d3.dump({"v": 42})
    print("[tester] T8 auto mkdir OK")

    # ----- Test 9: list_dump_pids -----
    pids = list_dump_pids(tmp_dir)
    assert pids == [os.getpid()], f"expected only this pid, got {pids}"
    print("[tester] T9 list_dump_pids OK")

    shutil.rmtree(tmp_dir)
    print("[tester] all OK")


def _bench_one_rank(rank: int, dump_dir: str, shape: tuple, n_iters: int, queue) -> None:
    """Run inside a child process: alloc GPU tensor, dump N times, report timings."""
    import time

    import torch

    n_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % n_gpus)
    try:
        t = torch.zeros(shape, dtype=torch.int32, device="cuda")
    except torch.cuda.OutOfMemoryError as e:
        queue.put((rank, "OOM", str(e)[:200]))
        return
    torch.cuda.synchronize()
    nbytes = t.numel() * t.element_size()

    d = MmapDumper(os.path.join(dump_dir, f"rank{rank}"))
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        d.dump({"req_to_token_partial": t})
        times.append(time.perf_counter() - t0)
    queue.put((rank, "OK", times, nbytes))


def _bench_speed(dump_dir_root: str = "/dev/shm/mmap_dumper_bench") -> None:
    import multiprocessing as mp
    import shutil

    import torch

    if not torch.cuda.is_available():
        print("[bench] no CUDA, skip")
        return

    n_gpus = torch.cuda.device_count()
    n_ranks = min(8, n_gpus)
    print(f"[bench] using {n_ranks} ranks (GPUs available: {n_gpus})")
    print(f"[bench] dump dir root = {dump_dir_root}")

    shutil.rmtree(dump_dir_root, ignore_errors=True)
    os.makedirs(dump_dir_root, exist_ok=True)

    try:
        _bench_speed_inner(dump_dir_root, n_ranks)
    finally:
        shutil.rmtree(dump_dir_root, ignore_errors=True)
        print(f"[bench] cleaned {dump_dir_root}")
    print("[bench] done")


def _bench_speed_inner(dump_dir_root: str, n_ranks: int) -> None:
    import multiprocessing as mp
    import shutil

    # Default: only run SMALL (the new fast path).
    # Set SGLANG_DUMP_BENCH_BIG=1 to also run BIG (old slow path) for comparison.
    big = os.environ.get("SGLANG_DUMP_BENCH_BIG", "0") not in ("", "0", "false", "False")
    scenarios = [
        ("SMALL (4608, 10000) ~ 184 MB", (4608, 10000), 5),
    ]
    if big:
        scenarios.insert(0, ("BIG (4608, 1048580) ~ 18 GB", (4608, 1048580), 3))

    ctx = mp.get_context("spawn")
    for label, shape, n_iters in scenarios:
        nbytes = shape[0] * shape[1] * 4
        print(f"\n=== {label} ===")
        if dump_dir_root.startswith("/host/data") and nbytes * n_ranks > 130 * 2**30:
            print(f"  skip: would write {nbytes * n_ranks / 2**30:.1f} GB to /host/data (~93% full)")
            continue

        # Concurrent: 8 ranks at once
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_bench_one_rank,
                args=(r, os.path.join(dump_dir_root, "concurrent"), shape, n_iters, queue),
            )
            for r in range(n_ranks)
        ]
        import time as _time

        wall_t0 = _time.perf_counter()
        for p in procs:
            p.start()
        results = [queue.get() for _ in range(n_ranks)]
        for p in procs:
            p.join()
        wall_total = _time.perf_counter() - wall_t0

        results.sort(key=lambda x: x[0])
        all_oom = all(r[1] == "OOM" for r in results)
        if all_oom:
            print("  all ranks OOM, skip")
            continue
        print(
            f"  CONCURRENT 8-rank wall total = {wall_total*1000:.0f} ms "
            f"(includes process spawn + GPU init + tensor alloc)"
        )
        for rank, status, *rest in results:
            if status == "OOM":
                print(f"    rank {rank}: OOM ({rest[0]})")
                continue
            times, nb = rest
            t_min = min(times) * 1000
            t_med = sorted(times)[len(times) // 2] * 1000
            t_max = max(times) * 1000
            mb = nb / 2**20
            print(
                f"    rank {rank}: dump nbytes={mb:.0f} MB, "
                f"per-call min={t_min:.1f} ms, median={t_med:.1f} ms, max={t_max:.1f} ms"
            )

        shutil.rmtree(os.path.join(dump_dir_root, "concurrent"), ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        dump_dir = sys.argv[2] if len(sys.argv) > 2 else "/dev/shm/mmap_dumper_bench"
        _bench_speed(dump_dir)
    else:
        _tester()
