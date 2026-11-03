"""Experimental remote Engram reader for the opt-in SGLang integration."""

import atexit
import base64
import hashlib
import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.srt.engram_remote.owner_control import OwnerClient


@contextmanager
def _reader_device_scope(mapping, gpu):
    """Select this visible GPU's NIC only while creating its native contexts."""
    key = "GPUNETIO_DEVICE_API_RDMA"
    previous = os.environ.get(key)
    if mapping is None:
        yield previous
        return
    devices = [device.strip() for device in mapping.split(",")]
    if any(not device for device in devices) or not 0 <= gpu < len(devices):
        raise ValueError("reader device map must name every visible GPU NIC")
    os.environ[key] = devices[gpu]
    try:
        yield devices[gpu]
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


_quarantined = []


class ReaderRuntime:
    def __init__(self):
        config = json.loads(Path(os.environ["ENGRAM_SHARED_READY"]).read_text())
        self.client = OwnerClient(
            config["host"],
            config["port"],
            config["checkpoint_id"],
            config["generation"],
        )
        self.lease = self.client.attach()
        self.context = None
        self.closed = False
        try:
            self._prepare()
            if os.environ.get("ENGRAM_SHARED_CONTROLLED_DRAIN") == "1":
                if signal.getsignal(signal.SIGUSR2) != signal.SIG_DFL:
                    raise RuntimeError(
                        "refuse to overwrite an existing SIGUSR2 handler"
                    )
                signal.signal(signal.SIGUSR2, self._controlled_drain)
        except BaseException:
            try:
                if self.context is not None:
                    self.context.close()
                self.lease.release()
            except BaseException:
                # A failed device retirement must retain scratch and its lease.
                _quarantined.append(self)
            raise
        atexit.register(self.close)
        print(
            json.dumps(
                {
                    "event": "shared-engram-reader-ready",
                    "pid": os.getpid(),
                    "gpu": torch.cuda.current_device(),
                    "scratch_bytes": self.arena.numel(),
                    "registered_table_bytes": sum(self.metadata["lengths"]),
                    "reader_full_table_replica_bytes": 0,
                }
            ),
            flush=True,
        )

    def _prepare(self):
        md = self.lease.metadata
        self.metadata = md
        self.storage_layout = md.get("storage_layout")
        if self.storage_layout != "row264":
            raise ValueError("remote Engram requires row264 storage")
        # Validate actual serving checkpoint, not merely a same-shape table.
        model = Path(os.environ["ENGRAM_SHARED_MODEL"])
        for name, field in (
            ("config.json", "config_sha256"),
            ("model.safetensors.index.json", "index_sha256"),
        ):
            if hashlib.sha256((model / name).read_bytes()).hexdigest() != md[field]:
                raise ValueError(
                    "shared owner/serving checkpoint identity mismatch: " + name
                )
        cfg = json.loads((model / "config.json").read_text())
        cfg = cfg.get("text_config", cfg)
        rows = cfg["engram_num_embeddings"]
        specs = md.get("table_specs", ())
        if len(specs) != 2 or tuple(md["lengths"]) != tuple(int(n) * 264 for n in rows):
            raise ValueError("row264 MR count/length mismatch")
        expected = {
            "weight": {"offset": 0, "bytes_per_row": 256, "dtype": "F8_E4M3"},
            "scale": {"offset": 256, "bytes_per_row": 8, "dtype": "F8_E8M0"},
        }
        if any(
            s.get("layer") != layer or s.get("component_layout") != expected
            for s, layer in zip(specs, (1, 14))
        ):
            raise ValueError("row264 component layout mismatch")
        self.max_queries = int(os.environ.get("ENGRAM_SHARED_MAX_QUERIES", "196608"))
        self.prefetch_enabled = os.environ.get("ENGRAM_SHARED_PREFETCH") == "1"
        if (
            not self.prefetch_enabled
            or os.environ.get("ENGRAM_SHARED_READ_PLAN") != "1"
        ):
            raise ValueError("remote Engram requires prefetch and a prepared READ plan")
        self.p_off = 64
        self.s_off = self.p_off + self.max_queries * 256 + 64
        self.dump = (self.s_off + self.max_queries * 8 + 127) // 64 * 64
        gpu = torch.cuda.current_device()
        self.span = self.dump + 64
        self.arena = torch.empty(
            self.span * (2 if self.prefetch_enabled else 1),
            dtype=torch.uint8,
            device="cuda",
        )
        self.layer_stats = torch.zeros(
            (2 if self.prefetch_enabled else 1, 8), dtype=torch.uint64, device="cuda"
        )
        self.ranges = {}
        self.prefetch_ids = None
        self.prefetch_issued = set()
        self.defer_layer14 = os.environ.get("ENGRAM_SHARED_DEFER_LAYER14") == "1"
        self.forward_cp_expected = False
        self.comm_stream = torch.cuda.Stream(device=gpu)
        self.ready_events = {layer: torch.cuda.Event() for layer in (1, 14)}
        self.consumed_events = {layer: torch.cuda.Event() for layer in (1, 14)}
        # A separate base per model instance prevents same-node endpoint reuse.
        port = int(os.environ["ENGRAM_SHARED_READER_PORT_BASE"]) + gpu * max(
            4, int(os.environ.get("ENGRAM_SHARED_READ_LANES", "1"))
        )
        lanes = int(os.environ.get("ENGRAM_SHARED_READ_LANES", "1"))
        from sglang.srt.engram_remote.read_plan import ReadPlan

        graph_sizes = tuple(
            int(n)
            for n in os.environ.get("ENGRAM_SHARED_READ_GRAPH_SIZES", "49152").split(
                ","
            )
        )
        if not graph_sizes or any(n <= 0 or n > self.max_queries for n in graph_sizes):
            raise ValueError("invalid planned READ graph capacity")
        with _reader_device_scope(
            os.environ.get("ENGRAM_SHARED_READER_DEVICES"), gpu
        ) as self.reader_device:
            self.context = ReadPlan(
                os.environ["ENGRAM_SHARED_LIBRARY"],
                gpu,
                port,
                lanes,
                self.max_queries,
                graph_sizes,
            )
        self.context.register([self.arena], host=False)
        self.context.connect_raw(
            base64.b64decode(md["native_md"], validate=True),
            md["addresses"],
            md["lengths"],
            md["owner_gpu"],
        )

    def fetch(self, indices, layer, lo, hi):
        if self.closed:
            raise RuntimeError(
                "shared Engram reader drained; new admission must stay stopped"
            )
        ids = indices.reshape(-1).contiguous()
        n = ids.numel()
        if n > self.max_queries:
            raise ValueError(f"shared Engram scratch capacity {n}>{self.max_queries}")
        li = (1, 14).index(layer)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prefetch consumer must remain the native eager Engram break"
            )
        if self.prefetch_ids is None:
            if self.forward_cp_expected:
                raise RuntimeError(
                    "CP forward did not execute early hash prefetch hook"
                )
            # Tiny/non-CP forwards (including health requests) never run
            # the full-CP hash hook. Keep the same producer stream even
            # when a synchronous read is required.
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                ids.record_stream(self.comm_stream)
                self.context.fetch(
                    ids,
                    n,
                    li * self.span + self.p_off,
                    0,
                    li * self.span + self.dump,
                    512,
                    self.layer_stats[li],
                    weight_index=li,
                    scale_index=li,
                    row_lo=lo,
                    row_hi=hi,
                    row264=True,
                )
                self.ready_events[layer].record()
        elif self.prefetch_ids[layer].numel() != n or self.ranges[layer] != (
            lo,
            hi,
        ):
            raise RuntimeError("prefetch consumer shape/range was not prepared")
        if self.prefetch_ids is not None and layer not in self.prefetch_issued:
            self.issue_prefetch_layer(
                layer
            )  # Safe synchronous fallback for non-scatter paths.
        torch.cuda.current_stream().wait_event(self.ready_events[layer])
        if self.prefetch_ids is not None:
            torch._assert_async(
                (ids == self.prefetch_ids[layer]).all(),
                "prefetched Engram IDs differ",
            )
        torch._assert_async(
            self.layer_stats[li].view(torch.int64)[0] == 0,
            "prefetched Engram GET failed",
        )
        offset = li * self.span + self.p_off
        raw = self.arena[offset : offset + n * 264].view(n, 264)
        return ids, raw[:, :256].view(torch.float8_e4m3fn), raw[:, 256:]

    def begin_prefetch(self, full_hash_ids, local_rows, cp_size):
        if not self.prefetch_enabled or self.closed:
            raise RuntimeError("prefetch runtime unavailable")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "full-hash prefetch must start before custom graph capture"
            )
        if (
            set(self.ranges) != {1, 14}
            or full_hash_ids.ndim != 3
            or full_hash_ids.shape[1] != 2
        ):
            raise ValueError("both trained layer ranges/full hash tensor are required")
        if cp_size not in (4, 8) or full_hash_ids.shape[0] > cp_size * local_rows:
            raise ValueError("unsupported CP/padded hash shape")
        n = cp_size * local_rows * full_hash_ids.shape[2]
        if n == 0:
            return
        self.forward_cp_expected = True
        if n > self.max_queries:
            raise ValueError("prefetch exceeds registered scratch")
        # Match native cat([hash[r::CP] padded to hidden_rows] for r in CP).
        ordered = full_hash_ids.new_zeros(
            (cp_size * local_rows, *full_hash_ids.shape[1:])
        )
        for rank in range(cp_size):
            part = full_hash_ids[rank::cp_size]
            ordered[rank * local_rows : rank * local_rows + part.shape[0]].copy_(part)
        ids = {
            layer: ordered[:, li, :].contiguous().reshape(-1)
            for li, layer in enumerate((1, 14))
        }
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        self.prefetch_ids = ids
        with torch.cuda.stream(self.comm_stream):
            for layer in (1,) if self.defer_layer14 else (1, 14):
                self._enqueue_prefetch_layer(layer)

    def _enqueue_prefetch_layer(self, layer):
        li = (1, 14).index(layer)
        ids = self.prefetch_ids[layer]
        lo, hi = self.ranges[layer]
        ids.record_stream(self.comm_stream)
        self.context.fetch(
            ids,
            ids.numel(),
            li * self.span + self.p_off,
            0,
            li * self.span + self.dump,
            512,
            self.layer_stats[li],
            weight_index=li,
            scale_index=li,
            row_lo=lo,
            row_hi=hi,
            row264=True,
        )
        self.ready_events[layer].record()
        self.prefetch_issued.add(layer)

    def issue_prefetch_layer(self, layer):
        if self.closed:
            raise RuntimeError("cannot prefetch after drain")
        if self.prefetch_ids is None or layer in self.prefetch_issued:
            return
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            self._enqueue_prefetch_layer(layer)

    def start_forward(self, cp_extend):
        self.prefetch_ids = None
        self.prefetch_issued = set()
        self.forward_cp_expected = bool(cp_extend)

    def record_consumed(self, layer):
        self.consumed_events[layer].record(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(self.consumed_events[layer])
            self.context.record_consumed()

    def close(self):
        if not self.closed:
            # Native close retires CUDA/NIC consumers before the owner lease.
            self.context.close()
            self.lease.release()
            self.closed = True

    def _controlled_drain(self, *_):
        # Research controller stops admission and waits for all HTTP requests
        # before signaling every rank. Keep workers alive until all release:
        # an early rank exit could make the parent kill undrained peer ranks.
        self.close()
        print(
            json.dumps(
                {
                    "event": "shared-engram-reader-drained",
                    "pid": os.getpid(),
                    "lease_released": True,
                }
            ),
            flush=True,
        )
