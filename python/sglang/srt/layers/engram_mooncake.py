"""Mooncake host rows: graph-external fetch, graph-captured per-layer H2D.

Only the optional host backend imports Mooncake. The local/GPU table remains
unchanged. Buffers live for the model lifetime and are never replaced after
capture. Sequential scheduling is required for host-buffer ownership.
"""

import functools
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)


@functools.lru_cache(maxsize=1)
def connect_store(config_path):
    from mooncake.store import EngramStore, EngramStoreConfig, MooncakeDistributedStore

    config = json.loads(Path(config_path).read_text())
    mode = config.get("mode", "store")
    if mode not in ("store", "local"):
        raise ValueError(f"Unknown Engram mode: {mode}")
    store = None
    if mode == "store":
        store = MooncakeDistributedStore()
        rc = store.setup(**config["connection"])
        if rc != 0:
            raise RuntimeError(f"Mooncake setup failed: {rc}")
    layers = {}
    for layer_id, layout in config["layers"].items():
        cfg = EngramStoreConfig()
        cfg.table_vocab_sizes = layout["table_vocab_sizes"]
        cfg.row_bytes = layout["row_bytes"]
        layers[int(layer_id)] = cfg
    table = EngramStore(layers, store)
    if mode == "local":
        for layer_id, cfg in layers.items():
            paths = config["local_tables"][str(layer_id)]
            if len(paths) != len(cfg.table_vocab_sizes):
                raise ValueError("Local Engram table count mismatch")
            arrays = []
            for path, rows in zip(paths, cfg.table_vocab_sizes):
                if Path(path).stat().st_size != rows * cfg.row_bytes:
                    raise ValueError(f"Local Engram table size mismatch: {path}")
                arrays.append(
                    np.memmap(
                        path, mode="r", dtype=np.uint8, shape=(rows, cfg.row_bytes)
                    )
                )
            table.bind_local(layer_id, arrays)
    return store, table, config


def _lookup_rows(table, layer_id, ids, padding, output):
    # A prefix view stays contiguous and inside the registered output allocation.
    # Avoid reading a full graph bucket for a short or converted decode batch.
    end = np.flatnonzero(~padding)[-1] + 1 if padding.any() else len(ids)
    table.lookup_into(layer_id, ids[None, :end], output[:, :end])
    output[0, padding] = 0


def _release_buffers(store, buffers, executor):
    # A worker may still be writing after an interrupted forward.
    executor.shutdown(wait=True)
    for host in buffers.values():
        if store is not None and host.numel():
            store.unregister_buffer(host.data_ptr())
    buffers.clear()


class MooncakeEngramEmbedding(nn.Module):
    _shared = False  # Do not activate the local-table layer-14 project prefetch.

    def __init__(self, layout, layer_id):
        super().__init__()
        from sglang.srt.runtime_context import get_exec, get_server_args

        args = get_server_args()
        if not envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            raise ValueError(
                "Mooncake Engram requires SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1"
            )
        if (
            args.pp_size != 1
            or args.attn_cp_size != 1
            or args.speculative_algorithm
            or not args.disable_overlap_schedule
        ):
            raise ValueError(
                "Mooncake Engram requires PP=CP=1, no speculation and --disable-overlap-schedule"
            )
        graph = get_exec().graph.cuda_graph_config
        if graph.decode.backend != "breakable" or graph.prefill.backend != "breakable":
            raise ValueError(
                "Mooncake Engram requires --cuda-graph-backend-decode breakable --cuda-graph-backend-prefill breakable"
            )
        if envs.SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH.get():
            raise ValueError(
                "Mooncake row prefetch cannot be combined with local Engram KV prefetch"
            )
        if envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get() == "private":
            raise ValueError(
                "Mooncake Engram uses complete rows, not private TP shards"
            )
        self.dim = layout.head_dim
        self.layer_id = layer_id
        self.store, self.table, manifest = connect_store(
            envs.SGLANG_DSV41_ENGRAM_MOONCAKE_CONFIG.get()
        )
        primes = [
            p
            for group in layout.primes[layout.layer_ids.index(layer_id)]
            for p in group
        ]
        expected = {
            "table_vocab_sizes": primes,
            "head_dim": self.dim,
            "row_bytes": self.dim + self.dim // 32,
        }
        if manifest["layers"][str(layer_id)] != expected:
            raise ValueError(f"Mooncake Engram layout mismatch for layer {layer_id}")
        if self.store is not None:
            ready = self.store.get("engram:ready")
            if not ready or json.loads(ready) != manifest["layers"]:
                raise ValueError(
                    "Mooncake Engram tables have not been published with this layout"
                )
        self.offsets = np.cumsum([0] + primes[:-1], dtype=np.int64)
        self.host_buffers = {}
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"engram-{layer_id}"
        )
        self.pending = None
        self._release = weakref.finalize(
            self, _release_buffers, self.store, self.host_buffers, self.executor
        )
        self.lookup_count = 0

    def buffer(self, num_tokens):
        if num_tokens not in self.host_buffers:
            host = torch.zeros(
                (1, num_tokens, len(self.offsets), self.dim + self.dim // 32),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            if (
                host.numel()
                and self.store is not None
                and self.store.register_buffer(host.data_ptr(), host.numel()) != 0
            ):
                raise RuntimeError("Could not register Mooncake Engram staging buffer")
            self.host_buffers[num_tokens] = host
        return self.host_buffers[num_tokens]

    def prefetch(self, global_ids):
        # Drain an unconsumed read if an earlier forward stopped before this layer.
        self.wait()
        # Blocking D2H also drains earlier same-stream graph reads of these host
        # buffers before reuse. Overlap scheduling is rejected at construction.
        ids = global_ids.cpu().numpy()
        host = self.buffer(len(ids))
        if len(ids):
            # All-zero hash rows are padding, not per-head global row zero.
            padding = np.all(ids == 0, axis=-1)
            if padding.all():
                host.zero_()  # Idle DPA ranks need no embedding reads.
                return
            local_ids = np.ascontiguousarray(ids - self.offsets)
            local_ids[padding] = 0
            # Only CPU arrays enter the worker. The binding releases the GIL
            # during Store I/O, allowing GPU submission to continue on this thread.
            self.pending = self.executor.submit(
                _lookup_rows,
                self.table,
                self.layer_id,
                local_ids,
                padding,
                host.numpy(),
            )

    @eager_on_graph(True, capture_stub=lambda self: None)
    def wait(self):
        from sglang.srt.model_executor.runner_utils import capture_mode

        if capture_mode.is_capture_mode:
            return
        if self.pending is not None:
            try:
                self.pending.result()
                self.lookup_count += 1
            finally:
                self.pending = None

    def forward(self, indices, forward_batch=None, *, cp_all_tokens=False):
        # Replay waits outside the graph immediately before this layer's H2D.
        # An I/O failure prevents stale or partially written rows reaching the GPU.
        self.wait()
        host = self.host_buffers[indices.shape[0]]
        raw = torch.empty(host.shape[1:], dtype=torch.uint8, device=indices.device)
        raw.copy_(host[0], non_blocking=True)  # Captured at this Engram layer.
        weight = raw[..., : self.dim].contiguous().view(torch.float8_e4m3fn).float()
        scale = raw[..., self.dim :].contiguous().view(torch.float8_e8m0fnu).float()
        return (
            (weight.unflatten(-1, (-1, 32)) * scale.unsqueeze(-1))
            .flatten(-2)
            .to(torch.bfloat16)
        )

    def finish_load(self, label=""):
        pass  # Parameters are owned by the external Store, not this module.
