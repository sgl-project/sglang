from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import time
import torch

from .infllmv2_pooler import PoolerBase, TorchPooler, _out_len

@dataclass
class KVViewsConfig:
    c1_kernel: int = 32
    c1_stride: int = 16
    c1_to_block_by_maxpool: bool = True
    c1_maxpool_kernel: int = 5
    c1_maxpool_stride: int = 4
    c1_maxpool_padding: int = 1
    c2_kernel: int = 128
    c2_stride: int = 64

class InfLLM2KVViews:
    def __init__(self, cfg: KVViewsConfig = KVViewsConfig(), pooler: Optional[PoolerBase] = None):
        self.cfg = cfg
        self.pooler: PoolerBase = pooler or TorchPooler()
    def _avg_pool_seq(self, x: torch.Tensor, kernel: int, stride: int) -> torch.Tensor:
        return self.pooler.avg_pool1d(x, kernel, stride)
    def _max_pool_seq(self, x: torch.Tensor, kernel: int, stride: int, padding: int) -> torch.Tensor:
        return self.pooler.max_pool1d(x, kernel, stride, padding)

    # convenience for one-shot build
    def build_views_from_k(self, k_BHSD: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c1_avg = self._avg_pool_seq(k_BHSD, self.cfg.c1_kernel, self.cfg.c1_stride)
        c1 = self._max_pool_seq(c1_avg, self.cfg.c1_maxpool_kernel, self.cfg.c1_maxpool_stride, self.cfg.c1_maxpool_padding) if self.cfg.c1_to_block_by_maxpool else c1_avg
        c2 = self._avg_pool_seq(k_BHSD, self.cfg.c2_kernel, self.cfg.c2_stride)
        return c1, c2

class KVViewsSidecar:
    KEY = "_infllm2_views"
    def __init__(self, kv_views: InfLLM2KVViews):
        self.kv_views = kv_views

    def ensure_handle(self, state_dict: Dict[str, Any]):
        if self.KEY not in state_dict:
            state_dict[self.KEY] = {}
        return state_dict[self.KEY]

    def get_or_build(self, state_dict: Dict[str, Any], layer_id: int, k_local_BHSD: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        views = self.ensure_handle(state_dict)
        entry = views.get(layer_id)
        B, HK, SC, _ = k_local_BHSD.shape
        if entry is None:
            c1_avg = self.kv_views._avg_pool_seq(k_local_BHSD, self.kv_views.cfg.c1_kernel, self.kv_views.cfg.c1_stride)
            c1 = self.kv_views._max_pool_seq(c1_avg, self.kv_views.cfg.c1_maxpool_kernel, self.kv_views.cfg.c1_maxpool_stride, self.kv_views.cfg.c1_maxpool_padding) if self.kv_views.cfg.c1_to_block_by_maxpool else c1_avg
            c2 = self.kv_views._avg_pool_seq(k_local_BHSD, self.kv_views.cfg.c2_kernel, self.kv_views.cfg.c2_stride)
            sc_done_bhk = torch.full((B, HK), SC, device=k_local_BHSD.device, dtype=torch.int32)
            entry = views[layer_id] = {
                "c1": c1, "c1_avg": c1_avg, "c2": c2,
                "sc_done_bhk": sc_done_bhk, "stamp": time.time(),
            }
        return entry["c1"], entry["c2"]

    def update_append_batched(self, state_dict: Dict[str, Any], layer_id: int,
                              k_old: torch.Tensor, k_new: torch.Tensor,
                              sc_old: torch.Tensor, sc_add: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        views = self.ensure_handle(state_dict)
        entry = views.get(layer_id)
        if entry is None:
            k_full = torch.cat([k_old[:, :, :sc_old.max(), :], k_new[:, :, :sc_add.max(), :]], dim=2)
            return self.get_or_build(state_dict, layer_id, k_full)

        c1_prev: torch.Tensor = entry["c1"]
        c1_avg_prev: torch.Tensor = entry["c1_avg"]
        c2_prev: torch.Tensor = entry["c2"]

        c2_new, _ = self.kv_views.pooler.append_avg_pool1d_batched(
            c2_prev, k_old, k_new, sc_old, sc_add, self.kv_views.cfg.c2_kernel, self.kv_views.cfg.c2_stride
        )
        c1_avg_new, d1_avg = self.kv_views.pooler.append_avg_pool1d_batched(
            c1_avg_prev, k_old, k_new, sc_old, sc_add, self.kv_views.cfg.c1_kernel, self.kv_views.cfg.c1_stride
        )
        try:
            c1_new, _ = self.kv_views.pooler.append_max_pool1d_batched(
                c1_prev, c1_avg_prev, c1_avg_new,
                _out_len(sc_old, self.kv_views.cfg.c1_kernel, self.kv_views.cfg.c1_stride, 0),
                d1_avg,
                self.kv_views.cfg.c1_maxpool_kernel, self.kv_views.cfg.c1_maxpool_stride, self.kv_views.cfg.c1_maxpool_padding
            )
        except NotImplementedError:
            c1_full = self.kv_views._max_pool_seq(c1_avg_new, self.kv_views.cfg.c1_maxpool_kernel, self.kv_views.cfg.c1_maxpool_stride, self.kv_views.cfg.c1_maxpool_padding)
            add = c1_full.shape[2] - c1_prev.shape[2]
            c1_new = torch.cat([c1_prev, c1_full[:, :, -max(add, 0):, :]], dim=2)

        entry["c1"], entry["c1_avg"], entry["c2"] = c1_new, c1_avg_new, c2_new
        entry["sc_done_bhk"] = sc_old + sc_add
        entry["stamp"] = time.time()
        return c1_new, c2_new

    def get_lengths(self, state_dict: Dict[str, Any], layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        views = self.ensure_handle(state_dict)
        entry = views.get(layer_id)
        assert entry is not None
        sc_done = entry["sc_done_bhk"]
        sc1 = _out_len(sc_done, self.kv_views.cfg.c1_kernel, self.kv_views.cfg.c1_stride)
        sc2 = _out_len(sc_done, self.kv_views.cfg.c2_kernel, self.kv_views.cfg.c2_stride)
        return sc1, sc2

class ContextMemoryManager:
    def __init__(self, ttl_seconds: int = 900, max_entries: int = 4096):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._pool: Dict[str, Dict[str, Any]] = {}
        self._meta: Dict[str, float] = {}

    def _sweep(self):
        import time as _t
        now = _t.time()
        to_del = [rid for rid, ts in self._meta.items() if self._ttl > 0 and now - ts > self._ttl]
        for rid in to_del:
            self._pool.pop(rid, None); self._meta.pop(rid, None)
        if len(self._pool) > self._max:
            overflow = len(self._pool) - self._max
            for rid in list(sorted(self._meta, key=self._meta.get))[:overflow]:
                self._pool.pop(rid, None); self._meta.pop(rid, None)

    def get_state(self, request_id: str) -> Dict[str, Any]:
        import time as _t
        self._sweep()
        st = self._pool.get(request_id)
        if st is None:
            st = {}
            self._pool[request_id] = st
        self._meta[request_id] = _t.time()
        return st

    def clear_state(self, request_id: str):
        self._pool.pop(request_id, None)
        self._meta.pop(request_id, None)

    def clear_all(self):
        self._pool.clear(); self._meta.clear()