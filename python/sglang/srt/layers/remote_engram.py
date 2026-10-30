"""Experimental adapter for the external shared-Engram research runtime.

This adapter is opt-in and is not a supported NIXL integration. It expects the
legacy research runtime on PYTHONPATH. The reader owns registered scratch
until GPU work retires.
"""

import os

import torch

_runtime = None


def prepare_forward(cp_extend: bool):
    if _runtime is None or not _runtime.prefetch_enabled:
        raise RuntimeError("remote Engram prefetch runtime is not initialized")
    _runtime.start_forward(cp_extend)


def prefetch_full_hash(hash_ids, local_rows: int, cp_size: int):
    if _runtime is None:
        raise RuntimeError("remote Engram reader has not finished loading")
    _runtime.begin_prefetch(hash_ids, local_rows, cp_size)


class RemoteEngramTable:
    def __init__(self, num_embeddings: int, dim: int, layer_id: int):
        if dim != 256 or layer_id not in (1, 14):
            raise ValueError("research remote Engram supports 256-wide layers 1 and 14")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.layer_id = layer_id
        self.loaded = set()
        self.runtime = None

    def check_weight(self, param, loaded_weight, weight):
        kind = "weight" if param is weight else "scale"
        width = self.dim if kind == "weight" else self.dim // 32
        if tuple(loaded_weight.shape) != (self.num_embeddings, width):
            raise ValueError("remote Engram checkpoint tensor shape mismatch")
        if loaded_weight.dtype != param.dtype:
            raise ValueError("remote Engram checkpoint tensor dtype mismatch")
        self.loaded.add(kind)

    def finish_load(self, row_start: int, rows: int):
        global _runtime
        if self.loaded != {"weight", "scale"}:
            raise RuntimeError("remote Engram checkpoint parameters not loaded")
        if _runtime is None:
            from shared_model_adapter import ReaderRuntime

            _runtime = ReaderRuntime()
        self.runtime = _runtime
        layer_index = (1, 14).index(self.layer_id)
        if self.runtime.storage_layout == "row264":
            expected = (self.num_embeddings * 264,)
            actual = self.runtime.metadata["lengths"][layer_index : layer_index + 1]
        else:
            expected = (self.num_embeddings * 256, self.num_embeddings * 8)
            actual = self.runtime.metadata["lengths"][
                2 * layer_index : 2 * layer_index + 2
            ]
        if tuple(actual) != expected:
            raise ValueError("remote Engram table size mismatch")
        self.runtime.ranges[self.layer_id] = (row_start, row_start + rows)
        if os.environ.get("ENGRAM_SHARED_READ_PLAN") == "1" and set(
            self.runtime.ranges
        ) == {1, 14}:
            with torch.cuda.stream(self.runtime.comm_stream):
                self.runtime.context.prime(
                    self.runtime.ranges,
                    self.runtime.p_off,
                    self.runtime.dump,
                    self.runtime.layer_stats,
                )

    def lookup(self, indices, row_start: int, rows: int, new_rows):
        if self.runtime is None:
            raise RuntimeError("remote Engram reader has not finished loading")
        out = new_rows(indices)
        if not indices.numel():
            return out
        ids, weight, scale = self.runtime.fetch(
            indices, self.layer_id, row_start, row_start + rows
        )
        if self.runtime.storage_layout == "row264":
            from row264_kernels import gather_row264

            representatives = (
                self.runtime.context.representatives((1, 14).index(self.layer_id))
                if hasattr(self.runtime.context, "representatives")
                else None
            )
            gather_row264(
                weight.data_ptr(),
                ids,
                out.view(-1, self.dim),
                row_start,
                row_start + rows,
                query_order=True,
                representatives=representatives,
            )
        else:
            from sglang.kernels.ops.embeddings.engram_gather import engram_gather
            from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE

            n = ids.numel()
            owned = (ids >= row_start) & (ids < row_start + rows)
            local_ids = torch.where(owned, self.runtime.sequence[:n], n)
            engram_gather(
                weight.data_ptr(),
                scale.data_ptr(),
                local_ids,
                out.view(-1, self.dim),
                self.dim,
                FP8_BLOCK_SIZE,
                row_lo=0,
                row_hi=n,
            )
        self.runtime.record_consumed(self.layer_id)
        return out

    def after_reduction(self):
        if (
            self.layer_id == 1
            and self.runtime.prefetch_enabled
            and self.runtime.defer_layer14
        ):
            self.runtime.issue_prefetch_layer(14)
