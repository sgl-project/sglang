# SPDX-License-Identifier: Apache-2.0
"""KVarN configuration.

KVarN = Hadamard rotation + iterative variance
normalization (Sinkhorn-like) + asymmetric RTN quantization.

The configuration object is pure (no framework imports) so it can be
unit-tested in isolation.
"""

import math
import os
from dataclasses import dataclass

# Named KVarN presets: each maps to a frozen set of config parameters.
# The trailing g<N> encodes the variance-normalization tile size, which must
# equal the page size. g128 is the current design point; g64 trades a little
# compression (more per-tile scale overhead) for finer quantization granularity.
KVARN_PRESETS: dict[str, dict] = {
    "kvarn_k4v2_g128": {"key_bits": 4, "value_bits": 2, "group": 128},
    "kvarn_k4v4_g128": {"key_bits": 4, "value_bits": 4, "group": 128},
    "kvarn_k4v2_g64": {"key_bits": 4, "value_bits": 2, "group": 64},
    "kvarn_k4v4_g64": {"key_bits": 4, "value_bits": 4, "group": 64},
}


def is_kvarn_dtype(dtype_str: str) -> bool:
    """Return True if *dtype_str* is a KVarN preset name."""
    return isinstance(dtype_str, str) and dtype_str.startswith("kvarn_")


@dataclass
class KVarNConfig:
    """Configuration for KVarN KV-cache quantization.

    Pipeline per (block, head):
      1. Hadamard rotation along head_dim.
      2. Iterative log-domain variance-normalization (Sinkhorn-like).
      3. Asymmetric per-row RTN at key_bits / value_bits.
      4. Absorb the per-row RTN scale and zero-point into the matching
         sinkhorn scale axis.

    Cache layout (per (block, head)) is a single packed record — see
    ``tile_bytes`` / ``tile_bytes_aligned``.
    """

    head_dim: int = 128
    key_bits: int = 4
    value_bits: int = 4
    group: int = 128
    sinkhorn_iters: int = 8
    sink_tokens: int = 128
    boundary_skip_layers: int = 0

    # ── derived: storage layout ──────────────────────────────────────────────
    @property
    def k_packed_bytes(self) -> int:
        return math.ceil(self.head_dim * self.group * self.key_bits / 8)

    @property
    def v_packed_bytes(self) -> int:
        return math.ceil(self.group * self.head_dim * self.value_bits / 8)

    @property
    def k_scale_bytes(self) -> int:
        """fp16 bytes for K scales: s_col_K' [D] + zp_K' [D] + s_row_K [group]."""
        return (2 * self.head_dim + self.group) * 2

    @property
    def v_scale_bytes(self) -> int:
        """fp16 bytes for V scales: s_col_V [D] + s_row_V' [group] + zp_V' [group]."""
        return (self.head_dim + 2 * self.group) * 2

    @property
    def tile_bytes(self) -> int:
        return (
            self.k_packed_bytes
            + self.k_scale_bytes
            + self.v_packed_bytes
            + self.v_scale_bytes
        )

    @property
    def tile_bytes_aligned(self) -> int:
        """tile_bytes rounded up for nicer Triton loads."""
        if self.head_dim >= 256:
            slot = math.ceil(self.tile_bytes / self.group)
            slot_pow2 = 1 << (slot - 1).bit_length()
            return slot_pow2 * self.group
        return ((self.tile_bytes + 7) // 8) * 8

    # ── slot byte offsets within one tile (used by the kernels) ──────────────
    @property
    def k_packed_offset(self) -> int:
        return 0

    @property
    def k_s_col_offset(self) -> int:
        return self.k_packed_offset + self.k_packed_bytes

    @property
    def k_zp_offset(self) -> int:
        return self.k_s_col_offset + self.head_dim * 2

    @property
    def k_s_row_offset(self) -> int:
        return self.k_zp_offset + self.head_dim * 2

    @property
    def v_packed_offset(self) -> int:
        return self.k_s_row_offset + self.group * 2

    @property
    def v_s_col_offset(self) -> int:
        return self.v_packed_offset + self.v_packed_bytes

    @property
    def v_s_row_offset(self) -> int:
        return self.v_s_col_offset + self.head_dim * 2

    @property
    def v_zp_offset(self) -> int:
        return self.v_s_row_offset + self.group * 2

    # ── fp16 tail-pool sizing ────────────────────────────────────────────────
    POOL_MEM_FRAC_DEFAULT = 0.08
    POOL_USABLE_SHARE_DEFAULT = 0.5

    def _slot_bytes_per_layer(self, num_kv_heads: int) -> int:
        return self.group * num_kv_heads * self.head_dim * 4

    def pool_slots(self, max_num_seqs: int, max_num_batched_tokens: int) -> int:
        prefill_blocks = (max_num_batched_tokens + self.group - 1) // self.group
        return max(2 * max_num_seqs + prefill_blocks + 8, 8)

    def pool_budget_bytes(
        self,
        total_gpu_bytes: int,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        if weight_bytes is not None and gpu_memory_utilization is not None:
            share = self.POOL_USABLE_SHARE_DEFAULT
            usable = gpu_memory_utilization * total_gpu_bytes - weight_bytes
            return max(0, int(share * usable))
        return int(total_gpu_bytes * self.POOL_MEM_FRAC_DEFAULT)

    def max_supported_seqs(
        self,
        total_gpu_bytes: int,
        num_kv_heads: int,
        num_layers: int,
        max_num_batched_tokens: int,
        frac: float | None = None,
        gpu_memory_utilization: float | None = None,
        weight_bytes: int | None = None,
    ) -> int:
        if frac is not None:
            budget = int(total_gpu_bytes * frac)
        else:
            budget = self.pool_budget_bytes(
                total_gpu_bytes, gpu_memory_utilization, weight_bytes
            )
        slot_bytes = self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)
        max_slots = int(budget / slot_bytes)
        prefill_blocks = (max_num_batched_tokens + self.group - 1) // self.group
        return max(1, (max_slots - prefill_blocks - 8) // 2)

    def pool_bytes(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_kv_heads: int,
        num_layers: int,
    ) -> int:
        slots = self.pool_slots(max_num_seqs, max_num_batched_tokens)
        return slots * self._slot_bytes_per_layer(num_kv_heads) * max(num_layers, 1)

    @staticmethod
    def num_kvarn_layers(model_config, parallel_config=None) -> int:
        """Number of layers the KVarN fp16 tail pool actually spans = the
        full-attention layers. On a hybrid model (Qwen3.5/3.6, Jamba, ...) the
        Mamba/linear-attention layers have no KVarN pool, so sizing the pool by
        ALL layers over-reserves it ~Nx and starves the Mamba/KV caches (OOM or
        cap collapse). For a dense transformer this equals total layers, so the
        dense path is unchanged. Falls back to total layers if the per-type
        count is unavailable."""
        try:
            if hasattr(model_config, "get_num_layers_by_block_type"):
                n = model_config.get_num_layers_by_block_type(
                    parallel_config, "attention"
                )
                if n and n > 0:
                    return n
            if hasattr(model_config, "num_attention_layers"):
                n = model_config.num_attention_layers
                if n and n > 0:
                    return n
        except Exception:
            pass

        # Fallback: total layers
        if hasattr(model_config, "num_layers"):
            return model_config.num_layers
        if hasattr(model_config, "get_num_layers"):
            return (
                model_config.get_num_layers(parallel_config)
                if parallel_config
                else model_config.get_num_layers()
            )
        return 24  # Safe default

    @staticmethod
    def estimate_weight_bytes(model: str, tensor_parallel_size: int = 1) -> int | None:
        """Best-effort per-rank model weight size in bytes, read from the
        checkpoint files on disk (exact, and cheap, with no CUDA context).
        Returns None if the files can't be located, so the caller falls back
        to the legacy budget.

        Resolves a local directory directly, or the local HF cache snapshot for
        a repo id (never downloads). Prefers the shards named in a
        ``*.safetensors.index.json`` manifest, which is exactly the set the
        loader reads. This avoids double-counting a repo that ships both a
        single consolidated checkpoint and the sharded HF set. Divides by the
        tensor-parallel degree (weights shard ~evenly across ranks)."""
        import glob as _glob
        import json as _json

        try:
            d = model
            if not os.path.isdir(d):
                try:
                    from huggingface_hub import snapshot_download

                    d = snapshot_download(model, local_files_only=True)
                except Exception:
                    return None

            # 1) Prefer the loader's own manifest
            for ext in ("safetensors", "bin"):
                indexes = _glob.glob(
                    os.path.join(d, "**", f"*.{ext}.index.json"), recursive=True
                )
                if not indexes:
                    continue
                try:
                    with open(indexes[0]) as fh:
                        weight_map = _json.load(fh).get("weight_map", {})
                    base = os.path.dirname(indexes[0])
                    names = sorted(set(weight_map.values()))
                    shards = [os.path.join(base, s) for s in names]
                    if names and all(os.path.exists(p) for p in shards):
                        total = sum(os.path.getsize(p) for p in shards)
                        if total > 0:
                            return total // max(tensor_parallel_size, 1)
                except Exception:
                    pass

            # 2) No usable manifest: prefer a canonical single-file checkpoint
            for single in ("model.safetensors", "consolidated.safetensors"):
                p = os.path.join(d, single)
                if os.path.exists(p):
                    total = os.path.getsize(p)
                    if total > 0:
                        return total // max(tensor_parallel_size, 1)

            # 3) Fallback: sum whatever weight shards are present
            files = _glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True)
            if not files:
                files = _glob.glob(os.path.join(d, "**", "*.bin"), recursive=True)
            if not files:
                return None
            total = sum(os.path.getsize(f) for f in files)
            if total <= 0:
                return None
            return total // max(tensor_parallel_size, 1)
        except Exception:
            return None

    @staticmethod
    def get_boundary_skip_layers(num_layers: int, n: int = 2) -> list[str]:
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        return [str(i) for i in sorted(set(first + last))]

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "KVarNConfig":
        """Create a config from a preset string like ``"kvarn_k4v4_g128"``."""
        if cache_dtype not in KVARN_PRESETS:
            valid = ", ".join(KVARN_PRESETS.keys())
            raise ValueError(
                f"Unknown KVarN cache dtype: {cache_dtype!r}. Valid: {valid}"
            )
        preset = KVARN_PRESETS[cache_dtype]
        return KVarNConfig(
            head_dim=head_dim,
            key_bits=preset["key_bits"],
            value_bits=preset["value_bits"],
            group=preset["group"],
            sinkhorn_iters=8,
            sink_tokens=128,
        )
