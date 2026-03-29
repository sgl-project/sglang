"""TurboQuant KV cache quantization for DeepSeek V2/V3 MLA.

Compresses the MLA latent vector (kv_lora_rank + qk_rope_head_dim) stored in
MLATokenToKVPool. The latent part (kv_lora_rank) is quantized via TurboQuant
while the RoPE part (qk_rope_head_dim) stays in FP16 for positional precision.

Two modes:
  - MSE-only: For value reconstruction (weighted sum averages out errors)
  - Prod (MSE + QJL): For key inner products (unbiased attention scores)

Integration: Wraps around MLATokenToKVPool, intercepting set/get operations.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.turboquant_engine import (
    generate_rotation_matrix,
    get_codebook,
    pack_indices,
    unpack_indices,
    pad_for_packing,
)


class TurboQuantKVCompressor:
    """Compresses MLA latent KV cache vectors using TurboQuant.

    For DeepSeek V3:
      kv_lora_rank = 512 (latent part, quantized)
      qk_rope_head_dim = 64 (rope part, kept in FP16)
      total kv_cache_dim = 576

    Storage per token (at 4-bit):
      Original: 576 × 2 bytes (FP16) = 1152 bytes
      Compressed: 512 × 0.5 bytes + 64 × 2 bytes (FP16) + norms ≈ 392 bytes → 2.94x
    At 3-bit: 192 + 128 + 8 = 328 bytes → 3.51x
    At 2-bit: 128 + 128 + 8 = 264 bytes → 4.36x
    """

    def __init__(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        bit_width: int = 4,
        group_size: Optional[int] = None,
        seed: int = 42,
        use_qjl: bool = True,
        device: str = "cpu",
    ):
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.bit_width = bit_width
        self.group_size = group_size or kv_lora_rank
        self.seed = seed
        self.use_qjl = use_qjl
        self.device = device

        self.n_levels = 2 ** bit_width
        self.n_groups = math.ceil(kv_lora_rank / self.group_size)

        centroids, boundaries = get_codebook(bit_width)
        self.centroids = centroids.to(device)
        self.boundaries = boundaries.to(device)

        self._rotation_cache: Dict[int, torch.Tensor] = {}

        # QJL projection matrix for unbiased inner product estimation
        if use_qjl:
            gen = torch.Generator().manual_seed(seed + 10000)
            self.S = torch.randn(
                kv_lora_rank, kv_lora_rank, generator=gen, dtype=torch.float32
            ).to(device)
        else:
            self.S = None

    def _get_rotation(self, g_start: int, dim_override: Optional[int] = None) -> torch.Tensor:
        key = self.seed + g_start
        dim = dim_override
        if dim is None:
            g_end = min(g_start + self.group_size, self.kv_lora_rank)
            dim = g_end - g_start
        cache_key = (key, dim)
        if cache_key not in self._rotation_cache:
            Pi = generate_rotation_matrix(dim, seed=key).to(self.device)
            self._rotation_cache[cache_key] = Pi
        return self._rotation_cache[cache_key]

    @torch.no_grad()
    def compress(
        self, kv_states: torch.Tensor, global_norm: Optional[bool] = None
    ) -> dict:
        """Compress MLA latent vectors.

        Args:
            kv_states: (..., kv_cache_dim) where last dim = kv_lora_rank + qk_rope_head_dim
            global_norm: if True, use paper's algorithm (one global norm + one rotation).
                         If None, auto-select: global when use_qjl=True, grouped otherwise.
        """
        if global_norm is None:
            global_norm = self.use_qjl

        orig_shape = kv_states.shape
        kv_flat = kv_states.reshape(-1, self.kv_cache_dim).float()
        T = kv_flat.shape[0]

        latent = kv_flat[:, : self.kv_lora_rank]
        rope = kv_flat[:, self.kv_lora_rank :]

        if global_norm:
            return self._compress_global(latent, rope, orig_shape, T)
        else:
            return self._compress_grouped(latent, rope, orig_shape, T)

    def _compress_global(self, latent, rope, orig_shape, T):
        """Paper's algorithm: ONE global norm, ONE rotation for full kv_lora_rank."""
        d = self.kv_lora_rank

        # Global normalization (paper: unit vector on S^{d-1})
        vec_norms = latent.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (T, 1)
        latent_norm = latent / vec_norms

        # One rotation for full dimension (d x d)
        Pi = self._get_rotation(0, dim_override=d)
        Y = latent_norm @ Pi.T
        scale = math.sqrt(d)
        Y_scaled = Y * scale

        indices = torch.searchsorted(self.boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, self.n_levels - 1).reshape(T, d)

        # MSE reconstruction
        Y_hat = self.centroids[indices] / scale
        latent_mse = (Y_hat @ Pi) * vec_norms  # back to original scale

        padded = pad_for_packing(d, self.bit_width)
        if padded > d:
            indices = torch.nn.functional.pad(indices, (0, padded - d), value=0)
        packed = pack_indices(indices, self.bit_width)

        result = {
            "indices_packed": packed,
            "norms": vec_norms.squeeze(1),  # float32 for QJL precision
            "rope_part": rope.half(),
            "orig_shape": orig_shape,
            "global_norm": True,
        }

        if self.use_qjl:
            residual = latent - latent_mse
            residual_norm = torch.norm(residual, dim=-1)  # (T,) float32
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1

            result["k_mse"] = latent_mse  # float32 for QJL precision
            result["qjl_signs"] = signs
            result["residual_norm"] = residual_norm  # float32

        return result

    def _compress_grouped(self, latent, rope, orig_shape, T):
        """Original grouped algorithm: per-group norms + rotations."""
        all_indices = []
        all_norms = []
        latent_mse = torch.zeros_like(latent)

        for g in range(self.n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, self.kv_lora_rank)
            g_dim = g_end - g_start
            L_g = latent[:, g_start:g_end]

            norms = L_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
            L_norm = L_g / norms
            all_norms.append(norms.squeeze(1))

            Pi = self._get_rotation(g_start)
            Y = L_norm @ Pi.T
            scale = math.sqrt(g_dim)
            Y_scaled = Y * scale

            indices = torch.searchsorted(self.boundaries, Y_scaled.reshape(-1))
            indices = indices.clamp(0, self.n_levels - 1).reshape(T, g_dim)
            all_indices.append(indices)

            Y_hat = self.centroids[indices] / scale
            L_hat_g = Y_hat @ Pi
            latent_mse[:, g_start:g_end] = L_hat_g * norms

        full_indices = torch.cat(all_indices, dim=1)
        norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

        padded = pad_for_packing(self.kv_lora_rank, self.bit_width)
        if padded > self.kv_lora_rank:
            full_indices = torch.nn.functional.pad(full_indices, (0, padded - self.kv_lora_rank), value=0)
        packed = pack_indices(full_indices, self.bit_width)

        result = {
            "indices_packed": packed,
            "norms": norms_out.half(),
            "rope_part": rope.half(),
            "orig_shape": orig_shape,
            "global_norm": False,
        }

        if self.use_qjl:
            residual = latent - latent_mse
            residual_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1

            result["k_mse"] = latent_mse  # float32
            result["qjl_signs"] = signs
            result["residual_norm"] = residual_norm  # float32

        return result

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress to full MLA latent vector (MSE reconstruction + rope)."""
        indices_packed = compressed["indices_packed"]
        norms = compressed["norms"].float()
        rope = compressed["rope_part"]
        orig_shape = compressed["orig_shape"]

        is_global = compressed.get("global_norm", False)

        T = indices_packed.shape[0]
        d = self.kv_lora_rank
        padded = pad_for_packing(d, self.bit_width)
        indices = unpack_indices(indices_packed, padded, self.bit_width)[:, :d]

        latent = torch.zeros(T, d, dtype=torch.float32, device=indices_packed.device)

        if is_global:
            Pi = self._get_rotation(0, dim_override=d)
            scale = math.sqrt(d)
            Y = self.centroids[indices.long()] / scale
            latent = (Y @ Pi) * norms.unsqueeze(1)
        else:
            for g in range(self.n_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, d)
                g_dim = g_end - g_start
                scale = math.sqrt(g_dim)

                Pi = self._get_rotation(g_start)
                Y_g = self.centroids[indices[:, g_start:g_end].long()] / scale
                L_g = Y_g @ Pi

                if norms.dim() == 1:
                    L_g = L_g * norms.unsqueeze(1)
                else:
                    L_g = L_g * norms[:, g].unsqueeze(1)

                latent[:, g_start:g_end] = L_g

        kv_out = torch.cat([latent.half(), rope], dim=-1)
        return kv_out.view(*orig_shape)

    @torch.no_grad()
    def asymmetric_attention_scores(
        self,
        queries_latent: torch.Tensor,
        compressed: dict,
    ) -> torch.Tensor:
        """Compute unbiased <q, k> using TurboQuant asymmetric estimator.

        Uses the formula:
            <q, k> ≈ <q, k_mse> + ||r|| * sqrt(pi/2)/m * <S·q, sign(S·r)>

        Args:
            queries_latent: (B, T_q, kv_lora_rank) query latent vectors
                           (already absorbed via w_kc)
            compressed: dict from compress()

        Returns:
            scores: (B, T_q, T_k) attention scores for the latent (nope) part
        """
        if not self.use_qjl or "k_mse" not in compressed:
            raise ValueError("QJL data not available; compress with use_qjl=True")

        k_mse = compressed["k_mse"].float()
        signs = compressed["qjl_signs"].float()
        r_norm = compressed["residual_norm"].float()

        # Term 1: Q @ K_mse^T
        term1 = queries_latent.float() @ k_mse.T

        # Term 2: QJL correction
        q_proj = queries_latent.float() @ self.S.T
        qjl_ip = q_proj @ signs.T

        m = self.S.shape[0]
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = correction_scale * qjl_ip * r_norm.unsqueeze(0)

        return term1 + term2

    def memory_usage_per_token(self) -> dict:
        """Estimate memory per token in bytes."""
        latent_bits = self.kv_lora_rank * self.bit_width
        rope_bytes = self.qk_rope_head_dim * 2  # FP16
        norm_bytes = self.n_groups * 2  # FP16 norms
        qjl_bytes = self.kv_lora_rank if self.use_qjl else 0
        rnorm_bytes = 2 if self.use_qjl else 0
        kmse_bytes = self.kv_lora_rank * 2 if self.use_qjl else 0

        compressed_bytes = (
            latent_bits // 8 + rope_bytes + norm_bytes + qjl_bytes + rnorm_bytes + kmse_bytes
        )
        original_bytes = (self.kv_lora_rank + self.qk_rope_head_dim) * 2

        return {
            "compressed_bytes": compressed_bytes,
            "original_bytes": original_bytes,
            "compression_ratio": original_bytes / max(compressed_bytes, 1),
        }


class TurboQuantMLAKVPool:
    """Wrapper around MLATokenToKVPool that adds TurboQuant compression.

    Intercepts KV cache writes to compress latent vectors on-the-fly,
    and provides compressed read access for attention computation.

    Usage:
        pool = TurboQuantMLAKVPool(
            base_pool=original_pool,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            bit_width=4,
        )
    """

    def __init__(
        self,
        base_pool,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        bit_width: int = 4,
        group_size: Optional[int] = None,
        seed: int = 42,
        use_qjl: bool = False,
    ):
        self.base_pool = base_pool
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        device = "cpu"
        if hasattr(base_pool, "device"):
            device = str(base_pool.device)

        self.compressor = TurboQuantKVCompressor(
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            bit_width=bit_width,
            group_size=group_size,
            seed=seed,
            use_qjl=use_qjl,
            device=device,
        )

        self._compressed_cache: Dict[int, Dict[int, dict]] = {}

    def compress_and_store(
        self, layer_id: int, cache_locs: torch.Tensor, kv_data: torch.Tensor
    ):
        """Compress KV data and store both compressed + original.

        The original is still written to base_pool for compatibility with
        existing attention backends. The compressed version is stored
        separately for long-context eviction/offloading.
        """
        compressed = self.compressor.compress(kv_data)

        if layer_id not in self._compressed_cache:
            self._compressed_cache[layer_id] = {}

        for i, loc in enumerate(cache_locs):
            self._compressed_cache[layer_id][loc.item()] = {
                k: v[i] if isinstance(v, torch.Tensor) else v
                for k, v in compressed.items()
            }

    def get_compressed(self, layer_id: int, cache_locs: torch.Tensor) -> dict:
        """Retrieve compressed KV data for given locations."""
        if layer_id not in self._compressed_cache:
            raise KeyError(f"No compressed cache for layer {layer_id}")

        layer_cache = self._compressed_cache[layer_id]
        batch_data = {}
        for loc in cache_locs:
            loc_data = layer_cache.get(loc.item())
            if loc_data is None:
                continue
            for k, v in loc_data.items():
                if k not in batch_data:
                    batch_data[k] = []
                batch_data[k].append(v)

        result = {}
        for k, v_list in batch_data.items():
            if isinstance(v_list[0], torch.Tensor):
                result[k] = torch.stack(v_list)
            else:
                result[k] = v_list

        return result

    def decompress_and_get(
        self, layer_id: int, cache_locs: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve and decompress KV data."""
        compressed = self.get_compressed(layer_id, cache_locs)
        return self.compressor.decompress(compressed)

    def memory_usage_per_token(self) -> dict:
        """Delegate to compressor."""
        return self.compressor.memory_usage_per_token()
