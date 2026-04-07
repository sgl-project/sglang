"""
TurboQuant memory pool for KV cache compression.

Implements Google's TurboQuant (ICLR 2026) KV cache quantization.
Stores bit-packed centroid indices + L2 norms per head per token.

Reads use selective dequantization: only the active token positions
(set via set_active_kv_indices) are dequantized into a shared workspace.
This gives O(active_tokens) reads with real memory savings (~3.76x at 4-bit).
"""

import math
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.quantization.turboquant_kernels import (
    HadamardTransform,
    _get_centroids_tensor,
    _next_power_of_2,
    compute_packed_dim,
    compute_packed_dim_mixed,
    initialize_centroids_cache,
    parse_bits,
    turboquant_dequant_fused,
    turboquant_dequant_fused_mixed,
    turboquant_dequantize,
    turboquant_dequantize_mixed,
    turboquant_quantize,
    turboquant_quantize_mixed,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    get_tensor_size_bytes,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

# Deterministic seeds for the randomized Hadamard rotation.
_HADAMARD_SEED_K = 42
_HADAMARD_SEED_K_LO = 43
_HADAMARD_SEED_V = 137
_HADAMARD_SEED_V_LO = 138


class MHATokenToKVPoolTurboQuant(MHATokenToKVPool):
    """Memory pool that stores KV cache compressed via TurboQuant.

    Storage per token per head per layer:
      - Bit-packed centroid indices (uint8, packed at b bits/coord)
      - L2 norm (float32, 1 per token-head)

    On read, only the active token positions are dequantized into a shared
    workspace buffer (one K, one V — NOT per-layer).  The attention backend
    calls set_active_kv_indices() before get_kv_buffer() to specify which
    positions will be read.

    Memory: compressed storage (per-layer) + shared workspace (one pair).
    At 4-bit this gives ~3.76x compression of the per-layer storage, with
    the workspace adding a small constant overhead.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        bits: float = 4,
        mode: str = "mse",
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.bits = bits
        self.mode = mode
        self.is_mixed, self.bits_hi, self.bits_lo = parse_bits(bits)

        self.padded_head_dim = _next_power_of_2(head_dim)
        effective_v = v_head_dim if v_head_dim is not None else head_dim
        self.v_padded_head_dim = _next_power_of_2(effective_v)

        # Hadamard transforms
        torch_device = torch.device(device)
        if self.is_mixed:
            k_split = head_dim // 2
            v_split = effective_v // 2
            self.k_hadamard_hi = HadamardTransform(
                k_split, seed=_HADAMARD_SEED_K, device=torch_device
            )
            self.k_hadamard_lo = HadamardTransform(
                head_dim - k_split, seed=_HADAMARD_SEED_K_LO, device=torch_device
            )
            self.v_hadamard_hi = HadamardTransform(
                v_split, seed=_HADAMARD_SEED_V, device=torch_device
            )
            self.v_hadamard_lo = HadamardTransform(
                effective_v - v_split, seed=_HADAMARD_SEED_V_LO, device=torch_device
            )
            self._k_split_dim = k_split
            self._v_split_dim = v_split
            self.k_hadamard = self.k_hadamard_hi
            self.v_hadamard = self.v_hadamard_hi
        else:
            self.k_hadamard = HadamardTransform(
                head_dim, seed=_HADAMARD_SEED_K, device=torch_device
            )
            self.v_hadamard = HadamardTransform(
                effective_v, seed=_HADAMARD_SEED_V, device=torch_device
            )

        # Active indices: set by attention backend before get_kv_buffer
        self._active_indices: Optional[torch.Tensor] = None

        # Rotated-domain mode: skip Hadamard inverse in dequant, instead
        # the attention backend rotates Q and output. Eliminates N×layers
        # Hadamard transforms, replacing with 2 per layer on single vectors.
        # Rotated-domain: skip Hadamard in dequant, rotate Q/output instead.
        # Faster for long contexts (1000+ tokens) but slower for short ones
        # due to Q/output rotation overhead on GQA models (many Q heads).
        # Disabled by default; enable via environment variable for long-context workloads.
        self._rotated_domain = (
            not self.is_mixed and self.mode == "mse"
            and os.environ.get("TURBOQUANT_ROTATED_DOMAIN", "0") == "1"
        )

        # TurboQuant uses compressed buffers incompatible with the
        # parent's Triton-based kv_cache_copy (which expects contiguous bf16).
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=False,
        )

        initialize_centroids_cache(torch_device)

        # Pre-scaled centroids for fused dequant path (centroids / sqrt(d))
        if self.is_mixed:
            k_split_padded = _next_power_of_2(self._k_split_dim)
            k_lo_padded = _next_power_of_2(head_dim - self._k_split_dim)
            v_split_padded = _next_power_of_2(self._v_split_dim)
            v_lo_padded = _next_power_of_2(effective_v - self._v_split_dim)
            c_hi = _get_centroids_tensor(self.bits_hi, torch_device)
            c_lo = _get_centroids_tensor(self.bits_lo, torch_device)
            self._k_scaled_centroids_hi = c_hi / math.sqrt(k_split_padded)
            self._k_scaled_centroids_lo = c_lo / math.sqrt(k_lo_padded)
            self._v_scaled_centroids_hi = c_hi / math.sqrt(v_split_padded)
            self._v_scaled_centroids_lo = c_lo / math.sqrt(v_lo_padded)
        elif self.mode == "mse":
            bits_int = int(self.bits)
            c = _get_centroids_tensor(bits_int, torch_device)
            self._k_scaled_centroids = c / math.sqrt(self.padded_head_dim)
            self._v_scaled_centroids = c / math.sqrt(self.v_padded_head_dim)

    def _create_buffers(self):
        """Allocate compressed storage + shared workspace."""
        self.store_dtype = torch.uint8

        m = self.size + self.page_size
        k_packed_dim = compute_packed_dim_mixed(self.head_dim, self.bits)
        v_packed_dim = compute_packed_dim_mixed(self.v_head_dim, self.bits)

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # Bit-packed centroid indices — per layer
                self.k_buffer = [
                    torch.zeros(
                        (m, self.head_num, k_packed_dim),
                        dtype=torch.uint8, device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, self.head_num, v_packed_dim),
                        dtype=torch.uint8, device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                # L2 norms — per layer
                norm_shape = (
                    (m, self.head_num, 2) if self.is_mixed else (m, self.head_num)
                )
                self.k_norms_buffer = [
                    torch.zeros(norm_shape, dtype=torch.float32, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_norms_buffer = [
                    torch.zeros(norm_shape, dtype=torch.float32, device=self.device)
                    for _ in range(self.layer_num)
                ]

                # QJL — only for "prod" mode
                if self.mode == "prod":
                    k_qjl_dim = compute_packed_dim(self.padded_head_dim, 1)
                    v_qjl_dim = compute_packed_dim(self.v_padded_head_dim, 1)
                    self.k_qjl_buffer = [
                        torch.zeros(
                            (m, self.head_num, k_qjl_dim),
                            dtype=torch.uint8, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_qjl_buffer = [
                        torch.zeros(
                            (m, self.head_num, v_qjl_dim),
                            dtype=torch.uint8, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.k_residual_norms_buffer = [
                        torch.zeros(
                            (m, self.head_num),
                            dtype=torch.float32, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_residual_norms_buffer = [
                        torch.zeros(
                            (m, self.head_num),
                            dtype=torch.float32, device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

                # Shared workspace: pool-sized, NOT per-layer.  Only the
                # active positions are filled on each _get call.  FlashInfer
                # indexes into this via its page table — inactive positions
                # are never read.
                self._k_ws = torch.zeros(
                    (m, self.head_num, self.head_dim),
                    dtype=self.dtype, device=self.device,
                )
                self._v_ws = torch.zeros(
                    (m, self.head_num, self.v_head_dim),
                    dtype=self.dtype, device=self.device,
                )

    def _clear_buffers(self):
        del self.k_buffer, self.v_buffer
        del self.k_norms_buffer, self.v_norms_buffer
        del self._k_ws, self._v_ws
        if self.mode == "prod":
            del self.k_qjl_buffer, self.v_qjl_buffer
            del self.k_residual_norms_buffer, self.v_residual_norms_buffer

    def get_kv_size_bytes(self):
        k_size = sum(get_tensor_size_bytes(b) for b in self.k_buffer)
        k_size += sum(get_tensor_size_bytes(b) for b in self.k_norms_buffer)
        k_size += get_tensor_size_bytes(self._k_ws)
        v_size = sum(get_tensor_size_bytes(b) for b in self.v_buffer)
        v_size += sum(get_tensor_size_bytes(b) for b in self.v_norms_buffer)
        v_size += get_tensor_size_bytes(self._v_ws)
        if self.mode == "prod":
            k_size += sum(get_tensor_size_bytes(b) for b in self.k_qjl_buffer)
            k_size += sum(
                get_tensor_size_bytes(b) for b in self.k_residual_norms_buffer
            )
            v_size += sum(get_tensor_size_bytes(b) for b in self.v_qjl_buffer)
            v_size += sum(
                get_tensor_size_bytes(b) for b in self.v_residual_norms_buffer
            )
        return k_size, v_size

    # ── Active indices: set by attention backend before get_kv_buffer ──

    def set_active_kv_indices(self, indices: torch.Tensor):
        """Set the token positions that will be read in the next attention call.

        Called by the attention backend after begin_forward() computes kv_indices.
        Only these positions are dequantized in _get_key_buffer/_get_value_buffer.
        """
        self._active_indices = indices

    # ── Rotated-domain attention: Q rotation + output de-rotation ──

    def rotate_q(self, q: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Rotate Q into the Hadamard domain for rotated-domain attention.

        Math: q_rot = forward_k(q) = (1/√d) * H * S_k * q.
        Then: q^T @ k_original = q_rot^T @ k_stored, because
        k_stored = (scaled_centroids * norm) and k_original = inverse(k_stored).
        """
        if not self._rotated_domain:
            return q
        return self.k_hadamard.forward(q.float()).to(q.dtype)

    def rotate_output(self, o: torch.Tensor, layer_id: int) -> torch.Tensor:
        """De-rotate attention output from the Hadamard domain.

        Math: output = inverse_v(output_rot).
        Because v_original = inverse_v(v_stored), linearity gives
        sum(alpha_i * v_original_i) = inverse_v(sum(alpha_i * v_stored_i)).
        """
        if not self._rotated_domain:
            return o
        return self.v_hadamard.inverse(o.float()).to(o.dtype)

    # ── Read path: selective dequant into shared workspace ──

    def _dequant_positions(self, layer_id: int, which: str):
        """Dequant active positions for one layer into the shared workspace.

        Uses direct indexing (no torch.unique) so all ops have fixed shapes
        determined by len(indices). Duplicate indices produce redundant but
        correct writes — same data to same position.

        For 4-bit MSE mode, uses the fused path (3 kernels instead of 6+).
        """
        idx = layer_id - self.start_layer
        indices = self._active_indices

        # Rotated-domain path: skip Hadamard inverse in dequant.
        # Q is pre-rotated and output is post-rotated by the attention backend.
        if self._rotated_domain:
            if which == "k":
                turboquant_dequant_fused(
                    self.k_buffer[idx], self.k_norms_buffer[idx],
                    indices, self.k_hadamard, self._k_scaled_centroids,
                    self._k_ws, self.head_dim, self.padded_head_dim,
                    skip_hadamard=True,
                )
            else:
                turboquant_dequant_fused(
                    self.v_buffer[idx], self.v_norms_buffer[idx],
                    indices, self.v_hadamard, self._v_scaled_centroids,
                    self._v_ws, self.v_head_dim, self.v_padded_head_dim,
                    skip_hadamard=True,
                )
            return

        # Full dequant path (with Hadamard inverse)
        if self.is_mixed:
            if which == "k":
                turboquant_dequant_fused_mixed(
                    self.k_buffer[idx], self.k_norms_buffer[idx],
                    indices, self.k_hadamard_hi, self.k_hadamard_lo,
                    self._k_scaled_centroids_hi, self._k_scaled_centroids_lo,
                    self._k_ws, self.head_dim, self._k_split_dim,
                    self.bits_hi, self.bits_lo,
                )
            else:
                turboquant_dequant_fused_mixed(
                    self.v_buffer[idx], self.v_norms_buffer[idx],
                    indices, self.v_hadamard_hi, self.v_hadamard_lo,
                    self._v_scaled_centroids_hi, self._v_scaled_centroids_lo,
                    self._v_ws, self.v_head_dim, self._v_split_dim,
                    self.bits_hi, self.bits_lo,
                )
            return

        if self.mode == "mse":
            if which == "k":
                turboquant_dequant_fused(
                    self.k_buffer[idx], self.k_norms_buffer[idx],
                    indices, self.k_hadamard, self._k_scaled_centroids,
                    self._k_ws, self.head_dim, self.padded_head_dim,
                )
            else:
                turboquant_dequant_fused(
                    self.v_buffer[idx], self.v_norms_buffer[idx],
                    indices, self.v_hadamard, self._v_scaled_centroids,
                    self._v_ws, self.v_head_dim, self.v_padded_head_dim,
                )
            return

        # Fallback path for prod mode
        if which == "k":
            packed = self.k_buffer[idx]
            norms = self.k_norms_buffer[idx]
            ws = self._k_ws
            out_dim = self.head_dim
        else:
            packed = self.v_buffer[idx]
            norms = self.v_norms_buffer[idx]
            ws = self._v_ws
            out_dim = self.v_head_dim

        sel_packed = packed[indices]   # (n, heads, packed_dim)
        sel_norms = norms[indices]     # (n, heads) or (n, heads, 2)
        n = indices.shape[0]

        # Flatten for dequant: (n * heads, packed_dim)
        flat_packed = sel_packed.reshape(-1, sel_packed.shape[-1])

        if self.is_mixed:
            flat_norms_hi = sel_norms[..., 0].reshape(-1).contiguous()
            flat_norms_lo = sel_norms[..., 1].reshape(-1).contiguous()
            hi_padded = _next_power_of_2(self._k_split_dim if which == "k" else self._v_split_dim)
            hi_packed_dim = compute_packed_dim(hi_padded, self.bits_hi)
            split_dim = self._k_split_dim if which == "k" else self._v_split_dim
            q = {
                "packed_hi": flat_packed[:, :hi_packed_dim],
                "packed_lo": flat_packed[:, hi_packed_dim:],
                "norms_hi": flat_norms_hi,
                "norms_lo": flat_norms_lo,
                "padded_dim_hi": hi_padded,
                "padded_dim_lo": _next_power_of_2(out_dim - split_dim),
                "split_dim": split_dim,
                "bits_hi": self.bits_hi,
                "bits_lo": self.bits_lo,
            }
            h_hi = self.k_hadamard_hi if which == "k" else self.v_hadamard_hi
            h_lo = self.k_hadamard_lo if which == "k" else self.v_hadamard_lo
            recon = turboquant_dequantize_mixed(q, h_hi, h_lo, self.dtype)
        else:
            flat_norms = sel_norms.reshape(-1)
            q = {
                "packed_indices": flat_packed,
                "norms": flat_norms,
                "padded_dim": self.padded_head_dim if which == "k" else self.v_padded_head_dim,
            }
            if self.mode == "prod":
                qjl_buf = self.k_qjl_buffer[idx] if which == "k" else self.v_qjl_buffer[idx]
                res_buf = self.k_residual_norms_buffer[idx] if which == "k" else self.v_residual_norms_buffer[idx]
                q["qjl_signs"] = qjl_buf[indices].reshape(-1, qjl_buf.shape[-1])
                q["residual_norms"] = res_buf[indices].reshape(-1)
            h = self.k_hadamard if which == "k" else self.v_hadamard
            recon = turboquant_dequantize(q, h, int(self.bits), self.mode, self.dtype)

        recon = recon[:, :out_dim].reshape(n, self.head_num, out_dim)
        ws[indices] = recon

    def _get_key_buffer(self, layer_id: int):
        """Dequant active positions into shared workspace and return it."""
        if self._active_indices is not None and self._active_indices.numel() > 0:
            self._dequant_positions(layer_id, "k")
        return self._k_ws

    def _get_value_buffer(self, layer_id: int):
        """Dequant active positions into shared workspace and return it."""
        if self._active_indices is not None and self._active_indices.numel() > 0:
            self._dequant_positions(layer_id, "v")
        return self._v_ws

    # ── Write path: quantize and store compressed only ──

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Quantize and store compressed KV entries."""
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        idx = layer_id - self.start_layer
        num_tokens = cache_k.shape[0]

        k_flat = cache_k.reshape(-1, self.head_dim)
        v_flat = cache_v.reshape(-1, self.v_head_dim)

        # Quantize
        if self.is_mixed:
            k_q = turboquant_quantize_mixed(
                k_flat, self.k_hadamard_hi, self.k_hadamard_lo,
                self.bits_hi, self.bits_lo, self._k_split_dim,
            )
            v_q = turboquant_quantize_mixed(
                v_flat, self.v_hadamard_hi, self.v_hadamard_lo,
                self.bits_hi, self.bits_lo, self._v_split_dim,
            )
        else:
            k_q = turboquant_quantize(k_flat, self.k_hadamard, int(self.bits), self.mode)
            v_q = turboquant_quantize(v_flat, self.v_hadamard, int(self.bits), self.mode)

        # Store compressed
        if self.is_mixed:
            packed_k = torch.cat([k_q["packed_hi"], k_q["packed_lo"]], dim=-1)
            packed_v = torch.cat([v_q["packed_hi"], v_q["packed_lo"]], dim=-1)
            self.k_buffer[idx][loc] = packed_k.reshape(num_tokens, self.head_num, -1)
            self.v_buffer[idx][loc] = packed_v.reshape(num_tokens, self.head_num, -1)
            self.k_norms_buffer[idx][loc] = torch.stack(
                [k_q["norms_hi"], k_q["norms_lo"]], dim=-1
            ).reshape(num_tokens, self.head_num, 2)
            self.v_norms_buffer[idx][loc] = torch.stack(
                [v_q["norms_hi"], v_q["norms_lo"]], dim=-1
            ).reshape(num_tokens, self.head_num, 2)
        else:
            self.k_buffer[idx][loc] = k_q["packed_indices"].reshape(num_tokens, self.head_num, -1)
            self.v_buffer[idx][loc] = v_q["packed_indices"].reshape(num_tokens, self.head_num, -1)
            self.k_norms_buffer[idx][loc] = k_q["norms"].reshape(num_tokens, self.head_num)
            self.v_norms_buffer[idx][loc] = v_q["norms"].reshape(num_tokens, self.head_num)

        if not self.is_mixed and self.mode == "prod":
            self.k_qjl_buffer[idx][loc] = k_q["qjl_signs"].reshape(num_tokens, self.head_num, -1)
            self.v_qjl_buffer[idx][loc] = v_q["qjl_signs"].reshape(num_tokens, self.head_num, -1)
            self.k_residual_norms_buffer[idx][loc] = k_q["residual_norms"].reshape(num_tokens, self.head_num)
            self.v_residual_norms_buffer[idx][loc] = v_q["residual_norms"].reshape(num_tokens, self.head_num)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        for i in range(self.layer_num):
            self.k_buffer[i][tgt_loc] = self.k_buffer[i][src_loc]
            self.v_buffer[i][tgt_loc] = self.v_buffer[i][src_loc]
            self.k_norms_buffer[i][tgt_loc] = self.k_norms_buffer[i][src_loc]
            self.v_norms_buffer[i][tgt_loc] = self.v_norms_buffer[i][src_loc]
            if self.mode == "prod":
                self.k_qjl_buffer[i][tgt_loc] = self.k_qjl_buffer[i][src_loc]
                self.v_qjl_buffer[i][tgt_loc] = self.v_qjl_buffer[i][src_loc]
                self.k_residual_norms_buffer[i][tgt_loc] = self.k_residual_norms_buffer[i][src_loc]
                self.v_residual_norms_buffer[i][tgt_loc] = self.v_residual_norms_buffer[i][src_loc]
