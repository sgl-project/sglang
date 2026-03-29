"""TurboQuant quantization for SGLang: weight quantization with on-the-fly dequant.

Integrates TurboQuant (random rotation + Lloyd-Max scalar quantization) into SGLang's
quantization framework. Supports single-pass 4-bit and residual 4+4 (8-bit total).

Usage:
    model_config.quantization = "turboquant"
    # In quantization_config (HF or sidecar JSON):
    {
        "quant_method": "turboquant",
        "bit_width": 4,
        "group_size": 128,
        "seed": 42,
        "residual_bit_width": 4,   // optional, for 4+4 lossless
        "residual_seed": 1042
    }
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.turboquant_engine import (
    generate_rotation_matrix,
    get_codebook,
    pack_4bit,
    turboquant_matmul_pytorch,
    unpack_4bit,
)
from sglang.srt.layers.quantization.turboquant_triton import (
    has_triton,
    triton_fused_dual_pass_matmul,
    triton_fused_matmul,
)

logger = logging.getLogger(__name__)


class TurboquantConfig(QuantizationConfig):
    """Configuration for TurboQuant weight quantization."""

    def __init__(
        self,
        bit_width: int = 4,
        group_size: int = 128,
        seed: int = 42,
        residual_bit_width: Optional[int] = None,
        residual_seed: int = 1042,
    ):
        self.bit_width = bit_width
        self.group_size = group_size
        self.seed = seed
        self.residual_bit_width = residual_bit_width
        self.residual_seed = residual_seed

    def __repr__(self) -> str:
        bits = f"{self.bit_width}"
        if self.residual_bit_width:
            bits += f"+{self.residual_bit_width}"
        return f"TurboquantConfig(bits={bits}, group_size={self.group_size})"

    @classmethod
    def get_name(cls) -> str:
        return "turboquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # No GPU capability requirement (works on ROCm/CUDA)

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["turboquant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TurboquantConfig":
        return cls(
            bit_width=cls.get_from_keys(config, ["bit_width", "bits"]),
            group_size=cls.get_from_keys_or(config, ["group_size"], 128),
            seed=cls.get_from_keys_or(config, ["seed"], 42),
            residual_bit_width=cls.get_from_keys_or(
                config, ["residual_bit_width"], None
            ),
            residual_seed=cls.get_from_keys_or(config, ["residual_seed"], 1042),
        )

    def get_quant_method(
        self, layer: nn.Module, prefix: str = ""
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return TurboquantLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @property
    def has_residual(self) -> bool:
        return self.residual_bit_width is not None

    @property
    def total_bits(self) -> int:
        return self.bit_width + (self.residual_bit_width or 0)


class TurboquantLinearMethod(LinearMethodBase):
    """Linear method with TurboQuant on-the-fly 4-bit dequantization.

    Storage per layer:
      - indices_packed: (M, N//2) uint8 — packed 4-bit quantization indices
      - weight_norms: (M,) or (M, n_groups) float32 — per-row L2 norms
      - codebook: (16,) float32 — Lloyd-Max centroids
      - [optional] pass2 buffers for residual quantization
    """

    def __init__(self, quant_config: TurboquantConfig):
        self.quant_config = quant_config
        self._use_triton = has_triton()
        self._rotation_cache: Dict[int, torch.Tensor] = {}

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        K = input_size_per_partition
        N = output_size_per_partition
        group_size = self.quant_config.group_size
        bit_width = self.quant_config.bit_width
        n_levels = 2 ** bit_width
        n_groups = math.ceil(K / group_size)
        packed_dim = math.ceil(K / 2)

        # Pass 1 buffers
        layer.register_buffer(
            "tq_indices_packed",
            torch.zeros(N, packed_dim, dtype=torch.uint8),
        )
        layer.register_buffer(
            "tq_codebook",
            torch.zeros(n_levels, dtype=torch.float32),
        )
        if n_groups == 1:
            layer.register_buffer(
                "tq_weight_norms",
                torch.ones(N, dtype=torch.float32),
            )
        else:
            layer.register_buffer(
                "tq_weight_norms",
                torch.ones(N, n_groups, dtype=torch.float32),
            )

        # Pass 2 (residual) buffers
        if self.quant_config.has_residual:
            r_levels = 2 ** self.quant_config.residual_bit_width
            layer.register_buffer(
                "tq_pass2_indices_packed",
                torch.zeros(N, packed_dim, dtype=torch.uint8),
            )
            layer.register_buffer(
                "tq_pass2_codebook",
                torch.zeros(r_levels, dtype=torch.float32),
            )
            if n_groups == 1:
                layer.register_buffer(
                    "tq_pass2_weight_norms",
                    torch.ones(N, dtype=torch.float32),
                )
            else:
                layer.register_buffer(
                    "tq_pass2_weight_norms",
                    torch.ones(N, n_groups, dtype=torch.float32),
                )

        # Metadata
        layer.tq_in_features = K
        layer.tq_out_features = N
        layer.tq_group_size = group_size
        layer.tq_seed = self.quant_config.seed
        layer.tq_has_residual = self.quant_config.has_residual
        layer.tq_residual_seed = self.quant_config.residual_seed
        layer.tq_scale = math.sqrt(group_size)

        # We also create a standard weight for loading from HF checkpoints,
        # which will be converted in process_weights_after_loading.
        weight = torch.nn.Parameter(
            torch.empty(N, K, dtype=params_dtype), requires_grad=False
        )
        layer.register_parameter("weight", weight)

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is not None:
            weight.weight_loader = weight_loader

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Convert loaded fp16/bf16 weights into TurboQuant packed format."""
        if not hasattr(layer, "weight") or layer.weight is None:
            return

        W = layer.weight.data
        if W.numel() == 0:
            return

        device = W.device
        M, N = W.shape
        group_size = layer.tq_group_size
        seed = layer.tq_seed
        bit_width = self.quant_config.bit_width

        centroids, boundaries = get_codebook(bit_width)
        centroids_dev = centroids.to(device)
        boundaries_dev = boundaries.to(device)

        # Quantize pass 1
        packed, norms = self._quantize_weight(
            W, bit_width, group_size, seed, centroids_dev, boundaries_dev
        )
        layer.tq_indices_packed.copy_(packed)
        layer.tq_weight_norms.copy_(norms)
        layer.tq_codebook.copy_(centroids_dev)

        # Residual pass 2
        if layer.tq_has_residual:
            W_hat1 = self._dequantize_weight(
                packed, norms, centroids_dev, N, group_size, seed, device
            )
            residual = W.float() - W_hat1

            r_bit = self.quant_config.residual_bit_width
            r_seed = layer.tq_residual_seed
            r_centroids, r_boundaries = get_codebook(r_bit)
            r_centroids_dev = r_centroids.to(device)
            r_boundaries_dev = r_boundaries.to(device)

            r_packed, r_norms = self._quantize_weight(
                residual, r_bit, group_size, r_seed, r_centroids_dev, r_boundaries_dev
            )
            layer.tq_pass2_indices_packed.copy_(r_packed)
            layer.tq_pass2_weight_norms.copy_(r_norms)
            layer.tq_pass2_codebook.copy_(r_centroids_dev)

        # Free the original weight to save memory
        layer.weight = None

    def _quantize_weight(
        self,
        W: torch.Tensor,
        bit_width: int,
        group_size: int,
        seed: int,
        centroids: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight matrix → (indices_packed, norms)."""
        M, N_w = W.shape
        W = W.float()
        device = W.device
        all_norms = []
        all_indices = []

        for g_start in range(0, N_w, group_size):
            g_end = min(g_start + group_size, N_w)
            g_dim = g_end - g_start
            W_g = W[:, g_start:g_end]

            norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
            W_norm = W_g / norms
            all_norms.append(norms.squeeze(1))

            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
            Y = W_norm @ Pi.T
            scale = math.sqrt(g_dim)
            Y_scaled = Y * scale

            indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
            indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
            all_indices.append(indices)

        full_indices = torch.cat(all_indices, dim=1)
        norms_out = (
            torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]
        )

        if N_w % 2 != 0:
            full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

        packed = pack_4bit(full_indices)
        return packed, norms_out

    def _dequantize_weight(
        self,
        indices_packed: torch.Tensor,
        norms: torch.Tensor,
        codebook: torch.Tensor,
        in_features: int,
        group_size: int,
        seed: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Dequantize for residual computation."""
        M = indices_packed.shape[0]
        N = in_features
        padded_N = N if N % 2 == 0 else N + 1
        indices = unpack_4bit(indices_packed, padded_N)[:, :N]
        n_groups = math.ceil(N / group_size)
        scale = math.sqrt(group_size)

        W_approx = torch.zeros(M, N, dtype=torch.float32, device=device)
        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(g_start + group_size, N)
            g_dim = g_end - g_start

            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
            Y_g = codebook[indices[:, g_start:g_end].long()] / scale
            W_g = Y_g @ Pi

            if norms.dim() == 1:
                W_g = W_g * norms.unsqueeze(1)
            else:
                W_g = W_g * norms[:, g].unsqueeze(1)

            W_approx[:, g_start:g_end] = W_g

        return W_approx

    def _get_rotation(self, dim: int, seed: int, device: torch.device) -> torch.Tensor:
        key = (dim, seed)
        if key not in self._rotation_cache:
            self._rotation_cache[key] = generate_rotation_matrix(dim, seed).to(device)
        Pi = self._rotation_cache[key]
        if Pi.device != device:
            Pi = Pi.to(device)
            self._rotation_cache[key] = Pi
        return Pi

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """On-the-fly dequant forward: rotate input → fused matmul → rescale."""
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        device = x.device
        K = layer.tq_in_features
        group_size = layer.tq_group_size
        n_groups = math.ceil(K / group_size)

        x_f = x.float()
        output = torch.zeros(
            x_f.shape[0], layer.tq_out_features, dtype=torch.float32, device=device
        )

        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(g_start + group_size, K)
            g_dim = g_end - g_start

            Pi_g = self._get_rotation(g_dim, layer.tq_seed + g_start, device)
            x_rot_g = x_f[:, g_start:g_end] @ Pi_g.T

            if layer.tq_weight_norms.dim() == 1:
                norms_g = layer.tq_weight_norms
            else:
                norms_g = layer.tq_weight_norms[:, g]

            use_triton = self._use_triton and g_dim == group_size and x_f.is_cuda
            if use_triton:
                packed_g = layer.tq_indices_packed[:, g_start // 2 : g_end // 2]
                out_g = triton_fused_matmul(
                    x_rot_g.contiguous(),
                    packed_g.contiguous(),
                    layer.tq_codebook,
                    norms_g.contiguous(),
                    g_dim,
                    layer.tq_scale,
                )
            else:
                padded = K if K % 2 == 0 else K + 1
                indices = unpack_4bit(layer.tq_indices_packed, padded)[:, :K]
                idx_g = indices[:, g_start:g_end]
                W_g = layer.tq_codebook[idx_g.long()]
                out_g = x_rot_g @ W_g.T
                out_g = out_g * (norms_g[None, :] / layer.tq_scale)

            output += out_g

        # Residual pass 2
        if layer.tq_has_residual:
            for g in range(n_groups):
                g_start = g * group_size
                g_end = min(g_start + group_size, K)
                g_dim = g_end - g_start

                Pi2_g = self._get_rotation(
                    g_dim, layer.tq_residual_seed + g_start, device
                )
                x_rot2_g = x_f[:, g_start:g_end] @ Pi2_g.T

                if layer.tq_pass2_weight_norms.dim() == 1:
                    norms2_g = layer.tq_pass2_weight_norms
                else:
                    norms2_g = layer.tq_pass2_weight_norms[:, g]

                use_triton2 = self._use_triton and g_dim == group_size and x_f.is_cuda
                if use_triton2:
                    packed2_g = layer.tq_pass2_indices_packed[
                        :, g_start // 2 : g_end // 2
                    ]
                    out2_g = triton_fused_matmul(
                        x_rot2_g.contiguous(),
                        packed2_g.contiguous(),
                        layer.tq_pass2_codebook,
                        norms2_g.contiguous(),
                        g_dim,
                        layer.tq_scale,
                    )
                else:
                    padded = K if K % 2 == 0 else K + 1
                    idx2 = unpack_4bit(layer.tq_pass2_indices_packed, padded)[:, :K]
                    idx2_g = idx2[:, g_start:g_end]
                    W2_g = layer.tq_pass2_codebook[idx2_g.long()]
                    out2_g = x_rot2_g @ W2_g.T
                    out2_g = out2_g * (norms2_g[None, :] / layer.tq_scale)

                output += out2_g

        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], layer.tq_out_features)

        result = output.to(x.dtype)
        if bias is not None:
            result = result + bias.to(result.dtype)
        return result
