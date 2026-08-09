import math
from typing import Any

# deep_gemm and sgl_kernel
import deep_gemm
import torch
import torch.nn.functional as F
from deep_gemm import (
    get_mn_major_tma_aligned_tensor,
    per_block_cast_to_fp8,
)
from sgl_kernel import silu_and_mul
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from torch import nn
from torch.nn import LayerNorm
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .dots_omni_vision_attention import (
    VisionAttention,
    VisionRotaryEmbedding,
    apply_vision_attention_residual,
)


class DotsMoEVitConfig(PretrainedConfig):
    model_type: str = "dots_moe_vit"

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_size: int = 2048,
        intermediate_size: int = 4224,
        moe_intermediate_size: int = 2112,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 24,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        use_qk_norm: bool = True,
        attn_implementation="flash_attention_3",
        initializer_range=0.02,
        is_causal=False,
        post_norm=True,
        gradient_checkpointing=False,
        pyramid_num_routed: list[int] | None = None,
        capacity_factor: float = 2.0,
        router_scoring_func: str = "sigmoid",
        router_scale: float = 1.0,
        adapter_in_dim: int = 1536,
        adapter_out_dim: int = 2048,
        adapter_merge_size: int = 2,
        # ``pixel_shuffle_mlp`` (legacy, e.g. dotsvlm2.lite.omni_fp8) reshapes 2x2 spatial neighbours via NHWC pixel-shuffle then runs LN+2-layer MLP.
        # ``patch_merger`` (cybertron PatchMerger, e.g. fireall_iter02275) skips the pixel-shuffle permutation and instead views every 4 consecutive 2x2-grouped tokens as one row.
        adapter_type: str = "pixel_shuffle_mlp",
        # If True the preprocessor already emits patches in 2x2-grouped order (qwen ``merge_size=2`` flatten path), so RoPE positions must also be regrouped
        # to match the in-block layout. Old checkpoints (e.g. omni_fp8) were trained with ``pre_pixel_shuffle=False`` even though the data is grouped, so the legacy default keeps row-major RoPE.
        pre_pixel_shuffle: bool = False,
        # If True, use FP8 MoE implementation
        enable_fp8_moe: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.use_qk_norm = use_qk_norm
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.pyramid_num_routed = pyramid_num_routed or []
        self.capacity_factor = capacity_factor
        self.router_scoring_func = router_scoring_func
        self.router_scale = router_scale
        self.adapter_in_dim = adapter_in_dim
        self.adapter_out_dim = adapter_out_dim
        self.adapter_merge_size = adapter_merge_size
        if adapter_type not in ("pixel_shuffle_mlp", "patch_merger"):
            raise ValueError(
                f"adapter_type must be 'pixel_shuffle_mlp' or 'patch_merger', got {adapter_type!r}"
            )
        self.adapter_type = adapter_type
        self.pre_pixel_shuffle = pre_pixel_shuffle
        self.enable_fp8_moe = enable_fp8_moe


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ---- FFN modules ----


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features, bias=False):
        super().__init__()
        self.fc13 = nn.Linear(in_features, hidden_features * 2, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.register_load_state_dict_pre_hook(
            DotsSwiGLUFFN._load_fused_fc13_from_split
        )

    @staticmethod
    def _load_fused_fc13_from_split(
        module: "DotsSwiGLUFFN",
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        fc13_weight_key = prefix + "fc13.weight"
        fc1_weight_key = prefix + "fc1.weight"
        fc3_weight_key = prefix + "fc3.weight"
        fc1_weight = state_dict.get(fc1_weight_key)
        fc3_weight = state_dict.get(fc3_weight_key)
        if fc1_weight is not None and fc3_weight is not None:
            if fc13_weight_key not in state_dict:
                state_dict[fc13_weight_key] = torch.cat((fc1_weight, fc3_weight), dim=0)
            state_dict.pop(fc1_weight_key)
            state_dict.pop(fc3_weight_key)

        fc13_bias_key = prefix + "fc13.bias"
        fc1_bias_key = prefix + "fc1.bias"
        fc3_bias_key = prefix + "fc3.bias"
        fc1_bias = state_dict.get(fc1_bias_key)
        fc3_bias = state_dict.get(fc3_bias_key)
        if fc1_bias is not None and fc3_bias is not None:
            if module.fc13.bias is not None and fc13_bias_key not in state_dict:
                state_dict[fc13_bias_key] = torch.cat((fc1_bias, fc3_bias), dim=0)
            state_dict.pop(fc1_bias_key)
            state_dict.pop(fc3_bias_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc13(x)
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)
        x = self.fc2(out)
        return x


def _ceil_to_multiple(v: int, multiple: int) -> int:
    return ((v + multiple - 1) // multiple) * multiple


def _per_block_cast_to_fp8_padded(
    x: torch.Tensor,
    *,
    use_ue8m0: bool = False,
    gran_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`per_block_cast_to_fp8` with zero-padding for non-divisible 2D tensors.

    DeepGEMM block FP8 path expects block-aligned dimensions. When `x.shape` is
    not divisible by `gran_k`, this helper pads zeros on both dims to the nearest
    multiple and then calls `per_block_cast_to_fp8`.
    """
    if x.dim() != 2:
        raise ValueError(f"expected 2D tensor, got shape={tuple(x.shape)}")
    if gran_k <= 0:
        raise ValueError(f"gran_k must be positive, got {gran_k}")

    m, n = int(x.shape[0]), int(x.shape[1])
    m_pad = _ceil_to_multiple(m, gran_k)
    n_pad = _ceil_to_multiple(n, gran_k)

    if m_pad == m and n_pad == n:
        return per_block_cast_to_fp8(x.contiguous(), use_ue8m0=use_ue8m0, gran_k=gran_k)

    x_pad = torch.zeros((m_pad, n_pad), dtype=x.dtype, device=x.device)
    x_pad[:m, :n] = x
    return per_block_cast_to_fp8(x_pad.contiguous(), use_ue8m0=use_ue8m0, gran_k=gran_k)


def _fp8_blockwise_linear(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    out_features: int,
    x_scale_col_major: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepGEMM implementation of FP8 GEMM"""
    m, n = x_bf16.shape[0], out_features
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    if x_scale_col_major is None:
        x_fp8, x_scale = sglang_per_token_group_quant_fp8(x_bf16, group_size=128)
        x_scale_col_major = get_mn_major_tma_aligned_tensor(x_scale)
    else:
        x_fp8 = x_bf16

    # Run DeepGEMM kernel
    deep_gemm.fp8_gemm_nt((x_fp8, x_scale_col_major), (w_fp8, w_scale), out)

    if bias is not None:
        out = out + bias
    return out


class DotsSwiGLUFFNFP8(nn.Module):
    """SwiGLU FFN with FP8 weights (128×128 blocks) and per-token group-128 activation quant; bf16 output.

    ``nn.Linear`` siblings keep standard ``fc{1,2,3}.weight`` keys for bf16 checkpoints; after each
    ``load_state_dict`` the FP8 buffers are rebuilt from those weights.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        block_size: int = 128,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.block_size = block_size
        self.group_size = group_size
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.register_load_state_dict_post_hook(DotsSwiGLUFFNFP8._post_load_repack_fp8)
        self._repack_fp8_weights()

    @staticmethod
    def _post_load_repack_fp8(module: "DotsSwiGLUFFNFP8", _incompatible_keys) -> None:
        module._repack_fp8_weights()

    @torch.no_grad()
    def _repack_fp8_weights(self) -> None:
        fc13_weight = torch.cat((self.fc1.weight, self.fc3.weight), dim=0).contiguous()
        fc13_w_fp8, fc13_w_scale = _per_block_cast_to_fp8_padded(
            fc13_weight, use_ue8m0=False, gran_k=self.block_size
        )
        fc2_w_fp8, fc2_w_scale = _per_block_cast_to_fp8_padded(
            self.fc2.weight, use_ue8m0=False, gran_k=self.block_size
        )
        # Non-persistent: bf16 checkpoints only store ``fc*.weight`` / ``fc*.bias``.
        self.register_buffer("_fc13_w_fp8", fc13_w_fp8, persistent=False)
        self.register_buffer("_fc13_w_scale", fc13_w_scale, persistent=False)
        self.register_buffer("_fc2_w_fp8", fc2_w_fp8, persistent=False)
        self.register_buffer("_fc2_w_scale", fc2_w_scale, persistent=False)

    def _fc13_bias(self) -> torch.Tensor | None:
        if self.fc1.bias is None:
            return None
        return torch.cat((self.fc1.bias, self.fc3.bias), dim=0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h13 = _fp8_blockwise_linear(
            x,
            self._fc13_w_fp8,
            self._fc13_w_scale,
            out_features=self.hidden_features * 2,
            bias=self._fc13_bias(),
        )
        h1, h3 = h13.chunk(2, dim=-1)
        return _fp8_blockwise_linear(
            F.silu(h1) * h3,
            self._fc2_w_fp8,
            self._fc2_w_scale,
            out_features=self.in_features,
            bias=self.fc2.bias,
        )


class MoESwiGLUFFN(nn.Module):
    """MoE FFN with per-expert SwiGLU experts, sigmoid/softmax gating, top-k routing."""

    def __init__(self, config: DotsMoEVitConfig, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.hidden_size = config.embed_dim
        self.num_routed = config.pyramid_num_routed[layer_number]
        self.capacity_factor = config.capacity_factor
        self.router_scoring_func = config.router_scoring_func
        self.router_scale = config.router_scale

        self.register_buffer(
            "router_bias", torch.zeros(self.num_routed, dtype=torch.float32)
        )

        self.experts = nn.ModuleList(
            [
                DotsSwiGLUFFN(
                    self.hidden_size, config.moe_intermediate_size, bias=config.use_bias
                )
                for _ in range(self.num_routed)
            ]
        )

        self.gate_weight = nn.Parameter(
            torch.empty((self.num_routed, self.hidden_size), dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror cybertron's ``AIMv2MoEGEMMSwiGLUFFN.forward``: keep ``gating_prob`` /
        # router-bias add in fp32 and run topk in fp32. The legacy ``.type_as(x_flat)``
        # path on bf16 made expert routing diverge whenever two routed-expert scores
        # tied within bf16 precision, which is the dominant source of numerical drift
        # observed against latest cybertron checkpoints.
        epsilon = 1e-9
        x_flat = x.contiguous()
        num_tokens = x_flat.shape[0]

        gate_logits = F.linear(x_flat.float(), self.gate_weight.float())

        if self.router_scoring_func == "sigmoid":
            gating_prob = torch.sigmoid(gate_logits)
        else:
            gating_prob = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)

        aggregated_output = torch.zeros_like(x_flat)
        aggregated_gate = torch.zeros(num_tokens, dtype=x.dtype, device=x.device)

        topk = min(int(self.capacity_factor), self.num_routed)

        gating_with_bias = gating_prob + self.router_bias.to(torch.float32).unsqueeze(0)
        _, topk_indices = torch.topk(gating_with_bias, k=topk, dim=-1, sorted=False)

        routed_weights = gating_prob.gather(1, topk_indices)
        if self.router_scoring_func == "sigmoid" and topk > 1:
            routed_weights = routed_weights / (
                routed_weights.sum(dim=-1, keepdim=True) + epsilon
            )
        routed_weights = (routed_weights * self.router_scale).to(x_flat.dtype)

        for expert_idx in range(self.num_routed):
            selected_mask = topk_indices == expert_idx
            if selected_mask.sum() == 0:
                continue
            n_idx, top = torch.where(selected_mask)
            # Fancy indexing can yield non-contiguous rows; cuBLAS bf16 GEMM may then fail
            # with ``CUBLAS_STATUS_INVALID_VALUE`` inside ``F.linear``.
            x_selected = x_flat[n_idx].contiguous()
            expert_output = self.experts[expert_idx](x_selected)
            contrib = expert_output * routed_weights[n_idx, top].unsqueeze(-1)
            aggregated_output[n_idx] = aggregated_output[n_idx] + contrib
            aggregated_gate[n_idx] = aggregated_gate[n_idx] + routed_weights[n_idx, top]

        aggregated_output = aggregated_output / (
            aggregated_gate.unsqueeze(-1) + epsilon
        )
        return aggregated_output


class MoESwiGLUFFNFP8(MoESwiGLUFFN):
    """Same routing and checkpoint layout as :class:`MoESwiGLUFFN` (bf16 ``experts.*.fc13``).

    Expert compute uses SGLang's fused MoE Triton path (same stack as
    :class:`sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE` + block FP8), so
    there is no per-expert Python dispatch loop. Stacked gate/up and down weights
    are repacked from the linears after each ``load_state_dict`` (same hook pattern
    as :class:`DotsSwiGLUFFNFP8`).

    Requires CUDA, ``sglang`` (with ``torch.ops.sglang`` fused MoE), and
    ``deep_gemm`` for 128×128 block FP8 quantization of expert weights.
    """

    def __init__(self, config: DotsMoEVitConfig, layer_number: int):
        super().__init__(config, layer_number)
        # ``MoESwiGLUFFN`` already builds ``DotsSwiGLUFFN`` experts.
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

        self._moe_runner_config = MoeRunnerConfig(inplace=False)
        self.register_buffer("_fused_w13_fp8", None, persistent=False)
        self.register_buffer("_fused_w13_scale", None, persistent=False)
        self.register_buffer("_fused_w2_fp8", None, persistent=False)
        self.register_buffer("_fused_w2_scale", None, persistent=False)
        self.register_load_state_dict_post_hook(
            MoESwiGLUFFNFP8._post_load_pack_fused_fp8
        )

    @staticmethod
    def _post_load_pack_fused_fp8(
        module: "MoESwiGLUFFNFP8", _incompatible_keys
    ) -> None:
        module._pack_fused_fp8_weights()

    @torch.no_grad()
    def _pack_fused_fp8_weights(self) -> None:
        """Stack gate+up per expert, block-quantize with DeepGEMM (128×128), layout for ``fused_moe``."""
        e_list = list(self.experts)
        if not e_list:
            return
        w13_chunks: list[torch.Tensor] = []
        s13_chunks: list[torch.Tensor] = []
        w2_chunks: list[torch.Tensor] = []
        s2_chunks: list[torch.Tensor] = []
        for ex in e_list:
            # Block-quantize gate and up separately, then stack for fused MoE w13.
            w1_weight, w3_weight = ex.fc13.weight.detach().chunk(2, dim=0)
            w1_bf16 = w1_weight.to(torch.bfloat16)
            w3_bf16 = w3_weight.to(torch.bfloat16)
            q1, s1 = _per_block_cast_to_fp8_padded(w1_bf16, use_ue8m0=False, gran_k=128)
            q3, s3 = _per_block_cast_to_fp8_padded(w3_bf16, use_ue8m0=False, gran_k=128)
            w13_fp8 = torch.cat([q1, q3], dim=0).contiguous()
            s13 = torch.cat([s1, s3], dim=0).contiguous()
            w13_chunks.append(w13_fp8)
            s13_chunks.append(s13)

            w2_bf16 = ex.fc2.weight.detach().to(torch.bfloat16)
            q2, s2 = _per_block_cast_to_fp8_padded(w2_bf16, use_ue8m0=False, gran_k=128)
            w2_chunks.append(q2.contiguous())
            s2_chunks.append(s2)

        self._fused_w13_fp8 = torch.stack(w13_chunks, dim=0).contiguous()
        self._fused_w13_scale = torch.stack(s13_chunks, dim=0).contiguous()
        self._fused_w2_fp8 = torch.stack(w2_chunks, dim=0).contiguous()
        self._fused_w2_scale = torch.stack(s2_chunks, dim=0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        # Mirror cybertron's ``AIMv2MoEGEMMSwiGLUFFN.forward``: routing decisions stay in fp32 to
        # avoid topk ties under bf16 precision (see :class:`MoESwiGLUFFN`).
        epsilon = 1e-9
        x_flat = x.contiguous()

        gate_logits = F.linear(x_flat.float(), self.gate_weight.float())

        if self.router_scoring_func == "sigmoid":
            gating_prob = torch.sigmoid(gate_logits)
        else:
            gating_prob = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)

        topk = min(int(self.capacity_factor), self.num_routed)
        gating_with_bias = gating_prob + self.router_bias.to(torch.float32).unsqueeze(0)

        _, topk_indices = torch.topk(gating_with_bias, k=topk, dim=-1, sorted=False)

        routed_weights = gating_prob.gather(1, topk_indices)
        if self.router_scoring_func == "sigmoid" and topk > 1:
            routed_weights = routed_weights / (
                routed_weights.sum(dim=-1, keepdim=True) + epsilon
            )
        routed_weights = routed_weights * float(self.router_scale)

        topk_ids = topk_indices.to(torch.int32)
        topk_output = StandardTopKOutput(routed_weights, topk_ids, gate_logits)

        if self._fused_w13_fp8 is None:
            self._pack_fused_fp8_weights()

        b1 = b2 = None
        if self.config.use_bias:
            b1_list = []
            b2_list = []
            for ex in self.experts:
                b1_list.append(ex.fc13.bias.detach().to(x.dtype))
                b2_list.append(ex.fc2.bias.detach().to(x.dtype))
            b1 = torch.stack(b1_list, dim=0).contiguous()
            b2 = torch.stack(b2_list, dim=0).contiguous()

        fused_out = fused_moe(
            x_flat,
            self._fused_w13_fp8,
            self._fused_w2_fp8,
            topk_output,
            moe_runner_config=self._moe_runner_config,
            b1=b1,
            b2=b2,
            use_fp8_w8a8=True,
            w1_scale=self._fused_w13_scale,
            w2_scale=self._fused_w2_scale,
            block_shape=[128, 128],
        )
        denom = routed_weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
        return (fused_out / denom).type_as(x)


# ---- PatchEmbed ----


class DotsPatchEmbed(nn.Module):
    def __init__(self, config: DotsMoEVitConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


# ---- Block ----


class MoEVisionBlock(nn.Module):
    def __init__(self, config: DotsMoEVitConfig, layer_number: int):
        super().__init__()
        self.attn = VisionAttention(config)
        self.norm_1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.norm_2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        is_moe = (
            config.pyramid_num_routed
            and layer_number < len(config.pyramid_num_routed)
            and config.pyramid_num_routed[layer_number] > 0
        )
        if is_moe and config.enable_fp8_moe:
            self.mlp = MoESwiGLUFFNFP8(config, layer_number)
        elif is_moe:
            self.mlp = MoESwiGLUFFN(config, layer_number)
        else:
            self.mlp = DotsSwiGLUFFN(
                config.embed_dim, config.intermediate_size, bias=config.use_bias
            )

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = apply_vision_attention_residual(
            self.attn,
            self.norm_1,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states


# ---- Adapter (pixel_shuffle + MLP) ----


def _pixel_shuffle(x, scale_factor=0.5):
    if x.size(1) % 2 == 1:
        x = torch.cat([x[:, :1], x], dim=1)
    if x.size(2) % 2 == 1:
        x = torch.cat([x[:, :, :1], x], dim=2)
    n, h, w, c = x.size()
    x = x.reshape(n, h, int(w * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.reshape(
        n,
        int(w * scale_factor),
        int(h * scale_factor),
        int(c / (scale_factor * scale_factor)),
    )
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class PixelShuffleAdapter(nn.Module):
    """Legacy adapter: NHWC pixel-shuffle spatial merge + LayerNorm + 2-layer MLP.

    Mirrors ``cybertron`` ``FCAdapter(pool_kind='pixel_shuffle', proj_kind='mlp2x_ln_gelu')``.
    State-dict keys: ``proj.0`` (LayerNorm of in_dim*merge**2), ``proj.1`` / ``proj.3`` (Linear).
    """

    def __init__(self, config: DotsMoEVitConfig):
        super().__init__()
        in_dim = config.adapter_in_dim
        out_dim = config.adapter_out_dim
        merge_size = config.adapter_merge_size
        merged_dim = in_dim * merge_size**2
        self.proj = nn.Sequential(
            LayerNorm(merged_dim),
            nn.Linear(merged_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        patch_embed: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        assert patch_embed.dim() == 2 and grid_thw is not None
        image_features = []
        token_index = 0
        for i in range(grid_thw.shape[0]):
            grid_t, grid_h, grid_w = grid_thw[i]
            images_token_length = grid_t * grid_h * grid_w
            _pe = patch_embed[token_index : token_index + images_token_length]
            token_index += images_token_length
            if grid_t == 1:
                _pe = _pe.reshape(int(grid_h), int(grid_w), -1).unsqueeze(0)
            else:
                _pe = _pe.reshape(int(grid_t), int(grid_h), int(grid_w), -1)
            _pe = _pixel_shuffle(_pe, scale_factor=0.5)
            if grid_t == 1:
                _pe = _pe.squeeze(0)
            else:
                _pe = _pe.reshape(-1, _pe.shape[-1])
            image_features.append(_pe.reshape(-1, _pe.shape[-1]))
        out = torch.cat(image_features, dim=0)
        out = self.proj(out)
        return out


class PatchMergerAdapter(nn.Module):
    """Cybertron ``PatchMerger`` (``pool_kind='patch_merger', proj_kind='identity'``).

    Assumes the encoder output is already laid out in ``merge_size``x``merge_size`` groups
    (qwen ``pre_pixel_shuffle`` preprocessor + RoPE grouped accordingly), so merging is a
    simple ``view(-1, merge**2 * in_dim)`` of consecutive tokens. State-dict layout matches
    cybertron's ``PatchMerger`` (``ln_q`` over the per-token dim, ``mlp.0`` / ``mlp.2`` Linear).
    """

    def __init__(self, config: DotsMoEVitConfig):
        super().__init__()
        in_dim = config.adapter_in_dim
        out_dim = config.adapter_out_dim
        merge_size = config.adapter_merge_size
        merged_dim = in_dim * merge_size**2
        self.merge_size = merge_size
        self.merged_dim = merged_dim
        self.ln_q = LayerNorm(in_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, out_dim),
        )

    def forward(
        self,
        patch_embed: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        assert patch_embed.dim() == 2 and grid_thw is not None
        x = self.ln_q(patch_embed)
        x = x.reshape(-1, self.merged_dim)
        return self.mlp(x)


_ADAPTER_CLASSES = {
    "pixel_shuffle_mlp": PixelShuffleAdapter,
    "patch_merger": PatchMergerAdapter,
}


# ---- Full Model ----


class DotsMoEVitModel(PreTrainedModel):
    config_class = DotsMoEVitConfig

    def __init__(self, config: DotsMoEVitConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsPatchEmbed(config)

        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, cache_seq_len=100000)

        self.blocks = nn.ModuleList(
            [MoEVisionBlock(config, i) for i in range(config.num_hidden_layers)]
        )

        if config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        adapter_cls = _ADAPTER_CLASSES.get(config.adapter_type)
        if adapter_cls is None:
            raise ValueError(f"Unknown adapter_type {config.adapter_type!r}")
        self.adapter = adapter_cls(config)

        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    @property
    def dtype(self) -> torch.dtype:
        mlp = self.blocks[0].mlp
        if isinstance(mlp, DotsSwiGLUFFN):
            return mlp.fc13.weight.dtype
        if isinstance(mlp, DotsSwiGLUFFNFP8):
            return mlp.fc1.weight.dtype
        expert = mlp.experts[0]
        if isinstance(expert, DotsSwiGLUFFN):
            return expert.fc13.weight.dtype
        return expert.fc1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def get_pos_ids_by_grid(self, grid_thw):
        # Mirrors ``cybertron`` ``AIMv2NativeModel.rot_pos_emb``: when ``pre_pixel_shuffle``
        # is set, RoPE positions follow the qwen ``merge_size`` grouped layout (default 2x2);
        # otherwise positions are flat row-major regardless of ``spatial_merge_size``.
        if self.config.pre_pixel_shuffle:
            rope_merge_size = (
                self.spatial_merge_size if self.spatial_merge_size > 1 else 2
            )
        else:
            rope_merge_size = 1
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def _build_cu_seqlens_from_grid(self, grid_thw: torch.Tensor):
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        # Same as (cu_seqlens[1:] - cu_seqlens[:-1]).max(); computed here to avoid D2H inside attention.
        max_seqlen = int((grid_thw[:, 1] * grid_thw[:, 2]).max().item())
        return cu_seqlens, max_seqlen

    def _build_single_temporal_cu_seqlens_from_grid(self, grid_thw: torch.Tensor):
        seq_lens = grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = torch.empty(
            (grid_thw.shape[0] + 1,), device=grid_thw.device, dtype=torch.int32
        )
        cu_seqlens[0] = 0
        torch.cumsum(seq_lens, dim=0, dtype=torch.int32, out=cu_seqlens[1:])
        max_seqlen = int(seq_lens.max().item())
        return cu_seqlens, max_seqlen

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, bf16=True
    ) -> torch.Tensor:
        if bf16:
            hidden_states = hidden_states.bfloat16()
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        if grid_thw[:, 0].sum().item() == grid_thw.shape[0]:
            cu_seqlens, max_seqlen = self._build_single_temporal_cu_seqlens_from_grid(
                grid_thw
            )
        else:
            cu_seqlens, max_seqlen = self._build_cu_seqlens_from_grid(grid_thw)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb,
                    max_seqlen,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb,
                    max_seqlen,
                )

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.adapter(hidden_states, grid_thw)
        return hidden_states
