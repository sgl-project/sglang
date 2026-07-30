"""Inference-only Mage-VL model for sglang."""

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix


class VisionRotaryEmbedding(nn.Module):
    """3D (T,H,W) RoPE with 4:6:6 split, ported from Mage-VL checkpoint.

    head_dim must be divisible by 16; half = head_dim // 2;
    unit = half // 16; t_size = 4*unit, h_size = 6*unit, w_size = 6*unit.
    """

    def __init__(self, config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        assert head_dim % 2 == 0 and head_dim % 16 == 0
        half = head_dim // 2
        assert half % 16 == 0
        unit = half // 16
        self.head_dim = head_dim
        self.half = half
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit
        self.base = base
        self.register_buffer(
            "inv_freq_t",
            1.0
            / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0
            / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0
            / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """patch_positions: [L, 3] long -> freqs: [L, half] float32.

        Force fp32 on inv_freq buffers: a module ``.to(bfloat16)`` (the standard
        sglang load path) would downcast these non-persistent buffers and
        truncate high-frequency entries. HF keeps inv_freq in fp32, so we mirror
        that here regardless of the load dtype.
        """
        pp = patch_positions.to(self.inv_freq_t.device)
        t, h, w = pp[:, 0].float(), pp[:, 1].float(), pp[:, 2].float()
        freqs_t = t.unsqueeze(-1) * self.inv_freq_t.float().unsqueeze(0)  # [L, t_size]
        freqs_h = h.unsqueeze(-1) * self.inv_freq_h.float().unsqueeze(0)
        freqs_w = w.unsqueeze(-1) * self.inv_freq_w.float().unsqueeze(0)
        return torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)  # [L, half]


def _mage_vl_rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """HF Mage-VL interleaved rotate_half implementation.

    sglang's default ``apply_rotary_pos_emb`` uses the half-split convention
    ``cat(-x[..., half:], x[..., :half])`` which pairs RoPE dimensions
    differently from HF Mage-VL (which interleaves adjacent dim pairs). The two are
    NOT numerically equivalent — using sglang's default here gives the wrong
    vision tower output (see HF ``modeling_mage_vl.py::rotate_half``).
    """
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


def _mage_vl_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    x_shape: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Customised position embedding applier for Mage-VL's VisionAttention.

    Computes the rotation in fp32 (matching HF) and casts back to the input
    dtype, using the interleaved ``rotate_half`` convention from HF Mage-VL.
    Signature follows ``VisionAttention.customized_position_embedding_applier``.
    """
    cos, sin = position_embeddings
    oqd, okd = q.dtype, k.dtype
    qf = q.float()
    kf = k.float()
    cos = cos.float()
    sin = sin.float()
    while cos.dim() < qf.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    qe = (qf * cos) + (_mage_vl_rotate_half_interleaved(qf) * sin)
    ke = (kf * cos) + (_mage_vl_rotate_half_interleaved(kf) * sin)
    return qe.to(oqd), ke.to(okd)


def build_cu_seqlens(
    grid_thw: torch.Tensor,
    total_patches: int,
    fixed_t: Optional[int] = 4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """Port of MageVLVisionPretrainedModel._build_cu_seqlens.

    Splits grids whose t > fixed_t into chunks of fixed_t (+ remainder).
    Returns (cu_seqlens int32[N+1], max_seqlen int).
    """
    if grid_thw is None or grid_thw.numel() == 0:
        return (
            torch.tensor([0, total_patches], dtype=torch.int32, device=device),
            total_patches,
        )
    if device is None:
        device = grid_thw.device

    cu_seqlens = [0]
    max_seqlen = 0
    current_len = 0
    for idx in range(grid_thw.shape[0]):
        t_val = int(grid_thw[idx, 0])
        h_val = int(grid_thw[idx, 1])
        w_val = int(grid_thw[idx, 2])
        if fixed_t is not None and fixed_t > 0 and t_val > fixed_t:
            num_full = t_val // fixed_t
            rem = t_val % fixed_t
            for _ in range(num_full):
                chunk = fixed_t * h_val * w_val
                current_len += chunk
                max_seqlen = max(max_seqlen, chunk)
                cu_seqlens.append(current_len)
            if rem > 0:
                chunk = rem * h_val * w_val
                current_len += chunk
                max_seqlen = max(max_seqlen, chunk)
                cu_seqlens.append(current_len)
        else:
            chunk = t_val * h_val * w_val
            current_len += chunk
            max_seqlen = max(max_seqlen, chunk)
            cu_seqlens.append(current_len)

    if cu_seqlens[-1] != total_patches:
        raise ValueError(
            f"cu_seqlens calculation mismatch: total_patches={total_patches}, "
            f"calculated={cu_seqlens[-1]}, grid_thw={grid_thw}"
        )
    return torch.tensor(cu_seqlens, dtype=torch.int32, device=device), max_seqlen


def _get_norm(config):
    """Return LayerNorm or RMSNorm based on `config.layer_norm_type`.

    Raises ValueError on unknown values (defensive — typos like 'rmsnorm'
    silently fell through to LayerNorm before).
    """
    norm_type = getattr(config, "layer_norm_type", "layer_norm")
    if norm_type == "layer_norm":
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    if norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    raise ValueError(f"Unknown layer_norm_type: {norm_type!r}")


class MageVLVisionEmbeddings(nn.Module):
    """Patch embedding via Conv2d (kernel_size=stride=patch_size, bias=False).

    Mirrors HF ``MageVLVisionEmbeddings`` so the checkpoint
    ``patch_embedding.weight`` loads directly.

    Input  : [N, C, P, P] or [N, C*P*P]
    Output : [N, hidden_size]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.patch_embedding(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class MageVLVisionPatchMerger(nn.Module):
    """LayerNorm + 2x Linear (TP-aware), merges spatial_merge_size^2 patches into one token.

    Optionally adds H/W absolute position embeddings (controlled by
    ``use_patch_position_encoding``).
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-5,
        use_patch_position_encoding: bool = False,
        patch_position_encoding_type: str = "absolute",
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.spatial_merge_size = spatial_merge_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.ln_q = nn.LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                ),
            ]
        )
        if use_patch_position_encoding:
            if patch_position_encoding_type != "absolute":
                raise ValueError(
                    f"Unknown patch_position_encoding_type: {patch_position_encoding_type}"
                )
            self.pos_emb_h = nn.Embedding(max_position_embeddings, dim)
            self.pos_emb_w = nn.Embedding(max_position_embeddings, dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        x = self.ln_q(x).view(-1, self.hidden_size)
        fc1, act, fc2 = self.mlp
        x, _ = fc1(x)
        x = act(x)
        x, _ = fc2(x)
        if self.use_patch_position_encoding and patch_positions is not None:
            pp = patch_positions.view(-1, self.spatial_merge_size**2, 3)[:, 0, :]
            pp = (pp // self.spatial_merge_size).long()
            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])
        return x


class MageVLVisionMLP(nn.Module):
    """Siglip-style MLP via TP linear (ColumnParallel + RowParallel)."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class MageVLVisionBlock(nn.Module):
    """Pre-norm + VisionAttention + pre-norm + MLP (Mage-VL vision block).

    Convention (matches OneVision 1.5 ``RiceBlock``): ``x`` carries shape ``[s, b, d]``.
    Inside, we rearrange to ``[b, s, d]`` for VisionAttention then back.

    ``qkv_backend`` defaults to ``None`` so ``VisionAttention`` auto-selects
    based on platform (fa3/triton on CUDA Hopper+, sdpa on CPU). Production
    callers can override.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        qkv_backend: Optional[str] = None,
    ):
        super().__init__()
        self.layer_norm1 = _get_norm(config)
        self.layer_norm2 = _get_norm(config)
        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            rotary_embed="normal",
            proj_bias=True,
            qkv_backend=qkv_backend,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            customized_position_embedding_applier=_mage_vl_apply_rotary_pos_emb,
        )
        self.mlp = MageVLVisionMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        h = self.layer_norm1(x)
        h = rearrange(h, "s b ... -> b s ...")
        # NOTE: VisionAttention.forward accepts **kwargs; max_seqlen is forwarded
        # for FA varlen backends that can consume it (current backends recompute
        # it from cu_seqlens, so this is a forward-compatible pass-through).
        attn = self.attn(
            h,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn
        x = x + self.mlp(self.layer_norm2(x))
        return x


class MageVLVisionEncoder(nn.Module):
    """Thin wrapper that holds the list of MageVLVisionBlock layers.

    Mirrors HF ``MageVLVisionEncoder.layers`` so checkpoint weights at
    ``encoder.layers.{i}.*`` load directly into our sglang port.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        qkv_backend: Optional[str] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MageVLVisionBlock(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    qkv_backend=qkv_backend,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
            )
        return hidden_states


class MageVLVisionTransformer(nn.Module):
    """Top-level Mage-VL vision tower.

    Mirrors HF ``MageVLVisionPretrainedModel`` so the checkpoint
    state_dict (``embeddings.patch_embedding.weight``, ``layernorm_pre.*``,
    ``encoder.layers.{i}.*``, ``merger.*``) loads directly.

    Pipeline:
        embeddings -> layernorm_pre -> encoder(window-attn) -> [optional layernorm_post]
        -> merger(spatial_merge_size^2 -> 1)

    Input:
        pixel_values:    [N, C, P, P]  (or flattened [N, C*P*P])
        grid_thw:        [num_samples, 3]  rows of [t, h, w]
        patch_positions: [N, 3]  rows of [t, h, w] per patch

    Output:
        merged tokens: [N // spatial_merge_size**2, out_hidden_size]
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        qkv_backend: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.frame_windows_size = getattr(config, "frame_windows_size", 4)

        self.embeddings = MageVLVisionEmbeddings(config)
        self.layernorm_pre = _get_norm(config)
        self.video_rope = VisionRotaryEmbedding(config)

        self.encoder = MageVLVisionEncoder(
            config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
            qkv_backend=qkv_backend,
        )

        if getattr(config, "use_head", False):
            self.layernorm_post = _get_norm(config)
        else:
            self.layernorm_post = None

        self.merger = MageVLVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
            use_patch_position_encoding=getattr(
                config, "use_patch_position_encoding", False
            ),
            patch_position_encoding_type=getattr(
                config, "patch_position_encoding_type", "absolute"
            ),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        patch_positions: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Embeddings: [N, C, P, P] -> [N, D] -> [1, N, D]
        h = self.embeddings(pixel_values)
        if h.dim() == 2:
            h = h.unsqueeze(0)

        # 2. 3D RoPE freqs from patch_positions
        if patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)
        freqs = self.video_rope.forward_from_positions(patch_positions)  # [N, half]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [N, head_dim]
        cos = freqs.cos().to(h.dtype)
        sin = freqs.sin().to(h.dtype)

        # 3. Pre-norm + encoder (window attn per cu_seqlens)
        h = self.layernorm_pre(h)
        cu_seqlens, max_seqlen = build_cu_seqlens(
            grid_thw=grid_thw,
            total_patches=h.shape[1],
            fixed_t=self.frame_windows_size,
            device=h.device,
        )

        # Block convention: [s, b, d]
        h = rearrange(h, "b s d -> s b d")
        h = self.encoder(
            h,
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
            max_seqlen=max_seqlen,
        )
        h = rearrange(h, "s b d -> b s d")

        if self.layernorm_post is not None:
            h = self.layernorm_post(h)

        # 4. Merger: [1, N, D] -> [N, D] -> [N/merge^2, out_hidden_size]
        h = h.squeeze(0)
        return self.merger(h, patch_positions=patch_positions)


class MageVLForConditionalGeneration(nn.Module):
    """sglang inference module for Mage-VL.

    Top-level layout (flat, not HF's two-level ``model.visual`` / ``model.language_model``):
        self.visual    : MageVLVisionTransformer
        self.model     : Qwen3Model (text)
        self.lm_head   : ParallelLMHead (TP-aware logits)

    Checkpoint keys ``model.visual.*`` / ``model.language_model.*`` / ``lm_head.*``
    are remapped to this layout by ``load_weights`` (Task 8).

    Multimodal forward goes through sglang's ``general_mm_embed_routine``:
    text token embeddings are obtained from ``self.model.embed_tokens``,
    image patches are encoded by ``self.visual.forward`` and spliced in at the
    image-token positions.
    """

    # BitsAndBytes scaffolding (matches OneVision 1.5; harmless if BnB unused)
    default_bitsandbytes_target_modules = [
        ".fc2.",
        ".fc1.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.visual = MageVLVisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )
        self.model = Qwen3Model(
            config.text_config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if getattr(config, "tie_word_embeddings", False):
            import logging

            logging.warning("tied word embeddings is not supported in Mage-VL.")

        # mrope is enabled when the text config has ``rope_scaling`` with ``mrope_section``
        text_rope = getattr(config.text_config, "rope_scaling", None)
        self.is_mrope_enabled = text_rope is not None and "mrope_section" in text_rope

        self.logits_processor = LogitsProcessor(config.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    # ---- multimodal padding pattern ----

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids,
            mm_inputs,
        )

    # ---- vision feature path (image-only; video aliases) ----

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Concatenate per-item ``feature`` (pixel patches), ``image_grid_thw``,
        and ``patch_positions`` (carried in ``model_specific_data`` by the
        Mage-VL multimodal processor) and run the vision tower.

        Note: ``item.patch_positions`` is accessed via
        ``MultimodalDataItem.__getattr__``, which looks up
        ``model_specific_data["patch_positions"]`` — so the docstring's
        "model_specific_data" reference and the attribute access in code
        are consistent.
        """
        # Precondition: Mage-VL multimodal processor must attach `patch_positions`
        # to each MultimodalDataItem via model_specific_data. Accessed as item.patch_positions
        # via MultimodalDataItem.__getattr__.
        for it in items:
            if "patch_positions" not in it.model_specific_data:
                raise RuntimeError(
                    "Mage-VL vision tower requires `patch_positions` per multimodal item; "
                    "ensure the Mage-VL multimodal processor populates it."
                )
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            self.visual.dtype,
        )
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)
        patch_positions = torch.cat([item.patch_positions for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        return self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            patch_positions=patch_positions,
        )

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Mage-VL processor folds video frames into the image path; alias.
        return self.get_image_feature(items)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    # ---- sglang forward protocol ----

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )
        if not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )
        return self.pooler(hidden_states, forward_batch)


def map_hf_name(name: str) -> str:
    """Translate an HF Mage-VL checkpoint key to an sglang parameter key.

    Vision: model.visual.X -> visual.X; self_attn.qkv -> attn.qkv_proj;
            self_attn.proj -> attn.proj.
    Text:   model.language_model.X -> model.X.
    """
    if name.startswith("model.visual"):
        name = name[len("model.") :]
        name = name.replace(".self_attn.qkv.", ".attn.qkv_proj.")
        name = name.replace(".self_attn.proj.", ".attn.proj.")
    elif name.startswith("model.language_model"):
        name = "model." + name[len("model.language_model.") :]
    return name


def _load_weights_into(model, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
    """Stream HF Mage-VL safetensors weights into the sglang model.

    Text branch uses standard q/k/v -> qkv_proj and gate/up -> gate_up_proj
    stacked-param merging via the bound weight_loader. Vision branch uses
    default_weight_loader because VisionAttention.qkv_proj is already a single
    ColumnParallelLinear (no shard merging).
    """
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(model.named_parameters(remove_duplicate=False))

    for raw_name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in raw_name:
            continue

        name = map_hf_name(raw_name)

        matched = False
        if "visual" not in name:
            for tgt, src, shard_id in stacked_params_mapping:
                if src in name:
                    pname = name.replace(src, tgt)
                    if pname.endswith(".bias") and pname not in params_dict:
                        matched = True
                        break
                    if pname not in params_dict:
                        # not a real stacked merge target; fall through
                        break
                    param = params_dict[pname]
                    param.weight_loader(param, loaded_weight, shard_id)
                    matched = True
                    break
        if matched:
            continue

        if name.endswith(".bias") and name not in params_dict:
            continue
        if name not in params_dict:
            raise KeyError(
                f"Mage-VL weight key not in model params: raw={raw_name!r}, mapped={name!r}"
            )
        param = params_dict[name]
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, loaded_weight)


MageVLForConditionalGeneration.load_weights = _load_weights_into


EntryClass = [MageVLForConditionalGeneration]
