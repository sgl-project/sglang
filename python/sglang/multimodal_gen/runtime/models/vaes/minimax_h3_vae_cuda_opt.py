# SPDX-License-Identifier: Apache-2.0
"""CUDA fast path for the MiniMax-H3 video VAE decoder (ViT3DDecoder).

Fuses each decoder attention block's per-head QK RMSNorm + NeoX RoPE into one
in-place launch over the strided Q/K views of the QKV buffer and runs the
block's attention on cuDNN SDPA. Forwards are bound once at VAE load and
dispatch on a decode-scoped :class:`VaeFastPathGate`: ``quality="extra-high"``
and ``"high"`` take the fast path (rounding-level differences), the
``"lossless"`` default runs the original module path bit-for-bit. Install is
all-or-nothing and fail-closed.
"""

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
    _sdpa_attention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    _env_flag,
    apply_rotary_pos_emb_qk,
    native_rope_cache,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _fused_qknorm_rope(self, query, key, rotary_pos_emb) -> bool:
    native = native_rope_cache(query, key, rotary_pos_emb)
    if native is None:
        return False
    cache, positions = native
    if not can_use_fused_inplace_qknorm_rope(
        self.dim_head, cache.shape[1], True, query.dtype, cache.dtype, True
    ):
        return False
    weight = self._sgl_unit_weight
    if weight is None or weight.dtype != query.dtype or weight.device != query.device:
        weight = torch.ones(self.dim_head, dtype=query.dtype, device=query.device)
        self._sgl_unit_weight = weight
    fused_inplace_qknorm_rope(
        query[0],
        key[0],
        weight,
        weight,
        cache,
        positions,
        is_neox=True,
        eps=self.norm_q.eps,
        head_dim=self.dim_head,
        rope_dim=cache.shape[1],
        round_norm_before_rope=True,
    )
    return True


def _cudnn_attention(self, query, key, value):
    if not self._sgl_cudnn_failed:
        try:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                ).transpose(1, 2)
        except RuntimeError as e:
            logger.warning(
                "MiniMax-H3 VAE: cuDNN SDPA failed (%s); using the layer's "
                "attention backend.",
                e,
            )
            self._sgl_cudnn_failed = True
    return self.attn(query, key, value)


def _attn_fast_forward(self, hidden_states, rotary_pos_emb=None):
    if (
        not self._sgl_gate.enabled
        or rotary_pos_emb is None
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return type(self).forward(self, hidden_states, rotary_pos_emb)

    batch_size, seq_len, _ = hidden_states.shape
    qkv = self.to_qkv(hidden_states)
    qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)

    if not _fused_qknorm_rope(self, query, key, rotary_pos_emb):
        query = _apply_qk_norm(self.norm_q, query)
        key = _apply_qk_norm(self.norm_k, key)
        query, key = apply_rotary_pos_emb_qk(query, key, rotary_pos_emb)

    if self.attn is not None and query.dtype in (torch.float16, torch.bfloat16):
        hidden_states = _cudnn_attention(self, query, key, value)
    else:
        hidden_states = _sdpa_attention(query, key, value)

    hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
    return self.to_out(hidden_states)


def _plain_rms_norm(module) -> bool:
    return isinstance(module, nn.RMSNorm) and module.weight is None


def _attn_fast_compatible(m) -> bool:
    return (
        type(m) is Attention
        and _plain_rms_norm(m.norm_q)
        and _plain_rms_norm(m.norm_k)
        and m.norm_q.eps == m.norm_k.eps
    )


def install_qknorm_rope(attn_modules: list[nn.Module], gate: VaeFastPathGate) -> None:
    for m in attn_modules:
        m._sgl_gate = gate
        m._sgl_unit_weight = None
        m._sgl_cudnn_failed = False
        m.forward = MethodType(_attn_fast_forward, m)


def maybe_optimize_minimax_h3_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA MiniMax-H3 VAE decoder fast path."""
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
        ViT3DDecoder,
    )

    if not isinstance(vae, MiniMaxH3VideoVAE):
        return vae
    decoder = getattr(vae, "decoder", None)
    if type(decoder) is not ViT3DDecoder:
        return vae
    if not _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        logger.info("MiniMax-H3 VAE: fp32 QK norm disabled; skipping fast path.")
        return vae

    attn_modules = [block.attn for block in decoder.transformer_blocks]
    eligible = [m for m in attn_modules if _attn_fast_compatible(m)]
    if len(eligible) != len(attn_modules):
        logger.warning(
            "MiniMax-H3 VAE: %d/%d decoder attention blocks non-standard; "
            "skipping fast path.",
            len(attn_modules) - len(eligible),
            len(attn_modules),
        )
        return vae

    gate = VaeFastPathGate()
    install_qknorm_rope(eligible, gate)
    register_vae_fast_path_gate(vae, gate)
    logger.info(
        "MiniMax-H3 VAE: installed quality-gated fast path (%d QK RMSNorm+RoPE "
        "fusions, cuDNN SDPA).",
        len(eligible),
    )
    return vae
