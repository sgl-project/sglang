# Adapted from qwen2.py
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.true_on_policy import (
    should_disable_fused_qk_norm_mrope,
    should_force_bfloat16_dense_tensor_math,
)
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu
from sglang.srt.utils.hf_transformers_utils import get_rope_config

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_has_fused_qk_norm_mrope = False
if _use_aiter:
    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
        logger.info("aiter fused_qk_norm_mrope_3d kernel available")
    except ImportError:
        pass

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

    from sglang.srt.hardware_backend.npu.cmo import get_cmo_stream, wait_cmo_stream


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
            true_on_policy_weight_dtype=torch.float32,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=rms_norm_eps,
            true_on_policy_weight_dtype=torch.float32,
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

        # --- VLCache (Stage B: image-KV reuse) ---
        self.layer_id = layer_id
        _sa = get_global_server_args()
        self.vlcache_enabled = getattr(_sa, "enable_vlcache", False)
        if self.vlcache_enabled:
            from sglang.srt.managers.mock_kv_manager import mock_kv_manager

            self.mock_kv_manager = mock_kv_manager
            ratio = getattr(_sa, "vlcache_recompute_ratio", 0.3)
            num_layers = getattr(_sa, "_vlcache_num_layers", layer_id + 1)
            self.recompute_ratio_in_layer = [ratio] * max(num_layers, layer_id + 1)
            self.max_recompute_layer_id = int(
                torch.argmax(torch.tensor(self.recompute_ratio_in_layer)).item()
            )
            self.is_max_recompute_layer = (
                max(self.recompute_ratio_in_layer)
                == self.recompute_ratio_in_layer[self.layer_id]
            )

        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
        if self.use_fused_qk_norm_mrope:
            # Scale tensors MUST stay on CPU: the C++ kernel uses .item<float>()
            # which triggers hipMemcpy D2H + sync on CUDA tensors, breaking graph capture.
            # Explicit device='cpu' is required because SGLang constructs models inside
            # a `with torch.device('cuda'):` context that changes the default device.
            self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    def maybe_write_kv(self, k: torch.Tensor, v: torch.Tensor, write_info) -> None:
        """Write freshly-computed image K/V slices to the VLCache store (Stage B).

        ``write_info[layer_id]`` is a list of ``[start, end, uid_k, uid_v]`` slices
        into the *compressed* K/V (rows the image occupies after reused tokens were
        dropped). Each slice is a cache-miss image being stored for future reuse.

        R3 (tensor parallelism): each TP rank computes and writes its own KV shard
        under a rank-scoped uid, so shards never collide and a later read on the same
        rank retrieves the matching shard. Reuse only fires when every rank's shard is
        present, so a rank that missed a write simply forces a recompute rather than
        reading another rank's data.
        """
        if write_info is None or self.layer_id not in write_info:
            return
        k = k.contiguous()
        v = v.contiguous()
        # uids come from write_info (built by the mask-builder), which already scopes
        # them per TP rank -- do NOT re-suffix here, or write/read uids won't match.
        for start_idx, end_idx, uid_k, uid_v in write_info[self.layer_id]:
            # Skip empty slices: at recompute_ratio >= 1.0 the reuse portion is empty
            # (nothing to cache), which would otherwise store a 0-row tensor.
            if end_idx < start_idx:
                continue
            part_k = k[start_idx : end_idx + 1, :]
            part_v = v[start_idx : end_idx + 1, :]
            # No torch.cuda.synchronize() here: write_kv's copy is issued on the current
            # stream and is ordered after the projection that produced part_k/part_v, so
            # it observes materialized data without a device-wide flush. The previous
            # per-layer sync stalled the whole GPU ~num_layers times per prefill.
            self.mock_kv_manager.write_kv(part_k, uid_k, non_blocking=True)
            self.mock_kv_manager.write_kv(part_v, uid_v, non_blocking=True)

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward_prepare_aiter_fused_mrope(
        self, positions, hidden_states, forward_batch
    ):
        """Fused QK-norm + 3D mRoPE + KV cache write for decode (ROCm/aiter).

        The fused HIP kernel replaces split → QK norm → mRoPE → cache write,
        so KV is already in the paged cache when this returns.
        Returns (q, None, None); caller must pass save_kv_cache=False to attn.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]

        qkv_3d = qkv.view(num_tokens, -1, self.head_dim)

        token_to_kv_pool = forward_batch.token_to_kv_pool
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(self.attn.layer_id)
        slot_mapping = forward_batch.out_cache_loc

        cos_sin = self.rotary_emb.cos_sin_cache
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(dtype=qkv.dtype)

        q_out = torch.empty(
            num_tokens,
            self.num_heads,
            self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )

        fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            qkv_3d,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin,
            positions,
            num_tokens,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.is_neox_style,
            self.rotary_emb.mrope_section,
            self.rotary_emb.mrope_interleaved,
            self.q_norm.variance_epsilon,
            q_out,
            k_cache,
            v_cache,
            slot_mapping,
            self._fused_k_scale,
            self._fused_v_scale,
            None,
            None,
            False,
            False,
            0,
            0,
        )

        q = q_out.reshape(num_tokens, -1)
        return q, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # --- VLCache (Stage B): reuse-aware attention for this layer ---
        # Fires only when enabled AND this layer has a reuse plan (recompute_info),
        # i.e. an image in the batch is a cache hit. Otherwise fall through to stock.
        if (
            self.vlcache_enabled
            and forward_batch.recompute_info is not None
            and self.layer_id in forward_batch.recompute_info
        ):
            return self._forward_vlcache_reuse(positions, hidden_states, forward_batch)

        if (
            should_force_bfloat16_dense_tensor_math()
            or hidden_states.dtype != self.qkv_proj.weight.dtype
        ):
            # True-on-policy RMSNorm can produce fp32 activations while dense
            # projections remain bf16, including during cuda-graph capture when
            # the global on-policy flag is temporarily cleared.
            hidden_states = hidden_states.to(self.qkv_proj.weight.dtype)

        save_kv_cache = True
        use_aiter_fused = (
            self.use_fused_qk_norm_mrope
            and forward_batch.forward_mode.is_decode()
            and not should_disable_fused_qk_norm_mrope()
        )

        # VLCache (Stage B): on a cache MISS this layer must STORE the image's K/V
        # pre-RoPE (so a later hit can re-RoPE it at the shifted position). The stock
        # prepare paths RoPE before returning, so we project + write + RoPE inline
        # here, mirroring the reference. Only fires when there is a write plan for
        # this layer (an image was a miss this batch).
        if (
            self.vlcache_enabled
            and not use_aiter_fused
            and forward_batch.write_info is not None
            and self.layer_id in forward_batch.write_info
        ):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Store pre-QK-norm, pre-RoPE K/V under the image's per-layer uids.
            self.maybe_write_kv(k, v, forward_batch.write_info)
            # allow_inplace=True: write_kv already took a synchronous .clone() of k, so
            # the stored copy is independent -- normalizing k in place here is safe and
            # keeps the FUSED qk-norm kernel (allow_inplace=False forced the slow
            # unfused fallback on every layer, the dominant cost on cache-miss prefills).
            q, k = apply_qk_norm(
                q=q, k=k, q_norm=self.q_norm, k_norm=self.k_norm,
                head_dim=self.head_dim, alt_stream=self.alt_stream, allow_inplace=True,
            )
            q, k = self.rotary_emb(positions, q, k)
            if should_force_bfloat16_dense_tensor_math() or q.dtype != v.dtype:
                q = q.to(v.dtype)
                k = k.to(v.dtype)
            attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=True)
            output, _ = self.o_proj(attn_output)
            return output

        if use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused_mrope(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif (
            not _is_npu
            or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        ):
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if should_force_bfloat16_dense_tensor_math() or q.dtype != v.dtype:
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        output, _ = self.o_proj(attn_output)
        return output

    def _forward_vlcache_reuse(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Reuse-aware attention for one layer when an image is a cache hit (Stage B).

        Only the *recompute* tokens (compute_mask) get a fresh Q/K/V projection; the
        reused image K/V come from the store (already loaded into recompute_info by
        the mask-builder). We:
          1. project + QK-norm only the recompute tokens (compressed hidden states),
          2. store the freshly-computed image K/V (maybe_write_kv) BEFORE RoPE,
          3. re-apply RoPE to the recompute tokens at their *current* positions,
          4. re-apply RoPE to the reused K at their current positions (pre-RoPE keys
             were stored, so this rotates them to where the image now sits),
          5. splice recomputed K/V into the reused K/V at the compute_mask rows,
          6. run attention with the sparse (reuse-aware) backend path.

        Assumes a uniform recompute ratio across layers (every layer is the
        max-recompute layer), which is how VLCache is configured here. The
        non-uniform (per-layer differing ratio) case is not yet supported.
        """
        assert (
            self.is_max_recompute_layer
        ), "VLCache non-uniform per-layer recompute ratio is not supported yet"

        compute_mask = forward_batch.compute_mask[self.max_recompute_layer_id]
        k, v = forward_batch.recompute_info[self.layer_id]

        # 1. Project + QK-norm ONLY the recompute tokens.
        #    The hidden stream is compressed exactly once (at the first reuse layer):
        #    Qwen3DecoderLayer slices the residual+hidden to the recompute rows after
        #    self_attn returns, so every downstream reuse layer receives an ALREADY
        #    compressed tensor. compute_mask stays full batch length (it also masks the
        #    full-length reused K/V below), so only index hidden when it's still full.
        if hidden_states.shape[0] == compute_mask.shape[0]:
            compute_hidden = hidden_states[compute_mask]
        else:
            assert hidden_states.shape[0] == int(compute_mask.sum()), (
                f"VLCache: pre-compressed hidden rows {hidden_states.shape[0]} != "
                f"recompute rows {int(compute_mask.sum())} (layer {self.layer_id})"
            )
            compute_hidden = hidden_states
        qkv, _ = self.qkv_proj(compute_hidden)
        q, k_part, v_part = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. Store the freshly-computed (pre-RoPE) image K/V for future reuse.
        self.maybe_write_kv(k_part, v_part, forward_batch.write_info)

        # 3. QK-norm, then RoPE the recompute tokens at their current positions.
        #    allow_inplace=True: write_kv already cloned k_part, so in-place norm is safe
        #    and keeps the fused kernel.
        q, k_part = apply_qk_norm(
            q=q, k=k_part, q_norm=self.q_norm, k_norm=self.k_norm,
            head_dim=self.head_dim, alt_stream=self.alt_stream, allow_inplace=True,
        )
        q, k_part = self.rotary_emb(positions[..., compute_mask], q.contiguous(), k_part.contiguous())

        # 4. The reused K were stored pre-RoPE; QK-norm + RoPE them at CURRENT positions
        #    (a dummy q satisfies the rotary_emb signature; only k is used).
        #    Only the REUSED rows (~compute_mask) need this: the recompute rows of k are
        #    overwritten by k_part in step 5, so RoPE-ing the full batch here wasted work
        #    on every row that gets discarded. RoPE is per-position independent, so
        #    processing just the reused subset is exactly equivalent. positions is
        #    [3, total_tokens] (mRoPE); slicing the reuse columns keeps them aligned.
        reuse_sel = ~compute_mask
        k_reuse = k[reuse_sel]
        dummy_q = torch.empty_like(k_reuse)
        dummy_q, k_reuse = apply_qk_norm(
            q=dummy_q, k=k_reuse, q_norm=self.q_norm, k_norm=self.k_norm,
            head_dim=self.head_dim, alt_stream=self.alt_stream,
        )
        _, k_reuse = self.rotary_emb(
            positions[..., reuse_sel], dummy_q, k_reuse.contiguous()
        )

        # 5. Assemble the full per-layer K/V: re-RoPE'd reused rows + fresh recompute rows.
        k[reuse_sel, :] = k_reuse
        k[compute_mask, :] = k_part
        v[compute_mask, :] = v_part

        # 6. Attention: q holds only recompute tokens; k/v hold the full image KV.
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta, rope_scaling = get_rope_config(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            true_on_policy_weight_dtype=torch.float32,
            true_on_policy_override_orig_dtype=torch.float32,
            true_on_policy_fp32_residual=True,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            true_on_policy_weight_dtype=torch.float32,
            true_on_policy_override_orig_dtype=torch.float32,
            true_on_policy_fp32_residual=True,
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            post_residual_addition=post_residual_addition,
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            # VLCache (Stage B): if this layer reused an image, self_attn returned only
            # the recompute-token rows, so the residual (full length) must be sliced to
            # match before the residual add downstream.
            if (
                self.self_attn.vlcache_enabled
                and forward_batch.recompute_info is not None
                and self.self_attn.layer_id in forward_batch.recompute_info
                and residual is not None
            ):
                # Compress the residual to match self_attn's recompute-only output.
                # The stream compresses exactly once (first reuse layer); downstream
                # reuse layers already receive a compressed residual, so only slice
                # when it is still full batch length.
                _mask = forward_batch.compute_mask[self.self_attn.max_recompute_layer_id]
                if residual.shape[0] == _mask.shape[0]:
                    residual = residual[_mask]

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache=(
                [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                if _is_npu
                and not get_global_server_args().disable_piecewise_cuda_graph
                and (
                    hasattr(self.mlp.gate_up_proj, "weight")
                    and hasattr(self.mlp.down_proj, "weight")
                )
                else None
            ),
        )
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # Stacked params mapping for unified weight loading API
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = self.stacked_params_mapping
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        # SGLang captures "before layer i". To capture the hidden state after target
        # layer `k` (HF-style), we capture before layer `k + 1`.
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen3ForCausalLM
