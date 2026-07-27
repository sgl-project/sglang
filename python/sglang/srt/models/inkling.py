from __future__ import annotations

import copy
import logging
import re
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.inkling import (
    InklingAudioConfig,
    InklingMMConfig,
    InklingModelConfig,
    InklingVisionConfig,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import force_eager_attention
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.inkling_common.attn import (
    InklingAttention,
    compute_log_scaling_tau,
)
from sglang.srt.models.inkling_common.dense_mlp import InklingDenseMLP
from sglang.srt.models.inkling_common.hmlp import HMLPPatchEncoder
from sglang.srt.models.inkling_common.kernels.comm import (
    all_gather_hidden,
    ar_fullwidth_sconv_fused,
    ar_scattered_sconv_fused,
    ar_sconv_norm_fusable,
    ar_sconv_norm_fused,
    ensure_inkling_ar_resources,
    fullwidth_ar_sconv_fusable,
    scattered_ar_sconv_fusable,
)
from sglang.srt.models.inkling_common.moe import InklingMoE
from sglang.srt.models.inkling_common.sconv import SconvType, ShortConvolution
from sglang.srt.models.inkling_common.util import (
    bf16_routed_uses_stock_fused_moe,
    deinterleave_gate_up,
    lora_compatible_layout_enabled,
    shared_sink_uses_trtllm_bf16,
    trtllm_bf16_weight_prep_enabled,
    use_inkling_shared_fused_moe,
)
from sglang.srt.runtime_context import get_model, get_parallel, get_server_args
from sglang.srt.utils import add_prefix, is_cuda, make_layers

logger = logging.getLogger(__name__)

ATTENTION_PARAMS_MAPPING = [
    ("qkvr", "wq_du", 0),
    ("qkvr", "wk_dv", 1),
    ("qkvr", "wv_dv", 2),
    ("qkvr", "wr_du", 3),
]

STACKED_DENSE_PARAMS_MAPPING = [
    ("gate_up_proj", "w1", 0),
    ("gate_up_proj", "w3", 1),
    ("down_proj", "w2", None),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("down_proj", "down_proj", None),
]

# Online RL weight-sync streams routed experts one at a time as FULL (unsharded)
# per-expert tensors named `...mlp.experts.{j}.gate_proj/up_proj/down_proj.weight`.
# Disk checkpoints only ever carry the fused w13_weight/w2_weight, so this pattern
# never fires on the ordinary loading path.
_PER_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<pfx>.+\.mlp\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def _shard_full_to_local(
    loaded_weight: torch.Tensor, dst: torch.Tensor, dim: int
) -> torch.Tensor:
    """Slice a FULL (unsharded) per-expert weight to this MoE-TP rank's shard along `dim`.

    The online weight-sync ships full per-expert tensors (parallelism-agnostic HF
    layout); sglang owns its own MoE-TP sharding, so narrow here per
    get_parallel().moe_tp_rank. With TP1 the dims already match and this is the
    identity, so the ordinary path is byte-for-byte unchanged.
    """
    if loaded_weight.shape[dim] == dst.shape[dim]:
        return loaded_weight
    rank = get_parallel().moe_tp_rank
    return loaded_weight.narrow(dim, rank * dst.shape[dim], dst.shape[dim])


KV_REPLICATED_SUFFIXES = (
    ".wk_dv.weight",
    ".wv_dv.weight",
    ".k_sconv.weight",
    ".v_sconv.weight",
)


def _normalize_mm_weight_name(name: str) -> str:
    if name.startswith("visual.model."):
        return name.replace("visual.model.", "visual.vision_encoder.", 1)
    if name.startswith("visual.") and not name.startswith("visual.vision_encoder."):
        return name.replace("visual.", "visual.vision_encoder.", 1)
    return name


def _is_unsupported_mm_weight_name(name: str) -> bool:
    return name.startswith(
        (
            "audio.decoder.",
            "audio.logits_processor.",
            "vision.",
            "image.",
        )
    )


class InklingDecoderLayer(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        layer_id: int,
        is_local: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.attn = InklingAttention(
            hidden_size=config.hidden_size,
            num_heads=(
                config.swa_num_attention_heads
                if is_local
                else config.num_attention_heads
            ),
            num_kv_heads=(
                config.swa_num_key_value_heads
                if is_local
                else config.num_key_value_heads
            ),
            head_dim=config.swa_head_dim if is_local else config.head_dim,
            d_rel=config.d_rel,
            rel_extent=config.rel_extent,
            local_extent=config.sliding_window_size,
            norm_eps=config.rms_norm_eps,
            is_local=is_local,
            layer_id=layer_id,
            q_bias=config.q_bias,
            o_bias=config.o_bias,
            quant_config=quant_config,
            kv_conv=config.use_sconv,
            sconv_kernel_size=config.sconv_kernel_size,
            prefix=add_prefix("attn", prefix),
            alt_stream=alt_stream,
        )
        if layer_id < config.dense_mlp_idx:
            self.mlp = InklingDenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                use_global_scale=config.use_global_scale,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                fused=True,
                tp_rank=get_parallel().attn_tp_rank,
                tp_size=get_parallel().attn_tp_size,
                tp_group=get_parallel().attn_tp_group,
                use_dp_attention_reduce=True,
            )
        else:
            # Routed experts use FusedMoE. Under LoRA the shared expert remains
            # InklingBatchDenseMLP and applies its adapter delta directly.
            self.mlp = InklingMoE(
                config=config,
                layer_id=layer_id,
                prefix=add_prefix("mlp", prefix),
                quant_config=quant_config,
                alt_stream=alt_stream,
            )

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --enable-scattered-sconv: the output sconvs are channelwise, so they
        # run on the [T, H/P] hidden shard the reduce-scatter produces; weights
        # (ShortConvolution.weight_loader narrows by tp_rank) and conv-state
        # cache (configs/inkling.py stream_dim) shard with them. The layer
        # all-gathers back to [T, H] after each sconv, before the residual add.
        self.attn_tp_group = get_parallel().attn_tp_group
        self.scattered_sconv = get_server_args().enable_scattered_sconv
        sconv_hidden = config.hidden_size
        if self.scattered_sconv:
            assert config.use_sconv, "--enable-scattered-sconv requires use_sconv"
            assert config.hidden_size % self.attn_tp_group.world_size == 0
            sconv_hidden = config.hidden_size // self.attn_tp_group.world_size
        self.attn_sconv = (
            ShortConvolution(
                sconv_hidden,
                config.sconv_kernel_size,
                sconv_type=SconvType.ATTN,
                layer_id=layer_id,
            )
            if config.use_sconv
            else None
        )
        self.mlp_sconv = (
            ShortConvolution(
                sconv_hidden,
                config.sconv_kernel_size,
                sconv_type=SconvType.MLP,
                layer_id=layer_id,
            )
            if config.use_sconv
            else None
        )
        # The fused decode path needs an MoE and this layer's MLP convolution;
        # scattered convolution disables the fusion separately.
        self.mlp_ar_fusable = (
            isinstance(self.mlp, InklingMoE) and self.mlp_sconv is not None
        )

        # Under BCG the short-conv metadata (cu_seqlens/seq_idx) is baked at bs=1
        # during capture, which is wrong for multi-seq prefill. Running every
        # sconv (and the attn whose k/v_sconv it wraps) eagerly makes them re-read
        # the LIVE per-seq metadata at replay. `_breakable_attn_group` groups the
        # prior layer's (deferred) mlp_sconv + attn_norm + attn + attn_sconv into
        # ONE eager break; only mlp_norm + MoE stay captured. Outside a capture
        # these wrappers just run inline. `_breakable_mlp_sconv` runs the final
        # layer's deferred mlp_sconv after the layer loop.
        self._breakable_attn_group = eager_on_graph(True)(self._attn_group_impl)
        self._breakable_mlp_sconv = eager_on_graph(True)(self._mlp_sconv_impl)

    def _attn_block(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_mlp_sconv: Optional[ShortConvolution],
        log_scaling_tau: Optional[torch.Tensor],
        *,
        eager_attn: bool,
        prev_mlp_partial: bool = False,
        fuse_attn_ar: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The {deferred prior-layer mlp_sconv -> attn_norm -> attn -> attn_sconv}
        region, shared by the inline `forward` path and the BCG eager group. Returns
        (hidden_states, residual).

        When ``eager_attn`` is set (the BCG eager break), the attention runs eagerly
        and ``forward_batch.out_cache_loc`` is narrowed to the token count for the KV
        write: the eager attn path (unlike the custom op) does NOT narrow it, and under
        BCG it is a full-bucket buffer, so the KV-write kernel would size-mismatch. The
        caller is responsible for narrowing hidden_states/positions/log_scaling_tau to
        the real (non-padded) token count before calling with ``eager_attn=True``.
        Otherwise the attention runs normally.

        ``prev_mlp_partial``: ``hidden_states`` holds the previous layer's
        UNREDUCED MoE partial sums (``reduce=False``); the {all-reduce ->
        prev_mlp_sconv -> attn_norm} chain runs as ONE fused kernel."""
        hs, res = hidden_states, residual
        if prev_mlp_partial and self.scattered_sconv:
            fm = forward_batch.forward_mode
            if fm.is_decode() or fm.is_target_verify():
                # Fused decode/verify {AR + scattered sconv + attn_norm}: the
                # add+RMSNorm tail is fused in-kernel (residual always live
                # here -- partials only ever come from a previous layer's MoE).
                hs, res = ar_scattered_sconv_fused(
                    hs,
                    prev_mlp_sconv,
                    forward_batch,
                    get_tensor_model_parallel_group(),
                    norm=self.attn_norm,
                    norm_residual=res,
                )
            else:
                # Fused extend {AR + scattered sconv}: hs holds the previous
                # MoE's unreduced partials; the kernel returns the gathered
                # post-conv [T, H], and the norm runs unfused below.
                hs = ar_scattered_sconv_fused(
                    hs, prev_mlp_sconv, forward_batch, get_tensor_model_parallel_group()
                )
                hs, res = self.attn_norm(hs, res)
        elif prev_mlp_partial:
            fm = forward_batch.forward_mode
            if fm.is_decode() or fm.is_target_verify():
                # Fused decode {AR -> sconv -> add+norm}; residual is always
                # live here (partials only ever come from a prior layer's MoE).
                hs, res = ar_sconv_norm_fused(
                    hs,
                    res,
                    prev_mlp_sconv,
                    self.attn_norm,
                    forward_batch,
                    get_tensor_model_parallel_group(),
                )
            else:
                # Fused extend {AR + full-width sconv + cache update}
                # (non-scattered); norm runs unfused on the gathered [T, H].
                hs = ar_fullwidth_sconv_fused(
                    hs, prev_mlp_sconv, forward_batch, get_tensor_model_parallel_group()
                )
                hs, res = self.attn_norm(hs, res)
        else:
            if prev_mlp_sconv is not None:
                hs = prev_mlp_sconv(hs, positions, forward_batch)
                if self.scattered_sconv:
                    # hs was the previous layer's reduce-scattered [T, H/P] MoE
                    # shard; gather back to [T, H] before the residual add.
                    hs = all_gather_hidden(hs, self.attn_tp_group)

            # Fused residual-add + norm for attention input. First layer: no prior
            # residual yet, so just norm the (post-deferred-sconv) embeddings.
            if res is None:
                res = hs
                hs = self.attn_norm(hs)
            else:
                hs, res = self.attn_norm(hs, res)

        if eager_attn:
            # Routing through the split op here would start a nested break that
            # asserts on the already-ended segment, so force the eager attn path;
            # narrow out_cache_loc to the real token count for the KV write (see above),
            # then restore the full buffer on forward_batch (shared across layers/replays).
            orig_out_cache_loc = forward_batch.out_cache_loc
            forward_batch.out_cache_loc = orig_out_cache_loc[: hs.shape[0]]
            with force_eager_attention():
                hs = self.attn(
                    hs, positions, forward_batch, log_scaling_tau=log_scaling_tau
                )
            forward_batch.out_cache_loc = orig_out_cache_loc
        else:
            hs = self.attn(
                hs,
                positions,
                forward_batch,
                log_scaling_tau=log_scaling_tau,
                reduce=not fuse_attn_ar,
            )

        if fuse_attn_ar:
            # hs holds the UNREDUCED wo_ud partials; the caller (forward) fuses
            # {AR -> attn_sconv -> mlp_norm} into one kernel.
            return hs, res
        if self.attn_sconv is not None:
            # Under scattered sconv, hs is the attn output's reduce-scattered
            # [T, H/P] shard (attn.py routed the reduction); gather after.
            hs = self.attn_sconv(hs, positions, forward_batch)
            if self.scattered_sconv:
                hs = all_gather_hidden(hs, self.attn_tp_group)
        return hs, res

    def _attn_group_impl(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        attn_out: torch.Tensor,
        residual_out: torch.Tensor,
        prev_mlp_sconv: Optional[ShortConvolution],
        log_scaling_tau: Optional[torch.Tensor],
    ) -> None:
        """Eager break: run `_attn_block` on the REAL (non-padded) tokens with the LIVE
        forward_batch and write the result into the padded output buffers. Mutates
        attn_out / residual_out and returns None (the eager_on_graph copy-back is
        per-tensor, not per-tuple, so outputs must be pre-allocated buffers)."""
        forward_batch = get_tc_piecewise_forward_context().forward_batch
        n = forward_batch.num_token_non_padded_cpu
        # log_scaling_tau is per-token, so narrow it to match the real tokens too.
        hs, res = self._attn_block(
            hidden_states[:n],
            residual[:n] if residual is not None else None,
            positions[:n],
            forward_batch,
            prev_mlp_sconv,
            log_scaling_tau[:n] if log_scaling_tau is not None else None,
            eager_attn=True,
        )
        torch._foreach_copy_((attn_out[:n], residual_out[:n]), (hs, res))
        if attn_out.shape[0] != n:
            torch._foreach_zero_((attn_out[n:], residual_out[n:]))

    def _mlp_sconv_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Eager break for the final layer's deferred mlp_sconv: run on the real
        tokens with the live forward_batch, write the padded output buffer."""
        forward_batch = get_tc_piecewise_forward_context().forward_batch
        n = forward_batch.num_token_non_padded_cpu
        y = self.mlp_sconv(hidden_states[:n], positions[:n], forward_batch)
        if self.scattered_sconv:
            # y is the [n, H/P] shard; the output buffer is post-gather [n, H].
            y = all_gather_hidden(y, self.attn_tp_group)
        out[:n].copy_(y)
        if out.shape[0] != n:
            out[n:].zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        prev_mlp_sconv: Optional[ShortConvolution] = None,
        *,
        log_scaling_tau: torch.Tensor | None = None,
        prev_mlp_partial: bool = False,
        fuse_ar_sconv: bool = False,
        fuse_attn_ar: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """mlp_sconv is DEFERRED: this layer applies the *previous* layer's
        mlp_sconv (`prev_mlp_sconv`) at the head of its eager region, and its own
        mlp_sconv is applied by the next layer (or, for the last layer, by
        InklingCausalLLM after the loop). Deferral is behavior-identical -- attn_norm
        consumes the sconv output either way -- and lets the whole
        {mlp_sconv, attn_norm, attn, attn_sconv} region be ONE eager break.

        ``prev_mlp_partial``: ``hidden_states`` holds the previous layer's
        UNREDUCED MoE partials; the head chain runs as the fused AR kernel.
        ``fuse_ar_sconv``: this layer's own MoE runs ``reduce=False`` so the
        NEXT consumer (layer or tail) fuses its AR the same way. The caller
        (InklingCausalLLM) threads both consistently."""
        if forward_batch.forward_mode.is_idle():
            return hidden_states, residual

        # The eager group reads the LIVE forward_batch from the tc_piecewise context
        # (the only hook evaluated at BCG replay time — the break's captured args are
        # frozen at capture-bucket shapes). Only the prefill BCG runner installs that
        # context; the decode breakable backend sets is_in_breakable_cuda_graph() but
        # NOT the context, so gate on both and otherwise fall through to the inline
        # path below (which uses the passed forward_batch — correct for decode).
        if (
            is_in_breakable_cuda_graph()
            and get_tc_piecewise_forward_context() is not None
        ):
            # BCG prefill path: the AR fusion is decode-only, so partials never
            # reach (or leave) this branch.
            assert not prev_mlp_partial and not fuse_ar_sconv and not fuse_attn_ar
            # BCG: {prev mlp_sconv, attn_norm, attn, attn_sconv} run eagerly (one
            # break under capture); mlp_norm + MoE stay captured. (The live
            # forward_batch inside the break is read from the shared tc_piecewise
            # context, which the prefill BCG runner populates at capture and replay.)
            # Under scattered sconv the group's INPUT can be the previous layer's
            # [T, H/P] MoE shard while its OUTPUT is post-all-gather [T, H], so
            # size the output buffers explicitly (residual is always [T, H]).
            out_shape = (hidden_states.shape[0], self.attn_norm.weight.shape[0])
            attn_out = hidden_states.new_empty(out_shape)
            residual_out = hidden_states.new_empty(out_shape)
            self._breakable_attn_group(
                hidden_states,
                residual,
                positions,
                attn_out,
                residual_out,
                prev_mlp_sconv,
                log_scaling_tau,
            )
            hidden_states, residual = self.mlp_norm(attn_out, residual_out)
            del attn_out
            del residual_out
            hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
            return hidden_states, residual

        # Plain eager / decode: run inline (still deferring mlp_sconv so the
        # InklingCausalLLM loop threads prev_mlp_sconv uniformly across modes).
        fuse_attn = fuse_attn_ar and self.attn_sconv is not None
        hidden_states, residual = self._attn_block(
            hidden_states,
            residual,
            positions,
            forward_batch,
            prev_mlp_sconv,
            log_scaling_tau,
            eager_attn=False,
            prev_mlp_partial=prev_mlp_partial,
            fuse_attn_ar=fuse_attn,
        )
        if fuse_attn and self.scattered_sconv:
            fm = forward_batch.forward_mode
            if fm.is_decode() or fm.is_target_verify():
                # Fused decode/verify {wo_ud AR + scattered attn_sconv +
                # mlp_norm} (attn-side chain), norm tail in-kernel.
                hidden_states, residual = ar_scattered_sconv_fused(
                    hidden_states,
                    self.attn_sconv,
                    forward_batch,
                    self.attn_tp_group,
                    norm=self.mlp_norm,
                    norm_residual=residual,
                )
            else:
                # Fused extend {AR + scattered sconv} (attn-side chain); the
                # norm runs unfused on the gathered [T, H].
                hidden_states = ar_scattered_sconv_fused(
                    hidden_states, self.attn_sconv, forward_batch, self.attn_tp_group
                )
                hidden_states, residual = self.mlp_norm(hidden_states, residual)
        elif fuse_attn:
            fm = forward_batch.forward_mode
            if fm.is_decode() or fm.is_target_verify():
                # Fused {wo_ud AR -> attn_sconv -> mlp_norm} (attn-side chain).
                hidden_states, residual = ar_sconv_norm_fused(
                    hidden_states,
                    residual,
                    self.attn_sconv,
                    self.mlp_norm,
                    forward_batch,
                    get_parallel().attn_tp_group,
                )
            else:
                # Fused extend {AR + full-width attn_sconv + cache update}
                # (attn-side chain, non-scattered); norm runs unfused.
                hidden_states = ar_fullwidth_sconv_fused(
                    hidden_states,
                    self.attn_sconv,
                    forward_batch,
                    get_parallel().attn_tp_group,
                )
                hidden_states, residual = self.mlp_norm(hidden_states, residual)
        else:
            hidden_states, residual = self.mlp_norm(hidden_states, residual)
        if fuse_ar_sconv and self.mlp_ar_fusable:
            # Skip the MoE's own all-reduce; the next layer (or the model tail)
            # fuses {AR -> this layer's mlp_sconv -> norm} into one kernel.
            hidden_states = self.mlp(
                hidden_states, forward_batch=forward_batch, reduce=False
            )
        else:
            hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        return hidden_states, residual


class InklingCausalLLM(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padded_vocab_size = config.padded_vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.padded_vocab_size,
            config.hidden_size,
            org_num_embeddings=self.padded_vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )

        self.alt_stream = torch.cuda.Stream() if is_cuda() else None

        def get_layer(idx: int, prefix: str) -> InklingDecoderLayer:
            return InklingDecoderLayer(
                config=config,
                layer_id=idx,
                is_local=idx in set(config.local_layer_ids),
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Custom-AR resources must exist BEFORE any CUDA-graph capture (with
        # the prefill graph disabled and --skip-server-warmup there is no eager
        # forward to build them lazily; decode capture would bake the fallback).
        if envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get():
            ensure_inkling_ar_resources(get_tensor_model_parallel_group())
            ensure_inkling_ar_resources(get_parallel().attn_tp_group)

        # Warm the fused decode {AR -> mlp_sconv -> norm} JIT module (both
        # track variants) so the first fused call -- which can land inside a
        # CUDA-graph capture -- doesn't pay the nvcc compile there.
        sconv0 = self.layers[0].mlp_sconv
        world = get_parallel().tp_size
        if (
            is_cuda()
            and envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
            and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM.get()
            and sconv0 is not None
            and world in (4, 8)  # symm-mem multimem worlds, power-of-two
        ):
            from sglang.kernels.ops.communication.inkling_ar_fused import (
                compile_inkling_ar_sconv_norm,
            )

            for do_track in (False, True):
                compile_inkling_ar_sconv_norm(
                    torch.bfloat16,
                    world,
                    sconv0.kernel_size[0],
                    sconv0.activation in ("silu", "swish"),
                    sconv0.use_residual,
                    do_track,
                )

        # Warm the fused attention-prologue JIT module(s) so the first
        # target-verify call (which lands inside a CUDA-graph capture) doesn't
        # nvcc-compile there. fused_prologue is decided PER layer, so warm every
        # distinct eligible signature -- not just layer 0, which may be a
        # local/SWA layer (head_dim != 128) that never uses the prologue while
        # later full-attention layers do.
        if is_cuda() and envs.SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE.get():
            from sglang.kernels.ops.attention.inkling_attn_prologue import (
                compile_inkling_attn_prologue,
            )

            warmed: set = set()
            warm_mxfp8 = get_model().kv_cache_dtype == "mxfp8"
            for layer in self.layers:
                attn = layer.attn
                ks = attn.k_sconv
                if ks is None or attn.head_dim != 128:
                    continue
                sig = (
                    ks.kernel_size[0],
                    ks.activation in ("silu", "swish"),
                    ks.use_residual,
                )
                if sig in warmed:
                    continue
                warmed.add(sig)
                compile_inkling_attn_prologue(torch.bfloat16, *sig)
                if warm_mxfp8:
                    compile_inkling_attn_prologue(torch.bfloat16, *sig, use_mxfp8=True)

        self.lm_head = ParallelLMHead(
            self.padded_vocab_size,
            config.hidden_size,
            org_num_embeddings=self.padded_vocab_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self):
        # Fold embed_norm into the embedding so general_mm_embed_routine norms the text
        # tokens (MM positions are overwritten by the tower features, which keep their own norm).
        embed_tokens, embed_norm = self.embed_tokens, self.embed_norm

        def embed(input_ids: torch.Tensor) -> torch.Tensor:
            embeds = embed_tokens(input_ids)
            return embed_norm(embeds) if embed_norm is not None else embeds

        embed.num_embeddings = self.config.vocab_size
        return embed

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            if self.embed_norm is not None:
                hidden_states = self.embed_norm(hidden_states)
        else:
            # embed_norm was already applied during the MM embed; don't re-norm here.
            hidden_states = input_embeds
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        log_scaling_tau = (
            compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )
            if self.config.log_scaling_n_floor is not None
            else None
        )
        residual = None
        # mlp_sconv is deferred one layer: each layer applies the previous layer's
        # mlp_sconv at the head of its (eager) attn region, so the whole sconv+attn
        # region is one BCG break. prev_mlp_sconv=None for layer 0.
        prev_mlp_sconv = None
        # Fused decode {MoE AR -> mlp_sconv -> attn_norm}: decided ONCE per
        # forward (a pure function of per-forward state, so the producing MoE
        # and the consuming layer/tail always agree). When on, an eligible
        # layer's MoE returns UNREDUCED partials (reduce=False) and the next
        # consumer runs the fused kernel.
        fuse_ar_sconv = (
            not forward_batch.forward_mode.is_idle()
            and ar_sconv_norm_fusable(
                get_tensor_model_parallel_group(),
                forward_batch,
                hidden_states.shape[0],
                hidden_states.shape[-1],
                hidden_states.dtype,
            )
        )
        # The attn-side chain reduces over the ATTENTION TP group (identical to
        # the model TP group without DP attention, but gate on it explicitly).
        fuse_attn_ar = (
            not forward_batch.forward_mode.is_idle()
            and ar_sconv_norm_fusable(
                get_parallel().attn_tp_group,
                forward_batch,
                hidden_states.shape[0],
                hidden_states.shape[-1],
                hidden_states.dtype,
            )
        )
        # Fused extend {AR + scattered sconv} (--enable-scattered-sconv +
        # SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV): same producer contract as the
        # decode fusion above (reduce=False), consumer runs the scattered
        # kernel. Mutually exclusive with ar_sconv_norm_fusable by mode
        # (extend vs decode/verify) and by the scattered gate inside it.
        if not forward_batch.forward_mode.is_idle() and scattered_ar_sconv_fusable(
            get_tensor_model_parallel_group(),
            forward_batch,
            hidden_states.shape[0],
            hidden_states.shape[-1],
            hidden_states.dtype,
        ):
            fuse_ar_sconv = True
            fuse_attn_ar = True
        # Fused extend {AR + full-width sconv + cache update} (NON-scattered):
        # same producer contract (reduce=False); the consumer sites dispatch
        # by mode -- extend runs the full-width column kernel, decode/verify
        # keep ar_sconv_norm_fused. Mutually exclusive with both gates above
        # (mode for ar_sconv_norm_fusable, the scattered flag for
        # scattered_ar_sconv_fusable).
        if not forward_batch.forward_mode.is_idle() and fullwidth_ar_sconv_fusable(
            get_tensor_model_parallel_group(),
            forward_batch,
            hidden_states.shape[0],
            hidden_states.shape[-1],
            hidden_states.dtype,
        ):
            fuse_ar_sconv = True
            fuse_attn_ar = True
        prev_mlp_partial = False
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                positions,
                forward_batch,
                residual,
                prev_mlp_sconv,
                log_scaling_tau=log_scaling_tau,
                prev_mlp_partial=prev_mlp_partial,
                fuse_ar_sconv=fuse_ar_sconv,
                fuse_attn_ar=fuse_attn_ar,
            )
            prev_mlp_sconv = layer.mlp_sconv
            prev_mlp_partial = fuse_ar_sconv and layer.mlp_ar_fusable
        # The final layer's mlp_sconv was deferred; run it now — as an eager break
        # under BCG (so it re-reads live per-seq metadata at replay), else inline.
        if prev_mlp_sconv is not None and not forward_batch.forward_mode.is_idle():
            if prev_mlp_partial and self.layers[-1].scattered_sconv:
                fm = forward_batch.forward_mode
                if fm.is_decode() or fm.is_target_verify():
                    # Fused decode/verify tail: {AR + scattered sconv + final
                    # norm} in one kernel.
                    hidden_states, _ = ar_scattered_sconv_fused(
                        hidden_states,
                        prev_mlp_sconv,
                        forward_batch,
                        get_tensor_model_parallel_group(),
                        norm=self.norm,
                        norm_residual=residual,
                    )
                    return hidden_states
                # Fused extend tail: {AR + scattered sconv}, then the final
                # norm unfused on the gathered [T, H].
                hidden_states = ar_scattered_sconv_fused(
                    hidden_states,
                    prev_mlp_sconv,
                    forward_batch,
                    get_tensor_model_parallel_group(),
                )
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            if prev_mlp_partial:
                fm = forward_batch.forward_mode
                if fm.is_decode() or fm.is_target_verify():
                    # Fused tail: {AR -> final mlp_sconv -> final norm} in one
                    # kernel.
                    hidden_states, _ = ar_sconv_norm_fused(
                        hidden_states,
                        residual,
                        prev_mlp_sconv,
                        self.norm,
                        forward_batch,
                        get_tensor_model_parallel_group(),
                    )
                    return hidden_states
                # Fused extend tail: {AR + full-width sconv + cache update}
                # (non-scattered), then the final norm unfused.
                hidden_states = ar_fullwidth_sconv_fused(
                    hidden_states,
                    prev_mlp_sconv,
                    forward_batch,
                    get_tensor_model_parallel_group(),
                )
                hidden_states, _ = self.norm(hidden_states, residual)
                return hidden_states
            # Same gate as the per-layer group: the eager break needs the tc_piecewise
            # context (installed only by the prefill BCG runner) to read the live
            # forward_batch at replay; else run inline with the passed forward_batch.
            scattered = self.layers[-1].scattered_sconv
            if (
                is_in_breakable_cuda_graph()
                and get_tc_piecewise_forward_context() is not None
            ):
                # Under scattered sconv the input is the last MoE's [T, H/P]
                # shard; the break's output buffer is post-all-gather [T, H].
                out_shape = (
                    (hidden_states.shape[0], self.norm.weight.shape[0])
                    if scattered
                    else hidden_states.shape
                )
                mlp_sconv_out = hidden_states.new_empty(out_shape)
                self.layers[-1]._breakable_mlp_sconv(
                    hidden_states, positions, mlp_sconv_out
                )
                hidden_states = mlp_sconv_out
            else:
                hidden_states = prev_mlp_sconv(hidden_states, positions, forward_batch)
                if scattered:
                    hidden_states = all_gather_hidden(
                        hidden_states, self.layers[-1].attn_tp_group
                    )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class InklingAudio(nn.Module):
    def __init__(self, config: InklingAudioConfig, prefix: str = ""):
        del prefix
        super().__init__()
        assert config.audio_mode == "dmel"
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.use_audio_norm = config.use_audio_norm
        self.encoder = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size, config.decoder_dmodel
        )
        self.final_norm: RMSNorm | None = None
        if self.use_audio_norm:
            self.final_norm = RMSNorm(config.decoder_dmodel, eps=1e-6)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        assert audio_features.shape[1] == self.n_mel_bins

        audio_features = audio_features.to(
            dtype=self.encoder.weight.dtype, device=self.encoder.weight.device
        )

        embedding_indices = (
            torch.arange(self.n_mel_bins, device=audio_features.device)
            * self.mel_vocab_size
        ).unsqueeze(0) + audio_features.to(torch.int32)

        hidden_states = (
            self.encoder(embedding_indices.reshape(-1))
            .reshape(audio_features.shape[0], audio_features.shape[1], -1)
            .sum(axis=1)
        )

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states


class InklingVision(nn.Module):
    def __init__(self, config: InklingVisionConfig, prefix: str = ""):
        del prefix
        super().__init__()
        assert config.vision_encoder_type == "hmlp"
        self.vision_encoder = HMLPPatchEncoder(config)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(vision_features)


class InklingForConditionalGeneration(nn.Module):
    fall_back_to_pt_during_load = False
    supported_lora_modules = [
        "qkvr",
        "wo_ud",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]

    def __init__(
        self,
        config: InklingMMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.text_config = config.text_config

        server_args = get_server_args()
        assert envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get()
        if server_args.disaggregation_mode != "decode":
            assert not server_args.disable_radix_cache
            assert not server_args.disable_hybrid_swa_memory
            assert server_args.enable_mamba_extra_buffer()

        from types import SimpleNamespace

        from sglang.srt.models.inkling_common.quantization import (
            get_quantization_config,
        )

        inkling_quant_config = get_quantization_config(
            SimpleNamespace(hf_config=self.config, model_path=server_args.model_path)
        )
        if inkling_quant_config is not None:
            quant_config = inkling_quant_config

        self.quant_config = quant_config
        self.llm = InklingCausalLLM(
            self.text_config,
            quant_config=quant_config,
            prefix=add_prefix("llm", prefix),
        )
        # Only build the vision/audio towers when multimodal is actually
        # enabled. Inkling is in ModelConfig.mm_disabled_models, so multimodal
        # defaults OFF (enable_multimodal auto -> False); a multimodal
        # checkpoint served text-only must not allocate/load the towers (wasted
        # GPU memory / avoidable startup OOM). The mm dispatch (forward) and the
        # weight loader already skip audio./visual. when these are None.
        build_multimodal = bool(server_args.enable_multimodal)
        self.audio = (
            InklingAudio(self.config.audio_config)
            if build_multimodal and self.config.audio_config.decoder_dmodel is not None
            else None
        )
        self.visual = (
            InklingVision(self.config.vision_config, prefix=prefix)
            if build_multimodal and self.config.vision_config.decoder_dmodel is not None
            else None
        )
        self.mm_pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    @property
    def model(self) -> nn.Module:
        # Expose the language model under `.model` so the prefill CUDA graph
        # (BCG) setup treats Inkling as a language model: model_runner's
        # `hasattr(self.model, "model")` gate and `resolve_language_model`
        # both look for `.model`.
        #
        # This is a property rather than just renaming the `llm` submodule to
        # `model`: the NVFP4 hf_quant_config.json names the LM `model.llm.*`
        # and the exclude-module matching is keyed to the `llm` name, so
        # renaming flips fp4/bf16 treatment on modules and breaks weight
        # loading with a packed-vs-bf16 shape mismatch. A property (not a
        # submodule alias) also avoids registering `llm` twice in the module
        # tree / state_dict.
        return self.llm

    def get_hidden_dim(self, module_name: str, layer_idx: int) -> tuple[int, int]:
        def base_layer(module: torch.nn.Module) -> torch.nn.Module:
            from sglang.srt.lora.layers import BaseLayerWithLoRA

            if isinstance(module, BaseLayerWithLoRA):
                return module.base_layer
            return module

        config = self.text_config
        hidden_size = config.hidden_size
        layer = self.llm.layers[layer_idx]
        if module_name == "qkvr":
            qkvr = base_layer(layer.attn.qkvr)
            return qkvr.input_size, sum(qkvr.output_sizes)
        if module_name == "wo_ud":
            wo_ud = base_layer(layer.attn.wo_ud)
            return wo_ud.input_size, wo_ud.output_size
        # gate_up_proj / down_proj exist only on dense-MLP layers; MoE layers serve
        # them via *_moe buffers, so return config dims for the unused buffer alloc.
        if module_name == "gate_up_proj":
            if isinstance(layer.mlp, InklingDenseMLP):
                gate_up_proj = base_layer(layer.mlp.gate_up_proj)
                return gate_up_proj.input_size, gate_up_proj.output_size
            return hidden_size, config.intermediate_size * 2
        if module_name == "down_proj":
            if isinstance(layer.mlp, InklingDenseMLP):
                down_proj = base_layer(layer.mlp.down_proj)
                return down_proj.input_size, down_proj.output_size
            return config.intermediate_size, hidden_size
        if module_name in ("gate_up_proj_moe", "gate_up_proj_shared_moe"):
            return hidden_size, config.intermediate_size * 2
        if module_name in ("down_proj_moe", "down_proj_shared_moe"):
            return config.intermediate_size, hidden_size
        if module_name == "embed_tokens":
            return config.vocab_size, hidden_size
        if module_name == "lm_head":
            return hidden_size, config.vocab_size
        raise NotImplementedError(f"get_hidden_dim not implemented for {module_name}")

    def get_stacked_multiply(self, module_name: str) -> int:
        if module_name == "qkvr":
            return 4
        if module_name in (
            "gate_up_proj",
            "gate_up_proj_moe",
            "gate_up_proj_shared_moe",
        ):
            return 2
        return 1

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        # The processor expands one placeholder per media item into a run of the same
        # token id; the scheduler calls this to replace each run with the item's
        # pad_value (radix hash), which _embed_mm then masks on to scatter the embeds.
        return self.mm_pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.embed_tokens

    def get_embed_and_head(self):
        return self.llm.embed_tokens.weight, self.llm.lm_head.weight

    def get_num_kv_cache_layers(self) -> int:
        return self.text_config.num_hidden_layers

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return self.text_config.sliding_window_size - 1

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        dmel = torch.cat([item.feature for item in items], dim=0)
        return self.audio(dmel)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        patches = torch.cat([item.feature for item in items], dim=0)
        param = next(self.visual.parameters())
        return self.visual(patches.to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        data_embedding_funcs = {}
        if self.audio is not None:
            data_embedding_funcs[Modality.AUDIO] = self.get_audio_feature
        if self.visual is not None:
            data_embedding_funcs[Modality.IMAGE] = self.get_image_feature
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.llm,
            data_embedding_funcs=data_embedding_funcs,
            positions=positions,
        )
        mup_width_multiplier = self.config.text_config.logits_mup_width_multiplier
        hidden_states_for_logits = (
            hidden_states
            if not mup_width_multiplier
            else hidden_states / mup_width_multiplier
        )
        # The MTP chain needs the undivided hidden (the mup division is
        # lm_head-only); passed unconditionally because the target
        # verify/prefill forwards never set return_hidden_states_before_norm.
        return self.llm.logits_processor(
            input_ids,
            hidden_states_for_logits,
            self.llm.lm_head,
            forward_batch,
            hidden_states_before_norm=hidden_states,
        )

    def update_conv_state_after_mtp_verify(
        self,
        req_to_token_pool,
        req_pool_indices: torch.Tensor,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
    ) -> None:
        """Commit the per-step sconv windows saved during TARGET_VERIFY into the
        persistent conv caches at each request's last accepted step.

        Inkling bypasses the HybridLinearAttnBackend wrapper (ShortConvolution reads
        the mamba pool directly), so the model owns this commit instead of an
        attention-backend hook. The pool is passed in because this runs from the
        spec worker after the forward context has exited.
        """
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            scatter_mamba_states_after_mtp_verify,
        )

        pool = req_to_token_pool
        mamba_indices = pool.translate_mamba_indices(
            pool.get_mamba_indices(req_pool_indices)
        )
        scatter_mamba_states_after_mtp_verify(
            pool.get_speculative_mamba2_params_all_layers(),
            mamba_indices,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )

    def _load_regular_param(
        self,
        params_dict: dict[str, torch.nn.Parameter],
        loaded_params: Set[str],
        name: str,
        loaded_weight: torch.Tensor,
        shard_id: Optional[int] = None,
    ) -> bool:
        if name not in params_dict:
            return False
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is None:
            if param.data.shape == loaded_weight.shape:
                default_weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight)
        else:
            weight_loader(param, loaded_weight, shard_id)
        loaded_params.add(name)
        return True

    def _load_nvfp4_scale_param(
        self,
        params_dict: dict[str, torch.nn.Parameter],
        loaded_params: Set[str],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """Load an NVFP4 auxiliary tensor (block scale / scale2 / input_amax /
        original_shape); shard appropriately.
        """
        if name not in params_dict:
            return False
        param = params_dict[name]
        if loaded_weight.shape != param.shape:
            # shared experts shard over the full tp group; routed over moe_tp
            tp_rank = (
                get_parallel().tp_rank
                if ".shared_experts" in name
                else get_parallel().moe_tp_rank
            )
            for dim in range(loaded_weight.ndim):
                if loaded_weight.shape[dim] == param.shape[dim]:
                    continue
                if loaded_weight.shape[dim] % param.shape[dim] != 0:
                    raise ValueError(
                        f"Cannot TP-shard NVFP4 scale {name}: checkpoint dim "
                        f"{dim} ({loaded_weight.shape[dim]}) is not divisible "
                        f"by the local param dim ({param.shape[dim]})"
                    )
                loaded_weight = loaded_weight.narrow(
                    dim, tp_rank * param.shape[dim], param.shape[dim]
                )
        default_weight_loader(param, loaded_weight)
        loaded_params.add(name)
        return True

    def _ckpt_scale_to_modelopt(
        self,
        suf: str,
        loaded: torch.Tensor,
        param: torch.nn.Parameter,
    ) -> torch.Tensor:
        """Convert a Inkling-checkpoint NVFP4 aux tensor to the layout
        ModelOptNvFp4FusedMoEMethod expects.

        - scale  (block scales): same (E, 2F, H/16) layout -- TP sharding and the
          interleaved-w13 de-interleave are handled by the loader / process_weights.
        - scale2 (per-tensor weight scale): the checkpoint stores one per expert (E,);
          ModelOpt's w13 wants one per (gate, up) -> (E, 2); w2 stays (E,).
        - input_amax -> input_scale = amax / (448*6) (FP8 e4m3 max * FP4 e2m1 max). The
          checkpoint stores a single global activation amax; broadcast to the param shape.
        """
        if suf == "scale":
            return loaded
        if suf == "scale2":
            v = loaded.to(param.dtype)
            if v.ndim == 1 and param.ndim == 2 and param.shape[0] == v.shape[0]:
                return v[:, None].expand(param.shape[0], param.shape[1]).contiguous()
            return v
        if suf == "input_amax":
            scale = float(loaded.reshape(-1)[0].to(torch.float32)) / (448.0 * 6.0)
            return torch.full(tuple(param.shape), scale, dtype=param.dtype)
        return loaded

    def _slice_local_experts(
        self, name: str, loaded_weight: torch.Tensor
    ) -> torch.Tensor:
        """Narrow a routed-expert checkpoint tensor to this rank's local ep block.

        The checkpoint packs all n_routed_experts in dim 0, but under EP each rank's params
        hold only its contiguous slice (the fused loader shards the intermediate dim only).
        No-op when EP is off or for replicated shared-expert tensors.
        """
        ep_size = get_parallel().moe_ep_size
        if (
            ep_size <= 1
            or ".experts." not in name
            # per-expert RL sync tensors do their own EP remap in _load_per_expert_param;
            # a full per-expert tensor whose dim 0 happens to equal n_routed_experts must
            # not be pre-narrowed here.
            or _PER_EXPERT_WEIGHT_RE.match(name) is not None
            or loaded_weight.ndim == 0
            or loaded_weight.shape[0] != self.text_config.n_routed_experts
        ):
            return loaded_weight
        local = self.text_config.n_routed_experts // ep_size
        start = get_parallel().moe_ep_rank * local
        return loaded_weight.narrow(0, start, local).contiguous()

    def _load_fused_moe_param(
        self,
        params_dict: dict[str, torch.nn.Parameter],
        loaded_params: Set[str],
        name: str,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ) -> bool:
        if name not in params_dict:
            return False
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if (
            shard_id == "w13"
            and not self.text_config.inference_moe_w13_interleaved
            and weight_loader is not default_weight_loader
        ):
            from sglang.srt.layers.quantization.modelopt_quant import (
                deinterleave_w13,
            )

            loaded_weight = deinterleave_w13(loaded_weight)
        if (
            loaded_weight.ndim > 0
            and param.data.ndim > 0
            and loaded_weight.shape[0] != param.data.shape[0]
        ):
            raise ValueError(
                f"Unexpected fused MoE expert dimension for {name}: "
                f"loaded={loaded_weight.shape[0]}, expected={param.data.shape[0]}"
            )
        if weight_loader is default_weight_loader:
            default_weight_loader(param, loaded_weight)
        else:
            weight_loader(param, loaded_weight, name, shard_id)
        loaded_params.add(name)
        return True

    def _load_per_expert_param(
        self,
        params_dict: dict[str, torch.nn.Parameter],
        loaded_params: Set[str],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """Load ONE routed expert shipped by the online RL weight-sync.

        The trainer streams routed experts one at a time as FULL (unsharded)
        per-expert tensors — ``...mlp.experts.{j}.gate_proj/up_proj/down_proj.weight``
        — to avoid materializing the multi-GB fused stack. Disk checkpoints only
        carry the fused ``w13_weight``/``w2_weight``, so this never fires for them.

        Writes expert ``j``'s slice of the fused buffer, narrowed to this rank's
        MoE-TP shard (w13 shards the intermediate/output dim, w2 the
        intermediate/input dim). Under EP the global expert id is remapped to this
        rank's local slot and non-owned experts are skipped, mirroring the stock
        ``FusedMoE.weight_loader``. The w13 row layout follows the serving mode:
        Inkling-interleaved ([g0, u0, g1, u1, ...]; the grouped gemm reads 0::2/1::2)
        when the config stores w13 interleaved, contiguous [gate || up] under
        ``lora_compatible_layout_enabled()`` or ``inference_moe_w13_interleaved=False``
        (both make the fused param hold contiguous rows).
        """
        m = _PER_EXPERT_WEIGHT_RE.match(name)
        if m is None:
            return False
        pfx, eid, proj = m.group("pfx"), int(m.group("eid")), m.group("proj")
        leaf = "w2_weight" if proj == "down_proj" else "w13_weight"
        # Under --enable-lora the FusedMoE is wrapped and the base tensor lives one
        # level down at `base_layer.<leaf>`.
        target = next(
            (
                t
                for t in (f"{pfx}.{leaf}", f"{pfx}.base_layer.{leaf}")
                if t in params_dict
            ),
            None,
        )
        if target is None:
            return False
        moe = self.get_submodule(target.rsplit(".", 1)[0])
        if getattr(moe, "use_flashinfer_trtllm_moe", False) or getattr(
            getattr(moe, "quant_method", None), "use_flashinfer_trtllm_moe", False
        ):
            # The trtllm runners keep w13/w2 block-shuffled (and [up || gate]) after
            # process_weights_after_loading; writing canonical rows into that layout
            # would corrupt the stack. Only the triton runner is validated for RL sync.
            raise NotImplementedError(
                f"per-expert RL weight-sync does not support the trtllm MoE layout ({target}); "
                "serve RL rollouts with the triton MoE runner"
            )
        ep_size = get_parallel().moe_ep_size
        if ep_size > 1:
            local = self.text_config.n_routed_experts // ep_size
            first = get_parallel().moe_ep_rank * local
            if not (first <= eid < first + local):
                loaded_params.add(target)  # another rank owns this expert
                return True
            eid -= first
        if proj == "down_proj":
            dst = params_dict[target].data[
                eid
            ]  # [H, I_local]; shard intermediate (dim 1)
            dst.copy_(_shard_full_to_local(loaded_weight, dst, dim=1))
        else:
            w13 = params_dict[target].data[
                eid
            ]  # [2*I_local, H]; shard intermediate (dim 0)
            idx = 0 if proj == "gate_proj" else 1
            if (
                lora_compatible_layout_enabled()
                or not self.text_config.inference_moe_w13_interleaved
            ):
                half = w13.shape[0] // 2
                dst = w13[idx * half : (idx + 1) * half]  # contiguous [gate || up]
            else:
                dst = w13[idx::2]  # Inkling-interleaved rows
            dst.copy_(_shard_full_to_local(loaded_weight, dst, dim=0))
        loaded_params.add(target)
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        embed_tokens_weight: Optional[torch.Tensor] = None

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # The checkpoint nests the sub-models under model.{llm,audio,visual}.;
            # strip the container prefix so the llm./audio./visual. mapping below
            # (and the visual/audio normalization) sees the submodule names directly.
            name = name.removeprefix("model.")
            name = _normalize_mm_weight_name(name)
            if _is_unsupported_mm_weight_name(name):
                continue
            if name.startswith("audio.") and self.audio is None:
                continue
            if name.startswith("visual.") and self.visual is None:
                continue

            if any(name.endswith(suffix) for suffix in KV_REPLICATED_SUFFIXES):
                layer_id = get_layer_id(name)
                if layer_id is not None:
                    num_kv_heads, head_dim = (
                        (
                            self.text_config.swa_num_key_value_heads,
                            self.text_config.swa_head_dim,
                        )
                        if layer_id in set(self.text_config.local_layer_ids)
                        else (
                            self.text_config.num_key_value_heads,
                            self.text_config.head_dim,
                        )
                    )
                    attn_tp_size = get_parallel().attn_tp_size
                    if (
                        attn_tp_size > num_kv_heads
                        and loaded_weight.shape[0] == num_kv_heads * head_dim
                    ):
                        assert attn_tp_size % num_kv_heads == 0
                        replicas = attn_tp_size // num_kv_heads
                        if name.endswith((".wk_dv.weight", ".wv_dv.weight")):
                            loaded_weight = (
                                loaded_weight.view(num_kv_heads, head_dim, -1)
                                .repeat_interleave(replicas, dim=0)
                                .reshape(attn_tp_size * head_dim, -1)
                            )
                        else:
                            kv_head_idx = get_parallel().attn_tp_rank // replicas
                            loaded_weight = loaded_weight.narrow(
                                0, kv_head_idx * head_dim, head_dim
                            )

            if name.endswith(".embed.weight"):
                name = name.replace(".embed.weight", ".embed_tokens.weight")
            elif name.endswith(".unembed.weight"):
                name = name.replace(".unembed.weight", ".lm_head.weight")
            elif name.endswith(".embed_tokens.embed_norm.weight"):
                name = name.replace(
                    ".embed_tokens.embed_norm.weight", ".embed_norm.weight"
                )

            if name == "llm.embed_tokens.weight":
                embed_tokens_weight = loaded_weight

            loaded_weight = self._slice_local_experts(name, loaded_weight)

            matched = False
            for param_name, weight_name, shard_id in ATTENTION_PARAMS_MAPPING:
                if f".attn.{weight_name}." not in name:
                    continue
                sgl_name = name.replace(f".{weight_name}.", f".{param_name}.")
                matched = self._load_regular_param(
                    params_dict, loaded_params, sgl_name, loaded_weight, shard_id
                )
                break
            if matched:
                continue

            if ".mlp.w13_dn.weight" in name:
                sgl_name = name.replace(".w13_dn.", ".gate_up_proj.")
                if sgl_name in params_dict:
                    param = params_dict[sgl_name]
                    if loaded_weight.shape != param.data.shape:
                        shard_size = param.data.shape[0]
                        start = get_parallel().attn_tp_rank * shard_size
                        loaded_weight = loaded_weight.narrow(0, start, shard_size)
                    if lora_compatible_layout_enabled():
                        # Local interleaved rows -> [gate||up] so contiguous swiglu and
                        # stock LoRA gate_up slicing line up (see InklingDenseMLP.__init__).
                        loaded_weight = deinterleave_gate_up(loaded_weight, dim=0)
                    default_weight_loader(param, loaded_weight)
                    loaded_params.add(sgl_name)
                    continue

            if ".mlp.w2_md.weight" in name:
                sgl_name = name.replace(".w2_md.", ".down_proj.")
                if self._load_regular_param(
                    params_dict, loaded_params, sgl_name, loaded_weight
                ):
                    continue

            for param_name, weight_name, shard_id in STACKED_DENSE_PARAMS_MAPPING:
                if f".mlp.{weight_name}." in name:
                    sgl_name = name.replace(f".{weight_name}.", f".{param_name}.")
                    matched = self._load_regular_param(
                        params_dict, loaded_params, sgl_name, loaded_weight, shard_id
                    )
                    break
            if matched:
                continue

            _scale_matched = False
            if ".experts." in name or ".shared_experts." in name:
                # Map checkpoint NVFP4 aux tensors to the owning quant method's params.
                # Routed experts use ModelOptNvFp4FusedMoEMethod, which registers
                # w13_weight_scale / w13_weight_scale_2 / w13_input_scale; shared experts
                # still use InklingNvfp4MoEMethod (w13_scale / w13_scale2 / w13_input_amax).
                # Prefer the ModelOpt param (applying the scale2 (E,)->(E,2) and the
                # input_amax -> input_scale = amax/(448*6) conversions); otherwise fall
                # back to the Inkling param name with a direct copy.
                for _suf in ("scale2", "scale", "input_amax", "original_shape"):
                    if not name.endswith(f"_weight.{_suf}"):
                        continue
                    _scale_matched = True
                    base = name.replace("shared_w13_weight", "w13_weight").replace(
                        "shared_w2_weight", "w2_weight"
                    )
                    prefix = base[: base.rfind("_weight.")]
                    mo_suf = {
                        "scale": "weight_scale",
                        "scale2": "weight_scale_2",
                        "input_amax": "input_scale",
                    }.get(_suf)
                    mo_name = f"{prefix}_{mo_suf}" if mo_suf else None
                    inkling_name = f"{prefix}_{_suf}"
                    if mo_name is not None and mo_name in params_dict:
                        conv = self._ckpt_scale_to_modelopt(
                            _suf, loaded_weight, params_dict[mo_name]
                        )
                        self._load_nvfp4_scale_param(
                            params_dict, loaded_params, mo_name, conv
                        )
                    elif inkling_name in params_dict:
                        self._load_nvfp4_scale_param(
                            params_dict, loaded_params, inkling_name, loaded_weight
                        )
                    elif _suf != "original_shape":
                        # ModelOpt path has no original_shape param, so dropping that one
                        # is expected; anything else going missing is a real problem.
                        logger.warning(
                            "NVFP4 scale tensor %s not mapped to any param; dropped",
                            name,
                        )
                    break
            if _scale_matched:
                continue

            if ".experts.w13_weight" in name:
                # bf16 routed layers run the stock FusedMoE forward (not moe_tp_forward)
                # under --enable-lora, or natively on trtllm_routed for UNQUANTIZED
                # checkpoints: de-interleave per moe_tp block for the stock weight prep.
                if (
                    loaded_weight.dtype != torch.uint8
                    and self.text_config.inference_moe_w13_interleaved
                    and (
                        lora_compatible_layout_enabled()
                        or bf16_routed_uses_stock_fused_moe(self.quant_config)
                    )
                ):
                    tp = get_parallel().moe_tp_size
                    n_e, two_f, hid = loaded_weight.shape
                    loaded_weight = deinterleave_gate_up(
                        loaded_weight.view(n_e, tp, two_f // tp, hid), dim=2
                    )
                    if trtllm_bf16_weight_prep_enabled():
                        # trtllm bf16 weight prep consumes [up || gate] per rank
                        # ("w3_w1" order); triton/marlin consume [gate || up].
                        half = two_f // tp // 2
                        loaded_weight = torch.cat(
                            [loaded_weight[:, :, half:], loaded_weight[:, :, :half]],
                            dim=2,
                        )
                    loaded_weight = loaded_weight.view(n_e, two_f, hid)
                if self._load_fused_moe_param(
                    params_dict, loaded_params, name, loaded_weight, "w13"
                ):
                    continue
            if ".experts.w2_weight" in name:
                if self._load_fused_moe_param(
                    params_dict, loaded_params, name, loaded_weight, "w2"
                ):
                    continue
            if ".shared_experts.shared_w13_weight" in name:
                # InklingSharedFusedMoE's bf16 path needs contiguous [gate||up] w13 for the
                # SRT runner's silu_and_mul, unlike the interleaved bmm/moe_tp_forward paths.
                # Per-rank blocks are sized by the FULL tp group (InklingSharedFusedMoE always
                # shards over it at EP=1), NOT moe_tp (= tp/ep, wrong under --ep-size > 1).
                if (
                    loaded_weight.dtype != torch.uint8
                    and self.text_config.inference_moe_w13_interleaved
                    and use_inkling_shared_fused_moe()
                ):
                    tp = get_parallel().tp_size
                    n_e, two_f, hid = loaded_weight.shape
                    loaded_weight = deinterleave_gate_up(
                        loaded_weight.view(n_e, tp, two_f // tp, hid), dim=2
                    )
                    if (
                        get_moe_runner_backend().is_experimental_sgl_trtllm()
                        or bf16_routed_uses_stock_fused_moe(self.quant_config)
                        or shared_sink_uses_trtllm_bf16()
                    ):
                        # TRT-LLM BF16 weight preparation consumes [up || gate]
                        # per rank, including the shared sink on quantized models.
                        half = two_f // tp // 2
                        loaded_weight = torch.cat(
                            [loaded_weight[:, :, half:], loaded_weight[:, :, :half]],
                            dim=2,
                        )
                    loaded_weight = loaded_weight.view(n_e, two_f, hid)
                sgl_name = name.replace("shared_w13_weight", "w13_weight")
                if self._load_fused_moe_param(
                    params_dict, loaded_params, sgl_name, loaded_weight, "w13"
                ):
                    continue
            if ".shared_experts.shared_w2_weight" in name:
                sgl_name = name.replace("shared_w2_weight", "w2_weight")
                if self._load_fused_moe_param(
                    params_dict, loaded_params, sgl_name, loaded_weight, "w2"
                ):
                    continue
            if self._load_per_expert_param(
                params_dict, loaded_params, name, loaded_weight
            ):
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if self._load_regular_param(
                params_dict, loaded_params, name, loaded_weight
            ):
                continue

        if (
            "llm.lm_head.weight" not in loaded_params
            and "llm.lm_head.weight" in params_dict
        ):
            if embed_tokens_weight is not None:
                self._load_regular_param(
                    params_dict,
                    loaded_params,
                    "llm.lm_head.weight",
                    embed_tokens_weight,
                )
        return loaded_params


class InklingMTPLayer(nn.Module):
    """Single MTP layer following torchtitan's MultiTokenPredictorModule structure."""

    def __init__(
        self,
        config: InklingModelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_id: int | None = None,
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()

        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.padded_vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.main_model_embed_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )
        self.mup_width_multiplier = config.logits_mup_width_multiplier
        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha
        self.mtp_layer_id = layer_id if layer_id is not None else 0
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Shared (non-layer-indexed) chain post-norm applied per depth before both
        # the chain handoff and the LM head.
        self.chain_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.chain_hidden_post_norm
            else None
        )
        self.input_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        # MTP blocks always use the dense MLP; the dense_mlp_idx override forces it.
        mtp_config = copy.copy(config)
        mtp_config.dense_mlp_idx = (layer_id or 0) + 1
        is_local = self.mtp_layer_id in config.mtp_local_layer_ids
        if is_local:
            # A banded depth runs at the HEAD's window, not the trunk's: the
            # checkpoint's rel_logits_proj was trained at the head's window.
            mtp_config.sliding_window_size = config.mtp_local_extent
            mtp_config.swa_num_attention_heads = config.mtp_swa_num_attention_heads
            mtp_config.swa_num_key_value_heads = config.mtp_swa_num_key_value_heads
            mtp_config.swa_head_dim = config.mtp_swa_head_dim
        self.alt_stream = torch.cuda.Stream() if is_cuda() else None
        self.transformer_block = InklingDecoderLayer(
            config=mtp_config,
            layer_id=layer_id if layer_id is not None else 0,
            is_local=is_local,
            quant_config=quant_config,
            prefix=add_prefix("transformer_block", prefix),
            alt_stream=alt_stream,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embed_tokens(input_ids)
        if self.main_model_embed_norm is not None:
            embeds = self.main_model_embed_norm(embeds)

        hnorm = self.hidden_norm(forward_batch.spec_info.hidden_states)
        enorm = self.embed_norm(embeds)
        combined = torch.cat((hnorm, enorm), dim=-1)
        h_inproj = self.input_proj(combined)
        log_scaling_tau = (
            compute_log_scaling_tau(
                positions,
                self.log_scaling_n_floor,
                self.log_scaling_alpha,
            )
            if self.log_scaling_n_floor is not None
            else None
        )
        fm_idle = forward_batch.forward_mode.is_idle()
        fuse_attn_ar = not fm_idle and ar_sconv_norm_fusable(
            get_parallel().attn_tp_group,
            forward_batch,
            h_inproj.shape[0],
            h_inproj.shape[-1],
            h_inproj.dtype,
        )
        if not fm_idle and scattered_ar_sconv_fusable(
            get_tensor_model_parallel_group(),
            forward_batch,
            h_inproj.shape[0],
            h_inproj.shape[-1],
            h_inproj.dtype,
        ):
            fuse_attn_ar = True
        # MTP is intentionally outside the main-model AttnRes loop.
        h, residual = self.transformer_block(
            h_inproj,
            positions,
            forward_batch,
            None,
            log_scaling_tau=log_scaling_tau,
            fuse_attn_ar=fuse_attn_ar,
        )
        mlp_sconv = self.transformer_block.mlp_sconv
        if mlp_sconv is not None and not fm_idle:
            h = mlp_sconv(h, positions, forward_batch)
            if self.transformer_block.scattered_sconv:
                # h was the [T, H/P] shard; gather before the residual add.
                h = all_gather_hidden(h, self.transformer_block.attn_tp_group)
        # transformer_block defers the final residual add to the caller.
        if residual is not None:
            h = h + residual
        # chain_norm applies to the raw block output, before the mup division.
        if self.chain_norm is not None:
            h = self.chain_norm(h)
        if self.mup_width_multiplier is not None:
            return h / self.mup_width_multiplier, h
        return h, h


class InklingForConditionalGenerationMTP(nn.Module):
    """MTP draft model for Inkling speculative decoding.

    Each instance represents a single MTP layer. In multi-layer MTP,
    MultiLayerEagleDraftWorker creates one instance per layer, each loading
    different weights via draft_model_idx filtering.
    """

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: InklingMMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        draft_model_idx: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.text_config = config.text_config
        if config.text_config.mtp_local_layer_ids and draft_model_idx is None:
            # The banded per-depth pool routing (SWA ring at full capacity with
            # an identity mapping) only exists on the multi-layer draft path.
            raise NotImplementedError(
                "a banded MTP head (mtp_local_layer_ids set) requires "
                "--enable-multi-layer-eagle"
            )
        self.draft_model_idx = draft_model_idx if draft_model_idx is not None else 0
        # chain_hidden_post_norm lives in the checkpoint's mtp_config; InklingMTPLayer
        # reads it off text_config.
        if isinstance(config.mtp_config, dict):
            config.text_config.chain_hidden_post_norm = config.mtp_config.get(
                "chain_hidden_post_norm", config.text_config.chain_hidden_post_norm
            )
        # The MTP block is bf16 in the checkpoint (no mtp.* quant excludes), so build
        # the draft unquantized.
        quant_config = None
        # Without an alt_stream the InklingAttention fused-prologue gate can
        # never pass in the draft, leaving every draft forward on the unfused
        # {2x causal_conv1d + qk-norm + KV-store scatter (+ tau scale)} chain.
        self.alt_stream = torch.cuda.Stream() if is_cuda() else None
        self.model = InklingMTPLayer(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            layer_id=self.draft_model_idx,
            alt_stream=self.alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.text_config.padded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=config.text_config.padded_vocab_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config.text_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        hidden_states, hidden_states_before_norm = self.model(
            input_ids, positions, forward_batch
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=hidden_states_before_norm,
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        # Every de-tied MTP head loads the same bf16 embed/unembed as the target,
        # so alias the target's tensors rather than keep one duplicate per head.
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            name = name.removeprefix("model.")

            if "mtp.chain_norm." in name:
                name = "model.chain_norm." + name.split("mtp.chain_norm.", 1)[1]
            elif ".mtp.layers." in name or name.startswith("mtp."):
                # The loader's _filter_mtp_weights already kept only this head's
                # layer and remapped it to mtp.layers.0.
                name = re.sub(r".*mtp\.(?:model\.)?layers\.\d+\.", "model.", name)

            if name in ("llm.embed_tokens.weight", "llm.embed.weight", "embed.weight"):
                name = "model.embed_tokens.weight"
            elif name in ("llm.lm_head.weight", "llm.unembed.weight", "unembed.weight"):
                name = "lm_head.weight"
            elif name in ("embed_norm.weight", "llm.embed_norm.weight"):
                name = "model.main_model_embed_norm.weight"
            elif name.startswith("llm.") and ".mtp." not in name:
                continue

            matched = False
            for param_name, weight_name, shard_id in ATTENTION_PARAMS_MAPPING:
                if f".attn.{weight_name}." not in name:
                    continue
                sgl_name = name.replace(f".{weight_name}.", f".{param_name}.")
                if sgl_name in params_dict:
                    param = params_dict[sgl_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(sgl_name)
                matched = True
                break
            if matched:
                continue

            if ".mlp.w13_dn.weight" in name:
                sgl_name = name.replace(".w13_dn.", ".gate_up_proj.")
                if sgl_name in params_dict:
                    param = params_dict[sgl_name]
                    if loaded_weight.shape != param.data.shape:
                        shard_size = param.data.shape[0]
                        start = get_parallel().attn_tp_rank * shard_size
                        loaded_weight = loaded_weight.narrow(0, start, shard_size)
                    default_weight_loader(param, loaded_weight)
                    loaded_params.add(sgl_name)
                continue
            if ".mlp.w2_md.weight" in name:
                name = name.replace(".w2_md.", ".down_proj.")
            else:
                for param_name, weight_name, shard_id in STACKED_DENSE_PARAMS_MAPPING:
                    if f".mlp.{weight_name}." in name:
                        sgl_name = name.replace(f".{weight_name}.", f".{param_name}.")
                        if sgl_name in params_dict:
                            param = params_dict[sgl_name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            if shard_id is None:
                                weight_loader(param, loaded_weight)
                            else:
                                weight_loader(param, loaded_weight, shard_id)
                            loaded_params.add(sgl_name)
                        matched = True
                        break
                if matched:
                    continue

            if ".mlp.shared_experts." in name:
                name = name.replace(
                    ".mlp.shared_experts.", ".mlp.experts.shared_experts."
                )
            for needle, shard in (
                (".experts.w13_weight", "w13"),
                (".experts.w2_weight", "w2"),
            ):
                if needle in name and name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if weight_loader is default_weight_loader:
                        default_weight_loader(param, loaded_weight)
                    else:
                        weight_loader(param, loaded_weight, name, shard)
                    loaded_params.add(name)
                    matched = True
                    break
            if matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        unloaded = sorted(set(params_dict) - loaded_params)
        if unloaded:
            msg = (
                f"MTP draft (idx {self.draft_model_idx}): {len(unloaded)} unloaded "
                f"weights (loaded {len(loaded_params)}/{len(params_dict)}); an unloaded "
                f"RMSNorm stays all-ones and silently miscalibrates the draft. "
                f"First 20: {unloaded[:20]}"
            )
            raise RuntimeError(msg)
        return loaded_params


EntryClass = [InklingForConditionalGeneration, InklingForConditionalGenerationMTP]
