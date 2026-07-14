from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.jit_kernel.dsv4 import fused_q_norm_rope, fused_rope_inplace
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v4 import (
    DEEPSEEK_V4_STACKED_PARAMS_MAPPING,
    DeepseekV4DecoderLayer,
    MqaAttentionBase,
    _dequant_fp8_wo_a,
    hc_head_torch,
    make_hc_head_params,
)
from sglang.srt.models.dspark import (
    DSparkConfidenceHead,
    StepSampler,
    gather_and_crop_vocab,
    run_markov_block,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_draft_model import (
    BuildStepLocal,
    CommitKvProj,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.utils import add_prefix, is_blackwell_supported
from sglang.srt.utils.async_probe import maybe_detect_in_closed_range

logger = logging.getLogger(__name__)

_PAD_NUM_HEADS = 64


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(x.size(0), 1, x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class DSparkAttention(MqaAttentionBase):

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            attn_tp_rank=get_parallel().attn_tp_rank,
            attn_tp_size=get_parallel().attn_tp_size,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=False,
            wo_a_keeps_quant_config=False,
            wo_b_reduce_results=True,
            rope_original_seq_len=0,
        )
        assert (
            self.compress_ratio == 0
        ), "DSpark draft attention requires compress_ratio == 0."
        self.window_size = int(
            getattr(config, "sliding_window", None) or config.window_size
        )

        self.attn = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self._use_fast_kernel = envs.SGLANG_DSPARK_FAST_KERNEL.get()
        self.alt_streams = alt_streams
        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

    def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:
        kv, _ = self.wkv(x)
        return kv

    def _local_attn_sink(self) -> torch.Tensor:
        if self.attn_tp_size == 1:
            return self.attn_sink
        if self._attn_sink_local is None:
            rank = self.attn_tp_rank
            num_heads = self.n_local_heads
            sink = self.attn_sink.new_zeros(max(num_heads, _PAD_NUM_HEADS))
            sink[:num_heads] = self.attn_sink[rank * num_heads : (rank + 1) * num_heads]
            self._attn_sink_local = sink
        return self._attn_sink_local

    def _store_block_kv(
        self,
        *,
        kv: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        if is_unified_kv_triton():
            # unified_kv: SWA K lives in the shared bf16 ring (swa_kv_pool is
            # None). Use the unified ring write target -- get_unified_swa_loc
            # recomputes it from live positions for multi-step draft decode.
            pool.set_unified_key_buffer_radix_fused_norm_rope(
                layer_id=self.layer_id,
                swa_loc=attn_backend.get_unified_swa_loc(forward_batch),
                kv=kv,
                kv_weight=self.kv_norm.weight.data,
                eps=self.eps,
                freqs_cis=self.freqs_cis,
                positions=positions,
            )
            return
        pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def _compute_q(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self._use_fast_kernel:
            if q_out is None:
                q_out = torch.empty_like(q)
            fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
            return q_out
        else:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
            apply_rotary_emb(q[..., -self.rope_head_dim :], self.freqs_cis[positions])
            if q_out is not None:
                q_out.copy_(q)
                return q_out
            return q

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        pool = _resolve_dspark_pool()
        attn_backend = get_attn_backend()
        rd = self.rope_head_dim

        enable_multi_stream = (
            self.alt_streams is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] <= self._multi_stream_bs_limit
        )

        q_padded: Optional[torch.Tensor] = None
        q_out: Optional[torch.Tensor] = None
        if self.n_local_heads < _PAD_NUM_HEADS:
            q_padded = hidden_states.new_empty(
                hidden_states.shape[0], _PAD_NUM_HEADS, self.head_dim
            )
            q_out = q_padded[:, : self.n_local_heads, :]

        if enable_multi_stream:
            current_stream = torch.cuda.current_stream()
            stream_kv = self.alt_streams[0]
            stream_kv.wait_stream(current_stream)
            with torch.cuda.stream(stream_kv):
                kv = self.kv_proj_only(hidden_states)
                self._store_block_kv(
                    kv=kv,
                    positions=positions,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    pool=pool,
                )
            q = self._compute_q(hidden_states, positions, q_out=q_out)
            current_stream.wait_stream(stream_kv)
        else:
            kv = self.kv_proj_only(hidden_states)
            self._store_block_kv(
                kv=kv,
                positions=positions,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                pool=pool,
            )
            q = self._compute_q(hidden_states, positions, q_out=q_out)

        if q_padded is not None:
            q = q_padded
        attn_sink = self._local_attn_sink()

        o = attn_backend.forward(
            q=q,
            k=kv,
            v=kv,
            layer=self.attn,
            forward_batch=forward_batch,
            compress_ratio=0,
            attn_sink=attn_sink,
            save_kv_cache=False,
        )
        if o.shape[1] != self.n_local_heads:
            o = o[:, : self.n_local_heads, :]

        if self._use_fast_kernel:
            fused_rope_inplace(
                o[..., -rd:], None, self.freqs_cis, positions=positions, inverse=True
            )
        else:
            apply_rotary_emb(o[..., -rd:], self.freqs_cis[positions], inverse=True)

        o = o.view(
            o.shape[0],
            self.n_local_groups,
            o.shape[1] * o.shape[2] // self.n_local_groups,
        )
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if self._use_fast_kernel:
            o = torch.einsum("bgd,grd->bgr", o, wo_a)
        else:
            o = torch.einsum("bgd,grd->bgr", o.float(), wo_a.float()).to(q.dtype)
        out, _ = self.wo_b(o.reshape(o.shape[0], o.shape[1] * o.shape[2]))
        return out


def _resolve_dspark_pool() -> DeepSeekV4TokenToKVPool:
    pool = get_token_to_kv_pool()
    assert isinstance(pool, DeepSeekV4TokenToKVPool), (
        "DSpark draft attention requires a DeepSeekV4TokenToKVPool, "
        f"got {type(pool).__name__}."
    )
    return pool


class MarkovW2ShardGeometry(msgspec.Struct, frozen=True):

    tp_size: int
    org_vocab_start: int
    org_vocab_end: int
    num_embeddings_per_partition: int
    num_embeddings_padded: int


class DSparkV4MarkovHead(nn.Module):

    markov_head_type = "vanilla"

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        if self.markov_rank <= 0:
            raise ValueError(
                f"DSparkV4MarkovHead requires markov_rank > 0, got {self.markov_rank}."
            )
        self.markov_w1 = VocabParallelEmbedding(
            self.vocab_size, self.markov_rank, enable_tp=False
        )
        self._opt_markov_w2_bf16 = envs.SGLANG_DSPARK_OPT_MARKOV_W2_BF16.get()
        self._opt_markov_w2_tp_shard = envs.SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD.get()
        markov_w2_dtype = torch.bfloat16 if self._opt_markov_w2_bf16 else torch.float32
        self.markov_w2 = nn.Linear(
            self.markov_rank, self.vocab_size, bias=False, dtype=markov_w2_dtype
        )
        self._tp_shard: Optional[MarkovW2ShardGeometry] = None

    def configure_tp_shard(self, *, lm_head: nn.Module) -> None:
        if not self._opt_markov_w2_tp_shard:
            return
        if int(lm_head.org_vocab_size) != self.vocab_size:
            raise ValueError(
                "DSpark markov_w2 TP-shard requires lm_head.org_vocab_size == "
                f"markov vocab_size, got {int(lm_head.org_vocab_size)} vs "
                f"{self.vocab_size}."
            )
        tp_size = int(lm_head.tp_size)
        per_partition = int(lm_head.num_embeddings_per_partition)
        num_padded = int(lm_head.num_embeddings_padded)
        if per_partition * tp_size != num_padded:
            raise ValueError(
                "DSpark markov_w2 TP-shard could not align to the lm_head partition: "
                f"num_embeddings_per_partition({per_partition}) * tp_size({tp_size}) != "
                f"num_embeddings_padded({num_padded})."
            )
        attn_tp_size = get_parallel().attn_tp_group.world_size
        if attn_tp_size != tp_size:
            raise ValueError(
                "DSpark markov_w2 TP-shard needs the attn-TP group (used for the per-step "
                f"all-gather) to equal the lm_head shard group, got attn_tp_size="
                f"{attn_tp_size} vs lm_head tp_size={tp_size}. This config (e.g. DP "
                "attention without --enable-dp-lm-head, where lm_head shards over the "
                "global TP group) is unsupported; disable "
                "SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD."
            )
        self._tp_shard = MarkovW2ShardGeometry(
            tp_size=tp_size,
            org_vocab_start=int(lm_head.shard_indices.org_vocab_start_index),
            org_vocab_end=int(lm_head.shard_indices.org_vocab_end_index),
            num_embeddings_per_partition=per_partition,
            num_embeddings_padded=num_padded,
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(
        self, latent_states: torch.Tensor, *, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = self.markov_w2.weight if weight is None else weight
        if self._opt_markov_w2_bf16:
            return F.linear(latent_states.to(weight.dtype), weight).float()
        return F.linear(latent_states.float(), weight)

    def compute_step_bias(
        self, token_ids: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._tp_shard is not None:
            return self._apply_step_logits_sharded(
                base_local=logits, token_ids=token_ids
            )
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def _apply_step_logits_sharded(
        self, *, base_local: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        shard = self._tp_shard
        latent = self.get_prev_embeddings(token_ids)
        weight_local = self.markov_w2.weight[
            shard.org_vocab_start : shard.org_vocab_end
        ]
        if self._opt_markov_w2_bf16:
            bias = F.linear(latent.to(weight_local.dtype), weight_local)
        else:
            bias = F.linear(latent.float(), weight_local)
        step_local = BuildStepLocal.execute(bias=bias, base_local=base_local)
        if shard.tp_size > 1:
            full = get_parallel().attn_tp_group.all_gather(step_local, dim=-1)
        else:
            full = step_local
        return full[..., : self.vocab_size]

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.get_prev_embeddings(token_ids)
        logits = self.project_bias(embed)
        return logits, embed

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        sampler: StepSampler,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return run_markov_block(
            self,
            base_logits,
            first_prev_tokens=first_prev_tokens,
            hidden_states=hidden_states,
            sampler=sampler,
        )


def build_dspark_v4_confidence_head(
    *, config: DeepSeekV4Config, markov_rank: int
) -> Optional[DSparkConfidenceHead]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
    if not hasattr(config, "enable_confidence_head"):
        logger.warning(
            "DSpark draft config has no enable_confidence_head field; treating the "
            "confidence head as enabled."
        )
    with_markov_cfg = getattr(config, "confidence_head_with_markov", None)
    with_markov = (
        (markov_rank > 0) if with_markov_cfg is None else bool(with_markov_cfg)
    )
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark V4 confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=int(config.hidden_size),
        markov_rank=int(markov_rank),
        with_markov=with_markov,
        bias=False,
    )


class DSparkV4Stage(DeepseekV4DecoderLayer):

    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        stage_id: int,
        num_stages: int,
        num_target_layers: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            is_nextn=True,
            alt_streams=alt_streams,
        )
        self.stage_id = stage_id
        self.dim = config.hidden_size

        if stage_id == 0:
            if num_target_layers <= 0:
                raise ValueError(
                    "DSpark needs target layers for the target-hidden projection."
                )
            self.main_proj = ReplicatedLinear(
                config.hidden_size * num_target_layers,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("main_proj", prefix),
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if stage_id == num_stages - 1:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            (
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            ) = make_hc_head_params(config.hc_mult, config.hidden_size)

    def _build_self_attn(
        self,
        *,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[List[torch.cuda.Stream]],
        compress_ratio_override: Optional[int],
    ) -> nn.Module:
        del compress_ratio_override
        return DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
        )

    def _hc_pre_block(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, post, comb, _ = self.hc_pre(x, hc_fn, hc_scale, hc_base)
        return y, post, comb

    def _hc_post_block(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return self.hc_post(x, residual, post, comb)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        x, post, comb = self._hc_pre_block(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, forward_batch)
        x = self._hc_post_block(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre_block(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.post_attention_layernorm(x)
        x = self._run_ffn(x, forward_batch)
        x = self._hc_post_block(x, residual, post, comb)
        return x

    def _run_ffn(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.dim)
        y = self._run_moe_ffn_dp_sync(
            x, forward_batch, input_ids=None, input_ids_global=None
        )
        return y.view(shape)


class DeepseekV4ForCausalLMDSpark(nn.Module):

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark V4 draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(
            dspark_config.resolve_gamma(default=int(config.num_hidden_layers))
        )
        self.block_size = self.gamma
        if dspark_config.target_layer_ids is not None:
            self.num_stages = len(dspark_config.target_layer_ids)
        else:
            self.num_stages = int(getattr(config, "num_nextn_predict_layers", 1) or 1)

        target_num_layers = (
            int(dspark_config.num_target_layers)
            if dspark_config.num_target_layers is not None
            else int(getattr(config, "num_hidden_layers", 1))
        )
        if dspark_config.target_layer_ids is not None:
            self.num_target_features = len(dspark_config.target_layer_ids)
        else:
            self.num_target_features = target_num_layers

        self.start_layer = 0
        self.end_layer = self.num_stages
        use_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and envs.SGLANG_DSPARK_ENABLE_MULTI_STREAM.get()
            and torch.cuda.is_available()
        )
        self.alt_streams: Optional[List[torch.cuda.Stream]] = (
            [torch.cuda.Stream()] if use_multi_stream else None
        )
        self.stages = nn.ModuleList(
            [
                DSparkV4Stage(
                    config=config,
                    layer_id=stage_id,
                    stage_id=stage_id,
                    num_stages=self.num_stages,
                    num_target_layers=self.num_target_features,
                    quant_config=quant_config,
                    prefix=add_prefix(f"stages.{stage_id}", prefix),
                    alt_streams=self.alt_streams,
                )
                for stage_id in range(self.num_stages)
            ]
        )
        self.markov_head = DSparkV4MarkovHead(
            vocab_size=int(config.vocab_size),
            markov_rank=int(dspark_config.markov_rank),
        )
        self.confidence_head = build_dspark_v4_confidence_head(
            config=config, markov_rank=int(dspark_config.markov_rank)
        )
        self.hc_mult = int(config.hc_mult)
        self.norm_eps = float(config.rms_norm_eps)
        self.hc_eps = float(config.hc_eps)

        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()
        self._opt_markov_w2_tp_shard = envs.SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD.get()

    @property
    def enable_confidence_head(self) -> bool:
        return self.confidence_head is not None

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.markov_head.configure_tp_shard(lm_head=lm_head)

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        stage0 = self.stages[0]
        projected, _ = stage0.main_proj(main_hidden)
        return stage0.main_norm(projected)

    def write_target_hidden_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        main_x = self.project_target_hidden(main_hidden)
        swa_loc = swa_loc.to(torch.int32)
        kvs = CommitKvProj.execute(
            main_x=main_x,
            wkv_linears=[stage.self_attn.wkv for stage in self.stages],
        )
        # Under unified_kv the swa_kv_pool is None; the caller passes a unified
        # ring loc (state_slot * ring + pos % ring, -1 for uncommitted) so the
        # store just needs to target the bf16 ring instead of the fp8 flashmla
        # buffer. Same swa_loc/positions contract either way.
        store_kv = (
            pool.set_unified_key_buffer_radix_fused_norm_rope
            if is_unified_kv_triton()
            else pool.set_swa_key_buffer_radix_fused_norm_rope
        )
        for stage, kv in zip(self.stages, kvs):
            attn = stage.self_attn
            store_kv(
                layer_id=attn.layer_id,
                swa_loc=swa_loc,
                kv=kv,
                kv_weight=attn.kv_norm.weight.data,
                eps=attn.eps,
                freqs_cis=attn.freqs_cis,
                positions=positions,
            )

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise ValueError(
                "DeepseekV4ForCausalLMDSpark requires the target embed_tokens "
                "(call attach_shared_modules first)."
            )
        x = self.embed_tokens(input_ids)
        x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors
        if input_embeds is None:
            input_embeds = self.forward_embed(input_ids)
        x = input_embeds
        for stage in self.stages:
            x = stage(positions, x, forward_batch)

        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)

    def collapse_hc_head(self, x: torch.Tensor) -> torch.Tensor:
        last = self.stages[-1]
        return hc_head_torch(
            x,
            last.hc_head_fn,
            last.hc_head_scale,
            last.hc_head_base,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def compute_base_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x_post_hc = self.collapse_hc_head(x)
        return self._logits_from_x_post_hc(x_post_hc), x_post_hc

    def _logits_from_x_post_hc(self, x_post_hc: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise ValueError(
                "DeepseekV4ForCausalLMDSpark requires the target lm_head "
                "(call attach_shared_modules first)."
            )
        last = self.stages[-1]
        x = last.norm(x_post_hc)
        weight = self.lm_head.weight
        if self._use_fp32_lm_head:
            local_logits = F.linear(x.float(), weight.float())
        else:
            local_logits = torch.matmul(x.to(weight.dtype), weight.T)
        if self._opt_markov_w2_tp_shard:
            return local_logits
        return gather_and_crop_vocab(local_logits, self.lm_head)

    def compute_confidence(
        self,
        *,
        anchor_tokens: torch.Tensor,
        sampled_tokens: torch.Tensor,
        x_post_hc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        confidence_head = self.confidence_head
        if confidence_head is None:
            return None
        bs = int(anchor_tokens.shape[0])
        x_post_hc = x_post_hc.view(bs, self.gamma, -1)
        if confidence_head.with_markov:
            prev_seq = torch.cat(
                [anchor_tokens.view(-1, 1), sampled_tokens[:, : self.gamma - 1]], dim=1
            )
            markov_embed_stack = self.markov_head.get_prev_embeddings(prev_seq)
        else:
            markov_embed_stack = None
        confidence_raw = confidence_head(x_post_hc, markov_embed_stack)
        confidence = confidence_head.apply_sts(confidence_raw)
        maybe_detect_in_closed_range(
            confidence, 0.0, 1.0, "DSpark confidence must lie in [0, 1]."
        )
        return confidence

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        weights = list(weights)
        if any(name.endswith(".wo_a.scale") for name, _ in weights):
            weights = list(_dequant_fp8_wo_a(weights))

        stacked_params_mapping = DEEPSEEK_V4_STACKED_PARAMS_MAPPING
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_weight_name(name)
            if mapped is None:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped:
                    continue
                candidate = mapped.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(candidate)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in mapped:
                        continue
                    candidate = mapped.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    param = params_dict[candidate]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        candidate,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(candidate)
                    break
                else:
                    if mapped not in params_dict:
                        logger.warning(
                            "DSpark V4 draft: unexpected weight %r -> %r", name, mapped
                        )
                        continue
                    param = params_dict[mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(mapped)

        self._assert_confidence_head_loaded(
            params_dict=params_dict, loaded_params=loaded_params
        )

    def _assert_confidence_head_loaded(
        self, *, params_dict: dict, loaded_params: set
    ) -> None:
        if self.confidence_head is None:
            return
        confidence_param_names = {
            name for name in params_dict if name.startswith("confidence_head.")
        }
        missing = confidence_param_names - loaded_params
        if missing:
            raise ValueError(
                f"DSpark V4 confidence head is enabled but the checkpoint is missing "
                f"{sorted(missing)}. Provide a checkpoint with trained confidence weights, "
                f"or disable the confidence head (enable_confidence_head=False)."
            )

    def _remap_dspark_weight_name(self, name: str) -> Optional[str]:
        if name.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
            return None
        if "rotary_emb.inv_freq" in name:
            return None

        if not name.startswith("mtp."):
            return None
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        stage_id, rest = parts[1], parts[2]

        if rest.startswith("markov_head."):
            return f"markov_head.{rest[len('markov_head.'):]}"

        if rest.startswith("confidence_head."):
            if self.confidence_head is None:
                return None
            return f"confidence_head.{rest[len('confidence_head.'):]}"

        mapped_rest = rest
        mapped_rest = mapped_rest.replace("attn.", "self_attn.", 1)
        mapped_rest = mapped_rest.replace("ffn.", "mlp.", 1)
        mapped_rest = mapped_rest.replace("attn_norm.", "input_layernorm.", 1)
        mapped_rest = mapped_rest.replace("ffn_norm.", "post_attention_layernorm.", 1)
        mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
        mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
        mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
        mapped_rest = mapped_rest.replace(".gate.tid2eid", ".topk.tid2eid")
        mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
        mapped_rest = mapped_rest.replace(".scale", ".weight_scale_inv")
        return f"stages.{stage_id}.{mapped_rest}"


EntryClass = [DeepseekV4ForCausalLMDSpark]
