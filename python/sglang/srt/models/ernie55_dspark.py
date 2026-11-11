# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The DSpark draft an ERNIE 5.5 Flash checkpoint bundles under ``model.mtp.*``.

Each stage is an ERNIE MoE decoder layer with the uncompressed DSpark draft
attention; stage 0 also projects the target's captured hidden states into every
stage's KV.
:func:`sglang.srt.configs.ernie55.to_dspark_draft_config` reshapes the target
config into this one.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.activation.multimax import multimax
from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
from sglang.kernels.ops.attention.dsv4.rope import precompute_freqs_cis
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj
from sglang.srt.configs.ernie55 import Ernie5_5_MoeConfig
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
from sglang.srt.layers.layernorm import RMSNorm, exact_rms_norm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul
from sglang.srt.models.deepseek_v4_dspark import (
    _PAD_NUM_HEADS,
    DSparkAttention,
    DSparkV4MarkovHead,
    _resolve_dspark_pool,
    build_dspark_v4_confidence_head,
)
from sglang.srt.models.dspark import gather_and_crop_vocab, project_through_lm_head
from sglang.srt.models.ernie55 import (
    Ernie5_5DecoderLayer,
    Ernie5_5GatedQMixin,
    Ernie5_5HcHead,
    Ernie5_5WeightLoader,
    VhaPostMix,
    check_supported_deployment,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.speculative.dspark_components.dspark_config import (
    get_dspark_sample_from_anchor,
    parse_dspark_draft_config,
)


class Ernie5_5DSparkAttention(Ernie5_5GatedQMixin, DSparkAttention):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[list] = None,
    ):
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
        )
        # compress_ratio 0 selected the main rope, but the stages were trained on
        # the compressed YaRN rope of the HCA layers.
        self.rope_base = config.compress_rope_theta
        self.rope_kwargs.update(
            base=self.rope_base,
            original_seq_len=self.rope_scaling["original_max_position_embeddings"],
        )
        self.freqs_cis = precompute_freqs_cis(**self.rope_kwargs)
        self._init_gated_wq_b(quant_config, prefix)
        self.postmix = VhaPostMix(self.n_local_heads, self.n_heads)

    def _compute_q(self, x, positions, q_out=None):
        q, _ = self.wq_a(x)
        return self._gated_q_norm_rope(self.q_norm(q), positions, q_out)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """The HCA attention's signature, so ``Ernie5_5DecoderLayer`` drives it."""
        # Idle ranks have no draft KV rows but still join the MoE collectives.
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(x)
        pool = _resolve_dspark_pool()
        attn_backend = get_attn_backend()
        kv = self.kv_proj_only(x)
        self._store_block_kv(
            kv=kv,
            positions=positions,
            forward_batch=forward_batch,
            attn_backend=attn_backend,
            pool=pool,
        )
        q_padded = q_out = None
        if self.n_local_heads < _PAD_NUM_HEADS:
            q_padded = x.new_empty(x.shape[0], _PAD_NUM_HEADS, self.head_dim)
            q_out = q_padded[:, : self.n_local_heads]
        q = self._compute_q(x, positions, q_out=q_out)
        o = attn_backend.forward(
            q=q if q_padded is None else q_padded,
            k=kv,
            v=kv,
            layer=self.attn,
            forward_batch=forward_batch,
            compress_ratio=0,
            attn_sink=self._local_attn_sink(),
            save_kv_cache=False,
        )
        o = o[:, : self.n_local_heads]
        fused_rope_inplace(
            o[..., -self.rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )
        o = self.postmix(o).reshape(o.shape[0], self.n_local_groups, -1)
        o = _apply_wo_a_bf16_matmul(
            o,
            self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1),
            is_decode=forward_batch.forward_mode.is_decode(),
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
        )
        o = fused_sigmoid_mul(
            o.reshape(o.shape[0], -1), self._take_gate(), round_sigmoid=True
        )
        out, _ = self.wo_b(o)
        return out


class Ernie5_5DSparkStage(Ernie5_5DecoderLayer):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        stage_id: int,
        num_target_layers: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[list] = None,
    ):
        super().__init__(config, stage_id, quant_config, prefix, alt_streams)
        if stage_id == 0:
            # Shipped in bf16, so it takes no quant config.
            self.main_proj = ReplicatedLinear(
                config.hidden_size * num_target_layers,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=add_prefix("main_proj", prefix),
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def _build_self_attn(config, layer_id, quant_config, prefix, alt_streams):
        return Ernie5_5DSparkAttention(
            config, layer_id, quant_config, prefix, alt_streams
        )


class Ernie5_5_MoeForCausalLMDSpark(nn.Module):
    # The draft embeds and projects through the target's vocabulary modules.
    uses_own_vocab_modules = False

    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        config.validate_checkpoint_fields()
        check_supported_deployment(config)
        self.config = config
        self.quant_config = quant_config
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "the ERNIE 5.5 Flash DSpark draft needs a Markov refine head "
                f"(dspark_markov_rank > 0), got {dspark_config.markov_rank}"
            )
        self.gamma = int(dspark_config.resolve_gamma(default=config.num_hidden_layers))
        self.block_size = self.gamma
        self.sample_from_anchor = get_dspark_sample_from_anchor(config)
        self.num_stages = len(dspark_config.target_layer_ids)
        self.start_layer, self.end_layer = 0, self.num_stages
        self.hc_mult = config.hc_mult
        alt_streams = (
            [torch.cuda.Stream()]
            if torch.cuda.is_available()
            and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and envs.SGLANG_DSPARK_ENABLE_MULTI_STREAM.get()
            else None
        )
        self.stages = nn.ModuleList(
            [
                Ernie5_5DSparkStage(
                    config,
                    stage_id,
                    self.num_stages,
                    quant_config,
                    add_prefix(f"stages.{stage_id}", prefix),
                    alt_streams,
                )
                for stage_id in range(self.num_stages)
            ]
        )
        self.hc_head = Ernie5_5HcHead(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkV4MarkovHead(
            vocab_size=int(config.vocab_size),
            markov_rank=int(dspark_config.markov_rank),
        )
        self.confidence_head = build_dspark_v4_confidence_head(
            config=config, markov_rank=int(dspark_config.markov_rank)
        )
        # The target's multimax, applied to the draft logits as in training.
        self.multimax_ranges = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        self.multimax_ts = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
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
        return exact_rms_norm(projected, stage0.main_norm)

    def write_target_hidden_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        kvs = self.project_commit_kv(main_hidden=main_hidden, pool=pool)
        self.store_commit_kv(kvs=kvs, swa_loc=swa_loc, positions=positions, pool=pool)

    def project_commit_kv(
        self, *, main_hidden: torch.Tensor, pool: DeepSeekV4TokenToKVPool
    ) -> List[torch.Tensor]:
        """The context rows' per-stage KV before norm / rope / store, computed
        before the accept decides which rows are committed."""
        main_x = self.project_target_hidden(main_hidden)
        return CommitKvProj.execute(
            main_x=main_x,
            wkv_linears=[stage.self_attn.wkv for stage in self.stages],
            # The FlashMLA writer reads an explicit KV row stride, so views are fine.
            allow_strided_output=(
                get_platform().is_blackwell
                and not is_unified_kv_triton()
                and not pool.uniform_fp8
            ),
        )

    def store_commit_kv(
        self,
        *,
        kvs: List[torch.Tensor],
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        store_kv = (
            pool.set_unified_key_buffer_radix_fused_norm_rope
            if is_unified_kv_triton()
            else pool.set_swa_key_buffer_radix_fused_norm_rope
        )
        swa_loc = swa_loc.to(torch.int32)
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
            raise RuntimeError("attach_shared_modules must run before the draft")
        return self._expand_streams(self.embed_tokens(input_ids))

    def _expand_streams(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            x = self.forward_embed(input_ids)
        elif input_embeds.dim() == 2:
            x = self._expand_streams(input_embeds)
        else:
            x = input_embeds
        # The draft stages are not the target's layers the recorder counts.
        with get_global_expert_distribution_recorder().disable_this_region():
            for stage in self.stages:
                x = stage(positions, x, forward_batch, None)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)

    def compute_base_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_post_hc = self.hc_head(x)
        x_normed = exact_rms_norm(x_post_hc, self.norm)
        weight = self.lm_head.weight
        if self._use_fp32_lm_head and weight.is_floating_point():
            local_logits = torch.nn.functional.linear(x_normed.float(), weight.float())
        else:
            local_logits = project_through_lm_head(x_normed, self.lm_head)
        local_logits = multimax(local_logits, self.multimax_ranges, self.multimax_ts)
        if not self._opt_markov_w2_tp_shard:
            local_logits = gather_and_crop_vocab(local_logits, self.lm_head)
        return local_logits, x_post_hc

    def compute_confidence(
        self,
        *,
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        confidence_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.confidence_head.predict(
            confidence_features,
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            markov_head=self.markov_head,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        _Ernie5_5DSparkWeightLoader(self).load_all(weights)
        self.post_load_weights()

    def post_load_weights(self) -> None:
        for stage in self.stages:
            stage.self_attn.postmix.fold()


class _Ernie5_5DSparkWeightLoader(Ernie5_5WeightLoader):
    # The draft attention keeps wq_a and wkv apart, which _HCA_UNFUSED_QKV_A loads.
    _HCA_STACKED = {}
    _DRAFT_RENAMES = {
        "model.mtp.feature_projection.weight": "stages.0.main_proj.weight",
        "model.mtp.feature_norm.weight": "stages.0.main_norm.weight",
        "model.mtp.norm.weight": "norm.weight",
        "model.mtp.mhc_contract.hc_head_fn": "hc_head.fn",
        "model.mtp.mhc_contract.hc_head_base": "hc_head.base",
        "model.mtp.mhc_contract.hc_head_scale": "hc_head.scale",
        "model.mtp.markov_head.markov_w2.weight": "markov_head.markov_w2.weight",
        "model.mtp.confidence_head.projection.weight": "confidence_head.proj.weight",
        "model.lm_head.multimax_ranges": "multimax_ranges",
        "model.lm_head.multimax_ts": "multimax_ts",
    }
    # Stored [in, out]; the parameters are [out, in].
    _DRAFT_TRANSPOSED = {
        "model.mtp.markov_head.markov_w1.weight": "markov_head.markov_w1.weight",
    }
    _DRAFT_PREFIX = "model.mtp."

    def __init__(self, model: Ernie5_5_MoeForCausalLMDSpark):
        self._init_tables(model, layers=model.stages, mla_layers=())
        # The static ragged-verify mode builds no confidence head.
        self.skipped = (
            () if model.confidence_head is not None else ("confidence_head.",)
        )

    def load(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._DRAFT_TRANSPOSED:
            target = self._DRAFT_TRANSPOSED[name]
            self.params[target].data.copy_(tensor.t())
            self.loaded.add(target)
            return
        target = self._DRAFT_RENAMES.get(name)
        if target is not None:
            if not target.startswith(self.skipped):
                self._load(target, tensor)
            return
        if not name.startswith(self._DRAFT_PREFIX):
            return
        stage_text, _, suffix = name[len(self._DRAFT_PREFIX) :].partition(".")
        if not stage_text.isdigit():
            raise RuntimeError(f"unexpected DSpark draft tensor {name}")
        stage_id = int(stage_text)
        self._load_layer_tensor(f"stages.{stage_id}.", stage_id, suffix, tensor)


EntryClass = [Ernie5_5_MoeForCausalLMDSpark]
