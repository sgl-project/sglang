import functools
import logging
from typing import Iterable, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2MLP,
    DeepseekV2MoE,
)
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import BumpAllocator, get_device_capability, is_cuda


def permute_hyv4_indexer_weight(name, loaded_weight, config):
    if ".self_attn.indexer.wq_b." in name:
        group_count = config.index_n_heads
    elif any(
        key in name
        for key in (
            ".self_attn.indexer.wk.",
            ".self_attn.indexer.k_norm.",
        )
    ):
        group_count = 1
    else:
        return loaded_weight

    shape = loaded_weight.shape
    loaded_weight = loaded_weight.reshape(
        group_count,
        config.index_head_dim,
        *shape[1:],
    )
    rope_dim = config.qk_rope_head_dim
    return torch.cat(
        (
            loaded_weight[:, -rope_dim:],
            loaded_weight[:, :-rope_dim],
        ),
        dim=1,
    ).reshape(shape)


class HYV4HCPreLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_fn = ReplicatedLinear(
            config.hidden_size * config.hc_mult,
            2 * config.hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.hc_fn",
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(
            torch.empty(2 * config.hc_mult, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor):
        if hidden_states.is_cuda:
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_pre
            except ImportError:
                pass
            else:
                try:
                    return fused_hy4_ihc_pre(
                        hidden_states,
                        self.hc_fn.weight,
                        self.hc_scale,
                        self.hc_base,
                        self.magnitude,
                        self.rms_norm_eps,
                        self.hc_eps,
                    )
                except Exception:
                    logger.warning("fused_hy4_ihc_pre failed, using eager path")
        shape = hidden_states.shape
        flat = hidden_states.flatten(1).float()
        scale = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_norm_eps)
        gates = self.hc_fn(flat)[0] * scale
        pre = (
            torch.sigmoid(
                gates[..., : self.hc_mult] * self.hc_scale[0]
                + self.hc_base[: self.hc_mult]
            )
            + self.hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(
                gates[..., self.hc_mult :] * self.hc_scale[1]
                + self.hc_base[self.hc_mult :]
            )
            + self.hc_eps
        )
        reduced = torch.sum(pre.unsqueeze(-1) * hidden_states.reshape(shape), dim=1)
        return reduced.to(hidden_states.dtype), post


class HYV4HCLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_pre = HYV4HCPreLayer(config, f"{prefix}.hc_pre")

    def prepare_input(self, hidden_states: torch.Tensor):
        if hidden_states.ndim == 3:
            return hidden_states
        if hidden_states.ndim != 2:
            raise RuntimeError(
                f"iHC expects a 2D or 3D tensor, got {hidden_states.shape}"
            )
        if hidden_states.shape[-1] == self.hidden_size:
            return hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        if hidden_states.shape[-1] == self.hidden_size * self.hc_mult:
            return hidden_states.reshape(-1, self.hc_mult, self.hidden_size)
        raise RuntimeError(
            "iHC input width must equal hidden_size or hc_mult * hidden_size"
        )

    def pre(self, hidden_states: torch.Tensor):
        reduced, post = self.hc_pre(hidden_states)
        return reduced, post, hidden_states

    def post(self, output, residual, post):
        if output.is_cuda:
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_post
            except ImportError:
                pass
            else:
                try:
                    return fused_hy4_ihc_post(output, residual, post)
                except Exception:
                    logger.warning("fused_hy4_ihc_post failed, using eager path")
        result = post.float().unsqueeze(-1) * output.float().unsqueeze(1)
        return (result + residual.float()).to(output.dtype)


class HYV4HCHeadLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.config = config
        self.hc_head_fn = ReplicatedLinear(
            config.hc_mult * config.hidden_size,
            config.hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.hc_head_fn",
        )
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor):
        shape = hidden_states.shape
        flat = hidden_states.flatten(1).float()
        scale = torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        gates = self.hc_head_fn(flat)[0] * scale
        gates = (
            torch.sigmoid(gates * self.hc_head_scale + self.hc_head_base)
            + self.config.hc_eps
        )
        output = torch.sum(gates.unsqueeze(-1) * flat.reshape(shape), dim=1)
        return output.to(hidden_states.dtype)


class HYV4Attention(DeepseekV2AttentionMLA):
    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        is_nextn=False,
    ):
        rope_parameters = config.rope_parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_parameters["rope_theta"],
            rope_scaling=(
                None
                if rope_parameters.get("rope_type") == "default"
                else rope_parameters
            ),
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            reduce_results=True,
            layer_id=layer_id,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )
        parallel = get_parallel()
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=parallel.attn_tp_rank,
            tp_size=parallel.attn_tp_size,
            prefix=f"{prefix}.linear_gate",
        )
        self.learnable_sink_param = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32)
        )
        self.learnable_sink_param.weight_loader = self._sink_weight_loader
        self._gate_backend = self._resolve_gate_backend(
            getattr(config, "gating_type", None)
        )
        if prefix.endswith(".0.self_attn"):
            logger.info("HYV4 MLA output gate backend: %s", self._gate_backend)

    @staticmethod
    def _sink_weight_loader(param, loaded_weight):
        parallel = get_parallel()
        heads = loaded_weight.shape[0] // parallel.attn_tp_size
        start = parallel.attn_tp_rank * heads
        param.data.copy_(loaded_weight[start : start + heads].float())

    def _resolve_gate_backend(self, gating_type: str | None) -> str:
        """Select hpc, Triton, or eager for the MLA output gate."""
        weight = getattr(self.linear_gate, "weight", None)
        if self._hpc_gated_mla_supported(
            gating_type,
            None if weight is None else weight.dtype,
            None if weight is None else weight.shape[0],
        ):
            return "hpc"
        try:
            from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (  # noqa: F401
                sigmoid_gate_mul,
            )
        except ImportError:
            return "eager"
        return "triton"

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _hpc_gated_mla_supported(
        gating_type: str | None,
        weight_dtype: torch.dtype | None,
        weight_width: int | None,
    ) -> bool:
        """Check whether ``hpc.gated_mla_gemm`` supports this model shard."""
        try:
            import hpc
        except ImportError:
            return False

        if not hasattr(getattr(hpc, "gemm", None), "gated_mla_gemm"):
            logger.info(
                "HY4 gated MLA: the installed hpc build (%s) has no "
                "gemm.gated_mla_gemm; falling back.",
                getattr(hpc, "__version__", "unknown"),
            )
            return False

        major, minor = get_device_capability()
        if (major, minor) not in ((10, 0), (10, 3)):
            logger.warning(
                "HY4 gated MLA: hpc.gated_mla_gemm is built for sm100/sm103, "
                "got sm%d%d; falling back.",
                major,
                minor,
            )
            return False

        # Headwise gating broadcasts one scalar per head over v_head_dim; the
        # kernel only implements the elementwise product.
        if gating_type != "elementwise":
            return False

        if weight_dtype != torch.bfloat16:
            logger.warning(
                "HY4 gated MLA: gate weight dtype is %s, the kernel needs "
                "bfloat16 (keep linear_gate out of quantization).",
                weight_dtype,
            )
            return False
        if weight_width is None or weight_width % 256 != 0:
            logger.warning(
                "HY4 gated MLA: local gate width %s is not a multiple of 256.",
                weight_width,
            )
            return False
        return True

    def prepare_attention_output_gate(self, hidden_states):
        # The hpc kernel fuses the projection in, so hand it the activations and
        # let apply_attention_output_gate do the GEMM. The other two tiers want
        # the projected gate.
        if self._gate_backend == "hpc":
            return hidden_states
        return self.linear_gate(hidden_states)[0]

    def apply_attention_output_gate(self, attn_out, gate):
        """Apply the MLA output gate. ``gate`` is whatever prepare returned."""
        if self._gate_backend == "hpc":
            from hpc.gemm import gated_mla_gemm

            # linear_gate is column-parallel and unbiased, so the local weight
            # shard lines up with the local attn_out columns.
            return gated_mla_gemm(
                gate.contiguous(), self.linear_gate.weight, attn_out.contiguous()
            )
        if self._gate_backend == "triton":
            from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
                sigmoid_gate_mul,
            )

            return sigmoid_gate_mul(attn_out, gate)
        return attn_out * torch.sigmoid(gate)

    def dispatch_attn_forward_method(self, forward_batch):
        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)
        if hasattr(backend, "use_mha") and backend.use_mha is not False:
            backend.use_mha = False
        method = super().dispatch_attn_forward_method(forward_batch)
        if method != AttnForwardMethod.MLA:
            raise RuntimeError("HYV4 requires the sparse MLA attention path")
        return method


class HYV4DecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix="", alt_stream=None):
        super().__init__()
        self.self_attn = HYV4Attention(
            config, layer_id, quant_config, f"{prefix}.self_attn", alt_stream
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_id] == "dense":
            self.mlp = DeepseekV2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MoE(
                config,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
            )
            if hasattr(self.mlp, "shared_experts"):
                self.mlp.shared_experts.swiglu_limit = None
        self.hc_attn_layer = HYV4HCLayer(config, f"{prefix}.hc_attn_layer")
        self.hc_mlp_layer = HYV4HCLayer(config, f"{prefix}.hc_mlp_layer")

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
    ):
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(
                hidden_states, forward_batch, self.self_attn.prepare_qkv_latent
            )
        )
        try:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=prev_topk_indices,
            )
        finally:
            get_attn_tp_context().clear_attn_inputs()
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        hidden_states = self.hc_attn_layer.post(hidden_states, residual, post)

        hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
        hidden_states, post, residual = self.hc_mlp_layer.pre(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post)
        return hidden_states, topk_indices


class HYV4Model(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if get_pp_group().world_size != 1:
            raise ValueError("HYV4 pipeline parallelism is not supported")
        self.config = config
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
            **get_embedding_tp_kwargs(),
        )
        self.alt_stream = get_stream("alt") if is_cuda() else None
        self.layers = nn.ModuleList(
            [
                HYV4DecoderLayer(
                    config,
                    i,
                    quant_config,
                    f"{prefix}.layers.{i}",
                    self.alt_stream,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.hc_head = HYV4HCHeadLayer(config, f"{prefix}.hc_head")
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        zero_allocator = BumpAllocator(
            buffer_size=2 * len(self.layers),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_share = IndexTopKShareState(forward_batch, None)
        for layer in self.layers:
            hidden_states, topk_indices = layer(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                topk_share.topk_indices,
            )
            topk_share.update(topk_indices)
        topk_share.publish()
        return self.norm(self.hc_head(hidden_states))


class HYV4ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return "HYV4 applies the SwiGLU limit only to routed experts."

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.model = HYV4Model(config, quant_config, f"{prefix}.model")
        self.num_fused_shared_experts = max(
            (
                layer.mlp.num_fused_shared_experts
                for layer in self.model.layers
                if isinstance(layer.mlp, DeepseekV2MoE)
            ),
            default=0,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            # Keep checkpoint weights in bf16; LogitsProcessor emits fp32
            # logits when config.enable_lm_head_fp32 is set.
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
            use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        def mapped_weights():
            for name, loaded_weight in weights:
                if name.startswith("model.mtp_layers."):
                    continue
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                if name.endswith((".hc_fn", ".hc_head_fn")):
                    name += ".weight"
                if name.endswith(".weight_scale"):
                    name += "_inv"
                yield name, loaded_weight

        self.do_load_weights(mapped_weights())


EntryClass = [HYV4ForCausalLM]
