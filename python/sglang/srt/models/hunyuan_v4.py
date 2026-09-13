import functools
import logging
from typing import Iterable, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

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

logger = logging.getLogger(__name__)


def _hpc_ihc_available(op_name: str, hc_mult: int, hidden_size: int) -> bool:
    try:
        from sglang.kernels.ops.layernorm.hy4_ihc import _hpc_ihc_op
    except ImportError:
        return False
    return _hpc_ihc_op(op_name, hc_mult, hidden_size) is not None


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
        self._fused_ihc_pre_disabled = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        rms_weight: torch.Tensor | None = None,
        rms_eps: float = 0.0,
    ):
        if hidden_states.is_cuda and not self._fused_ihc_pre_disabled:
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
                        rms_weight,
                        rms_eps,
                    )
                except Exception:
                    self._fused_ihc_pre_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_pre failed, disabling fused path",
                        exc_info=True,
                    )
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
        reduced = reduced.to(hidden_states.dtype)
        if rms_weight is not None:
            reduced_float = reduced.float()
            reduced = (
                reduced_float
                * torch.rsqrt(
                    reduced_float.square().mean(dim=-1, keepdim=True) + rms_eps
                )
                * rms_weight.float()
            ).to(hidden_states.dtype)
        return reduced, post


class HYV4HCLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_pre = HYV4HCPreLayer(config, f"{prefix}.hc_pre")
        self._fused_ihc_post_disabled = False
        self._fused_ihc_post_pre_disabled = False

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

    def pre(self, hidden_states: torch.Tensor, norm: RMSNorm | None = None):
        fuse_norm = (
            norm is not None
            and hidden_states.is_cuda
            and _hpc_ihc_available("fuse_ihc_pre", self.hc_mult, self.hidden_size)
        )
        reduced, post = self.hc_pre(
            hidden_states,
            norm.weight if fuse_norm else None,
            norm.variance_epsilon if fuse_norm else 0.0,
        )
        if norm is not None and not fuse_norm:
            reduced = norm(reduced)
        return reduced, post, hidden_states

    def post(self, output, residual, post):
        if output.is_cuda and not self._fused_ihc_post_disabled:
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_post
            except ImportError:
                pass
            else:
                try:
                    return fused_hy4_ihc_post(output, residual, post)
                except Exception:
                    self._fused_ihc_post_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_post failed, disabling fused path",
                        exc_info=True,
                    )
        result = post.float().unsqueeze(-1) * output.float().unsqueeze(1)
        return (result + residual.float()).to(output.dtype)

    def post_pre(self, output, residual, post, next_layer, norm):
        if (
            output.is_cuda
            and not self._fused_ihc_post_pre_disabled
            and _hpc_ihc_available("fuse_ihc_post_pre", self.hc_mult, self.hidden_size)
        ):
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import (
                    fused_hy4_ihc_post_pre,
                )
            except ImportError:
                pass
            else:
                try:
                    next_residual, reduced, next_post = fused_hy4_ihc_post_pre(
                        output,
                        residual,
                        post,
                        next_layer.hc_pre.hc_fn.weight,
                        next_layer.hc_pre.hc_scale,
                        next_layer.hc_pre.hc_base,
                        next_layer.hc_pre.magnitude,
                        next_layer.hc_pre.rms_norm_eps,
                        next_layer.hc_pre.hc_eps,
                        norm.weight,
                        norm.variance_epsilon,
                    )
                    return reduced, next_post, next_residual
                except Exception:
                    self._fused_ihc_post_pre_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_post_pre failed, disabling fused path",
                        exc_info=True,
                    )
        next_residual = self.post(output, residual, post)
        next_residual = next_layer.prepare_input(next_residual)
        return next_layer.pre(next_residual, norm)


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
        self._fused_ihc_head_disabled = False

    def forward(self, hidden_states: torch.Tensor, norm: RMSNorm | None = None):
        if (
            hidden_states.is_cuda
            and not self._fused_ihc_head_disabled
            and _hpc_ihc_available(
                "fuse_ihc_head", self.config.hc_mult, self.config.hidden_size
            )
        ):
            try:
                from sglang.kernels.ops.layernorm.hy4_ihc import fused_hy4_ihc_head
            except ImportError:
                pass
            else:
                try:
                    return fused_hy4_ihc_head(
                        hidden_states,
                        self.hc_head_fn.weight,
                        self.hc_head_scale,
                        self.hc_head_base,
                        self.config.rms_norm_eps,
                        self.config.hc_eps,
                        None if norm is None else norm.weight,
                        0.0 if norm is None else norm.variance_epsilon,
                    )
                except Exception:
                    self._fused_ihc_head_disabled = True
                    logger.warning(
                        "fused_hy4_ihc_head failed, disabling fused path",
                        exc_info=True,
                    )
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
        output = output.to(hidden_states.dtype)
        return output if norm is None else norm(output)


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
        self.local_gate_width = self.num_local_heads * config.v_head_dim
        if self.linear_gate.output_size_per_partition != self.local_gate_width:
            raise ValueError(
                "HYV4 attention gate shard width must match the local attention "
                f"output width: {self.linear_gate.output_size_per_partition} != "
                f"{self.local_gate_width}"
            )
        self.learnable_sink_param = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32)
        )
        self.learnable_sink_param.weight_loader = self._sink_weight_loader
        self._gate_fallback_backend = self._resolve_gate_fallback_backend()
        self._gate_backend = self._resolve_gate_backend(
            getattr(config, "gating_type", None), self._gate_fallback_backend
        )
        if prefix.endswith(".0.self_attn"):
            logger.info("HYV4 MLA output gate backend: %s", self._gate_backend)

    @staticmethod
    def _sink_weight_loader(param, loaded_weight):
        parallel = get_parallel()
        heads = loaded_weight.shape[0] // parallel.attn_tp_size
        start = parallel.attn_tp_rank * heads
        param.data.copy_(loaded_weight[start : start + heads].float())

    @staticmethod
    def _resolve_gate_fallback_backend() -> str:
        try:
            from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (  # noqa: F401
                sigmoid_gate_mul,
            )
        except ImportError:
            return "eager"
        return "triton"

    def _resolve_gate_backend(
        self, gating_type: str | None, fallback_backend: str
    ) -> str:
        weight = getattr(self.linear_gate, "weight", None)
        if self._hpc_gated_mla_supported(
            gating_type,
            None if weight is None else weight.dtype,
            None if weight is None else tuple(weight.shape),
            self.local_gate_width,
            self.hidden_size,
        ):
            return "hpc"
        return fallback_backend

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _hpc_gated_mla_supported(
        gating_type: str | None,
        weight_dtype: torch.dtype | None,
        weight_shape: tuple[int, ...] | None,
        local_gate_width: int,
        hidden_size: int,
    ) -> bool:
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
        expected_shape = (local_gate_width, hidden_size)
        if weight_shape != expected_shape:
            logger.warning(
                "HYV4 gated MLA: local gate weight shape is %s, expected %s.",
                weight_shape,
                expected_shape,
            )
            return False
        if local_gate_width % 256 != 0:
            logger.warning(
                "HY4 gated MLA: local gate width %s is not a multiple of 256.",
                local_gate_width,
            )
            return False
        return True

    def _hpc_gated_mla_inputs_supported(self, hidden_states, attn_out) -> bool:
        weight = self.linear_gate.weight
        return (
            hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and attn_out.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and weight.ndim == 2
            and attn_out.ndim == 2
            and hidden_states.shape[0] == attn_out.shape[0]
            and hidden_states.shape[1] == weight.shape[1]
            and weight.shape[0] == self.local_gate_width
            and attn_out.shape[1] == self.local_gate_width
            and hidden_states.device == weight.device == attn_out.device
        )

    @staticmethod
    def _apply_attention_output_gate_fallback(attn_out, gate, backend):
        if gate.shape != attn_out.shape:
            raise ValueError(
                "HYV4 projected attention gate shape must match the local attention "
                f"output shape: {tuple(gate.shape)} != {tuple(attn_out.shape)}"
            )
        if backend == "triton" and attn_out.is_cuda:
            from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
                sigmoid_gate_mul,
            )

            return sigmoid_gate_mul(attn_out, gate)
        return attn_out * torch.sigmoid(gate)

    def prepare_attention_output_gate(self, hidden_states):
        # The hpc kernel fuses the projection in, so hand it the activations and
        # let apply_attention_output_gate do the GEMM. The other two tiers want
        # the projected gate.
        if self._gate_backend == "hpc":
            return hidden_states
        return self.linear_gate(hidden_states)[0]

    def apply_attention_output_gate(self, attn_out, gate):
        if self._gate_backend == "hpc" and self._hpc_gated_mla_inputs_supported(
            gate, attn_out
        ):
            from hpc.gemm import gated_mla_gemm

            # linear_gate is column-parallel and unbiased, so the local weight
            # shard lines up with the local attn_out columns.
            return gated_mla_gemm(
                gate.contiguous(), self.linear_gate.weight, attn_out.contiguous()
            )
        if self._gate_backend == "hpc":
            gate = self.linear_gate(gate)[0]
            backend = self._gate_fallback_backend
        else:
            backend = self._gate_backend
        return self._apply_attention_output_gate_fallback(attn_out, gate, backend)

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
        hidden_states, post, residual = self.hc_attn_layer.pre(
            hidden_states, self.input_layernorm
        )
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
        hidden_states, post, residual = self.hc_attn_layer.post_pre(
            hidden_states,
            residual,
            post,
            self.hc_mlp_layer,
            self.post_attention_layernorm,
        )
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
        return self.hc_head(hidden_states, self.norm)


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
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
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
