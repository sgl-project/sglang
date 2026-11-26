import contextlib
import copy
import functools
import logging
from typing import Iterable, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.hy_v4_dfly import DFLY_CHECKPOINT_PREFIX
from sglang.srt.configs.model_config import dsa_layer_skips_topk, get_dsa_index_topk
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.layer_boundary import (
    IHCState,
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_pp_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
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


def _hpc_ihc_available(
    op_name: str, hc_mult: int, hidden_size: int, num_tokens: int
) -> bool:
    try:
        from sglang.kernels.ops.layernorm.hy4_ihc import _selected_hpc_ihc_op
    except ImportError:
        return False
    return _selected_hpc_ihc_op(op_name, hc_mult, hidden_size, num_tokens) is not None


def _validate_ihc_policy():
    try:
        from sglang.kernels.ops.layernorm.hy4_ihc import _ihc_policy
    except ImportError:
        return
    _ihc_policy()


def _ihc_forward_mode(forward_batch):
    try:
        from sglang.kernels.ops.layernorm.hy4_ihc import hy4_ihc_forward_mode
    except ImportError:
        return contextlib.nullcontext()
    return hy4_ihc_forward_mode(forward_batch.forward_mode.is_decode_or_idle())


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
    rows = group_count * config.index_head_dim
    if shape[0] != rows:
        # Block scales: the permutation stays inside each head, so it leaves a
        # scale row unchanged when that row's block covers whole heads.
        if rows % shape[0] or (rows // shape[0]) % config.index_head_dim:
            raise ValueError(
                f"{name}: {shape[0]} scale rows for {rows} indexer rows do not "
                f"align with index_head_dim={config.index_head_dim}"
            )
        return loaded_weight
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


def hyv4_quant_config(quant_config):
    # HYV4 compressed-tensors checkpoints select quantized modules with regex
    # targets and store every unselected module unquantized.
    if not isinstance(quant_config, CompressedTensorsConfig) or not all(
        target.startswith("re:") for target in quant_config.target_scheme_map
    ):
        return quant_config
    quant_config = copy.copy(quant_config)
    quant_config.ignore_unmatched_layers = True
    return quant_config


def check_hyv4_scale_has_target(name, quant_config, layer_prefix):
    # Unselected modules load unquantized only while the checkpoint agrees: a
    # stored weight scale must belong to a module the config selects.
    if not (
        isinstance(quant_config, CompressedTensorsConfig)
        and quant_config.ignore_unmatched_layers
        and name.endswith(".weight_scale")
    ):
        return
    layer_name = f"{layer_prefix}.{name.removesuffix('.weight_scale')}"
    if should_ignore_layer(
        layer_name,
        ignore=quant_config.ignore,
        fused_mapping=quant_config.packed_modules_mapping,
    ) or not check_equal_or_regex_match(
        layer_name=layer_name, targets=quant_config.target_scheme_map
    ):
        raise ValueError(
            f"HYV4 checkpoint stores {name}, but the compressed-tensors config "
            f"leaves {layer_name} unquantized"
        )


def normalize_hyv4_weight_scale(name, loaded_weight, quant_config):
    # compressed-tensors schemes register the checkpoint's own `weight_scale`.
    if isinstance(quant_config, CompressedTensorsConfig):
        return name, loaded_weight
    for alias in (".weight_scale", ".scale"):
        if name.endswith(alias):
            name = name.removesuffix(alias) + ".weight_scale_inv"
            break
    if (
        name.endswith(".weight_scale_inv")
        and loaded_weight.dtype == torch.uint8
        and isinstance(quant_config, Fp8Config)
        and not quant_config.use_mxfp8
    ):
        # UE8M0 exponent bytes. MXFP8 scale params keep the raw bytes; the FP8
        # block and MXFP4 expert params take the value 2**(byte - 127).
        loaded_weight = loaded_weight.view(torch.float8_e8m0fnu).float()
    return name, loaded_weight


def check_hyv4_indexer_pairs(unresolved_indexer_pairs: Set[str]) -> None:
    if unresolved_indexer_pairs:
        raise ValueError(
            "Incomplete FP8 indexer weight/scale pairs in the HYV4 checkpoint: "
            + ", ".join(sorted(unresolved_indexer_pairs))
        )


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
            and _hpc_ihc_available(
                "fuse_ihc_pre", self.hc_mult, self.hidden_size, hidden_states.shape[0]
            )
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
            and _hpc_ihc_available(
                "fuse_ihc_post_pre", self.hc_mult, self.hidden_size, output.shape[0]
            )
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
                "fuse_ihc_head",
                self.config.hc_mult,
                self.config.hidden_size,
                hidden_states.shape[0],
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


class HYV4MRotaryEmbedding(nn.Module):
    """Sectioned T/H/W rotary for [3, num_tokens] positions. 1-D positions
    keep the text rotary, whose cache the 1-D fused kernels read."""

    def __init__(self, rope, mrope):
        super().__init__()
        self.rope = rope
        self.mrope = mrope

    @property
    def cos_sin_cache(self):
        return self.rope.cos_sin_cache

    @property
    def is_neox_style(self):
        return self.rope.is_neox_style

    def forward(self, positions, query, key):
        if positions.ndim == 1:
            return self.rope(positions, query, key)
        # The mrope kernels take token-major [num_tokens, heads * rope_dim].
        rotated_query, rotated_key = self.mrope(
            positions, query.flatten(1), key.flatten(1)
        )
        return rotated_query.view_as(query), rotated_key.view_as(key)


def _build_hyv4_mrope(config, rope_parameters, is_neox_style):
    if rope_parameters.get("rope_type") != "default":
        raise ValueError(
            "HYV4 mRoPE supports only rope_type='default'; got "
            f"{rope_parameters.get('rope_type')!r}"
        )
    mrope_section = list(rope_parameters["mrope_section"])
    if len(mrope_section) != 3 or sum(mrope_section) != config.qk_rope_head_dim // 2:
        raise ValueError(
            "HYV4 mrope_section must be three sections summing to "
            f"qk_rope_head_dim // 2 = {config.qk_rope_head_dim // 2}; "
            f"got {mrope_section!r}"
        )
    return get_rope(
        config.qk_rope_head_dim,
        rotary_dim=config.qk_rope_head_dim,
        max_position=config.max_position_embeddings,
        base=rope_parameters["rope_theta"],
        is_neox_style=is_neox_style,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": mrope_section,
            "mrope_interleaved": rope_parameters.get("mrope_interleaved", False),
        },
    )


def hyv4_mrope_rows_are_equal(forward_mode):
    # Decode and spec batches build every mRoPE row as pos + delta, where sectioned
    # mRoPE equals 1-D rope; row 0 keeps the 1-D fused rope and indexer kernels.
    return (
        forward_mode.is_decode()
        or forward_mode.is_target_verify()
        or forward_mode.is_draft_extend_v2()
    )


class HYV4Attention(DeepseekV2AttentionMLA):
    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        is_nextn=False,
        *,
        use_mrope=False,
        attn_post_proj_norm=False,
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
            # The FFN input boundary completes this stage's sum.
            reduce_results=False,
            layer_id=layer_id,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )
        if use_mrope:
            self.rotary_emb = HYV4MRotaryEmbedding(
                self.rotary_emb,
                _build_hyv4_mrope(
                    config, rope_parameters, self.rotary_emb.is_neox_style
                ),
            )
            if self.indexer is not None:
                self.indexer.rotary_emb = HYV4MRotaryEmbedding(
                    self.indexer.rotary_emb,
                    _build_hyv4_mrope(
                        config, rope_parameters, self.indexer.is_neox_style
                    ),
                )
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            bias=False,
            quant_config=quant_config,
            parallel_group="attn_tp",
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
        # A DFly worker sets this on its target index layers.
        self.dfly_topk_capture = None
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if attn_post_proj_norm
            else None
        )

    @staticmethod
    def _sink_weight_loader(param, loaded_weight):
        parallel = get_parallel()
        heads = loaded_weight.shape[0] // parallel.attn_tp_size
        start = parallel.attn_tp_rank * heads
        param.data.copy_(loaded_weight[start : start + heads].float())

    @staticmethod
    def _resolve_gate_fallback_backend() -> str:
        try:
            from sglang.kernels.ops.elementwise.elementwise import (  # noqa: F401
                fused_sigmoid_mul,
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
            from sglang.kernels.ops.elementwise.elementwise import (
                fused_sigmoid_mul,
            )

            return fused_sigmoid_mul(attn_out, gate)
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

    def _fuse_rope_for_trtllm_mla(self, forward_batch):
        # The fused trtllm rope reads forward_batch.positions, which lacks the mRoPE
        # delta of this model's positions. Graph buffers always carry mrope_positions.
        if isinstance(self.rotary_emb, HYV4MRotaryEmbedding):
            return False
        return super()._fuse_rope_for_trtllm_mla(forward_batch)

    def forward_absorb_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        *args,
    ):
        # Output keeps top-k only for a sharing successor; record the producer's rows.
        if self.dfly_topk_capture is not None:
            self.dfly_topk_capture.record(self.layer_id, topk_indices)
        return super().forward_absorb_core(
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            *args,
        )

    def dispatch_attn_forward_method(self, forward_batch):
        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)
        if hasattr(backend, "use_mha") and backend.use_mha is not False:
            backend.use_mha = False
        method = super().dispatch_attn_forward_method(forward_batch)
        if method != AttnForwardMethod.MLA:
            raise RuntimeError("HYV4 requires the sparse MLA attention path")
        return method


class _HeadNorm:
    """The layer stack's terminal read: the learned head mixes the streams down
    and fuses the final norm into the same kernel."""

    def __init__(self, hc_head, norm):
        self.hc_head = hc_head
        self.norm = norm

    def __call__(self, hidden_states):
        return self.hc_head(hidden_states, self.norm)


def _build_stages(
    hc_attn_layer,
    hc_mlp_layer,
    input_layernorm,
    post_attention_layernorm,
    *,
    qkv_latent_func,
    layer_id,
    config,
    post_proj_norm,
):
    """The attention and FFN stage boundaries of one iHC decoder layer."""

    def post_pre(hidden_states, residual, post_gate, out_norm):
        return hc_attn_layer.post_pre(
            hidden_states, residual, post_gate, hc_mlp_layer, out_norm
        )

    residual = IHCState(
        expand=hc_attn_layer.prepare_input,
        attn_pre=hc_attn_layer.pre,
        ffn_pre=hc_mlp_layer.pre,
        attn_post=hc_attn_layer.post,
        ffn_post=hc_mlp_layer.post,
        post_pre=post_pre,
    ).residual_ops()

    attn, ffn = _stage_facts(
        config, layer_id, residual=residual, post_proj_norm=post_proj_norm
    )
    return append_stages(
        (attn, input_layernorm, {"qkv_latent_func": qkv_latent_func}),
        (ffn, post_attention_layernorm),
    )


def _stage_facts(config, layer_id, *, residual=None, post_proj_norm=None):
    """The stages the layer at ``layer_id`` declares, shared by the layer and
    by make_layers for layers built on another pipeline rank."""
    if residual is None:
        residual = IHCState.facts()
    layer_types = config.mlp_layer_types
    next_id = layer_id + 1
    return (
        declare_attn(
            read=residual.attn_readout,
            update=residual.attn_update,
            output_transform=(
                OutputTransform(post_proj_norm)
                if getattr(config, "attn_post_proj_norm", False)
                else None
            ),
        ),
        declare_ffn(
            sparse=layer_types[layer_id] != "dense",
            next_layer_sparse=(
                next_id < len(layer_types) and layer_types[next_id] != "dense"
            ),
            read=residual.ffn_readout,
            update=residual.ffn_update,
        ),
    )


class HYV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        *,
        use_mrope=False,
    ):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            layer_id,
            quant_config,
            f"{prefix}.self_attn",
            alt_stream,
            use_mrope=use_mrope,
            attn_post_proj_norm=getattr(config, "attn_post_proj_norm", False),
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
                reduce_results=False,
                allow_fused_down=False,
            )
        else:
            self.mlp = DeepseekV2MoE(
                config,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
                reduce_results=False,
                # The loader keeps compressed-tensors' per-expert weight_packed names.
                serves_fused_mxfp4=False,
            )
            if self.mlp.shared_experts is not None:
                self.mlp.shared_experts.swiglu_limit = None
        self.hc_attn_layer = HYV4HCLayer(config, f"{prefix}.hc_attn_layer")
        self.hc_mlp_layer = HYV4HCLayer(config, f"{prefix}.hc_mlp_layer")
        self.attn_boundary, self.ffn_boundary = _build_stages(
            self.hc_attn_layer,
            self.hc_mlp_layer,
            self.input_layernorm,
            self.post_attention_layernorm,
            qkv_latent_func=self.self_attn.prepare_qkv_latent,
            layer_id=layer_id,
            config=config,
            post_proj_norm=self.self_attn.post_proj_norm,
        )

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
    ):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
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
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        if isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)
        return hidden_states, topk_indices


class HYV4Model(nn.Module):
    def __init__(self, config, quant_config=None, prefix="", *, use_mrope=False):
        super().__init__()
        # Fail at startup: no hpc, or an empty batch, would otherwise skip the check.
        _validate_ihc_policy()
        self.config = config
        self.pp_group = get_parallel().pp_group
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
                **get_embedding_tp_kwargs(),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.alt_stream = get_stream("alt") if is_cuda() else None
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: HYV4DecoderLayer(
                config,
                idx,
                quant_config,
                prefix,
                self.alt_stream,
                use_mrope=use_mrope,
            ),
            prefix=add_prefix("layers", prefix),
            stage_facts=lambda idx: _stage_facts(config, idx),
        )
        if self.pp_group.is_last_rank:
            self.hc_head = HYV4HCHeadLayer(config, f"{prefix}.hc_head")
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            # The head mixes the streams down and fuses the final norm into one kernel.
            self._head_norm = _HeadNorm(self.hc_head, self.norm)
        else:
            self.hc_head = PPMissingLayer()
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual_batch.start(forward_batch)
            topk_share = IndexTopKShareState(forward_batch, None)
        else:
            hidden_states = self._from_previous_stage(pp_proxy_tensors, forward_batch)
            topk_share = IndexTopKShareState(
                forward_batch, pp_proxy_tensors.tensors.get("topk_indices")
            )
        zero_allocator = BumpAllocator(
            buffer_size=2 * (self.end_layer - self.start_layer),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        with _ihc_forward_mode(forward_batch):
            for layer in self.layers[self.start_layer : self.end_layer]:
                hidden_states, topk_indices = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    zero_allocator,
                    topk_share.topk_indices,
                )
                topk_share.update(topk_indices)
            if not self.pp_group.is_last_rank:
                return self._to_next_stage(hidden_states, forward_batch, topk_share)
            topk_share.publish()
            return residual_batch.final_norm(
                hidden_states, forward_batch, self._head_norm
            )

    def _from_previous_stage(self, pp_proxy_tensors, forward_batch):
        first_layer = self.layers[self.start_layer]
        tensors = dict(pp_proxy_tensors.tensors)
        tensors["hidden_states"] = first_layer.hc_attn_layer.prepare_input(
            tensors["hidden_states"]
        )
        mm_input_embeds = tensors.pop("mm_input_embeds", None)
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        return first_layer.attn_boundary.from_pp(PPProxyTensors(tensors), forward_batch)

    def _to_next_stage(self, hidden_states, forward_batch, topk_share):
        proxy = residual_batch.to_pp(hidden_states, forward_batch)
        # The receiving buffers are 2-D at hc_mult * hidden_size.
        streams = proxy.tensors["hidden_states"]
        proxy.tensors["hidden_states"] = streams.flatten(1)
        if self.end_layer < self.config.num_hidden_layers and dsa_layer_skips_topk(
            self.config, self.end_layer
        ):
            topk_indices = topk_share.topk_indices
            if topk_indices is None:
                topk_indices = streams.new_empty(
                    (0, get_dsa_index_topk(self.config)), dtype=torch.int32
                )
            proxy.tensors["topk_indices"] = topk_indices
        # Only the first stage embeds media; the MTP draft on the last stage
        # reads the target's merged embeddings in its prefill draft extend.
        spec_algorithm = forward_batch.spec_algorithm
        if (
            forward_batch.mm_input_embeds is not None
            and spec_algorithm is not None
            and spec_algorithm.is_eagle()
        ):
            proxy.tensors["mm_input_embeds"] = forward_batch.mm_input_embeds
        return proxy


HYV4_PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
}
# Draft heads bundled in the target checkpoint; their own draft models load them.
HYV4_DRAFT_CHECKPOINT_PREFIXES = ("model.mtp_layers.", DFLY_CHECKPOINT_PREFIX)


class HYV4ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = HYV4_PACKED_MODULES_MAPPING

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        return "HYV4 applies the SwiGLU limit only to routed experts."

    def __init__(self, config, quant_config=None, prefix="", *, use_mrope=False):
        super().__init__()
        quant_config = hyv4_quant_config(quant_config)
        self.config = config
        self.quant_config = quant_config
        self.layer_name_prefix = prefix
        self.pp_group = get_parallel().pp_group
        self.model = HYV4Model(
            config, quant_config, f"{prefix}.model", use_mrope=use_mrope
        )
        self.num_fused_shared_experts = max(
            (
                layer.mlp.num_fused_shared_experts
                for layer in self.model.layers[self.start_layer : self.end_layer]
                if isinstance(layer.mlp, DeepseekV2MoE)
            ),
            default=0,
        )
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                # Keep checkpoint weights in bf16; LogitsProcessor emits fp32
                # logits when config.enable_lm_head_fp32 is set.
                quant_config=quant_config,
                prefix=f"{prefix}.lm_head",
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        # A DFly worker sets this on the top-level target model only.
        self.dfly_execution = None

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ):
        if (
            self.dfly_execution is not None
            and forward_batch.forward_mode.is_target_verify()
        ):
            self.dfly_execution.draft_into_verify(
                input_ids=input_ids, forward_batch=forward_batch
            )
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        head = self.lm_head.weight if self.pp_group.is_last_rank else None
        return embed, head

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        norm_names = {
            f"model.layers.{i}.self_attn.post_proj_norm.weight"
            for i in range(self.config.num_hidden_layers)
        }
        seen_norms = set()

        def mapped_weights():
            for name, loaded_weight in weights:
                if name.startswith(HYV4_DRAFT_CHECKPOINT_PREFIXES):
                    continue
                if ".post_proj_norm." in name:
                    if (
                        not getattr(self.config, "attn_post_proj_norm", False)
                        or name not in norm_names
                    ):
                        raise ValueError(
                            f"Unexpected HYV4 post-projection norm: {name}"
                        )
                    if name in seen_norms:
                        raise ValueError(f"Duplicate HYV4 post-projection norm: {name}")
                    if tuple(loaded_weight.shape) != (self.config.hidden_size,):
                        raise ValueError(
                            f"Invalid HYV4 post-projection norm shape: {name}"
                        )
                    seen_norms.add(name)
                if not self.pp_group.is_last_rank and name.startswith(
                    ("model.hc_head.", "lm_head.")
                ):
                    continue
                check_hyv4_scale_has_target(
                    name, self.quant_config, self.layer_name_prefix
                )
                name, loaded_weight = normalize_hyv4_weight_scale(
                    name, loaded_weight, self.quant_config
                )
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                if name.endswith((".hc_fn", ".hc_head_fn")):
                    name += ".weight"
                yield name, loaded_weight
            if getattr(self.config, "attn_post_proj_norm", False):
                missing = norm_names - seen_norms
                if missing:
                    raise ValueError(
                        f"HYV4 checkpoint is missing post-projection norms: {sorted(missing)}"
                    )

        self.do_load_weights(mapped_weights())

    def _check_loaded_weights(
        self, unloaded_params: Set[str], unresolved_indexer_pairs: Set[str]
    ) -> None:
        check_hyv4_indexer_pairs(unresolved_indexer_pairs)
        missing = {n for n in unloaded_params if ".post_proj_norm." in n}
        if missing:
            raise ValueError(
                f"HYV4 post-projection norms left unloaded: {sorted(missing)}"
            )


EntryClass = [HYV4ForCausalLM]
