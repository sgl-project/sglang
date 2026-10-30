# Copyright 2026 SGLang Team
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

import atexit
import logging
import math
from functools import partial
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.ops.diffusion.rope.qknorm_rope_jit import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
)
from sglang.kernels.ops.elementwise.beam_fused import BeamAttentionGateOp
from sglang.kernels.ops.elementwise.row_scale import apply_log_scaling_tau
from sglang.kernels.ops.gemm.router_gemv import (
    router_gemm_bf16x3_cublas,
    router_gemv,
    router_gemv_supported,
)
from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
from sglang.kernels.ops.layernorm.beam_fused import BeamSandwichResidualRMSNormOp
from sglang.kernels.spec import KernelBackend
from sglang.srt.configs.beam import BeamConfig
from sglang.srt.distributed import (
    moe_expert_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import (
    dp_gather_replicate,
    dp_scatter,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    post_experts_output_is_complete,
    should_skip_mlp_all_reduce,
    should_skip_post_experts_all_reduce,
    should_use_dp_reduce_scatterv,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_forward, get_lora, get_parallel
from sglang.srt.utils import (
    add_prefix,
    is_cuda,
    make_layers,
    print_warning_once,
    set_weight_attrs,
)

_is_cuda = is_cuda()


def compute_ssmax_scale(positions: torch.Tensor) -> torch.Tensor:
    return torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))


def apply_ssmax(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale Q rows by an FP32 scale, rounding only after multiplication."""
    if (
        q.is_cuda
        and not torch.compiler.is_compiling()
        and get_fused_op_backend() != KernelBackend.TORCH
    ):
        # One launch with the same FP32 product and single rounding.
        return apply_log_scaling_tau(q, scale)
    return (q.float() * scale.unsqueeze(-1)).to(q.dtype)


def depth_scaled_residual(
    branch: torch.Tensor, residual: torch.Tensor, scales: Tuple[float, float]
) -> torch.Tensor:
    branch_scale, residual_scale = scales
    residual = residual.to(branch.dtype)
    # Round both scaled operands before adding in the activation dtype.
    branch = branch * branch_scale
    residual = residual * residual_scale
    return branch + residual


def _can_fuse_qk_norm_rope(head_dim: int) -> bool:
    cuda_graph_config = get_exec().graph.cuda_graph_config
    return (
        _is_cuda
        and not get_exec().deterministic.enable_deterministic_inference
        and (
            cuda_graph_config is None
            or cuda_graph_config.prefill.tc_compiler != "inductor"
        )
        and can_use_fused_inplace_qknorm_rope(
            head_dim=head_dim,
            rope_dim=head_dim,
            is_neox=True,
            dtype=torch.get_default_dtype(),
            round_norm_before_rope=True,
        )
    )


class BeamRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config: BeamConfig):
        super().__init__(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_parameters["sliding_attention"]["rope_theta"],
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

    def _torch_compile_forward(self, num_tokens: int):
        # Preserve FP32 CUDA rotation instead of the eager BF16 intermediates.
        return None


class BeamSiluAndMul(SiluAndMul):
    def _torch_compile_forward(self, num_tokens: int):
        # Keep the fused activation/product and its single output cast.
        return None


class BeamUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def _torch_compile_forward(self, num_tokens: int):
        # The torch-native fallback rounds route weights and omits the runner
        # scale. Keep the CUDA runner, including during single-token compile.
        return None


# Token counts per forward where FlashInfer's MNNVL all-reduce beats the best
# CustomAllReduceV2 algorithm with the same norm, measured on GB300 TP4 between
# real-shape GEMMs. Other GPUs and TP sizes are unmeasured.
_FLASHINFER_AR_TOKEN_BAND = {((10, 3), 4): (96, 640)}

logger = logging.getLogger(__name__)


def _flashinfer_all_reduce(x: torch.Tensor, workspace) -> torch.Tensor:
    import flashinfer.comm as fi_comm

    x = x.contiguous()
    out = torch.empty_like(x)
    # With PDL the MNNVL kernel releases the next launch before its output store;
    # the PDL norm's gdc_wait orders its read of `out`.
    fi_comm.allreduce_fusion(
        input=x,
        workspace=workspace,
        pattern=fi_comm.AllReduceFusionPattern.kAllReduce,
        launch_with_pdl=True,
        output=out,
    )
    return out


def _mlp_owes_single_tp_sum() -> bool:
    """Whether the MLP output owes one TP-wide sum that no combine already ran."""
    parallel = get_parallel()
    one_group = (
        parallel.moe_ep_size == 1 and parallel.moe_tp_size == parallel.tp_size
    ) or (parallel.moe_tp_size == 1 and parallel.moe_ep_size == parallel.tp_size)
    return one_group and not (
        post_experts_output_is_complete(is_tp_path=True)
        or post_experts_output_is_complete(is_tp_path=False)
    )


def _beam_owns_tp_boundaries() -> bool:
    """Whether Beam may replace the TP all-reduces of its layer boundaries."""
    parallel = get_parallel()
    execution = get_exec()
    return not (
        parallel.attn_tp_size != parallel.tp_size
        or parallel.attn_cp_size != 1
        or execution.graph.enable_torch_compile
        or execution.deterministic.enable_deterministic_inference
        or execution.comm.disable_custom_all_reduce
        or get_fused_op_backend() is not None
        or torch.version.hip is not None
    )


class BeamMLP(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = BeamSiluAndMul()

    def forward(self, x: torch.Tensor, skip_all_reduce: bool = False) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, skip_all_reduce=skip_all_reduce)
        return x


class BeamMoE(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.tp_size = get_parallel().moe_tp_size
        self.ep_size = get_parallel().moe_ep_size
        backend = get_moe_runner_backend()
        quant_scheme = get_beam_quantization(quant_config)
        supported_runners = get_beam_supported_moe_runners(quant_scheme)
        if get_moe_a2a_backend().is_flashinfer_megamoe():
            # MegaMOE dispatches each DP rank's own tokens and combines them back
            # complete, so the post-experts all-reduces below are skipped.
            # Only the ModelOpt NVFP4 MoE method has MegaMOE weight hooks.
            if (
                quant_scheme != "nvfp4"
                or not isinstance(quant_config, ModelOptFp4Config)
                or not backend.is_flashinfer_megamoe()
            ):
                raise ValueError(
                    "Beam supports the flashinfer_megamoe all-to-all only for "
                    "ModelOpt NVFP4 with the flashinfer_megamoe runner"
                )
        elif (
            backend.value not in supported_runners
            or not get_moe_a2a_backend().is_none()
        ):
            raise ValueError(
                "Beam supports the auto or triton MoE runner for BF16/FP8 "
                "and flashinfer_trtllm for NVFP4, with no MoE all-to-all; "
                "NVFP4 also supports the flashinfer_megamoe all-to-all"
            )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32), requires_grad=False
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            renormalize=True,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            output_format=(
                TopKOutputFormat.BYPASSED
                if backend.is_flashinfer_trtllm()
                else TopKOutputFormat.STANDARD
            ),
        )
        bf16_experts = quant_config is None or (
            quant_scheme == "fp8"
            and all(
                should_ignore_layer(
                    add_prefix(f"experts.0.{proj}", prefix),
                    ignore=quant_config.ignore,
                    fused_mapping=quant_config.packed_modules_mapping,
                )
                for proj in ("gate_proj", "up_proj", "down_proj")
            )
        )
        # Keep normalized routing coefficients in FP32. The runner applies the
        # routed scale during combination, after weighting each expert output.
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            inplace=False,
            routed_scaling_factor=config.routed_scaling_factor,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            quant_config=quant_config,
            quant_method=BeamUnquantizedFusedMoEMethod() if bf16_experts else None,
            prefix=add_prefix("experts", prefix),
        )
        # Static eligibility for the opt-in single-token GEMV on this SM103 shard;
        # forward also needs one contiguous BF16 token and interleaved w13.
        self.use_decode_gemv = (
            envs.SGLANG_OPT_BEAM_DECODE_GEMV.get()
            and _is_cuda
            and self.tp_size == 4
            and self.ep_size == 1
            and self.experts.w13_weight.dtype == torch.bfloat16
            and tuple(self.experts.w13_weight.shape) == (256, 1280, 5120)
            and torch.cuda.get_device_capability() == (10, 3)
        )
        if envs.SGLANG_OPT_BEAM_DECODE_GEMV.get() and not self.use_decode_gemv:
            print_warning_once(
                "SGLANG_OPT_BEAM_DECODE_GEMV is unused: it needs BF16 experts "
                "with MoE TP4/EP1 on SM103."
            )
        # Some expert methods consume routing weights with scale already applied.
        if self.experts.should_fuse_routed_scaling_factor_in_topk:
            self.topk.topk_config.routed_scaling_factor = config.routed_scaling_factor
            self.topk.topk_config.apply_routed_scaling_factor_on_output = True
        # Offloading replaces parameters, but TopKConfig keeps an alias.
        self.refresh_correction_bias = get_exec().offload.cpu_offload_gb > 0
        # Preserve the normal linear path for offloading and strict
        # deterministic inference. Never cache or downcast router weights.
        self.use_fast_router = (
            _is_cuda
            and not self.refresh_correction_bias
            and not get_exec().deterministic.enable_deterministic_inference
            and get_fused_op_backend() != KernelBackend.TORCH
        )
        # Tiny GEMM up to 16 tokens.
        self.use_fp32_router = (
            self.use_fast_router
            # Full-model validation covers the BF16 MoE runner.
            and quant_config is None
            and config.hidden_size == 5120
            and config.num_experts == 256
            and torch.cuda.get_device_capability()[0] in (9, 10)
        )

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # All fast paths read BF16 activations and the live FP32 gate; tiny GEMM and
        # the GEMV are IEEE FP32, the BF16x3 GEMM accumulates on tensor cores.
        if (
            not self.use_fast_router
            or torch.compiler.is_compiling()
            or hidden_states.dtype != torch.bfloat16
        ):
            router_logits, _ = self.gate(hidden_states.float())
            return router_logits
        if (
            self.use_fp32_router
            and hidden_states.is_contiguous()
            and hidden_states.data_ptr() % 16 == 0
            # Larger batches win in isolation but did not improve the
            # full BF16 decode step.
            and hidden_states.shape[0] <= 16
        ):
            return tiny_gemm_bf16(
                hidden_states, self.gate.weight, out_dtype=torch.float32
            )
        if router_gemv_supported(hidden_states, self.gate.weight):
            return router_gemv(hidden_states, self.gate.weight)
        return router_gemm_bf16x3_cublas(hidden_states, self.gate.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        defer_sum: bool = False,
        skip_all_reduce: bool = False,
    ) -> torch.Tensor:
        """Return combined outputs, or local expert outputs for Beam's fusion.

        `skip_all_reduce=True` returns this rank's partial sum and leaves the
        post-experts all-reduce to the caller.

        `defer_sum=True` is restricted by the caller to the unquantized Triton
        TP path. It returns contiguous BF16 [tokens, top_k, hidden_size] outputs
        with routing weights already applied in GEMM2. The consumer must sum
        experts in FP32, apply `routed_scaling_factor`, round to BF16, and perform
        the TP all-reduce before normalization. No expert sum, routed scale or
        collective is applied by this deferred path.
        """
        if hidden_states.shape[0] > 0:
            router_logits = self._router_logits(hidden_states)
            if self.refresh_correction_bias:
                self.topk.topk_config.correction_bias = (
                    self.gate.e_score_correction_bias
                )
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device, layer_id=self.layer_id
            )
        if defer_sum:
            if (
                self.use_decode_gemv
                and hidden_states.shape[0] == 1
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.is_contiguous()
                # Interleaving is decided after weight loading.
                and self.experts.quant_method.w13_swiglu_interleaved
            ):
                from sglang.kernels.ops.moe.beam_decode_gemv import beam_decode_gemv

                return beam_decode_gemv(
                    hidden_states,
                    self.experts.w13_weight,
                    self.experts.w2_weight,
                    topk_output.topk_weights,
                    topk_output.topk_ids,
                )
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                fused_experts_impl,
            )

            return fused_experts_impl(
                hidden_states,
                self.experts.w13_weight,
                self.experts.w2_weight,
                topk_output.topk_weights,
                topk_output.topk_ids,
                inplace=False,
                filter_expert=False,
                defer_sum=True,
                fuse_swiglu_interleaved=self.experts.quant_method.w13_swiglu_interleaved,
            )
        output = self.experts(hidden_states, topk_output)
        if skip_all_reduce:
            return output
        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False
        ):
            output = moe_expert_parallel_all_reduce(output)
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True
        ):
            output = moe_tensor_model_parallel_all_reduce(output)
        return output


class BeamAttention(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        rotary_emb: BeamRotaryEmbedding | None = None,
    ):
        super().__init__()
        self.tp_rank = get_parallel().attn_tp_rank
        self.tp_size = get_parallel().attn_tp_size
        if config.num_attention_heads % self.tp_size:
            raise ValueError("Beam query heads must be divisible by attention TP size")
        if config.num_key_value_heads >= self.tp_size:
            if config.num_key_value_heads % self.tp_size:
                raise ValueError("Beam KV heads must be divisible by attention TP size")
        elif self.tp_size % config.num_key_value_heads:
            raise ValueError("Beam attention TP size must be divisible by KV heads")
        self.num_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Sliding layers use RoPE; full-attention layers use NoPE with SSMax.
        self.apply_rope = config.layer_types[layer_id] == "sliding_attention"
        self.sliding_window = config.sliding_window - 1 if self.apply_rope else -1
        # MergedColumnParallelLinear shards each projection independently, but
        # does not replicate KV heads. Keep the existing quantized/replicated
        # paths until their loading and backend contracts are validated.
        self.use_packed_projection = (
            quant_config is None and config.num_key_value_heads >= self.tp_size
        )
        if self.use_packed_projection:
            self.qkv_gate_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [
                    config.num_attention_heads * self.head_dim,
                    config.num_key_value_heads * self.head_dim,
                    config.num_key_value_heads * self.head_dim,
                    config.num_attention_heads * self.head_dim,
                ],
                bias=False,
                parallel_group="attn_tp",
                prefix=add_prefix("qkv_gate_proj", prefix),
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                config.hidden_size,
                self.head_dim,
                config.num_attention_heads,
                config.num_key_value_heads,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("qkv_proj", prefix),
            )
            self.gate_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                parallel_group="attn_tp",
                prefix=add_prefix("gate_proj", prefix),
            )
        self.attention_gate = BeamAttentionGateOp()
        self.gate_bias = nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
        set_weight_attrs(self.gate_bias, {"weight_loader": self.load_gate_bias})
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            parallel_group="attn_tp",
            use_dp_attention_reduce=True,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = (
            (rotary_emb if rotary_emb is not None else BeamRotaryEmbedding(config))
            if self.apply_rope
            else None
        )
        self.use_fused_qk_norm_rope = self.apply_rope and _can_fuse_qk_norm_rope(
            self.head_dim
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window,
            prefix=add_prefix("attn", prefix),
        )

    def _fused_qk_norm_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> None:
        # In place on the packed QKV views. Rounding the norm to the activation
        # dtype before the FP32 rotation matches RMSNorm followed by RoPE.
        fused_inplace_qknorm_rope(
            q=q.view(-1, self.num_heads, self.head_dim),
            k=k.view(-1, self.num_kv_heads, self.head_dim),
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            positions=positions,
            is_neox=True,
            eps=self.q_norm.variance_epsilon,
            head_dim=self.head_dim,
            rope_dim=self.head_dim,
            round_norm_before_rope=True,
        )

    def load_gate_bias(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        default_weight_loader(
            param,
            loaded_weight.narrow(0, self.tp_rank * self.num_heads, self.num_heads),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        ssmax_scale: torch.Tensor | None,
        *,
        skip_all_reduce: bool = False,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states
        if self.use_packed_projection:
            qkv_gate, _ = self.qkv_gate_proj(hidden_states)
            q, k, v, gate = qkv_gate.split(
                [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        force_torch = get_fused_op_backend() == KernelBackend.TORCH
        if self.use_fused_qk_norm_rope and not force_torch:
            self._fused_qk_norm_rope(positions, q, k)
        else:
            if force_torch:
                q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
                k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
            else:
                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    head_dim=self.head_dim,
                    allow_inplace=_is_cuda,
                )
            if self.rotary_emb is not None:
                q, k = self.rotary_emb(positions, q, k)
            else:
                q = apply_ssmax(q, ssmax_scale)
        attention_output = self.attn(q, k, v, forward_batch)
        if not self.use_packed_projection:
            gate, _ = self.gate_proj(hidden_states)
        attention_output = self.attention_gate(
            attention_output.view(-1, self.q_size),
            gate.view(-1, self.q_size),
            self.gate_bias,
            self.head_dim,
        )
        output, _ = self.o_proj(attention_output, skip_all_reduce=skip_all_reduce)
        return output


class BeamDecoderLayer(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        layer_id: int,
        prefix: str = "",
        rotary_emb: BeamRotaryEmbedding | None = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.rms_norm_eps = config.rms_norm_eps
        self.sandwich_residual_norm = BeamSandwichResidualRMSNormOp()
        attn_depth = 2 * (layer_id + 1)
        mlp_depth = attn_depth + 1
        attn_denominator = math.sqrt(attn_depth)
        mlp_denominator = math.sqrt(mlp_depth)
        self.attn_residual_scales = (
            1.0 / attn_denominator,
            math.sqrt(attn_depth - 1) / attn_denominator,
        )
        self.mlp_residual_scales = (
            1.0 / mlp_denominator,
            math.sqrt(mlp_depth - 1) / mlp_denominator,
        )
        self.self_attn = BeamAttention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            rotary_emb=rotary_emb,
        )
        self.mlp = (
            BeamMLP(config, prefix=add_prefix("mlp", prefix), quant_config=quant_config)
            if config.mlp_layer_types[layer_id] == "dense"
            else BeamMoE(
                config,
                layer_id,
                prefix=add_prefix("mlp", prefix),
                quant_config=quant_config,
            )
        )
        # Round once, after FP32 normalization and learned-weight multiplication.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # The MLP is sharded over the full TP group, so attention-DP ranks
        # gather their tokens around it; norms and residuals stay DP-local.
        # The MegaMOE all-to-all instead takes each DP rank's own tokens; every
        # EP rank enters it on every forward, including ranks with no tokens.
        self.gather_mlp_tokens = get_parallel().attn_dp_size > 1 and not (
            config.mlp_layer_types[layer_id] != "dense"
            and get_moe_a2a_backend().is_flashinfer_megamoe()
        )

    def _forward_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_all_reduce: bool = False,
    ) -> torch.Tensor:
        """`skip_all_reduce=True` returns this rank's partial sum for the caller
        to reduce; DP attention gathers tokens instead and never skips."""
        if not self.gather_mlp_tokens:
            return self.mlp(hidden_states, skip_all_reduce=skip_all_reduce)
        assert not skip_all_reduce, "DP-gathered MLPs reduce-scatter their output"
        local_hidden_states = hidden_states
        hidden_states = get_global_dp_buffer(get_parallel().tp_group)
        dp_gather_replicate(
            global_tokens=hidden_states,
            local_tokens=local_hidden_states,
            forward_batch=forward_batch,
        )
        use_reduce_scatterv = (
            isinstance(self.mlp, BeamMoE) and should_use_dp_reduce_scatterv()
        )
        with get_forward().scoped(mlp_reduce_scatter=use_reduce_scatterv):
            global_output = self.mlp(hidden_states)
        output = get_local_dp_buffer(get_parallel().tp_group)
        # BeamMoE skips its all-reduces in this layout; reduce-scatter instead.
        if use_reduce_scatterv:
            get_parallel().tp_group.reduce_scatterv(
                global_output, output=output, sizes=get_dp_global_num_tokens()
            )
        else:
            dp_scatter(
                local_tokens=output,
                global_tokens=global_output,
                forward_batch=forward_batch,
            )
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        ssmax_scale: torch.Tensor | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions, hidden_states, forward_batch, ssmax_scale
        )
        hidden_states = depth_scaled_residual(
            self.post_attention_layernorm(hidden_states),
            residual,
            self.attn_residual_scales,
        )
        residual = hidden_states
        hidden_states = self._forward_mlp(
            hidden_states=self.pre_feedforward_layernorm(hidden_states),
            forward_batch=forward_batch,
        )
        return depth_scaled_residual(
            self.post_feedforward_layernorm(hidden_states),
            residual,
            self.mlp_residual_scales,
        )

    def forward_fused(
        self,
        positions: torch.Tensor,
        normed: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        next_norm_weight: Optional[torch.Tensor],
        ssmax_scale: torch.Tensor | None,
        attention_beam_ar_comm=None,
        attention_flashinfer_ar=None,
        mlp_beam_ar_topk_comm=None,
        mlp_beam_ar_comm=None,
        mlp_flashinfer_ar=None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Same arithmetic as `forward` with the two norm/residual transitions fused.

        `normed` is this layer's `input_layernorm(residual)`, produced by the
        previous transition. Returns the next layer's normalized input (or None
        when `next_norm_weight` is None, on the last layer of a pipeline stage
        that does not own the final norm) and the new residual stream.
        Each boundary takes at most one replacement for its TP all-reduce:
        `*_beam_ar_comm` fuses the all-reduce and norm in `beam_all_reduce_norm`
        (`mlp_beam_ar_topk_comm` also sums the top-k expert outputs, and wins
        when it applies); `*_flashinfer_ar` runs FlashInfer's all-reduce ahead
        of the unfused norm.
        """
        # A replacement, when given, owns the all-reduce o_proj would run.
        attn = self.self_attn(
            positions,
            normed,
            forward_batch,
            ssmax_scale,
            skip_all_reduce=attention_beam_ar_comm is not None
            or attention_flashinfer_ar is not None,
        )
        normed, residual = self._boundary(
            attn,
            residual,
            self.post_attention_layernorm.weight,
            self.pre_feedforward_layernorm.weight,
            self.attn_residual_scales,
            attention_beam_ar_comm,
            attention_flashinfer_ar,
        )
        if (
            mlp_beam_ar_topk_comm is not None
            and isinstance(self.mlp, BeamMoE)
            and self.mlp.experts.quant_method.runner.runner_backend.is_triton()
            and not self.mlp.experts._dwdp_bound
        ):
            # Per-expert outputs; the kernel sums them and applies the routed scale.
            return self._boundary(
                self.mlp(normed, defer_sum=True),
                residual,
                self.post_feedforward_layernorm.weight,
                next_norm_weight,
                self.mlp_residual_scales,
                mlp_beam_ar_topk_comm,
                None,
                routed_scale=self.mlp.config.routed_scaling_factor,
            )
        mlp = self._forward_mlp(
            normed,
            forward_batch,
            skip_all_reduce=mlp_beam_ar_comm is not None
            or mlp_flashinfer_ar is not None,
        )
        return self._boundary(
            mlp,
            residual,
            self.post_feedforward_layernorm.weight,
            next_norm_weight,
            self.mlp_residual_scales,
            mlp_beam_ar_comm,
            mlp_flashinfer_ar,
        )

    def _boundary(
        self,
        branch,
        residual,
        post_weight,
        next_weight,
        scales,
        beam_ar_comm,
        flashinfer_ar,
        routed_scale=1.0,
    ):
        """Reduce `branch` by the given replacement (or not at all, when it was
        reduced already) and apply the sandwich norm and residual combine."""
        if beam_ar_comm is not None:
            from sglang.kernels.ops.communication.beam_all_reduce import (
                beam_all_reduce_norm,
            )

            return beam_all_reduce_norm(
                beam_ar_comm,
                branch,
                residual,
                post_weight,
                next_weight,
                self.rms_norm_eps,
                scales,
                routed_scale=routed_scale,
            )
        if flashinfer_ar is not None:
            branch = flashinfer_ar(branch)
        return self.sandwich_residual_norm(
            branch, residual, post_weight, next_weight, self.rms_norm_eps, scales
        )


class BeamModel(nn.Module):
    def __init__(
        self,
        config: BeamConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=get_parallel().attn_dp_enabled,
                prefix=add_prefix("embed_tokens", prefix),
            )
            if self.pp_group.is_first_rank
            else PPMissingLayer()
        )
        # Reuse the FP32 position cache across this model's rotary layers.
        rotary_emb = (
            BeamRotaryEmbedding(config)
            if "sliding_attention" in config.layer_types
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: BeamDecoderLayer(
                config, idx, prefix, rotary_emb=rotary_emb, quant_config=quant_config
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )
        self.has_full_attention = any(
            config.layer_types[layer_id] == "full_attention"
            for layer_id in range(self.start_layer, self.end_layer)
        )
        # Decoder layers whose outputs feed a speculative drafter (DFlash).
        self.layers_to_capture = ()
        # Deferred expert sums need the BF16 Triton runner's per-expert outputs.
        self.fuse_moe_all_reduce = (
            quant_config is None and config.num_experts_per_tok == 8
        )
        self._beam_all_reduce_comm = None
        # Offload wrappers load weights through each layer's regular forward.
        # Fusing across layer boundaries would bypass those wrappers.
        # A custom PP partition can leave a stage with no layer to fuse from.
        self.fuse_layers = (
            get_exec().offload.cpu_offload_gb <= 0
            and get_exec().offload.offload_group_size <= 0
            and self.start_layer < self.end_layer
        )
        # FlashInfer's all-reduce serves this token band once set up, on the
        # first eager forward (see _flashinfer_ar_for).
        self._flashinfer_ar_tokens = (
            self._flashinfer_ar_band() if self.fuse_layers else None
        )
        self._flashinfer_ar = None
        self._mlp_has_single_tp_sum = _mlp_owes_single_tp_sum()

    @staticmethod
    def _flashinfer_ar_band() -> Optional[Tuple[int, int]]:
        """This GPU and TP size's measured token band, if FlashInfer is available."""
        parallel = get_parallel()
        if (
            parallel.tp_size < 2
            or not _beam_owns_tp_boundaries()
            # The global FlashInfer all-reduce already serves every size.
            or get_exec().comm.flashinfer_allreduce_fusion_backend is not None
            or get_exec().comm.enforce_disable_flashinfer_allreduce_fusion
            or not torch.cuda.is_available()
        ):
            return None
        from sglang.srt.utils import is_flashinfer_available

        band = _FLASHINFER_AR_TOKEN_BAND.get(
            (torch.cuda.get_device_capability(), parallel.tp_size)
        )
        return band if band is not None and is_flashinfer_available() else None

    def _setup_flashinfer_ar(self, hidden_size: int) -> None:
        # Runs on the first eager forward, so meta-built models are materialized
        # and no CUDA graph is capturing. Every TP rank runs it in the same
        # forward; one failed allocation disables the band on all of them.
        from sglang.srt.layers.flashinfer_comm_fusion import (
            FlashInferWorkspaceManager,
            is_flashinfer_allreduce_unavailable,
        )

        group = get_parallel().tp_group
        manager = FlashInferWorkspaceManager()
        if not is_flashinfer_allreduce_unavailable():
            manager.initialize(
                world_size=group.world_size,
                rank=group.rank_in_group,
                max_token_num=self._flashinfer_ar_tokens[1],
                hidden_dim=hidden_size,
                backend="mnnvl",
                dtype=torch.bfloat16,
                device_group=group.device_group,
                cpu_group=group.cpu_group,
            )
        ready = torch.tensor(
            [int(manager.workspace is not None)], dtype=torch.int32, device="cpu"
        )
        torch.distributed.all_reduce(
            ready, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        if not ready.item():
            manager.cleanup()
            self._flashinfer_ar_tokens = None
            return
        atexit.register(manager.cleanup)
        self._flashinfer_ar = partial(
            _flashinfer_all_reduce, workspace=manager.workspace
        )
        logger.info(
            "Beam uses FlashInfer's all-reduce for %d-%d tokens per forward",
            *self._flashinfer_ar_tokens,
        )

    def _flashinfer_ar_for(self, hidden_states: torch.Tensor):
        """FlashInfer's all-reduce if the token count is in its band, else None."""
        band = self._flashinfer_ar_tokens
        if band is None or torch.compiler.is_compiling():
            return None
        if self._flashinfer_ar is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._setup_flashinfer_ar(hidden_states.shape[1])
            band = self._flashinfer_ar_tokens
        if (
            band is None
            or not band[0] <= hidden_states.shape[0] <= band[1]
            or hidden_states.dtype != torch.bfloat16
            or get_forward().sp_active
        ):
            return None
        return self._flashinfer_ar

    def _can_replace_mlp_all_reduce(self) -> bool:
        """Whether a fused or FlashInfer all-reduce may replace the MLP's own."""
        return self._mlp_has_single_tp_sum and not should_skip_mlp_all_reduce()

    def _all_reduce_limits(self, hidden_states) -> Optional[Tuple[int, int]]:
        """Return attention/top-k MLP token limits for eligible inputs; no allocation."""
        if not hidden_states.is_cuda or torch.compiler.is_compiling():
            return None
        if (
            hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or hidden_states.data_ptr() % 16 != 0
            or hidden_states.shape[1] != 5120
            or not 0 < hidden_states.shape[0] <= 64
        ):
            return None
        # Attention owns a TP reduction independently of the expert layout.
        parallel = get_parallel()
        capability = torch.cuda.get_device_capability(hidden_states.device)
        architecture = capability[0]
        if (
            parallel.tp_size not in (2, 4, 8)
            or architecture not in (9, 10)
            or get_forward().sp_active
            or not _beam_owns_tp_boundaries()
        ):
            return None
        # Keep the existing collective above the measured one-shot crossover.
        # Attention and MoE have different savings: only MoE removes a local sum.
        attention_max_tokens, topk_max_tokens = (
            {2: (32, 32), 4: (32, 64), 8: (1, 16)}[parallel.tp_size]
            if architecture == 9
            else (64, 64)
        )
        # On GB300 TP4 the fused norm's reduction order changed trained-model
        # decode output at 64 tokens; cap it at 32 until the orders match.
        if capability == (10, 3) and parallel.tp_size == 4:
            attention_max_tokens = min(attention_max_tokens, 32)
            topk_max_tokens = min(topk_max_tokens, 32)
        # Deferred expert outputs currently require unsharded expert IDs and
        # one TP reduction. EP keeps its normal expert mapping and reduction.
        if (
            not self.fuse_moe_all_reduce
            or parallel.moe_ep_size != 1
            or parallel.moe_tp_size != parallel.tp_size
            or should_skip_post_experts_all_reduce(is_tp_path=True)
        ):
            topk_max_tokens = 0
        max_tokens = max(attention_max_tokens, topk_max_tokens)
        if hidden_states.shape[0] > max_tokens:
            return None
        return attention_max_tokens, topk_max_tokens

    def _prepare_all_reduce_comm(self, hidden_states, max_tokens: int):
        """Reuse or initialize the shared push workspace during eager warmup.

        The caller supplies the capacity selected by `_all_reduce_limits`.
        Capture may reuse an initialized communicator, but never allocates one.
        """
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        comm = get_parallel().tp_group.ca_comm
        if not isinstance(comm, CustomAllReduceV2) or comm.disabled:
            return None
        if self._beam_all_reduce_comm is None:
            # Initialize once in eager warmup, before any CUDA graph capture.
            # Hopper's native push plane is tuned for much smaller messages.
            # Share one bounded plane across all attention and MoE boundaries.
            if torch.cuda.is_current_stream_capturing():
                return None
            required_bytes = max_tokens * hidden_states.shape[1] * 2
            if not comm.supports_push(nbytes=required_bytes, num_blocks=max_tokens):
                from sglang.srt.distributed.comm_cleanup import register_comm_cleanup

                comm = CustomAllReduceV2(
                    comm.group,
                    comm.device,
                    max_pull_size=0,
                    max_pull_blocks=0,
                    max_push_size=required_bytes,
                    max_push_blocks=max_tokens,
                )
                register_comm_cleanup(comm)
            self._beam_all_reduce_comm = comm
        comm = self._beam_all_reduce_comm
        if not comm.supports_push(
            nbytes=hidden_states.numel() * hidden_states.element_size(),
            num_blocks=hidden_states.shape[0],
        ):
            return None
        return comm.obj

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
        else:
            hidden_states = pp_proxy_tensors["hidden_states"]
        # Positions are shared by every full-attention layer of this stage.
        scale = compute_ssmax_scale(positions) if self.has_full_attention else None
        aux_hidden_states = []
        if self.fuse_layers:
            return self._forward_fused(
                positions, hidden_states, forward_batch, aux_hidden_states, scale
            )
        for layer_id in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[layer_id](
                positions, hidden_states, forward_batch, scale
            )
            if layer_id in self.layers_to_capture:
                # The post-layer residual stream, before the final norm.
                aux_hidden_states.append(hidden_states)
        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states})
        hidden_states = self.norm(hidden_states)
        if self.layers_to_capture:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _forward_fused(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        aux_hidden_states: list,
        ssmax_scale: torch.Tensor | None,
    ):
        # Each fused transition also produces the next layer's normalized input,
        # so the first norm of this stage is the only standalone one.
        attention_beam_ar_comm = mlp_beam_ar_topk_comm = mlp_beam_ar_comm = None
        limits = self._all_reduce_limits(hidden_states)
        if limits is not None:
            attention_max_tokens, topk_max_tokens = limits
            comm = self._prepare_all_reduce_comm(hidden_states, max(limits))
            if hidden_states.shape[0] <= topk_max_tokens:
                mlp_beam_ar_topk_comm = comm
            if hidden_states.shape[0] <= attention_max_tokens:
                attention_beam_ar_comm = comm
                # A summed MLP output is the same [tokens, hidden] partial sum as
                # attention's, so it fuses under the same limit.
                if self._can_replace_mlp_all_reduce():
                    mlp_beam_ar_comm = comm
        attention_flashinfer_ar = self._flashinfer_ar_for(hidden_states)
        mlp_flashinfer_ar = None
        if attention_flashinfer_ar is not None and self._can_replace_mlp_all_reduce():
            mlp_flashinfer_ar = attention_flashinfer_ar
        residual = hidden_states
        normed = self.layers[self.start_layer].input_layernorm(residual)
        for layer_id in range(self.start_layer, self.end_layer):
            if layer_id + 1 < self.end_layer:
                next_norm_weight = self.layers[layer_id + 1].input_layernorm.weight
            elif self.pp_group.is_last_rank:
                next_norm_weight = self.norm.weight
            else:
                next_norm_weight = None
            normed, residual = self.layers[layer_id].forward_fused(
                positions,
                normed,
                residual,
                forward_batch,
                next_norm_weight,
                ssmax_scale,
                attention_beam_ar_comm=attention_beam_ar_comm,
                attention_flashinfer_ar=attention_flashinfer_ar,
                mlp_beam_ar_topk_comm=mlp_beam_ar_topk_comm,
                mlp_beam_ar_comm=mlp_beam_ar_comm,
                mlp_flashinfer_ar=mlp_flashinfer_ar,
            )
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(residual)
        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": residual})
        # `normed` already holds `self.norm(residual)`.
        if self.layers_to_capture:
            return normed, aux_hidden_states
        return normed


def get_beam_quantization(quant_config: Optional[QuantizationConfig]) -> Optional[str]:
    """Identify supported formats for validation, independently of layer dispatch."""
    if quant_config is None:
        return "bf16"
    if quant_config.get_name() == "compressed_tensors":
        if (
            quant_config.quant_format != "float-quantized"
            or quant_config.linear_fp8_config is not None
        ):
            return None
        for scheme in quant_config.target_scheme_map.values():
            weights = scheme.get("weights")
            activations = scheme.get("input_activations")
            if not (
                all(
                    q is not None
                    and q.type == "float"
                    and q.num_bits == 8
                    and q.symmetric
                    for q in (weights, activations)
                )
                and weights.strategy == "channel"
                and not weights.dynamic
                and activations.strategy == "token"
                and activations.dynamic is True
                and scheme.get("format") in (None, "float-quantized")
            ):
                return None
        return "fp8" if quant_config.target_scheme_map else None
    if (
        isinstance(quant_config, ModelOptFp4Config)
        and quant_config.is_checkpoint_nvfp4_serialized
        and not quant_config.is_awq
        and not quant_config.is_w4a16
    ):
        return "nvfp4"
    return None


def get_beam_supported_moe_runners(quant_scheme: Optional[str]) -> Tuple[str, ...]:
    return {
        "bf16": ("auto", "triton"),
        "fp8": ("auto", "triton"),
        "nvfp4": ("flashinfer_trtllm",),
    }.get(quant_scheme, ())


class BeamForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: BeamConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if get_beam_quantization(quant_config) is None:
            raise NotImplementedError(
                "Beam supports BF16, compressed-tensors FP8 (per-channel "
                "weights and dynamic per-token activations), or ModelOpt "
                "serialized W4A4 NVFP4"
            )
        if get_parallel().moe_dense_tp_size == 1:
            raise NotImplementedError(
                "Beam shards dense MLP layers over the full TP group; "
                "moe_dense_tp_size=1 is not supported"
            )
        if (
            get_exec().moe.enable_eplb
            or get_exec().moe.init_expert_location != "trivial"
            or get_exec().moe.ep_num_redundant_experts > 0
        ):
            raise NotImplementedError(
                "Beam has not been validated with expert relocation or "
                "redundant experts; use trivial expert placement"
            )
        if get_lora().enable_lora or get_lora().lora_paths:
            raise NotImplementedError("LoRA adapters are not supported for Beam")
        self.config = config
        self.model = BeamModel(config, quant_config, prefix=add_prefix("model", prefix))
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        self.lm_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            if self.model.pp_group.is_last_rank
            else PPMissingLayer()
        )
        # The default head projection returns BF16; LogitsProcessor promotes
        # its output to FP32 before softcap and logprobs.
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.model.pp_group.is_last_rank:
            return hidden_states
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def set_dflash_layers_to_capture(self, layer_ids):
        """Capture the listed decoder layers' outputs for the DFlash drafter."""
        if self.model.pp_group.world_size != 1:
            raise ValueError("Beam DFlash capture currently requires PP=1")
        layer_ids = tuple(layer_ids)
        if (
            not layer_ids
            or list(layer_ids) != sorted(set(layer_ids))
            or any(i < 0 or i >= self.config.num_hidden_layers for i in layer_ids)
        ):
            raise ValueError(
                "DFlash capture layers must be sorted, unique, and in range"
            )
        self.model.layers_to_capture = layer_ids
        self.capture_aux_hidden_states = True

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self):
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        packed_projections = [
            (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"),
            (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
            (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
            (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
            (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
        ]
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params = dict(self.named_parameters())
        # Select by the actual destination parameter so PP shards and the
        # replicated-KV/quantized fallback retain their existing loaders.
        attention_shards = {"q_proj": 0, "k_proj": 1, "v_proj": 2, "gate_proj": 3}
        loaded = set()
        for name, weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and not self.start_layer <= layer_id < self.end_layer
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if (
                name == "model.embed_tokens.weight"
                and not self.model.pp_group.is_first_rank
            ):
                continue
            if (
                name == "model.norm.weight" or name.startswith("lm_head.")
            ) and not self.model.pp_group.is_last_rank:
                continue
            if ".self_attn." in name:
                prefix, suffix = name.split(".self_attn.", 1)
                projection, separator, field = suffix.partition(".")
                target_name = f"{prefix}.self_attn.qkv_gate_proj.{field}"
                if (
                    separator
                    and projection in attention_shards
                    and target_name in params
                ):
                    param = params[target_name]
                    param.weight_loader(param, weight, attention_shards[projection])
                    loaded.add(target_name)
                    continue
            for source, target, shard_id in packed_projections:
                if source in name:
                    target_name = name.replace(source, target)
                    param = params[target_name]
                    param.weight_loader(param, weight, shard_id)
                    loaded.add(target_name)
                    break
            else:
                for target, source, expert_id, shard_id in expert_mapping:
                    if source in name:
                        target_name = name.replace(source, target)
                        param = params[target_name]
                        param.weight_loader(
                            param,
                            weight,
                            target_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded.add(target_name)
                        break
                else:
                    # Packed expert checkpoints use [expert, 2*intermediate, hidden]
                    # in gate/up order, or [expert, hidden, intermediate] for down.
                    if name.endswith(".mlp.experts.gate_up_proj"):
                        target_name = name.replace("gate_up_proj", "w13_weight")
                        param = params[target_name]
                        for expert_id, expert_weight in enumerate(weight):
                            gate, up = expert_weight.chunk(2, dim=0)
                            for shard_id, shard in (("w1", gate), ("w3", up)):
                                param.weight_loader(
                                    param,
                                    shard,
                                    target_name,
                                    shard_id=shard_id,
                                    expert_id=expert_id,
                                )
                        loaded.add(target_name)
                    elif name.endswith(".mlp.experts.down_proj"):
                        target_name = name.replace("down_proj", "w2_weight")
                        param = params[target_name]
                        for expert_id, shard in enumerate(weight):
                            param.weight_loader(
                                param,
                                shard,
                                target_name,
                                shard_id="w2",
                                expert_id=expert_id,
                            )
                        loaded.add(target_name)
                    else:
                        param = params[name]
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, weight)
                        loaded.add(name)
        return loaded


EntryClass = BeamForCausalLM
