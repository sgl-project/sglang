"""Which kernel decides each TOP op's bits, as references into the code; ``kernel_inventory`` renders and audits them.

This file holds references only. Each reason lives at the code it points to: a TOP kernel's docstring opens with
"Stands in for <stock kernel>: <reason>", a gate carries its reason in a comment on or next to the anchored line, and
a pin is the keyword set of the pinned call or a module-level constant. A reference is ``"<module>:<qualname>"``;
an empty qualname names the module (its docstring carries the claim).

Plain NamedTuples, not msgspec: ``kernel_inventory`` runs on the standard library alone, without importing sglang.
"""

from __future__ import annotations

from typing import NamedTuple, Union


class Stock(NamedTuple):
    """A stock kernel TOP runs in one configuration."""

    kernel: str
    # The function whose call to ``kernel`` carries the pinned arguments, if one does.
    pinned_at: str = ""
    # Callee spellings at ``pinned_at`` (an import alias, a kwargs.update); default: the kernel's own name.
    calls: tuple[str, ...] = ()
    # Module-level constants that hold the pinned configuration.
    pins: tuple[str, ...] = ()


class TopKernel(NamedTuple):
    """A TOP-written kernel; its docstring states which stock kernel it stands in for and why."""

    kernel: str
    stands_in_for: tuple[str, ...]


class Gate(NamedTuple):
    """A stock fast path TOP turns off; the reason is the comment at ``anchor`` inside ``at``, or for a startup
    refusal of an unqualified path, the refusal's message."""

    at: str
    anchor: str
    fast_path: str


class OpKernels(NamedTuple):
    kernels: tuple[Union[Stock, TopKernel], ...] = ()
    gates: tuple[Gate, ...] = ()
    # For an op without a kernel of its own (a draft running the target's ops): the function that admits it.
    admitted_by: str = ""


_TOP = "sglang.srt.true_on_policy"
_BI = "sglang.srt.batch_invariant_ops.batch_invariant_ops"
_FA4 = "sglang.kernels.ops.attention.flash_attn.cute.interface"
_FA4_PIN = ("fa4_varlen", "kwargs.update", "set_batch_invariant")
_DG = "sglang.srt.layers.deep_gemm_wrapper.entrypoint"
_DG_RUNNER = "sglang.srt.layers.moe.moe_runner.deep_gemm"
_EP_KERNELS = "sglang.kernels.ops.moe.ep_moe_kernels"
_ACT = "sglang.kernels.ops.activation.activation"
_FLA = "sglang.kernels.ops.attention.fla"
_FLASH_MLA = "sglang.kernels.ops.attention.flash_mla_interface"
_MHC = "sglang.kernels.ops.layernorm.mhc"
_FP8_UTILS = "sglang.srt.layers.quantization.fp8_utils"
_DSA = "sglang.srt.layers.attention.dsa"
_DSV4_BACKEND = "sglang.srt.layers.attention.deepseek_v4_backend"
_QSA = "sglang.srt.layers.attention.qsa"

_BI_GEMM = Stock(
    kernel="deep_gemm:bf16_gemm_nn",
    pinned_at=f"{_BI}:_matmul_persistent_deepgemm",
    calls=("deep_gemm.bf16_gemm_nn", "deep_gemm.use_deterministic_algorithms"),
)
_BI_GEMM_GATE = Gate(
    at=f"{_BI}:matmul_persistent",
    anchor="if _is_true_on_policy():",
    fast_path=f"{_BI}:_matmul_persistent_triton",
)
_FA4_CAUSAL = Stock(
    kernel=f"{_FA4}:flash_attn_varlen_func",
    pinned_at=f"{_TOP}.attention:flash_attn_varlen_func",
    calls=_FA4_PIN,
)
_FA4_NONCAUSAL = Stock(
    kernel=f"{_FA4}:flash_attn_varlen_func",
    pinned_at=f"{_TOP}.attention:_forward",
    calls=_FA4_PIN,
)
_ORDERED_TOPK = Stock(
    kernel="flashinfer:top_k_ragged_transform",
    pinned_at=f"{_TOP}.dsa.ordered_topk:ordered_dsa_topk",
)
_SLOT_COMBINE = Stock(
    kernel=f"{_EP_KERNELS}:ep_gather",
    pinned_at=f"{_TOP}.slot_order_deepep:owner_reduce",
)
_BF16_GROUPED = (
    Stock(
        kernel=f"{_DG}:grouped_gemm_nt_bf16_contig",
        pinned_at=f"{_DG_RUNNER}:DeepGemmRunnerCore._run_bf16_contiguous_gemm",
    ),
    Stock(
        kernel=f"{_DG}:grouped_gemm_nt_bf16_masked",
        pinned_at=f"{_DG_RUNNER}:DeepGemmRunnerCore._run_masked_bf16_gemm",
    ),
    Stock(
        kernel=f"{_EP_KERNELS}:silu_and_mul_masked_fwd",
        pinned_at=f"{_DG_RUNNER}:DeepGemmRunnerCore._run_masked_bf16_gemm",
    ),
    Stock(
        kernel=f"{_ACT}:silu_and_mul_accurate",
        pinned_at=f"{_DG_RUNNER}:_contiguous_silu_and_mul",
    ),
)
_PRECISE_SILU_GATES = (
    Gate(
        at=f"{_DG_RUNNER}:_contiguous_silu_and_mul",
        anchor="if should_use_precise_masked_silu():",
        fast_path=f"{_ACT}:silu_and_mul",
    ),
    Gate(
        at=f"{_DG_RUNNER}:DeepGemmRunnerCore._run_masked_bf16_gemm",
        anchor="precise=should_use_precise_masked_silu(),",
        fast_path=f"{_EP_KERNELS}:silu_and_mul_masked_fwd",
    ),
)
_FP8_QUANT = Stock(
    kernel="sglang.kernels.ops.quantization.per_token_group_quant:_jit_module",
    pinned_at=f"{_TOP}.fp8.fp8_expert_forward:quantize_rows",
)
# Decode's expert input: DeepEP's fused FP8 cast, the program of quantize_rows with the DeepEP amax floor
# (fp8_expert_forward.quantize_expert_input), which prefill and the trainer run.
_FP8_EXPERT_INPUT = Stock(
    kernel="deep_ep:Buffer.low_latency_dispatch",
    pinned_at="sglang.srt.layers.moe.token_dispatcher.deepep:_DeepEPDispatcherImplLowLatency._dispatch_core",
    calls=("buffer.low_latency_dispatch",),
)
_FP8_GEMM = Stock(
    kernel=f"{_DG}:gemm_nt_f8f8bf16",
    pinned_at=f"{_TOP}.fp8.fp8_linear:_gemm",
)
_FP8_GROUPED = (
    Stock(
        kernel=f"{_DG}:grouped_gemm_nt_f8f8bf16_contig",
        pinned_at=f"{_TOP}.fp8.fp8_expert_forward:expert_forward",
    ),
    Stock(
        kernel=f"{_DG}:grouped_gemm_nt_f8f8bf16_masked",
        pinned_at=f"{_TOP}.fp8.fp8_expert_forward:expert_forward",
    ),
)
_HOPPER_SCALES = TopKernel(
    kernel=f"{_TOP}.fp8.hopper_scales:",
    stands_in_for=(f"{_FP8_UTILS}:inverse_transform_scale_ue8m0",),
)
_FP8_EXPERT_GATES = (
    Gate(
        at=f"{_DG_RUNNER}:DeepGemmRunnerCore.run",
        anchor="if top_hidden_states is not None:",
        fast_path=f"{_DG_RUNNER}:DeepGemmRunnerCore._run_contiguous_gemm",
    ),
    Gate(
        at="sglang.srt.layers.quantization.fp8:Fp8MoEMethod.process_weights_after_loading_block_quant",
        anchor="if not top_fp8.requant_ue8m0_experts(weight, weight_scale):",
        fast_path=f"{_FP8_UTILS}:requant_block_scale_ue8m0_for_deepgemm",
    ),
)
_OWNER_COMBINE_GATE = Gate(
    at="sglang.srt.models.deepseek_v2:DeepseekV2MoE.forward_deepep",
    anchor="return top_glm_combine(final_hidden_states, shared_output)",
    fast_path="sglang.srt.models.deepseek_v2:_scale_routed_and_add_shared",
)
_DSV4_SPARSE = (
    TopKernel(
        kernel=f"{_TOP}.dsv4_attention:",
        stands_in_for=(f"{_FLASH_MLA}:flash_mla_with_kvcache",),
    ),
    Stock(
        kernel=f"{_FLASH_MLA}:flash_mla_sparse_fwd",
        pinned_at=f"{_TOP}.dsv4_attention:sparse_attention",
    ),
)
_DSV4_PAGED_DECODE_GATE = Gate(
    at=f"{_DSV4_BACKEND}:DeepseekV4AttnBackend._forward_attention",
    anchor="if self.top_sparse_attention:",
    fast_path=f"{_FLASH_MLA}:flash_mla_with_kvcache",
)
_DSV4_ROUTER_GATE = Gate(
    at=f"{_TOP}.dsv4:validate_top_dsv4_topk",
    anchor="or cfg.apply_routed_scaling_factor_on_output",
    fast_path="sglang.srt.layers.moe.topk:is_routed_scaling_factor_fused_in_topk",
)
_FUSED_GATE = Stock(
    kernel="sglang.kernels.ops.moe.moe_fused_gate:moe_fused_gate",
    pinned_at="sglang.srt.layers.moe.topk:biased_topk_jit_kernel_impl",
)
_FP32_GATE_LOGITS = Stock(
    kernel=f"{_BI}:_matmul_persistent_deepgemm",
    pinned_at=f"{_BI}:_matmul_persistent_top",
)
_MAYBE_LINEAR_GATE = Gate(
    at="sglang.srt.layers.quantization.fp8:Fp8LinearMethod.apply",
    anchor="top_fp8.maybe_linear(self, layer, x, bias)",
    fast_path=f"{_FP8_UTILS}:dispatch_w8a8_block_fp8_linear",
)
_CLAMPED_MLP_GATE = Gate(
    at="sglang.srt.models.deepseek_v2:DeepseekV2MLP.forward",
    anchor="top_fp8.maybe_clamped_mlp(self, x)",
    fast_path="sglang.kernels.ops.activation.silu_and_mul_clamp:silu_and_mul_contig_post_quant",
)
_VISION_GATES = (
    Gate(
        at="sglang.srt.layers.attention.vision:VisionAttention.forward",
        anchor="output = self._top_attention(",
        fast_path="sglang.srt.layers.attention.vision:VisionFlash4Attention",
    ),
    Gate(
        at=f"{_TOP}.vision:refuse_unqualified_serving",
        anchor='get_mm().mm_attention_backend not in (None, "fa4")',
        fast_path="sglang.srt.layers.attention.vision:VisionFlash4Attention",
    ),
)
_GDN_REPLAYSSM_GATE = Gate(
    at=f"{_TOP}.gdn_contract:validate_gdn_contract",
    anchor="if args.enable_linear_replayssm:",
    fast_path="sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm:fused_recurrent_linear_replayssm_decode",
)

OPS: dict[str, OpKernels] = {
    "bi_linear": OpKernels(
        kernels=(_BI_GEMM,),
        gates=(
            _BI_GEMM_GATE,
            Gate(
                at="sglang.srt.layers.linear:RowParallelLinear.forward",
                anchor="if top_output is not None:",
                fast_path="sglang.srt.layers.quantization.unquant:UnquantizedLinearMethod.apply",
            ),
        ),
    ),
    "rms_norm": OpKernels(
        kernels=(
            Stock(kernel="sgl_kernel:rmsnorm"),
            Stock(kernel="sgl_kernel:gemma_rmsnorm"),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.layernorm:RMSNorm.forward_cuda",
                anchor="if is_batch_invariant_mode_enabled() and not top_enabled:",
                fast_path=f"{_BI}:rms_norm_batch_invariant",
            ),
        ),
    ),
    "packed_residual": OpKernels(
        kernels=(
            Stock(kernel="sgl_kernel:fused_add_rmsnorm"),
            Stock(kernel="sgl_kernel:gemma_fused_add_rmsnorm"),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.layer_boundary.ops:update_attn_tp_gather_output",
                anchor="if is_true_on_policy_enabled() and update is PLAIN_ADD:",
                fast_path="sglang.srt.layers.layer_boundary.residual.add_norm:PlainAdd.update",
            ),
        ),
    ),
    "rope": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.attention.rope:apply_rope_with_cos_sin_cache_inplace"
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.rotary_embedding.mrope:MRotaryEmbedding.__init__",
                anchor="if self._force_native:",
                fast_path="sglang.srt.layers.rotary_embedding.mrope:MRotaryEmbedding.forward_cuda",
            ),
            Gate(
                at="sglang.srt.models.qwen3_moe:Qwen3MoeAttention.__init__",
                anchor="(not is_true_on_policy_enabled() or _top_fused_rope(self.rotary_emb))",
                fast_path="sglang.kernels.ops.attention.rope:FusedSetKVBufferArg",
            ),
        ),
    ),
    "swiglu": OpKernels(
        kernels=(
            Stock(kernel=f"{_ACT}:silu_and_mul_with_activation_rounding"),
            Stock(
                kernel=f"{_ACT}:activation_module",
                pinned_at=f"{_ACT}:_run_activation_with_rounding_inplace",
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.activation:SiluAndMul.__init__",
                anchor="self._forward_method = self.forward_top",
                fast_path="sglang.srt.layers.activation:SiluAndMul.forward_cuda",
            ),
            Gate(
                at="sglang.srt.models.qwen2:Qwen2MLP.forward",
                anchor="if not is_true_on_policy_enabled() and self.dual_gemm.can_run(",
                fast_path="sglang.srt.layers.dual_gemm:DualGemm",
            ),
        ),
    ),
    "fa4_attention": OpKernels(
        kernels=(
            _FA4_CAUSAL,
            Stock(
                kernel="sglang.kernels.ops.layernorm.norm:fused_inplace_qknorm",
                pinned_at=f"{_TOP}.qk_norm:fused_qk_norm",
            ),
        ),
        gates=(
            Gate(
                at="sglang.kernels.ops.attention.flash_attention_v4:flash_attn_varlen_func",
                anchor="if true_on_policy:",
                fast_path=f"{_FA4}:flash_attn_varlen_func",
            ),
            Gate(
                at="sglang.srt.models.qwen3_moe:Qwen3MoeAttention.__init__",
                anchor="and not should_disable_fused_qk_norm_mrope()",
                fast_path="sglang.kernels.ops.attention.fused_qknorm_rope:fused_qk_norm_rope",
            ),
        ),
    ),
    "lm_head": OpKernels(
        kernels=(
            _BI_GEMM,
            TopKernel(
                kernel=f"{_TOP}.log_softmax:tiled_log_softmax",
                stands_in_for=(f"{_BI}:_log_softmax_kernel",),
            ),
        ),
        gates=(
            Gate(
                at=f"{_BI}:log_softmax",
                anchor="_is_true_on_policy()",
                fast_path=f"{_BI}:_log_softmax_kernel",
            ),
        ),
    ),
    "sampling": OpKernels(
        kernels=(
            TopKernel(
                kernel=f"{_TOP}.exponential_race:",
                stands_in_for=("sglang.srt.layers.sampler:multinomial_with_seed",),
            ),
            Stock(kernel="sglang.kernels.ops.sampling.murmur_hash:murmur_hash32"),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.sampler:Sampler.forward",
                anchor="and simple_sampling_case",
                fast_path="sglang.srt.layers.sampler:Sampler._sample_from_probs",
            ),
            Gate(
                at="sglang.srt.speculative.eagle_utils:eagle_sample",
                anchor="elif _verify_uses_top_race(sampling_info):",
                fast_path="sglang.srt.speculative.eagle_utils:_rejection_sampling_target_probs",
            ),
        ),
    ),
    "qwen_router": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.moe.moe_fused_gate:moe_fused_gate",
                pinned_at=f"{_TOP}.qwen_route:_route_forward",
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.qwen2_moe:Qwen2MoeSparseMoeBlock.__init__",
                anchor="self.topk = (top_router or TopK)(",
                fast_path="sglang.srt.layers.moe.topk:TopK.forward_cuda",
            ),
        ),
    ),
    "ep_slot_transport": OpKernels(
        kernels=(
            TopKernel(
                kernel=f"{_TOP}.slot_order_deepep:",
                stands_in_for=(
                    "sglang.srt.layers.moe.token_dispatcher.deepep:_DeepEPDispatcherImplNormal._combine_core",
                ),
            ),
            _SLOT_COMBINE,
            Stock(
                kernel="deep_ep:Buffer.dispatch",
                pinned_at=f"{_TOP}.slot_order_deepep:raw_dispatch",
                calls=("buffer.dispatch",),
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.moe.fused_moe_triton.layer:create_moe_dispatcher",
                anchor="if a2a_backend.is_deepep() and should_use_slot_order_moe_combine():",
                fast_path="sglang.srt.batch_overlap.two_batch_overlap:MaybeTboDeepEPDispatcher",
            ),
            Gate(
                at=f"{_TOP}.moe_ep:validate_ep_runtime",
                anchor='or args.deepep_dispatcher_output_dtype != "bf16"',
                fast_path="sglang.srt.layers.moe.utils:get_deepep_output_dtype",
            ),
        ),
    ),
    "qwen3_moe_experts": OpKernels(
        kernels=(
            Stock(
                kernel=f"{_DG}:grouped_gemm_nt_bf16_contig",
                pinned_at=f"{_TOP}.moe:_LocalExperts.forward",
            ),
            Stock(
                kernel=f"{_ACT}:silu_and_mul_accurate",
                pinned_at=f"{_TOP}.moe:_LocalExperts.forward",
            ),
            Stock(
                kernel=f"{_EP_KERNELS}:ep_gather",
                pinned_at=f"{_TOP}.moe:_LocalExperts.forward",
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.qwen3_moe:Qwen3MoeSparseMoeBlock.forward",
                anchor="if self._top is not None:",
                fast_path="sglang.srt.models.qwen3_moe:Qwen3MoeSparseMoeBlock.forward_normal",
            ),
        ),
    ),
    "gdn": OpKernels(
        kernels=(
            Stock(
                kernel=f"{_FLA}.chunk:chunk_gated_delta_rule",
                pinned_at=f"{_TOP}.gdn_chunked_engine:ChunkedGDNEngine.extend",
                pins=(
                    f"{_FLA}.chunk_fwd:TOP_CHUNKED_KKT_CONFIG",
                    f"{_FLA}.chunk_delta_h:TOP_CHUNKED_FMA_CLOSE",
                ),
            ),
            TopKernel(
                kernel=f"{_FLA}.chunk_rows:",
                stands_in_for=(
                    f"{_FLA}.fused_recurrent:fused_recurrent_gated_delta_rule_packed_decode",
                    "sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent:fused_sigmoid_gating_delta_rule_update",
                ),
            ),
            TopKernel(
                kernel=f"{_FLA}.chunk_rows2:",
                stands_in_for=(
                    f"{_FLA}.fused_recurrent:fused_recurrent_gated_delta_rule_packed_decode",
                    "sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent:fused_sigmoid_gating_delta_rule_update",
                ),
            ),
        ),
        gates=(
            Gate(
                at=f"{_FLA}.chunk_fwd:_kkt_solve_configs",
                anchor='if envs.SGLANG_TOP_GDN_MODE.get() == "chunked":',
                fast_path=f"{_FLA}.chunk_fwd:chunk_gated_delta_rule_fwd_kkt_solve_kernel",
            ),
            Gate(
                at=f"{_TOP}.gdn_contract:validate_gdn_contract",
                anchor='if args.linear_attn_backend != "triton" or args.mamba_ssm_dtype != "float32":',
                fast_path="sglang.srt.layers.attention.linear.kernels.gdn_cutedsl:CuteDSLGDNKernel",
            ),
            _GDN_REPLAYSSM_GATE,
        ),
    ),
    "kda": OpKernels(
        kernels=(
            Stock(
                kernel=f"{_FLA}.kda:chunk_kda",
                pinned_at="sglang.srt.layers.attention.linear.kernels.kda_triton:TritonKDAKernel.extend",
                pins=(f"{_FLA}.kda_top:TOP_KDA_CONFIG",),
            ),
            Stock(
                kernel=f"{_FLA}.chunk_intra:chunk_kda_fwd_intra",
                pinned_at=f"{_FLA}.kda:chunk_kda_fwd",
                pins=(f"{_FLA}.kda_top:TOP_KDA_CONFIG",),
            ),
            TopKernel(
                kernel=f"{_FLA}.kda_rows:",
                stands_in_for=(
                    f"{_FLA}.fused_recurrent:fused_recurrent_kda_packed_decode",
                ),
            ),
        ),
        gates=(
            Gate(
                at=f"{_FLA}.kda:chunk_kda_fwd",
                anchor="kda_top.top_kda_chunked() and lower_bound is not None",
                fast_path=f"{_FLA}.chunk_intra:chunk_kda_fwd_kernel_intra_sub_chunk",
            ),
            Gate(
                at=f"{_TOP}.kda_chunked:_TopKDAAttnBackend.__init__",
                anchor="dispatcher.supports_packed_decode = False",
                fast_path=f"{_FLA}.fused_recurrent:fused_recurrent_kda_packed_decode",
            ),
            Gate(
                at=f"{_TOP}.kda_chunked:_validate_kda_kernel_envs",
                anchor="if envs.SGLANG_OPT_FUSED_KDA_VERIFY.get():",
                fast_path="sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify:fused_kda_conv_gating_verify",
            ),
        ),
    ),
    "hc_mean_head": OpKernels(
        kernels=(
            Stock(kernel=f"{_MHC}:hc_contract"),
            Stock(kernel=f"{_BI}:mean_batch_invariant"),
        ),
    ),
    "glm_clamped_experts": OpKernels(
        kernels=(
            *_BF16_GROUPED,
            TopKernel(
                kernel=f"{_TOP}.moe:top_clamp_swiglu_input",
                stands_in_for=(
                    "sglang.kernels.ops.activation.silu_and_mul_clamp:silu_and_mul_clamp",
                ),
            ),
            _SLOT_COMBINE,
        ),
        gates=(*_PRECISE_SILU_GATES, _OWNER_COMBINE_GATE),
    ),
    "clamped_swiglu_bf16": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.activation.silu_and_mul_clamp:silu_and_mul_clamp",
                pinned_at="sglang.srt.models.deepseek_v2:DeepseekV2MLP.forward",
            ),
        ),
    ),
    "gdn_full_attention": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate:fused_qk_gemma_rmsnorm_rope_gate"
            ),
            _FA4_CAUSAL,
            Stock(
                kernel="sglang.kernels.ops.elementwise.elementwise:fused_sigmoid_mul"
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.qwen3_5:Qwen3_5AttentionDecoderLayer.__init__",
                anchor="and self.rotary_emb.cos_sin_cache.dtype != torch.float32",
                fast_path="sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate:fused_qk_gemma_rmsnorm_rope_gate",
            ),
            Gate(
                at="sglang.srt.models.qwen3_5:Qwen3_5AttentionDecoderLayer._apply_qk_norm",
                anchor="and not self.top_enabled",
                fast_path="sglang.srt.models.qwen3_5:Qwen3_5AttentionDecoderLayer._apply_qk_norm",
            ),
        ),
    ),
    "decode_kv_split": OpKernels(
        kernels=(
            _FA4_CAUSAL,
            Stock(
                kernel=f"{_FA4}:flash_attn_combine",
                pinned_at=f"{_TOP}.attention:_combine_pieces",
                pins=(f"{_TOP}.attention:MAX_DECODE_KV_PIECES",),
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.attention.flashattention_backend:FlashAttentionBackend.forward_decode",
                anchor="if self.top_decode_kv_split:",
                fast_path="sglang.kernels.ops.attention.flash_attention_v4:flash_attn_with_kvcache",
            ),
            Gate(
                at="sglang.srt.layers.attention.flashattention_backend:FlashAttentionBackend.forward_extend",
                anchor="and forward_batch.forward_mode.is_target_verify()",
                fast_path="sglang.kernels.ops.attention.flash_attention_v4:flash_attn_with_kvcache",
            ),
        ),
    ),
    "shared_expert_gate": OpKernels(
        kernels=(
            Stock(
                kernel=f"{_BI}:_matmul_persistent_triton",
                pinned_at=f"{_TOP}.qwen_moe_router:shared_gate_linear",
                pins=(f"{_TOP}.qwen_moe_router:_SHARED_GATE_CONFIG",),
            ),
            Stock(kernel="sglang.srt.layers.activation:SiluAndMul.forward_cuda"),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.qwen2_moe:Qwen2MoeSparseMoeBlock._forward_shared_experts",
                anchor="if self._top:",
                fast_path="sglang.srt.models.qwen2_moe:Qwen2MoeSparseMoeBlock._forward_shared_experts",
            ),
        ),
    ),
    "qwen2_moe_experts": OpKernels(
        kernels=(*_BF16_GROUPED, _SLOT_COMBINE),
        gates=_PRECISE_SILU_GATES,
    ),
    "dsa": OpKernels(
        kernels=(
            Stock(
                kernel=f"{_FLASH_MLA}:flash_mla_sparse_fwd",
                pinned_at=f"{_DSA}.flashmla:flashmla_sparse_bf16",
            ),
            Stock(kernel="flashinfer:top_k", pins=(f"{_TOP}.glm:DSA_SELECTION_ENVS",)),
            _ORDERED_TOPK,
            Stock(
                kernel=f"{_BI}:_matmul_persistent_deepgemm",
                pinned_at=f"{_TOP}.dsa_head_gate:project_head_gate",
            ),
            Stock(kernel="deep_gemm:fp8_paged_mqa_logits"),
            TopKernel(
                kernel=f"{_TOP}.dsa.flashmla_fp8:",
                stands_in_for=(f"{_FLASH_MLA}:flash_mla_with_kvcache",),
            ),
            TopKernel(
                kernel="sglang.kernels.ops.attention.dsa.dequant_k_cache:dequantize_k_cache_selected",
                stands_in_for=(
                    "sglang.kernels.ops.attention.dsa.dequant_k_cache:dequantize_k_cache_paged_dsa",
                ),
            ),
        ),
        gates=(
            Gate(
                at=f"{_TOP}.dsa.flashmla_fp8:validate_backend",
                anchor='or backend.dsa_decode_impl != "flashmla_sparse"',
                fast_path="sglang.srt.layers.attention.dsa_backend:DeepseekSparseAttnBackend._forward_flashmla_kv",
            ),
            Gate(
                at=f"{_TOP}.glm:validate_glm",
                anchor='or args.dsa_decode_backend != "flashmla_sparse"',
                fast_path="sglang.srt.layers.attention.dsa_backend:DeepseekSparseAttnBackend._forward_flashmla_kv",
            ),
            Gate(
                at=f"{_TOP}.glm:validate_glm",
                anchor="validate_envs(_DSA_SERVING_ENVS)",
                fast_path=f"{_DSA}.dsa_topk_backend:_topk_transform_v2_paged",
            ),
            Gate(
                at=f"{_DSA}.dsa_topk_backend:DSATopKBackend.topk_transform",
                anchor="if top_fp8():",
                fast_path=f"{_DSA}.dsa_topk_backend:_topk_transform_v2_paged",
            ),
            Gate(
                at=f"{_TOP}.glm:validate_glm",
                anchor="or args.enable_hisparse",
                fast_path="sglang.srt.managers.hisparse_coordinator:HiSparseCoordinator.swap_in_selected_pages",
            ),
        ),
    ),
    "glm_router": OpKernels(
        kernels=(
            TopKernel(
                kernel=f"{_TOP}.glm_router:_select_and_normalize_kernel",
                stands_in_for=("sglang.srt.layers.moe.topk:biased_grouped_topk_gpu",),
            ),
            Stock(
                kernel="flashinfer:top_k",
                pinned_at=f"{_TOP}.glm_topk:_select_ids",
                calls=("flashinfer.top_k",),
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.deepseek_v2:MoEGate.forward",
                anchor="if self._top_glm_router:",
                fast_path="sglang.kernels.ops.gemm.bf16_fp32:linear_bf16_fp32",
            ),
            Gate(
                at="sglang.srt.models.deepseek_v2:DeepseekV2MoE.__init__",
                anchor="self.topk = (top_glm_router_cls(is_deepseek_v4=is_deepseek_v4) or TopK)(",
                fast_path="sglang.srt.layers.moe.topk:biased_grouped_topk_gpu",
            ),
        ),
    ),
    "glm_experts": OpKernels(
        kernels=(*_BF16_GROUPED, _SLOT_COMBINE),
        gates=(
            *_PRECISE_SILU_GATES,
            _OWNER_COMBINE_GATE,
            Gate(
                at=f"{_TOP}.glm:validate_glm",
                anchor="if args.enable_waterfill or args.enforce_shared_experts_fusion:",
                fast_path="sglang.srt.layers.moe.utils:get_num_fused_shared_experts",
            ),
            Gate(
                at=f"{_TOP}.glm:validate_glm",
                anchor="args.enable_single_batch_overlap",
                fast_path="sglang.srt.batch_overlap.single_batch_overlap:compute_overlap_args",
            ),
        ),
    ),
    "fp8_linear": OpKernels(
        kernels=(_FP8_QUANT, _FP8_GEMM, _HOPPER_SCALES),
        gates=(
            _MAYBE_LINEAR_GATE,
            Gate(
                at=f"{_TOP}.fp8:maybe_linear",
                anchor="or method.weight_block_size != [128, 128]",
                fast_path=f"{_FP8_UTILS}:dispatch_block_fp8_mxfp8_linear",
            ),
        ),
    ),
    "glm_fp8_experts": OpKernels(
        kernels=(
            _FP8_EXPERT_INPUT,
            _FP8_QUANT,
            *_FP8_GROUPED,
            _SLOT_COMBINE,
            _HOPPER_SCALES,
        ),
        gates=(
            *_FP8_EXPERT_GATES,
            Gate(
                at=f"{_TOP}.glm:validate_glm",
                anchor="if args.enable_waterfill or args.enforce_shared_experts_fusion:",
                fast_path="sglang.srt.layers.moe.utils:get_num_fused_shared_experts",
            ),
        ),
    ),
    "glm_mtp_draft": OpKernels(admitted_by=f"{_TOP}.mtp:validate_mtp"),
    "clamped_swiglu": OpKernels(
        kernels=(_FP8_QUANT, _FP8_GEMM, _HOPPER_SCALES),
        gates=(_CLAMPED_MLP_GATE,),
    ),
    "dsv4_router": OpKernels(
        kernels=(
            _FP32_GATE_LOGITS,
            _FUSED_GATE,
            Stock(kernel="sglang.kernels.ops.moe.dsv4:hash_topk"),
        ),
        gates=(_DSV4_ROUTER_GATE,),
    ),
    "dsv4_experts": OpKernels(
        kernels=(
            _FP8_EXPERT_INPUT,
            _FP8_QUANT,
            *_FP8_GROUPED,
            _SLOT_COMBINE,
            _HOPPER_SCALES,
        ),
        gates=(
            *_FP8_EXPERT_GATES,
            _OWNER_COMBINE_GATE,
            Gate(
                at=f"{_TOP}.dsv4:validate_dsv4",
                anchor='("shared-expert fusion", args.enforce_shared_experts_fusion),',
                fast_path="sglang.srt.layers.moe.utils:get_num_fused_shared_experts",
            ),
        ),
    ),
    "mhc": OpKernels(
        kernels=(
            Stock(
                kernel=f"{_DG}:tf32_hc_prenorm_gemm",
                pinned_at=f"{_MHC}:mhc_pre",
                pins=(f"{_TOP}.dsv4:MHC_PRE_NUM_SPLITS",),
            ),
            Stock(
                kernel=f"{_MHC}:mhc_pre_big_fuse_with_norm_tilelang",
                pinned_at=f"{_MHC}:mhc_pre",
            ),
        ),
        gates=(
            Gate(
                at=f"{_MHC}:mhc_fused_post_pre",
                anchor="fma_token_threshold = 0 if top_mhc_pre_num_splits() is not None else 32",
                fast_path=f"{_MHC}:mhc_fused_post_pre_fma_tilelang",
            ),
            Gate(
                at=f"{_TOP}.dsv4:validate_dsv4",
                anchor='("FlashInfer mHC", envs.SGLANG_OPT_USE_FLASHINFER_MHC.get()),',
                fast_path="sglang.srt.models.deepseek_v4:_flashinfer_hc_pre",
            ),
        ),
    ),
    "dsv4_attention": OpKernels(
        kernels=(
            *_DSV4_SPARSE,
            Stock(
                kernel="sglang.kernels.ops.attention.dsv4.dequant_k_cache:dequantize_k_cache_paged_dsv4",
                pinned_at=f"{_TOP}.dsv4_attention:_gather_chunk",
            ),
            _ORDERED_TOPK,
        ),
        gates=(
            _DSV4_PAGED_DECODE_GATE,
            Gate(
                at="sglang.srt.layers.attention.dsv4.indexer:C4IndexerBackendMixin.forward_c4_indexer",
                anchor="if self.top_ordered_topk:",
                fast_path="sglang.kernels.ops.attention.dsv4.topk:topk_transform_paged_v2",
            ),
            Gate(
                at=f"{_TOP}.dsv4:validate_dsv4",
                anchor='("the TileLang indexer", envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()),',
                fast_path="sglang.kernels.ops.attention.dsa.tilelang_kernel:tilelang_fp8_paged_mqa_logits",
            ),
        ),
    ),
    "dsv4_mtp_draft": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.layernorm.mhc_head:fused_hc_head",
                pinned_at="sglang.srt.models.deepseek_v4_nextn:DeepseekV4ModelNextN.hc_head",
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.deepseek_v4_nextn:DeepseekV4ModelNextN.hc_head",
                anchor="if is_true_on_policy_enabled():",
                fast_path="sglang.srt.models.deepseek_v4_nextn:DeepseekV4ModelNextN.hc_head",
            ),
        ),
        admitted_by=f"{_TOP}.mtp:validate_mtp",
    ),
    "fp8_b32_linear": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.quantization.fp8_kernel:sglang_per_token_group_quant_fp8",
                pinned_at=f"{_TOP}.fp8.fp8_b32_linear:quantize_rows",
                pins=(f"{_TOP}.fp8.fp8_b32_linear:GROUP",),
            ),
            Stock(
                kernel=f"{_DG}:gemm_nt_f8f8bf16",
                pinned_at=f"{_TOP}.fp8.fp8_b32_linear:linear",
                pins=(f"{_TOP}.fp8.fp8_b32_linear:RECIPE",),
            ),
        ),
        gates=(
            _MAYBE_LINEAR_GATE,
            Gate(
                at=f"{_TOP}.fp8:maybe_linear",
                anchor="or method.block_fp8_as_mxfp8",
                fast_path=f"{_FP8_UTILS}:dispatch_block_fp8_mxfp8_linear",
            ),
        ),
    ),
    "clamped_swiglu_b32": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.activation.silu_and_mul_clamp:silu_and_mul_contig_post_quant",
                pinned_at=f"{_TOP}.fp8.fp8_b32_linear:clamped_swiglu_mlp",
                pins=(
                    f"{_TOP}.fp8.fp8_b32_linear:GROUP",
                    f"{_TOP}.fp8.fp8_b32_linear:SWIGLU_LIMIT",
                ),
            ),
            Stock(
                kernel=f"{_DG}:gemm_nt_f8f8bf16",
                pinned_at=f"{_TOP}.fp8.fp8_b32_linear:clamped_swiglu_mlp",
                pins=(f"{_TOP}.fp8.fp8_b32_linear:RECIPE",),
            ),
        ),
        gates=(_CLAMPED_MLP_GATE,),
    ),
    "mhc_prev": OpKernels(
        kernels=(
            Stock(
                kernel=f"{_MHC}:hc_mix_stats_sinkhorn",
                pinned_at=f"{_TOP}.dsv41_mhc:mix_stats",
            ),
            Stock(kernel=f"{_MHC}:mhc_combine", pinned_at=f"{_TOP}.dsv41_mhc:collapse"),
            Stock(
                kernel="sglang.kernels.ops.layernorm.mhc_post_split_h:mhc_post_split_h",
                pinned_at=f"{_TOP}.dsv41_mhc:post",
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.deepseek_v4_mhc:_combine",
                anchor="if cfg.top:",
                fast_path="sglang.kernels.ops.layernorm.hc_combine_norm:hc_combine_norm",
            ),
            Gate(
                at="sglang.srt.models.deepseek_v4_mhc:_post_fusion",
                anchor="if cfg.top:",
                fast_path="sglang.kernels.ops.layernorm.mhc_mega:mhc_mega_boundary",
            ),
            Gate(
                at="sglang.srt.models.deepseek_v4:DeepseekV4DecoderLayerBase.refresh_mhc_norm_weight_cache",
                anchor="and not self.hc_cfg.top",
                fast_path=f"{_MHC}:split_bf16_hc_weight",
            ),
        ),
    ),
    "engram": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.embeddings.engram_gate:fused_engram_gate",
                pinned_at=f"{_TOP}.dsv41_engram:gate",
            ),
            Stock(
                kernel="sglang.kernels.ops.embeddings.engram_hash:engram_hash_ids",
                pinned_at=f"{_TOP}.dsv41_engram:hash_packed",
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.engram:Engram.forward",
                anchor="if self.top:",
                fast_path="sglang.srt.layers.engram:engram_gate",
            ),
        ),
    ),
    "dsv41_attention": OpKernels(
        kernels=(
            *_DSV4_SPARSE,
            Stock(
                kernel="sglang.kernels.ops.attention.dsv4.fp4_indexer:fp4_index_logits_paged",
                pinned_at=f"{_TOP}.dsv41_attention:select",
            ),
            _ORDERED_TOPK,
            Stock(
                kernel=f"{_BI}:matmul_persistent",
                pinned_at=f"{_TOP}.dsv41_attention:project_bf16",
            ),
        ),
        gates=(
            _DSV4_PAGED_DECODE_GATE,
            Gate(
                at="sglang.srt.models.deepseek_v4:MQALayer.__init__",
                anchor="and not self.top_v41",
                fast_path="sglang.kernels.ops.attention.dsv4.wo_a:fused_rope_wo_a_bf16",
            ),
            Gate(
                at="sglang.srt.models.deepseek_v4:MQALayer._compute_q_b",
                anchor="if self.top_v41:",
                fast_path="sglang.kernels.ops.attention.dsv4.norm_rope:fused_rope_inplace",
            ),
            Gate(
                at="sglang.srt.layers.attention.dsv4.dsv41_sparse:DeepseekV41Indexer.forward_wk",
                anchor="if self.top:",
                fast_path="sglang.kernels.ops.gemm.small_gemm_bf16:n128k512_gemm_bf16",
            ),
            Gate(
                at=f"{_DSV4_BACKEND}:DeepseekV4AttnBackend.forward_low_ratio_sources",
                anchor="if self.top_sparse_attention:",
                fast_path=f"{_DSV4_BACKEND}:DeepseekV4AttnBackend._low_ratio_compress_fused",
            ),
        ),
    ),
    "dsv41_router": OpKernels(
        kernels=(_FP32_GATE_LOGITS, _FUSED_GATE),
        gates=(
            _DSV4_ROUTER_GATE,
            Gate(
                at="sglang.srt.models.deepseek_v2:DeepseekV2MoE.__init__",
                anchor="self.top_vision_routing = vl_correction_bias and is_true_on_policy_enabled()",
                fast_path="sglang.srt.layers.moe.topk:TopK.forward_cuda",
            ),
        ),
    ),
    "dsv41_dspark_draft": OpKernels(
        kernels=(
            TopKernel(
                kernel=f"{_TOP}.dsv41_dspark:race_draft",
                stands_in_for=(
                    "sglang.kernels.ops.speculative.dspark.markov_sample:MarkovSampleStep",
                ),
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.speculative.dspark_components.dspark_draft:_draft_step",
                anchor="if race_key is not None:",
                fast_path="sglang.kernels.ops.speculative.dspark.markov_sample:MarkovSampleStep",
            ),
        ),
        admitted_by=f"{_TOP}.dsv41_dspark:validate_dspark",
    ),
    "hyper_connection": OpKernels(
        kernels=(
            TopKernel(
                kernel=f"{_TOP}.hyper_connection:hc_mix",
                stands_in_for=(
                    "sglang.kernels.ops.gemm.hc_mix:fused_hc_mix",
                    "sglang.kernels.ops.gemm.hc_mix_splitk:hc_mix",
                ),
            ),
            TopKernel(
                kernel=f"{_TOP}.hyper_connection:hc_inject_combine",
                stands_in_for=("sglang.kernels.ops.elementwise.hc_combine:hc_combine",),
            ),
        ),
        gates=(
            Gate(
                at="sglang.srt.layers.hyperconnection:GatedResidual.mix",
                anchor="if self.top_enabled:",
                fast_path="sglang.kernels.ops.gemm.hc_mix:fused_hc_mix",
            ),
            Gate(
                at="sglang.srt.layers.hyperconnection:GatedResidual.combine",
                anchor="if self.top_enabled:",
                fast_path="sglang.kernels.ops.elementwise.hc_combine:hc_combine",
            ),
        ),
    ),
    "ple": OpKernels(
        kernels=(
            TopKernel(
                kernel=f"{_TOP}.ple:",
                stands_in_for=(
                    "sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_reduce",
                    "sglang.kernels.ops.mamba.qwen4_short_conv:fused_qwen4_short_conv_state",
                ),
            ),
            Stock(kernel=f"{_BI}:matmul_persistent"),
        ),
        gates=(
            Gate(
                at="sglang.srt.models.qwen4_exp:Qwen4ExpNGramEmbedding.__init__",
                anchor="self.enable_ngram_hash_fusion and not is_true_on_policy_enabled()",
                fast_path="sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_reduce",
            ),
        ),
    ),
    "qsa": OpKernels(
        kernels=(
            TopKernel(
                kernel=f"{_TOP}.qsa:sparse_attention",
                stands_in_for=(f"{_QSA}.kernel:qsa_sparse_attention",),
            ),
            TopKernel(
                kernel=f"{_TOP}.qsa:select_paged",
                stands_in_for=(f"{_QSA}.mqa:qsa_mqa_decode",),
            ),
            TopKernel(
                kernel=f"{_TOP}.qsa:compress_norm_rope",
                stands_in_for=(f"{_QSA}.kernel:average_pool_qsa_keys",),
            ),
            _ORDERED_TOPK,
            Stock(
                kernel=f"{_BI}:matmul_persistent",
                pinned_at=f"{_TOP}.qsa:index_projection",
            ),
        ),
        gates=(
            Gate(
                at=f"{_QSA}.qsa_indexer:QSAIndexer._use_fused_prep",
                anchor="return not self.top_enabled and (",
                fast_path=f"{_QSA}.qsa_indexer:QSAIndexer._fused_compress_store",
            ),
            Gate(
                at="sglang.srt.layers.attention.qwen_sparse_attn_backend:QwenSparseAttnBackend._try_fused_kv_attention",
                anchor="self.top_enabled",
                fast_path="sglang.srt.layers.attention.qwen_sparse_attn_backend:QwenSparseAttnBackend._forward_trtllm_sparse",
            ),
            Gate(
                at=f"{_TOP}.gdn_contract:validate_gdn_contract",
                anchor='args.qsa_indexer_dtype not in ("auto", "bfloat16")',
                fast_path="sglang.srt.mem_cache.qsa_kv_pool:resolve_qsa_indexer_dtype",
            ),
        ),
    ),
    "qwen4_mtp_draft": OpKernels(
        admitted_by=f"{_TOP}.gdn_contract:validate_gdn_contract"
    ),
    "qwen_vision_tower": OpKernels(
        kernels=(
            _BI_GEMM,
            _FA4_NONCAUSAL,
            Stock(
                kernel="sglang.srt.layers.rotary_embedding.utils:apply_rotary_pos_emb_native_eager"
            ),
        ),
        gates=(
            *_VISION_GATES,
            Gate(
                at="sglang.srt.models.qwen3_vl:Qwen3VLVisionPatchEmbed.forward",
                anchor="if is_true_on_policy_enabled():",
                fast_path="sglang.srt.layers.conv:Conv3dLayer",
            ),
            Gate(
                at="sglang.srt.models.qwen3_vl:Qwen3VLMoeVisionModel._use_vectorized_pos_embed",
                anchor="and not is_true_on_policy_enabled()",
                fast_path="sglang.srt.models.qwen3_vl:Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_vectorized",
            ),
            Gate(
                at="sglang.srt.models.qwen3_vl:Qwen3VLMoeVisionModel.forward",
                anchor="if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get() and not is_true_on_policy_enabled():",
                fast_path="sglang.srt.models.qwen3_vl:Qwen3VLMoeVisionModel.forward_with_cuda_graph",
            ),
        ),
    ),
    "dsv41_vision_tower": OpKernels(
        kernels=(
            _BI_GEMM,
            _FA4_NONCAUSAL,
            Stock(kernel="sglang.srt.models.deepseek_v41_vit:apply_rotary"),
            TopKernel(
                kernel=f"{_TOP}.vision:rms_norm",
                stands_in_for=("sglang.srt.layers.layernorm:RMSNorm.forward_native",),
            ),
        ),
        gates=(
            *_VISION_GATES,
            Gate(
                at="sglang.srt.models.deepseek_v41_vit:_norm",
                anchor="if is_true_on_policy_enabled():",
                fast_path="sglang.srt.layers.layernorm:RMSNorm.forward_native",
            ),
        ),
    ),
    "glm5_next_dsa": OpKernels(
        kernels=(
            Stock(
                kernel="sglang.kernels.ops.attention.dsa.tilelang_kernel:tilelang_sparse_fwd"
            ),
            Stock(
                kernel="flashinfer:top_k_ragged_transform",
                pinned_at=f"{_TOP}.dsa.ordered_topk:ordered_dsa_topk",
                pins=(f"{_TOP}.glm5_next_dsa:GROUP_TOPK",),
            ),
            Stock(
                kernel=f"{_DSA}.kpool_fp8_index:kpool_softmax_rotate_write_cache",
                pinned_at=f"{_TOP}.glm5_next_dsa:write_extend_groups",
            ),
            Stock(
                kernel=f"{_BI}:_matmul_persistent_top",
                pinned_at=f"{_TOP}.glm5_next_dsa:head_gate",
            ),
            Stock(kernel="deep_gemm:fp8_mqa_logits"),
        ),
        gates=(
            Gate(
                at=f"{_DSA}.dsa_indexer_kpool:IndexerKPool._forward_cuda_impl",
                anchor="enable_dual_stream = (",
                fast_path=f"{_DSA}.dsa_indexer_kpool:IndexerKPool._project_and_scale_head_gates",
            ),
            Gate(
                at=f"{_DSA}.dsa_indexer_kpool:IndexerKPool._forward_cuda_impl",
                anchor="skip_logits_computation = not self.top and max_kv_len <= self.index_topk",
                fast_path=f"{_DSA}.dsa_indexer_kpool:IndexerKPool._forward_cuda_skip_logits",
            ),
            Gate(
                at=f"{_DSA}.dsa_indexer_kpool:IndexerKPool._topk_from_kpool_logits",
                anchor="if self.top:",
                fast_path=f"{_DSA}.kpool_fp8_index:topk_from_pooled_history_logits",
            ),
            Gate(
                at=f"{_TOP}.glm5_next_dsa:validate_glm5_next_dsa",
                anchor='server_args.dsa_prefill_backend != "tilelang"',
                fast_path="sglang.srt.layers.attention.dsa_backend:DeepseekSparseAttnBackend._forward_flashmla_sparse",
            ),
        ),
    ),
    "glm5_next_mtp_draft": OpKernels(admitted_by=f"{_TOP}.mtp:validate_mtp"),
}

# Stock fast paths the contract turns off for every op.
CONTRACT_GATES: tuple[Gate, ...] = (
    Gate(
        at=f"{_TOP}.contracts:SGLangTrueOnPolicyContract.policy",
        anchor="disable_mlp_allreduce_fusion=True,",
        fast_path="sglang.srt.layers.flashinfer_comm_fusion:flashinfer_allreduce_residual_rmsnorm",
    ),
    Gate(
        at=f"{_TOP}.contracts:SGLangTrueOnPolicyContract.policy",
        anchor="disable_reduce_scatter=True,",
        fast_path="sglang.srt.layers.dp_attention:attn_tp_reduce_scatter_tensor",
    ),
    Gate(
        at=f"{_TOP}.contracts:SGLangTrueOnPolicyContract.policy",
        anchor="disable_fused_qk_norm_mrope=True,",
        fast_path="sglang.kernels.ops.attention.fused_qknorm_rope:fused_qk_norm_rope",
    ),
    Gate(
        at=f"{_TOP}.contracts:_unqualified_features",
        anchor='("--enable-torch-compile", server_args.enable_torch_compile),',
        fast_path="sglang.srt.compilation.torch_compile_decoration:patch_model",
    ),
    Gate(
        at=f"{_TOP}.contracts:_unqualified_features",
        anchor='("--enable-mixed-chunk", server_args.enable_mixed_chunk),',
        fast_path="sglang.srt.managers.schedule_batch:ScheduleBatch.mix_with_running",
    ),
)

# Per family, the modules under true_on_policy/ that admit its serving program (backends, envs, layouts).
FAMILY_PROGRAMS: dict[str, tuple[str, ...]] = {
    "Qwen3 dense": (f"{_TOP}.attention",),
    "Qwen3 MoE": (f"{_TOP}.attention", f"{_TOP}.qwen_moe", f"{_TOP}.moe_ep"),
    "Qwen GDN dense": (f"{_TOP}.gdn_contract", f"{_TOP}.gdn_chunked", f"{_TOP}.vision"),
    "Qwen GDN MoE": (
        f"{_TOP}.gdn_contract",
        f"{_TOP}.gdn_chunked",
        f"{_TOP}.moe_ep",
        f"{_TOP}.vision",
    ),
    "GLM": (f"{_TOP}.glm", f"{_TOP}.glm_moe", f"{_TOP}.fp8"),
    "Qwen3.8-Flash-Next": (
        f"{_TOP}.gdn_contract",
        f"{_TOP}.gdn_chunked",
        f"{_TOP}.qsa",
        f"{_TOP}.ple",
    ),
    "GLM-5.3-Flash": (
        f"{_TOP}.glm5_next",
        f"{_TOP}.kda_chunked",
        f"{_TOP}.glm5_next_dsa",
    ),
    "DeepSeek-V4": (f"{_TOP}.dsv4", f"{_TOP}.fp8"),
    "DeepSeek-V4.1": (f"{_TOP}.dsv41", f"{_TOP}.fp8", f"{_TOP}.vision"),
}
