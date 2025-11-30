import logging
from importlib.util import find_spec

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from sglang.srt.compilation.inductor_pass import enable_fake_mode
from sglang.srt.compilation.matcher_utils import MatcherFusedAddRMSNorm, MatcherRMSNorm
from sglang.srt.compilation.sglang_config import SGLangConfig
from sglang.srt.compilation.sglang_inductor_pass import SGLangPatternMatcherPass
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.utils import direct_register_custom_op

if find_spec("flashinfer"):
    try:
        import flashinfer.comm as flashinfer_comm

        flashinfer_comm = (
            flashinfer_comm
            if hasattr(flashinfer_comm, "trtllm_allreduce_fusion")
            else None
        )
    except ImportError:
        flashinfer_comm = None
else:
    flashinfer_comm = None

logger = logging.getLogger(__name__)


class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str):
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


if flashinfer_comm is not None:
    _FI_WORKSPACE_TENSOR = None

    MiB = 1024 * 1024
    # Max size of the input tensor per world size
    # to use flashinfer fused allreduce
    _FI_MAX_SIZES = {
        2: 64 * MiB,  # 64MB
        4: 2 * MiB,  # 2MB
        8: MiB // 2,  # 512KB
    }

    # opt for a more conservative default value
    # when world size is not in _FI_MAX_SIZES
    _DEFAULT_FI_MAX_SIZE = MiB // 2

    def call_flashinfer_fused_allreduce_norm(
        allreduce_in: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        world_rank: int,
        world_size: int,
        launch_with_pdl: bool,
        trigger_completion_at_end: bool,
        fp32_acc: bool,
        max_token_num: int,
        pattern_code: int,
        fuse_rms_quant: bool,
        norm_out: torch.Tensor | None = None,
        quant_out: torch.Tensor | None = None,
        scale_out: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ) -> None:
        num_tokens, hidden_size = allreduce_in.shape
        element_size = allreduce_in.element_size()
        current_tensor_size = num_tokens * hidden_size * element_size
        max_fusion_size = max_token_num * hidden_size * element_size
        use_flashinfer = current_tensor_size <= min(
            _FI_MAX_SIZES.get(world_size, _DEFAULT_FI_MAX_SIZE),
            max_fusion_size,
        )
        if use_flashinfer:
            assert (
                _FI_WORKSPACE_TENSOR is not None
            ), "Flashinfer must be enabled when using flashinfer"
            if norm_out is None:
                norm_out = allreduce_in
                residual_out = residual
            else:
                # return residual_out as allreduce_out with zeroed residual_in
                # as flashinfer does not support rms_norm
                # and allreduce_out together
                residual_out = allreduce_in
            # For the sizes that are smaller than the max size,
            # we only use flashinfer one shot allreduce
            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=allreduce_in,
                token_num=allreduce_in.shape[0],
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                world_rank=world_rank,
                world_size=world_size,
                hidden_dim=allreduce_in.shape[-1],
                workspace_ptrs=_FI_WORKSPACE_TENSOR,
                launch_with_pdl=launch_with_pdl,
                use_oneshot=True,
                trigger_completion_at_end=trigger_completion_at_end,
                fp32_acc=fp32_acc,
                pattern_code=pattern_code,
                allreduce_out=None,
                quant_out=quant_out,
                scale_out=scale_out,
                # in sglang we only support swizzled layout
                layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
                scale_factor=scale_factor,
            )
        else:
            allreduce_out = tensor_model_parallel_all_reduce(allreduce_in)
            if scale_factor is not None and scale_out is None and fuse_rms_quant:
                # Do fused rms norm static fp8 quant fused op
                if norm_out is None:
                    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                        quant_out,
                        allreduce_out,
                        residual,
                        rms_gamma,
                        scale_factor,
                        rms_eps,
                    )
                else:
                    torch.ops._C.rms_norm_static_fp8_quant(
                        quant_out, allreduce_out, rms_gamma, scale_factor, rms_eps
                    )
            else:
                if norm_out is None:
                    torch.ops._C.fused_add_rms_norm(
                        allreduce_out, residual, rms_gamma, rms_eps
                    )
                    norm_out = allreduce_out
                else:
                    torch.ops._C.rms_norm(norm_out, allreduce_out, rms_gamma, rms_eps)
                if scale_factor is not None:
                    if scale_out is not None:
                        torch.ops._C.scaled_fp4_quant(
                            quant_out, norm_out, scale_out, scale_factor
                        )
                    else:
                        torch.ops._C.static_scaled_fp8_quant(
                            quant_out, norm_out, scale_factor
                        )
            if scale_factor is None or norm_out is not None:
                # we need to return allreduce output
                # in cases of non quant fused AR + RMS norm
                # and fused AR + RMS norm + quant without fused add
                allreduce_in.copy_(allreduce_out)

    def call_flashinfer_fused_allreduce_norm_fake(
        allreduce_in: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        world_rank: int,
        world_size: int,
        launch_with_pdl: bool,
        trigger_completion_at_end: bool,
        fp32_acc: bool,
        max_token_num: int,
        pattern_code: int,
        fuse_rms_quant: bool,
        norm_out: torch.Tensor | None = None,
        quant_out: torch.Tensor | None = None,
        scale_out: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="flashinfer_trtllm_fused_allreduce_norm",
        op_func=call_flashinfer_fused_allreduce_norm,
        mutates_args=[
            "allreduce_in",
            "residual",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
        fake_impl=call_flashinfer_fused_allreduce_norm_fake,
    )
    flashinfer_trtllm_fused_allreduce_norm = (
        # TODO(yuan-luo): SGLang kernel
        torch.ops.sglang.flashinfer_trtllm_fused_allreduce_norm.default
    )


class FlashInferFusedAllReduceParams:
    """Parameters for FlashInfer fused allreduce operations."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        use_fp32_lamport: bool = False,
        max_token_num: int = 1024,
        fuse_rms_quant: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.use_fp32_lamport = use_fp32_lamport
        self.trigger_completion_at_end = True
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.use_oneshot = False
        self.max_token_num = max_token_num
        self.fuse_rms_quant = fuse_rms_quant

    def get_trtllm_fused_allreduce_kwargs(self):
        return {
            "world_rank": self.rank,
            "world_size": self.world_size,
            "launch_with_pdl": self.launch_with_pdl,
            "trigger_completion_at_end": self.trigger_completion_at_end,
            "fp32_acc": self.fp32_acc,
            "max_token_num": self.max_token_num,
            "fuse_rms_quant": self.fuse_rms_quant,
        }


class AllReduceRMSNormPattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (without residual)
    with fused flashinfer implementation.
    Applies to allreduce + rmsnorm before attn in the first Transformer block.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
        allreduce_params: FlashInferFusedAllReduceParams,
    ):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.rmsnorm_matcher = MatcherRMSNorm(epsilon)

    def get_inputs(self):
        input, weight = self.rmsnorm_matcher.inputs()

        # input goes through allreduce first, always 16-bit
        return [input.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input: torch.Tensor, weight: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = self.rmsnorm_matcher(allreduce_output, weight)

            return rms, allreduce_output

        def replacement(input: torch.Tensor, weight: torch.Tensor):
            residual = torch.zeros_like(input)
            rms_result = torch.empty_like(input)
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=rms_result,
                quant_out=None,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # rms_result, allreduce_in
            return allreduce[3], allreduce[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusedAddRMSNormPattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (with residual)
    with fused flashinfer implementation.
    Applies to o_proj + rmsnorm after attn and mlp + rmsnorm before attn.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
        allreduce_params: FlashInferFusedAllReduceParams,
    ):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)

    def get_inputs(self):
        input, residual, weight = self.rmsnorm_matcher.inputs()

        # input goes through allreduce first, always 16-bit
        return [residual, input.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = self.rmsnorm_matcher(allreduce_output, weight, residual)
            return rms, residual

        def replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ):
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                quant_out=None,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # allreduce_in, residual
            return allreduce[1], allreduce[2]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        # Same pattern, but only return the output and not residual
        # (helpful for end of graph where residual is not used again)
        first_return_only = lambda fn: lambda a, b, c: fn(a, b, c)[0]

        pm.register_replacement(
            first_return_only(pattern),
            first_return_only(replacement),
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AllReduceFusionPass(SGLangPatternMatcherPass):
    def __init__(self, config: SGLangConfig):
        super().__init__(config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            return
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="all_reduce_fusion_pass"
        )
        if config.model_config is None:
            return
        self.hidden_dim = config.model_config.get_hidden_size()
        self.group = get_tp_group().device_group
        rank = get_tensor_model_parallel_rank()
        use_fp32_lamport = self.model_dtype == torch.float32
        if flashinfer_comm is None:
            logger.warning(
                "Flashinfer is not installed or comm module not found, "
                "skipping allreduce fusion pass"
            )
            return
        # Check if the world size is supported
        if self.tp_size not in _FI_MAX_SIZES:
            logger.warning(
                "Flashinfer allreduce fusion is not supported for world size %s",
                self.tp_size,
            )
            return
        max_num_token = min(
            _FI_MAX_SIZES.get(self.tp_size, _DEFAULT_FI_MAX_SIZE)
            // (self.hidden_dim * self.tp_size * (4 if use_fp32_lamport else 2)),
            config.compilation_config.pass_config.fi_allreduce_fusion_max_token_num,
        )
        self.ipc_handles, workspace_tensor = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=rank,
                tp_size=self.tp_size,
                max_token_num=max_num_token,
                hidden_dim=self.hidden_dim,
                group=self.group,
                use_fp32_lamport=use_fp32_lamport,
            )
        )

        global _FI_WORKSPACE_TENSOR
        _FI_WORKSPACE_TENSOR = workspace_tensor
        self.allreduce_params = FlashInferFusedAllReduceParams(
            rank=rank,
            world_size=self.tp_size,
            use_fp32_lamport=use_fp32_lamport,
            max_token_num=max_num_token,
            # fuse rms norm static fp8 quant fused op
            # in fallback path, when we don't use flashinfer
            fuse_rms_quant=config.compilation_config.pass_config.enable_fusion,
        )

        self.register_patterns()

    @enable_fake_mode
    def register_patterns(self):
        for epsilon in [1e-5, 1e-6]:
            # TODO(yuan-luo):
            # register AllReduceFusedRMSNormStaticQuantFP8Pattern
            # register AllReduceFusedAddRMSNormStaticQuantFP8Pattern
            # register AllReduceFusedRMSNormStaticQuantNVFP4Pattern
            # register AllReduceFusedAddRMSNormStaticQuantNVFP4Pattern
            AllReduceRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.allreduce_params,
            ).register(self.patterns)
            # AllReduceFusedAddRMSNormPattern(
            #     epsilon,
            #     self.model_dtype,
            #     self.device,
            #     self.allreduce_params,
            # ).register(self.patterns)

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    def __call__(self, graph: fx.Graph):
        if self.disabled:
            logger.debug("AllReduceFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def __del__(self):
        if getattr(self, "disabled", True):
            return
        if flashinfer_comm is not None:
            flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                self.ipc_handles, self.group
            )
