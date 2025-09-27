import logging
from importlib.util import find_spec
from typing import Optional

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from sglang.srt.configs.compilation_config import CompilationConfig
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.utils import (
    direct_register_custom_op,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()


if _is_cuda:
    if _is_flashinfer_available:
        from flashinfer.norm import fused_add_rmsnorm
    else:
        from sgl_kernel import fused_add_rmsnorm

from .inductor_pass import enable_fake_mode
from .sglang_inductor_pass import SglangInductorPass, SglangPatternMatcherPass

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
        4: MiB,  # 1MB
        6: MiB // 2,  # 512KB
        8: MiB // 2,  # 512KB
    }

    # opt for a more conservative default value
    # when world size is not in _FI_MAX_SIZES
    _DEFAULT_FI_MAX_SIZE = MiB // 2

    def call_trtllm_fused_allreduce_norm(
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
        norm_out: Optional[torch.Tensor] = None,
        quant_out: Optional[torch.Tensor] = None,
        scale_out: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
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
            # TODO
            pass

    def call_trtllm_fused_allreduce_norm_fake(
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
        norm_out: Optional[torch.Tensor] = None,
        quant_out: Optional[torch.Tensor] = None,
        scale_out: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="flashinfer_allreduce_residual_rmsnorm",
        op_func=call_trtllm_fused_allreduce_norm,
        mutates_args=[
            "allreduce_in",
            "residual",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
        fake_impl=call_trtllm_fused_allreduce_norm_fake,
    )
    flashinfer_allreduce_residual_rmsnorm = (
        torch.ops.sglang.flashinfer_allreduce_residual_rmsnorm.default
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

    def get_inputs(self):
        input = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [
            residual,
            input,
            weight,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = auto_functionalized_v2(
                torch.ops.sgl_kernel.fused_add_rmsnorm.default,
                input=allreduce_output,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )
            # input, residual
            return rms[1], rms[2]

        def replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ):
            allreduce = auto_functionalized_v2(
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


class AllReduceFusionPass(SglangPatternMatcherPass):
    # TODO(yuan-luo): replace with SglangConfig
    def __init__(
        self,
        compilation_config: CompilationConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ):
        super().__init__(compilation_config, model_config, device_config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            return
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="all_reduce_fusion_pass"
        )
        if model_config is None:
            return
        self.hidden_dim = model_config.get_hidden_size()
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
                "Flashinfer allreduce fusion is not " "supported for world size %s",
                self.tp_size,
            )
            return
        max_num_token = min(
            _FI_MAX_SIZES.get(self.tp_size, _DEFAULT_FI_MAX_SIZE)
            // (self.hidden_dim * self.tp_size * (4 if use_fp32_lamport else 2)),
            compilation_config.pass_config.fi_allreduce_fusion_max_token_num,
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
            fuse_rms_quant=compilation_config.pass_config.enable_fusion,
        )

        self.register_patterns()

    @enable_fake_mode
    def register_patterns(self):
        for epsilon in [1e-5, 1e-6]:
            AllReduceFusedAddRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.allreduce_params,
            ).register(self.patterns)

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
