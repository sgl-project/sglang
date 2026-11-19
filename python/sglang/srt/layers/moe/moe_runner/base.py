from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, TypeGuard

import torch

from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    MoeRunnerBackendLike,
    RoutingMethodType,
)
from sglang.srt.runtime_context import get_forward

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonRunnerCore,
        TritonRunnerInput,
        TritonRunnerOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        CombineInputFormat,
        DispatchOutput,
        DispatchOutputFormat,
    )


def moe_output_buffer_ctx(buf: torch.Tensor):
    """Provide the MoE output buffer for the current forward scope."""

    return get_forward().scoped(moe_output_buffer=buf)


@dataclass
class MoeRunnerConfig:
    # MoE parameters
    num_experts: Optional[int] = None
    num_local_experts: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size_per_partition: Optional[int] = None
    layer_id: Optional[int] = None
    top_k: Optional[int] = None
    num_fused_shared_experts: Optional[int] = None
    params_dtype: Optional[torch.dtype] = None
    routing_method_type: Optional[RoutingMethodType] = None

    # None denotes a direct kernel with standard TopK and local expert IDs.
    runner_backend: Optional[MoeRunnerBackendLike] = None

    # Runner configuration
    activation: str = "silu"
    is_gated: bool = True
    apply_router_weight_on_input: bool = False
    inplace: bool = True
    no_combine: bool = False
    # With no_combine: still apply the router weight, leaving only the top-k sum to
    # the caller (triton and marlin runners).
    no_combine_keep_router_weight: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_beta: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None
    swiglu_limit: Optional[float] = None
    # Whether gate/up weights are stored interleaved (vs split). Only the
    # silu+is_gated swiglu path consumes it (interleaved -> swiglu_gpt_oss_*,
    # otherwise chunk gate/up then apply alpha/limit).
    gate_up_interleaved: bool = True
    layer: Optional[torch.nn.Module] = None
    use_tp_all_gather_activation: bool = False
    # Request FP32 SiLU/multiply intermediates until FP8 quantization.
    # False preserves backend defaults, including their existing FP32 paths.
    silu_mul_keep_fp32: bool = False
    # Resolved token threshold for Dual-Runner (0 disables; set once at runner
    # init so forward hot paths avoid reading environment variables).
    dual_runner_threshold: int = 0

    # The MoE placement a fused func needs is the layer's own, not the
    # deployment's: DWDP's `FusedMoE.bind_full_expert_weights` collapses a
    # layer's EP view after load. Read it off the layer so there is one
    # address to keep correct.
    @property
    def moe_tp_size(self) -> int:
        return self._layer_placement("moe_tp_size")

    @property
    def moe_tp_rank(self) -> int:
        return self._layer_placement("moe_tp_rank")

    @property
    def moe_ep_size(self) -> int:
        return self._layer_placement("moe_ep_size")

    @property
    def moe_ep_rank(self) -> int:
        return self._layer_placement("moe_ep_rank")

    def _layer_placement(self, name: str) -> int:
        if self.layer is None:
            raise AttributeError(
                f"{name} is the layer's MoE placement, but this runner config "
                "has no layer. A layer that runs fused MoE funcs states itself "
                "on its runner config."
            )
        return getattr(self.layer, name)


@dataclass
class RunnerInput(ABC):
    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerInput]:
        return self.runner_backend == MoeRunnerBackend.TRITON


class RunnerOutput(ABC):
    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerOutput]:
        return self.runner_backend == MoeRunnerBackend.TRITON


@dataclass
class MoeQuantInfo(ABC):
    """Moe quantization data."""

    pass


class MoeRunnerCore(ABC):
    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    @abstractmethod
    def run(
        self,
        runner_input: RunnerInput,
        quant_info: MoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> RunnerOutput:
        pass

    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerCore]:
        return self.runner_backend == MoeRunnerBackend.TRITON


class DispatchMoeRunnerCore(ABC):
    """Runner core that consumes the standard dispatch representation directly."""

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackendLike: ...

    @abstractmethod
    def run_from_dispatch(
        self,
        dispatch_output: DispatchOutput,
        quant_info: MoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks: Any = None,
    ) -> CombineInput: ...


class FusedOpPool:
    _fused_funcs: dict[str, Callable] = {}

    @classmethod
    def register_fused_func(
        cls, a2a_backend_name: str, runner_backend_name: str, fused_func: Callable
    ):
        # Key by the canonical value, which is what lookups pass, so a
        # deprecated alias such as "flashinfer" still registers correctly.
        a2a_backend_name = MoeA2ABackend(a2a_backend_name).value
        key = (a2a_backend_name, runner_backend_name)
        if key in cls._fused_funcs:
            raise ValueError(
                f"Fused function for {a2a_backend_name} to {runner_backend_name} is already registered."
            )
        assert MoeA2ABackend(a2a_backend_name), (
            f"Invalid dispatch name: {a2a_backend_name}"
        )
        assert MoeRunnerBackend(runner_backend_name), (
            f"Invalid runner name: {runner_backend_name}"
        )
        cls._fused_funcs[key] = fused_func

    @classmethod
    def get_fused_func(cls, dispatch_name: str, runner_name: str) -> Optional[Callable]:
        key = (dispatch_name, runner_name)
        fused_func = cls._fused_funcs.get(key)
        return fused_func


class PermuteMethodPool:
    _pre_permute_methods: dict[
        Tuple[DispatchOutputFormat, MoeRunnerBackend], Callable
    ] = {}
    _post_permute_methods: dict[
        Tuple[MoeRunnerBackend, CombineInputFormat], Callable
    ] = {}

    @classmethod
    def register_pre_permute(
        cls,
        dispatch_output_name: str,
        runner_backend_name: str,
        permute_func: Callable,
    ):
        """
        Register a customized pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.

        :param dispatch_output_name: The DispatchOutputFormat name.
        :param runner_backend_name: The MoeRunnerBackend name.
        :param permute_func: The permute function to register.
        """
        # TODO: check if registration is valid
        key = (dispatch_output_name, runner_backend_name)
        if key in cls._pre_permute_methods:
            raise ValueError(
                f"Pre-permute method for {dispatch_output_name} to {runner_backend_name} is already registered."
            )
        cls._pre_permute_methods[key] = permute_func

    @classmethod
    def register_post_permute(
        cls,
        runner_backend_name: str,
        combine_input_name: str,
        permute_func: Callable,
    ):
        """
        Register a customized post-permute function for the given MoeRunnerBackend and CombineInputFormat.

        :param runner_backend_name: The MoeRunnerBackend name.
        :param combine_input_name: The CombineInputFormat name.
        :param permute_func: The permute function to register.
        """
        # TODO: check if registration is valid
        key = (runner_backend_name, combine_input_name)
        if key in cls._post_permute_methods:
            raise ValueError(
                f"Post-permute method for {runner_backend_name} to {combine_input_name} is already registered."
            )
        cls._post_permute_methods[key] = permute_func

    @classmethod
    def get_pre_permute(
        cls,
        dispatch_output_format: DispatchOutputFormat,
        runner_input_format: MoeRunnerBackend,
    ) -> Callable:
        """
        Retrieve the pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.

        :param dispatch_output_format: The DispatchOutputFormat type.
        :param runner_input_format: The MoeRunnerBackend type.
        :return: The registered permute function or None if not found.
        """
        key = (dispatch_output_format, runner_input_format)
        pre_permute_func = cls._pre_permute_methods.get(key)
        assert pre_permute_func is not None, (
            f"Pre-permute function for {dispatch_output_format} to {runner_input_format} is not registered"
        )
        return pre_permute_func

    @classmethod
    def get_post_permute(
        cls,
        runner_output_format: MoeRunnerBackend,
        combine_input_format: CombineInputFormat,
    ) -> Callable:
        """
        Retrieve the post-permute function for the given MoeRunnerBackend and CombineInputFormat.

        :param runner_output_format: The MoeRunnerBackend type.
        :param combine_input_format: The CombineInputFormat type.
        :return: The registered permute function or None if not found.
        """
        key = (runner_output_format, combine_input_format)
        post_permute_func = cls._post_permute_methods.get(key)
        assert post_permute_func is not None, (
            f"Post-permute function for {runner_output_format} to {combine_input_format} is not registered"
        )
        return post_permute_func


def register_fused_func(
    a2a_backend_name: str,
    runner_backend_name: str,
) -> Callable:
    """
    Decorator to register a fused function for the given DispatchOutputFormat and MoeRunnerBackend.

    :param a2a_backend_name: The A2A backend name.
    :param runner_backend_name: The MoeRunnerBackend name.
    :return: The decorator function.
    """

    def decorator(fused_func: Callable):
        FusedOpPool.register_fused_func(
            a2a_backend_name, runner_backend_name, fused_func
        )
        return fused_func

    return decorator


def register_pre_permute(
    dispatch_output_name: str,
    runner_backend_name: str,
) -> Callable:
    """
    Decorator to register a pre-permute function for the given DispatchOutputFormat and MoeRunnerBackend.

    :param dispatch_output_name: The DispatchOutputFormat name.
    :param runner_backend_name: The MoeRunnerBackend name.
    :return: The decorator function.
    """

    def decorator(
        permute_func: Callable[
            [DispatchOutput, MoeQuantInfo, MoeRunnerConfig, dict], RunnerInput
        ],
    ) -> Callable:
        PermuteMethodPool.register_pre_permute(
            dispatch_output_name, runner_backend_name, permute_func
        )
        return permute_func

    return decorator


def register_post_permute(
    runner_backend_name: str,
    combine_input_name: str,
) -> Callable:
    """
    Decorator to register a post-permute function for the given MoeRunnerBackend and CombineInputFormat.

    :param runner_backend_name: The MoeRunnerBackend name.
    :param combine_input_name: The CombineInputFormat name.
    :return: The decorator function.
    """

    def decorator(
        permute_func: Callable[
            [RunnerOutput, MoeQuantInfo, MoeRunnerConfig, dict], CombineInput
        ],
    ) -> Callable:
        PermuteMethodPool.register_post_permute(
            runner_backend_name, combine_input_name, permute_func
        )
        return permute_func

    return decorator


def resolve_moe_dual_runner_threshold(
    raw_threshold: int,
    runner_config: MoeRunnerConfig,
    *,
    is_fp8: bool,
    scale_ue8m0: Optional[bool] = None,
) -> int:
    """Resolve `SGLANG_MOE_DUAL_RUNNER_THRESHOLD` into a concrete token cutoff.

    `0` disables Dual-Runner, `> 0` preserves an explicit cutoff, and `< 0`
    selects an M_eff-based crossover where DeepGEMM compact's block_e=128
    padding inflation (`1 + 128 / M_eff`) becomes lower than Triton's WGMMA gap.
    """
    if raw_threshold >= 0:
        return raw_threshold

    num_local_experts = max(1, int(runner_config.num_local_experts or 1))
    top_k = max(1, int(runner_config.top_k or 1))
    if is_fp8:
        if scale_ue8m0 is None:
            from sglang.srt.layers import deep_gemm_wrapper

            scale_ue8m0 = bool(deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0)
        # On SM90 (!DEEPGEMM_SCALE_UE8M0), DEEPGEMM_NEED_TMA_ALIGNED_SCALES adds
        # two unfused quant + tma_align_input_scale HBM transpose passes per layer,
        # pushing the DeepGEMM crossover to M_eff=1024 (16384 tokens on E=128, k=8).
        target_m_eff = 256 if scale_ue8m0 else 1024
        min_tokens = 1024 if scale_ue8m0 else 2048
    else:
        target_m_eff = 512
        min_tokens = 1024
    return max(min_tokens, (target_m_eff * num_local_experts + top_k - 1) // top_k)


def get_moe_dual_runner_unsupported_reason(
    runner_config: MoeRunnerConfig,
    *,
    with_bias: bool,
) -> Optional[str]:
    """Return a short reason if shared Dual-Runner prerequisites are unmet."""
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.runtime_context import get_exec, get_lora
    from sglang.srt.utils import get_device_sm, is_cuda

    if not get_moe_runner_backend().is_auto():
        return f"moe_runner_backend is {get_moe_runner_backend().name}, not auto"
    if not get_moe_a2a_backend().is_none():
        return f"moe_a2a_backend is {get_moe_a2a_backend().name}, not none"
    if not is_cuda() or get_device_sm() not in (90, 100):
        return (
            f"requires SM90/SM100 CUDA "
            f"(got sm={get_device_sm() if is_cuda() else 'non-cuda'})"
        )
    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return "ENABLE_JIT_DEEPGEMM is False"
    if not runner_config.is_gated or runner_config.activation != "silu":
        return (
            f"unsupported activation "
            f"({runner_config.activation}, is_gated={runner_config.is_gated})"
        )
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
        or runner_config.swiglu_limit is not None
    ):
        return "custom SwiGLU alpha/clamp/limit is unsupported"
    if (
        runner_config.apply_router_weight_on_input
        or runner_config.no_combine
        or with_bias
    ):
        return "apply_router_weight_on_input, no_combine, or expert bias is unsupported"

    try:
        if get_lora().enable_lora or get_lora().lora_paths:
            return "LoRA is enabled"
    except Exception:
        pass
    try:
        moe_exec = get_exec().moe
        if moe_exec.enable_eplb or moe_exec.enable_paged_experts:
            return "EPLB or paged experts enabled"
    except Exception:
        pass

    hidden_size = int(runner_config.hidden_size or 0)
    if hidden_size <= 0 or hidden_size % 128 != 0:
        return f"hidden_size={hidden_size} is not divisible by 128"
    return None
