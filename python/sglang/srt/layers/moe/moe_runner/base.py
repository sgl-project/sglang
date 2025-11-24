from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, TypeGuard

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

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

    # Runner configuration
    activation: str = "silu"
    is_gated: bool = True
    apply_router_weight_on_input: bool = False
    inplace: bool = True
    no_combine: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None


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
        self, runner_input: RunnerInput, quant_info: MoeQuantInfo, running_state: dict
    ) -> RunnerOutput:
        pass

    @property
    @abstractmethod
    def runner_backend(self) -> MoeRunnerBackend: ...

    def runner_backend_is_triton(self) -> TypeGuard[TritonRunnerCore]:
        return self.runner_backend == MoeRunnerBackend.TRITON


class FusedOpPool:
    _fused_funcs: dict[str, Callable] = {}

    @classmethod
    def register_fused_func(
        cls, a2a_backend_name: str, runner_backend_name: str, fused_func: Callable
    ):
        key = (a2a_backend_name, runner_backend_name)
        if key in cls._fused_funcs:
            raise ValueError(
                f"Fused function for {a2a_backend_name} to {runner_backend_name} is already registered."
            )
        assert MoeA2ABackend(
            a2a_backend_name
        ), f"Invalid dispatch name: {a2a_backend_name}"
        assert MoeRunnerBackend(
            runner_backend_name
        ), f"Invalid runner name: {runner_backend_name}"
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
        assert (
            pre_permute_func is not None
        ), f"Pre-permute function for {dispatch_output_format} to {runner_input_format} is not registered"
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
        assert (
            post_permute_func is not None
        ), f"Post-permute function for {runner_output_format} to {combine_input_format} is not registered"
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
