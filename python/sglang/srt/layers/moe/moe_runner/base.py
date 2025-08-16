from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, TypeGuard

import torch

from sglang.srt.layers.moe.token_dispatcher import (
    DispatchOutput,
    CombineInput,
    CombineInputFormat,
    DispatchOutputFormat,
)

from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.triton import (
        TritonRunnerInput,
        TritonRunnerOutput,
    )
    from sglang.srt.layers.quantization.base_config import MoeQuantInfo
    from sglang.srt.layers.moe.utils import MoeRunnerBackend


@dataclass
class MoeRunnerConfig:

    # MoE parameters
    num_experts: Optional[int] = None
    num_local_experts: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    layer_id: Optional[int] = None
    top_k: Optional[int] = None
    num_fused_shared_experts: Optional[int] = None
    params_dtype: Optional[torch.dtype] = None

    # Runner configuration
    activation: str = "silu"
    apply_router_weight_on_input: bool = False
    inplace: bool = True
    no_combine: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None


class RunnerInputFormat(Enum):
    TRITON = "triton"


class RunnerOutputFormat(Enum):
    TRITON = "triton"


class RunnerInput(ABC):
    @abstractmethod
    def get_format(self) -> RunnerInputFormat:
        pass

    def format_is_triton(self) -> TypeGuard[TritonRunnerInput]:
        return self.get_format() == RunnerInputFormat.TRITON


class RunnerOutput(ABC):
    @abstractmethod
    def get_format(self) -> RunnerOutputFormat:
        pass

    def format_is_triton(self) -> TypeGuard[TritonRunnerOutput]:
        return self.get_format() == RunnerOutputFormat.TRITON


class MoeRunnerCore(ABC):

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    @abstractmethod
    def run(self, runner_input):
        pass

    @abstractmethod
    @property
    def input_format(cls) -> RunnerInputFormat:
        pass

    @abstractmethod
    @property
    def output_format(cls) -> RunnerOutputFormat:
        pass


class PermuteMethodPool:

    _pre_permute_methods: dict[
        Tuple[DispatchOutputFormat, RunnerInputFormat], Callable
    ] = {}
    _post_permute_methods: dict[
        Tuple[RunnerOutputFormat, CombineInputFormat], Callable
    ] = {}

    @classmethod
    def register_pre_permute(
        cls,
        dispatch_output_format: DispatchOutputFormat,
        runner_input_format: RunnerInputFormat,
        permute_func: Callable,
    ):
        """
        Register a customized pre-permute function for the given DispatchOutputFormat and RunnerInputFormat.

        :param dispatch_output_format: The DispatchOutputFormat type.
        :param runner_input_format: The RunnerInputFormat type.
        :param permute_func: The permute function to register.
        """
        key = (dispatch_output_format, runner_input_format)
        if key in cls._pre_permute_methods:
            raise ValueError(f"Pre-permute method for {key} is already registered.")
        cls._pre_permute_methods[key] = permute_func

    @classmethod
    def register_post_permute(
        cls,
        runner_output_format: RunnerOutputFormat,
        combine_input_format: CombineInputFormat,
        permute_func: Callable,
    ):
        """
        Register a customized post-permute function for the given RunnerOutputFormat and CombineInputFormat.

        :param runner_output_format: The RunnerOutputFormat type.
        :param combine_input_format: The CombineInputFormat type.
        :param permute_func: The permute function to register.
        """
        key = (runner_output_format, combine_input_format)
        if key in cls._post_permute_methods:
            raise ValueError(f"Post-permute method for {key} is already registered.")
        cls._post_permute_methods[key] = permute_func

    @classmethod
    def get_pre_permute(
        cls,
        dispatch_output_format: DispatchOutputFormat,
        runner_input_format: RunnerInputFormat,
    ) -> Callable:
        """
        Retrieve the pre-permute function for the given DispatchOutputFormat and RunnerInputFormat.

        :param dispatch_output_format: The DispatchOutputFormat type.
        :param runner_input_format: The RunnerInputFormat type.
        :return: The registered permute function or None if not found.
        """
        key = (dispatch_output_format, runner_input_format)
        return cls._pre_permute_methods.get(key)

    @classmethod
    def get_post_permute(
        cls,
        runner_output_format: RunnerOutputFormat,
        combine_input_format: CombineInputFormat,
    ) -> Callable:
        """
        Retrieve the post-permute function for the given RunnerOutputFormat and CombineInputFormat.

        :param runner_output_format: The RunnerOutputFormat type.
        :param combine_input_format: The CombineInputFormat type.
        :return: The registered permute function or None if not found.
        """
        key = (runner_output_format, combine_input_format)
        return cls._post_permute_methods.get(key)


def register_pre_permute(
    dispatch_output_format: DispatchOutputFormat,
    runner_input_format: RunnerInputFormat,
) -> Callable:
    """
    Decorator to register a pre-permute function for the given DispatchOutputFormat and RunnerInputFormat.

    :param dispatch_output_format: The DispatchOutputFormat type.
    :param runner_input_format: The RunnerInputFormat type.
    :return: The decorator function.
    """

    def decorator(permute_func: Callable) -> Callable:
        PermuteMethodPool.register_pre_permute(
            dispatch_output_format, runner_input_format, permute_func
        )
        return permute_func

    return decorator


def register_post_permute(
    runner_output_format: RunnerOutputFormat,
    combine_input_format: CombineInputFormat,
) -> Callable:
    """
    Decorator to register a post-permute function for the given RunnerOutputFormat and CombineInputFormat.

    :param runner_output_format: The RunnerOutputFormat type.
    :param combine_input_format: The CombineInputFormat type.
    :return: The decorator function.
    """

    def decorator(permute_func: Callable) -> Callable:
        PermuteMethodPool.register_post_permute(
            runner_output_format, combine_input_format, permute_func
        )
        return permute_func

    return decorator


class MoeRunner:

    def __init__(self, runner_backend: MoeRunnerBackend, config: MoeRunnerConfig):
        self.runner_backend = runner_backend
        self.config = config

        if runner_backend.is_triton():
            self.runner_core = TritonRunnerCore(config)
        else:
            raise NotImplementedError(f"Unsupported runner backend: {runner_backend}")

    def forward(self, dispatch_output: DispatchOutput, quant_info: MoeQuantInfo) -> CombineInput:
        dispatch_output_format = dispatch_output.format
        runner_input_format = self.runner_core.input_format
        pre_permute_func = PermuteMethodPool.get_pre_permute(dispatch_output_format, runner_input_format)

        runner_input = pre_permute_func(dispatch_output)
        runner_output = self.runner_core.run(runner_input, quant_info)

        runner_output_format = self.runner_core.output_format
        combine_input_format = CombineInputFormat.STANDARD
        post_permute_func = PermuteMethodPool.get_post_permute(runner_output_format, combine_input_format)
        combine_input = post_permute_func(runner_output)

        return combine_input
