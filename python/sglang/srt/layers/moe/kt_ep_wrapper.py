# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import inspect
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if server_args.kt_weight_path is None:
        return None

    # Try to get num_layers from model config
    num_layers = None
    try:
        hf_config = server_args.get_hf_config()
        num_layers = getattr(hf_config, "num_hidden_layers", None)
    except Exception:
        # If we can't get the config, num_layers will be None
        pass

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=server_args.kt_num_gpu_experts,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    topk_ids[topk_ids >= num_gpu_experts] = -1
    return topk_ids


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    """Mask CPU expert IDs and remap GPU expert IDs to GPU-local indices."""
    is_gpu_expert = gpu_experts_mask[topk_ids]
    return torch.where(is_gpu_expert, logical_to_gpu_index[topk_ids], -1)


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.gpu_experts_mask = torch.empty(0, dtype=torch.bool)
        self.logical_to_gpu_index = torch.empty(0, dtype=torch.int32)
        self.gpu_experts_mask_cuda: Optional[torch.Tensor] = None
        self.logical_to_gpu_index_cuda: Optional[torch.Tensor] = None
        self.tp_rank = get_parallel().tp_rank

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Store parameters needed for KT initialization
        self._layer_params = None

    def _build_kt_wrapper_kwargs(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size_full: int,
        layer_max_deferred: int,
    ) -> dict:
        wrapper_kwargs = {
            "layer_idx": self.kt_config.layer_idx,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "hidden_size": hidden_size,
            "moe_intermediate_size": intermediate_size_full,
            "cpuinfer_threads": self.kt_config.cpuinfer_threads,
            "threadpool_count": self.kt_config.threadpool_count,
            "weight_path": self.kt_config.weight_path,
            "chunked_prefill_size": self.kt_config.chunked_prefill_size,
            "method": self.kt_config.method,
            "max_deferred_experts_per_token": layer_max_deferred,
        }

        signature = inspect.signature(KTMoEWrapper)
        if "gpu_experts_mask" in signature.parameters:
            wrapper_kwargs["gpu_experts_mask"] = self.gpu_experts_mask
        else:
            wrapper_kwargs["num_gpu_experts"] = self.num_gpu_experts

        return wrapper_kwargs

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        if self.moe_ep_size != 1:
            raise NotImplementedError(
                "KTransformers heterogeneous MoE does not currently support expert parallelism. "
                "Please launch with --expert-parallel-size 1."
            )
        if self.num_gpu_experts > num_experts:
            raise ValueError(
                f"kt_num_gpu_experts ({self.num_gpu_experts}) exceeds the number of local experts ({num_experts})."
            )

        self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
        self.gpu_experts_mask[: self.num_gpu_experts] = True
        gpu_expert_indices = torch.where(self.gpu_experts_mask)[0]
        self.logical_to_gpu_index = torch.full(
            (num_experts,), -1, dtype=torch.int32
        )
        self.logical_to_gpu_index[gpu_expert_indices] = torch.arange(
            len(gpu_expert_indices), dtype=torch.int32
        )

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        target_device = next(layer.parameters()).device
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device=target_device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(
            device=target_device
        )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.tp_rank == 0:
            self.wrapper = KTMoEWrapper(
                **self._build_kt_wrapper_kwargs(
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    hidden_size=hidden_size,
                    intermediate_size_full=intermediate_size_full,
                    layer_max_deferred=layer_max_deferred,
                )
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            if (
                metadata is not None
                and getattr(metadata, "physical_to_logical_map_cpu", None) is not None
            ):
                physical_to_logical_map_cpu = (
                    metadata.physical_to_logical_map_cpu[self.kt_config.layer_idx]
                    .contiguous()
                )
            else:
                physical_to_logical_map_cpu = torch.arange(
                    layer.num_experts, dtype=torch.int64, device="cpu"
                )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        gpu_runner_config = replace(moe_runner_config)
        if self.override_num_local_experts:
            gpu_runner_config = replace(
                gpu_runner_config, num_local_experts=self.num_gpu_experts
            )
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, gpu_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        # Step 1: Submit CPU expert computation (non-blocking)
        if self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2: Prepare GPU computation by masking CPU experts and remapping GPU experts
        topk_ids = topk_output.topk_ids
        masked_topk_ids = mask_and_remap_expert_ids(
            topk_ids,
            self.gpu_experts_mask_cuda,
            self.logical_to_gpu_index_cuda,
        )

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation (any quantization method)
        # This runs in parallel with CPU computation
        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)

        # Step 4: Synchronize CPU results and merge with GPU results
        output = gpu_combine_input.hidden_states
        if self.tp_rank == 0:
            cpu_output = self.sync(x)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
