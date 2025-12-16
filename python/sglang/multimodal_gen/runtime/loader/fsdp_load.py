# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune
# Copyright 2024 The TorchTune Authors.
# Copyright 2025 The sglang-diffusion Authors.

import contextlib
from collections.abc import Callable, Generator
from itertools import chain
from typing import Any

import torch
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.nn.modules.module import _IncompatibleKeys

from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import set_mixed_precision_policy

logger = init_logger(__name__)


# TODO(PY): move this to utils elsewhere
@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


# TODO(PY): add compile option
def maybe_load_fsdp_model(
    model_cls: type[nn.Module],
    init_params: dict[str, Any],
    weight_dir_list: list[str],
    device: torch.device,
    hsdp_replicate_dim: int,
    hsdp_shard_dim: int,
    default_dtype: torch.dtype,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    fsdp_inference: bool = False,
    output_dtype: torch.dtype | None = None,
    pin_cpu_memory: bool = True,
) -> torch.nn.Module:
    """
    Load the model with FSDP if is training, else load the model without FSDP.
    """
    # NOTE(will): cast_forward_inputs=True shouldn't be needed as we are
    # manually casting the inputs to the model
    mp_policy = MixedPrecisionPolicy(
        param_dtype, reduce_dtype, output_dtype, cast_forward_inputs=False
    )

    set_mixed_precision_policy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )

    with set_default_dtype(default_dtype), torch.device("meta"):
        model = model_cls(**init_params)

    # Check if we should use FSDP
    use_fsdp = fsdp_inference

    # Disable FSDP for MPS as it's not compatible
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.is_mps():
        use_fsdp = False
        logger.info("Disabling FSDP for MPS platform as it's not compatible")

    if use_fsdp:
        world_size = hsdp_replicate_dim * hsdp_shard_dim
        if not fsdp_inference:
            hsdp_replicate_dim = world_size
            hsdp_shard_dim = 1

        device_mesh = init_device_mesh(
            "cuda",
            # (Replicate(), Shard(dim=0))
            mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
            mesh_dim_names=("replicate", "shard"),
        )
        shard_model(
            model,
            cpu_offload=cpu_offload,
            reshard_after_forward=True,
            mp_policy=mp_policy,
            mesh=device_mesh,
            fsdp_shard_conditions=model._fsdp_shard_conditions,
            pin_cpu_memory=pin_cpu_memory,
        )

    weight_iterator = safetensors_weights_iterator(weight_dir_list)
    param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)
    load_model_from_full_model_state_dict(
        model,
        weight_iterator,
        device,
        default_dtype,
        strict=True,
        cpu_offload=cpu_offload,
        param_names_mapping=param_names_mapping_fn,
    )
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
        # Avoid unintended computation graph accumulation during inference
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = False
    return model


def shard_model(
    model,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = MixedPrecisionPolicy(),  # noqa
    mesh: DeviceMesh | None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]] = [],  # noqa
    pin_cpu_memory: bool = True,
) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under multiple parallelism.
            Default to None.
        fsdp_shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP.
        pin_cpu_memory (bool): If set to True, FSDP will pin the CPU memory of the offloaded parameters.

    Raises:
        ValueError: If no layer modules were sharded, indicating that no shard_condition was triggered.
    """
    if fsdp_shard_conditions is None or len(fsdp_shard_conditions) == 0:
        logger.warning(
            "The FSDP shard condition list is empty or None. No modules will be sharded in %s",
            type(model).__name__,
        )
        return

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=pin_cpu_memory)

    # iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0
    # TODO(will): don't reshard after forward for the last layer to save on the
    # all-gather that will immediately happen Shard the model with FSDP,
    for n, m in reversed(list(model.named_modules())):
        if any([shard_condition(n, m) for shard_condition in fsdp_shard_conditions]):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)


# TODO(PY): device mesh for cfg parallel
def load_model_from_full_model_state_dict(
    model: FSDPModule | torch.nn.Module,
    full_sd_iterator: Generator[tuple[str, torch.Tensor], None, None],
    device: torch.device,
    param_dtype: torch.dtype,
    strict: bool = False,
    cpu_offload: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
) -> _IncompatibleKeys:
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model (if training) or normal huggingface model
    Args:
        model (Union[FSDPModule, torch.nn.Module]): Model to generate fully qualified names for cpu_state_dict
        full_sd_iterator (Generator): an iterator yielding (param_name, tensor) pairs
        device (torch.device): device used to move full state dict tensors
        param_dtype (torch.dtype): dtype used to move full state dict tensors
        strict (bool): flag to check if to load the model in strict mode
        cpu_offload (bool): flag to check if FSDP offload is enabled
        param_names_mapping (Optional[Callable[[str], str]]): a function that maps full param name to sharded param name
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Raises:
        NotImplementedError: If got FSDP with more than 1D.
    """
    meta_sd = model.state_dict()
    sharded_sd = {}
    custom_param_sd, reverse_param_names_mapping = hf_to_custom_state_dict(
        full_sd_iterator, param_names_mapping
    )  # type: ignore
    for target_param_name, full_tensor in custom_param_sd.items():
        meta_sharded_param = meta_sd.get(target_param_name)
        if meta_sharded_param is None:
            raise ValueError(
                f"Parameter {target_param_name} not found in custom model state dict. The hf to custom mapping may be incorrect."
            )
        if not hasattr(meta_sharded_param, "device_mesh"):
            full_tensor = full_tensor.to(device=device, dtype=param_dtype)
            # In cases where parts of the model aren't sharded, some parameters will be plain tensors
            sharded_tensor = full_tensor
        else:
            full_tensor = full_tensor.to(device=device, dtype=param_dtype)
            sharded_tensor = distribute_tensor(
                full_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.to("cpu")
        sharded_sd[target_param_name] = nn.Parameter(sharded_tensor)

    model.reverse_param_names_mapping = reverse_param_names_mapping
    unused_keys = set(meta_sd.keys()) - set(sharded_sd.keys())
    if unused_keys:
        logger.warning("Found unloaded parameters in meta state dict: %s", unused_keys)

    # List of allowed parameter name patterns
    ALLOWED_NEW_PARAM_PATTERNS = ["gate_compress"]  # Can be extended as needed
    for new_param_name in unused_keys:
        if not any(pattern in new_param_name for pattern in ALLOWED_NEW_PARAM_PATTERNS):
            logger.error(
                "Unsupported new parameter: %s. Allowed patterns: %s",
                new_param_name,
                ALLOWED_NEW_PARAM_PATTERNS,
            )
            raise ValueError(
                f"New parameter '{new_param_name}' is not supported. "
                f"Currently only parameters containing {ALLOWED_NEW_PARAM_PATTERNS} are allowed."
            )
        meta_sharded_param = meta_sd.get(new_param_name)
        if not hasattr(meta_sharded_param, "device_mesh"):
            # Initialize with zeros
            sharded_tensor = torch.zeros_like(
                meta_sharded_param, device=device, dtype=param_dtype
            )
        else:
            # Initialize with zeros and distribute
            full_tensor = torch.zeros_like(
                meta_sharded_param, device=device, dtype=param_dtype
            )
            sharded_tensor = distribute_tensor(
                full_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
        sharded_sd[new_param_name] = nn.Parameter(sharded_tensor)

    # choose `assign=True` since we cannot call `copy_` on meta tensor
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)
