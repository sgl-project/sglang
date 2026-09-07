# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune
# Copyright 2024 The TorchTune Authors.
# Copyright 2025 The sglang-diffusion Authors.

from collections import Counter, defaultdict
from collections.abc import Callable, Generator
from types import MethodType
from typing import Any

import torch
import torch.distributed.tensor as dist_tensor
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)
from torch.nn.modules.module import _IncompatibleKeys

from sglang.multimodal_gen.configs.models.fsdp import is_module_list_entry_in
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.bitsandbytes import (
    attach_bitsandbytes_4bit_quant_states,
    build_bitsandbytes_4bit_quant_states,
    split_bitsandbytes_4bit_state,
)
from sglang.multimodal_gen.runtime.loader import rank_local_checkpoint
from sglang.multimodal_gen.runtime.loader.utils import (
    finalize_loaded_model,
    get_param_names_mapping,
    hf_to_custom_state_dict,
    initialize_model,
)
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    process_model_weights_after_loading,
)
from sglang.multimodal_gen.utils import set_mixed_precision_policy

logger = init_logger(__name__)

_DTYPE_MISMATCH_EXAMPLE_LIMIT = 3


def _is_bitsandbytes_quant_config(quant_config: Any | None) -> bool:
    if quant_config is None:
        return False
    return quant_config.get_name() == "bitsandbytes"


def _format_dtype_mismatch_summary(
    mismatch_counts: Counter[tuple[torch.dtype, torch.dtype]],
    mismatch_examples: dict[tuple[torch.dtype, torch.dtype], list[str]],
) -> str:
    parts: list[str] = []
    for (checkpoint_dtype, target_dtype), count in mismatch_counts.items():
        examples = mismatch_examples[(checkpoint_dtype, target_dtype)]
        part = f"{checkpoint_dtype}->{target_dtype} x{count}"
        if examples:
            part += f" (e.g. {', '.join(examples)})"
        parts.append(part)
    return "; ".join(parts)


def _make_param_like(
    actual_param: torch.nn.Parameter, tensor: torch.Tensor
) -> torch.nn.Parameter:
    cls = actual_param.__class__
    # nn.Parameter defaults to requires_grad=True, which is illegal for non-floating/complex dtypes (e.g., int8/FP8
    # quantized weights).
    try:
        new_param = cls.__new__(cls, tensor, requires_grad=False)
    except TypeError:
        try:
            new_param = cls.__new__(cls, tensor)
        except TypeError:
            new_param = nn.Parameter(tensor, requires_grad=False)
    new_param.__dict__.update(actual_param.__dict__)
    new_param.requires_grad = False
    return new_param


def _can_assign_tensor_without_copy(
    actual_param: torch.nn.Parameter,
    full_tensor: torch.Tensor,
    target_param: torch.Tensor,
) -> bool:
    """Return whether a TP=1 linear loader would only copy this tensor."""
    weight_loader = actual_param.__dict__.get("weight_loader")
    if not isinstance(weight_loader, MethodType):
        return False

    owner = weight_loader.__self__
    if not isinstance(
        owner,
        (ReplicatedLinear, ColumnParallelLinear, RowParallelLinear),
    ):
        return False
    if not isinstance(owner.quant_method, UnquantizedLinearMethod):
        return False
    if not isinstance(owner, ReplicatedLinear) and owner.tp_size != 1:
        return False
    if type(actual_param) is not nn.Parameter:
        return False
    if any(
        actual_param.__dict__.get(attribute, False)
        for attribute in (
            "is_metadata",
            "is_sharded_weight",
            "needs_scalar_to_array",
        )
    ):
        return False
    return (
        full_tensor.shape == target_param.shape
        and full_tensor.dtype == target_param.dtype
        and full_tensor.layout == target_param.layout
        and full_tensor.stride() == target_param.stride()
    )


def _make_class_name_shard_condition(class_names: set[str]):
    def shard_condition(n: str, m: nn.Module) -> bool:
        return type(m).__name__ in class_names

    return shard_condition


def _is_common_numbered_block(n: str, m: nn.Module) -> bool:
    return is_module_list_entry_in(
        n,
        (
            "blocks",
            "layers",
            "double_blocks",
            "single_blocks",
            "refiner_blocks",
            "noise_refiner",
            "context_refiner",
            "transformer_blocks",
            "single_transformer_blocks",
        ),
    )


def _resolve_fsdp_shard_conditions(
    model: torch.nn.Module,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]] | None,
) -> tuple[list[Callable[[str, nn.Module], bool]], str]:
    if fsdp_shard_conditions:
        return fsdp_shard_conditions, "explicit"

    block_class_names = set(getattr(model, "_repeated_blocks", []) or [])
    block_class_names.update(getattr(model, "_no_split_modules", []) or [])
    if block_class_names:
        return [_make_class_name_shard_condition(block_class_names)], "block-class"

    return [_is_common_numbered_block], "common-numbered-block"


def _maybe_dequantize_fp8(
    full_tensor: torch.Tensor,
    target_dtype: torch.dtype,
    target_param_name: str,
    param_sd: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Auto-dequantize an FP8 checkpoint weight when the model parameter expects a higher-precision type.

    Some modules (e.g. AdaLayerNormZero) don't accept quant_config, so their
    parameters remain in higher precision even when the checkpoint stores FP8
    weights.  In that case we multiply by the per-tensor weight_scale to
    recover the original unquantized value.
    """
    if not (
        full_tensor.dtype == torch.float8_e4m3fn and target_dtype != torch.float8_e4m3fn
    ):
        return full_tensor

    scale_key = target_param_name.rsplit(".", 1)[0] + ".weight_scale"
    scale_tensor = param_sd.get(scale_key)
    if scale_tensor is not None:
        full_tensor = full_tensor.to(torch.float32) * scale_tensor.float()
        logger.debug(
            "Auto-dequantized FP8 weight %s using %s",
            target_param_name,
            scale_key,
        )
    return full_tensor


def _move_to_device_preserving_meta(model: nn.Module, device: torch.device) -> None:
    # Buffers absent from the checkpoint (e.g. cosmos3's RoPE inv_freq) are
    # still on the meta device here and .to() cannot copy out of meta; leave
    # them for the model's post_load_weights() to rebuild on the real device.
    model._apply(lambda t: t if t.is_meta else t.to(device))


def register_fsdp_entrypoints(model: torch.nn.Module) -> None:
    """Let FSDP2 unshard around forward passes that bypass ``__call__``.

    FSDP2 only unshards around the wrapped module's own ``forward``. Parameters
    the shard conditions did not match stay in the catch-all root group, whose
    hook therefore never fires for a model driven through a custom method, and
    the first op mixing them with a plain tensor fails. Models declare those
    entry points in ``_fsdp_forward_methods``, which every model loaded through
    FSDP must define; ``BaseDiT`` and ``TextEncoder`` default it to ``()``.
    """
    for name in model._fsdp_forward_methods:
        register_fsdp_forward_method(model, name)


# TODO(PY): add compile option
def maybe_load_fsdp_model(
    model_cls: type[nn.Module],
    init_params: dict[str, Any],
    weight_dir_list: list[str],
    device: torch.device,
    hsdp_replicate_dim: int,
    hsdp_shard_dim: int,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    component_starts_on_cpu: bool = False,
    fsdp_inference: bool = False,
    output_dtype: torch.dtype | None = None,
    pin_cpu_memory: bool = True,
    strict: bool = True,
    weight_load_plan: WeightLoadPlan | None = None,
    checkpoint_key_filter: Callable[[str], bool] | None = None,
    weights_iterator: Generator[tuple[str, torch.Tensor], None, None] | None = None,
) -> torch.nn.Module:
    """Load a model with optional FSDP (Fully Sharded Data Parallel) support.

    ``model_cls`` must declare ``_fsdp_forward_methods``, the entry points FSDP2
    has to unshard around (empty when the model is driven through ``__call__``).

    Args:
        param_dtype: Data type for model parameters, also used for:
            - Model initialization context (set_default_torch_dtype)
            - FSDP mixed precision policy unless the model preserves mixed
              original parameter dtypes
            - Weight loading and casting
        reduce_dtype: Data type for gradient reduction in FSDP mixed precision.
        component_starts_on_cpu: Load a non-FSDP component onto CPU initially.
            Runtime residency strategies move it to the compute device before use.
        strict: If True, enforce strict state dict loading (all keys must match).
        weight_load_plan: Optional checkpoint/postprocess device plan for this load.
        weights_iterator: Optional pre-built ``(name, tensor)`` source, used
            instead of reading ``weight_dir_list`` as safetensors. Set by callers
            whose checkpoint is not safetensors at all, such as GGUF.
    """
    # NOTE(will): cast_forward_inputs=True shouldn't be needed as we are
    # manually casting the inputs to the model

    # 1. prepare for loading
    default_torch_dtype = param_dtype if param_dtype else torch.bfloat16
    # Some native models deliberately mix FP32 projections with lower-precision
    # blocks.  FSDP must all-gather those parameters in their original dtypes;
    # the thread-local compute dtype below remains the requested default.
    fsdp_param_dtype = (
        None
        if fsdp_inference and getattr(model_cls, "_fsdp_mixed_dtype_params", False)
        else default_torch_dtype
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=fsdp_param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=False,
    )

    set_mixed_precision_policy(
        param_dtype=default_torch_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )

    model = initialize_model(
        model_cls, init_params, default_torch_dtype, torch.device("meta")
    )

    # Check if we should use FSDP
    use_fsdp = fsdp_inference

    # Disable FSDP for MPS as it's not compatible
    if current_platform.is_mps():
        use_fsdp = False
        logger.info("Disabling FSDP for MPS platform as it's not compatible")

    weight_load_plan = weight_load_plan or WeightLoadPlan(checkpoint_load_device=device)
    keep_checkpoint_mapping = bool(
        current_platform.is_mps()
        and weight_load_plan.mps_layerwise_cpu_staging
        and weight_load_plan.checkpoint_load_device.type == "cpu"
    )
    if keep_checkpoint_mapping:
        # layerwise offload replaces block parameters with placeholders after
        # load, so compatible checkpoint tensors stay file-backed on CPU
        model._keep_checkpoint_mapping = True
    defer_cpu_placement = bool(
        component_starts_on_cpu
        and weight_load_plan.defer_cpu_placement
        and not use_fsdp
    )
    load_on_cpu = bool(component_starts_on_cpu and not defer_cpu_placement)
    weight_postprocess_device = weight_load_plan.weight_postprocess_device
    if use_fsdp and weight_postprocess_device is not None:
        logger.warning("Ignoring weight postprocess device override for FSDP loading.")
        weight_postprocess_device = None

    if use_fsdp:
        model._pre_fsdp_weight_loader_params = {
            n: p
            for n, p in model.named_parameters()
            if getattr(p, "weight_loader", None)
        }
        world_size = hsdp_replicate_dim * hsdp_shard_dim
        if not fsdp_inference:
            hsdp_replicate_dim = world_size
            hsdp_shard_dim = 1

        device_mesh = init_device_mesh(
            current_platform.device_type,
            # (Replicate(), Shard(dim=0))
            mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
            mesh_dim_names=("replicate", "shard"),
        )
        shard_model(
            model,
            cpu_offload=False,
            reshard_after_forward=True,
            mp_policy=mp_policy,
            mesh=device_mesh,
            fsdp_shard_conditions=getattr(model, "_fsdp_shard_conditions", None),
            pin_cpu_memory=pin_cpu_memory,
        )
        register_fsdp_entrypoints(model)

    param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)

    # 2. load model from disk
    preprocess_loaded_state_dict = getattr(model, "preprocess_loaded_state_dict", None)
    bnb_quant_states = None
    preconverted_state_dict = None
    is_bnb_quantized = _is_bitsandbytes_quant_config(init_params.get("quant_config"))
    if (
        not weight_load_plan.load_full_state_dict_on_device
        and use_fsdp
        and weight_dir_list
        and weights_iterator is None
        and preprocess_loaded_state_dict is None
        and checkpoint_key_filter is None
        and not is_bnb_quantized
    ):
        preconverted_state_dict = (
            rank_local_checkpoint.try_load_rank_local_fsdp_state_dict(
                model,
                weight_dir_list,
                param_names_mapping_fn,
            )
        )
    elif (
        not weight_load_plan.load_full_state_dict_on_device
        and not use_fsdp
        and weight_dir_list
        and weights_iterator is None
        and preprocess_loaded_state_dict is None
        and checkpoint_key_filter is None
        and not is_bnb_quantized
    ):
        preconverted_state_dict = (
            rank_local_checkpoint.try_load_rank_local_tp_state_dict(
                model,
                weight_dir_list,
                param_names_mapping_fn,
            )
        )

    if preconverted_state_dict is None:
        if weights_iterator is not None:
            weight_iterator = weights_iterator
        elif weight_load_plan.load_full_state_dict_on_device:
            weight_iterator = safetensors_weights_iterator(
                weight_dir_list,
                key_filter=checkpoint_key_filter,
                weight_load_plan=weight_load_plan,
            )
        else:
            weight_iterator = safetensors_weights_iterator(
                weight_dir_list,
                key_filter=checkpoint_key_filter,
            )
        if preprocess_loaded_state_dict is not None:
            weight_iterator = preprocess_loaded_state_dict(weight_iterator)
        if is_bnb_quantized:
            normal_weights, raw_quant_state = split_bitsandbytes_4bit_state(
                weight_iterator
            )
            bnb_quant_states = build_bitsandbytes_4bit_quant_states(
                [name for name, _ in normal_weights],
                raw_quant_state,
                device,
                param_names_mapping_fn,
            )
            weight_iterator = iter(normal_weights)
    else:
        weight_iterator = iter(())

    load_model_from_full_model_state_dict(
        model,
        weight_iterator,
        weight_load_plan.checkpoint_load_device,
        param_dtype,
        strict=strict,
        cpu_offload=load_on_cpu,
        param_names_mapping=param_names_mapping_fn,
        keep_checkpoint_mapping=keep_checkpoint_mapping,
        allow_device_tensor_assignment=(
            weight_load_plan.load_full_state_dict_on_device
        ),
        preconverted_state_dict=preconverted_state_dict,
    )
    if bnb_quant_states:
        attach_bitsandbytes_4bit_quant_states(
            dict(model.named_parameters()), bnb_quant_states
        )

    # 3. postprocessing
    if weight_postprocess_device is not None:
        # move to device to perform postprocessing
        _move_to_device_preserving_meta(model, weight_postprocess_device)

    process_model_weights_after_loading(model)
    model.post_load_weights()

    finalize_loaded_model(model)

    # 4. deferred cpu offload
    if defer_cpu_placement:
        model.to("cpu")

    return model


def shard_model(
    model,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = MixedPrecisionPolicy(),  # noqa
    mesh: DeviceMesh | None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]] | None = None,
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

    """
    fsdp_shard_conditions, condition_source = _resolve_fsdp_shard_conditions(
        model, fsdp_shard_conditions
    )
    if condition_source != "explicit":
        logger.warning(
            "Using %s FSDP shard condition fallback for %s",
            condition_source,
            type(model).__name__,
        )

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
        if any([shard_condition(n, m) for shard_condition in fsdp_shard_conditions]):  # type: ignore
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            f"No layer modules were sharded in {type(model).__name__}. "
            f"FSDP shard condition source: {condition_source}."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)
    logger.info(
        "Applied FSDP to %d submodules in %s using %s shard conditions",
        num_layers_sharded,
        type(model).__name__,
        condition_source,
    )


# TODO(mick): need refactor, to move out checkpoint-specific adjustments
def load_model_from_full_model_state_dict(
    model: FSDPModule | torch.nn.Module,
    full_sd_iterator: Generator[tuple[str, torch.Tensor], None, None],
    checkpoint_load_device: torch.device,
    param_dtype: torch.dtype | None,
    strict: bool = False,
    cpu_offload: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
    keep_checkpoint_mapping: bool = False,
    preconverted_state_dict: (
        tuple[
            dict[
                str,
                torch.Tensor
                | rank_local_checkpoint.LocalFSDPShard
                | rank_local_checkpoint.LocalTPShard,
            ],
            dict[str, tuple[str, Any, Any]],
        ]
        | None
    ) = None,
    allow_device_tensor_assignment: bool = False,
) -> _IncompatibleKeys:
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model (if training) or normal huggingface model
    Args:
        model (Union[FSDPModule, torch.nn.Module]): Model to generate fully qualified names for cpu_state_dict
        full_sd_iterator (Generator): an iterator yielding (param_name, tensor) pairs
        checkpoint_load_device (torch.device): device used to move full state dict tensors
        param_dtype (torch.dtype): dtype used to move full state dict tensors. If none, respect original dtype from checkpoint
        strict (bool): flag to check if to load the model in strict mode
        cpu_offload (bool): flag to check if FSDP offload is enabled
        param_names_mapping (Optional[Callable[[str], str]]): a function that maps full param name to sharded param name
        keep_checkpoint_mapping (bool): retain compatible CPU checkpoint tensors instead of copying them
        allow_device_tensor_assignment (bool): adopt compatible checkpoint tensors
            already materialized on the target device. This is reserved for an
            explicit full-state direct-device load; ordinary loading keeps its
            established parameter materialization path.
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    """
    meta_sd = model.state_dict()
    param_dict = dict(model.named_parameters())

    # map names from checkpoint to customized names
    if preconverted_state_dict is None:
        custom_param_sd, reverse_param_names_mapping = hf_to_custom_state_dict(
            full_sd_iterator,
            param_names_mapping,
            valid_target_names=set(meta_sd.keys()),
        )  # type: ignore
    else:
        custom_param_sd, reverse_param_names_mapping = preconverted_state_dict

    is_fsdp_model = isinstance(model, FSDPModule) or any(
        isinstance(param, dist_tensor.DTensor) for param in meta_sd.values()
    )

    # sort parameter names to ensure all ranks process parameters in the same order
    sorted_param_names = sorted(custom_param_sd.keys())

    sharded_sd = {}
    skipped_checkpoint_keys: list[str] = []
    non_quantized_dtype_mismatch_counts: Counter[tuple[torch.dtype, torch.dtype]] = (
        Counter()
    )
    non_quantized_dtype_mismatch_examples: dict[
        tuple[torch.dtype, torch.dtype], list[str]
    ] = defaultdict(list)
    quantized_dtype_mismatch_counts: Counter[tuple[torch.dtype, torch.dtype]] = (
        Counter()
    )
    quantized_dtype_mismatch_examples: dict[
        tuple[torch.dtype, torch.dtype], list[str]
    ] = defaultdict(list)

    # shard from loaded state_dict, custom_param_sd -> sharded_sd
    for target_param_name in sorted_param_names:
        loaded_tensor = custom_param_sd[target_param_name]
        meta_sharded_param = meta_sd.get(target_param_name)

        if meta_sharded_param is None:
            # For FSDP models, ensure all ranks process parameters consistently
            if strict or is_fsdp_model:
                raise ValueError(
                    f"Parameter {target_param_name} not found in custom model state dict. The hf to custom mapping may be incorrect."
                )
            else:
                skipped_checkpoint_keys.append(target_param_name)
                continue

        target_dtype = meta_sharded_param.dtype
        is_rank_local_fsdp_shard = isinstance(
            loaded_tensor, rank_local_checkpoint.LocalFSDPShard
        )
        is_rank_local_tp_shard = isinstance(
            loaded_tensor, rank_local_checkpoint.LocalTPShard
        )
        is_rank_local_shard = is_rank_local_fsdp_shard or is_rank_local_tp_shard
        full_tensor = loaded_tensor.tensor if is_rank_local_shard else loaded_tensor

        if not is_rank_local_shard:
            full_tensor = _maybe_dequantize_fp8(
                full_tensor,
                target_dtype,
                target_param_name,
                custom_param_sd,  # type: ignore[arg-type]
            )

        if full_tensor.dtype != target_dtype:
            mismatch_key = (full_tensor.dtype, target_dtype)
            if (
                full_tensor.dtype in rank_local_checkpoint.QUANTIZED_DTYPES
                or target_dtype in rank_local_checkpoint.QUANTIZED_DTYPES
            ):
                quantized_dtype_mismatch_counts[mismatch_key] += 1
                if (
                    len(quantized_dtype_mismatch_examples[mismatch_key])
                    < _DTYPE_MISMATCH_EXAMPLE_LIMIT
                ):
                    quantized_dtype_mismatch_examples[mismatch_key].append(
                        target_param_name
                    )
            else:
                non_quantized_dtype_mismatch_counts[mismatch_key] += 1
                if (
                    len(non_quantized_dtype_mismatch_examples[mismatch_key])
                    < _DTYPE_MISMATCH_EXAMPLE_LIMIT
                ):
                    non_quantized_dtype_mismatch_examples[mismatch_key].append(
                        target_param_name
                    )

        if is_rank_local_fsdp_shard:
            if not isinstance(meta_sharded_param, dist_tensor.DTensor):
                raise TypeError(
                    f"Rank-local FSDP shard produced for non-DTensor parameter {target_param_name}"
                )
            local_tensor = full_tensor.to(
                device=checkpoint_load_device,
                dtype=target_dtype,
            )
            sharded_tensor = dist_tensor.DTensor.from_local(
                local_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
                run_check=False,
                shape=meta_sharded_param.shape,
                stride=meta_sharded_param.stride(),
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.to("cpu")
        elif is_rank_local_tp_shard:
            if isinstance(meta_sharded_param, dist_tensor.DTensor):
                raise TypeError(
                    f"Rank-local TP shard produced for DTensor parameter {target_param_name}"
                )
            sharded_tensor = full_tensor.to(
                device=checkpoint_load_device,
                dtype=target_dtype,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
        elif not isinstance(meta_sharded_param, dist_tensor.DTensor):
            full_tensor = full_tensor.to(
                device=checkpoint_load_device,
                dtype=target_dtype,
            )
            actual_param = rank_local_checkpoint.get_param_for_weight_loading(
                model, param_dict, target_param_name
            )
            weight_loader = (
                getattr(actual_param, "weight_loader", None)
                if actual_param is not None
                else None
            )
            use_checkpoint_tensor_directly = bool(
                keep_checkpoint_mapping
                and actual_param is not None
                and not getattr(actual_param, "checkpoint_mapping_unsafe", False)
                and tuple(meta_sharded_param.shape) == tuple(full_tensor.shape)
                and full_tensor.device.type == "cpu"
                and full_tensor.dtype == target_dtype
            )
            if use_checkpoint_tensor_directly:
                sharded_tensor = full_tensor
            elif weight_loader is not None:
                assert actual_param is not None
                if (
                    full_tensor.device.type == "cpu" or allow_device_tensor_assignment
                ) and _can_assign_tensor_without_copy(
                    actual_param, full_tensor, meta_sharded_param
                ):
                    sharded_tensor = full_tensor
                else:
                    sharded_tensor = torch.empty_like(
                        meta_sharded_param,
                        device=checkpoint_load_device,
                        dtype=target_dtype,
                    )
                    # Preserve requires_grad flag to avoid errors with non-floating dtypes
                    requires_grad = meta_sharded_param.requires_grad
                    temp_param = _make_param_like(actual_param, sharded_tensor)
                    if not (
                        sharded_tensor.is_floating_point()
                        or sharded_tensor.is_complex()
                    ):
                        requires_grad = False
                    temp_param.requires_grad = requires_grad
                    try:
                        weight_loader(temp_param, full_tensor)
                    except AssertionError as exc:
                        raise AssertionError(
                            "Failed to shard/load parameter "
                            f"{target_param_name}: full_tensor.shape={tuple(full_tensor.shape)}, "
                            f"meta_sharded_param.shape={tuple(meta_sharded_param.shape)}, "
                            f"temp_param.shape={tuple(temp_param.shape)}, "
                            f"param_cls={type(actual_param).__name__}"
                        ) from exc
                    sharded_tensor = temp_param.data
            else:
                # In cases where parts of the model aren't sharded, some parameters will be plain tensors
                sharded_tensor = full_tensor

            # Important: `cpu_offload` is intended for FSDP-managed parameter movement.
            # If a parameter is not sharded into a DTensor (i.e., no `device_mesh`), FSDP
            # will NOT manage it. Offloading it here would leave CPU parameters that
            # later participate in GPU kernels (e.g., conv/embedding), causing device/dtype
            # mismatches like "Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)".
            #
            # Therefore:
            # - For non-FSDP models, keep the historical behavior (allow CPU offload).
            # - For FSDP models, do NOT offload non-sharded parameters here.
            if cpu_offload and not is_fsdp_model:
                sharded_tensor = sharded_tensor.cpu()
        else:
            full_tensor = full_tensor.to(
                device=checkpoint_load_device, dtype=target_dtype
            )
            actual_param = rank_local_checkpoint.get_param_for_weight_loading(
                model, param_dict, target_param_name
            )
            weight_loader = (
                getattr(actual_param, "weight_loader", None)
                if actual_param is not None
                else None
            )
            if weight_loader is not None:
                assert actual_param is not None
                tp_sharded_tensor = torch.empty(
                    tuple(actual_param.shape),
                    device=checkpoint_load_device,
                    dtype=target_dtype,
                )
                temp_param = _make_param_like(actual_param, tp_sharded_tensor)
                if not (
                    tp_sharded_tensor.is_floating_point()
                    or tp_sharded_tensor.is_complex()
                ):
                    temp_param.requires_grad = False
                try:
                    weight_loader(temp_param, full_tensor)
                except AssertionError as exc:
                    raise AssertionError(
                        "Failed to TP-shard/load FSDP parameter "
                        f"{target_param_name}: full_tensor.shape={tuple(full_tensor.shape)}, "
                        f"meta_sharded_param.shape={tuple(meta_sharded_param.shape)}, "
                        f"temp_param.shape={tuple(temp_param.shape)}, "
                        f"param_cls={type(actual_param).__name__}"
                    ) from exc
                full_tensor = temp_param.data
            sharded_tensor = distribute_tensor(
                full_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.to("cpu")

        actual_param = param_dict.get(target_param_name)
        if actual_param is not None:
            sharded_sd[target_param_name] = _make_param_like(
                actual_param, sharded_tensor
            )
        else:
            sharded_sd[target_param_name] = nn.Parameter(
                sharded_tensor, requires_grad=False
            )

    model.reverse_param_names_mapping = reverse_param_names_mapping

    if non_quantized_dtype_mismatch_counts:
        logger.debug(
            "Casting checkpoint tensors to target dtype during load: %s",
            _format_dtype_mismatch_summary(
                non_quantized_dtype_mismatch_counts,
                non_quantized_dtype_mismatch_examples,
            ),
            main_process_only=True,
            local_main_process_only=True,
        )

    if quantized_dtype_mismatch_counts:
        logger.warning(
            "Dtype mismatches detected for quantized parameters during load: %s",
            _format_dtype_mismatch_summary(
                quantized_dtype_mismatch_counts,
                quantized_dtype_mismatch_examples,
            ),
            main_process_only=True,
            local_main_process_only=True,
        )

    if skipped_checkpoint_keys:
        logger.warning(
            "Checkpoint keys not loaded (no matching model parameter) %s",
            (
                skipped_checkpoint_keys[:20]
                if len(skipped_checkpoint_keys) > 20
                else skipped_checkpoint_keys
            ),
        )
        if len(skipped_checkpoint_keys) > 20:
            logger.warning(
                "... and %d more skipped keys.",
                len(skipped_checkpoint_keys) - 20,
            )

    # parameters in nn.Module that doesn't exist in safetensor files
    unused_keys = set(meta_sd.keys()) - set(sharded_sd.keys())
    if unused_keys:
        logger.warning("Found unloaded parameters in meta state dict: %s", unused_keys)

    # Legacy allowlist for parameter families synthesized after loading.
    # New formats should declare missing_param_init on the parameter instead.
    LEGACY_ALLOWED_NEW_PARAM_PATTERNS = [
        "gate_compress",
        "wcscales",
        "wtscale",
        "input_scale",
        "weight_scale",
        "bias",
        "norm_q",
        "norm_k",
        "weight_scale",
    ]
    for new_param_name in unused_keys:
        meta_sharded_param = meta_sd.get(new_param_name)
        meta_sharded_param_dtype = meta_sharded_param.dtype
        actual_param = param_dict.get(new_param_name)
        missing_param_init = (
            getattr(actual_param, "missing_param_init", None)
            if actual_param is not None
            else None
        )

        if missing_param_init == "error":
            raise ValueError(
                f"Required checkpoint parameter '{new_param_name}' was not loaded. "
                "This usually indicates a checkpoint/model-arch mismatch or a "
                "broken weight-name mapping."
            )

        if missing_param_init is None and not any(
            pattern in new_param_name for pattern in LEGACY_ALLOWED_NEW_PARAM_PATTERNS
        ):
            logger.error(
                "Unsupported new parameter: %s. Allowed legacy patterns: %s",
                new_param_name,
                LEGACY_ALLOWED_NEW_PARAM_PATTERNS,
            )
            raise ValueError(
                f"New parameter '{new_param_name}' is not supported. "
                "Checkpoint-specific synthesized parameters should either match "
                f"{LEGACY_ALLOWED_NEW_PARAM_PATTERNS} or declare missing_param_init."
            )

        if missing_param_init == "ones" or any(
            p in new_param_name
            for p in (
                "wcscales",
                "wtscale",
                "input_scale",
                "weight_scale",
                "norm_q",
                "norm_k",
            )
        ):
            init_like = torch.ones_like
        elif missing_param_init == "zeros" or missing_param_init is None:
            init_like = torch.zeros_like
        else:
            raise ValueError(
                f"Unsupported missing_param_init={missing_param_init!r} for {new_param_name}"
            )

        if not hasattr(meta_sharded_param, "device_mesh"):
            sharded_tensor = init_like(
                meta_sharded_param,
                device=checkpoint_load_device,
                dtype=meta_sharded_param_dtype,
            )
            if cpu_offload and not is_fsdp_model:
                sharded_tensor = sharded_tensor.cpu()
        else:
            full_tensor = init_like(
                meta_sharded_param,
                device=checkpoint_load_device,
                dtype=meta_sharded_param_dtype,
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
