import hashlib
import logging
from typing import Dict, Iterable, Optional, Tuple

import torch

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.layers.layernorm import Gemma3RMSNorm, GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    inverse_transform_scale_ue8m0,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = logging.getLogger(__name__)


class WeightChecker:
    def __init__(self, model_runner):
        self._model_runner = model_runner
        self._model = model_runner.model
        self._module_state = dict(self._model.named_modules())
        self._logical_to_packed_mapping = _get_logical_to_packed_mapping(self._model)
        self._snapshot_tensors = None

    def handle(self, action: str, checksums: Optional[Dict[str, str]] = None):
        logger.info(f"[WeightChecker] handle action={action}")
        if action == "snapshot":
            self._snapshot()
        elif action == "reset_tensors":
            self._reset_tensors()
        elif action == "compare":
            self._compare()
        elif action == "compare_checksum":
            self._compare_checksum(checksums)
        else:
            raise Exception(f"Unsupported {action=}")

    def _compare_checksum(self, expected_checksums: Optional[Dict[str, str]]):
        """Compare Miles rollout-payload checksums against full logical tensors in SGLang."""
        if not expected_checksums:
            return

        tp_group = ps.get_tp_group()
        tp_rank = ps.get_tensor_model_parallel_rank()
        actual_state = dict(self._model_state())

        errors = []
        matched_count = 0

        for name in sorted(expected_checksums.keys()):
            resolved = _find_logical_tensor_view(
                name,
                actual_state,
                self._module_state,
                self._logical_to_packed_mapping,
            )
            if resolved is None:
                errors.append(f"name={name} TP_rank={tp_rank} missing!")
                continue

            expected_hash = expected_checksums[name]
            data, module, suffix, shard_index = resolved
            full_param = _reconstruct_full_tensor(
                data,
                module,
                suffix,
                shard_index,
                tp_group,
            )
            actual_hash = _compute_hash(full_param)

            if actual_hash == expected_hash:
                matched_count += 1
            else:
                errors.append(f"name={name} TP_rank={tp_rank} mismatch!")

            del data

        if matched_count > 0:
            logger.info(
                f"[WeightChecker] verified {matched_count} parameters on TP_rank {tp_rank}"
            )

        if errors:
            raise Exception(
                "Weight checksum verification failed:\n" + "\n".join(errors[:5])
            )

    def _snapshot(self):
        """
        Snapshot the model state from GPU to CPU. Keep the tensors with
        `self._snapshot_tensors` for later comparison.
        """
        named_tensors = [
            (name, param.data.detach().cpu()) for name, param in self._model_state()
        ]
        self._snapshot_tensors = dict(named_tensors)
        assert len(self._snapshot_tensors) == len(
            named_tensors
        ), f"should not have duplicated tensor name"

    def _reset_tensors(self):
        """
        Reset the model state to random values, to simulate situations like silent
        data corruption. Only use this for verification purpose.
        """
        for name, param in self._model_state():
            param.copy_(_random_like(param))

    def _compare(self):
        """
        Compare the model state between the snapshot and the current model state.
        """
        assert self._snapshot_tensors is not None

        _check_tensors(
            expect_tensors=_postprocess_tensors(self._snapshot_tensors),
            actual_tensors=_postprocess_tensors(dict(self._model_state())),
        )

    def _model_state(self):
        # TODO: support EAGLE etc (e.g. yield from both main model and draft model)
        yield from self._model.named_parameters()
        for name, buffer in self._model.named_buffers():
            module_name, _, local_name = name.rpartition(".")
            if (
                local_name
                in self._module_state[module_name]._non_persistent_buffers_set
            ):
                continue
            yield name, buffer


def _compute_hash(t: torch.Tensor) -> str:
    """Encapsulates hashing chain: detach -> cpu -> contiguous -> view -> numpy."""
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy()
    ).hexdigest()


def _find_logical_tensor_view(
    name: str,
    actual_state: Dict[str, torch.Tensor],
    module_state: Dict[str, torch.nn.Module],
    logical_to_packed_mapping: Dict[str, tuple[str, int]],
) -> Optional[Tuple[torch.Tensor, torch.nn.Module, str, Optional[int]]]:
    """Find the local SGLang tensor view for one Miles logical tensor name."""
    if name in actual_state:
        module_name, _, suffix = name.rpartition(".")
        return actual_state[name].data, module_state[module_name], suffix, None

    return _find_packed_tensor_view(
        name,
        actual_state,
        module_state,
        logical_to_packed_mapping,
    )


def _find_packed_tensor_view(
    name: str,
    actual_state: Dict[str, torch.Tensor],
    module_state: Dict[str, torch.nn.Module],
    logical_to_packed_mapping: Dict[str, tuple[str, int]],
) -> Optional[Tuple[torch.Tensor, torch.nn.Module, str, Optional[int]]]:
    """Recover one logical HF tensor view from a packed SGLang parameter."""
    parts = name.split(".")
    if len(parts) < 3:
        return None

    shard_name = parts[-2]
    suffix = parts[-1]
    packed_info = logical_to_packed_mapping.get(shard_name)
    if packed_info is None:
        return None

    packed_name, shard_index = packed_info
    packed_state_name = ".".join(parts[:-2] + [packed_name, suffix])
    packed_tensor = actual_state.get(packed_state_name)
    if packed_tensor is None:
        return None

    packed_module_name = ".".join(parts[:-2] + [packed_name])
    packed_module = module_state[packed_module_name]

    if isinstance(packed_module, QKVParallelLinear):
        return (
            _split_qkv_tensor(packed_tensor.data, packed_module, shard_index, suffix),
            packed_module,
            suffix,
            shard_index,
        )

    if isinstance(packed_module, MergedColumnParallelLinear):
        return (
            _split_merged_column_tensor(
                packed_tensor.data,
                packed_module,
                shard_index,
                suffix,
            ),
            packed_module,
            suffix,
            shard_index,
        )

    raise AssertionError(
        "Checksum comparison only supports standard qkv_proj and gate_up_proj packing "
        f"for Miles-synced models, got {type(packed_module).__name__}"
    )


def _get_logical_to_packed_mapping(
    model: torch.nn.Module,
) -> Dict[str, tuple[str, int]]:
    """Collect logical-name to packed-name metadata published by the model."""
    mapping = _get_standard_packed_projection_mapping()

    _add_bitsandbytes_mapping(
        mapping,
        getattr(model, "bitsandbytes_stacked_params_mapping", None),
    )
    _add_packed_modules_mapping(mapping, getattr(model, "packed_modules_mapping", None))
    _add_stacked_mapping_list(mapping, getattr(model, "stacked_params_mapping", None))

    for cls in type(model).__mro__:
        attrs = vars(cls)
        _add_bitsandbytes_mapping(
            mapping,
            attrs.get("bitsandbytes_stacked_params_mapping"),
        )
        _add_packed_modules_mapping(mapping, attrs.get("packed_modules_mapping"))
        _add_stacked_mapping_list(mapping, attrs.get("stacked_params_mapping"))

    return mapping


def _normalize_weight_name(name: str) -> str:
    return name.strip(".")


def _add_bitsandbytes_mapping(
    mapping: Dict[str, tuple[str, int]],
    stacked_mapping,
) -> None:
    if stacked_mapping is None:
        return

    for shard_name, packed_info in stacked_mapping.items():
        normalized_shard_name = _normalize_weight_name(shard_name)
        if normalized_shard_name in mapping:
            continue
        packed_name, shard_index = packed_info
        mapping[normalized_shard_name] = (
            _normalize_weight_name(packed_name),
            _normalize_shard_index(shard_index),
        )


def _add_packed_modules_mapping(
    mapping: Dict[str, tuple[str, int]],
    packed_modules_mapping,
) -> None:
    if packed_modules_mapping is None:
        return

    for packed_name, shard_names in packed_modules_mapping.items():
        normalized_packed_name = _normalize_weight_name(packed_name)
        for shard_index, shard_name in enumerate(shard_names):
            normalized_shard_name = _normalize_weight_name(shard_name)
            if normalized_shard_name in mapping:
                continue
            mapping[normalized_shard_name] = (
                normalized_packed_name,
                shard_index,
            )


def _add_stacked_mapping_list(
    mapping: Dict[str, tuple[str, int]],
    stacked_params_mapping,
) -> None:
    if stacked_params_mapping is None:
        return

    for packed_name, shard_name, shard_index, *_ in stacked_params_mapping:
        normalized_shard_name = _normalize_weight_name(shard_name)
        if normalized_shard_name in mapping:
            continue
        mapping[normalized_shard_name] = (
            _normalize_weight_name(packed_name),
            _normalize_shard_index(shard_index),
        )


def _get_standard_packed_projection_mapping() -> Dict[str, tuple[str, int]]:
    return {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }


def _normalize_shard_index(shard_index: str | int) -> int:
    if isinstance(shard_index, int):
        return shard_index

    if shard_index == "q":
        return 0
    if shard_index == "k":
        return 1
    if shard_index == "v":
        return 2

    raise AssertionError(f"Unsupported shard index: {shard_index}")


def _split_qkv_tensor(
    tensor: torch.Tensor,
    module: QKVParallelLinear,
    shard_index: int,
    suffix: str,
) -> torch.Tensor:
    local_sizes = [
        module.num_heads * module.head_size,
        module.num_kv_heads * module.head_size,
        module.num_kv_heads * module.v_head_size,
    ]
    return _split_packed_tensor(tensor, local_sizes, shard_index, suffix, module)


def _split_merged_column_tensor(
    tensor: torch.Tensor,
    module: MergedColumnParallelLinear,
    shard_index: int,
    suffix: str,
) -> torch.Tensor:
    local_sizes = [size // module.tp_size for size in module.output_sizes]
    return _split_packed_tensor(tensor, local_sizes, shard_index, suffix, module)


def _split_packed_tensor(
    tensor: torch.Tensor,
    local_sizes: list[int],
    shard_index: int,
    suffix: str,
    module: torch.nn.Module,
) -> torch.Tensor:
    split_sizes = local_sizes
    if suffix == "weight_scale_inv":
        split_sizes = _scale_split_sizes(local_sizes, tensor, module)
    return torch.split(tensor, split_sizes, dim=0)[shard_index]


def _scale_split_sizes(
    local_sizes: list[int],
    tensor: torch.Tensor,
    module: torch.nn.Module,
) -> list[int]:
    if tensor.ndim == 0:
        return [1] * len(local_sizes)
    if tensor.ndim == 1:
        return [1] * len(local_sizes)

    block_size = getattr(getattr(module, "quant_method", None), "quant_config", None)
    block_n = None
    if block_size is not None:
        weight_block_size = getattr(block_size, "weight_block_size", None)
        if weight_block_size is not None:
            block_n = weight_block_size[0]

    if block_n is None:
        return local_sizes

    if getattr(tensor, "format_ue8m0", False):
        block_n = 1

    return [(size + block_n - 1) // block_n for size in local_sizes]


def _reconstruct_full_tensor(
    tensor: torch.Tensor,
    module: torch.nn.Module,
    suffix: str,
    shard_index: Optional[int],
    tp_group,
) -> torch.Tensor:
    """Rebuild the full logical tensor across TP ranks before hashing it."""
    if tp_group.world_size == 1:
        return tensor

    if isinstance(module, QKVParallelLinear):
        return _reconstruct_qkv_tensor(tensor, module, suffix, shard_index, tp_group)

    if isinstance(module, MergedColumnParallelLinear):
        return _gather_tensor_shards(
            tensor,
            _column_parallel_shard_dim(tensor.ndim),
            tp_group,
        )

    if isinstance(module, ColumnParallelLinear):
        return _gather_tensor_shards(
            tensor,
            _column_parallel_shard_dim(tensor.ndim),
            tp_group,
        )

    if isinstance(module, RowParallelLinear):
        return _gather_tensor_shards(
            tensor,
            _row_parallel_shard_dim(tensor.ndim, suffix),
            tp_group,
        )

    if isinstance(module, ReplicatedLinear):
        return tensor

    if isinstance(module, (RMSNorm, GemmaRMSNorm, Gemma3RMSNorm)):
        return tensor

    if isinstance(module, VocabParallelEmbedding):
        return _gather_tensor_shards(
            tensor,
            _vocab_parallel_shard_dim(tensor.ndim),
            tp_group,
        )

    raise AssertionError(
        "Checksum comparison only supports explicit Miles sync layouts, "
        f"got {type(module).__name__}"
    )


def _reconstruct_qkv_tensor(
    tensor: torch.Tensor,
    module: QKVParallelLinear,
    suffix: str,
    shard_index: Optional[int],
    tp_group,
) -> torch.Tensor:
    shard_dim = _column_parallel_shard_dim(tensor.ndim)
    if shard_dim is None:
        return tensor

    if shard_index is not None:
        replica_group_size = 1 if shard_index == 0 else module.num_kv_head_replicas
        return _gather_tensor_shards(tensor, shard_dim, tp_group, replica_group_size)

    local_sizes = [
        module.q_proj_shard_size,
        module.kv_proj_shard_size,
        module.v_proj_shard_size,
    ]
    split_sizes = local_sizes
    if suffix == "weight_scale_inv":
        split_sizes = _scale_split_sizes(local_sizes, tensor, module)
    shards = torch.split(tensor, split_sizes, dim=0)
    full_q = _gather_tensor_shards(shards[0], shard_dim, tp_group)
    full_k = _gather_tensor_shards(
        shards[1],
        shard_dim,
        tp_group,
        module.num_kv_head_replicas,
    )
    full_v = _gather_tensor_shards(
        shards[2],
        shard_dim,
        tp_group,
        module.num_kv_head_replicas,
    )
    return torch.cat([full_q, full_k, full_v], dim=0)


def _gather_tensor_shards(
    tensor: torch.Tensor,
    shard_dim: Optional[int],
    tp_group,
    replica_group_size: int = 1,
) -> torch.Tensor:
    if shard_dim is None or tp_group.world_size == 1:
        return tensor

    gathered = tp_group.all_gather(tensor, dim=shard_dim)
    if replica_group_size > 1:
        assert tp_group.world_size % replica_group_size == 0
        shard_size = tensor.shape[shard_dim]
        unique_world_size = tp_group.world_size // replica_group_size
        unique_shards = torch.split(gathered, shard_size, dim=shard_dim)[
            :unique_world_size
        ]
        return torch.cat(unique_shards, dim=shard_dim)
    return gathered


def _column_parallel_shard_dim(ndim: int) -> Optional[int]:
    if ndim in (1, 2):
        return 0
    if ndim == 3:
        return 1
    return None


def _row_parallel_shard_dim(ndim: int, suffix: str) -> Optional[int]:
    if suffix == "bias":
        return None
    if ndim == 2:
        return 1
    if ndim == 3:
        return 2
    return None


def _vocab_parallel_shard_dim(ndim: int) -> Optional[int]:
    if ndim in (1, 2):
        return 0
    return None


def _check_tensors(
    expect_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
    actual_tensors: Iterable[Tuple[str, bool, torch.Tensor]],
):
    from sglang.srt.debug_utils.dumper import get_tensor_info

    good_names = []
    error_messages = []
    info_messages = []

    for (expect_name, expect_should_compare, expect), (
        actual_name,
        actual_should_compare,
        actual,
    ) in zip(expect_tensors, actual_tensors, strict=True):
        assert expect_name == actual_name, f"{expect_name=} {actual_name=}"
        assert (
            expect_should_compare == actual_should_compare
        ), f"{expect_should_compare=} {actual_should_compare=}"
        name = expect_name
        should_compare = expect_should_compare

        expect = expect.cuda()
        actual = actual.cuda()

        if torch.all(expect == actual):
            good_names.append(name)
        else:
            abs_diff = (actual.float() - expect.float()).abs()
            msg = (
                f"name={name} "
                f"max_abs_err={abs_diff.max()} "
                f"mean_abs_err={abs_diff.mean()} "
                f"{get_tensor_info(expect)=} "
                f"{get_tensor_info(actual)=} "
            )
            (error_messages if should_compare else info_messages).append(msg)

    logger.info(f"[check_tensors] equal tensors: {good_names}")
    if len(info_messages) > 0:
        logger.info(f"[check_tensors] info: {info_messages}")
    if len(error_messages) > 0:
        raise Exception(f"check tensor equality failed:\n" + "\n".join(error_messages))


def _random_like(t: torch.Tensor):
    device = t.device
    shape = t.shape
    dtype = t.dtype

    if dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch.float32).to(dtype)

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5

    info = torch.iinfo(dtype)
    return torch.randint(
        low=int(info.min), high=int(info.max), size=shape, device=device, dtype=dtype
    )


def _postprocess_tensors(
    raw: Dict[str, torch.Tensor],
) -> Iterable[Tuple[str, bool, torch.Tensor]]:
    from sglang.srt.debug_utils.dumper import get_tensor_info

    skip_compare_names = []

    # dequant fp8
    quant_names = [
        name
        for name in raw
        # Match: `something.weight`, `something.experts.w2_weight`
        if name.endswith("weight") and name.replace("weight", "weight_scale_inv") in raw
    ]
    skip_compare_names += quant_names
    for name in quant_names:
        w_q = raw[name]
        w_s = raw[name.replace("weight", "weight_scale_inv")]

        try:
            # TODO this is only needed for Blackwell
            w_s_inverse_transformed = inverse_transform_scale_ue8m0(
                w_s, mn=w_q.shape[-2]
            )
            w_dequant = block_quant_dequant(
                w_q,
                w_s_inverse_transformed,
                # TODO do not hardcode
                block_size=[128, 128],
                dtype=torch.bfloat16,
            )
            yield name, True, w_dequant
        except Exception as e:
            e.add_note(
                f"when handling {name=} {get_tensor_info(w_q)=} {get_tensor_info(w_s)=}"
            )
            raise

    for name in raw:
        should_compare = name not in skip_compare_names
        yield name, should_compare, raw[name]
