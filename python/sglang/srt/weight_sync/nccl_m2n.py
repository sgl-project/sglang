"""nccl-rl destination adapter for Miles weight updates."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist

_ROUTED_EXPERT_WEIGHT_RECIPES = frozenset(("expert_gate", "expert_up", "expert_down"))
_FP8_WEIGHT_RECIPES = {
    "expert_gate": "expert_fc1_0",
    "expert_up": "expert_fc1_1",
    "expert_down": "expert_fc2",
}
_FP8_SCALE_RECIPES = {
    f"{recipe}_scale": f"{source}_scale"
    for recipe, source in _FP8_WEIGHT_RECIPES.items()
}
_FP8_BLOCK_SIZE = 128
_FP8_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE],
    "weight_dtype": "float8_e4m3fn",
    "scale_dtype": "float32",
    "scale_format": "canonical",
}


def _nccl_rl() -> Any:
    try:
        from nccl import m2n
    except Exception as exc:
        raise RuntimeError(
            "nccl-rl was selected, but its nccl.m2n package or native library "
            "is unavailable"
        ) from exc
    return m2n


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported nccl-rl dtype {name!r}")
    return dtype


def _block_scale_shape(shape: Sequence[int]) -> tuple[int, ...]:
    if len(shape) != 3:
        raise ValueError(f"Block-FP8 expert tensor must be 3-D, got {tuple(shape)}")
    return (
        int(shape[0]),
        (int(shape[1]) + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
        (int(shape[2]) + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
    )


class M2NFP8Storage:
    """Keep graph-visible FP8 buffers alive until an update is finalized.

    The receiver needs canonical tensors while loading, whereas inference may
    use packed UE8M0 scales. Quantization hooks can also replace Parameters.
    Hold views of the original inference storage (not copies), then copy the
    post-processed values back into those exact buffers before generation resumes.
    This state belongs to the model runner, not a reconnectable communicator.
    """

    def __init__(
        self, model: torch.nn.Module, manifests: Sequence[Mapping[str, Any]]
    ) -> None:
        names = {
            entry["destination"]["parameter"]
            for manifest in manifests
            for entry in manifest["entries"]
            if entry.get("tensor_role") in ("weight", "scale")
        }
        params = dict(model.named_parameters()) if names else {}
        self._buffers = {name: params[name].detach() for name in sorted(names)}

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self._buffers:
            return
        params = dict(model.named_parameters())
        # Validate every result before copying any buffer. A changed inference
        # shape/dtype requires graph recapture, not an implicit storage rebind.
        for name, buffer in self._buffers.items():
            param = params.get(name)
            if (
                param is None
                or param.shape != buffer.shape
                or param.dtype != buffer.dtype
                or param.device != buffer.device
            ):
                actual = (
                    "missing"
                    if param is None
                    else (
                        f"shape={tuple(param.shape)}, dtype={param.dtype}, "
                        f"device={param.device}"
                    )
                )
                raise RuntimeError(
                    f"M2N FP8 post-processing changed the inference layout of {name}: "
                    f"{actual}; expected shape={tuple(buffer.shape)}, "
                    f"dtype={buffer.dtype}, device={buffer.device}. "
                    "Cannot resume generation with the captured CUDA graphs."
                )

        for name, buffer in self._buffers.items():
            param = params[name]
            value = param.detach()
            if (
                value.data_ptr() != buffer.data_ptr()
                or value.stride() != buffer.stride()
            ):
                if (
                    value.untyped_storage().data_ptr()
                    == buffer.untyped_storage().data_ptr()
                ):
                    # A hook may return a different view of the same allocation.
                    value = value.clone()
                buffer.copy_(value)
            # Restore the original strides too (packed scales can be strided),
            # keeping the current Parameter and its post-load attributes intact.
            param.data = buffer


@dataclass(frozen=True)
class _Layout:
    mesh: list[list[int]]
    placements: tuple[tuple[str, int | None], ...]
    local_shape: tuple[int, ...]


def _layout(
    descriptor: Mapping[str, Any],
    global_shape: Sequence[int],
    world_size: int,
    label: str,
) -> _Layout:
    mesh = [list(row) for row in descriptor["mesh"]]
    if not mesh or not mesh[0] or any(len(row) != len(mesh[0]) for row in mesh):
        raise ValueError(f"{label} mesh must be a non-empty rectangle")
    ranks = [rank for row in mesh for rank in row]
    if any(
        not isinstance(rank, int) or rank < 0 or rank >= world_size for rank in ranks
    ) or len(ranks) != len(set(ranks)):
        raise ValueError(f"{label} mesh contains invalid or duplicate ranks")
    if ranks != list(range(min(ranks), min(ranks) + len(ranks))):
        raise ValueError(f"{label} mesh must be a row-major contiguous rank interval")

    records = descriptor["placements"]
    if len(records) != 2:
        raise ValueError(f"{label} must have one placement per mesh axis")
    placements: list[tuple[str, int | None]] = []
    shape = list(global_shape)
    for axis_size, record in zip((len(mesh), len(mesh[0])), records, strict=True):
        kind = record.get("type")
        if kind == "replicate":
            placements.append((kind, None))
        elif kind == "shard":
            dim = record.get("dim")
            if not isinstance(dim, int) or dim < 0 or dim >= len(shape):
                raise ValueError(f"{label} has invalid shard dimension {dim!r}")
            if shape[dim] % axis_size:
                raise ValueError(
                    f"{label} shape {shape} cannot shard dimension {dim} "
                    f"over {axis_size} ranks"
                )
            shape[dim] //= axis_size
            placements.append((kind, dim))
        else:
            raise ValueError(f"{label} has unknown placement {record!r}")

    local_shape = tuple(descriptor["local_shape"])
    if local_shape != tuple(shape):
        raise ValueError(
            f"{label} local shape {local_shape} does not match placements; "
            f"expected {tuple(shape)}"
        )
    return _Layout(mesh, tuple(placements), local_shape)


def _shard_dim(layout: _Layout, label: str) -> int:
    if layout.placements[0] != ("replicate", None):
        raise ValueError(f"{label} first mesh axis must be replicated")
    kind, dim = layout.placements[1]
    if kind != "shard" or dim is None:
        raise ValueError(f"{label} second mesh axis must be sharded")
    return dim


def _coordinate(mesh: Sequence[Sequence[int]], rank: int) -> tuple[int, int]:
    coordinates = [
        (replica, row.index(rank)) for replica, row in enumerate(mesh) if rank in row
    ]
    if len(coordinates) != 1:
        raise ValueError(
            f"Communicator rank {rank} must occur once in every destination mesh"
        )
    return coordinates[0]


def _m2n_placements(m2n: Any, layout: _Layout) -> list[Any]:
    return [
        m2n.Replicate() if kind == "replicate" else m2n.Shard(dim)
        for kind, dim in layout.placements
    ]


def _warm_and_borrow_nccl_comm(pg: dist.ProcessGroup, device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(f"nccl-rl requires CUDA, got {device}")
    torch.cuda.set_device(device)
    dist.all_reduce(torch.zeros(1, device=device), group=pg)
    torch.cuda.synchronize(device)
    comm_ptr = int(pg._get_backend(device)._comm_ptr())
    if not comm_ptr:
        raise RuntimeError("ProcessGroupNCCL returned a null communicator pointer")
    return comm_ptr


class NcclM2NReceiver:
    def __init__(
        self,
        *,
        pg: dist.ProcessGroup,
        manifest: dict[str, Any],
        model: torch.nn.Module,
        device: torch.device,
        topology: Mapping[str, int],
        static_expert_placement: bool,
    ) -> None:
        if manifest.get("schema_version") != 1 or not manifest.get("entries"):
            raise ValueError("Unsupported or empty Miles nccl-rl manifest")
        expected_hash = manifest.get("manifest_hash")
        if expected_hash is not None:
            payload = {
                key: value for key, value in manifest.items() if key != "manifest_hash"
            }
            actual_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if expected_hash != actual_hash:
                raise ValueError("Miles nccl-rl manifest hash mismatch")
        self.manifest = manifest
        self._pg = pg
        self.model = model
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.type == "cuda" and device.index is None
            else device
        )
        _nccl_rl()

        # Warm first so one worker's model validation cannot strand its peers.
        self.comm_ptr = _warm_and_borrow_nccl_comm(pg, self.device)
        try:
            world_size = dist.get_world_size(pg)
            if world_size != manifest["communicator_world_size"]:
                raise ValueError("Process-group and manifest world sizes differ")
            self._world_size = world_size
            self._validate_topology(topology, static_expert_placement)
            self._topology = dict(topology)
            self._comm_rank = pg.rank()
            self._params = dict(model.named_parameters())
            self._entries = self._validate_manifest(
                world_size,
                allow_packed_expert_weights=True,
            )
            self.stream = torch.cuda.Stream(device=self.device)
        except Exception:
            self.destroy()
            raise

    @staticmethod
    def _validate_topology(
        topology: Mapping[str, int], static_expert_placement: bool
    ) -> None:
        required = (
            "tp_rank",
            "tp_size",
            "moe_ep_rank",
            "moe_ep_size",
            "moe_tp_rank",
            "moe_tp_size",
            "dp_rank",
            "dp_size",
            "pp_rank",
            "pp_size",
        )
        if any(key not in topology for key in required):
            raise ValueError("Incomplete SGLang topology for nccl-rl")
        ep_layout = (
            topology["tp_size"] == topology["moe_ep_size"]
            and topology["moe_tp_size"] == 1
        )
        moe_tp_layout = (
            topology["moe_ep_size"] == 1
            and topology["moe_tp_size"] == topology["tp_size"]
        )
        if (
            not (ep_layout or moe_tp_layout)
            or topology["dp_size"] != 1
            or topology["dp_rank"] != 0
            or topology["pp_size"] != 1
            or topology["pp_rank"] != 0
        ):
            raise ValueError(
                "nccl-rl requires SGLang EP=TP with MoE-TP=1 or EP=1 "
                "with MoE-TP=TP, plus DP=1 and PP=1; "
                f"got {dict(topology)}"
            )
        if not (0 <= topology["tp_rank"] < topology["tp_size"]):
            raise ValueError(f"Invalid SGLang TP rank in {dict(topology)}")
        if not (0 <= topology["moe_ep_rank"] < topology["moe_ep_size"]):
            raise ValueError(f"Invalid SGLang EP rank in {dict(topology)}")
        if not (0 <= topology["moe_tp_rank"] < topology["moe_tp_size"]):
            raise ValueError(f"Invalid SGLang MoE-TP rank in {dict(topology)}")
        if not static_expert_placement:
            raise ValueError(
                "nccl-rl requires static contiguous experts without EPLB, "
                "elastic EP, or redundant experts"
            )

    def _validate_fp8_target(self, entry: Mapping[str, Any]) -> None:
        parameter = entry["destination"]["parameter"]
        module = self.model.get_submodule(parameter.rsplit(".", 1)[0])
        method = getattr(module, "quant_method", None)
        config = getattr(method, "quant_config", None)
        if (
            method is None
            or config is None
            or not getattr(method, "block_quant", False)
            or getattr(method, "use_mxfp8", False)
            or getattr(config, "use_mxfp8", False)
            or getattr(method, "is_fp4_expert", False)
            or getattr(config, "is_fp4_experts", False)
            or not getattr(config, "is_checkpoint_fp8_serialized", False)
            or getattr(config, "activation_scheme", None) != "dynamic"
            or list(getattr(config, "weight_block_size", ()) or ())
            != [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE]
        ):
            raise ValueError(
                f"{parameter} is not canonical serialized 128x128 block FP8"
            )

        weight_name = (
            parameter.removesuffix("_scale_inv")
            if entry["tensor_role"] == "scale"
            else parameter
        )
        weight = self._params.get(weight_name)
        if (
            weight is None
            or weight.dtype != torch.float8_e4m3fn
            or getattr(weight, "is_shuffled", False)
        ):
            raise ValueError(f"{weight_name} is not an unshuffled float8_e4m3fn weight")

    def _validate_fp8_pairs(
        self,
        pairs: Mapping[
            str,
            Mapping[
                str,
                tuple[Mapping[str, Any], _Layout, _Layout],
            ],
        ],
    ) -> None:
        components_by_module: dict[str, set[str]] = {}
        for pair_id, roles in pairs.items():
            if set(roles) != {"weight", "scale"}:
                raise ValueError(
                    f"FP8 pair {pair_id!r} must contain one weight and one scale"
                )
            weight, weight_src, weight_dst = roles["weight"]
            scale, scale_src, scale_dst = roles["scale"]
            recipe = weight["destination"]["recipe"]
            source_recipe = _FP8_WEIGHT_RECIPES[recipe]
            component = recipe.removeprefix("expert_")
            scale_name = f"{pair_id.removesuffix('.weight')}.weight_scale_inv"
            if (
                weight["name"] != pair_id
                or not pair_id.endswith(f".{component}_proj.weight")
                or scale["name"] != scale_name
                or scale["destination"]["recipe"] != f"{recipe}_scale"
                or weight["source"]["recipe"] != source_recipe
                or scale["source"]["recipe"] != f"{source_recipe}_scale"
                or weight["source"]["names_by_rank"] != scale["source"]["names_by_rank"]
            ):
                raise ValueError(f"FP8 pair {pair_id!r} has mismatched recipes")
            if (
                weight["family"] != scale["family"]
                or weight["pp_rank"] != scale["pp_rank"]
                or weight_src.mesh != scale_src.mesh
                or weight_src.placements != scale_src.placements
                or weight_dst.mesh != scale_dst.mesh
                or weight_dst.placements != scale_dst.placements
            ):
                raise ValueError(f"FP8 pair {pair_id!r} has mismatched layouts")

            expected_scale_global = _block_scale_shape(weight["global_shape"])
            expected_scale_source = _block_scale_shape(weight_src.local_shape)
            expected_scale_destination = _block_scale_shape(weight_dst.local_shape)
            if (
                tuple(scale["global_shape"]) != expected_scale_global
                or scale_src.local_shape != expected_scale_source
                or scale_dst.local_shape != expected_scale_destination
            ):
                raise ValueError(
                    f"FP8 pair {pair_id!r} scale shape does not match its "
                    "128x128 weight blocks"
                )

            weight_parameter = weight["destination"]["parameter"]
            if scale["destination"]["parameter"] != f"{weight_parameter}_scale_inv":
                raise ValueError(
                    f"FP8 pair {pair_id!r} does not target the matching scale"
                )
            module_name = weight_parameter.rsplit(".", 1)[0]
            components = components_by_module.setdefault(module_name, set())
            if recipe in components:
                raise ValueError(
                    f"FP8 module {module_name!r} has duplicate {recipe} pairs"
                )
            components.add(recipe)

        expected_components = set(_FP8_WEIGHT_RECIPES)
        for module_name, components in components_by_module.items():
            if components != expected_components:
                raise ValueError(
                    f"FP8 module {module_name!r} must update gate, up, and down "
                    f"atomically; got {sorted(components)}"
                )

    def _validate_manifest(
        self,
        world_size: int,
        *,
        allow_packed_expert_weights: bool = False,
    ) -> list[tuple[Mapping[str, Any], _Layout, _Layout]]:
        entries: list[tuple[Mapping[str, Any], _Layout, _Layout]] = []
        names: set[str] = set()
        pairs: dict[
            str,
            dict[str, tuple[Mapping[str, Any], _Layout, _Layout]],
        ] = {}
        for entry in self.manifest["entries"]:
            name = entry["name"]
            family = entry["family"]
            pp_rank = entry.get("pp_rank")
            if "pp_rank" in self.manifest and pp_rank != self.manifest["pp_rank"]:
                raise ValueError(
                    f"{name} belongs to another PP stage's M2N communicator"
                )
            if (
                name in names
                or family not in ("dense", "routed_expert")
                or not isinstance(pp_rank, int)
                or pp_rank < 0
            ):
                raise ValueError(f"Invalid nccl-rl entry {name!r} family {family!r}")
            names.add(name)

            global_shape = tuple(entry["global_shape"])
            if not global_shape or any(
                not isinstance(dim, int) or dim <= 0 for dim in global_shape
            ):
                raise ValueError(f"{name} has invalid global shape {global_shape}")
            source = entry["source"]
            destination = entry["destination"]
            src_layout = _layout(source, global_shape, world_size, f"{name} source")
            dst_layout = _layout(
                destination, global_shape, world_size, f"{name} destination"
            )
            src_ranks = {rank for row in src_layout.mesh for rank in row}
            dst_ranks = {rank for row in dst_layout.mesh for rank in row}
            if src_ranks & dst_ranks:
                raise ValueError(f"{name} source and destination meshes overlap")
            if any(len(row) != self._topology["tp_size"] for row in dst_layout.mesh):
                raise ValueError(
                    f"{name} destination rows must have TP/EP size "
                    f"{self._topology['tp_size']}"
                )

            _shard_dim(src_layout, f"{name} source")
            dst_shard_dim = _shard_dim(dst_layout, f"{name} destination")
            _, shard_rank = _coordinate(dst_layout.mesh, self._comm_rank)
            if family == "routed_expert" and self._topology["moe_tp_size"] == 1:
                expected_rank = self._topology["moe_ep_rank"]
                parallelism = "EP"
            elif family == "routed_expert":
                expected_rank = self._topology["moe_tp_rank"]
                parallelism = "MoE-TP"
            else:
                expected_rank = self._topology["tp_rank"]
                parallelism = "TP"
            if shard_rank != expected_rank:
                raise ValueError(
                    f"{name} destination shard {shard_rank} does not match "
                    f"SGLang {parallelism} rank {expected_rank}"
                )

            source_ranks = {str(rank) for rank in src_ranks}
            if (
                set(source["names_by_rank"]) != source_ranks
                or any(not names for names in source["names_by_rank"].values())
                or not source.get("recipe")
                or not destination.get("recipe")
            ):
                raise ValueError(f"{name} has an incomplete source/destination recipe")
            parameter = destination["parameter"]
            if parameter not in self._params:
                raise ValueError(f"Missing nccl-rl destination parameter {parameter!r}")
            recipe = destination["recipe"]
            if (family == "routed_expert") != recipe.startswith("expert_"):
                raise ValueError(
                    f"{name} family {family!r} does not match recipe {recipe!r}"
                )
            if family == "routed_expert":
                if self._topology["moe_tp_size"] == 1:
                    expected_shard_dim = 0
                elif recipe.removesuffix("_scale") in ("expert_gate", "expert_up"):
                    expected_shard_dim = 1
                elif recipe.removesuffix("_scale") == "expert_down":
                    expected_shard_dim = 2
                else:
                    raise ValueError(f"Unknown nccl-rl routed-expert recipe {recipe!r}")
                if dst_shard_dim != expected_shard_dim:
                    raise ValueError(
                        f"{name} destination must shard dimension {expected_shard_dim} "
                        f"for {recipe} with EP={self._topology['moe_ep_size']} and "
                        f"MoE-TP={self._topology['moe_tp_size']}; got dimension {dst_shard_dim}"
                    )

            pair_id = entry.get("pair_id")
            tensor_role = entry.get("tensor_role")
            dtype_name = entry.get("dtype")
            if not isinstance(dtype_name, str):
                raise ValueError(f"{name} has invalid dtype {dtype_name!r}")
            if (pair_id is None) != (tensor_role is None):
                raise ValueError(
                    f"{name} must specify pair_id and tensor_role together"
                )
            if pair_id is not None:
                expected_recipes = (
                    _FP8_WEIGHT_RECIPES
                    if tensor_role == "weight"
                    else _FP8_SCALE_RECIPES if tensor_role == "scale" else None
                )
                expected_dtype = (
                    "float8_e4m3fn" if tensor_role == "weight" else "float32"
                )
                if (
                    not isinstance(pair_id, str)
                    or not pair_id
                    or family != "routed_expert"
                    or expected_recipes is None
                    or recipe not in expected_recipes
                    or dtype_name != expected_dtype
                ):
                    raise ValueError(
                        f"{name} has invalid routed-expert FP8 pair metadata"
                    )
                pair = pairs.setdefault(pair_id, {})
                if tensor_role in pair:
                    raise ValueError(
                        f"FP8 pair {pair_id!r} has duplicate {tensor_role} entries"
                    )
                self._validate_fp8_target(entry)
            elif recipe in _FP8_SCALE_RECIPES or dtype_name.startswith("float8"):
                raise ValueError(f"{name} FP8 weights and scales require pair metadata")

            param = self._params[parameter]
            live_scale = tensor_role == "scale" and bool(
                getattr(param, "format_ue8m0", False)
            )
            if param.dtype != _dtype(dtype_name) and not live_scale:
                raise ValueError(
                    f"{parameter} has dtype {param.dtype}, expected {entry['dtype']}"
                )
            self._validate_parameter(
                entry,
                dst_layout.local_shape,
                param,
                allow_packed_expert_weights=allow_packed_expert_weights,
            )
            record = (entry, src_layout, dst_layout)
            entries.append(record)
            if pair_id is not None:
                pairs[pair_id][tensor_role] = record
        if pairs:
            if self.manifest.get("quantization") != _FP8_QUANTIZATION:
                raise ValueError(
                    "Paired FP8 entries require canonical manifest quantization "
                    f"metadata {_FP8_QUANTIZATION}"
                )
            self._validate_fp8_pairs(pairs)
        elif "quantization" in self.manifest:
            raise ValueError(
                "Unquantized nccl-rl manifests must not include quantization metadata"
            )
        return entries

    def _validate_parameter(
        self,
        entry: Mapping[str, Any],
        local_shape: tuple[int, ...],
        param: torch.Tensor,
        *,
        allow_packed_expert_weights: bool = False,
    ) -> None:
        name = entry["destination"]["parameter"]
        recipe = entry["destination"]["recipe"]
        expected_ndim = 3 if recipe.startswith("expert_") else 2
        if len(local_shape) != expected_ndim:
            raise ValueError(
                f"{name} recipe {recipe} requires a {expected_ndim}-D tensor, "
                f"got local shape {local_shape}"
            )
        if recipe in ("dense_gate", "dense_up"):
            valid = tuple(param.shape) == (
                local_shape[0] * 2,
                local_shape[1],
            )
        elif recipe == "dense_down":
            valid = tuple(param.shape) == local_shape and param.is_contiguous()
        elif recipe in ("expert_gate", "expert_up"):
            canonical = (local_shape[0], local_shape[1] * 2, local_shape[2])
            expected = (
                (canonical[0], canonical[2], canonical[1])
                if self._unquantized_expert_is_transposed(entry)
                else canonical
            )
            valid = tuple(param.shape) == expected
        elif recipe in ("expert_gate_scale", "expert_up_scale"):
            valid = bool(getattr(param, "format_ue8m0", False)) or tuple(
                param.shape
            ) == (
                local_shape[0],
                local_shape[1] * 2,
                local_shape[2],
            )
        elif recipe == "expert_down":
            expected = (
                (local_shape[0], local_shape[2], local_shape[1])
                if self._unquantized_expert_is_transposed(entry)
                else local_shape
            )
            valid = tuple(param.shape) == expected and (
                entry.get("tensor_role") != "weight" or param.is_contiguous()
            )
        elif recipe == "expert_down_scale":
            valid = bool(getattr(param, "format_ue8m0", False)) or (
                tuple(param.shape) == local_shape and param.is_contiguous()
            )
        else:
            raise ValueError(f"Unknown nccl-rl destination recipe {recipe!r}")
        if (
            not valid
            and allow_packed_expert_weights
            and entry["family"] == "routed_expert"
            and recipe in _ROUTED_EXPERT_WEIGHT_RECIPES
            and (
                entry.get("tensor_role") == "weight"
                or self._is_blocked_bf16_expert(entry)
            )
        ):
            canonical_shape = self._canonical_parameter_shape(recipe, local_shape)
            canonical_numel = 1
            for dim in canonical_shape:
                canonical_numel *= dim
            # MoE post-processing may retain BF16 or FP8 values in a
            # backend-specific blocked view. The receive path replaces that
            # storage with a canonical buffer before running any collective.
            valid = param.is_contiguous() and param.numel() == canonical_numel
        if not valid:
            raise ValueError(
                f"{name} has shape {tuple(param.shape)}, incompatible with "
                f"{recipe} local shape {local_shape}"
            )

        if entry["family"] == "routed_expert":
            module = self.model.get_submodule(name.rsplit(".", 1)[0])
            for attr, expected in (
                ("moe_ep_rank", self._topology["moe_ep_rank"]),
                ("moe_ep_size", self._topology["moe_ep_size"]),
                ("moe_tp_rank", self._topology["moe_tp_rank"]),
                ("moe_tp_size", self._topology["moe_tp_size"]),
                ("_num_local_routed", local_shape[0]),
                ("_num_global_routed", entry["global_shape"][0]),
            ):
                actual = getattr(module, attr, expected)
                if actual != expected:
                    raise ValueError(
                        f"{name} has non-contiguous expert ownership: "
                        f"{attr}={actual}, expected {expected}"
                    )

    def _expert_gate_up_starts(
        self, param_name: str, intermediate: int
    ) -> tuple[int, int]:
        module = self.model.get_submodule(param_name.rsplit(".", 1)[0])
        method_up_first = bool(
            getattr(
                getattr(module, "quant_method", None),
                "load_up_proj_weight_first",
                False,
            )
        )
        # FusedMoE's regular weight loader swaps W1/W3 before applying the
        # quantization method's gate/up ordering when FlashInfer TRT-LLM is in
        # use. M2N bypasses that loader, so reproduce both transformations here.
        trtllm_swaps_w13 = bool(getattr(module, "use_flashinfer_trtllm_moe", False))
        up_first = method_up_first ^ trtllm_swaps_w13
        return (intermediate, 0) if up_first else (0, intermediate)

    @staticmethod
    def _canonical_parameter_shape(
        recipe: str, local_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        if recipe.removesuffix("_scale") in ("expert_gate", "expert_up"):
            return (local_shape[0], local_shape[1] * 2, local_shape[2])
        return local_shape

    def _unquantized_expert_is_transposed(self, entry: Mapping[str, Any]) -> bool:
        if entry["family"] != "routed_expert" or entry.get("tensor_role") is not None:
            return False
        name = entry["destination"]["parameter"]
        module = self.model.get_submodule(name.rsplit(".", 1)[0])
        # Match FusedMoE's weight loader. Shapes alone are ambiguous for square
        # matrices, and triton_kernel deliberately stores both weights as KxN.
        return bool(
            getattr(module, "use_triton_kernels", False)
            or getattr(self._params[name], "is_transposed", False)
        )

    def _is_blocked_bf16_expert(self, entry: Mapping[str, Any]) -> bool:
        if entry["family"] != "routed_expert" or entry.get("tensor_role") is not None:
            return False
        name = entry["destination"]["parameter"]
        module = self.model.get_submodule(name.rsplit(".", 1)[0])
        return (
            self._params[name].dtype == torch.bfloat16
            and bool(getattr(module, "use_flashinfer_trtllm_moe", False))
            and not self._unquantized_expert_is_transposed(entry)
        )

    def _prepare_fp8_destinations(self) -> None:
        for entry, _, _ in self._entries:
            if entry.get("tensor_role") in ("weight", "scale"):
                self._validate_fp8_target(entry)

        was_packed = {
            entry["destination"]["parameter"]: bool(
                getattr(
                    self._params[entry["destination"]["parameter"]],
                    "format_ue8m0",
                    False,
                )
            )
            for entry, _, _ in self._entries
            if entry.get("tensor_role") == "scale"
        }
        prepared: set[str] = set()
        for entry, _, dst_layout in self._entries:
            role = entry.get("tensor_role")
            if role not in ("weight", "scale"):
                continue
            descriptor = entry["destination"]
            parameter = descriptor["parameter"]
            if parameter in prepared:
                continue
            prepared.add(parameter)

            param = self._params[parameter]
            shape = self._canonical_parameter_shape(
                descriptor["recipe"], dst_layout.local_shape
            )
            dtype = torch.float8_e4m3fn if role == "weight" else torch.float32
            scale_name = f"{parameter}_scale_inv" if role == "weight" else parameter
            if (
                was_packed[scale_name]
                or param.dtype != dtype
                or tuple(param.shape) != shape
                or not param.is_contiguous()
            ):
                param.data = torch.empty(
                    shape,
                    dtype=dtype,
                    device=param.device,
                )
            if role == "scale":
                param.format_ue8m0 = False

    def _prepare_unquantized_expert_destinations(self) -> None:
        prepared: set[str] = set()
        for entry, _, dst_layout in self._entries:
            descriptor = entry["destination"]
            recipe = descriptor["recipe"]
            parameter = descriptor["parameter"]
            if (
                entry["family"] != "routed_expert"
                or entry.get("tensor_role") is not None
                or recipe not in _ROUTED_EXPERT_WEIGHT_RECIPES
                or parameter in prepared
                or not self._is_blocked_bf16_expert(entry)
            ):
                continue
            prepared.add(parameter)

            param = self._params[parameter]
            shape = self._canonical_parameter_shape(recipe, dst_layout.local_shape)
            if tuple(param.shape) == shape and param.is_contiguous():
                continue
            expected_numel = 1
            for dim in shape:
                expected_numel *= dim
            if param.numel() != expected_numel:
                raise ValueError(
                    f"{parameter} has shape {tuple(param.shape)}, incompatible with "
                    f"canonical {recipe} shape {shape}"
                )
            # FlashInfer TRT-LLM BF16 MoE weights remain in a permuted blocked
            # layout after model post-processing. Rebind that storage to its
            # canonical load shape; every element is overwritten by this update
            # before the model post-load hook packs it again.
            param.data = param.data.reshape(shape)

    def _destination(
        self, entry: Mapping[str, Any], shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, Callable[[], None] | None]:
        descriptor = entry["destination"]
        param = self._params[descriptor["parameter"]].data
        recipe = descriptor["recipe"]
        if self._unquantized_expert_is_transposed(entry):
            # Copy through a logical NxK view without changing the parameter's
            # inference shape, strides, or CUDA-graph-visible storage.
            param = param.transpose(1, 2)
        if recipe == "dense_down":
            return param, None

        buffer = torch.empty(shape, dtype=param.dtype, device=self.device)
        if recipe in ("dense_gate", "dense_up"):
            start = 0 if recipe == "dense_gate" else shape[0]
            return buffer, lambda: param.narrow(0, start, shape[0]).copy_(buffer)
        if recipe in (
            "expert_gate",
            "expert_up",
            "expert_gate_scale",
            "expert_up_scale",
        ):
            component = recipe.removesuffix("_scale")
            starts = self._expert_gate_up_starts(descriptor["parameter"], shape[1])
            start = starts[component == "expert_up"]
            if tuple(param.shape) == (shape[0], shape[1] * 2, shape[2]):
                return buffer, lambda: param.narrow(1, start, shape[1]).copy_(buffer)
            return buffer, lambda: param.narrow(2, start, shape[1]).copy_(
                buffer.transpose(1, 2)
            )
        if tuple(param.shape) == shape:
            if param.is_contiguous():
                return param, None
            return buffer, lambda: param.copy_(buffer)
        return buffer, lambda: param.copy_(buffer.transpose(1, 2))

    def _prepare_receive(self) -> None:
        if getattr(self, "_failed_receive_buffers", None) is not None:
            raise RuntimeError("A failed M2N stream must be destroyed before retrying")
        # Quantization hooks may replace Parameter objects between updates.
        # Always target the loadable storage restored by begin_weight_update().
        self._params = dict(self.model.named_parameters())
        self._prepare_unquantized_expert_destinations()
        self._prepare_fp8_destinations()
        self._entries = self._validate_manifest(self._world_size)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))

    def _receive_entries(self, in_flight: list):
        """Enqueue one entry at a time, retaining its buffers until completion."""
        m2n = _nccl_rl()
        previous_source_mesh = None
        for entry, src_layout, dst_layout in self._entries:
            with torch.cuda.stream(self.stream):
                if (
                    previous_source_mesh is not None
                    and src_layout.mesh != previous_source_mesh
                ):
                    # Dense and expert ownership may differ within a PP stage.
                    # Bystander sources must not run ahead into shared staging.
                    dist.barrier(group=self._pg)
                previous_source_mesh = src_layout.mesh
                destination, post_copy = self._destination(
                    entry, dst_layout.local_shape
                )
                # Keep ownership outside the generator too: an exception from
                # reshard/post_copy unwinds its frame before receive_many drains.
                in_flight[:] = [destination, post_copy]
                m2n.reshard(
                    None,
                    destination,
                    self.comm_ptr,
                    self.stream,
                    src_mesh=src_layout.mesh,
                    src_placements=_m2n_placements(m2n, src_layout),
                    src_local_shape=src_layout.local_shape,
                    src_dtype=_dtype(entry["dtype"]),
                    dst_mesh=dst_layout.mesh,
                    dst_placements=_m2n_placements(m2n, dst_layout),
                    dst_local_shape=dst_layout.local_shape,
                    dst_dtype=_dtype(entry["dtype"]),
                )
                if post_copy is not None:
                    post_copy()
            # Yield OUTSIDE the stream context: another stage must not inherit
            # this stage's current stream. Generator locals retain destination
            # and post_copy while the native copy and merge are still in flight.
            yield
            self.stream.synchronize()
            in_flight.clear()

    def receive(self) -> None:
        self.receive_many([self])

    @staticmethod
    def receive_many(receivers: Sequence[NcclM2NReceiver]) -> None:
        """Overlap one entry per PP stream using deterministic host launch order.

        The caller bounds the number of stages in this wave. No background
        threads or concurrent HTTP requests are needed. Every stream is drained
        before returning, including on failure. If a drain itself fails, its
        temporary buffers stay owned by that receiver until teardown succeeds.
        """
        if not receivers:
            raise ValueError("An M2N PP wave requires at least one receiver")
        parameters: set[str] = set()
        communicators: set[int] = set()
        stages: set[int] = set()
        for receiver in receivers:
            if (
                receiver.model is not receivers[0].model
                or receiver.device != receivers[0].device
            ):
                raise ValueError(
                    "M2N PP receivers must target the same model and device"
                )
            if receiver.comm_ptr is None or receiver.comm_ptr in communicators:
                raise ValueError(
                    "M2N PP receivers require distinct, live communicators"
                )
            communicators.add(receiver.comm_ptr)
            if len(receivers) > 1:
                stage = receiver.manifest.get("pp_rank")
                if not isinstance(stage, int) or stage in stages:
                    raise ValueError(
                        "Concurrent M2N receivers require distinct PP stages"
                    )
                stages.add(stage)
            names = {
                entry["destination"]["parameter"]
                for entry in receiver.manifest["entries"]
            }
            if parameters.intersection(names):
                raise ValueError(
                    "Concurrent M2N PP receivers must not share destination parameters"
                )
            parameters.update(names)

        iterators = []
        in_flight = [[] for _ in receivers]
        error = None
        try:
            # Prepare all stages on the caller stream before any native receive.
            for receiver, buffers in zip(receivers, in_flight):
                receiver._prepare_receive()
                iterators.append(receiver._receive_entries(buffers))
            pending = deque(iterators)
            while pending:
                iterator = pending.popleft()
                try:
                    next(iterator)
                except StopIteration:
                    continue
                pending.append(iterator)
        except Exception as exc:
            error = exc
        finally:
            # Keep every generator (and hence its temporary tensors) alive until
            # ALL streams have been drained, even if an earlier drain fails.
            for receiver, buffers in zip(receivers, in_flight):
                try:
                    receiver.stream.synchronize()
                except Exception as exc:
                    # A failed synchronization cannot prove that native kernels
                    # have stopped using these tensors. Keep them alive beyond
                    # this stack frame, until destroy can successfully drain.
                    if getattr(receiver, "_failed_receive_buffers", None) is None:
                        receiver._failed_receive_buffers = buffers
                    if error is None:
                        error = exc
            for iterator in iterators:
                iterator.close()
        if error is not None:
            raise error

    def destroy(self) -> None:
        if getattr(self, "stream", None) is not None:
            self.stream.synchronize()
        self._failed_receive_buffers = None
        if getattr(self, "comm_ptr", None) is not None:
            _nccl_rl().finalize()
        self.stream = None
        self.comm_ptr = None
