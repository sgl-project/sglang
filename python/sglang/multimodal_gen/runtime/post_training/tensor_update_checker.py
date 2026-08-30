"""Verification helpers for diffusion update_weights_from_tensor workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    iter_materialized_weights,
)

_MAX_DISPLAY_TENSORS = 5


def _materialize_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return tensor.detach().cpu().contiguous()


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = _materialize_local_tensor(tensor)
    hasher = hashlib.sha256()
    hasher.update(str(tensor.dtype).encode("utf-8"))
    hasher.update(repr(tuple(tensor.shape)).encode("utf-8"))
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def build_named_tensor_sha256(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, str]:
    sha256_by_name: dict[str, str] = {}
    for name, tensor in named_tensors:
        sha256_by_name[name] = compute_tensor_sha256(tensor)
    return sha256_by_name


class TensorUpdateChecker:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def verify_across_tp(
        self,
        target_module: str,
        expected_named_tensors_sha256: dict[str, str],
        tp_rank: int,
        tp_world_size: int,
        tp_cpu_group,
        tp_root_rank: int,
    ) -> tuple[bool, str]:
        if tp_world_size == 1:
            return self.verify(
                target_module=target_module,
                expected_named_tensors_sha256=expected_named_tensors_sha256,
            )

        module = self.pipeline.get_module(target_module)
        if module is None:
            return False, f"Module '{target_module}' is not initialized"

        local_named_tensors = dict(
            self._iter_module_named_tensors(
                module, expected_named_tensors_sha256.keys()
            )
        )
        reference_tensors = dict(module.named_parameters())
        reference_tensors.update(dict(module.named_buffers()))
        actual_named_tensors_sha256: dict[str, str] | None = (
            {} if tp_rank == 0 else None
        )

        for name, expected_sha256 in expected_named_tensors_sha256.items():
            gathered_tensors: list[torch.Tensor | None] | None = (
                [None] * tp_world_size if tp_rank == 0 else None
            )
            torch.distributed.gather_object(
                (
                    _materialize_local_tensor(local_named_tensors[name])
                    if name in local_named_tensors
                    else None
                ),
                gathered_tensors,
                dst=tp_root_rank,
                group=tp_cpu_group,
            )
            if tp_rank != 0:
                continue

            valid_tensors = [
                tensor for tensor in gathered_tensors if tensor is not None
            ]
            if len(valid_tensors) != len(gathered_tensors):
                continue

            local_sha256s = [compute_tensor_sha256(tensor) for tensor in valid_tensors]
            if all(local_sha256 == expected_sha256 for local_sha256 in local_sha256s):
                actual_named_tensors_sha256[name] = expected_sha256
                continue

            reference_tensor = reference_tensors.get(name)
            candidate_dims: list[int] = []
            if isinstance(reference_tensor, DTensor):
                for placement in reference_tensor.placements:
                    shard_dim = getattr(placement, "dim", None)
                    if isinstance(shard_dim, int) and shard_dim not in candidate_dims:
                        candidate_dims.append(shard_dim)
            for attr in ("input_dim", "output_dim"):
                shard_dim = getattr(reference_tensor, attr, None)
                if isinstance(shard_dim, int) and shard_dim not in candidate_dims:
                    candidate_dims.append(shard_dim)

            reconstructed_sha256 = None
            first_tensor = valid_tensors[0]
            for shard_dim in candidate_dims:
                if first_tensor.ndim == 0:
                    break

                shard_dim %= first_tensor.ndim
                compatible = True
                for tensor in valid_tensors[1:]:
                    if (
                        tensor.ndim != first_tensor.ndim
                        or tensor.dtype != first_tensor.dtype
                    ):
                        compatible = False
                        break
                    if any(
                        lhs != rhs
                        for dim, (lhs, rhs) in enumerate(
                            zip(first_tensor.shape, tensor.shape)
                        )
                        if dim != shard_dim
                    ):
                        compatible = False
                        break
                if not compatible:
                    continue

                reconstructed = torch.cat(valid_tensors, dim=shard_dim).contiguous()
                if compute_tensor_sha256(reconstructed) == expected_sha256:
                    reconstructed_sha256 = expected_sha256
                    break

            actual_named_tensors_sha256[name] = reconstructed_sha256 or local_sha256s[0]

        final_result: tuple[bool, str] | None = None
        if tp_rank == 0:
            final_result = self._compare_manifests(
                target_module=target_module,
                expected_named_tensors_sha256=expected_named_tensors_sha256,
                actual_named_tensors_sha256=actual_named_tensors_sha256,
            )
            if final_result[0]:
                final_result = (
                    True,
                    f"Verified module '{target_module}' update across {tp_world_size} TP ranks.",
                )

        final_result_holder = [final_result]
        torch.distributed.broadcast_object_list(
            final_result_holder,
            src=tp_root_rank,
            group=tp_cpu_group,
        )
        final_result = final_result_holder[0]
        assert final_result is not None
        return final_result

    def verify(
        self,
        target_module: str,
        expected_named_tensors_sha256: dict[str, str],
    ) -> tuple[bool, str]:
        module = self.pipeline.get_module(target_module)
        if module is None:
            return False, f"Module '{target_module}' is not initialized"

        actual_named_tensors_sha256 = build_named_tensor_sha256(
            self._iter_module_named_tensors(
                module, expected_named_tensors_sha256.keys()
            )
        )
        return self._compare_manifests(
            target_module=target_module,
            expected_named_tensors_sha256=expected_named_tensors_sha256,
            actual_named_tensors_sha256=actual_named_tensors_sha256,
        )

    def _iter_module_named_tensors(
        self,
        module: torch.nn.Module,
        expected_names: Iterable[str],
    ):
        expected_name_set = set(expected_names)
        seen_names: set[str] = set()

        for name, tensor in iter_materialized_weights(module):
            if name not in expected_name_set:
                continue
            seen_names.add(name)
            yield name, tensor

        for name, tensor in module.named_buffers():
            if name in seen_names or name not in expected_name_set:
                continue
            seen_names.add(name)
            yield name, tensor

    def _compare_manifests(
        self,
        *,
        target_module: str,
        expected_named_tensors_sha256: dict[str, str],
        actual_named_tensors_sha256: dict[str, str],
    ) -> tuple[bool, str]:
        missing_names = sorted(
            name
            for name in expected_named_tensors_sha256
            if name not in actual_named_tensors_sha256
        )
        mismatched_names = sorted(
            name
            for name, expected_sha256 in expected_named_tensors_sha256.items()
            if name in actual_named_tensors_sha256
            and actual_named_tensors_sha256[name] != expected_sha256
        )

        if missing_names or mismatched_names:
            parts: list[str] = []
            if missing_names:
                parts.append(
                    "missing "
                    f"{len(missing_names)} tensor(s): "
                    f"{self._format_tensor_names(missing_names)}"
                )
            if mismatched_names:
                parts.append(
                    "checksum mismatch for "
                    f"{len(mismatched_names)} tensor(s): "
                    f"{self._format_tensor_names(mismatched_names)}"
                )
            return (
                False,
                f"Module '{target_module}' update weight check failed: "
                + "; ".join(parts),
            )

        return (
            True,
            f"Verified module '{target_module}' update for "
            f"{len(expected_named_tensors_sha256)} tensor(s).",
        )

    def _format_tensor_names(self, names: list[str]) -> str:
        displayed = names[:_MAX_DISPLAY_TENSORS]
        formatted = ", ".join(displayed)
        if len(names) > _MAX_DISPLAY_TENSORS:
            formatted += f", ... (+{len(names) - _MAX_DISPLAY_TENSORS} more)"
        return formatted
