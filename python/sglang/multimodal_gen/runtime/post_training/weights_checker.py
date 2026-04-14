"""Verification helpers for diffusion update_weights_from_tensor workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
    iter_materialized_weights,
)

_MAX_DISPLAY_TENSORS = 5


def _materialize_tensor_for_sha256(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return tensor.detach().cpu().contiguous()


def _compute_sha256_from_materialized(materialized: torch.Tensor) -> str:
    materialized = materialized.contiguous()

    hasher = hashlib.sha256()
    hasher.update(str(materialized.dtype).encode("utf-8"))
    hasher.update(repr(tuple(materialized.shape)).encode("utf-8"))
    hasher.update(materialized.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    return _compute_sha256_from_materialized(_materialize_tensor_for_sha256(tensor))


def build_named_tensor_sha256(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, str]:
    sha256_by_name: dict[str, str] = {}
    for name, tensor in named_tensors:
        if name in sha256_by_name:
            raise ValueError(f"Duplicate tensor name in SHA256 manifest: {name}")
        sha256_by_name[name] = compute_tensor_sha256(tensor)
    return sha256_by_name


class UpdateWeightFromTensorChecker:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def verify_across_tp(
        self,
        target_module: str,
        expected_named_tensors_sha256: dict[str, str],
        tp_rank: int,
        tp_world_size: int,
        tp_cpu_group,
    ) -> tuple[bool, str]:
        result = self.verify(
            target_module=target_module,
            expected_named_tensors_sha256=expected_named_tensors_sha256,
        )
        if tp_world_size == 1:
            return result

        gathered_results: list[tuple[bool, str]] | None = (
            [result] * tp_world_size if tp_rank == 0 else None
        )
        torch.distributed.gather_object(
            result,
            gathered_results,
            dst=0,
            group=tp_cpu_group,
        )

        final_result: tuple[bool, str] | None = None
        if tp_rank == 0:
            assert gathered_results is not None
            failures = [
                (rank, message)
                for rank, (success, message) in enumerate(gathered_results)
                if not success
            ]
            if failures:
                rank, message = failures[0]
                if len(failures) == 1:
                    final_result = (False, f"TP rank {rank}: {message}")
                else:
                    final_result = (
                        False,
                        f"{len(failures)} TP ranks failed update_weight_from_tensor_checker; "
                        f"first failure on rank {rank}: {message}",
                    )
            else:
                final_result = (
                    True,
                    f"Verified module '{target_module}' update across {tp_world_size} TP ranks.",
                )

        final_result_holder = [final_result]
        torch.distributed.broadcast_object_list(
            final_result_holder,
            src=0,
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
        module, error_message = self._get_module_for_verification(
            target_module=target_module,
            expected_named_tensors_sha256=expected_named_tensors_sha256,
        )
        if error_message is not None:
            return False, error_message

        actual_named_tensors_sha256 = self._build_local_module_sha256(
            module=module,
            expected_named_tensors_sha256=expected_named_tensors_sha256,
        )
        return self._compare_manifests(
            target_module=target_module,
            expected_named_tensors_sha256=expected_named_tensors_sha256,
            actual_named_tensors_sha256=actual_named_tensors_sha256,
        )

    def _get_module_for_verification(
        self,
        *,
        target_module: str,
        expected_named_tensors_sha256: dict[str, str],
    ) -> tuple[torch.nn.Module | None, str | None]:
        if not target_module:
            return None, "target_module is required"
        if not expected_named_tensors_sha256:
            return None, "expected_named_tensors_sha256 is required"
        if not isinstance(expected_named_tensors_sha256, dict):
            return None, "expected_named_tensors_sha256 must be a dict[str, str]"

        module = self.pipeline.get_module(target_module)
        if module is None:
            return None, f"Module '{target_module}' is not initialized"
        return module, None

    def _build_local_module_sha256(
        self,
        *,
        module: torch.nn.Module,
        expected_named_tensors_sha256: dict[str, str],
    ) -> dict[str, str]:
        return build_named_tensor_sha256(
            self._iter_module_named_tensors(
                module, expected_named_tensors_sha256.keys()
            )
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
