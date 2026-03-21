"""Verification helpers for diffusion update_weights_from_tensor workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
    iter_materialized_weights,
)

_TRANSFORMER_MODULE_NAME = "transformer"
_MAX_DISPLAY_TENSORS = 5


def _materialize_tensor_for_sha256(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return tensor.detach().cpu().contiguous()


def compute_tensor_sha256(tensor: torch.Tensor) -> str:
    materialized = _materialize_tensor_for_sha256(tensor)

    hasher = hashlib.sha256()
    hasher.update(str(materialized.dtype).encode("utf-8"))
    hasher.update(repr(tuple(materialized.shape)).encode("utf-8"))
    hasher.update(materialized.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


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
        expected_transformer_sha256: dict[str, str],
        tp_rank: int,
        tp_world_size: int,
        tp_cpu_group,
    ) -> tuple[bool, str]:
        try:
            local_success, local_message = self.verify(expected_transformer_sha256)
        except Exception as e:
            local_success = False
            local_message = (
                "Exception while verifying transformer update from tensor: "
                f"{type(e).__name__}: {e}"
            )

        if tp_world_size == 1:
            return local_success, local_message

        gathered_results: list[tuple[int, bool, str] | None] = [None] * tp_world_size
        torch.distributed.all_gather_object(
            gathered_results,
            (tp_rank, local_success, local_message),
            group=tp_cpu_group,
        )
        return self._summarize_tp_results(gathered_results)

    def verify(
        self,
        expected_transformer_sha256: dict[str, str],
    ) -> tuple[bool, str]:
        if not expected_transformer_sha256:
            return False, "expected_transformer_sha256 is required"
        if not isinstance(expected_transformer_sha256, dict):
            return False, "expected_transformer_sha256 must be a dict[str, str]"

        transformer = self.pipeline.get_module(_TRANSFORMER_MODULE_NAME)
        if transformer is None:
            return False, "Transformer module is not initialized"
        if not isinstance(transformer, torch.nn.Module):
            return False, "Transformer module is not a torch.nn.Module"

        actual_transformer_sha256 = build_named_tensor_sha256(
            self._iter_transformer_named_tensors(
                transformer, expected_transformer_sha256.keys()
            )
        )
        return self._compare_manifests(
            expected_transformer_sha256,
            actual_transformer_sha256,
        )

    def _iter_transformer_named_tensors(
        self,
        transformer: torch.nn.Module,
        expected_names: Iterable[str],
    ):
        expected_name_set = set(expected_names)
        seen_names: set[str] = set()

        for name, tensor in iter_materialized_weights(transformer):
            if name not in expected_name_set:
                continue
            seen_names.add(name)
            yield name, tensor

        for name, tensor in transformer.named_buffers():
            if name in seen_names or name not in expected_name_set:
                continue
            seen_names.add(name)
            yield name, tensor

    def _compare_manifests(
        self,
        expected_transformer_sha256: dict[str, str],
        actual_transformer_sha256: dict[str, str],
    ) -> tuple[bool, str]:
        missing_names = sorted(
            name
            for name in expected_transformer_sha256
            if name not in actual_transformer_sha256
        )
        mismatched_names = sorted(
            name
            for name, expected_sha256 in expected_transformer_sha256.items()
            if name in actual_transformer_sha256
            and actual_transformer_sha256[name] != expected_sha256
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
                "Transformer update weight check failed: " + "; ".join(parts),
            )

        return (
            True,
            f"Verified transformer update for {len(expected_transformer_sha256)} tensor(s).",
        )

    def _summarize_tp_results(
        self,
        gathered_results: list[tuple[int, bool, str] | None],
    ) -> tuple[bool, str]:
        failures = [
            (rank, message)
            for result in gathered_results
            if result is not None
            for rank, success, message in [result]
            if not success
        ]
        if failures:
            rank, message = failures[0]
            if len(failures) == 1:
                return False, f"TP rank {rank}: {message}"
            return (
                False,
                f"{len(failures)} TP ranks failed update_weight_from_tensor_checker; "
                f"first failure on rank {rank}: {message}",
            )

        return (
            True,
            f"Verified transformer update across {len(gathered_results)} TP ranks.",
        )

    def _format_tensor_names(self, names: list[str]) -> str:
        displayed = names[:_MAX_DISPLAY_TENSORS]
        formatted = ", ".join(displayed)
        if len(names) > _MAX_DISPLAY_TENSORS:
            formatted += f", ... (+{len(names) - _MAX_DISPLAY_TENSORS} more)"
        return formatted
