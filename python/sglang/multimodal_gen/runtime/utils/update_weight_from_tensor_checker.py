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
_ROOT_TP_RANK = 0


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
        expected_transformer_sha256: dict[str, str],
        tp_rank: int,
        tp_world_size: int,
        tp_cpu_group,
    ) -> tuple[bool, str]:
        if tp_world_size == 1:
            try:
                return self.verify(expected_transformer_sha256)
            except Exception as e:
                return (
                    False,
                    "Exception while verifying transformer update from tensor: "
                    f"{type(e).__name__}: {e}",
                )

        return self._verify_on_tp_root(
            expected_transformer_sha256=expected_transformer_sha256,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            tp_cpu_group=tp_cpu_group,
        )

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

        actual_transformer_sha256 = self._build_local_transformer_sha256(
            transformer=transformer,
            expected_transformer_sha256=expected_transformer_sha256,
        )
        return self._compare_manifests(
            expected_transformer_sha256,
            actual_transformer_sha256,
        )

    def _verify_on_tp_root(
        self,
        *,
        expected_transformer_sha256: dict[str, str],
        tp_rank: int,
        tp_world_size: int,
        tp_cpu_group,
    ) -> tuple[bool, str]:
        transformer, error_message = self._get_transformer_for_verification(
            expected_transformer_sha256
        )
        gathered_errors: list[str | None] | None = (
            [None] * tp_world_size if tp_rank == _ROOT_TP_RANK else None
        )
        torch.distributed.gather_object(
            error_message,
            gathered_errors,
            dst=_ROOT_TP_RANK,
            group=tp_cpu_group,
        )

        result: tuple[bool, str] | None = None
        should_verify_holder = [False]
        if tp_rank == _ROOT_TP_RANK:
            assert gathered_errors is not None
            failures = [
                (rank, message)
                for rank, message in enumerate(gathered_errors)
                if message is not None
            ]
            if failures:
                rank, message = failures[0]
                if len(failures) == 1:
                    result = (False, f"TP rank {rank}: {message}")
                else:
                    result = (
                        False,
                        f"{len(failures)} TP ranks failed update_weight_from_tensor_checker; "
                        f"first failure on rank {rank}: {message}",
                    )
            else:
                should_verify_holder[0] = True

        torch.distributed.broadcast_object_list(
            should_verify_holder,
            src=_ROOT_TP_RANK,
            group=tp_cpu_group,
        )
        if should_verify_holder[0]:
            assert transformer is not None
            actual_transformer_sha256 = self._build_root_transformer_sha256(
                transformer=transformer,
                expected_transformer_sha256=expected_transformer_sha256,
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
                tp_cpu_group=tp_cpu_group,
            )
            if tp_rank == _ROOT_TP_RANK:
                assert actual_transformer_sha256 is not None
                success, message = self._compare_manifests(
                    expected_transformer_sha256,
                    actual_transformer_sha256,
                )
                if success:
                    message = (
                        f"Verified transformer update across {tp_world_size} TP ranks."
                    )
                result = (success, message)

        result_holder = [result]
        torch.distributed.broadcast_object_list(
            result_holder,
            src=_ROOT_TP_RANK,
            group=tp_cpu_group,
        )
        final_result = result_holder[0]
        assert final_result is not None
        return final_result

    def _get_transformer_for_verification(
        self,
        expected_transformer_sha256: dict[str, str],
    ) -> tuple[torch.nn.Module | None, str | None]:
        if not expected_transformer_sha256:
            return None, "expected_transformer_sha256 is required"
        if not isinstance(expected_transformer_sha256, dict):
            return None, "expected_transformer_sha256 must be a dict[str, str]"

        transformer = self.pipeline.get_module(_TRANSFORMER_MODULE_NAME)
        if transformer is None:
            return None, "Transformer module is not initialized"
        if not isinstance(transformer, torch.nn.Module):
            return None, "Transformer module is not a torch.nn.Module"
        return transformer, None

    def _build_local_transformer_sha256(
        self,
        *,
        transformer: torch.nn.Module,
        expected_transformer_sha256: dict[str, str],
    ) -> dict[str, str]:
        return build_named_tensor_sha256(
            self._iter_transformer_named_tensors(
                transformer, expected_transformer_sha256.keys()
            )
        )

    def _build_root_transformer_sha256(
        self,
        *,
        transformer: torch.nn.Module,
        expected_transformer_sha256: dict[str, str],
        tp_rank: int,
        tp_world_size: int,
        tp_cpu_group,
    ) -> dict[str, str] | None:
        local_named_tensors = dict(
            self._iter_transformer_named_tensors(
                transformer, expected_transformer_sha256.keys()
            )
        )
        reference_tensors = dict(transformer.named_parameters())
        reference_tensors.update(dict(transformer.named_buffers()))
        actual_transformer_sha256: dict[str, str] | None = (
            {} if tp_rank == _ROOT_TP_RANK else None
        )
        for name, expected_sha256 in expected_transformer_sha256.items():
            gathered_tensors = self._gather_materialized_tensors_to_root(
                materialized=(
                    _materialize_tensor_for_sha256(local_named_tensors[name])
                    if name in local_named_tensors
                    else None
                ),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
                tp_cpu_group=tp_cpu_group,
            )
            if tp_rank != _ROOT_TP_RANK:
                continue

            reconstructed_sha256 = self._compute_transformer_tensor_sha256_from_gathered(
                gathered_tensors=gathered_tensors,
                expected_sha256=expected_sha256,
                reference_tensor=reference_tensors.get(name),
            )
            if reconstructed_sha256 is not None:
                assert actual_transformer_sha256 is not None
                actual_transformer_sha256[name] = reconstructed_sha256

        return actual_transformer_sha256

    def _compute_transformer_tensor_sha256_from_gathered(
        self,
        *,
        gathered_tensors: list[torch.Tensor | None] | None,
        expected_sha256: str,
        reference_tensor: torch.Tensor | None,
    ) -> str | None:
        if gathered_tensors is None:
            return None

        valid_tensors = [tensor for tensor in gathered_tensors if tensor is not None]
        if len(valid_tensors) != len(gathered_tensors):
            return None

        local_sha256s = [
            _compute_sha256_from_materialized(materialized)
            for materialized in valid_tensors
        ]
        if all(local_sha256 == expected_sha256 for local_sha256 in local_sha256s):
            return expected_sha256

        candidate_dims = self._get_tp_candidate_dims(reference_tensor)
        for shard_dim in candidate_dims:
            reconstructed = self._reconstruct_tp_tensor(
                gathered_tensors=valid_tensors,
                shard_dim=shard_dim,
            )
            if reconstructed is None:
                continue
            reconstructed_sha256 = _compute_sha256_from_materialized(reconstructed)
            if reconstructed_sha256 == expected_sha256:
                return reconstructed_sha256

        return local_sha256s[0]

    def _get_tp_candidate_dims(
        self,
        reference_tensor: torch.Tensor | None,
    ) -> list[int]:
        if reference_tensor is None:
            return []

        candidate_dims: list[int] = []
        if isinstance(reference_tensor, DTensor):
            for placement in reference_tensor.placements:
                shard_dim = getattr(placement, "dim", None)
                if isinstance(shard_dim, int):
                    candidate_dims.append(shard_dim)

        for attr in ("input_dim", "output_dim"):
            shard_dim = getattr(reference_tensor, attr, None)
            if isinstance(shard_dim, int):
                candidate_dims.append(shard_dim)

        deduped_dims: list[int] = []
        for shard_dim in candidate_dims:
            if shard_dim not in deduped_dims:
                deduped_dims.append(shard_dim)
        return deduped_dims

    def _gather_materialized_tensors_to_root(
        self,
        *,
        materialized: torch.Tensor | None,
        tp_rank: int,
        tp_world_size: int,
        tp_cpu_group,
    ) -> list[torch.Tensor | None] | None:
        gathered_tensors: list[torch.Tensor | None] | None = (
            [None] * tp_world_size if tp_rank == _ROOT_TP_RANK else None
        )
        torch.distributed.gather_object(
            materialized,
            gathered_tensors,
            dst=_ROOT_TP_RANK,
            group=tp_cpu_group,
        )
        return gathered_tensors

    def _reconstruct_tp_tensor(
        self,
        *,
        gathered_tensors: list[torch.Tensor],
        shard_dim: int,
    ) -> torch.Tensor | None:
        if not gathered_tensors:
            return None

        first_tensor = gathered_tensors[0]
        if first_tensor.ndim == 0:
            return None

        shard_dim %= first_tensor.ndim
        for tensor in gathered_tensors[1:]:
            if tensor.ndim != first_tensor.ndim or tensor.dtype != first_tensor.dtype:
                return None
            if any(
                lhs != rhs
                for dim, (lhs, rhs) in enumerate(zip(first_tensor.shape, tensor.shape))
                if dim != shard_dim
            ):
                return None

        return torch.cat(gathered_tensors, dim=shard_dim).contiguous()

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

    def _format_tensor_names(self, names: list[str]) -> str:
        displayed = names[:_MAX_DISPLAY_TENSORS]
        formatted = ", ".join(displayed)
        if len(names) > _MAX_DISPLAY_TENSORS:
            formatted += f", ... (+{len(names) - _MAX_DISPLAY_TENSORS} more)"
        return formatted
