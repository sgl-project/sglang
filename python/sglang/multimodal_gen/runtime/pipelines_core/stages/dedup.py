"""Stage-local grouped-request dedup helpers."""

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


class StageDedupMixin:
    """Mixin for stage-local grouped-request deduplication.

    The mixin handles only stage-local reuse. It is not a global cache and does
    not decide which requests are equivalent for a stage. A stage opts into the
    common full-stage path by declaring the ``Req`` fields it writes through the
    ``deduplicated_*`` class attributes and by overriding
    ``build_dedup_fingerprint``.

    Stages that can reuse only part of their work should override
    ``run_grouped_requests`` directly and may still use
    ``_group_requests_by_fingerprint`` for stable grouping.
    """

    deduplicated_output_fields: ClassVar[tuple[str, ...]] = ()
    deduplicated_tensor_tree_output_fields: ClassVar[tuple[str, ...]] = ()
    deduplicated_deepcopy_output_fields: ClassVar[tuple[str, ...]] = ()
    deduplicated_extra_tensor_tree_output_keys: ClassVar[tuple[str, ...]] = ()

    def run_grouped_requests(
        self,
        batches: list["Req"],
        server_args: "ServerArgs",
    ) -> list[Any]:
        """Run this stage for a group of independent requests.

        A grouped request is still a list of normal ``Req`` objects. The group
        boundary only gives a stage the opportunity to reduce duplicate work.
        Stages that do not opt in keep the single-request behavior by running
        ``self(batch, server_args)`` for every request.

        Full-stage dedup is declarative: declare the stage-owned output fields
        and return a stage-local fingerprint. Partial reuse belongs in a custom
        override, because the reusable unit is smaller than the whole stage.
        """
        if self.has_deduplicated_output_fields():
            return self.run_deduplicated_group(batches, server_args)

        return [self(batch, server_args) for batch in batches]

    @classmethod
    def has_deduplicated_output_fields(cls) -> bool:
        """Return whether this stage opts into base full-stage dedup."""
        return bool(
            cls.deduplicated_output_fields
            or cls.deduplicated_tensor_tree_output_fields
            or cls.deduplicated_deepcopy_output_fields
            or cls.deduplicated_extra_tensor_tree_output_keys
        )

    def build_dedup_fingerprint(self, batch: "Req", server_args: "ServerArgs") -> Any:
        """Return this stage's semantic input fingerprint for grouped dedup.

        A fingerprint is the stage-local set of input values that fully
        determines the outputs this stage writes. The default is unique per
        request, so dedup is explicit and safe by default.

        Overrides should include every request/config field read by this stage
        and exclude fields that only matter to later stages. If a field is a
        tensor or nested container, use ``freeze_for_dedup`` so the fingerprint
        remains hashable.
        """
        return id(batch)

    def run_deduplicated_group(
        self,
        batches: list["Req"],
        server_args: "ServerArgs",
        copy_outputs=None,
    ) -> list["Req"]:
        """Run full-stage-equivalent requests once and fan out stage outputs."""
        if copy_outputs is None:
            copy_outputs = self.copy_deduplicated_outputs

        results: list[Req | None] = [None] * len(batches)

        for _, group in self._group_requests_by_fingerprint(
            batches, lambda batch: self.build_dedup_fingerprint(batch, server_args)
        ):
            first_index, first_batch = group[0]
            first_result = self(first_batch, server_args)
            results[first_index] = first_result

            for index, batch in group[1:]:
                copy_outputs(first_result, batch)
                results[index] = batch

        return [result for result in results if result is not None]

    def copy_deduplicated_outputs(self, src: "Req", dst: "Req") -> None:
        """Copy declared stage outputs from a computed request to a duplicate.

        ``deduplicated_output_fields`` uses shallow container copies and shares
        tensor references, which is the low-overhead path for read-only outputs
        such as embeddings. Tensor-tree fields recursively clone tensors.
        Deepcopy fields are for mutable request-local runtime objects, such as
        scheduler instances. Extra keys clone selected ``Req.extra`` entries
        without replacing the destination extra dict.
        """
        for field in self.deduplicated_output_fields:
            setattr(dst, field, self.copy_stage_output(getattr(src, field)))
        for field in self.deduplicated_tensor_tree_output_fields:
            setattr(dst, field, self.clone_tensor_tree(getattr(src, field)))
        for field in self.deduplicated_deepcopy_output_fields:
            setattr(dst, field, deepcopy(getattr(src, field)))
        for key in self.deduplicated_extra_tensor_tree_output_keys:
            if key in src.extra:
                dst.extra[key] = self.clone_tensor_tree(src.extra[key])

    @classmethod
    def copy_stage_output(cls, value):
        """Shallow-copy reusable containers while preserving tensor ownership."""
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(value)
        if isinstance(value, dict):
            return dict(value)
        return value

    @classmethod
    def clone_tensor_tree(cls, value):
        """Recursively clone tensors in a small output tree."""
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, list):
            return [cls.clone_tensor_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.clone_tensor_tree(item) for item in value)
        if isinstance(value, dict):
            return {key: cls.clone_tensor_tree(item) for key, item in value.items()}
        return value

    @staticmethod
    def freeze_for_dedup(value: Any) -> Any:
        """Convert common nested values into a hashable fingerprint fragment."""
        if isinstance(value, torch.Tensor):
            if value.numel() <= 256:
                return (
                    "tensor",
                    tuple(value.shape),
                    str(value.dtype),
                    tuple(value.detach().cpu().reshape(-1).tolist()),
                )
            return ("tensor", tuple(value.shape), str(value.dtype), value.device.type)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, StageDedupMixin.freeze_for_dedup(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(StageDedupMixin.freeze_for_dedup(item) for item in value)
        if isinstance(value, set):
            return tuple(
                sorted(StageDedupMixin.freeze_for_dedup(item) for item in value)
            )
        return value

    @staticmethod
    def _group_requests_by_fingerprint(
        batches: list["Req"],
        fingerprint_fn,
    ) -> list[tuple[Any, list[tuple[int, "Req"]]]]:
        """Group requests by a stage-local fingerprint while preserving order."""
        groups: dict[Any, list[tuple[int, "Req"]]] = {}
        for index, batch in enumerate(batches):
            fingerprint = fingerprint_fn(batch)
            groups.setdefault(fingerprint, []).append((index, batch))
        return list(groups.items())
