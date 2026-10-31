# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Immutable canonical layout admission and live-consumer identity checks."""

from __future__ import annotations

from itertools import chain

from sglang.srt.weight_sync.gpu_delta import profiling
from sglang.srt.weight_sync.gpu_delta.bindings import (
    BACKEND_LAYOUTS,
    ConsumerSnapshot,
    _digest,
    _tensor_identity,
)
from sglang.srt.weight_sync.gpu_delta.derived import nvfp4_derived_update, plan_updates
from sglang.srt.weight_sync.gpu_delta.flashinfer import consumer_objects
from sglang.srt.weight_sync.gpu_delta.models import model_mapping


class GpuDeltaLayout:
    """Frozen canonical bindings with generic backend and consumer admission."""

    def __init__(self, model, inventory):
        from sglang.srt.weight_sync.gpu_delta.host import _natural_key

        if not inventory:
            raise ValueError("canonical checkpoint metadata is unavailable")
        # Normalize negotiated wire metadata without changing checkpoint inventory.
        self.inventory = {name: dict(meta) for name, meta in inventory.items()}
        mapping = model_mapping(model)
        self.bindings = []
        self.excluded = mapping.parameters.excluded
        for name, meta in self.inventory.items():
            binding = mapping.bind(name, meta)
            if binding is not None:
                self.bindings.append(binding)
        self.bindings.sort(key=lambda b: _natural_key(b.name))
        if mapping.parameters.moe_layers:
            from sglang.srt.runtime_context import get_exec

            _require_fixed_moe_topology(get_exec().moe)
        self.derived = []
        self.derived_updates = []
        self._consumers = []
        for prefix, layer in mapping.parameters.moe_layers.items():
            layout = mapping.parameters.moe_layouts[prefix]
            if layout.format == "nvfp4":
                self.derived_updates.append(nvfp4_derived_update(layer, layout.backend))
            self._consumers.append(
                ConsumerSnapshot(
                    lambda layer=layer, layout=layout: consumer_objects(layer, layout)
                )
            )
        self.derived_updates = plan_updates(
            self.derived_updates,
            [
                target
                for binding in self.bindings
                if binding.encoding == "raw_bytes"
                for target in binding.storage
            ],
        )
        mapping.finish()
        self.derived.extend(mapping.derived)
        self._consumers.extend(mapping.consumers)
        self._parameter_roots = [
            (module, name, tensor, _tensor_identity(tensor))
            for module in mapping.parameters.modules.values()
            for name, tensor in module._parameters.items()
            if tensor is not None
        ]
        self._prepare_identity_checks()
        self._reject_overlaps(mapping.aliases)
        self.description = {
            "layouts": dict(BACKEND_LAYOUTS),
            "tensors": [b.describe() for b in self.bindings],
            "excluded": self.excluded,
        }
        self.rank_plan_digest = _digest(self.description)
        self.description["rank_plan_digest"] = self.rank_plan_digest

    def _prepare_identity_checks(self):
        # Deduplicate geometry once, not in the per-update validation loop.
        # Separate views sharing storage still require separate geometry checks.
        checks = chain(
            (
                (tensor, expected, "parameter identity")
                for _, _, tensor, expected in self._parameter_roots
            ),
            (
                (consumer.objects[index], expected, "consumer storage")
                for consumer in self._consumers
                for index, expected in consumer.tensors
            ),
        )
        unique = {}
        for tensor, expected, kind in checks:
            key = id(tensor)
            prior = unique.get(key)
            if prior is None:
                unique[key] = (tensor, expected, kind)
            elif prior[1] != expected:
                raise RuntimeError(
                    f"GPU delta {kind} changed during admission; readmission required"
                )
        self._tensor_checks = tuple(unique.values())

    @profiling.annotate("gpu_delta.check_identity")
    def check_identity(self):
        for module, name, tensor, _ in self._parameter_roots:
            if module._parameters.get(name) is not tensor:
                raise RuntimeError(
                    "GPU delta parameter identity changed; readmission required"
                )
        for consumer in self._consumers:
            consumer.check_objects()
        for tensor, expected, kind in self._tensor_checks:
            if _tensor_identity(tensor) != expected:
                raise RuntimeError(f"GPU delta {kind} changed; readmission required")

    def _reject_overlaps(self, model_aliases):
        # Gate/up intentionally share one interleaved weight/scale allocation;
        # their maps are disjoint. Other shared storage needs an explicit alias
        # adapter so a tied parameter is never XORed twice.
        owners = {}
        for binding in self.bindings:
            for tensor in binding.storage:
                key = tensor.untyped_storage().data_ptr()
                prior = owners.get(key)
                if prior and prior != binding.name:
                    pair = (prior, binding.name)
                    if not all(".experts." in name for name in pair) and not (
                        _gate_up_alias(*pair) or model_aliases(*pair)
                    ):
                        raise ValueError(f"unclassified live weight alias: {pair}")
                owners[key] = binding.name


def _gate_up_alias(a, b):
    return (
        a.replace("gate_proj", "up_proj") == b or b.replace("gate_proj", "up_proj") == a
    )


def _require_fixed_moe_topology(moe):
    # The canonical expert map is valid only while physical expert ownership is
    # the trivial EP partition. EPLB/custom/elastic maps can change ownership
    # without changing tensor pointers, so pointer checks alone cannot admit it.
    if (
        moe.enable_eplb
        or moe.elastic_ep_backend is not None
        or moe.init_expert_location != "trivial"
        or moe.ep_num_redundant_experts != 0
        or (moe.moe_runner_backend, moe.moe_a2a_backend)
        not in {
            ("flashinfer_trtllm", "none"),
            ("flashinfer_trtllm_routed", "none"),
            ("flashinfer_mxfp4", "none"),
            ("flashinfer_cutedsl", "none"),
            ("flashinfer_megamoe", "flashinfer_megamoe"),
        }
    ):
        raise ValueError(
            "direct GPU deltas require fixed trivial EP, no redundant experts or EPLB, "
            "and a native FlashInfer runner with its matching dispatcher"
        )
