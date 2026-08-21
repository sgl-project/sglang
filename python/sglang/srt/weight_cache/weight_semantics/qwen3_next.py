from __future__ import annotations

from math import prod
from typing import Any, Sequence

import msgspec

from ..weight_runtime_manifest import (
    LogicalTensorView,
    WeightManifestError,
    WeightParallelTopology,
)
from .qwen3_5 import (
    Qwen35WeightSemanticsAdapter,
    _canonical_name,
    _itemsize,
    _layer_id,
    _shape,
    _view,
)


class Qwen3NextWeightSemanticsAdapter(Qwen35WeightSemanticsAdapter):
    """Describe Qwen3-Next runtime packing in canonical global coordinates."""

    def __init__(
        self,
        *,
        config: Any,
        up_first_w13_parameter_ids: Sequence[int] = (),
        num_fused_shared_experts: int = 0,
    ) -> None:
        super().__init__(
            config=config,
            up_first_w13_parameter_ids=up_first_w13_parameter_ids,
        )
        if num_fused_shared_experts not in (0, 1):
            raise WeightManifestError(
                "Qwen3-Next weight manifests support at most one fused shared expert"
            )
        self._num_fused_shared_experts = num_fused_shared_experts

    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        canonical_names = tuple(dict.fromkeys(_canonical_name(name) for name in names))
        name = canonical_names[0]
        layer_id = _layer_id(name)
        if self._num_fused_shared_experts and name.endswith("experts.w13_weight"):
            return self._fused_moe_w13(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if self._num_fused_shared_experts and name.endswith("experts.w2_weight"):
            return self._fused_moe_w2(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("in_proj_qkvz.weight"):
            self._require_grouped_gdn_layout(parameter, name)
            return self._grouped_gdn_view(
                name=name,
                parameter=parameter,
                topology=topology,
                group_width=self._qkvz_group_width(),
                layout="gdn-qkvz-grouped",
            )
        if name.endswith("in_proj_ba.weight"):
            self._require_grouped_gdn_layout(parameter, name)
            return self._grouped_gdn_view(
                name=name,
                parameter=parameter,
                topology=topology,
                group_width=self._ba_group_width(),
                layout="gdn-ba-grouped",
            )

        views = super().describe_parameter(
            names=names,
            parameter=parameter,
            topology=topology,
        )
        return tuple(self._retag(view) for view in views)

    @staticmethod
    def _require_grouped_gdn_layout(parameter: Any, name: str) -> None:
        layout = getattr(parameter, "_sglang_qwen3_next_gdn_layout", None)
        if layout != "grouped":
            raise WeightManifestError(
                "Qwen3-Next GDN runtime layout marker must explicitly be "
                f"'grouped': {name}: {layout!r}"
            )

    def _routed_expert_ids(
        self, *, parameter: Any, topology: WeightParallelTopology
    ) -> tuple[int, ...]:
        num_experts = int(self._config.num_experts)
        if num_experts % topology.ep_size != 0:
            raise WeightManifestError(
                "Qwen3-Next experts are not evenly EP partitionable"
            )
        local_experts = num_experts // topology.ep_size
        expected_slots = local_experts + self._num_fused_shared_experts
        if not _shape(parameter) or _shape(parameter)[0] != expected_slots:
            raise WeightManifestError(
                f"Qwen3-Next local expert count mismatch: {_shape(parameter)}, "
                f"expected {expected_slots} slots"
            )
        start = topology.ep_rank * local_experts
        return tuple(range(start, start + local_experts))

    def _fused_moe_w13(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        expert_ids = self._routed_expert_ids(parameter=parameter, topology=topology)
        shape = _shape(parameter)
        intermediate = int(self._config.moe_intermediate_size)
        shared_intermediate = int(
            getattr(self._config, "shared_expert_intermediate_size", 0)
        )
        if shared_intermediate != intermediate:
            raise WeightManifestError(
                "Qwen3-Next fused shared expert must match routed intermediate size"
            )
        if intermediate % topology.moe_tp_size != 0:
            raise WeightManifestError("Qwen3-Next expert tensor is not TP divisible")
        local_intermediate = intermediate // topology.moe_tp_size
        expected = (
            len(expert_ids) + 1,
            local_intermediate * 2,
            int(self._config.hidden_size),
        )
        if shape != expected:
            raise WeightManifestError(
                f"Qwen3-Next w13 tensor shape mismatch: {shape}, expected {expected}"
            )
        prefix = name[: -len("experts.w13_weight")]
        expert_bytes = prod(shape[1:]) * _itemsize(parameter)
        component_bytes = local_intermediate * shape[2] * _itemsize(parameter)
        components = (
            ("up_proj", "gate_proj")
            if id(parameter) in self._up_first_w13_parameter_ids
            else ("gate_proj", "up_proj")
        )
        views = []
        num_experts = int(self._config.num_experts)
        for local_index, expert_id in enumerate(expert_ids):
            base = local_index * expert_bytes
            for component_index, component in enumerate(components):
                views.append(
                    _view(
                        tensor_id=f"{prefix}experts.{component}.weight",
                        global_shape=(num_experts, intermediate, shape[2]),
                        global_offset=(
                            expert_id,
                            topology.moe_tp_rank * local_intermediate,
                            0,
                        ),
                        local_shape=(1, local_intermediate, shape[2]),
                        partition_dim=None,
                        byte_offset=base + component_index * component_bytes,
                        layer_id=layer_id,
                        expert_id=None,
                        layout="moe-w13",
                        shard_dims=(0, 1),
                    )
                )
        shared_base = len(expert_ids) * expert_bytes
        for component_index, component in enumerate(components):
            views.append(
                _view(
                    tensor_id=f"{prefix}shared_expert.{component}.weight",
                    global_shape=(shared_intermediate, shape[2]),
                    global_offset=(
                        topology.moe_tp_rank * local_intermediate,
                        0,
                    ),
                    local_shape=(local_intermediate, shape[2]),
                    partition_dim=0,
                    byte_offset=shared_base + component_index * component_bytes,
                    layer_id=layer_id,
                    expert_id=None,
                    layout="gate-up",
                )
            )
        return tuple(self._retag(view) for view in views)

    def _fused_moe_w2(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        expert_ids = self._routed_expert_ids(parameter=parameter, topology=topology)
        shape = _shape(parameter)
        intermediate = int(self._config.moe_intermediate_size)
        shared_intermediate = int(
            getattr(self._config, "shared_expert_intermediate_size", 0)
        )
        if shared_intermediate != intermediate:
            raise WeightManifestError(
                "Qwen3-Next fused shared expert must match routed intermediate size"
            )
        if intermediate % topology.moe_tp_size != 0:
            raise WeightManifestError("Qwen3-Next expert tensor is not TP divisible")
        local_intermediate = intermediate // topology.moe_tp_size
        expected = (
            len(expert_ids) + 1,
            int(self._config.hidden_size),
            local_intermediate,
        )
        if shape != expected:
            raise WeightManifestError(
                f"Qwen3-Next w2 tensor shape mismatch: {shape}, expected {expected}"
            )
        prefix = name[: -len("experts.w2_weight")]
        expert_bytes = prod(shape[1:]) * _itemsize(parameter)
        num_experts = int(self._config.num_experts)
        views = [
            _view(
                tensor_id=f"{prefix}experts.down_proj.weight",
                global_shape=(num_experts, shape[1], intermediate),
                global_offset=(
                    expert_id,
                    0,
                    topology.moe_tp_rank * local_intermediate,
                ),
                local_shape=(1, shape[1], local_intermediate),
                partition_dim=None,
                byte_offset=local_index * expert_bytes,
                layer_id=layer_id,
                expert_id=None,
                layout="moe-w2",
                shard_dims=(0, 2),
            )
            for local_index, expert_id in enumerate(expert_ids)
        ]
        views.append(
            _view(
                tensor_id=f"{prefix}shared_expert.down_proj.weight",
                global_shape=(shape[1], shared_intermediate),
                global_offset=(
                    0,
                    topology.moe_tp_rank * local_intermediate,
                ),
                local_shape=(shape[1], local_intermediate),
                partition_dim=1,
                byte_offset=len(expert_ids) * expert_bytes,
                layer_id=layer_id,
                expert_id=None,
                layout="row-parallel",
            )
        )
        return tuple(self._retag(view) for view in views)

    def _grouped_gdn_view(
        self,
        *,
        name: str,
        parameter: Any,
        topology: WeightParallelTopology,
        group_width: int,
        layout: str,
    ) -> tuple[LogicalTensorView, ...]:
        key_heads = int(self._config.linear_num_key_heads)
        tp_size = topology.attention_tp_size
        tp_rank = topology.attention_tp_rank
        if key_heads % tp_size != 0:
            raise WeightManifestError(
                f"Qwen3-Next GDN key heads are not TP divisible: {key_heads}"
            )
        local_key_heads = key_heads // tp_size
        hidden_size = int(self._config.hidden_size)
        local_shape = (local_key_heads, group_width, hidden_size)
        physical_shape = _shape(parameter)
        valid_shapes = (
            local_shape,
            (local_key_heads * group_width, hidden_size),
        )
        if physical_shape not in valid_shapes:
            raise WeightManifestError(
                f"Qwen3-Next grouped GDN tensor shape mismatch: {name}: "
                f"{physical_shape}, expected one of {valid_shapes}"
            )
        return (
            LogicalTensorView(
                tensor_id=name,
                global_shape=(key_heads, group_width, hidden_size),
                global_offset=(tp_rank * local_key_heads, 0, 0),
                local_shape=local_shape,
                partition_dim=0,
                byte_offset=0,
                layer_id=_layer_id(name),
                expert_id=None,
                layout_fingerprint=f"sglang:qwen3-next:{layout}:v1",
                shard_dims=(0,),
            ),
        )

    def _qkvz_group_width(self) -> int:
        key_heads = int(self._config.linear_num_key_heads)
        value_heads = int(self._config.linear_num_value_heads)
        if value_heads % key_heads != 0:
            raise WeightManifestError(
                "Qwen3-Next GDN value heads are not divisible by key heads"
            )
        values_per_key = value_heads // key_heads
        return 2 * int(self._config.linear_key_head_dim) + 2 * values_per_key * int(
            self._config.linear_value_head_dim
        )

    def _ba_group_width(self) -> int:
        key_heads = int(self._config.linear_num_key_heads)
        value_heads = int(self._config.linear_num_value_heads)
        if value_heads % key_heads != 0:
            raise WeightManifestError(
                "Qwen3-Next GDN value heads are not divisible by key heads"
            )
        return 2 * value_heads // key_heads

    @staticmethod
    def _retag(view: LogicalTensorView) -> LogicalTensorView:
        prefix = "sglang:qwen3.5:"
        if not view.layout_fingerprint.startswith(prefix):
            raise WeightManifestError(
                f"unexpected inherited Qwen layout: {view.layout_fingerprint}"
            )
        return msgspec.structs.replace(
            view,
            layout_fingerprint=(
                "sglang:qwen3-next:" + view.layout_fingerprint.removeprefix(prefix)
            ),
        )
