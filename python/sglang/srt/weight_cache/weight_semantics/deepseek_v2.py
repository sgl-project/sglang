from __future__ import annotations

from math import prod
from typing import Any, Sequence

from ..weight_runtime_manifest import (
    LogicalTensorView,
    WeightManifestError,
    WeightParallelTopology,
)
from .qwen3_5 import (
    _canonical_name,
    _itemsize,
    _layer_id,
    _replace_suffix,
    _shape,
)

_LAYOUT_PREFIX = "sglang:deepseek-v2:"

_REPLICATED_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_a_layernorm.weight",
    "kv_a_layernorm.weight",
    "norm.weight",
    "mlp.gate.weight",
    "e_score_correction_bias",
)


def _view(
    *,
    tensor_id: str,
    global_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    partition_dim: int | None,
    byte_offset: int,
    layer_id: int | None,
    expert_id: int | None = None,
    layout: str,
    shard_dims: tuple[int, ...] | None = None,
) -> LogicalTensorView:
    return LogicalTensorView(
        tensor_id=tensor_id,
        global_shape=global_shape,
        global_offset=global_offset,
        local_shape=local_shape,
        partition_dim=partition_dim,
        byte_offset=byte_offset,
        layer_id=layer_id,
        expert_id=expert_id,
        layout_fingerprint=f"{_LAYOUT_PREFIX}{layout}:v1",
        shard_dims=(
            shard_dims
            if shard_dims is not None
            else (() if partition_dim is None else (partition_dim,))
        ),
    )


def _replicated_view(
    *,
    parameter: Any,
    tensor_id: str,
    layer_id: int | None,
    layout: str,
    expected_shape: tuple[int, ...] | None = None,
    byte_offset: int = 0,
    global_shape: tuple[int, ...] | None = None,
) -> LogicalTensorView:
    shape = global_shape if global_shape is not None else _shape(parameter)
    if expected_shape is not None and shape != expected_shape:
        raise WeightManifestError(
            f"DeepSeek replicated tensor shape mismatch: {tensor_id}: {shape}, "
            f"expected {expected_shape}"
        )
    return _view(
        tensor_id=tensor_id,
        global_shape=shape,
        global_offset=(0,) * len(shape),
        local_shape=shape,
        partition_dim=None,
        byte_offset=byte_offset,
        layer_id=layer_id,
        layout=layout,
    )


def _split_dim_zero(
    *,
    parameter: Any,
    tensor_ids: Sequence[str],
    global_extents: Sequence[int],
    rank: int,
    size: int,
    layer_id: int | None,
    layout: str,
) -> tuple[LogicalTensorView, ...]:
    """Describe a dim-0 packed column-parallel parameter.

    The runtime tensor may either be sharded across ``size`` ranks or hold the
    full extent (a module that explicitly opts out of tensor parallelism, such
    as the DeepSeek shared expert on all-to-all MoE backends).
    """
    if len(tensor_ids) != len(global_extents):
        raise WeightManifestError("invalid packed DeepSeek tensor description")
    if size <= 0 or not 0 <= rank < size:
        raise WeightManifestError("invalid DeepSeek parallel rank")
    if any(extent <= 0 or extent % size != 0 for extent in global_extents):
        raise WeightManifestError(
            f"DeepSeek tensor is not evenly partitionable: {tensor_ids[0]}"
        )

    shape = _shape(parameter)
    if not shape:
        raise WeightManifestError(f"DeepSeek tensor has no axes: {tensor_ids[0]}")
    tail = shape[1:]
    local_extents = tuple(extent // size for extent in global_extents)
    if shape[0] == sum(local_extents):
        extents = local_extents
        offsets = tuple(rank * extent for extent in local_extents)
    elif shape[0] == sum(global_extents):
        extents = tuple(global_extents)
        offsets = (0,) * len(global_extents)
    else:
        raise WeightManifestError(
            f"DeepSeek packed tensor shape mismatch: {tensor_ids[0]}: {shape}, "
            f"expected leading extent {sum(local_extents)} or {sum(global_extents)}"
        )

    byte_offset = 0
    result = []
    for tensor_id, global_extent, extent, offset in zip(
        tensor_ids, global_extents, extents, offsets
    ):
        local_shape = (extent, *tail)
        result.append(
            _view(
                tensor_id=tensor_id,
                global_shape=(global_extent, *tail),
                global_offset=(offset, *((0,) * len(tail))),
                local_shape=local_shape,
                partition_dim=0,
                byte_offset=byte_offset,
                layer_id=layer_id,
                layout=layout,
            )
        )
        byte_offset += prod(local_shape) * _itemsize(parameter)
    return tuple(result)


def _row_parallel_view(
    *,
    parameter: Any,
    tensor_id: str,
    global_shape: tuple[int, int],
    rank: int,
    size: int,
    layer_id: int | None,
    layout: str,
) -> tuple[LogicalTensorView, ...]:
    if size <= 0 or not 0 <= rank < size or global_shape[1] % size != 0:
        raise WeightManifestError(f"DeepSeek tensor is not TP divisible: {tensor_id}")
    shape = _shape(parameter)
    sharded_shape = (global_shape[0], global_shape[1] // size)
    if shape == sharded_shape:
        offset = rank * sharded_shape[1]
    elif shape == global_shape:
        offset = 0
    else:
        raise WeightManifestError(
            f"DeepSeek row tensor shape mismatch: {tensor_id}: {shape}, "
            f"expected {sharded_shape} or {global_shape}"
        )
    return (
        _view(
            tensor_id=tensor_id,
            global_shape=global_shape,
            global_offset=(0, offset),
            local_shape=shape,
            partition_dim=1,
            byte_offset=0,
            layer_id=layer_id,
            layout=layout,
        ),
    )


class DeepseekV2WeightSemanticsAdapter:
    """Describe DeepSeek V2/V3 runtime packing in canonical global coordinates.

    The adapter covers Multi-head Latent Attention (MLA), the routed MoE
    experts, the loose or fused shared expert, and the dense MLP layers that
    precede ``first_k_dense_replace``. Derived MLA state (``w_kc`` / ``w_vc``)
    is not a runtime parameter; the loader rebuilds it from ``kv_b_proj``
    through ``post_load_weights`` after the transfer completes.
    """

    def __init__(
        self,
        *,
        config: Any,
        up_first_w13_parameter_ids: Sequence[int] = (),
        num_fused_shared_experts: int = 0,
        embed_vocab_group: str = "tp",
        lm_head_vocab_group: str = "tp",
    ) -> None:
        self._config = config
        self._up_first_w13_parameter_ids = frozenset(up_first_w13_parameter_ids)
        if num_fused_shared_experts not in (0, 1):
            raise WeightManifestError(
                "DeepSeek weight manifests support at most one fused shared expert"
            )
        self._num_fused_shared_experts = num_fused_shared_experts
        for vocab_group in (embed_vocab_group, lm_head_vocab_group):
            if vocab_group not in ("tp", "attn_tp", "replicated"):
                raise WeightManifestError(
                    f"invalid vocab-parallel group: {vocab_group!r}"
                )
        self._embed_vocab_group = embed_vocab_group
        self._lm_head_vocab_group = lm_head_vocab_group

    def _vocab_views(
        self,
        *,
        tensor_id: str,
        parameter: Any,
        topology: WeightParallelTopology,
        layer_id: int | None,
    ) -> tuple[LogicalTensorView, ...]:
        """Describe an embedding / lm_head shard in its vocab-parallel group.

        The group is discovered from the constructed modules: the full
        tensor-parallel world by default, the attention-TP group under DP
        attention, or no sharding at all when the table is replicated on
        every rank.
        """
        vocab_group = (
            self._embed_vocab_group
            if tensor_id == "embed_tokens.weight"
            else self._lm_head_vocab_group
        )
        if vocab_group == "attn_tp":
            rank, size = topology.attention_tp_rank, topology.attention_tp_size
        else:
            rank, size = topology.tp_rank, topology.tp_size
        if vocab_group == "replicated":
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=tensor_id,
                    layer_id=layer_id,
                    layout="vocab-parallel",
                ),
            )
        return _split_dim_zero(
            parameter=parameter,
            tensor_ids=(tensor_id,),
            global_extents=(int(self._config.vocab_size),),
            rank=rank,
            size=size,
            layer_id=layer_id,
            layout="vocab-parallel",
        )

    # ------------------------------------------------------------------
    # configuration helpers
    # ------------------------------------------------------------------
    def _hidden_size(self) -> int:
        return int(self._config.hidden_size)

    def _num_attention_heads(self) -> int:
        return int(self._config.num_attention_heads)

    def _qk_head_dim(self) -> int:
        return int(self._config.qk_nope_head_dim) + int(self._config.qk_rope_head_dim)

    def _v_head_dim(self) -> int:
        return int(self._config.v_head_dim)

    def _kv_lora_rank(self) -> int:
        return int(self._config.kv_lora_rank)

    def _q_lora_rank(self) -> int | None:
        value = getattr(self._config, "q_lora_rank", None)
        return None if value is None else int(value)

    def _num_routed_experts(self) -> int:
        return int(self._config.n_routed_experts)

    def _moe_intermediate_size(self) -> int:
        return int(self._config.moe_intermediate_size)

    def _num_shared_experts(self) -> int:
        value = getattr(self._config, "n_shared_experts", None)
        return 0 if value is None else int(value)

    def _index_n_heads(self) -> int:
        return int(self._config.index_n_heads)

    def _index_head_dim(self) -> int:
        return int(self._config.index_head_dim)

    def _mlp_intermediate_size(self, name: str) -> int:
        if "shared_experts" in name:
            shared = self._num_shared_experts()
            if shared <= 0:
                raise WeightManifestError(
                    "DeepSeek shared expert weights require n_shared_experts"
                )
            return self._moe_intermediate_size() * shared
        return int(self._config.intermediate_size)

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        canonical_names = tuple(
            dict.fromkeys(self._canonical_name(name) for name in names)
        )
        name = min(canonical_names)
        layer_id = _layer_id(name)

        if ".indexer." in name:
            return self._dsa_indexer(
                name=name,
                parameter=parameter,
                layer_id=layer_id,
            )
        if name.endswith("experts.w13_weight"):
            return self._moe_w13(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("experts.w2_weight"):
            return self._moe_w2(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("fused_qkv_a_proj_with_mqa.weight"):
            return self._fused_qkv_a_proj(
                name=name,
                parameter=parameter,
                layer_id=layer_id,
            )
        if name.endswith("kv_a_proj_with_mqa.weight"):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="mla-kv-a",
                    expected_shape=(
                        self._kv_lora_rank() + int(self._config.qk_rope_head_dim),
                        self._hidden_size(),
                    ),
                ),
            )
        if name.endswith("q_proj.weight"):
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(self._num_attention_heads() * self._qk_head_dim(),),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="mla-q",
            )
        if name.endswith("q_b_proj.weight"):
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(self._num_attention_heads() * self._qk_head_dim(),),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="mla-q-b",
            )
        if name.endswith("kv_b_proj.weight"):
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(
                    self._num_attention_heads()
                    * (int(self._config.qk_nope_head_dim) + self._v_head_dim()),
                ),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="mla-kv-b",
            )
        if name.endswith("o_proj.weight"):
            return _row_parallel_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(
                    self._hidden_size(),
                    self._num_attention_heads() * self._v_head_dim(),
                ),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="attention-row",
            )
        if name.endswith("gate_up_proj.weight"):
            intermediate = self._mlp_intermediate_size(name)
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(
                    _replace_suffix(name, "gate_up_proj.weight", "gate_proj.weight"),
                    _replace_suffix(name, "gate_up_proj.weight", "up_proj.weight"),
                ),
                global_extents=(intermediate, intermediate),
                rank=topology.tp_rank,
                size=topology.tp_size,
                layer_id=layer_id,
                layout="gate-up",
            )
        if name.endswith("down_proj.weight"):
            return _row_parallel_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(
                    self._hidden_size(),
                    self._mlp_intermediate_size(name),
                ),
                rank=topology.tp_rank,
                size=topology.tp_size,
                layer_id=layer_id,
                layout="row-parallel",
            )
        if name in ("embed_tokens.weight", "lm_head.weight"):
            return tuple(
                view
                for tensor_id in sorted(canonical_names)
                if tensor_id in ("embed_tokens.weight", "lm_head.weight")
                for view in self._vocab_views(
                    tensor_id=tensor_id,
                    parameter=parameter,
                    topology=topology,
                    layer_id=None,
                )
            )
        if name.endswith(_REPLICATED_SUFFIXES):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="replicated",
                ),
            )
        raise WeightManifestError(f"unsupported DeepSeek parameter: {names[0]}")

    # ------------------------------------------------------------------
    # DSA indexer (DeepSeek V3.2 sparse attention)
    # ------------------------------------------------------------------
    def _dsa_indexer(
        self, *, name: str, parameter: Any, layer_id: int | None
    ) -> tuple[LogicalTensorView, ...]:
        n_heads = self._index_n_heads()
        head_dim = self._index_head_dim()
        hidden = self._hidden_size()
        if name.endswith("wk_weights_proj.weight"):
            # CUDA-only runtime fusion of the bf16 indexer inputs: the top
            # ``index_head_dim`` rows hold ``wk`` (dequantized at load time when
            # the checkpoint is block-FP8) and the bottom ``index_n_heads`` rows
            # hold ``weights_proj``. Export both halves under their canonical
            # names so fused and unfused runtimes stay interchangeable.
            shape = _shape(parameter)
            expected = (head_dim + n_heads, hidden)
            if shape != expected:
                raise WeightManifestError(
                    f"DSA fused indexer tensor shape mismatch: {name}: {shape}, "
                    f"expected {expected}"
                )
            itemsize = _itemsize(parameter)
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=_replace_suffix(
                        name, "wk_weights_proj.weight", "wk.weight"
                    ),
                    layer_id=layer_id,
                    layout="dsa-indexer-wk",
                    global_shape=(head_dim, hidden),
                    byte_offset=0,
                ),
                _replicated_view(
                    parameter=parameter,
                    tensor_id=_replace_suffix(
                        name, "wk_weights_proj.weight", "weights_proj.weight"
                    ),
                    layer_id=layer_id,
                    layout="dsa-indexer-weights-proj",
                    global_shape=(n_heads, hidden),
                    byte_offset=head_dim * hidden * itemsize,
                ),
            )
        if name.endswith("wq_b.weight"):
            q_lora_rank = self._q_lora_rank()
            if q_lora_rank is None:
                raise WeightManifestError(
                    "DSA indexer wq_b requires q_lora_rank in the model config"
                )
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="dsa-indexer-wq-b",
                    expected_shape=(n_heads * head_dim, q_lora_rank),
                ),
            )
        if name.endswith("wk.weight"):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="dsa-indexer-wk",
                    expected_shape=(head_dim, hidden),
                ),
            )
        if name.endswith("weights_proj.weight"):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="dsa-indexer-weights-proj",
                    expected_shape=(n_heads, hidden),
                ),
            )
        if name.endswith(("k_norm.weight", "k_norm.bias")):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="replicated",
                ),
            )
        raise WeightManifestError(f"unsupported DSA indexer parameter: {name}")

    # ------------------------------------------------------------------
    # name canonicalization
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_name(name: str) -> str:
        # `DeepseekV2AttentionMLA` re-attaches `kv_b_proj` under `attn_mha`, so the
        # same storage is reachable through two module paths. Collapse the alias so
        # the logical tensor ID does not depend on which name sorts first.
        canonical = _canonical_name(name)
        return canonical.replace(".attn_mha.kv_b_proj.", ".kv_b_proj.")

    # ------------------------------------------------------------------
    # MLA fused low-rank projection
    # ------------------------------------------------------------------
    def _fused_qkv_a_proj(
        self, *, name: str, parameter: Any, layer_id: int | None
    ) -> tuple[LogicalTensorView, ...]:
        q_lora_rank = self._q_lora_rank()
        if q_lora_rank is None:
            raise WeightManifestError(
                "DeepSeek fused QKV-A projection requires q_lora_rank"
            )
        kv_extent = self._kv_lora_rank() + int(self._config.qk_rope_head_dim)
        hidden = self._hidden_size()
        shape = _shape(parameter)
        expected = (q_lora_rank + kv_extent, hidden)
        if shape != expected:
            raise WeightManifestError(
                f"DeepSeek fused QKV-A tensor shape mismatch: {shape}, "
                f"expected {expected}"
            )
        itemsize = _itemsize(parameter)
        # The fused projection is replicated, so each component is a full logical
        # tensor that starts at its own byte offset inside the packed parameter.
        return (
            _replicated_view(
                parameter=parameter,
                tensor_id=_replace_suffix(
                    name, "fused_qkv_a_proj_with_mqa.weight", "q_a_proj.weight"
                ),
                layer_id=layer_id,
                layout="mla-q-a",
                global_shape=(q_lora_rank, hidden),
                byte_offset=0,
            ),
            _replicated_view(
                parameter=parameter,
                tensor_id=_replace_suffix(
                    name,
                    "fused_qkv_a_proj_with_mqa.weight",
                    "kv_a_proj_with_mqa.weight",
                ),
                layer_id=layer_id,
                layout="mla-kv-a",
                global_shape=(kv_extent, hidden),
                byte_offset=q_lora_rank * hidden * itemsize,
            ),
        )

    # ------------------------------------------------------------------
    # MoE experts
    # ------------------------------------------------------------------
    def _routed_expert_ids(
        self, *, parameter: Any, topology: WeightParallelTopology
    ) -> tuple[int, ...]:
        num_experts = self._num_routed_experts()
        if num_experts % topology.ep_size != 0:
            raise WeightManifestError(
                "DeepSeek experts are not evenly EP partitionable"
            )
        local_experts = num_experts // topology.ep_size
        expected_slots = local_experts + self._num_fused_shared_experts
        shape = _shape(parameter)
        if not shape or shape[0] != expected_slots:
            raise WeightManifestError(
                f"DeepSeek local expert count mismatch: {shape}, "
                f"expected {expected_slots} slots"
            )
        start = topology.ep_rank * local_experts
        return tuple(range(start, start + local_experts))

    def _w13_components(self, parameter: Any) -> tuple[str, str]:
        return (
            ("up_proj", "gate_proj")
            if id(parameter) in self._up_first_w13_parameter_ids
            else ("gate_proj", "up_proj")
        )

    def _fused_shared_intermediate(self) -> int:
        shared = self._num_shared_experts()
        intermediate = self._moe_intermediate_size() * max(shared, 1)
        if intermediate != self._moe_intermediate_size():
            raise WeightManifestError(
                "DeepSeek fused shared expert must match the routed intermediate size"
            )
        return intermediate

    def _moe_w13(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        expert_ids = self._routed_expert_ids(parameter=parameter, topology=topology)
        shape = _shape(parameter)
        intermediate = self._moe_intermediate_size()
        if intermediate % topology.moe_tp_size != 0:
            raise WeightManifestError("DeepSeek expert tensor is not TP divisible")
        local_intermediate = intermediate // topology.moe_tp_size
        expected = (
            len(expert_ids) + self._num_fused_shared_experts,
            local_intermediate * 2,
            self._hidden_size(),
        )
        if shape != expected:
            raise WeightManifestError(
                f"DeepSeek w13 tensor shape mismatch: {shape}, expected {expected}"
            )
        prefix = name[: -len("experts.w13_weight")]
        expert_bytes = prod(shape[1:]) * _itemsize(parameter)
        component_bytes = local_intermediate * shape[2] * _itemsize(parameter)
        components = self._w13_components(parameter)
        num_experts = self._num_routed_experts()

        views = []
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
        if self._num_fused_shared_experts:
            shared_intermediate = self._fused_shared_intermediate()
            shared_base = len(expert_ids) * expert_bytes
            for component_index, component in enumerate(components):
                views.append(
                    _view(
                        tensor_id=f"{prefix}shared_experts.{component}.weight",
                        global_shape=(shared_intermediate, shape[2]),
                        global_offset=(topology.moe_tp_rank * local_intermediate, 0),
                        local_shape=(local_intermediate, shape[2]),
                        partition_dim=0,
                        byte_offset=shared_base + component_index * component_bytes,
                        layer_id=layer_id,
                        expert_id=None,
                        layout="gate-up",
                    )
                )
        return tuple(views)

    def _moe_w2(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        expert_ids = self._routed_expert_ids(parameter=parameter, topology=topology)
        shape = _shape(parameter)
        intermediate = self._moe_intermediate_size()
        if intermediate % topology.moe_tp_size != 0:
            raise WeightManifestError("DeepSeek expert tensor is not TP divisible")
        local_intermediate = intermediate // topology.moe_tp_size
        expected = (
            len(expert_ids) + self._num_fused_shared_experts,
            self._hidden_size(),
            local_intermediate,
        )
        if shape != expected:
            raise WeightManifestError(
                f"DeepSeek w2 tensor shape mismatch: {shape}, expected {expected}"
            )
        prefix = name[: -len("experts.w2_weight")]
        expert_bytes = prod(shape[1:]) * _itemsize(parameter)
        num_experts = self._num_routed_experts()

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
        if self._num_fused_shared_experts:
            shared_intermediate = self._fused_shared_intermediate()
            views.append(
                _view(
                    tensor_id=f"{prefix}shared_experts.down_proj.weight",
                    global_shape=(shape[1], shared_intermediate),
                    global_offset=(0, topology.moe_tp_rank * local_intermediate),
                    local_shape=(shape[1], local_intermediate),
                    partition_dim=1,
                    byte_offset=len(expert_ids) * expert_bytes,
                    layer_id=layer_id,
                    expert_id=None,
                    layout="row-parallel",
                )
            )
        return tuple(views)
