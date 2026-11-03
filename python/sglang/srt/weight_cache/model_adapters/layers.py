# SPDX-License-Identifier: Apache-2.0
"""Layout helpers for explicitly supported native, unquantized SGLang layers."""

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from ..weight_layout import WeightLayoutBuilder
from ..weight_runtime_manifest import WeightManifestError, WeightPartition


class LayerLayoutBuilder(WeightLayoutBuilder):
    @staticmethod
    def _shape(tensor, expected):
        if tuple(tensor.shape) != tuple(expected):
            raise WeightManifestError(
                f"unexpected weight shape: {tuple(tensor.shape)} != {tuple(expected)}"
            )

    @staticmethod
    def _linear(module, expected_type):
        if (
            type(module) is not expected_type
            or type(module.quant_method) is not UnquantizedLinearMethod
        ):
            raise WeightManifestError(
                f"unsupported linear layout: {type(module).__name__}"
            )

    def replicated(self, tensor, logical_name):
        self.region(tensor, logical_name, global_shape=tuple(tensor.shape))

    def replicated_linear(self, module, name):
        self._linear(module, ReplicatedLinear)
        self._shape(module.weight, (module.output_size, module.input_size))
        self.replicated(module.weight, name + ".weight")
        if module.bias is not None:
            self._shape(module.bias, (module.output_size,))
            self.replicated(module.bias, name + ".bias")

    def column_parallel(self, module, name):
        self._linear(module, ColumnParallelLinear)
        rank, size = get_group_rank_size(module.tp_group)
        if module.output_size % size:
            raise WeightManifestError(f"uneven column partition: {name}")
        rows = module.output_size // size
        self._shape(module.weight, (rows, module.input_size))
        self.region(
            module.weight,
            name + ".weight",
            global_shape=(module.output_size, module.input_size),
            global_offset=(rank * rows, 0),
            partition=WeightPartition(rank=rank, size=size, dim=0),
        )
        if module.bias is not None:
            self._shape(module.bias, (rows,))
            self.region(
                module.bias,
                name + ".bias",
                global_shape=(module.output_size,),
                global_offset=(rank * rows,),
                partition=WeightPartition(rank=rank, size=size, dim=0),
            )

    def row_parallel(self, module, name):
        self._linear(module, RowParallelLinear)
        rank, size = get_group_rank_size(module.tp_group)
        if module.input_size % size:
            raise WeightManifestError(f"uneven row partition: {name}")
        cols = module.input_size // size
        self._shape(module.weight, (module.output_size, cols))
        self.region(
            module.weight,
            name + ".weight",
            global_shape=(module.output_size, module.input_size),
            global_offset=(0, rank * cols),
            partition=WeightPartition(rank=rank, size=size, dim=1),
        )
        if module.bias is not None:
            self._shape(module.bias, (module.output_size,))
            self.replicated(module.bias, name + ".bias")

    def _segments(self, module, segments):
        """Each segment also declares its effective partition and replicas."""
        local_rows = sum(segment[3] for segment in segments)
        self._shape(module.weight, (local_rows, module.input_size))
        if module.bias is not None:
            self._shape(module.bias, (local_rows,))
        cursor = 0
        for name, total, start, rows, partition in segments:
            self.region(
                module.weight,
                name + ".weight",
                global_shape=(total, module.input_size),
                global_offset=(start, 0),
                local_shape=(rows, module.input_size),
                byte_offset=cursor * module.input_size * module.weight.element_size(),
                partition=partition,
            )
            if module.bias is not None:
                self.region(
                    module.bias,
                    name + ".bias",
                    global_shape=(total,),
                    global_offset=(start,),
                    local_shape=(rows,),
                    byte_offset=cursor * module.bias.element_size(),
                    partition=partition,
                )
            cursor += rows

    def merged_column(self, module, logical_names):
        self._linear(module, MergedColumnParallelLinear)
        rank, size = get_group_rank_size(module.tp_group)
        if len(logical_names) != len(module.output_sizes) or any(
            n % size for n in module.output_sizes
        ):
            raise WeightManifestError("invalid merged-column segments")
        self._segments(
            module,
            [
                (
                    name,
                    n,
                    rank * (n // size),
                    n // size,
                    WeightPartition(rank=rank, size=size, dim=0),
                )
                for name, n in zip(logical_names, module.output_sizes)
            ],
        )

    def qkv_parallel(self, module, logical_names):
        self._linear(module, QKVParallelLinear)
        if len(logical_names) != 3:
            raise WeightManifestError("QKV requires three logical names")
        qrank, qsize = get_group_rank_size(module.tp_group)
        kvrank, kvsize = get_group_rank_size(module.kv_tp_group)
        kvrank //= module.num_kv_head_replicas
        kvsize //= module.num_kv_head_replicas
        self._segments(
            module,
            [
                (
                    logical_names[0],
                    module.total_num_heads * module.head_size,
                    qrank * module.q_proj_shard_size,
                    module.q_proj_shard_size,
                    WeightPartition(rank=qrank, size=qsize, dim=0),
                ),
                (
                    logical_names[1],
                    module.total_num_kv_heads * module.head_size,
                    kvrank * module.kv_proj_shard_size,
                    module.kv_proj_shard_size,
                    WeightPartition(rank=kvrank, size=kvsize, dim=0),
                ),
                (
                    logical_names[2],
                    module.total_num_kv_heads * module.v_head_size,
                    kvrank * module.v_proj_shard_size,
                    module.v_proj_shard_size,
                    WeightPartition(rank=kvrank, size=kvsize, dim=0),
                ),
            ],
        )

    def vocab_parallel(self, module, name):
        if (
            type(module) not in (VocabParallelEmbedding, ParallelLMHead)
            or type(module.quant_method) is not UnquantizedEmbeddingMethod
        ):
            raise WeightManifestError(f"unsupported vocabulary layout: {name}")
        if module.num_embeddings != module.org_vocab_size:
            raise WeightManifestError(
                "added LoRA vocabulary is unsupported for reshard"
            )
        self._shape(
            module.weight, (module.num_embeddings_per_partition, module.embedding_dim)
        )
        indices = module.shard_indices
        rows = indices.num_org_elements
        rank, size = get_group_rank_size(module.tp_group)
        partition = WeightPartition(rank=rank, size=size, dim=0)
        self.declare(
            module.weight,
            name,
            global_shape=(module.org_vocab_size, module.embedding_dim),
            partition=partition,
        )
        if rows:
            self.region(
                module.weight,
                name,
                global_shape=(module.org_vocab_size, module.embedding_dim),
                global_offset=(indices.org_vocab_start_index, 0),
                local_shape=(rows, module.embedding_dim),
                partition=partition,
            )
        row_bytes = module.embedding_dim * module.weight.element_size()
        self.initialize(
            module.weight,
            byte_offset=rows * row_bytes,
            nbytes=(module.num_embeddings_per_partition - rows) * row_bytes,
        )

    def mlp(self, module, name):
        self.merged_column(
            module.gate_up_proj, (name + ".gate_proj", name + ".up_proj")
        )
        self.row_parallel(module.down_proj, name + ".down_proj")

    def rotary(self, module, name):
        # These constructor-generated caches do not depend on checkpoint bytes.
        for suffix, _ in module.named_buffers(remove_duplicate=False):
            if suffix not in ("cos_sin_cache", "inv_freq"):
                raise WeightManifestError(f"unsupported rotary buffer: {name}.{suffix}")
            self.local_buffer(name + "." + suffix)

    def fused_moe(self, module, name):
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        method = module.quant_method
        if (
            type(module) is not FusedMoE
            or type(method) is not UnquantizedFusedMoEMethod
            or not method.runner.runner_backend.is_triton()
            or method.use_triton_kernels
            or method.use_flashinfer_trtllm_moe
            or method.use_deep_gemm
            or method.with_bias
            or getattr(method, "load_up_proj_weight_first", False)
            or method.w13_swiglu_interleaved
            or module.num_fused_shared_experts
            or module._expert_storage_rank != module.moe_ep_rank
            or get_global_expert_location_metadata() is not None
        ):
            raise WeightManifestError(
                f"unsupported MoE layout: {name}; use the triton backend without EPLB, "
                "shared-expert fusion or SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED"
            )
        local_experts = module.num_local_experts
        intermediate = module.intermediate_size_per_partition
        hidden = module.hidden_size
        self._shape(module.w13_weight, (local_experts, 2 * intermediate, hidden))
        self._shape(module.w2_weight, (local_experts, hidden, intermediate))
        total_intermediate = intermediate * module.moe_tp_size
        start = intermediate * module.moe_tp_rank
        for slot in range(local_experts):
            expert = module._expert_storage_rank * module._num_local_routed + slot
            if module._map_global_expert_id_to_local_expert_id(expert) != slot:
                raise WeightManifestError(
                    f"unexpected expert ownership: {name}: {expert}"
                )
            prefix = f"{name}.{expert}"
            for half, proj in enumerate(("gate_proj", "up_proj")):
                self.region(
                    module.w13_weight,
                    f"{prefix}.{proj}.weight",
                    global_shape=(total_intermediate, hidden),
                    global_offset=(start, 0),
                    local_shape=(intermediate, hidden),
                    expert_id=expert,
                    partition=WeightPartition(
                        kind="expert",
                        rank=module.moe_tp_rank,
                        size=module.moe_tp_size,
                        dim=0,
                    ),
                    byte_offset=(slot * 2 + half)
                    * intermediate
                    * hidden
                    * module.w13_weight.element_size(),
                )
            self.region(
                module.w2_weight,
                f"{prefix}.down_proj.weight",
                global_shape=(hidden, total_intermediate),
                global_offset=(0, start),
                local_shape=(hidden, intermediate),
                expert_id=expert,
                partition=WeightPartition(
                    kind="expert",
                    rank=module.moe_tp_rank,
                    size=module.moe_tp_size,
                    dim=1,
                ),
                byte_offset=slot
                * hidden
                * intermediate
                * module.w2_weight.element_size(),
            )
