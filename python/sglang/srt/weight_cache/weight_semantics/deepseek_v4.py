from __future__ import annotations

from typing import Any

from ..weight_runtime_manifest import (
    LogicalTensorView,
    WeightManifestError,
    WeightParallelTopology,
)
from .deepseek_v2 import (
    DeepseekV2WeightSemanticsAdapter,
    _replicated_view,
    _row_parallel_view,
    _split_dim_zero,
)
from .qwen3_5 import (
    _itemsize,
    _layer_id,
    _replace_suffix,
    _shape,
)

# Parameters that are plain fp32 nn.Parameter attributes (no ``.weight``
# suffix) and replicated on every rank: the per-layer mHC mixing state, the
# model-level mHC head state, and the attention sink logits.
_V4_REPLICATED_SUFFIXES = (
    "hc_attn_fn",
    "hc_ffn_fn",
    "hc_attn_base",
    "hc_ffn_base",
    "hc_attn_scale",
    "hc_ffn_scale",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
    "attn_sink",
    "q_norm.weight",
    "kv_norm.weight",
    "topk.tid2eid",
)


class DeepseekV4WeightSemanticsAdapter(DeepseekV2WeightSemanticsAdapter):
    """Describe DeepSeek V4 runtime packing in canonical global coordinates.

    V4 replaces MLA with a grouped MQA + LoRA attention (``wq_a``/``wqkv_a``,
    ``wq_b``, ``wkv``, ``wo_a``/``wo_b``), adds mHC hyper-column mixing
    parameters, sliding-window compressors with a C4 indexer on
    ``compress_ratio == 4`` layers, and reuses the DeepSeek V2 MoE stack. The
    MoE, shared expert, gate, embedding, and lm_head branches are inherited
    from :class:`DeepseekV2WeightSemanticsAdapter`.

    ``Compressor.ape`` is described in its *runtime* (hotfixed) layout: the
    load-time ``apply_ape_hotfix`` permutation is not idempotent, so the
    remote-instance loader marks transferred compressors as already converted
    instead of re-permuting (see ``mark_weights_in_runtime_layout``).
    """

    # ------------------------------------------------------------------
    # configuration helpers
    # ------------------------------------------------------------------
    def _mqa_head_dim(self) -> int:
        configured = getattr(self._config, "head_dim", None)
        if configured:
            return int(configured)
        return self._qk_head_dim()

    def _o_groups(self) -> int:
        return int(self._config.o_groups)

    def _o_lora_rank(self) -> int:
        return int(self._config.o_lora_rank)

    def _required_q_lora_rank(self) -> int:
        q_lora_rank = self._q_lora_rank()
        if q_lora_rank is None:
            raise WeightManifestError(
                "DeepSeek V4 attention requires q_lora_rank in the model config"
            )
        return q_lora_rank

    def _compress_ratio(self, name: str, layer_id: int | None) -> int | None:
        compress_ratios = getattr(self._config, "compress_ratios", None)
        if not compress_ratios or layer_id is None or layer_id >= len(compress_ratios):
            return None
        # The C4 indexer embeds its own ratio-4 compressor regardless of the
        # attention compressor's ratio.
        if ".indexer." in name:
            return 4
        return int(compress_ratios[layer_id])

    def _compressor_head_dim(self, name: str) -> int:
        return self._index_head_dim() if ".indexer." in name else self._mqa_head_dim()

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

        # Compressors live both directly under self_attn and nested inside the
        # C4 indexer, so this branch must run before the inherited ``.indexer.``
        # dispatch.
        if ".compressor." in name:
            return self._compressor(
                name=name,
                parameter=parameter,
                layer_id=layer_id,
            )
        if name.endswith("wqkv_a.weight"):
            return self._fused_wqkv_a(
                name=name,
                parameter=parameter,
                layer_id=layer_id,
            )
        if name.endswith("wq_a.weight"):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="v4-mqa-wq-a",
                    expected_shape=(
                        self._required_q_lora_rank(),
                        self._hidden_size(),
                    ),
                ),
            )
        if name.endswith("wkv.weight"):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="v4-mqa-wkv",
                    expected_shape=(self._mqa_head_dim(), self._hidden_size()),
                ),
            )
        if name.endswith("wq_b.weight") and ".indexer." not in name:
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(self._num_attention_heads() * self._mqa_head_dim(),),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="v4-mqa-wq-b",
            )
        if name.endswith("wo_a.weight"):
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(self._o_groups() * self._o_lora_rank(),),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="v4-mqa-wo-a",
            )
        if name.endswith("wo_b.weight"):
            return _row_parallel_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(
                    self._hidden_size(),
                    self._o_groups() * self._o_lora_rank(),
                ),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="v4-mqa-wo-b",
            )
        if name.endswith(_V4_REPLICATED_SUFFIXES):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="replicated",
                ),
            )
        return super().describe_parameter(
            names=names,
            parameter=parameter,
            topology=topology,
        )

    # ------------------------------------------------------------------
    # fused low-rank QKV-A projection
    # ------------------------------------------------------------------
    def _fused_wqkv_a(
        self, *, name: str, parameter: Any, layer_id: int | None
    ) -> tuple[LogicalTensorView, ...]:
        q_lora_rank = self._required_q_lora_rank()
        head_dim = self._mqa_head_dim()
        hidden = self._hidden_size()
        shape = _shape(parameter)
        expected = (q_lora_rank + head_dim, hidden)
        if shape != expected:
            raise WeightManifestError(
                f"DeepSeek V4 fused wqkv_a tensor shape mismatch: {shape}, "
                f"expected {expected}"
            )
        itemsize = _itemsize(parameter)
        # Replicated packing of wq_a followed by wkv; export both halves under
        # their canonical names so fused and unfused runtimes interoperate.
        return (
            _replicated_view(
                parameter=parameter,
                tensor_id=_replace_suffix(name, "wqkv_a.weight", "wq_a.weight"),
                layer_id=layer_id,
                layout="v4-mqa-wq-a",
                global_shape=(q_lora_rank, hidden),
                byte_offset=0,
            ),
            _replicated_view(
                parameter=parameter,
                tensor_id=_replace_suffix(name, "wqkv_a.weight", "wkv.weight"),
                layer_id=layer_id,
                layout="v4-mqa-wkv",
                global_shape=(head_dim, hidden),
                byte_offset=q_lora_rank * hidden * itemsize,
            ),
        )

    # ------------------------------------------------------------------
    # sliding-window compressor (attention-level and inside the C4 indexer)
    # ------------------------------------------------------------------
    def _compressor(
        self, *, name: str, parameter: Any, layer_id: int | None
    ) -> tuple[LogicalTensorView, ...]:
        head_dim = self._compressor_head_dim(name)
        ratio = self._compress_ratio(name, layer_id)
        coff = None if ratio is None else (2 if ratio == 4 else 1)
        if name.endswith("compressor.ape"):
            # ``ape`` is stored in the runtime (post ``apply_ape_hotfix``)
            # layout on every rank; describe that layout verbatim. The distinct
            # fingerprint keeps checkpoint-layout descriptions from ever being
            # planned against this tensor.
            expected = None if coff is None else (ratio, coff * head_dim)
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="v4-compressor-ape-runtime",
                    expected_shape=expected,
                ),
            )
        if name.endswith("wkv_gate.weight"):
            expected = (
                None if coff is None else (2 * coff * head_dim, self._hidden_size())
            )
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="v4-compressor-wkv-gate",
                    expected_shape=expected,
                ),
            )
        if name.endswith("norm.weight"):
            return (
                _replicated_view(
                    parameter=parameter,
                    tensor_id=name,
                    layer_id=layer_id,
                    layout="replicated",
                    expected_shape=(head_dim,),
                ),
            )
        raise WeightManifestError(
            f"unsupported DeepSeek V4 compressor parameter: {name}"
        )
