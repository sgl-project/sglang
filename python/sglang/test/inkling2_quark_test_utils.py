"""Test helpers for the Inkling-2 Quark exports: the quantization_config of the four
exports, the tensors of a tiny export, the real attention at the dimensions of
Inkling-2-Mini, and a recorder of hidden copies of the packed weights."""

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.test.inkling2_test_utils import precision_group

VARIANTS = ("UINT4-AWQ", "MXFP4-RTN", "UINT4-AWQ-AttnFP8", "MXFP4-AttnFP8")
EXPERT_FORMAT = {
    "UINT4-AWQ": "uint4_awq",
    "MXFP4-RTN": "mxfp4",
    "UINT4-AWQ-AttnFP8": "uint4_awq",
    "MXFP4-AttnFP8": "mxfp4",
}
ATTENTION_FP8_LINEARS = ("wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn")


def has_fp8_attention(variant: str) -> bool:
    return variant.endswith("AttnFP8")


def _uint4_weight_spec():
    return dict(
        block_size=None,
        ch_axis=-1,
        clipped_ste=False,
        dtype="uint4",
        enable_buffer_reuse=False,
        group_size=128,
        is_dynamic=False,
        is_scale_quant=False,
        max_input_numel=4194304,
        mx_element_dtype=None,
        observer_cls="PerGroupMinMaxObserver",
        qscheme="per_group",
        round_method="half_even",
        scale_calculation_mode=None,
        scale_format=None,
        scale_type="float",
        symmetric=False,
        zero_point_type="int32",
    )


def _mxfp4_spec(is_dynamic):
    return dict(
        block_size=None,
        ch_axis=-1,
        clipped_ste=False,
        dtype="fp4",
        enable_buffer_reuse=False,
        group_size=32,
        is_dynamic=is_dynamic,
        is_scale_quant=False,
        max_input_numel=4194304,
        mx_element_dtype=None,
        observer_cls="PerBlockMXObserver",
        qscheme="per_group",
        round_method="half_even",
        scale_calculation_mode="even",
        scale_format="e8m0",
        scale_type="float",
        symmetric=None,
        zero_point_type="int32",
    )


def _fp8_attention_spec():
    common = dict(
        block_size=None,
        clipped_ste=False,
        dtype="fp8_e4m3",
        enable_buffer_reuse=False,
        group_size=None,
        is_scale_quant=False,
        max_input_numel=4194304,
        mx_element_dtype=None,
        observer_cls="PerChannelMinMaxObserver",
        qscheme="per_channel",
        round_method="half_even",
        scale_calculation_mode=None,
        scale_format=None,
        scale_type="float",
        symmetric=True,
        zero_point_type="int32",
    )
    return dict(
        bias=None,
        input_tensors=dict(common, ch_axis=1, is_dynamic=True, per_token=True),
        output_tensors=None,
        target_device=None,
        weight=dict(common, ch_axis=0, is_dynamic=False),
    )


def quantization_config(variant: str) -> dict:
    """The quantization_config of the config.json of an export, key for key."""
    uint4, attn_fp8 = variant.startswith("UINT4"), has_fp8_attention(variant)
    exclude = [
        "model.llm.unembed",
        "model.llm.layers.*.mlp.gate",
        "model.llm.layers.*.controllers.*",
        "model.llm.layers.*.attn.attn_sink_proj",
        "model.llm.layers.*.attn.wr_du",
        "model.llm.layers.*.attn.k_sconv*",
        "model.llm.layers.*.attn.v_sconv*",
        "model.llm.layers.*.attn.rel_logits_proj*",
        "model.visual*",
        "mtp*",
    ]
    if not attn_fp8:
        exclude.append("model.llm.layers.*.attn.*")
    return dict(
        algo_config=(
            [
                dict(
                    model_decoder_layers="model.language_model.layers",
                    name="awq",
                    scaling_layers=[
                        dict(
                            inp="mlp.experts.*.w13",
                            layers=["mlp.gate", "mlp.experts.*.w13"],
                            module2inspect="mlp",
                            prev_op="post_attention_layernorm",
                        )
                    ],
                )
            ]
            if variant == "UINT4-AWQ" or variant == "UINT4-AWQ-AttnFP8"
            else None
        ),
        exclude=exclude,
        export=dict(
            kv_cache_group=[],
            min_kv_scale=0.0,
            pack_method="reorder",
            weight_format="real_quantized",
            weight_merge_groups=None,
        ),
        global_quant_config=dict(
            bias=None,
            input_tensors=None if uint4 else _mxfp4_spec(True),
            output_tensors=None,
            target_device=None,
            weight=_uint4_weight_spec() if uint4 else _mxfp4_spec(False),
        ),
        kv_cache_post_rope=False,
        kv_cache_quant_config={},
        layer_quant_config=(
            {
                f"model.llm.layers.*.attn.{name}": _fp8_attention_spec()
                for name in ATTENTION_FP8_LINEARS
            }
            if attn_fp8
            else {}
        ),
        layer_type_quant_config={},
        quant_method="quark",
        quant_mode="eager_mode",
        softmax_quant_spec=None,
        version="0.14+5b5dba3",
    )


def native_quant_dict(compute="bfloat16", names=("x",)) -> dict:
    """The precision metadata of a native release: one group of tensors `names`."""
    return dict(
        quant_method="inkling2",
        model=dict(parameter_groups=[precision_group(compute, list(names))]),
    )


class Materialization(TorchDispatchMode):
    """Records (op, shape, dtype) of every output with new storage of an ATen op that
    reads one of the watched tensors: a cast or a copy, not a view."""

    def __init__(self, *watched):
        super().__init__()
        self.storages = {
            t.untyped_storage().data_ptr() for t in watched if t is not None
        }
        self.copies = []

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        leaves = _pytree.tree_leaves((args, kwargs))
        reads = any(
            isinstance(t, torch.Tensor)
            and t.untyped_storage().data_ptr() in self.storages
            for t in leaves
        )
        if reads:
            for o in _pytree.tree_leaves(out):
                if (
                    isinstance(o, torch.Tensor)
                    and o.untyped_storage().data_ptr() not in self.storages
                ):
                    self.copies.append((str(func), tuple(o.shape), str(o.dtype)))
        return out
