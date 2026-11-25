"""Test helpers for the Inkling-2 Quark exports: the quantization_config of the four
exports, the tensors of a tiny export, the real attention at the dimensions of
Inkling-2-Mini, and a recorder of hidden copies of the packed weights."""

import functools
import json
import tempfile
from pathlib import Path

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.test.inkling2_quark_kernel_test_utils import synth_mxfp4, synth_uint4
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


def expert_checkpoint_tensors(
    expert_format: str,
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    layer: int = 0,
    seed: int = 0,
    scale_dtype=torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """The routed experts of one layer: synthetic packed tensors with the names, shapes
    and dtypes of an export."""
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    shapes = {"w13": (2 * intermediate, hidden), "w2": (hidden, intermediate)}
    tensors = {}
    for i, (name, (n, k)) in enumerate(shapes.items()):
        if expert_format == "mxfp4":
            data = synth_mxfp4(experts, n, k, seed=seed + i)
        else:
            data = synth_uint4(experts, n, k, seed=seed + i, scale_dtype=scale_dtype)
        for field, value in data.items():
            if field != "expected":
                tensors[f"{prefix}{name}_{field}"] = value
    return tensors


def attention_checkpoint_tensors(
    *,
    fp8: bool,
    hidden: int,
    q_rows: int,
    kv_rows: int,
    r_rows: int,
    sink_rows: int,
    layer: int = 0,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """The attention linears of one layer: ATTENTION_FP8_LINEARS as FP8 with one BF16
    scale per output row if `fp8`, everything else BF16."""
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "wq_du": (q_rows, hidden),
        "wk_dv": (kv_rows, hidden),
        "wv_dv": (kv_rows, hidden),
        "wr_du": (r_rows, hidden),
        "gate_attn": (q_rows, hidden),
        "attn_sink_proj": (sink_rows, hidden),
        "wo_ud": (hidden, q_rows),
    }
    tensors = {}
    for name, shape in shapes.items():
        w = (torch.randn(shape, generator=generator) * 0.02).to(torch.bfloat16)
        base = f"model.llm.layers.{layer}.attn.{name}"
        if fp8 and name in ATTENTION_FP8_LINEARS:
            tensors[f"{base}.weight"] = w.to(torch.float8_e4m3fn)
            tensors[f"{base}.weight_scale"] = (
                torch.rand(shape[0], generator=generator) * 1e-3 + 1e-4
            ).to(torch.bfloat16)
        else:
            tensors[f"{base}.weight"] = w
    return tensors


# config.json of Inkling-2-Mini without its quantization_config
MINI_CONFIG = {
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": 200006,
    "text_config": {
        "model_type": "inkling2_loop_text",
        "model_max_length": 1048576,
        "torch_dtype": "bfloat16",
        "hidden_size": 2304,
        "num_hidden_layers": 24,
        "vocab_size": 201024,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "q_bias": False,
        "o_bias": False,
        "log_scaling_n_floor": 8192,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": True,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "local_layer_ids": [
            *range(0, 5),
            *range(6, 11),
            *range(12, 17),
            *range(18, 23),
        ],
        "sconv_kernel_size": 4,
        "unpadded_vocab_size": 200058,
        "final_logit_softcapping": 30.0,
        "swa_head_dim": 128,
        "swa_num_attention_heads": 24,
        "swa_num_key_value_heads": 8,
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "attn_res_normalize_values": True,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "n_shared_experts": 0,
        "intermediate_size": 1152,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": True,
        "loop_config": {
            "core_start": 6,
            "core_end": 18,
            "num_passes": 3,
            "gate_rank": 64,
            "is_varloop": False,
        },
    },
    "vision_config": {
        "model_type": "inkling2_vision",
        "text_hidden_size": 2304,
        "temporal_patch_size": 2,
        "native_patch_size": 20,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 4096,
        "projector_bias": False,
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "posemb_grid_size": 32,
        "layer_norm_eps": 1e-05,
    },
    "mtp_config": {
        "hidden_size": 2304,
        "num_hidden_layers": 3,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9216,
        "max_position_embeddings": 1048576,
        "torch_dtype": "bfloat16",
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "local_layer_ids": [0, 2],
        "sliding_window_size": 1024,
        "block_size": 8,
        "markov_rank": 256,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 23, 41, 48],
    },
}


@functools.cache
def mini_text_config():
    from sglang.srt.configs.model_config import ModelConfig

    with tempfile.TemporaryDirectory() as path:
        Path(path, "config.json").write_text(json.dumps(MINI_CONFIG))
        return ModelConfig(path, trust_remote_code=False).hf_config.text_config


def mini_layout(variant: str):
    from sglang.srt.layers.quantization.inkling2_quark import Inkling2QuarkLayout

    return Inkling2QuarkLayout.from_quantization_dict(
        quantization_config(variant), mini_text_config().num_hidden_layers
    )


def attention_rows(config) -> dict:
    """The keyword arguments of attention_checkpoint_tensors for a layer of `config`."""
    heads, head_dim = config.num_attention_heads, config.head_dim
    assert (config.swa_num_attention_heads, config.swa_head_dim) == (heads, head_dim)
    return dict(
        hidden=config.hidden_size,
        q_rows=heads * head_dim,
        kv_rows=config.num_key_value_heads * head_dim,
        r_rows=heads * config.d_rel,
        sink_rows=heads,
    )


def real_attention(config, quant_config, layer_id, tp=1, rank=0, device="cpu"):
    """Inkling2Attention of a layer on one attention-TP rank (needs a published context)."""
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.inkling_common.inkling2_attn import Inkling2Attention
    from sglang.test.parallel_groups import parallel_scope

    ranks = dict(tp_rank=rank, attn_tp_rank=rank, moe_ep_rank=rank)
    sizes = dict(tp_size=tp, attn_tp_size=tp, moe_ep_size=tp)
    with parallel_scope(**ranks, **sizes), torch.device(device):
        with set_default_torch_dtype(torch.bfloat16):
            return Inkling2Attention(
                config,
                layer_id,
                layer_id in config.local_layer_ids,
                prefix=f"llm.layers.{layer_id}.attn",
                quant_config=quant_config,
            )


def real_attention_export(config, layout, layer_id, tp=1, rank=0, device="cpu"):
    """(attention, a model around it, the checkpoint tensors of its layer) of an AttnFP8
    export; the parameters other than the linears are zeros."""
    from torch import nn

    from sglang.srt.models.inkling2 import Inkling2ForConditionalGeneration

    attn = real_attention(config, layout, layer_id, tp, rank, device)
    tensors = attention_checkpoint_tensors(
        fp8=True, layer=layer_id, **attention_rows(config)
    )
    base = f"model.llm.layers.{layer_id}.attn."
    for name, p in attn.named_parameters():
        if not name.startswith(("qkvr", "wo_ud")):
            tensors[base + name] = torch.zeros(p.shape, dtype=p.dtype)
    target = Inkling2ForConditionalGeneration.__new__(Inkling2ForConditionalGeneration)
    nn.Module.__init__(target)
    target.visual, target.quant_config = None, layout
    target.llm = nn.Module()
    layers = [nn.Module() for _ in range(layer_id + 1)]
    layers[layer_id].attn = attn
    target.llm.layers = nn.ModuleList(layers)
    return attn, target, tensors


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
