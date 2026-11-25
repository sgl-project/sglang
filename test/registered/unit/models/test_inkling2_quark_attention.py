"""The FP8 attention of the AttnFP8 exports, with the real Inkling2Attention at the
dimensions of Inkling-2-Mini (CPU only)."""

import copy
import re
import sys

import pytest
import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import quark_w8a8_fp8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.inkling_common.inkling2_attn import AttentionSources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.inkling2_quark_test_utils import (
    mini_layout,
    mini_text_config,
    native_quant_dict,
    real_attention,
    real_attention_export,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

BF16, F32, U8 = torch.bfloat16, torch.float32, torch.uint8
E4M3 = torch.float8_e4m3fn
FP8_PROJECTIONS = ("wq_du", "wk_dv", "wv_dv", "gate_attn")  # qkvr.fp8, in shard order
BF16_PROJECTIONS = ("wr_du", "attn_sink_proj")  # qkvr.bf16, in shard order


@pytest.fixture(scope="module")
def text_config():
    return mini_text_config()


@pytest.fixture(scope="module", autouse=True)
def _published_context():
    from sglang.srt.runtime_context import reset_context
    from sglang.srt.server_args import ServerArgs
    from sglang.test.parallel_groups import publish

    publish(ServerArgs(model_path="dummy"), role="test")
    yield
    reset_context()


def _parameter_table(module):
    return {n: (tuple(p.shape), p.dtype) for n, p in module.named_parameters()}


@pytest.mark.parametrize(
    "tp,rank,fp8_rows,bf16_rows,wo_columns",
    [(1, 0, 8192, 408, 3072), (8, 7, 1024, 56, 384)],
)
def test_fp8_attention_parameters(
    text_config, tp, rank, fp8_rows, bf16_rows, wo_columns
):
    attn = real_attention(text_config, mini_layout("MXFP4-AttnFP8"), 5, tp, rank)
    hidden = text_config.hidden_size
    # rows: q, k, v, gate | r, sinks (+ 5 alignment rows at TP=8)
    assert {
        n: t
        for n, t in _parameter_table(attn).items()
        if n.startswith(("qkvr", "wo_ud"))
    } == {
        "qkvr.fp8.weight": ((fp8_rows, hidden), E4M3),
        "qkvr.fp8.weight_scale": ((fp8_rows,), F32),
        "qkvr.bf16.weight": ((bf16_rows, hidden), BF16),
        "wo_ud.weight": ((hidden, wo_columns), E4M3),
        "wo_ud.weight_scale": ((hidden,), F32),
    }
    for module in (attn.qkvr["fp8"], attn.wo_ud):
        # exactly QuarkLinearMethod: the weight loaders match it by class name
        assert type(module.quant_method) is QuarkLinearMethod
    assert isinstance(attn.qkvr["bf16"].quant_method, UnquantizedLinearMethod)


@pytest.mark.parametrize("quant", ["UINT4-AWQ", "MXFP4-RTN", "native"])
def test_bf16_attention_is_built_as_without_a_quant_config(text_config, quant):
    if quant == "native":
        quant_config = Inkling2QuantConfig(native_quant_dict())
    else:
        quant_config = mini_layout(quant)
    attn = real_attention(text_config, quant_config, 5)
    assert _parameter_table(attn) == _parameter_table(
        real_attention(text_config, None, 5)
    )


def test_fp8_attention_refuses_attention_biases(text_config):
    layout = mini_layout("MXFP4-AttnFP8")
    for bias in ("q_bias", "o_bias"):
        biased = copy.copy(text_config)
        setattr(biased, bias, True)
        with pytest.raises(ValueError, match="does not support attention biases"):
            real_attention(biased, layout, 5)


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 1), (8, 7)])
@pytest.mark.parametrize("layer_id", [0, 5])
def test_fp8_attention_tp_slices(text_config, layer_id, tp, rank):
    attn, target, tensors = real_attention_export(
        text_config, mini_layout("MXFP4-AttnFP8"), layer_id, tp, rank
    )
    loaded = target.load_weights(list(tensors.items()))
    assert loaded == set(dict(target.named_parameters()))
    prefix = f"model.llm.layers.{layer_id}.attn."

    def rows(names, suffix="weight"):  # this rank's rows of each column-parallel shard
        return torch.cat(
            [tensors[f"{prefix}{n}.{suffix}"].chunk(tp)[rank] for n in names]
        )

    fp8, bf16 = attn.qkvr["fp8"], attn.qkvr["bf16"]
    assert torch.equal(fp8.weight.view(U8), rows(FP8_PROJECTIONS).view(U8))
    assert torch.equal(fp8.weight_scale, rows(FP8_PROJECTIONS, "weight_scale").to(F32))
    used = sum(tensors[f"{prefix}{n}.weight"].shape[0] // tp for n in BF16_PROJECTIONS)
    assert torch.equal(bf16.weight[:used].view(U8), rows(BF16_PROJECTIONS).view(U8))
    wo = tensors[f"{prefix}wo_ud.weight"].chunk(tp, dim=1)[rank]  # row-parallel
    assert torch.equal(attn.wo_ud.weight.view(U8), wo.view(U8))
    scale = tensors[f"{prefix}wo_ud.weight_scale"].to(F32)
    assert torch.equal(attn.wo_ud.weight_scale, scale)


@pytest.mark.parametrize(
    "dropped,message",
    [
        ("gate_attn.weight_scale", r"attn\.qkvr\.fp8\.weight_scale shards \[3\]"),
        ("attn_sink_proj.weight", r"attn\.qkvr\.bf16\.weight shards \[1\]"),
        ("wo_ud.weight_scale", r"attn\.wo_ud\.weight_scale"),
    ],
)
def test_fp8_attention_missing_shard(text_config, dropped, message):
    _, target, tensors = real_attention_export(
        text_config, mini_layout("MXFP4-AttnFP8"), 0
    )
    del tensors[f"model.llm.layers.0.attn.{dropped}"]
    with pytest.raises(ValueError, match="Missing checkpoint tensors") as error:
        target.load_weights(list(tensors.items()))
    assert re.search(message, str(error.value))


def test_fp8_qkvr_output_layout(text_config):
    """q, k, v, r, gate and the sinks come out as from one merged BF16 projection, which
    the prologue kernels read. The GEMMs are replaced by a projection whose column c of
    row i is 1000 * i + c."""
    attn = real_attention(text_config, mini_layout("MXFP4-AttnFP8"), 5, 8)
    sizes, hidden = attn.split_sizes, text_config.hidden_size
    total, ends = sum(sizes), torch.tensor(sizes).cumsum(0).tolist()
    columns = torch.arange(total)
    fp8_columns = torch.cat([columns[: ends[2]], columns[ends[3] : ends[4]]])
    bf16_columns = torch.cat([columns[ends[2] : ends[3]], columns[ends[4] :]])

    def merged(x):
        return x[:, :1] * 1000 + columns.to(x.dtype)

    attn.qkvr["fp8"].forward = lambda x: (merged(x)[:, fp8_columns], None)
    attn.qkvr["bf16"].forward = lambda x: (merged(x)[:, bf16_columns], None)

    def rows(first, count=3):
        return torch.arange(first, first + count, dtype=F32)[:, None].expand(-1, hidden)

    x, y, z = rows(0), rows(10), rows(20)
    pieces = attn._project_qkvr(x)
    assert len(pieces) == 6
    for piece, expected in zip(pieces, merged(x).split(sizes, -1)):
        assert piece.stride() == (total, 1) and torch.equal(piece, expected)
    q, k, v, r, gate, sinks = attn._project_sources(AttentionSources(x, y, z))
    # k comes from the second source, v from the third, the rest from the first
    expected = [merged(s).split(sizes, -1) for s in (x, y, z)]
    for got, want in zip(
        (q, k, v, r, gate, sinks),
        (expected[0][0], expected[1][1], expected[2][2], *expected[0][3:6]),
    ):
        assert torch.equal(got, want)
    assert len({t.stride() for t in (q, k, v, r)}) == 1


@pytest.mark.parametrize("outputs,error", [(32, None), (24, "lost its FP8")])
def test_a_scheme_that_dequantizes_is_refused_after_loading(
    monkeypatch, text_config, outputs, error
):
    """On HIP, QuarkW8A8Fp8 serves a per-channel linear whose N is not a multiple of 16
    dequantized; an FP8 attention linear must not run that way."""
    monkeypatch.setattr(quark_w8a8_fp8, "_is_hip", True)
    monkeypatch.setattr(quark_w8a8_fp8, "_is_fp8_fnuz", False)
    monkeypatch.setattr(quark_w8a8_fp8, "use_aiter_bpreshuffle_gemm", lambda n: False)
    module = MergedColumnParallelLinear(
        text_config.hidden_size,
        [outputs // 2, outputs // 2],
        bias=False,
        quant_config=mini_layout("MXFP4-AttnFP8"),
        parallel_group="attn_tp",
        prefix="llm.layers.0.attn.qkvr.fp8",
    )
    module.weight_scale.data.fill_(1.0)
    if error:
        with pytest.raises(ValueError, match=error):
            module.quant_method.process_weights_after_loading(module)
    else:
        module.quant_method.process_weights_after_loading(module)
        assert module.weight.dtype == E4M3 and module.weight_scale is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
