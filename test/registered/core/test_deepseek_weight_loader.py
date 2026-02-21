from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)


def _pack_int4_row(values: list[int]) -> int:
    packed = 0
    for i, value in enumerate(values):
        packed |= ((value + 8) & 0xF) << (4 * i)
    # Convert to signed int32 range.
    if packed >= 2**31:
        packed -= 2**32
    return packed


def test_dequantize_ct_wna16_weight():
    row0 = [-8, -7, -6, -5, -4, -3, -2, -1]
    row1 = [0, 1, 2, 3, 4, 5, 6, 7]
    weight_packed = torch.tensor(
        [[_pack_int4_row(row0)], [_pack_int4_row(row1)]], dtype=torch.int32
    )
    weight_scale = torch.tensor([[0.5], [2.0]], dtype=torch.bfloat16)
    weight_shape = torch.tensor([2, 8], dtype=torch.int64)

    layer = SimpleNamespace(
        weight_packed=weight_packed,
        weight_scale=weight_scale,
        weight_shape=weight_shape,
    )

    dequant = DeepseekV2WeightLoaderMixin._dequantize_ct_wna16_weight(layer)
    expected = torch.tensor(
        [
            [v * 0.5 for v in row0],
            [v * 2.0 for v in row1],
        ],
        dtype=torch.bfloat16,
    )
    assert dequant.dtype == torch.bfloat16
    assert tuple(dequant.shape) == (2, 8)
    assert torch.equal(dequant, expected)


def test_post_load_weights_dequantizes_ct_kv_b_proj():
    class _DummyLoader(DeepseekV2WeightLoaderMixin):
        pass

    kv_b_proj = SimpleNamespace(
        weight_packed=torch.tensor(
            [
                [_pack_int4_row([-8, -7, -6, -5, -4, -3, -2, -1])],
                [_pack_int4_row([0, 1, 2, 3, 4, 5, 6, 7])],
            ],
            dtype=torch.int32,
        ),
        weight_scale=torch.tensor([[1.0], [1.0]], dtype=torch.bfloat16),
        weight_shape=torch.tensor([2, 8], dtype=torch.int64),
    )
    self_attn = SimpleNamespace(
        kv_b_proj=kv_b_proj,
        qk_nope_head_dim=1,
        v_head_dim=1,
        w_kc=None,
        w_vc=None,
        w_scale=None,
    )

    loader = _DummyLoader()
    loader.config = SimpleNamespace(num_hidden_layers=1)
    loader.model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[SimpleNamespace(self_attn=self_attn)],
    )
    loader.quant_config = None

    loader.post_load_weights()

    assert self_attn.w_kc is not None
    assert self_attn.w_vc is not None
    assert tuple(self_attn.w_kc.shape) == (1, 1, 8)
    assert tuple(self_attn.w_vc.shape) == (1, 8, 1)
