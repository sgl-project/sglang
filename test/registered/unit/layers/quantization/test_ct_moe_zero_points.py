"""CT zero points use natural channel order, unlike AWQ's interleaved words."""

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.marlin_utils import (
    awq_to_marlin_zero_points,
    marlin_zero_points,
    pack_cols,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("groups,outputs", [(4, 512), (16, 1024)])
def test_ct_moe_zero_points_match_natural_order_dense_conversion(bits, groups, outputs):
    bound = 1 << (bits - 1)
    raw = torch.arange(3 * groups * outputs, dtype=torch.int32).reshape(
        3, groups, outputs
    )
    logical = (raw * 3 + raw // outputs + raw // (groups * outputs)) % (2 * bound)
    # The installed CT compressor packs the source output dimension naturally.
    packed = torch.stack(
        [
            pack_to_int32((expert.t() - bound).to(torch.int8), bits, packed_dim=0)
            .t()
            .contiguous()
            for expert in logical
        ]
    )
    expected = torch.stack(
        [marlin_zero_points(expert, groups, outputs, bits) for expert in logical]
    )
    scheme = SimpleNamespace(num_bits=bits, packed_factor=32 // bits)
    actual = CompressedTensorsWNA16MoE._convert_zero_points_to_marlin(scheme, packed)
    assert torch.equal(actual, expected)
    assert actual.shape == packed.shape and actual.dtype == torch.int32
    # The former AWQ conversion changes these nonuniform CT channel values.
    assert not torch.equal(
        awq_to_marlin_zero_points(packed[0], groups, outputs, bits), expected[0]
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_awq_conversion_retains_its_own_interleaving_contract(bits):
    groups, outputs = 4, 512
    logical = torch.arange(groups * outputs, dtype=torch.int32).reshape(
        groups, outputs
    ) % (1 << bits)
    interleave = [0, 2, 4, 6, 1, 3, 5, 7] if bits == 4 else [0, 2, 1, 3]
    interleaved = logical.reshape(-1, len(interleave))[:, interleave].reshape_as(
        logical
    )
    packed = pack_cols(interleaved, bits, groups, outputs)
    assert torch.equal(
        awq_to_marlin_zero_points(packed, groups, outputs, bits),
        marlin_zero_points(logical, groups, outputs, bits),
    )
