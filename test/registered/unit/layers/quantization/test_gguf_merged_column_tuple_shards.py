"""CPU coverage for GGUF fused logical shards in merged column linears."""

import pytest
import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.gguf import GGUFConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layer() -> MergedColumnParallelLinear:
    # The sizes deliberately differ so a split that happens after TP sharding
    # cannot accidentally pass these assertions.
    return MergedColumnParallelLinear(
        input_size=3,
        output_sizes=[4, 6, 2],
        bias=False,
        params_dtype=torch.float32,
        quant_config=GGUFConfig(),
        tp_rank=1,
        tp_size=2,
    )


def test_gguf_tuple_qweight_type_fans_out_to_every_logical_shard():
    layer = _layer()

    layer.weight_loader(
        layer.qweight_type,
        torch.tensor(12, dtype=torch.uint8),
        (2, 0, 1),
    )

    assert layer.qweight_type.tolist() == [12, 12, 12]
    assert layer.qweight_type.shard_weight_type == {2: 12, 0: 12, 1: 12}


def test_gguf_tuple_weight_splits_raw_rows_before_tp_and_preserves_order():
    layer = _layer()
    raw = torch.arange(12 * 3, dtype=torch.uint8).reshape(12, 3)

    # The checkpoint's fused tensor follows the supplied logical-shard order:
    # shard 2 has 2 rows, shard 0 has 4 rows, and shard 1 has 6 rows.
    layer.weight_loader(layer.qweight, raw, (2, 0, 1))

    assert layer.qweight.shard_id == [2, 0, 1]
    assert layer.qweight.shard_id_map == {2: 0, 0: 1, 1: 2}
    assert len(layer.qweight.data_container) == 3
    torch.testing.assert_close(layer.qweight.data_container[0], raw[1:2])
    torch.testing.assert_close(layer.qweight.data_container[1], raw[4:6])
    torch.testing.assert_close(layer.qweight.data_container[2], raw[9:12])


def test_gguf_tuple_weight_rejects_fused_rows_incompatible_with_output_sizes():
    layer = _layer()

    with pytest.raises(
        ValueError, match="Fused GGUF shard has incompatible output size"
    ):
        layer.weight_loader(
            layer.qweight,
            torch.zeros((11, 3), dtype=torch.uint8),
            (2, 0, 1),
        )
