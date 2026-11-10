import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Native QSA combine requires V100",
)


@pytest.mark.parametrize("batch,splits", [(1, 160), (2, 80), (4, 40)])
def test_combine_reference_masking_and_graph_replay(batch, splits):
    from sglang.kernels.ops.attention.sm70_qsa_combine import combine

    torch.manual_seed(18 + batch)
    partial = torch.randn((batch, splits, 6, 256), device="cuda", dtype=torch.float16)
    lse = torch.randn((batch, splits, 6), device="cuda") * 4 + 10
    lengths = torch.full((batch,), 25000, device="cuda", dtype=torch.int32)
    raw_partial, raw_lse = partial.clone(), lse.clone()
    combine(partial, lse, lengths, 2048)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = combine(partial, lse, lengths, 2048)
    for length in [1, 31, 32, 33, 2048, 25000, 33000]:
        lengths.copy_(torch.arange(batch, device="cuda", dtype=torch.int32) + length)
        partial.copy_(raw_partial)
        lse.copy_(raw_lse)
        expected = []
        for row in range(batch):
            active = min(splits, (min(length + row, 2048) + 31) // 32)
            # Inactive partials are uninitialized in the attention kernel.
            partial[row, active:].fill_(float("nan"))
            lse[row, active:].fill_(float("nan"))
            weights = torch.softmax(
                raw_lse[row, :active].double()
                * torch.log(torch.tensor(2.0, dtype=torch.float64, device="cuda")),
                dim=0,
            )
            expected.append(
                (weights[:, :, None] * raw_partial[row, :active].double()).sum(0).half()
            )
        graph.replay()
        reference = torch.stack(expected)
        torch.testing.assert_close(output, reference, rtol=0.001, atol=0.0002)
        assert (
            output.float() - reference.float()
        ).norm() / reference.float().norm() < 0.0005
