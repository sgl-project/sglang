"""Vector scale decoding must preserve scalar results and unaligned views."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The specialized NVFP4 MoE decode kernel requires an NVIDIA V100",
)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_vector_and_unaligned_scales_match_across_graph_replays(rows):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    torch.manual_seed(731 + rows)
    experts = 12
    w13 = torch.empty((experts, 160, 640), dtype=torch.int32, device="cuda").random_()
    w2 = torch.empty((experts, 10, 5120), dtype=torch.int32, device="cuda").random_()
    s13 = torch.empty((experts, 160, 320), dtype=torch.uint8, device="cuda")
    s2 = torch.empty((experts, 10, 2560), dtype=torch.uint8, device="cuda")

    def unaligned(tensor, offset):
        storage = torch.empty(
            tensor.numel() + offset, dtype=tensor.dtype, device=tensor.device
        )
        view = storage[offset:].view_as(tensor)
        assert view.is_contiguous() and view.data_ptr() % 8 == offset
        return view

    scalar13, scalar2 = unaligned(s13, 1), unaligned(s2, 3)
    assert s13.data_ptr() % 8 == s2.data_ptr() % 8 == 0
    g13 = torch.rand(experts, device="cuda") * 50 + 100
    g2 = torch.rand(experts, device="cuda") * 50 + 100
    x = torch.empty((rows, 2560), dtype=torch.float16, device="cuda")
    ids = torch.empty((rows, 10), dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, 10), device="cuda")

    def update(case):
        x.normal_()
        # Vary every stored scale column to expose byte-order mistakes.
        s13.random_(0x40, 0x79)
        s2.random_(0x40, 0x79)
        scalar13.copy_(s13)
        scalar2.copy_(s2)
        ids.copy_(torch.rand(rows, experts, device="cuda").topk(10).indices)
        ids[:, case::3] = -1
        if case == 2:
            ids[0].fill_(-1)
        weights.copy_(torch.randn_like(weights).softmax(-1))

    def run():
        # Exercise independent alignment dispatch for both projections.
        return [
            sm70_nvfp4_moe_decode(x, w13, w2, a, b, g13, g2, ids, weights)
            for a, b in (
                (s13, s2),
                (scalar13, scalar2),
                (scalar13, s2),
                (s13, scalar2),
            )
        ]

    update(0)
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for case in range(3):
        update(case)
        graph.replay()
        assert torch.isfinite(outputs[0]).all()
        assert outputs[0].abs().max() > 0 or (case == 2 and rows == 1)
        for output in outputs[1:]:
            torch.testing.assert_close(
                output.view(torch.int16), outputs[0].view(torch.int16), rtol=0, atol=0
            )
        if case == 2:
            assert torch.count_nonzero(outputs[0][0]) == 0
