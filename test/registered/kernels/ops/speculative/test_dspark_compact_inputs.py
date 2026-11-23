"""Direct packing must match dense inputs before and after graph replay."""

import os

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.native_compact import pack_verify_inputs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@pytest.mark.parametrize("write_window_index", [True, False])
@pytest.mark.parametrize("bs", [1, 2, 3, 8, 32, 64, 65, 96, 128, 256])
def test_direct_inputs_and_replay(bs, write_window_index):
    device = "cpu" if os.getenv("TRITON_INTERPRET") == "1" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA or TRITON_INTERPRET=1")
    generator = torch.Generator().manual_seed(412)
    lens = torch.full((bs,), 6, dtype=torch.int64, device=device)
    ptr = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int64), lens.cumsum(0)]
    ).int()
    reqs = torch.arange(1, bs + 1, device=device)
    prefixes = torch.arange(bs, device=device) % 10 + 60
    table = torch.arange((bs + 1) * 96, device=device).view(bs + 1, 96)
    anchors = torch.arange(bs * 10, device=device).view(bs, 10)[:, ::2]
    drafts = torch.zeros((bs, 10), dtype=torch.int64, device=device)[:, ::2]
    capacity = (bs * 6 + 7) // 8 * 8
    outputs = [
        torch.full((capacity,), -999, device=device, dtype=torch.int64)
        for _ in range(4)
    ]

    def operation():
        pack_verify_inputs(
            indptr=ptr,
            reqs=reqs,
            prefixes=prefixes,
            req_to_token=table,
            anchors=anchors,
            drafts=drafts,
            width=6,
            ids=outputs[0],
            positions=outputs[1],
            locations=outputs[2],
            window_index=outputs[3] if write_window_index else None,
        )

    operation()
    graph = None
    if device == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
    for variant in range(5):
        lengths = torch.randint(1, 7, (bs,), generator=generator)
        if variant == 0:
            lengths.fill_(6)
        elif variant == 1:
            lengths.fill_(1)
        elif variant == 2:
            lengths[::3] = 0
        elif variant == 4:
            lengths.zero_()
        lens.copy_(lengths)
        ptr.copy_(
            torch.cat(
                [torch.zeros(1, device=device, dtype=torch.int64), lens.cumsum(0)]
            )
        )
        drafts.copy_(torch.randint(0, 37, (bs, 5), generator=generator))
        if graph is None:
            operation()
        else:
            graph.replay()
        mask = torch.arange(6, device=device)[None, :] < lens[:, None]
        dense_ids = torch.cat([anchors[:, :1], drafts], dim=1)
        dense_pos = prefixes[:, None] + torch.arange(6, device=device)
        dense_loc = table[reqs[:, None], dense_pos]
        dense_window = torch.arange(bs * 6, device=device).view(bs, 6)
        count = int(lengths.sum())
        checked = (
            (dense_ids, dense_pos, dense_loc, dense_window)
            if write_window_index
            else (dense_ids, dense_pos, dense_loc)
        )
        for index, dense in enumerate(checked):
            expected = torch.full_like(outputs[index], -1 if index == 3 else 0)
            expected[:count] = dense[mask]
            torch.testing.assert_close(outputs[index], expected, rtol=0, atol=0)
        if not write_window_index:
            assert (outputs[3] == -999).all()
