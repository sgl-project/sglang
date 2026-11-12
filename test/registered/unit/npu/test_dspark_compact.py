"""Real-device scheduling, packing and metadata contracts for GLM compact.

These tests require an Ascend card and sgl-kernel-npu. They do not load a model
or replace the multi-rank serving and numerical-accuracy checks.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import dspark_schedule as schedule
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    build_ragged_verify_window,
    scatter_compact_to_strided,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=10, suite="nightly-1-npu-a3", nightly=True)

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires an Ascend NPU",
)


def make_backend():
    # Import the actual backend on NPU; no copied/AST-extracted methods or
    # replacement device operators in this suite.
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    backend = AscendAttnBackend.__new__(AscendAttnBackend)
    backend.req_to_token = torch.arange(192, dtype=torch.int32, device="npu").view(
        3, 64
    )
    backend.page_size = 16
    backend.device = "npu"
    backend._dspark_dsa_ragged_verify = True
    return backend


@pytest.mark.parametrize("budget", [2, 100])
def test_existing_npu_kernel_capture_replay_reads_updated_confidence(budget):
    cfg = DSparkScheduleConfig(gamma=3)
    static_input = torch.ones(2, 3, device="npu")
    for _ in range(2):
        schedule.ScheduleVerifyLensTopk.execute(
            confidence=static_input, budget=budget, cfg=cfg
        )
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        static_output = schedule.ScheduleVerifyLensTopk.execute(
            confidence=static_input, budget=budget, cfg=cfg
        )
    for values in (
        [[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]],
        [[0.2, 0.1, 0.1], [0.9, 0.8, 0.7]],
        [[1.0] * 3] * 2,
        [[0.0] * 3] * 2,
    ):
        confidence = torch.tensor(values)
        static_input.copy_(confidence)
        graph.replay()
        torch.npu.synchronize()
        reference = schedule.schedule_verify_lens_topk(
            confidence=confidence, budget=budget, cfg=cfg
        )
        torch.testing.assert_close(static_output.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [4, 8])
def test_npu_pack_metadata_and_scatter_with_padding(tokens):
    backend = make_backend()
    lens = schedule.ScheduleVerifyLensTopk.execute(
        confidence=torch.tensor([[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]], device="npu"),
        budget=2,
        cfg=DSparkScheduleConfig(gamma=3),
    )
    assert lens.cpu().tolist() == [3, 1]
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=lens, graph_num_tokens=tokens
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=2,
        seq_lens=torch.tensor([15, 31], device="npu"),
        req_pool_indices=torch.tensor([2, 0], device="npu"),
        spec_info=SimpleNamespace(ragged_verify_layout=layout),
    )
    candidates = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="npu")
    packed = build_ragged_verify_window(
        batch=batch,
        layout=layout,
        draft_block_ids=candidates,
        draft_tokens=candidates[:, 1:],
        bs=2,
        device="npu",
        verify_num_draft_tokens=4,
        model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=backend.req_to_token)
        ),
    )
    assert packed.verify_ids[:4].cpu().tolist() == [1, 2, 3, 5]
    assert packed.verify_cache_loc[:4].cpu().tolist() == [143, 144, 145, 31]
    backend.init_forward_metadata(batch)
    metadata = backend.forward_metadata
    assert metadata.actual_seq_lengths_q.cpu().tolist() == [3, 4, tokens]
    assert metadata.actual_seq_lengths_kv.cpu().tolist() == [18, 32, max(tokens - 4, 1)]
    assert metadata.block_tables[-1].cpu().tolist() == [0] * 4
    rows = scatter_compact_to_strided(
        compact=packed.verify_ids[:, None],
        layout=layout,
        fill_value=-1,
        verify_num_draft_tokens=4,
    )
    assert rows.cpu().flatten().tolist() == [1, 2, 3, -1, 5, -1, -1, -1]


def test_graph_metadata_refresh_reads_new_lengths_and_idle_rows():
    backend = make_backend()
    lens = torch.tensor([3, 1], dtype=torch.int32, device="npu")
    prefix = torch.tensor([15, 31], device="npu")
    slots = torch.tensor([2, 0], device="npu")
    metadata = backend._allocate_padded_verify_metadata(2, 8)
    addresses = [
        x.data_ptr()
        for x in (
            metadata.actual_seq_lengths_q,
            metadata.seq_lens,
            metadata.block_tables,
        )
    ]

    def update():
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=lens, graph_num_tokens=8
        )
        backend._update_ragged_verify_metadata(metadata, layout, prefix, slots)

    for _ in range(2):
        update()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        update()
    for lengths in ([1, 3], [0, 0], [4, 4]):
        lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        assert metadata.actual_seq_lengths_q.cpu().tolist() == [
            lengths[0],
            sum(lengths),
            8,
        ]
        expected_kv = [p + n if n else 0 for p, n in zip([15, 31], lengths)]
        assert metadata.seq_lens.cpu().tolist() == expected_kv + [
            max(8 - sum(lengths), 1)
        ]
        assert [
            x.data_ptr()
            for x in (
                metadata.actual_seq_lengths_q,
                metadata.seq_lens,
                metadata.block_tables,
            )
        ] == addresses
        assert not metadata.block_tables[-1].cpu().any()
        if lengths == [0, 0]:
            assert not metadata.block_tables.cpu().any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
