"""DSpark draft metadata must capture and refresh live GPU inputs on replay."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu")


def _backend(block_size):
    backend = dsv4.DeepseekV4AttnBackend.__new__(dsv4.DeepseekV4AttnBackend)
    device = torch.device("cuda")
    num_slots, context_size = 16, 2048
    req_to_token = torch.arange(
        num_slots * context_size, dtype=torch.int32, device=device
    ).reshape(num_slots, context_size)
    backend.device = device
    backend.cuda_int32_kwargs = dict(device=device, dtype=torch.int32)
    backend.req_to_token = req_to_token
    backend.page_size = 256
    backend.MAX_SEQ_LEN_FOR_CAPTURE = context_size
    backend.index_topk = 512
    backend.present_ratios = ()
    backend.low_ratios = ()
    backend.has_c4 = False
    backend.has_c128 = False
    backend.is_dspark_draft = True
    backend.is_draft_runner = True
    backend.needs_cpu_seq_lens = False
    backend.draft_metadata_glued = True
    backend._draft_extend_lens = {}
    backend.speculative_num_draft_tokens = block_size + 1
    backend.encoder_replay = False
    backend.encoder_row_floor = None
    backend.attn_backend_list = None
    backend.online_c128_mtp = SimpleNamespace(prepare_forward=lambda *args: None)
    backend.token_to_kv_pool = SimpleNamespace(
        request_window=None,
        swa_req_ring_size=None,
        swa_valid_from=torch.zeros(num_slots, dtype=torch.int32, device=device),
        full_to_swa_index_mapping=torch.arange(
            num_slots * context_size, dtype=torch.int32, device=device
        ),
    )
    backend.cuda_graph_metadata_of_bucket_and_bs = {
        bucket: {} for bucket in dsv4._GraphBucket
    }
    return backend


def _view(batch_size, block_size, dtype):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        actual_forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=batch_size,
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64, device="cuda"),
        seq_lens=torch.full((batch_size,), 127, dtype=dtype, device="cuda"),
        seq_lens_cpu=None,
        seq_lens_sum=0,
        out_cache_loc=torch.arange(
            20000, 20000 + batch_size * block_size, dtype=torch.int64, device="cuda"
        ),
    )


@pytest.mark.parametrize("batch_size,block_size", [(1, 1), (1, 5), (2, 5), (8, 5)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_draft_metadata_glue_replay_updates_live_inputs(batch_size, block_size, dtype):
    backend = _backend(block_size)
    view = _view(batch_size, block_size, dtype)
    glue = MetadataGlueGraph(torch.device("cuda"))
    key = (batch_size, str(ForwardMode.TARGET_VERIFY), str(ForwardMode.TARGET_VERIFY))
    for _ in range(glue.NUM_WARMUP + 1):
        glue.run(backend, view, key)
    assert not glue.disabled, "DSpark draft metadata capture fell back to eager"
    assert glue.is_captured(key)

    core = backend.forward_metadata.core_attn_metadata
    fields = ("seq_lens_casual", "positions_casual", "page_table", "raw_out_loc")
    pointers = {name: getattr(core, name).data_ptr() for name in fields}
    initial_positions = core.positions_casual.clone()
    boundaries = torch.tensor(
        [0, 1, 126, 127, 128, 255, 256, 1023], dtype=dtype, device="cuda"
    )
    for step in range(3):
        view.seq_lens.copy_(boundaries[:batch_size] + step)
        view.req_pool_indices.copy_(
            (torch.arange(batch_size, device="cuda") + step + 3) % 16
        )
        view.out_cache_loc.add_(batch_size * block_size)
        glue.run(backend, view, key)
        core = backend.forward_metadata.core_attn_metadata
        expected_positions = (
            view.seq_lens[:, None]
            + torch.arange(block_size, device="cuda", dtype=dtype)
        ).reshape(-1)
        expected_seq_lens = expected_positions + 1
        torch.testing.assert_close(
            core.seq_lens_casual, expected_seq_lens, rtol=0, atol=0, check_dtype=False
        )
        torch.testing.assert_close(
            core.positions_casual, expected_positions, rtol=0, atol=0, check_dtype=False
        )
        torch.testing.assert_close(core.raw_out_loc, view.out_cache_loc, rtol=0, atol=0)

        # Eager construction with the live host lengths is the original reference
        # path. It must agree with the captured, host-free construction.
        reference = backend.init_forward_metadata_dspark_draft_block(
            max_seq_len=2048,
            req_pool_indices=view.req_pool_indices,
            seq_lens=view.seq_lens,
            seq_lens_cpu=view.seq_lens.cpu(),
            out_cache_loc=view.out_cache_loc,
            block_size=block_size,
        ).core_attn_metadata
        torch.testing.assert_close(
            core.page_table, reference.page_table, rtol=0, atol=0
        )
        for name in fields:
            assert getattr(core, name).data_ptr() == pointers[name]
    assert not torch.equal(core.positions_casual, initial_positions)
    assert not glue.disabled
    assert glue.replays == 3


@pytest.mark.parametrize("block_size", [1, 5])
def test_captured_draft_swa_metadata_uses_live_request_mapping(block_size):
    backend = _backend(block_size)
    view = _view(8, block_size, torch.int64)
    kwargs = dict(
        max_seq_len=2048,
        req_pool_indices=view.req_pool_indices,
        seq_lens=view.seq_lens,
        seq_lens_cpu=None,
        out_cache_loc=view.out_cache_loc,
        block_size=block_size,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.init_forward_metadata_dspark_draft_block(**kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = backend.init_forward_metadata_dspark_draft_block(**kwargs)
    for step in range(3):
        view.seq_lens.copy_(
            torch.tensor(
                [0, 1, 127, 128, 129, 255, 256, 1023],
                dtype=view.seq_lens.dtype,
                device="cuda",
            )
            + step
        )
        view.req_pool_indices.add_(1).remainder_(16)
        view.out_cache_loc.add_(8 * block_size)
        backend.token_to_kv_pool.full_to_swa_index_mapping.add_(1)
        graph.replay()
        reference = backend.init_forward_metadata_dspark_draft_block(
            **dict(kwargs, seq_lens_cpu=view.seq_lens.cpu())
        )
        actual_core, expected_core = (
            captured.core_attn_metadata,
            reference.core_attn_metadata,
        )
        for name in (
            "seq_lens_casual",
            "positions_casual",
            "page_table",
            "swa_page_indices",
            "swa_topk_lengths",
            "raw_out_loc",
        ):
            torch.testing.assert_close(
                getattr(actual_core, name), getattr(expected_core, name), rtol=0, atol=0
            )
