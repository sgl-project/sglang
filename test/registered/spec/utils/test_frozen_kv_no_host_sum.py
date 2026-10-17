"""Regression checks for the opt-in Triton Frozen-KV draft path."""

from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.verify_mask import VerifyMask
from sglang.srt.speculative.eagle_utils import TreeMaskMode, build_tree_kernel_efficient
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def fixture(device="cpu", bs=3):
    target = object.__new__(TritonAttnBackend)
    target.max_context_len = 4352
    target._verify_mask = VerifyMask(
        buffer=torch.zeros(8 * 3 * (4352 + 3), dtype=torch.uint8, device=device),
        mode=TreeMaskMode.FULL_MASK,
        max_bs=8,
    )
    worker = NS(
        topk=1,
        cuda_graph_runner=object(),
        draft_attn_backend=target,
        target_worker=NS(model_runner=NS(attn_backend=target)),
        speculative_num_draft_tokens=3,
    )
    batch = NS(
        seq_lens=torch.empty(bs, dtype=torch.int64, device=device),
        seq_lens_cpu=None,
        reqs=[
            NS(
                origin_input_ids=[0] * n,
                sampling_params=NS(max_new_tokens=128),
                multimodal_inputs=None,
                session=None,
                session_id=None,
                input_embeds=None,
                retracted_stain=False,
                is_retracted=False,
                to_finish=None,
                finished_reason=None,
            )
            for n in [17, 255, 4096][:bs]
        ],
    )
    return worker, batch, target


def test_bound_and_fallbacks():
    worker, batch, target = fixture()
    method = FrozenKVMTPDraftWorker._no_host_sum_mask
    with envs.SGLANG_FROZEN_KV_NO_HOST_SUM.override(False):
        assert method(worker, batch) is None
    with envs.SGLANG_FROZEN_KV_NO_HOST_SUM.override(True):
        mask = method(worker, batch)
        assert mask.data_ptr() == target.verify_mask.buffer.data_ptr()
        # Covers every legal request length through completion, not just the
        # current device lengths, and does not initialize max-batch capacity.
        actual_max = sum(len(r.origin_input_ids) + 128 for r in batch.reqs)
        assert mask.numel() >= actual_max * 3 + 3 * 9
        assert mask.numel() < target.verify_mask.buffer.numel()
        worker.topk = 2
        assert method(worker, batch) is None
        worker.topk = 1
        batch.seq_lens_cpu = torch.ones(3)
        assert method(worker, batch) is None
        batch.seq_lens_cpu = None
        worker.cuda_graph_runner = None
        assert method(worker, batch) is None
        worker.cuda_graph_runner = object()
        target._verify_mask = VerifyMask(
            buffer=torch.empty(1), mode=TreeMaskMode.FULL_MASK, max_bs=1
        )
        assert method(worker, batch) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("multimodal_inputs", object()),
        ("session", object()),
        ("session_id", "continued-session"),
        ("input_embeds", object()),
        ("retracted_stain", True),
        ("to_finish", object()),
        ("finished_reason", object()),
    ],
)
def test_request_lifecycle_falls_back(field, value):
    worker, batch, _ = fixture()
    # One unsafe row must reject the whole batch, including after re-prefill
    # cleared is_retracted. Never size from a shortened/partial prompt.
    req = batch.reqs[1]
    setattr(req, field, value)
    req.origin_input_ids = [0]
    assert not req.is_retracted
    with envs.SGLANG_FROZEN_KV_NO_HOST_SUM.override(True):
        assert FrozenKVMTPDraftWorker._no_host_sum_mask(worker, batch) is None


def test_storage_capacity_not_just_batch_capacity():
    worker, batch, target = fixture()
    method = FrozenKVMTPDraftWorker._no_host_sum_mask
    with envs.SGLANG_FROZEN_KV_NO_HOST_SUM.override(True):
        required = method(worker, batch).numel()
        for size in (required - 1, required, required + 1):
            target._verify_mask = VerifyMask(
                buffer=torch.empty(size, dtype=torch.uint8),
                mode=TreeMaskMode.FULL_MASK,
                max_bs=8,
            )
            result = method(worker, batch)
            if size < required:
                assert result is None
            else:
                assert result.numel() == required
                assert result.data_ptr() == target.verify_mask.buffer.data_ptr()


def test_missing_output_limit_and_request_count_fall_back():
    worker, batch, _ = fixture()
    with envs.SGLANG_FROZEN_KV_NO_HOST_SUM.override(True):
        batch.reqs[1].sampling_params.max_new_tokens = None
        assert FrozenKVMTPDraftWorker._no_host_sum_mask(worker, batch) is None
        batch.reqs[1].sampling_params.max_new_tokens = 128
        batch.reqs.pop()
        assert FrozenKVMTPDraftWorker._no_host_sum_mask(worker, batch) is None


def test_request_bound_covers_overlap_lookahead_and_context_cap():
    worker, batch, target = fixture()
    width = worker.speculative_num_draft_tokens
    with envs.SGLANG_FROZEN_KV_NO_HOST_SUM.override(True):
        for max_new in (0, 1, 128, 4096):
            for req in batch.reqs:
                req.sampling_params.max_new_tokens = max_new
            buf = FrozenKVMTPDraftWorker._no_host_sum_mask(worker, batch)
            assert buf is not None
            for unpublished in range(2 * width + 1):
                lengths = [
                    min(
                        target.max_context_len,
                        len(req.origin_input_ids) + max_new + unpublished,
                    )
                    for req in batch.reqs
                ]
                assert width * sum(lengths) + len(lengths) * width**2 <= buf.numel()
            assert buf.numel() <= len(batch.reqs) * width * (
                target.max_context_len + width
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tree builder")
def test_reused_uint8_mask_matches_allocated_bool_across_length_changes():
    worker, batch, target = fixture("cuda")
    device = "cuda"
    kwargs = {
        "bonus_tokens": torch.tensor([101, 102, 103], dtype=torch.int32, device=device),
        "parent_list": torch.tensor([[0, 0]] * 3, dtype=torch.int64, device=device),
        "top_scores_index": torch.tensor(
            [[0, 1]] * 3, dtype=torch.int64, device=device
        ),
        "draft_tokens": torch.arange(6, dtype=torch.int64, device=device).view(3, 2),
        "topk": 1,
        "spec_steps": 2,
        "num_verify_tokens": 3,
    }
    # Shifting row offsets and SWA-boundary lengths must not preserve stale
    # False tree cells in what has now become an attended prefix.
    for lengths in (
        [17, 255, 4096],
        [20, 257, 4099],
        [145, 383, 4224],
        [18, 256, 4097],
    ):
        batch.seq_lens = torch.tensor(lengths, dtype=torch.int64, device=device)
        ref = build_tree_kernel_efficient(
            **kwargs, seq_lens=batch.seq_lens, seq_lens_sum=sum(lengths)
        )
        with envs.SGLANG_FROZEN_KV_NO_HOST_SUM.override(True):
            buf = FrozenKVMTPDraftWorker._no_host_sum_mask(worker, batch)
        got = build_tree_kernel_efficient(
            **kwargs, seq_lens=batch.seq_lens, seq_lens_sum=0, tree_mask_buf=buf
        )
        assert torch.equal(got[0][: ref[0].numel()].bool(), ref[0])
        for actual, expected in zip(got[1:], ref[1:]):
            assert torch.equal(actual, expected)
        assert got[0].data_ptr() == target.verify_mask.buffer.data_ptr()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
