"""The draft argmax exchange on a real TP group (torchrun).

SGLang's groups are set up as the server's bootstrap does. A draft
LogitsProcessor opted in by EagleDraftWorker returns all-reduced vocab-shard
argmax partials under the draft_argmax_only flag; its draft-decode and
draft-extend tokens equal the gathered-logits path of a processor that is not
opted in, eagerly and inside a CUDA graph."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.speculative.topk1 import (
    draft_topk1_argmax_from_partials,
    draft_topk1_postprocess,
    draft_topk1_postprocess_from_partials,
    draft_topk1_shard_partials,
    draft_topk1_shard_partials_width,
)
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers import flashinfer_comm_fusion as fi
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.layer_boundary import FUSE_ALLREDUCE_MAX_BATCH_SIZE
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_exec, get_forward, get_server_args
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

BF16 = torch.bfloat16
HIDDEN = 6144
VOCAB = 154880
ROWS = (1, 3, 16, 48)
EXTEND_ROWS_PER_REQ = 3


@cache_once
def _init_tp_group() -> ps.GroupCoordinator:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    blackwell = torch.cuda.get_device_capability()[0] >= 10
    publish_build_topology(
        world_rank=rank,
        tp_size=world,
        device="cuda",
        flashinfer_allreduce_fusion_backend="auto" if blackwell else None,
    )
    fusion = get_exec().comm.flashinfer_allreduce_fusion_backend
    ps.set_custom_all_reduce(not get_exec().comm.disable_custom_all_reduce)
    ps.set_flashinfer_allreduce_only(
        fusion is not None and not fi.uses_cutedsl_ar_fusion()
    )
    ps.initialize_model_parallel()
    ps._tag_groups_for_flashinfer_allreduce_only()
    initialize_dp_attention(server_args=get_server_args())
    if fusion is not None:
        fi.pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE, hidden_dim=HIDDEN, dtype=BF16
        )
    return ps.get_tp_group()


class _ModelStub(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.logits_processor = LogitsProcessor(SimpleNamespace(vocab_size=vocab))


@cache_once
def _processors_and_head():
    """(target processor, opted-in draft processor, LM head), opted in the way
    EagleDraftWorker does at init."""
    _init_tp_group()
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.topk = 1
    target, draft = _ModelStub(VOCAB), _ModelStub(VOCAB)
    worker.target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=target))
    worker.draft_runner = SimpleNamespace(model=draft)
    worker._init_draft_argmax_exchange()
    with torch.device("cuda"):
        head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=BF16)
    g = torch.Generator(device="cuda").manual_seed(100 + dist.get_rank())
    with torch.no_grad():
        head.weight.copy_(
            torch.randn(head.weight.shape, generator=g, device="cuda") * 0.02
        )
    return worker, target.logits_processor, draft.logits_processor, head


def _hidden(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)  # the same on every rank
    return torch.randn(rows, HIDDEN, generator=g, device="cuda").to(BF16)


def _forward(proc, head, hidden, mode, select, *, flag):
    metadata = LogitsMetadata(forward_mode=mode, draft_extend_select_index=select)
    input_ids = torch.zeros(hidden.shape[0], dtype=torch.long, device="cuda")
    with get_forward().scoped(draft_argmax_only=flag):
        out = proc(input_ids, hidden, head, metadata)
    torch.cuda.synchronize()
    dist.barrier()
    return out


def _exchange_method_or_skip(rows):
    tp = _init_tp_group()
    width = draft_topk1_shard_partials_width(
        local_width=VOCAB // tp.world_size, tp_size=tp.world_size
    )
    probe = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    method = tp.device_all_reduce_method(probe)
    if method is None:
        pytest.skip("the TP all-reduce would run on NCCL: the exchange stays off")
    return tp, method


def _assert_equal(a, b, tag):
    assert a.shape == b.shape and a.dtype == b.dtype, f"{tag}: {a.shape} vs {b.shape}"
    assert torch.equal(a, b), f"{tag}: differs at {(a != b).nonzero()[:4].tolist()}"


@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_partials_all_reduce_is_exact(rows):
    tp, _ = _exchange_method_or_skip(rows)
    local_width = VOCAB // tp.world_size
    g = torch.Generator(device="cuda").manual_seed(rows * 7 + tp.rank_in_group)
    local = (torch.randn(rows, local_width, generator=g, device="cuda") * 3).to(BF16)
    contrib = draft_topk1_shard_partials(
        local, tp_rank=tp.rank_in_group, tp_size=tp.world_size, vocab_size=VOCAB - 10
    )
    expected = contrib.clone()
    dist.all_reduce(expected, group=tp.device_group)
    got = tp.all_reduce(contrib)
    _assert_equal(got.view(torch.int32), expected.view(torch.int32), f"rows={rows}")


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.DRAFT_EXTEND_V2])
@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_logits_processor_returns_exact_partials(mode, rows):
    _exchange_method_or_skip(rows)
    worker, target, draft, head = _processors_and_head()
    assert worker.draft_argmax_exchange and draft.draft_argmax_exchange
    assert not target.draft_argmax_exchange
    if mode == ForwardMode.DECODE:
        hidden, select = _hidden(rows, seed=rows), None
    else:
        hidden = _hidden(rows * EXTEND_ROWS_PER_REQ, seed=rows + 1000)
        offsets = torch.arange(rows, device="cuda")
        select = offsets * EXTEND_ROWS_PER_REQ + offsets % EXTEND_ROWS_PER_REQ
    ref = _forward(target, head, hidden, mode, select, flag=True)
    new = _forward(draft, head, hidden, mode, select, flag=True)
    assert ref.next_token_argmax_partials is None and ref.next_token_logits is not None
    assert new.next_token_logits is None and new.next_token_argmax_partials is not None
    logits, partials = ref.next_token_logits, new.next_token_argmax_partials
    assert tuple(logits.shape) == (rows, VOCAB)

    pos0 = torch.arange(rows, device="cuda") * 3
    tok0 = torch.zeros(rows, 4, dtype=torch.long, device="cuda")
    pos_ref, tok_ref, pos_new, tok_new = (
        pos0.clone(),
        tok0.clone(),
        pos0.clone(),
        tok0.clone(),
    )
    p_ref, i_ref = draft_topk1_postprocess(logits, pos_ref, tok_ref, 2)
    p_new, i_new = draft_topk1_postprocess_from_partials(partials, pos_new, tok_new, 2)
    for what, a, b in (
        ("topk_index", i_ref, i_new),
        ("topk_p", p_ref, p_new),
        ("positions", pos_ref, pos_new),
        ("draft_tokens", tok_ref, tok_new),
    ):
        _assert_equal(a, b, f"{mode.name} rows={rows} {what}")
    _, i_new = draft_topk1_argmax_from_partials(partials)
    _assert_equal(torch.argmax(logits, dim=-1, keepdim=True), i_new, "argmax")
    # Without the flag the opted-in processor returns logits.
    off = _forward(draft, head, hidden, mode, select, flag=False)
    assert off.next_token_logits is not None and off.next_token_argmax_partials is None


@torch.inference_mode()
def test_non_vocab_parallel_head_falls_back_to_logits():
    _exchange_method_or_skip(4)
    _, _, draft, head = _processors_and_head()
    plain = nn.Module()
    plain.weight = head.weight
    out = _forward(
        draft, plain, _hidden(4, seed=5), ForwardMode.DECODE, None, flag=True
    )
    assert out.next_token_logits is not None and out.next_token_argmax_partials is None


@pytest.mark.parametrize("rows", (1, 4, 16))
@torch.inference_mode()
def test_cuda_graph_replay(rows):
    _exchange_method_or_skip(rows)
    _, target, draft, head = _processors_and_head()
    hidden = torch.zeros(rows, HIDDEN, dtype=BF16, device="cuda")
    pos = torch.zeros(rows, dtype=torch.long, device="cuda")
    toks = torch.zeros(rows, 3, dtype=torch.long, device="cuda")
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
    input_ids = torch.zeros(rows, dtype=torch.long, device="cuda")

    def step():
        with get_forward().scoped(draft_argmax_only=True):
            out = draft(input_ids, hidden, head, metadata)
        draft_topk1_postprocess_from_partials(
            out.next_token_argmax_partials, pos, toks, 1
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # compile and decide the all-reduce outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    for it in range(1, 4):
        hidden.copy_(_hidden(rows, seed=31 * it + rows))
        pos.copy_(torch.arange(rows, device="cuda") * 5 + it)
        toks.fill_(-1)
        pos_ref, tok_ref = pos.clone(), toks.clone()
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        ref = _forward(target, head, hidden, ForwardMode.DECODE, None, flag=True)
        draft_topk1_postprocess(ref.next_token_logits, pos_ref, tok_ref, 1)
        _assert_equal(pos_ref, pos, f"rows={rows} replay={it} positions")
        _assert_equal(tok_ref, toks, f"rows={rows} replay={it} draft_tokens")
    del graph


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
