"""Owner-encoded image spans: real gloo groups, production coordinator and entrypoints."""

import json
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing
from torch import nn

from sglang.srt.distributed import parallel_state
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.utils import prepare_cp_forward
from sglang.srt.managers import mm_owner_embedding, mm_schedule
from sglang.srt.managers.mm_owner_embedding import (
    MmOwnerProtocolError,
    select_owner_group,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

HIDDEN = 8
VOCAB = 64
IMAGE_TOKEN_ID = 7
CACHE_TAG = 50
DOWNSAMPLE = 2
GLOO_TIMEOUT = timedelta(seconds=90)


def _span(item_hash: int, rows: int, tag: int) -> torch.Tensor:
    """Distinguishable full span: the producing rank is visible in the values."""
    base = float(item_hash * 1000 + tag * 100)
    return torch.arange(rows * HIDDEN, dtype=torch.float32).view(rows, HIDDEN) + base


def _grid(rows: int):
    """A ViT grid whose downsampled span has exactly ``rows`` tokens."""
    return (0, 0) if rows == 2 else (DOWNSAMPLE, DOWNSAMPLE * (rows - 3))


def _image(item_hash: int, start: int, rows: int) -> MultimodalDataItem:
    h, w = _grid(rows)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.zeros(1),
        offsets=[(start, start + rows - 1)],
        model_specific_data={"n_vit_h": h, "n_vit_w": w},
    )
    item.set_hash(item_hash)
    return item


def _request(parts, prefix_len: int = 0, extend_len=None, rid: str = "rid"):
    """``parts`` mixes token ids with ``(hash, rows)`` images."""
    ids, items = [], []
    for part in parts:
        if isinstance(part, tuple):
            item = _image(part[0], len(ids), part[1])
            items.append(item)
            ids.extend([item.pad_value] * part[1])
        else:
            ids.append(part)
    if extend_len is None:
        extend_len = len(ids) - prefix_len
    return SimpleNamespace(
        ids=ids, items=items, prefix_len=prefix_len, extend_len=extend_len, rid=rid
    )


def _chunk_ids(req):
    return req.ids[req.prefix_len : req.prefix_len + req.extend_len]


def _batch(requests, mode=ForwardMode.EXTEND):
    return SimpleNamespace(
        forward_mode=mode,
        mm_inputs=[
            MultimodalInputs(mm_items=req.items, im_token_id=IMAGE_TOKEN_ID)
            if req.items
            else None
            for req in requests
        ],
        extend_prefix_lens_cpu=[req.prefix_len for req in requests],
        extend_seq_lens_cpu=[req.extend_len for req in requests],
        seq_lens_cpu=[req.prefix_len + req.extend_len for req in requests],
        input_ids=torch.tensor(
            [tok for req in requests for tok in _chunk_ids(req)], dtype=torch.long
        ),
        positions=torch.cat(
            [
                torch.arange(req.prefix_len, req.prefix_len + req.extend_len)
                for req in requests
            ]
        ),
        mm_input_embeds=None,
        attn_cp_metadata=None,
        global_num_tokens_cpu=None,
        out_cache_loc=None,
        input_ids_global=torch.zeros(1, dtype=torch.long),
        rids=[req.rid for req in requests],
    )


def _expected_embeds(embed: nn.Embedding, requests, tags):
    """``tags`` maps image hash to the tag of the span every rank must end up with."""
    rows = []
    with torch.no_grad():
        for req in requests:
            ids = torch.tensor(_chunk_ids(req), dtype=torch.long)
            chunk = nn.functional.embedding(ids.clamp(max=VOCAB - 1), embed.weight)
            chunk_end = req.prefix_len + req.extend_len
            for item in req.items:
                start, end = item.offsets[0]
                lo = max(start, req.prefix_len) - start
                hi = min(end + 1, chunk_end) - start
                if lo >= hi:
                    continue
                span = _span(item.hash, end - start + 1, tags[item.hash])
                dest = max(start, req.prefix_len) - req.prefix_len
                chunk[dest : dest + hi - lo] = span[lo:hi]
            rows.append(chunk)
    return torch.cat(rows)


def _seed_cache(item_hash: int, rows: int, tag: int) -> None:
    mm_schedule.embedding_cache.set(
        item_hash, mm_schedule.EmbeddingResult(embedding=_span(item_hash, rows, tag))
    )


class _RecordingBody:
    def __init__(self, embed):
        self.embed = embed
        self.calls = []

    def get_input_embeddings(self):
        return self.embed

    def __call__(self, input_ids, positions, forward_batch, input_embeds=None):
        self.calls.append(input_ids)
        return input_embeds, input_embeds


class _ReducingEmbedding(nn.Embedding):
    """Stands in for the TP vocab embedding: a real all-reduce on the device group."""

    def __init__(self, coordinator):
        torch.manual_seed(0)
        super().__init__(VOCAB, HIDDEN)
        self.coordinator = coordinator
        self.calls = 0

    def forward(self, input_ids):
        self.calls += 1
        if self.coordinator is not None:
            dist.all_reduce(torch.zeros(1), group=self.coordinator.device_group)
        return super().forward(input_ids)


class _VisionStub(DeepseekV4ForCausalLM):
    def __init__(self, embed, owner_group, tag: int):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN_ID,
            hidden_size=HIDDEN,
            vision_downsample_ratio=DOWNSAMPLE,
        )
        self.vision = object()
        self.tp_size = 1
        self.mm_owner_group = owner_group
        self.image_start = nn.Parameter(torch.zeros(HIDDEN))
        self.model = _RecordingBody(embed)
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.lm_head = object()
        self.capture_aux_hidden_states = False
        self.logits_calls = []
        self.tag = tag
        self.encoded = []

    def get_image_feature(self, items):
        self.encoded.extend(item.hash for item in items)
        return [
            _span(item.hash, item.offsets[0][1] - item.offsets[0][0] + 1, self.tag)
            for item in items
        ]

    def logits_processor(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states=None,
        hidden_states_before_norm=None,
    ):
        self.logits_calls.append(hidden_states)
        return object()

    def prepare(self, forward_batch):
        with torch.no_grad():
            return self.prepare_model_inputs(
                input_ids=forward_batch.input_ids,
                forward_batch=forward_batch,
                input_embeds=None,
            )


class _TracedGroup:
    def __init__(self, inner):
        self.inner = inner
        self.trace = []

    @property
    def world_size(self):
        return self.inner.world_size

    @property
    def rank_in_group(self):
        return self.inner.rank_in_group

    @property
    def ranks(self):
        return self.inner.ranks

    def all_gather_object(self, obj):
        self.trace.append(["all_gather_object", type(obj).__name__, self.world_size])
        return self.inner.all_gather_object(obj)

    def broadcast_object(self, obj=None, src=0):
        self.trace.append(["broadcast_object", src, self.world_size])
        return self.inner.broadcast_object(obj, src=src)

    def broadcast(self, tensor, src=0):
        self.trace.append(["broadcast", src, list(tensor.shape), self.world_size])
        return self.inner.broadcast(tensor, src=src)


class _ForbiddenGroup:
    world_size = 4
    rank_in_group = 0
    ranks = [0, 1, 2, 3]

    def __getattr__(self, name):
        raise AssertionError(f"collective {name!r} reached on a fast path")


def _coordinator(group_ranks, rank):
    return parallel_state.GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        use_message_queue_broadcaster=False,
        group_name="mm_owner_test",
        gloo_timeout=GLOO_TIMEOUT,
    )


def _init_rank(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=GLOO_TIMEOUT,
    )
    parallel_state._MODEL_PARALLEL_GROUP_TIMEOUT = GLOO_TIMEOUT


def _run_ranks(world_size: int, target):
    with TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "gloo-init")
        torch.multiprocessing.spawn(
            _rank_main,
            args=(world_size, init_file, directory, target),
            nprocs=world_size,
        )
        return [
            json.loads((Path(directory) / f"rank{rank}.json").read_text())
            for rank in range(world_size)
        ]


def _rank_main(rank, world_size, init_file, directory, target):
    _init_rank(rank, world_size, init_file)
    try:
        result = target(rank, world_size)
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@torch.no_grad()
def _run_cp_extend(model, forward_batch, coordinator):
    def gather(output, input_tensor):
        dist.all_gather_into_tensor(
            output, input_tensor, group=coordinator.device_group
        )

    runner = EagerRunner.__new__(EagerRunner)
    runner.model_runner = SimpleNamespace(model=model)
    with (
        patch("torch.cuda.current_stream", return_value=None),
        patch(
            "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
            side_effect=gather,
        ),
        patch(
            "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.cp.interleave.use_symmetric_memory",
            return_value=torch.no_grad(),
        ),
    ):
        prepare_cp_forward(forward_batch)
        runner._execute_extend_cp(forward_batch, {})


@contextmanager
def _interleave_cp(cp_size, **parallel):
    init_cp_strategy(enable_prefill_cp=True, cp_size=cp_size, cp_strategy="interleave")
    try:
        with get_parallel().override(attn_cp_size=cp_size, **parallel):
            yield
    finally:
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")


MANIFEST = ["all_gather_object", "RankManifest", 4]
PLAN = ["broadcast_object", 0, 4]
STATUS = ["all_gather_object", "RankStatus", 4]


def _bcast(src, rows, size=4):
    return ["broadcast", src, [rows, HIDDEN], size]


def _cold(*payloads):
    return [MANIFEST, PLAN, STATUS, *payloads, STATUS, STATUS]


# Global rank 0 sits outside the owner group, so a group-local source index that
# leaks out as a global rank is caught.
A, B, C, D, E = 400, 401, 402, 403, 404  # owner = hash % 4 -> group ranks 0,1,2,3,0
ROWS = {A: 3, B: 4, C: 2, D: 3, E: 6}


def _owner_lifetime_program(rank, world_size):
    group = _coordinator([[0], [1, 2, 3, 4]], rank)
    if rank == 0:
        return {"trace": [], "encoded": [], "outside": True}
    traced = _TracedGroup(group)
    local = group.rank_in_group
    owner_tag = {key: 1 + key % 4 for key in ROWS}
    mm_schedule.init_mm_embedding_cache(1 << 20)
    embed = _ReducingEmbedding(group)
    model = _VisionStub(embed, traced, tag=rank)

    # A valid only on its owner, B valid only on a non-owner, C valid
    # everywhere, D stale on its owner.
    if local == 0:
        _seed_cache(A, ROWS[A], CACHE_TAG + rank)
    if local == 2:
        _seed_cache(B, ROWS[B], CACHE_TAG + rank)
    _seed_cache(C, ROWS[C], CACHE_TAG)
    if local == 3:
        _seed_cache(D, ROWS[D] + 2, CACHE_TAG + rank)
    forwards = [
        (
            [
                _request([10, 11, (A, ROWS[A]), 12, (B, ROWS[B]), 13], rid="r1"),
                _request([20, (C, ROWS[C]), 21, (D, ROWS[D]), 22], rid="r2"),
            ],
            {A: CACHE_TAG + 1, B: owner_tag[B], C: CACHE_TAG, D: owner_tag[D]},
        ),
        (
            [
                _request([10, 11, (A, ROWS[A]), (B, ROWS[B])], rid="r3"),
                _request([20, (A, ROWS[A]), (C, ROWS[C]), (D, ROWS[D])], rid="r4"),
            ],
            {A: CACHE_TAG + 1, B: owner_tag[B], C: owner_tag[C], D: owner_tag[D]},
        ),
    ] + [
        (
            [_request([30, 31, (E, ROWS[E]), 32], prefix, extend, rid="r5")],
            {E: owner_tag[E]},
        )
        for prefix, extend in ((0, 5), (5, 4))
    ]
    encoded, traces = [], []
    for index, (requests, tags) in enumerate(forwards):
        if index == 1:
            # Local 2 loses every entry and admits nothing new; B's owner evicts B.
            if local == 2:
                mm_schedule.init_mm_embedding_cache(0)
            if local == 1:
                mm_schedule.embedding_cache.free(B, None)
        model.encoded.clear()
        traced.trace.clear()
        _, embeds = model.prepare(_batch(requests))
        assert torch.equal(embeds, _expected_embeds(embed, requests, tags))
        encoded.append(list(model.encoded))
        traces.append(list(traced.trace))
    return {"trace": traces, "encoded": encoded, "outside": False}


def test_owner_actions_and_cache_lifetime_across_asymmetric_ranks():
    results = _run_ranks(5, _owner_lifetime_program)
    members = range(1, 5)
    assert all(results[rank]["trace"] == results[1]["trace"] for rank in members)
    encoded = [results[rank]["encoded"] for rank in members]
    trace = results[1]["trace"]

    # Forward 1: A cache-broadcast, B and D owner-encoded, C a local hit.
    assert [e[0] for e in encoded] == [[], [B], [], [D]]
    assert trace[0] == _cold(_bcast(0, ROWS[A]), _bcast(1, ROWS[B]), _bcast(3, ROWS[D]))

    # Forward 2: evicted owners of B and C re-encode once; A recurs but moves once.
    assert [e[1] for e in encoded] == [[], [B], [C], []]
    assert trace[1] == _cold(*(_bcast(key % 4, ROWS[key]) for key in (A, B, C, D)))

    # Forwards 3 and 4: the chunk-crossing span is encoded fully once.
    assert [e[2] for e in encoded] == [[E], [], [], []]
    assert [e[3] for e in encoded] == [[], [], [], []]
    assert trace[2] == trace[3] == _cold(_bcast(0, ROWS[E]))
    assert results[0]["outside"]


def _cp8_program(rank, world_size):
    tp8 = _coordinator([list(range(8))], rank)
    singles = _coordinator([[r] for r in range(8)], rank)
    attn_cp = _TracedGroup(tp8)
    with _interleave_cp(
        8,
        tp_size=8,
        attn_dp_size=1,
        attn_cp_rank=rank,
        tp_group=_TracedGroup(tp8),
        attn_tp_group=_TracedGroup(singles),
        attn_cp_group=attn_cp,
    ):
        selected = select_owner_group(get_parallel())
        assert selected is attn_cp
        mm_schedule.init_mm_embedding_cache(1 << 20)
        torch.manual_seed(0)
        embed = nn.Embedding(VOCAB, HIDDEN)
        model = _VisionStub(embed, selected, tag=rank)
        w = 803  # owner 3; rows land on ranks 6,7,0,1,2,3 so ranks 4,5 hold no image row
        requests = [
            _request([10, 11, 12, 13, 14], rid="p1"),
            _request([20, (w, 6), 21, 22], rid="p2"),
        ]
        _run_cp_extend(model, _batch(requests), tp8)
    (logits,) = model.logits_calls
    assert torch.equal(logits, _expected_embeds(embed, requests, {w: 3}))
    (body,) = model.model.calls
    return {
        "encoded": list(model.encoded),
        "trace": list(attn_cp.trace),
        "image_rows": int((body == IMAGE_TOKEN_ID).sum()),
    }


def test_tp8_cp8_encodes_once_and_merges_before_shard():
    results = _run_ranks(8, _cp8_program)
    assert [r["encoded"] for r in results] == [[], [], [], [803], [], [], [], []]
    assert all(r["trace"] == results[0]["trace"] for r in results)
    assert [op for op in results[0]["trace"] if op[0] == "broadcast"] == [
        _bcast(3, 6, 8)
    ]
    assert [r["image_rows"] for r in results] == [1, 1, 1, 1, 0, 0, 1, 1]


def _raise_on(on_rank, target, name, exc_type=RuntimeError, after=0):
    def inject(rank, batch):
        if rank != on_rank:
            return nullcontext()
        original = getattr(target, name)
        calls = []

        def fail(*args, **kwargs):
            calls.append(None)
            if len(calls) > after:
                raise exc_type(f"injected {name}")
            return original(*args, **kwargs)

        return patch.object(target, name, fail)

    return inject


def _remap_oom_on(on_rank):
    def inject(rank, batch):
        if rank != on_rank:
            return nullcontext()
        original = torch.Tensor.masked_fill

        def masked_fill(self, mask, value):
            if self is batch.input_ids:
                raise torch.OutOfMemoryError("injected remap")
            return original(self, mask, value)

        return patch.object(torch.Tensor, "masked_fill", masked_fill)

    return inject


def _failure(requests, trace, errors, inject=None, vocab_calls=0, cp=False):
    return SimpleNamespace(
        requests=requests,
        trace=trace,
        errors=errors,
        inject=inject or (lambda rank, batch: nullcontext()),
        vocab_calls=vocab_calls,
        cp=cp,
    )


F, G = 300, 301  # owners 0 and 1 in a four-rank group
AT_PLAN = [MANIFEST, PLAN]
AT_FINALIZE_F = _cold(_bcast(0, 3))

FAILURES = {
    "prepare": _failure(
        [[10, (G, 3), 11]],
        AT_PLAN,
        ["during prepare on group rank 1", "OutOfMemoryError: injected as_tensor"],
        _raise_on(1, torch, "as_tensor", torch.OutOfMemoryError),
    ),
    "remap": _failure(
        [[10, (F, 3), 11]],
        AT_FINALIZE_F,
        ["during finalize on group rank 1", "injected remap"],
        _remap_oom_on(1),
        vocab_calls=1,
    ),
    "final_sync": _failure(
        [[10, (F, 3), 11]],
        AT_FINALIZE_F,
        ["during finalize on group rank 1", "injected _synchronize"],
        _raise_on(1, mm_owner_embedding, "_synchronize", after=2),
        vocab_calls=1,
    ),
    "leader_plan": _failure(
        [[10, (G, 3), 11]],
        AT_PLAN,
        ["during plan on group rank 0", "MemoryError: injected _make_plan"],
        _raise_on(0, mm_owner_embedding, "_make_plan", MemoryError),
    ),
    "remap_cp": _failure(
        [[10, (F, 3), 11]],
        AT_FINALIZE_F,
        ["during finalize on group rank 1", "injected remap"],
        _remap_oom_on(1),
        vocab_calls=1,
        cp=True,
    ),
}


def _failure_program(rank, world_size):
    group = _coordinator([[0, 1, 2, 3]], rank)
    out = {}
    for name, case in FAILURES.items():
        mm_schedule.init_mm_embedding_cache(1 << 20)
        traced = _TracedGroup(group)
        embed = _ReducingEmbedding(None if case.cp else group)
        model = _VisionStub(embed, traced, tag=rank)
        batch = _batch([_request(parts, rid=name) for parts in case.requests])
        with ExitStack() as stack, pytest.raises(MmOwnerProtocolError) as raised:
            stack.enter_context(case.inject(rank, batch))
            if case.cp:
                stack.enter_context(
                    _interleave_cp(4, attn_cp_rank=rank, attn_cp_group=traced)
                )
                _run_cp_extend(model, batch, group)
            else:
                model.prepare(batch)
        out[name] = {
            "error": str(raised.value),
            "trace": traced.trace,
            "vocab_calls": embed.calls,
            "body_calls": len(model.model.calls),
        }
    return out


def test_failures_agree_before_payload_text_embedding_or_body():
    results = _run_ranks(4, _failure_program)
    for name, case in FAILURES.items():
        runs = [r[name] for r in results]
        error = runs[0]["error"]
        assert all(run["error"] == error for run in runs), (name, runs)
        assert all(run["trace"] == case.trace for run in runs), (name, runs)
        for fragment in [*case.errors, f"rids=['{name}'"]:
            assert fragment in error, (name, fragment, error)
        assert [run["vocab_calls"] for run in runs] == [case.vocab_calls] * 4, name
        assert [run["body_calls"] for run in runs] == [0] * 4, name


def test_text_decode_prefilled_future_and_precomputed_paths_pay_no_collective():
    mm_schedule.init_mm_embedding_cache(1 << 20)
    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN)
    owner_model = _VisionStub(embed, _ForbiddenGroup(), tag=0)
    legacy_model = _VisionStub(embed, None, tag=0)

    text_only = _batch([_request([10, 11], rid="t0"), _request([20, 21, 22], rid="t1")])
    assert owner_model.prepare(text_only)[1] is None

    with_image = [_request([10, (A, 3), 11, 12], rid="v0")]
    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        assert owner_model.prepare(_batch(with_image, mode))[1] is None

    for request, chunk_ids in (
        (_request([10, (A, 3), 11, 12], prefix_len=5, rid="pf"), [12]),
        (_request([10, 11, 12, 13, (A, 3)], extend_len=2, rid="fut"), [10, 11]),
    ):
        ids, embeds = owner_model.prepare(_batch([request]))
        assert torch.equal(embeds, legacy_model.prepare(_batch([request]))[1])
        assert torch.equal(ids, torch.tensor(chunk_ids))

    precomputed = _request([10, 11, 12, 13], rid="pc")
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        precomputed_embeddings=torch.full((3, HIDDEN), 5.0),
        offsets=[(1, 3)],
    )
    item.set_hash(A)
    precomputed.items = [item]
    precomputed.ids[1:4] = [item.pad_value] * 3
    _, embeds = owner_model.prepare(_batch([precomputed]))
    assert torch.equal(embeds[1:4], torch.full((3, HIDDEN), 5.0))

    overlapping = [_request([10, (A, 3), 11, 12], prefix_len=2, rid="ov")]
    with pytest.raises(AssertionError, match="collective"):
        owner_model.prepare(_batch(overlapping))


def _parallel(tp_size, attn_dp_size, attn_cp_size, attn_tp_size, attn_cp_group_size):
    return SimpleNamespace(
        tp_size=tp_size,
        attn_dp_size=attn_dp_size,
        attn_cp_size=attn_cp_size,
        tp_group=SimpleNamespace(world_size=tp_size, name="tp"),
        attn_tp_group=SimpleNamespace(world_size=attn_tp_size, name="attn_tp"),
        attn_cp_group=SimpleNamespace(world_size=attn_cp_group_size, name="attn_cp"),
    )


def test_select_owner_group_follows_the_replication_domain():
    for layout, expected in (
        ((8, 1, 1, 8, 1), "attn_tp"),
        ((8, 1, 8, 1, 8), "attn_cp"),
        ((8, 2, 1, 4, 1), "attn_tp"),
        ((1, 1, 1, 1, 1), None),
        ((8, 2, 4, 1, 4), None),
        ((8, 1, 4, 2, 4), None),
    ):
        group = select_owner_group(_parallel(*layout))
        assert (group and group.name) == expected, layout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
