"""Owner-encoded image spans: real gloo groups, production coordinator and entrypoints."""

import json
import sys
from contextlib import nullcontext
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
from sglang.srt.managers import mm_owner_embedding, mm_schedule, mm_utils
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


def _image(item_hash: int, start: int, rows: int, grid=None) -> MultimodalDataItem:
    h, w = _grid(rows) if grid is None else grid
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.zeros(1),
        offsets=[(start, start + rows - 1)],
        model_specific_data={"n_vit_h": h, "n_vit_w": w},
    )
    item.set_hash(item_hash)
    return item


def _request(parts, prefix_len: int = 0, extend_len=None, rid: str = "rid"):
    """``parts`` mixes token ids with ``(hash, rows)`` or ``(hash, rows, grid)`` images."""
    ids, items = [], []
    for part in parts:
        if isinstance(part, tuple):
            item = _image(part[0], len(ids), part[1], *part[2:])
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
        self.calls.append(
            SimpleNamespace(input_ids=input_ids, input_embeds=input_embeds)
        )
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
        self.fail_hashes = set()

    def get_image_feature(self, items):
        hashes = [item.hash for item in items]
        if self.fail_hashes.intersection(hashes):
            raise RuntimeError("injected encoder failure")
        self.encoded.extend(hashes)
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
        self.logits_calls.append(SimpleNamespace(hidden_states=hidden_states))
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
        use_pymscclpp=False,
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


def _assert_traces_agree(results, ranks):
    traces = [results[rank]["trace"] for rank in ranks]
    assert all(trace == traces[0] for trace in traces), traces


def _expect_protocol_error(run):
    try:
        run()
    except MmOwnerProtocolError as exc:
        return str(exc)
    raise AssertionError("the entrypoint did not raise MmOwnerProtocolError")


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
    tags = {A: CACHE_TAG + 1, B: owner_tag[B], C: CACHE_TAG, D: owner_tag[D]}
    requests = [
        _request([10, 11, (A, ROWS[A]), 12, (B, ROWS[B]), 13], rid="r1"),
        _request([20, (C, ROWS[C]), 21, (D, ROWS[D]), 22], rid="r2"),
    ]
    _, embeds = model.prepare(_batch(requests))
    assert torch.equal(embeds, _expected_embeds(embed, requests, tags))
    encoded = [list(model.encoded)]
    traces = [list(traced.trace)]

    # Local 2 loses every entry and admits nothing new; B's owner evicts B.
    if local == 2:
        mm_schedule.init_mm_embedding_cache(0)
    if local == 1:
        mm_schedule.embedding_cache.free(B, None)
    model.encoded.clear()
    traced.trace.clear()
    tags = {A: CACHE_TAG + 1, B: owner_tag[B], C: owner_tag[C], D: owner_tag[D]}
    requests = [
        _request([10, 11, (A, ROWS[A]), (B, ROWS[B])], rid="r3"),
        _request([20, (A, ROWS[A]), (C, ROWS[C]), (D, ROWS[D])], rid="r4"),
    ]
    _, embeds = model.prepare(_batch(requests))
    assert torch.equal(embeds, _expected_embeds(embed, requests, tags))
    encoded.append(list(model.encoded))
    traces.append(list(traced.trace))

    # A cold span crossing the chunk boundary.
    for prefix_len, extend_len in ((0, 5), (5, 4)):
        model.encoded.clear()
        traced.trace.clear()
        request = _request(
            [30, 31, (E, ROWS[E]), 32],
            prefix_len=prefix_len,
            extend_len=extend_len,
            rid="r5",
        )
        _, embeds = model.prepare(_batch([request]))
        assert torch.equal(
            embeds, _expected_embeds(embed, [request], {E: owner_tag[E]})
        )
        encoded.append(list(model.encoded))
        traces.append(list(traced.trace))
    return {"trace": traces, "encoded": encoded, "outside": False}


def _topology_program(rank, world_size):
    tp8 = _coordinator([[0, 1, 2, 3, 4, 5, 6, 7]], rank)
    replicas = _coordinator([[0, 1, 2, 3], [4, 5, 6, 7]], rank)
    singles = _coordinator([[r] for r in range(8)], rank)
    out = {}

    tp_group, attn_tp, attn_cp = (
        _TracedGroup(tp8),
        _TracedGroup(tp8),
        _TracedGroup(singles),
    )
    with get_parallel().override(
        tp_size=8,
        attn_dp_size=1,
        attn_cp_size=1,
        tp_group=tp_group,
        attn_tp_group=attn_tp,
        attn_cp_group=attn_cp,
    ):
        selected = select_owner_group(get_parallel())
    assert selected is attn_tp
    mm_schedule.init_mm_embedding_cache(1 << 20)
    embed = _ReducingEmbedding(tp8)
    model = _VisionStub(embed, selected, tag=rank)
    x, y, z = 800, 805, 810  # owners 0, 5, 2
    requests = [
        _request([10, (x, 3), 11], rid="c1"),
        _request([20, (y, 2), (z, 4), 21], rid="c2"),
    ]
    _, embeds = model.prepare(_batch(requests))
    assert torch.equal(embeds, _expected_embeds(embed, requests, {x: 0, y: 5, z: 2}))
    out["cp1"] = {"encoded": list(model.encoded), "trace": list(attn_tp.trace)}

    tp_group, attn_tp, attn_cp = (
        _TracedGroup(tp8),
        _TracedGroup(singles),
        _TracedGroup(tp8),
    )
    init_cp_strategy(enable_prefill_cp=True, cp_size=8, cp_strategy="interleave")
    try:
        with get_parallel().override(
            tp_size=8,
            attn_dp_size=1,
            attn_cp_size=8,
            attn_cp_rank=rank,
            tp_group=tp_group,
            attn_tp_group=attn_tp,
            attn_cp_group=attn_cp,
        ):
            selected = select_owner_group(get_parallel())
            assert selected is attn_cp
            mm_schedule.init_mm_embedding_cache(1 << 20)
            torch.manual_seed(0)
            plain_embed = nn.Embedding(VOCAB, HIDDEN)
            model = _VisionStub(plain_embed, selected, tag=rank)
            w = 803  # owner 3; rows land on ranks 6,7,0,1,2,3 so ranks 4,5 hold no image row
            requests = [
                _request([10, 11, 12, 13, 14], rid="p1"),
                _request([20, (w, 6), 21, 22], rid="p2"),
            ]
            forward_batch = _batch(requests)
            full = _expected_embeds(plain_embed, requests, {w: 3})
            _run_cp_extend(model, forward_batch, tp8)
            physical = forward_batch.attn_cp_metadata.per_rank_actual_token[rank]
            (body,) = model.model.calls
            shard = full[rank::8]
            assert torch.equal(body.input_embeds[: shard.shape[0]], shard)
            assert body.input_embeds.shape[0] == physical
            (logits,) = model.logits_calls
            assert torch.equal(logits.hidden_states, full)
            assert torch.equal(forward_batch.mm_input_embeds, full)
            out["cp8"] = {
                "encoded": list(model.encoded),
                "trace": list(attn_cp.trace),
                "image_rows": int((body.input_ids == IMAGE_TOKEN_ID).sum()),
            }
    finally:
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")

    # One attention-DP replica is text-only first.
    tp_group, attn_tp, attn_cp = (
        _TracedGroup(tp8),
        _TracedGroup(replicas),
        _TracedGroup(singles),
    )
    with get_parallel().override(
        tp_size=8,
        attn_dp_size=2,
        attn_cp_size=1,
        tp_group=tp_group,
        attn_tp_group=attn_tp,
        attn_cp_group=attn_cp,
    ):
        selected = select_owner_group(get_parallel())
    assert selected is attn_tp
    mm_schedule.init_mm_embedding_cache(1 << 20)
    embed = _ReducingEmbedding(replicas)
    model = _VisionStub(embed, selected, tag=rank)
    p, q = 900, 901  # owners: local 0 (rank 0) and local 1 (rank 5)
    replica = rank // 4
    if replica == 0:
        requests = [_request([10, (p, 3), 11], rid="d0")]
    else:
        requests = [_request([10, 11, 12], rid="d1")]
    _, embeds = model.prepare(_batch(requests))
    if replica == 0:
        assert torch.equal(embeds, _expected_embeds(embed, requests, {p: 0}))
    else:
        assert embeds is None
    first = {"encoded": list(model.encoded), "trace": list(attn_tp.trace)}
    model.encoded.clear()
    attn_tp.trace.clear()
    if replica == 0:
        requests = [_request([10, (p, 3), 11], rid="d2")]
        tags = {p: 0}
    else:
        requests = [_request([30, (q, 2), 31], rid="d3")]
        tags = {q: 5}
    _, embeds = model.prepare(_batch(requests))
    assert torch.equal(embeds, _expected_embeds(embed, requests, tags))
    out["dp2"] = [first, {"encoded": list(model.encoded), "trace": list(attn_tp.trace)}]
    return out


F, G, H, I = 300, 301, 302, 303  # owners 0, 1, 2, 3 in a four-rank group
J, K = 310, 311  # owner 2 and owner 3, eight-row spans for the grid cases


def _failure_program(rank, world_size):
    group = _coordinator([[0, 1, 2, 3]], rank)
    out = {}

    def fresh(embed_cls=_ReducingEmbedding):
        mm_schedule.init_mm_embedding_cache(1 << 20)
        traced = _TracedGroup(group)
        embed = embed_cls(group)
        return traced, embed, _VisionStub(embed, traced, tag=rank)

    def inject(module, name, message, on_rank):
        if rank != on_rank:
            return nullcontext()
        return patch.object(module, name, side_effect=RuntimeError(message))

    def record(name, traced, embed, model, error, **extra):
        out[name] = {
            "error": error,
            "trace": list(traced.trace),
            "embed_calls": embed.calls,
            "encoded": list(model.encoded),
            **extra,
        }

    def remap_oom(target):
        original = torch.Tensor.masked_fill

        def masked_fill(self, mask, value):
            if self is target:
                raise torch.OutOfMemoryError("injected remap OOM")
            return original(self, mask, value)

        return patch.object(torch.Tensor, "masked_fill", masked_fill)

    def sync_failure(on_call):
        seen = []

        def synchronize(device):
            seen.append(device)
            if len(seen) == on_call:
                raise RuntimeError("injected final sync")

        return patch.object(mm_owner_embedding, "_synchronize", synchronize)

    traced, embed, model = fresh()
    batch = _batch([_request([10, (G, 3), 11], rid="fail-prepare")])
    with (
        patch.object(
            torch,
            "as_tensor",
            side_effect=torch.OutOfMemoryError("injected placeholder OOM"),
        )
        if rank == 1
        else nullcontext()
    ):
        error = _expect_protocol_error(lambda: model.prepare(batch))
    record("prepare", traced, embed, model, error)

    traced, embed, model = fresh()
    if rank == 1:
        model.fail_hashes = {G}
    error = _expect_protocol_error(
        lambda: model.prepare(_batch([_request([10, (G, 3), 11], rid="fail-encode")]))
    )
    record("encode", traced, embed, model, error)

    traced, embed, model = fresh()
    with inject(
        mm_owner_embedding, "_new_span_buffer", "injected allocation", on_rank=3
    ):
        error = _expect_protocol_error(
            lambda: model.prepare(
                _batch([_request([10, (F, 3), 11], rid="fail-alloc")])
            )
        )
    record("alloc", traced, embed, model, error)

    traced, embed, model = fresh()
    rows = 4 if rank == 2 else 3
    error = _expect_protocol_error(
        lambda: model.prepare(
            _batch([_request([10, (F, rows), 11], rid="fail-manifest")])
        )
    )
    record("manifest", traced, embed, model, error)

    traced, embed, model = fresh()
    grid = (4, 3) if rank == 2 else (3, 4)
    error = _expect_protocol_error(
        lambda: model.prepare(
            _batch([_request([10, (J, 8, grid), 11], rid="fail-grid")])
        )
    )
    record("grid", traced, embed, model, error)

    traced, embed, model = fresh()
    error = _expect_protocol_error(
        lambda: model.prepare(
            _batch(
                [
                    _request([10, (K, 8, (3, 4)), 11], rid="fail-dup-a"),
                    _request([20, (K, 8, (4, 3)), 21], rid="fail-dup-b"),
                ]
            )
        )
    )
    record("duplicate", traced, embed, model, error)

    traced, embed, model = fresh()
    error = _expect_protocol_error(
        lambda: model.prepare(
            _batch([_request([10, (F, 5, (2, 2)), 11], rid="fail-span-len")])
        )
    )
    record("span_len", traced, embed, model, error)

    traced, embed, model = fresh()
    with inject(
        mm_schedule, "_assemble_per_image_chunk", "injected assembly", on_rank=1
    ):
        error = _expect_protocol_error(
            lambda: model.prepare(
                _batch([_request([10, (H, 2), 11], rid="fail-assemble")])
            )
        )
    record("assemble", traced, embed, model, error)

    traced, embed, model = fresh()
    with inject(mm_utils, "_scatter_mm_embedding", "injected merge", on_rank=3):
        error = _expect_protocol_error(
            lambda: model.prepare(
                _batch([_request([10, (I, 2), 11], rid="fail-merge")])
            )
        )
    record("merge", traced, embed, model, error)

    traced, embed, model = fresh()
    batch = _batch([_request([10, (F, 3), 11], rid="fail-remap")])
    with remap_oom(batch.input_ids) if rank == 1 else nullcontext():
        error = _expect_protocol_error(lambda: model.prepare(batch))
    record("remap", traced, embed, model, error)

    traced, embed, model = fresh()
    with sync_failure(on_call=3) if rank == 1 else nullcontext():
        error = _expect_protocol_error(
            lambda: model.prepare(
                _batch([_request([10, (F, 3), 11], rid="fail-final-sync")])
            )
        )
    record("final_sync", traced, embed, model, error)

    traced, embed, model = fresh()
    with (
        patch.object(
            mm_owner_embedding,
            "_make_plan",
            side_effect=MemoryError("injected leader plan allocation"),
        )
        if rank == 0
        else nullcontext()
    ):
        error = _expect_protocol_error(
            lambda: model.prepare(_batch([_request([10, (G, 3), 11], rid="fail-plan")]))
        )
    record("leader_plan", traced, embed, model, error)

    init_cp_strategy(enable_prefill_cp=True, cp_size=4, cp_strategy="interleave")
    try:
        traced, embed, model = fresh(lambda group: _ReducingEmbedding(None))
        batch = _batch([_request([10, (F, 3), 11], rid="fail-remap-cp")])
        with (
            get_parallel().override(
                attn_cp_size=4, attn_cp_rank=rank, attn_cp_group=traced
            ),
            remap_oom(batch.input_ids) if rank == 1 else nullcontext(),
        ):
            error = _expect_protocol_error(lambda: _run_cp_extend(model, batch, group))
        record(
            "remap_cp", traced, embed, model, error, body_calls=len(model.model.calls)
        )
    finally:
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")
    return out


MANIFEST = ["all_gather_object", "RankManifest", 4]
PLAN = ["broadcast_object", 0, 4]
STATUS = ["all_gather_object", "RankStatus", 4]


def _bcast(src, rows, size=4):
    return ["broadcast", src, [rows, HIDDEN], size]


def test_owner_actions_and_cache_lifetime_across_asymmetric_ranks():
    """Owner hits broadcast, non-owner hits still receive, stale entries miss,
    all-hit moves no payload, per-forward references outlive the cache."""
    results = _run_ranks(5, _owner_lifetime_program)
    members = range(1, 5)
    _assert_traces_agree(results, members)
    encoded = [results[rank]["encoded"] for rank in members]
    trace = results[1]["trace"]

    # Forward 1: A cache-broadcast, B and D owner-encoded, C a local hit.
    assert [e[0] for e in encoded] == [[], [B], [], [D]]
    assert trace[0] == [
        MANIFEST,
        PLAN,
        STATUS,
        _bcast(0, ROWS[A]),
        _bcast(1, ROWS[B]),
        _bcast(3, ROWS[D]),
        STATUS,
        STATUS,
    ]

    # Forward 2: evicted owners of B and C re-encode once; A recurs but moves once.
    assert [e[1] for e in encoded] == [[], [B], [C], []]
    assert trace[1] == [
        MANIFEST,
        PLAN,
        STATUS,
        _bcast(0, ROWS[A]),
        _bcast(1, ROWS[B]),
        _bcast(2, ROWS[C]),
        _bcast(3, ROWS[D]),
        STATUS,
        STATUS,
    ]

    # Forwards 3 and 4: the chunk-crossing span is encoded fully once.
    assert [e[2] for e in encoded] == [[E], [], [], []]
    assert [e[3] for e in encoded] == [[], [], [], []]
    assert trace[2] == [MANIFEST, PLAN, STATUS, _bcast(0, ROWS[E]), STATUS, STATUS]
    assert trace[3] == [MANIFEST, PLAN, STATUS, _bcast(0, ROWS[E]), STATUS, STATUS]
    assert results[0]["outside"]


def test_tp8_cp1_and_cp8_dedupe_and_attention_dp_replicas_stay_isolated():
    results = _run_ranks(8, _topology_program)

    cp1 = [r["cp1"] for r in results]
    assert [c["encoded"] for c in cp1] == [[800], [], [810], [], [], [805], [], []]
    assert all(c["trace"] == cp1[0]["trace"] for c in cp1)
    assert [op for op in cp1[0]["trace"] if op[0] == "broadcast"] == [
        _bcast(0, 3, 8),
        _bcast(5, 2, 8),
        _bcast(2, 4, 8),
    ]

    cp8 = [r["cp8"] for r in results]
    assert [c["encoded"] for c in cp8] == [[], [], [], [803], [], [], [], []]
    assert all(c["trace"] == cp8[0]["trace"] for c in cp8)
    assert [op for op in cp8[0]["trace"] if op[0] == "broadcast"] == [_bcast(3, 6, 8)]
    assert [c["image_rows"] for c in cp8] == [1, 1, 1, 1, 0, 0, 1, 1]

    dp2 = [r["dp2"] for r in results]
    first = [d[0] for d in dp2]
    assert [f["encoded"] for f in first] == [[900], [], [], [], [], [], [], []]
    assert all(f["trace"] == [] for f in first[4:])
    assert all(f["trace"] == first[0]["trace"] for f in first[:4])
    assert all(op[-1] == 4 for op in first[0]["trace"])
    second = [d[1] for d in dp2]
    assert [s["encoded"] for s in second] == [[], [], [], [], [], [901], [], []]
    assert all(s["trace"] == second[0]["trace"] for s in second[:4])
    assert all(s["trace"] == second[4]["trace"] for s in second[4:])
    assert [op for op in second[0]["trace"] if op[0] == "broadcast"] == []
    assert [op for op in second[4]["trace"] if op[0] == "broadcast"] == [
        _bcast(1, 2, 4)
    ]


def test_failures_agree_before_payload_text_embedding_or_body():
    results = _run_ranks(4, _failure_program)
    cases = (
        "prepare",
        "encode",
        "alloc",
        "manifest",
        "grid",
        "duplicate",
        "span_len",
        "assemble",
        "merge",
        "remap",
        "final_sync",
        "leader_plan",
        "remap_cp",
    )
    for case in cases:
        errors = {r[case]["error"] for r in results}
        assert len(errors) == 1, (case, errors)
        assert all(r[case]["trace"] == results[0][case]["trace"] for r in results), case

    def calls(case, field="embed_calls"):
        return [r[case][field] for r in results]

    prepare = results[0]["prepare"]
    assert "during prepare on group rank 1" in prepare["error"]
    assert "injected placeholder OOM" in prepare["error"]
    assert prepare["trace"] == [MANIFEST, PLAN]
    assert calls("prepare") == [0, 0, 0, 0]

    encode = results[0]["encode"]
    assert "during encode on group rank 1" in encode["error"]
    assert "fail-encode" in encode["error"] and "301" in encode["error"]
    assert encode["trace"] == [MANIFEST, PLAN, STATUS]
    assert calls("encode") == [0, 0, 0, 0]

    alloc = results[0]["alloc"]
    assert "during encode on group rank 3" in alloc["error"]
    assert "hash 300 shape (3, 8)" in alloc["error"]
    assert "injected allocation" in alloc["error"]
    assert alloc["trace"] == [MANIFEST, PLAN, STATUS]
    assert calls("alloc", "encoded") == [[F], [], [], []]
    assert calls("alloc") == [0, 0, 0, 0]

    manifest = results[0]["manifest"]
    assert "manifest mismatch between group ranks 0 and 2" in manifest["error"]
    assert manifest["trace"] == [MANIFEST, PLAN]
    assert calls("manifest") == [0, 0, 0, 0]

    grid = results[0]["grid"]
    assert "manifest mismatch between group ranks 0 and 2" in grid["error"]
    assert "(3, 4" in grid["error"] and "(4, 3" in grid["error"]
    assert grid["trace"] == [MANIFEST, PLAN]

    duplicate = results[0]["duplicate"]
    assert "during manifest on group rank 0" in duplicate["error"]
    assert (
        f"image hash {K} (8 tokens) occurs with different geometry"
        in duplicate["error"]
    )
    assert duplicate["trace"] == [MANIFEST, PLAN]

    span_len = results[0]["span_len"]
    assert "yields 4 span tokens, placeholder has 5" in span_len["error"]
    assert span_len["trace"] == [MANIFEST, PLAN]

    assemble = results[0]["assemble"]
    assert "during features on group rank 1" in assemble["error"]
    assert "injected assembly" in assemble["error"]
    assert assemble["trace"] == [MANIFEST, PLAN, STATUS, _bcast(2, 2), STATUS]
    assert calls("assemble") == [0, 0, 0, 0]

    merge = results[0]["merge"]
    assert "during finalize on group rank 3" in merge["error"]
    assert merge["trace"] == [MANIFEST, PLAN, STATUS, _bcast(3, 2), STATUS, STATUS]
    assert calls("merge") == [1, 1, 1, 1]

    remap = results[0]["remap"]
    assert "during finalize on group rank 1" in remap["error"]
    assert "injected remap OOM" in remap["error"]
    assert remap["trace"] == [MANIFEST, PLAN, STATUS, _bcast(0, 3), STATUS, STATUS]
    assert calls("remap") == [1, 1, 1, 1]

    final_sync = results[0]["final_sync"]
    assert "during finalize on group rank 1" in final_sync["error"]
    assert "injected final sync" in final_sync["error"]
    assert final_sync["trace"] == [MANIFEST, PLAN, STATUS, _bcast(0, 3), STATUS, STATUS]
    assert calls("final_sync") == [1, 1, 1, 1]

    leader_plan = results[0]["leader_plan"]
    assert "during plan on group rank 0" in leader_plan["error"]
    assert "injected leader plan allocation" in leader_plan["error"]
    assert leader_plan["trace"] == [MANIFEST, PLAN]
    assert calls("leader_plan") == [0, 0, 0, 0]
    assert calls("leader_plan", "encoded") == [[], [], [], []]

    remap_cp = results[0]["remap_cp"]
    assert "during finalize on group rank 1" in remap_cp["error"]
    assert remap_cp["trace"] == [MANIFEST, PLAN, STATUS, _bcast(0, 3), STATUS, STATUS]
    assert calls("remap_cp", "body_calls") == [0, 0, 0, 0]


def test_text_decode_prefilled_future_and_precomputed_paths_pay_no_collective():
    """Chunks without owner-encoded image rows never touch the group; an overlapping chunk does."""
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

    prefilled = [_request([10, (A, 3), 11, 12], prefix_len=5, rid="pf")]
    ids, embeds = owner_model.prepare(_batch(prefilled))
    _, legacy_embeds = legacy_model.prepare(_batch(prefilled))
    assert torch.equal(embeds, legacy_embeds)
    assert torch.equal(ids, torch.tensor([12]))

    future = [_request([10, 11, 12, 13, (A, 3)], extend_len=2, rid="fut")]
    ids, embeds = owner_model.prepare(_batch(future))
    _, legacy_embeds = legacy_model.prepare(_batch(future))
    assert torch.equal(embeds, legacy_embeds)
    assert torch.equal(ids, torch.tensor([10, 11]))

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
        attn_tp_group=SimpleNamespace(world_size=attn_tp_size, name="attn_tp"),
        attn_cp_group=SimpleNamespace(world_size=attn_cp_group_size, name="attn_cp"),
    )


def test_select_owner_group_follows_the_replication_domain():
    """R = TP / attention-DP selects the attention-TP group, TP-aliased CP its handle, else None."""
    assert select_owner_group(_parallel(8, 1, 1, 8, 1)).name == "attn_tp"
    assert select_owner_group(_parallel(8, 1, 8, 1, 8)).name == "attn_cp"
    assert select_owner_group(_parallel(8, 2, 1, 4, 1)).name == "attn_tp"
    assert select_owner_group(_parallel(1, 1, 1, 1, 1)) is None
    assert select_owner_group(_parallel(8, 2, 4, 1, 4)) is None
    assert select_owner_group(_parallel(8, 1, 4, 2, 4)) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
