"""Where every attention layer writes and reads, pinned for a refactor of how
write locations are carried.

For a unified pool with one window sub-pool, one with several, a static
hybrid sliding-window pool and a fused draft pool, the test drives batches
through the Triton backend (the draft through its pool's write door), records
each layer's written rows, the read rails the backend builds and the draft's
layer routing, and compares them with a recorded golden. Under deterministic
inference an extend reads its own new tokens back from the cache through one
kernel, so those configurations also record what each layer's extend reads. Regenerate the golden
with ``SGLANG_REGEN_WINDOW_WRITE_GOLDEN=1`` only when a change is meant to move
KV.
"""

import json
import os
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel, reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods import (
    dense_attention as _kit,
)
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    MockModelRunner,
    TinyModelConfig,
)
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

# The kit enters a process-wide single-rank parallel override when imported;
# keep it to this module's tests so later tests in the process see none.
_kit._parallel_override.__exit__(None, None, None)

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HEADS, _KV_HEADS, _HEAD_DIM = 4, 2, 64
_DTYPE = torch.float16
_CONTEXT = 64
_GOLDEN = Path(__file__).with_name("window_write_locations_golden.json")
_ROWS = [1, 2, 3]
_PREFIX, _EXTEND = [5, 19, 2], [3, 4, 9]


def _geometry():
    return MHAGeometry(
        head_num=_KV_HEADS,
        head_dim=_HEAD_DIM,
        v_head_dim=_HEAD_DIM,
        kv_cache_dtype=_DTYPE,
    )


def _unified_storage(windows, *, fused_draft=None):
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_geometry(),
        windows=[
            WindowLayers(layer_ids=(i + 1,), geometry=_geometry(), window=w)
            for i, w in enumerate(windows)
        ],
    )
    plan = plan_kv_storage(decls, page_size=1, fused_draft=fused_draft)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=512,
            window_tokens=512,
            state_slots=0,
            max_running_requests=4,
            context_len=_CONTEXT,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device="cuda",
        start_layer=0,
        end_layer=len(windows) + 1,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=4,
        model_context_len=_CONTEXT,
        lazy_compaction=False,
    )
    return build_kv_storage(plan, sized, ctx), plan


def _static_pools(window):
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    pool = SWAKVPool(
        size=256,
        size_swa=256,
        page_size=1,
        dtype=_DTYPE,
        head_num=_KV_HEADS,
        head_dim=_HEAD_DIM,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cuda",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=256,
        size_swa=256,
        page_size=1,
        dtype=_DTYPE,
        device="cuda",
        kvcache=pool,
        need_sort=False,
    )
    requests = ReqToTokenPool(
        size=4, max_context_len=_CONTEXT, device="cuda", enable_memory_saver=False
    )
    return pool, allocator, requests


def _marker(layer_id, token, step):
    return (layer_id + 1, token + 1, step)


def _kv(layer_id, n, step):
    """K and V whose first three channels mark (layer, token, step)."""
    k = torch.zeros((n, _KV_HEADS, _HEAD_DIM), device="cuda", dtype=_DTYPE)
    for t in range(n):
        k[t, :, 0], k[t, :, 1], k[t, :, 2] = _marker(layer_id, t, step)
    return k, k.clone()


def _buffer_tag(buf):
    storage = buf.untyped_storage().data_ptr()
    return storage, buf.data_ptr() - storage


def _written_rows(pool, layer_ids, n, step):
    """Per layer: the buffer it writes (storage ordinal, byte offset) and, per
    token of the step, the rows holding that token's marker."""
    storages = []
    out = {}
    for layer_id in layer_ids:
        buf = pool.get_key_buffer(layer_id)
        storage, offset = _buffer_tag(buf)
        if storage not in storages:
            storages.append(storage)
        flat = buf[:, 0, :3].float()
        rows = []
        for t in range(n):
            a, b, c = _marker(layer_id, t, step)
            hit = ((flat[:, 0] == a) & (flat[:, 1] == b) & (flat[:, 2] == c)).nonzero()
            rows.append(hit.flatten().tolist())
        out[str(layer_id)] = {
            "storage": storages.index(storage),
            "offset": offset,
            "rows": rows,
        }
    return out


def _rail(indptr, indices, bs):
    end = int(indptr[bs].item())
    return {
        "indptr": indptr[: bs + 1].tolist(),
        "indices": indices[:end].tolist(),
    }


def _reads(backend, bs):
    meta = backend.forward_metadata
    out = {"full": _rail(meta.kv_indptr, meta.kv_indices, bs)}
    if meta.window_kv_indptr is not None:
        out["window"] = _rail(meta.window_kv_indptr, meta.window_kv_indices, bs)
    for name, rail in sorted((meta.other_windows or {}).items()):
        out[f"window:{name}"] = _rail(rail.kv_indptr, rail.kv_indices, bs)
    return out


class _Rig:
    """A Triton backend over ``pool`` with layer 0 full and one layer per
    window in ``windows``."""

    def __init__(self, windows, pool, allocator, requests, *, deterministic=False):
        case = DenseAttentionCase(
            name="window_write_locations",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=_HEADS,
            num_kv_heads=_KV_HEADS,
            page_size=1,
            prefix_lens=(0,) * 4,
            sliding_window_size=max(windows),
        )
        runner = MockModelRunner(
            case=case,
            model_config=TinyModelConfig(
                num_heads=_HEADS,
                num_kv_heads=_KV_HEADS,
                head_dim=_HEAD_DIM,
                hidden_size=_HEADS * _HEAD_DIM,
                context_len=_CONTEXT,
                sliding_window_size=max(windows),
            ),
            dtype=_DTYPE,
            device="cuda",
            max_context_len=_CONTEXT,
            head_dim=_HEAD_DIM,
            runner_batch_size=4,
        )
        self.pool = pool
        self.allocator = allocator
        runner.req_to_token_pool = requests
        runner.token_to_kv_pool = pool
        runner.token_to_kv_pool_allocator = allocator
        runner.sliding_window_size = max(windows)
        runner.init_runner_ids()
        self.runner = runner
        # The extend kernel's whole read list (prefix and new tokens), per
        # layer, captured under deterministic inference.
        self.extend_reads = None
        config = None
        if deterministic:
            config = get_context().override_server_args(
                enable_deterministic_inference=True
            )
            config.install()
        try:
            self.backend = TritonAttnBackend(runner)
        finally:
            if config is not None:
                config.restore()
        if deterministic:
            assert self.backend.enable_deterministic
            self.extend_reads = []
            build = self.backend.build_unified_kv_indices

            def capture(*args, **kwargs):
                indptr, indices, prefix_lens = build(*args, **kwargs)
                self.extend_reads.append((indptr, indices))
                return indptr, indices, prefix_lens

            self.backend.build_unified_kv_indices = capture
        runner.runner_ids.bind_and_verify_backends([self.backend])
        self.layers = [
            RadixAttention(
                num_heads=_HEADS,
                head_dim=_HEAD_DIM,
                scaling=_HEAD_DIM**-0.5,
                num_kv_heads=_KV_HEADS,
                layer_id=layer_id,
                sliding_window_size=window,
            )
            for layer_id, window in enumerate((-1, *windows))
        ]
        self.ids = {}

    def add_request(self, row, length):
        self.ids[row] = self.allocator.alloc(length)
        self.runner.req_to_token_pool.req_to_token[row, :length] = self.ids[row].to(
            torch.int32
        )

    def batch(self, mode, prefix_lens, input_lens):
        seq_lens = [p + n for p, n in zip(prefix_lens, input_lens)]
        positions = [p + i for p, n in zip(prefix_lens, input_lens) for i in range(n)]
        out = torch.cat(
            [self.ids[r][p : p + n] for r, p, n in zip(_ROWS, prefix_lens, input_lens)]
        )
        batch = ForwardBatch(
            forward_mode=mode,
            batch_size=len(_ROWS),
            input_ids=torch.zeros(len(positions), dtype=torch.int64, device="cuda"),
            req_pool_indices=torch.tensor(_ROWS, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            out_cache_loc=out,
            seq_lens_sum=sum(seq_lens),
            positions=torch.tensor(positions, dtype=torch.int64, device="cuda"),
        )
        self.runner.runner_ids.bind_own_plan(batch)
        if mode.is_extend():
            batch.extend_prefix_lens = torch.tensor(
                prefix_lens, dtype=torch.int32, device="cuda"
            )
            batch.extend_prefix_lens_cpu = list(prefix_lens)
            batch.extend_seq_lens = torch.tensor(
                input_lens, dtype=torch.int32, device="cuda"
            )
            batch.extend_seq_lens_cpu = list(input_lens)
            batch.extend_num_tokens = sum(input_lens)
        return batch

    def inputs(self, n, step):
        out = []
        for layer in self.layers:
            q = torch.zeros((n, _HEADS * _HEAD_DIM), device="cuda", dtype=_DTYPE)
            k, v = _kv(layer.layer_id, n, step)
            out.append((q, k, v))
        return out

    def forward(self, batch, inputs):
        outs = []
        for layer, (q, k, v) in zip(self.layers, inputs):
            fn = (
                self.backend.forward_decode
                if batch.forward_mode.is_decode()
                else self.backend.forward_extend
            )
            outs.append(fn(q, k, v, layer, batch))
        return outs

    def record(self, mode, prefix_lens, input_lens, step):
        batch = self.batch(mode, prefix_lens, input_lens)
        n = sum(input_lens)
        self.backend.init_forward_metadata(batch)
        reads = _reads(self.backend, len(_ROWS))
        if self.extend_reads is not None:
            self.extend_reads.clear()
        self.forward(batch, self.inputs(n, step))
        torch.cuda.synchronize()
        writes = _written_rows(self.pool, [l.layer_id for l in self.layers], n, step)
        record = {"reads": reads, "writes": writes}
        if self.extend_reads:
            # One kernel launch per layer, in layer order.
            record["extend_reads"] = {
                str(layer.layer_id): _rail(indptr, indices, len(_ROWS))
                for layer, (indptr, indices) in zip(self.layers, self.extend_reads)
            }
        return record

    def run_extends(self):
        """Prefix, then extend past it: each step's reads and writes."""
        for row, p, n in zip(_ROWS, _PREFIX, _EXTEND):
            self.add_request(row, p + n + 2)
        zeros = [0] * len(_ROWS)
        return {
            "prefix": self.record(ForwardMode.EXTEND, zeros, _PREFIX, 1),
            "extend": self.record(ForwardMode.EXTEND, _PREFIX, _EXTEND, 2),
        }

    def run(self):
        """Prefix, extend past it, decode one token: each step's reads and writes."""
        for row, p, n in zip(_ROWS, _PREFIX, _EXTEND):
            self.add_request(row, p + n + 2)
        zeros = [0] * len(_ROWS)
        ones = [1] * len(_ROWS)
        seq = [p + n for p, n in zip(_PREFIX, _EXTEND)]
        return {
            "prefix": self.record(ForwardMode.EXTEND, zeros, _PREFIX, 1),
            "extend": self.record(ForwardMode.EXTEND, _PREFIX, _EXTEND, 2),
            "decode": self.record(ForwardMode.DECODE, seq, ones, 3),
        }

    def run_graph(self):
        """A decode graph captured on one batch, replayed on another: the
        replay's reads and writes."""
        for row, p, n in zip(_ROWS, _PREFIX, _EXTEND):
            self.add_request(row, p + n + 2)
        zeros = [0] * len(_ROWS)
        ones = [1] * len(_ROWS)
        full = [p + n for p, n in zip(_PREFIX, _EXTEND)]
        self.record(ForwardMode.EXTEND, zeros, full, 1)
        backend = self.backend
        backend.init_cuda_graph_state(max_bs=len(_ROWS), max_num_tokens=len(_ROWS))
        capture = self.batch(ForwardMode.DECODE, [2, 3, 4], ones)
        static = self.inputs(len(_ROWS), 9)
        backend.init_forward_metadata_out_graph(capture, in_capture=True)
        backend.init_forward_metadata_in_graph(capture)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.forward(capture, static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.forward(capture, static)
        replay = self.batch(ForwardMode.DECODE, full, ones)
        capture.out_cache_loc.copy_(replay.out_cache_loc)
        for (sq, sk, sv), (q, k, v) in zip(static, self.inputs(len(_ROWS), 4)):
            sq.copy_(q)
            sk.copy_(k)
            sv.copy_(v)
        backend.init_forward_metadata_out_graph(replay)
        reads = _reads(backend, len(_ROWS))
        graph.replay()
        torch.cuda.synchronize()
        writes = _written_rows(
            self.pool, [l.layer_id for l in self.layers], len(_ROWS), 4
        )
        return {"reads": reads, "writes": writes}


def _unified_rig(windows, *, deterministic=False):
    setup_allocator_context()
    storage, _ = _unified_storage(windows)
    allocator = storage.allocator
    return _Rig(
        windows,
        allocator.get_kvcache(),
        allocator,
        storage.requests,
        deterministic=deterministic,
    )


def _static_rig(window, *, deterministic=False):
    setup_allocator_context()
    pool, allocator, requests = _static_pools(window)
    return _Rig((window,), pool, allocator, requests, deterministic=deterministic)


def _draft_write_loc(batch, runner_ids):
    """The write loc a backend bundles for a fused draft's layers."""
    window = runner_ids.write_ids(batch, runner_ids.window_kv_kind)
    return KVWriteLoc.for_batch(batch, swa_loc=window)


def _fused_draft():
    """A unified target (full layer 0, window 16 layer 1) with a fused draft
    of one full layer and one window-8 layer: the draft pool's routes and
    where its layers write."""
    from sglang.srt.mem_cache.layout.fused_draft import (
        LAYER_FULL,
        LAYER_WINDOW,
        DraftKVGeometry,
        DraftKVProfile,
        DraftLayerSet,
        PlacementContext,
        place_fused_draft,
    )
    from sglang.srt.mem_cache.unified_draft_pool import routed_draft_pool

    setup_allocator_context()
    geometry = DraftKVGeometry(
        head_num=_KV_HEADS, head_dim=_HEAD_DIM, v_head_dim=_HEAD_DIM
    )
    placement = place_fused_draft(
        profile=DraftKVProfile(
            num_layers=2,
            num_depths=1,
            kinds={
                LAYER_FULL: DraftLayerSet(layer_ids=(0,), geometry=geometry),
                LAYER_WINDOW: DraftLayerSet(
                    layer_ids=(1,), geometry=geometry, window=8
                ),
            },
        ),
        num_runners=1,
        ctx=PlacementContext(host_names=("full", "window_16"), target_windows=(16,)),
        store_dtype=_DTYPE,
    ).placement
    storage, plan = _unified_storage((16,), fused_draft=placement)
    allocator = storage.allocator
    # Draft layers by host: the full layer rides the owner, the window layer
    # its own band.
    hosts = list(placement.hosts())
    lanes = {host: list(placement.lanes_for(0, host)) for host in hosts}
    layers_by_host = {
        host: [1] if getattr(plan.sub_pool(host).lifetime, "window", None) else [0]
        for host in hosts
    }
    pool = routed_draft_pool(
        arena=allocator.arena,
        plan=plan,
        owner=plan.owner().name,
        page_size=1,
        lanes_by_host={
            host: dict(zip(layers_by_host[host], lanes[host])) for host in hosts
        },
    )
    runner_ids = RunnerIds(
        req_to_token=storage.requests.req_to_token,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool,
        page_size=1,
        device="cuda",
        is_draft_worker=True,
    )
    router = pool.layer_router
    routes = {
        str(layer_id): {
            "sub_pool": router.route(layer_id).sub_pool,
            "index": router.route(layer_id).index,
            "storage": router.storage_name(router.route(layer_id).sub_pool),
        }
        for layer_id in (0, 1)
    }
    ids = {row: allocator.alloc(p + n) for row, p, n in zip(_ROWS, _PREFIX, _EXTEND)}
    for row, p, n in zip(_ROWS, _PREFIX, _EXTEND):
        storage.requests.req_to_token[row, : p + n] = ids[row].to(torch.int32)
    seq = [p + n for p, n in zip(_PREFIX, _EXTEND)]
    out = torch.cat([ids[r] for r in _ROWS])
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(_ROWS),
        input_ids=torch.zeros(sum(seq), dtype=torch.int64, device="cuda"),
        req_pool_indices=torch.tensor(_ROWS, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(seq, dtype=torch.int32, device="cuda"),
        seq_lens_cpu=torch.tensor(seq, dtype=torch.int32),
        out_cache_loc=out,
        seq_lens_sum=sum(seq),
        positions=torch.zeros(sum(seq), dtype=torch.int64, device="cuda"),
    )
    runner_ids.bind_own_plan(batch)
    loc = _draft_write_loc(batch, runner_ids)
    layers = [
        RadixAttention(
            num_heads=_HEADS,
            head_dim=_HEAD_DIM,
            scaling=_HEAD_DIM**-0.5,
            num_kv_heads=_KV_HEADS,
            layer_id=layer_id,
            sliding_window_size=window,
        )
        for layer_id, window in ((0, -1), (1, 8))
    ]
    for layer in layers:
        k, v = _kv(layer.layer_id, sum(seq), 5)
        pool.set_kv_buffer(layer, loc, k, v)
    torch.cuda.synchronize()
    return {
        "routes": routes,
        "window_kind": str(
            getattr(runner_ids.window_kv_kind, "value", runner_ids.window_kv_kind)
        ),
        "writes": _written_rows(pool, [0, 1], sum(seq), 5),
    }


def _observe():
    return {
        "unified_one_window": _unified_rig((8,)).run(),
        "unified_one_window_graph": _unified_rig((8,)).run_graph(),
        "unified_several_windows": _unified_rig((8, 4)).run(),
        "unified_several_windows_graph": _unified_rig((16, 8, 3)).run_graph(),
        "static_window": _static_rig(8).run(),
        "static_window_graph": _static_rig(8).run_graph(),
        "unified_one_window_deterministic": _unified_rig(
            (8,), deterministic=True
        ).run_extends(),
        "static_window_deterministic": _static_rig(8, deterministic=True).run_extends(),
        "fused_draft": _fused_draft(),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestWindowWriteLocations(CustomTestCase):
    def setUp(self):
        override = get_parallel().override(attn_tp_size=1, attn_dcp_rank=0)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)

    def tearDown(self):
        # The pools published a runtime context; later tests start from none.
        reset_context()

    def test_reads_writes_and_routes_match_the_golden(self):
        observed = json.loads(json.dumps(_observe()))
        if os.environ.get("SGLANG_REGEN_WINDOW_WRITE_GOLDEN") == "1":
            _GOLDEN.write_text(json.dumps(observed, indent=1, sort_keys=True) + "\n")
        golden = json.loads(_GOLDEN.read_text())
        for config in golden:
            with self.subTest(config=config):
                self.assertEqual(observed[config], golden[config])
        self.assertEqual(sorted(observed), sorted(golden))
        # Every layer wrote every token of every step somewhere.
        for config, steps in observed.items():
            records = [steps] if "writes" in steps else list(steps.values())
            for record in records:
                for layer, rows in record["writes"].items():
                    self.assertTrue(
                        all(len(hit) == 1 for hit in rows["rows"]),
                        f"{config} layer {layer}: {rows['rows']}",
                    )


if __name__ == "__main__":
    unittest.main()
