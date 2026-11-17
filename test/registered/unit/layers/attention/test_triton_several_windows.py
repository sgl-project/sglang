"""Triton attention over a unified pool whose layers attend through several
window sub-pools: each layer reads and writes its own window's pages, and matches
attention computed from positions with that layer's window."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_WINDOW,
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    PlacementContext,
    place_fused_draft,
)
from sglang.srt.mem_cache.unified_draft_pool import bind_fused_draft
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
_ATOL = _RTOL = 3e-2


def _geometry(dim):
    return MHAGeometry(
        head_num=_KV_HEADS, head_dim=dim, v_head_dim=dim, kv_cache_dtype=_DTYPE
    )


def _per_window(dim, windows):
    return tuple(dim) if isinstance(dim, tuple) else (dim,) * len(windows)


def _storage(
    windows,
    *,
    full_dim=_HEAD_DIM,
    window_dim=_HEAD_DIM,
    full_last=False,
    fused_draft=None,
):
    """One full layer (first, or last) and one layer per window;
    ``window_dim`` is every window layer's width, or each one's."""
    window_dims = _per_window(window_dim, windows)
    full_layer = len(windows) if full_last else 0
    window_layers = [i for i in range(len(windows) + 1) if i != full_layer]
    decls = kv_layer_decls(
        full_layer_ids=(full_layer,),
        full_geometry=_geometry(full_dim),
        windows=[
            WindowLayers(layer_ids=(layer_id,), geometry=_geometry(dim), window=w)
            for layer_id, w, dim in zip(window_layers, windows, window_dims)
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
        lazy_compaction=True,
    )
    return build_kv_storage(plan, sized, ctx)


def _runner(storage, pool, *, model_window, is_draft_worker=False):
    """A mock runner over ``pool`` of the unified ``storage``."""
    case = DenseAttentionCase(
        name="several_windows",
        backend="triton",
        forward_mode=ForwardMode.EXTEND,
        num_heads=_HEADS,
        num_kv_heads=_KV_HEADS,
        page_size=1,
        prefix_lens=(0,) * 4,
        sliding_window_size=model_window,
    )
    runner = MockModelRunner(
        case=case,
        model_config=TinyModelConfig(
            num_heads=_HEADS,
            num_kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            hidden_size=_HEADS * _HEAD_DIM,
            context_len=_CONTEXT,
            sliding_window_size=model_window,
        ),
        dtype=_DTYPE,
        device="cuda",
        max_context_len=_CONTEXT,
        head_dim=_HEAD_DIM,
        runner_batch_size=4,
    )
    runner.req_to_token_pool = storage.requests
    runner.token_to_kv_pool = pool
    runner.token_to_kv_pool_allocator = storage.allocator
    runner.sliding_window_size = model_window
    runner.is_draft_worker = is_draft_worker
    runner.init_runner_ids()
    return runner


class _Model:
    """A unified pool over one full layer and one layer per window, a Triton
    backend bound to its runner, and the K/V every layer has written so far."""

    def __init__(
        self,
        windows,
        *,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        deterministic=False,
        full_dim=_HEAD_DIM,
        window_dim=_HEAD_DIM,
        full_last=False,
        model_window="longest",
    ):
        """``model_window`` is the model's own sliding window: the longest
        window by default, or None for a model that states none."""
        setup_allocator_context()
        storage = _storage(
            windows, full_dim=full_dim, window_dim=window_dim, full_last=full_last
        )
        if model_window == "longest":
            model_window = max(windows)
        case = DenseAttentionCase(
            name="several_windows",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_heads=_HEADS,
            num_kv_heads=_KV_HEADS,
            page_size=1,
            prefix_lens=(0,) * 4,
            sliding_window_size=model_window,
        )
        runner = MockModelRunner(
            case=case,
            model_config=TinyModelConfig(
                num_heads=_HEADS,
                num_kv_heads=_KV_HEADS,
                head_dim=full_dim,
                hidden_size=_HEADS * full_dim,
                context_len=_CONTEXT,
                sliding_window_size=model_window,
            ),
            dtype=_DTYPE,
            device="cuda",
            max_context_len=_CONTEXT,
            head_dim=full_dim,
            runner_batch_size=4,
        )
        self.allocator = storage.allocator
        runner.req_to_token_pool = storage.requests
        runner.token_to_kv_pool = self.allocator.get_kvcache()
        runner.token_to_kv_pool_allocator = self.allocator
        runner.sliding_window_size = model_window
        runner.spec_algorithm = spec_algorithm
        runner.init_runner_ids()
        self.runner = runner
        # Restored by the test (`config.restore`).
        self.config = None
        if deterministic:
            self.config = get_context().override_server_args(
                enable_deterministic_inference=True
            )
            self.config.install()
        self.backend = TritonAttnBackend(runner)
        layers = [(-1, full_dim), *zip(windows, _per_window(window_dim, windows))]
        if full_last:
            layers = layers[1:] + layers[:1]
        self.layers = [
            RadixAttention(
                num_heads=_HEADS,
                head_dim=dim,
                scaling=dim**-0.5,
                num_kv_heads=_KV_HEADS,
                layer_id=layer_id,
                sliding_window_size=window,
            )
            for layer_id, (window, dim) in enumerate(layers)
        ]
        # Per request row: its virtual token ids, then per layer its K and V.
        self.ids = {}
        self.kv = {}

    def bind(self):
        self.runner.runner_ids.bind_and_verify_backends([self.backend])

    def add_request(self, row, length, generator):
        self.ids[row] = self.allocator.alloc(length)
        self.runner.req_to_token_pool.req_to_token[row, :length] = self.ids[row].to(
            torch.int32
        )
        self.kv[row] = [
            (
                torch.randn(
                    (length, _KV_HEADS, layer.head_dim),
                    generator=generator,
                    device="cuda",
                    dtype=_DTYPE,
                ),
                torch.randn(
                    (length, _KV_HEADS, layer.head_dim),
                    generator=generator,
                    device="cuda",
                    dtype=_DTYPE,
                ),
            )
            for layer in self.layers
        ]

    def batch(self, mode, rows, prefix_lens, input_lens):
        seq_lens = [p + n for p, n in zip(prefix_lens, input_lens)]
        positions = [p + i for p, n in zip(prefix_lens, input_lens) for i in range(n)]
        out = torch.cat(
            [self.ids[r][p : p + n] for r, p, n in zip(rows, prefix_lens, input_lens)]
        )
        batch = ForwardBatch(
            forward_mode=mode,
            batch_size=len(rows),
            input_ids=torch.zeros(len(positions), dtype=torch.int64, device="cuda"),
            req_pool_indices=torch.tensor(rows, dtype=torch.int32, device="cuda"),
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

    def inputs(self, rows, prefix_lens, input_lens, generator):
        """Per layer: q, and the k and v the batch writes."""
        n = sum(input_lens)
        result = []
        for layer_id, layer in enumerate(self.layers):
            q = torch.randn(
                (n, _HEADS * layer.head_dim),
                generator=generator,
                device="cuda",
                dtype=_DTYPE,
            )
            k = torch.cat(
                [
                    self.kv[r][layer_id][0][p : p + m]
                    for r, p, m in zip(rows, prefix_lens, input_lens)
                ]
            )
            v = torch.cat(
                [
                    self.kv[r][layer_id][1][p : p + m]
                    for r, p, m in zip(rows, prefix_lens, input_lens)
                ]
            )
            result.append((q, k, v))
        return result

    def forward(self, batch, inputs):
        outputs = []
        for layer, (q, k, v) in zip(self.layers, inputs):
            fn = (
                self.backend.forward_decode
                if batch.forward_mode.is_decode()
                else self.backend.forward_extend
            )
            outputs.append(fn(q, k, v, layer, batch))
        return outputs

    def reference(self, mode, rows, prefix_lens, input_lens, inputs):
        """Attention from positions: a window layer's query at position p reads
        keys from p - window (extend) or p + 1 - window (decode) through p."""
        outputs = []
        for layer, (q, _, _) in zip(self.layers, inputs):
            window = layer.sliding_window_size
            out, start = [], 0
            for r, p, m in zip(rows, prefix_lens, input_lens):
                keys, values = self.kv[r][layer.layer_id]
                for i in range(m):
                    pos = p + i
                    first = 0
                    if window > -1:
                        first = max(
                            0, pos + 1 - window if mode.is_decode() else pos - window
                        )
                    query = q[start + i].view(_HEADS, layer.head_dim).float()
                    group = _HEADS // _KV_HEADS
                    k_ = keys[first : pos + 1].float().repeat_interleave(group, dim=1)
                    v_ = values[first : pos + 1].float().repeat_interleave(group, dim=1)
                    scores = torch.einsum("hd,khd->hk", query, k_) * layer.scaling
                    probs = torch.softmax(scores, dim=-1)
                    out.append(torch.einsum("hk,khd->hd", probs, v_).reshape(-1))
                start += m
            outputs.append(torch.stack(out).to(_DTYPE))
        return outputs


class SeveralWindowsCase(CustomTestCase):
    """Shared checks; every case runs a real Triton backend on CUDA."""

    def setUp(self):
        override = get_parallel().override(attn_tp_size=1, attn_dcp_rank=0)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)

    def _check(self, actual, expected, tag):
        for layer_id, (a, e) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(
                a,
                e,
                atol=_ATOL,
                rtol=_RTOL,
                msg=lambda m: f"{tag} layer {layer_id}: {m}",
            )

    def _prefill(self, model, rows, prefix_lens, extend_lens, generator):
        """Writes each request's prefix, then extends past it, checking both."""
        zeros = [0] * len(rows)
        steps = [(prefix_lens, extend_lens)]
        if any(prefix_lens):
            steps.insert(0, (zeros, prefix_lens))
        for prefix, inputs_lens in steps:
            mode = ForwardMode.EXTEND
            batch = model.batch(mode, rows, prefix, inputs_lens)
            inputs = model.inputs(rows, prefix, inputs_lens, generator)
            model.backend.init_forward_metadata(batch)
            actual = model.forward(batch, inputs)
            expected = model.reference(mode, rows, prefix, inputs_lens, inputs)
            self._check(actual, expected, f"extend prefix={prefix}")

    def _extend_and_decode(self, model):
        generator = torch.Generator(device="cuda").manual_seed(7)
        rows = [1, 2, 3]
        prefix_lens, extend_lens = [5, 19, 2], [3, 4, 9]
        for row, p, n in zip(rows, prefix_lens, extend_lens):
            model.add_request(row, p + n + 1, generator)
        self._prefill(model, rows, prefix_lens, extend_lens, generator)
        seq = [p + n for p, n in zip(prefix_lens, extend_lens)]
        ones = [1] * len(rows)
        batch = model.batch(ForwardMode.DECODE, rows, seq, ones)
        inputs = model.inputs(rows, seq, ones, generator)
        model.backend.init_forward_metadata(batch)
        actual = model.forward(batch, inputs)
        expected = model.reference(ForwardMode.DECODE, rows, seq, ones, inputs)
        self._check(actual, expected, "decode")

    def _graph_decode(self, model):
        """Capture a decode graph and replay it on the same rows and token
        count, with other sequence lengths and write ids; check each layer
        against attention from positions."""
        generator = torch.Generator(device="cuda").manual_seed(11)
        rows = [1, 2, 3]
        capture_seq, replay_seq = [2, 3, 4], [13, 6, 21]
        for row, n in zip(rows, replay_seq):
            model.add_request(row, n + 1, generator)
        self._prefill(model, rows, [0] * 3, replay_seq, generator)
        ones = [1] * len(rows)
        backend = model.backend
        backend.init_cuda_graph_state(max_bs=len(rows), max_num_tokens=len(rows))
        capture = model.batch(ForwardMode.DECODE, rows, capture_seq, ones)
        static = model.inputs(rows, capture_seq, ones, generator)
        backend.init_forward_metadata_out_graph(capture, in_capture=True)
        backend.init_forward_metadata_in_graph(capture)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            model.forward(capture, static)  # warm-up
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = model.forward(capture, static)

        replay = model.batch(ForwardMode.DECODE, rows, replay_seq, ones)
        inputs = model.inputs(rows, replay_seq, ones, generator)
        capture.out_cache_loc.copy_(replay.out_cache_loc)
        for (sq, sk, sv), (q, k, v) in zip(static, inputs):
            sq.copy_(q)
            sk.copy_(k)
            sv.copy_(v)
        backend.init_forward_metadata_out_graph(replay)
        graph.replay()
        torch.cuda.synchronize()
        expected = model.reference(ForwardMode.DECODE, rows, replay_seq, ones, inputs)
        self._check([o.clone() for o in outputs], expected, "graph decode")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTritonSeveralWindows(SeveralWindowsCase):
    def test_extend_and_decode_read_each_layers_window(self):
        """Windows of different lengths, eager extend (with and without a
        prefix longer than the windows) and decode."""
        for windows in ((8, 4), (16, 8, 3)):
            with self.subTest(windows=windows):
                model = _Model(windows)
                model.bind()
                self.assertGreater(model.backend.max_window_sub_pools, len(windows) - 1)
                self._extend_and_decode(model)

    def test_a_draft_window_band_is_the_drafts_alone(self):
        """A target without windows whose fused draft keeps a window band of
        its own: the target's backend builds no window rail over the draft's
        band, and the draft's builds its own."""
        setup_allocator_context()
        draft_window = 8
        geometry = DraftKVGeometry(
            head_num=_KV_HEADS, head_dim=_HEAD_DIM, v_head_dim=_HEAD_DIM
        )
        placement = place_fused_draft(
            profile=DraftKVProfile(
                num_layers=1,
                num_depths=1,
                kinds={
                    LAYER_WINDOW: DraftLayerSet(
                        layer_ids=(0,), geometry=geometry, window=draft_window
                    )
                },
            ),
            num_runners=1,
            ctx=PlacementContext(host_names=("full",), target_windows=()),
            store_dtype=_DTYPE,
        ).placement
        band = f"window_{draft_window}"
        self.assertIn(band, placement.new_bands)
        storage = _storage((), fused_draft=placement)
        target = _runner(storage, storage.allocator.get_kvcache(), model_window=None)
        self.assertEqual(target.runner_ids.window_sub_pools, ())
        target_backend = TritonAttnBackend(target)
        self.assertEqual(dict(target_backend.window_lengths), {})
        self.assertFalse(target_backend.use_sliding_window_kv_pool)
        target.runner_ids.bind_and_verify_backends([target_backend])

        layer = RadixAttention(
            num_heads=_HEADS,
            head_dim=_HEAD_DIM,
            scaling=_HEAD_DIM**-0.5,
            num_kv_heads=_KV_HEADS,
            layer_id=0,
            sliding_window_size=draft_window,
        )
        bound = bind_fused_draft(
            placement=placement,
            runner=0,
            arena=storage.allocator.arena,
            plan=storage.plan,
            owner=storage.plan.owner().name,
            target_state=None,
            model=SimpleNamespace(modules=lambda: [layer]),
            model_config=SimpleNamespace(),
            req_to_token_pool=storage.requests,
            page_size=1,
        )
        draft = _runner(
            storage,
            bound.token_to_kv_pool,
            model_window=draft_window,
            is_draft_worker=True,
        )
        draft_backend = TritonAttnBackend(draft)
        self.assertEqual(dict(draft_backend.window_lengths), {band: draft_window})
        self.assertTrue(draft_backend.use_sliding_window_kv_pool)

    def test_lean_scratch_takes_the_padded_widest_width(self):
        """REGRESSION: the Lean scratch is as wide as the widest value width padded
        to a power of two (96 -> 128); the logits scratch keeps each width exactly."""
        model = _Model((8, 4), full_dim=64, window_dim=(80, 96), model_window=None)
        backend = model.backend
        self.assertEqual(backend.extra_v_head_dims, (80, 96))
        self.assertEqual(backend.lean_v_head_dim, 128)
        backend.init_cuda_graph_state(max_bs=2, max_num_tokens=2)
        self.assertEqual(backend.cuda_graph_lean_Op.shape[-1], 128)
        self.assertEqual(
            {w: t.shape[-1] for w, t in backend.cuda_graph_extra_attn_logits.items()},
            {80: 80, 96: 96},
        )
        self.assertEqual(backend.cuda_graph_attn_logits.shape[-1], 64)

    def test_deterministic_extend_reads_the_window_ids(self):
        """Deterministic inference extends through one kernel that reads the
        new tokens' K/V back from the cache: a window layer reads them at its
        window sub-pool's ids, not the request's token ids."""
        model = _Model((8,), deterministic=True)
        self.addCleanup(model.config.restore)
        self.assertTrue(model.backend.enable_deterministic)
        model.bind()
        generator = torch.Generator(device="cuda").manual_seed(5)
        rows = [1, 2, 3]
        prefix_lens, extend_lens = [5, 19, 2], [3, 4, 9]
        for row, p, n in zip(rows, prefix_lens, extend_lens):
            model.add_request(row, p + n + 1, generator)
        self._prefill(model, rows, prefix_lens, extend_lens, generator)

    def test_deterministic_extend_refuses_several_windows(self):
        """The deterministic extend reads one window rail: binding a runner
        over several window sub-pools names the backend."""
        model = _Model((8, 4), deterministic=True)
        self.addCleanup(model.config.restore)
        with self.assertRaisesRegex(NotImplementedError, "TritonAttnBackend"):
            model.bind()

    def test_unconverted_modes_refuse_several_windows(self):
        """Speculative modes still build one window rail: binding a runner over
        several window sub-pools names the backend."""
        model = _Model((8, 4), spec_algorithm=SpeculativeAlgorithm.EAGLE)
        self.assertEqual(model.backend.max_window_sub_pools, 1)
        with self.assertRaisesRegex(NotImplementedError, "TritonAttnBackend"):
            model.bind()
        single = _Model((8,), spec_algorithm=SpeculativeAlgorithm.EAGLE)
        single.bind()
        self.assertIs(single.backend.runner_ids, single.runner.runner_ids)


if __name__ == "__main__":
    unittest.main()
