"""Real Store + CUDA Graph integration; start a dedicated test master first.

MOONCAKE_MASTER defaults to 127.0.0.1:50051, metadata to port 50052.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from mooncake.store import EngramStore, EngramStoreConfig, MooncakeDistributedStore

from sglang.srt.layers.engram import EngramHasher, EngramLayout
from sglang.srt.layers.engram_mooncake import MooncakeEngramEmbedding, connect_store
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.models.deepseek_v4 import _prefetch_mooncake_engram


@pytest.fixture(params=["store", "local"])
def setup(tmp_path, monkeypatch, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    layout = EngramLayout.build((1, 14), (0, 0), 4, 8, 256, 17)
    connection = dict(
        local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
        metadata_server=os.getenv(
            "MOONCAKE_METADATA", "http://127.0.0.1:50052/metadata"
        ),
        global_segment_size=0,
        local_buffer_size=1024**2,
        protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
        rdma_devices=os.getenv("MOONCAKE_RDMA_DEVICES", ""),
        master_server_addr=os.getenv("MOONCAKE_MASTER", "127.0.0.1:50051"),
    )
    owner = MooncakeDistributedStore()
    assert owner.setup(**dict(connection, global_segment_size=128 * 1024**2)) == 0
    if owner.is_exist("engram:ready"):
        owner.close()
        pytest.fail("Use a dedicated test Store without published Engram tables")
    manifest = dict(mode=request.param, connection=connection, layers={})
    if request.param == "local":
        manifest["local_tables"] = {}
    configs = {}
    for i, layer in enumerate(layout.layer_ids):
        cfg = EngramStoreConfig()
        cfg.table_vocab_sizes = [p for group in layout.primes[i] for p in group]
        cfg.row_bytes = 264
        configs[layer] = cfg
    table = EngramStore(configs, owner)
    reference = []
    for layer, cfg in configs.items():
        arrays = []
        for h, n in enumerate(cfg.table_vocab_sizes):
            raw = np.empty((n, 264), dtype=np.uint8)
            raw[:, :256] = (np.arange(n, dtype=np.uint32)[:, None] + h) % 112
            raw[:, 256:] = np.arange(8) + 123
            raw[:, :32] = 56  # FP8 1.0
            raw[:, 256] = 0  # E8M0 exponent zero is 2**-127, not zero.
            arrays.append(raw)
        if request.param == "store":
            table.populate(layer, arrays)
        else:
            paths = []
            for h, raw in enumerate(arrays):
                path = tmp_path / f"{layer}-{h}.bin"
                raw.tofile(path)
                paths.append(str(path))
            manifest["local_tables"][str(layer)] = paths
        reference.append(torch.from_numpy(np.concatenate(arrays)).cuda())
        manifest["layers"][str(layer)] = dict(
            table_vocab_sizes=cfg.table_vocab_sizes, head_dim=256, row_bytes=264
        )
    assert owner.put("engram:ready", json.dumps(manifest["layers"]).encode()) == 0
    path = tmp_path / "mooncake.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_MOONCAKE_CONFIG", str(path))
    args = SimpleNamespace(
        enable_dp_attention=True,
        pp_size=1,
        attn_cp_size=1,
        speculative_algorithm=None,
        disable_overlap_schedule=True,
    )
    graph_config = SimpleNamespace(
        decode=SimpleNamespace(backend="breakable"),
        prefill=SimpleNamespace(backend="breakable"),
    )
    with (
        patch("sglang.srt.runtime_context.get_server_args", return_value=args),
        patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(cuda_graph_config=graph_config)
            ),
        ),
    ):
        embeds = [MooncakeEngramEmbedding(layout, layer) for layer in layout.layer_ids]
        assert embeds[0].table is embeds[1].table
        assert embeds[0].table.get_layer_ids() == list(layout.layer_ids)
    with patch(
        "sglang.srt.layers.engram.build_compressed_token_map",
        return_value=(list(range(64)), 64),
    ):
        hasher = EngramHasher(layout, None, 2, 64).cuda()
        reference_hasher = EngramHasher(layout, None, 2, 64).cuda()
    hasher.init_history(8, "cuda")
    reference_hasher.init_history(8, "cuda")
    model = SimpleNamespace(
        engram_layout=layout,
        engram_hasher=hasher,
        layers={
            layer: SimpleNamespace(engram=SimpleNamespace(embed=embed))
            for layer, embed in zip(layout.layer_ids, embeds)
        },
    )
    yield model, reference_hasher, reference
    torch.cuda.synchronize()
    for embed in embeds:
        embed._release()
    if embeds[0].store is not None:
        embeds[0].store.close()
    connect_store.cache_clear()
    for layer in layout.layer_ids:
        table.remove_from_store(layer, force=True)
    owner.remove("engram:ready", True)
    owner.close()


def test_changing_tokens_padding_and_bucket_replay(setup):
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    model, reference_hasher, reference = setup
    captured = {}
    for batch_size in (1, 4):
        inputs = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(batch_size, device="cuda", dtype=torch.int64),
            positions=torch.zeros(batch_size, device="cuda", dtype=torch.int64),
            out_cache_loc=torch.ones(batch_size, device="cuda", dtype=torch.int64),
        )
        graph = BreakableCUDAGraph()
        before = model.engram_hasher.history.clone()
        torch.cuda.synchronize()
        with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
            ids = _prefetch_mooncake_engram(model, inputs, fb)
            outputs = [
                model.layers[layer].engram.embed(ids[:, i])
                for i, layer in enumerate((1, 14))
            ]
        torch.testing.assert_close(model.engram_hasher.history, before)
        assert len(graph._segments) >= 4 and len(graph._break_fns) == 3
        captured[batch_size] = graph, inputs, fb, outputs
    for step, batch_size in enumerate((1, 4, 4, 1, 4, 1)):
        graph, inputs, fb, outputs = captured[batch_size]
        inputs.copy_((torch.arange(batch_size, device="cuda") + step + 3) % 64)
        fb.positions.fill_(step + 3)
        fb.out_cache_loc.fill_(1)
        if batch_size == 4:
            fb.out_cache_loc[-1] = 0
        expected_ids = reference_hasher(inputs, fb)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            model.engram_hasher.history, reference_hasher.history
        )
        for i, out in enumerate(outputs):
            # Compare against the original host/device gather kernel, including
            # E8M0 exponent zero, rather than repeating the adapter arithmetic.
            weight = reference[i][..., :256].contiguous()
            scale = reference[i][..., 256:].contiguous()
            expected = torch.empty_like(out)
            engram_gather(
                weight.data_ptr(),
                scale.data_ptr(),
                expected_ids[:, i].reshape(-1),
                expected.view(-1, 256),
                256,
                32,
            )
            expected[torch.all(expected_ids[:, i] == 0, dim=-1)] = 0
            expected[fb.out_cache_loc == 0] = 0
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert all(model.layers[layer].engram.embed.lookup_count == 6 for layer in (1, 14))

    # Replay the same captured decode graph while the rank becomes idle and
    # active again; its capture-time ForwardMode remains DECODE throughout.
    graph, inputs, fb, outputs = captured[4]
    before = model.engram_hasher.history.clone()
    fb.out_cache_loc.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(model.engram_hasher.history, before)
    assert all(torch.count_nonzero(output) == 0 for output in outputs)
    assert all(model.layers[layer].engram.embed.lookup_count == 6 for layer in (1, 14))
    fb.out_cache_loc.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert all(model.layers[layer].engram.embed.lookup_count == 7 for layer in (1, 14))


def test_prefill_replay_uses_live_batch(setup):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        enable_breakable_cuda_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        set_tc_piecewise_forward_context,
    )

    model, reference_hasher, reference = setup
    inputs = torch.tensor([3, 4, 5, 0], device="cuda", dtype=torch.int64)

    def batch(lengths, slots):
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            ngram_history=None,
            out_cache_loc=torch.tensor(
                [1] * sum(lengths), device="cuda", dtype=torch.int64
            ),
            req_pool_indices=torch.tensor(slots, device="cuda", dtype=torch.int64),
            positions=torch.arange(4, device="cuda", dtype=torch.int64),
            extend_seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            extend_start_loc=torch.tensor(
                [0] + list(np.cumsum(lengths)[:-1]), device="cuda", dtype=torch.int32
            ),
        )

    capture_batch = batch([4], [0])
    graph = BreakableCUDAGraph()
    torch.cuda.synchronize()
    with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
        ids = _prefetch_mooncake_engram(model, inputs, capture_batch)
        outputs = [
            model.layers[layer].engram.embed(ids[:, i])
            for i, layer in enumerate((1, 14))
        ]
    for live in (batch([2, 1], [1, 2]), batch([1, 3], [3, 1])):
        expected_ids = reference_hasher(inputs, live)
        with (
            enable_breakable_cuda_graph(),
            set_tc_piecewise_forward_context(live, [], None, [], []),
        ):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            model.engram_hasher.history, reference_hasher.history
        )
        for i, out in enumerate(outputs):
            raw = reference[i][expected_ids[:, i]]
            weight = raw[..., :256].contiguous().view(torch.float8_e4m3fn).float()
            scale = raw[..., 256:].contiguous().view(torch.float8_e8m0fnu).float()
            expected = (
                (weight.unflatten(-1, (8, 32)) * scale.unsqueeze(-1))
                .flatten(-2)
                .to(torch.bfloat16)
            )
            expected[torch.all(expected_ids[:, i] == 0, dim=-1)] = 0
            expected[sum(live.extend_seq_lens.tolist()) :] = 0
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_missing_table_stops_before_h2d(setup):
    model, _, _ = setup
    if model.layers[1].engram.embed.store is None:
        pytest.skip("Local immutable tables have no Store removal")
    inputs = torch.tensor([3], device="cuda", dtype=torch.int64)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int64),
        positions=torch.tensor([3], device="cuda", dtype=torch.int64),
        out_cache_loc=torch.tensor([1], device="cuda", dtype=torch.int64),
    )
    graph = BreakableCUDAGraph()
    torch.cuda.synchronize()
    with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
        ids = _prefetch_mooncake_engram(model, inputs, fb)
        output = model.layers[14].engram.embed(ids[:, 1])
    graph.replay()
    torch.cuda.synchronize()
    before = output.clone()
    embed = model.layers[14].engram.embed
    assert embed.store.remove(embed.table.get_store_keys(embed.layer_id)[0], True) == 0
    inputs.fill_(7)
    with pytest.raises(RuntimeError, match="lookup_into failed"):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, before)


def test_late_layer_read_overlaps_earlier_graph(setup, monkeypatch):
    from threading import Event

    from sglang.srt.layers import engram_mooncake
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )

    model, _, _ = setup
    inputs = torch.tensor([3], device="cuda", dtype=torch.int64)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int64),
        positions=torch.tensor([3], device="cuda", dtype=torch.int64),
        out_cache_loc=torch.tensor([1], device="cuda", dtype=torch.int64),
    )
    started, release = Event(), Event()
    lookup = engram_mooncake._lookup_rows

    def delayed_lookup(table, layer_id, *args):
        if layer_id == 14:
            started.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Layer 14 read blocked earlier GPU computation")
        return lookup(table, layer_id, *args)

    monkeypatch.setattr(engram_mooncake, "_lookup_rows", delayed_lookup)
    reached = []

    @eager_on_graph(True, capture_stub=lambda output: None)
    def after_first_layer(output):
        assert started.wait(timeout=10)
        assert not model.layers[14].engram.embed.pending.done()
        # This graph segment really completed while the later read was blocked.
        torch.cuda.synchronize()
        assert torch.isfinite(output).all() and torch.count_nonzero(output) > 0
        reached.append(True)
        release.set()

    graph = BreakableCUDAGraph()
    torch.cuda.synchronize()
    with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
        ids = _prefetch_mooncake_engram(model, inputs, fb)
        first = model.layers[1].engram.embed(ids[:, 0])
        after_first_layer(first)
        last = model.layers[14].engram.embed(ids[:, 1])
    assert not started.is_set()  # Capture must not start background reads.
    try:
        graph.replay()
        torch.cuda.synchronize()
    finally:
        release.set()
    assert reached == [True]
    assert torch.isfinite(last).all()
    assert all(model.layers[layer].engram.embed.pending is None for layer in (1, 14))


def test_async_failure_can_retry_next_step(setup, monkeypatch):
    from sglang.srt.layers import engram_mooncake

    model, _, _ = setup
    inputs = torch.tensor([3], device="cuda", dtype=torch.int64)
    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([0], device="cuda", dtype=torch.int64),
        positions=torch.tensor([3], device="cuda", dtype=torch.int64),
        out_cache_loc=torch.tensor([1], device="cuda", dtype=torch.int64),
    )
    graph = BreakableCUDAGraph()
    torch.cuda.synchronize()
    with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
        ids = _prefetch_mooncake_engram(model, inputs, fb)
        outputs = [
            model.layers[layer].engram.embed(ids[:, i])
            for i, layer in enumerate((1, 14))
        ]
    lookup = engram_mooncake._lookup_rows

    def fail_first_layer(table, layer_id, *args):
        if layer_id == 1:
            raise RuntimeError("Injected background read failure")
        return lookup(table, layer_id, *args)

    monkeypatch.setattr(engram_mooncake, "_lookup_rows", fail_first_layer)
    with pytest.raises(RuntimeError, match="Injected background read failure"):
        graph.replay()
    # Layer 14 may still be running after layer 1 failed. The next prefetch
    # must drain that task before reusing its registered output buffer.
    monkeypatch.setattr(engram_mooncake, "_lookup_rows", lookup)
    inputs.fill_(7)
    fb.positions.add_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.isfinite(output).all() for output in outputs)
    assert all(model.layers[layer].engram.embed.pending is None for layer in (1, 14))


def test_lookup_graph_benchmark(setup):
    """Optional wall-time benchmark of hash + real Store fetch + captured H2D/dequant."""
    import time
    from pathlib import Path

    output_path = os.getenv("ENGRAM_BENCH_OUTPUT")
    if not output_path:
        pytest.skip("Set ENGRAM_BENCH_OUTPUT to record component performance")
    model, _, _ = setup
    results = []
    for size in (1, 4, 8):
        inputs = torch.arange(size, device="cuda", dtype=torch.int64) + 3
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(size, device="cuda", dtype=torch.int64),
            positions=torch.full((size,), 10, device="cuda", dtype=torch.int64),
            out_cache_loc=torch.ones(size, device="cuda", dtype=torch.int64),
        )
        graph = BreakableCUDAGraph()
        torch.cuda.synchronize()
        with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
            ids = _prefetch_mooncake_engram(model, inputs, fb)
            outputs = [
                model.layers[layer].engram.embed(ids[:, i])
                for i, layer in enumerate((1, 14))
            ]
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(30):
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        assert all(torch.isfinite(out).all() for out in outputs)
        results.append(
            dict(
                batch_size=size,
                iterations=len(samples),
                median_step_ms=float(np.median(samples)),
                p95_step_ms=float(np.percentile(samples, 95)),
                cuda_graph_segments=len(graph._segments),
                eager_breaks=len(graph._break_fns),
            )
        )
    Path(output_path).write_text(
        json.dumps(
            dict(
                scope="small-table Engram component, not full-model performance",
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


def test_loader_skips_table_before_materializing(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from sglang.srt.model_loader import weight_utils

    table_name = "layers.1.engram.embed.weight"
    gate_name = "layers.1.engram.q_weight"
    checkpoint = tmp_path / "mixed.safetensors"
    save_file(
        {table_name: torch.zeros(32, 256), gate_name: torch.ones(4, 16)}, checkpoint
    )
    original_open = weight_utils.safetensors.safe_open
    reads = []

    class TrackingOpen:
        def __init__(self, *args, **kwargs):
            self.file = original_open(*args, **kwargs)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def keys(self):
            return self.file.keys()

        def get_tensor(self, name):
            reads.append(name)
            return self.file.get_tensor(name)

    monkeypatch.setattr(weight_utils.safetensors, "safe_open", TrackingOpen)
    tensors = dict(
        weight_utils.safetensors_weights_iterator(
            [str(checkpoint)], skip_tensor_names={table_name}
        )
    )
    assert reads == [gate_name]
    torch.testing.assert_close(tensors[gate_name], torch.ones(4, 16))
    with pytest.raises(ValueError, match="requires mmap"):
        list(
            weight_utils.safetensors_weights_iterator(
                [str(checkpoint)], disable_mmap=True, skip_tensor_names={table_name}
            )
        )


def test_low_ratio_sources_live_cpu_count():
    from unittest.mock import Mock

    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _low_ratio_source_projections,
    )
    from sglang.srt.models.deepseek_v4 import deepseek_v4_low_ratio_sources

    # The old num_token_non_padded_cpu field no longer exists on ForwardBatch.
    fb = SimpleNamespace(global_num_token_non_padded_cpu=2)
    context = SimpleNamespace(forward_batch=fb)
    backend = Mock()
    x = torch.arange(12).reshape(4, 3).float()
    positions = torch.arange(4)
    layer = SimpleNamespace(
        compressor=SimpleNamespace(project=lambda value: (value + 1, None)),
        indexer=None,
    )
    buffers = {"kv": torch.full_like(x, -1)}
    with (
        patch(
            "sglang.srt.models.deepseek_v4.get_tc_piecewise_forward_context",
            return_value=context,
        ),
        patch("sglang.srt.models.deepseek_v4.get_attn_backend", return_value=backend),
        patch(
            "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.get_tc_piecewise_forward_context",
            return_value=context,
        ),
    ):
        deepseek_v4_low_ratio_sources(layer, x, x, positions)
        torch.testing.assert_close(
            backend.forward_low_ratio_sources.call_args.kwargs["x"], x[:2]
        )
        _low_ratio_source_projections(layer, x, x, positions, buffers)
        torch.testing.assert_close(buffers["kv"][:2], x[:2] + 1)
        assert torch.count_nonzero(buffers["kv"][2:]) == 0
        fb.global_num_token_non_padded_cpu = 0
        deepseek_v4_low_ratio_sources(layer, x, x, positions)
        _low_ratio_source_projections(layer, x, x, positions, buffers)
        assert backend.forward_low_ratio_sources.call_count == 1
        assert torch.count_nonzero(buffers["kv"]) == 0


def test_breakable_decode_logits_output():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.runner.shape_key import ShapeKey
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner = SimpleNamespace(
        device_module=torch.cuda,
        model_runner=SimpleNamespace(tp_group=SimpleNamespace(barrier=lambda: None)),
    )
    backend = BreakableCudaGraphBackend(runner)
    backend._pool = torch.cuda.graph_pool_handle()
    backend._capture_stream = torch.cuda.Stream()
    inputs = {}
    for size in (4, 1):
        key = ShapeKey(size=size)
        x = torch.ones(size, 16, device="cuda")
        inputs[key] = x

        def forward(x=x):
            return LogitsProcessorOutput(next_token_logits=x * 3, hidden_states=x + 2)

        with backend.replay_session():
            backend.capture_one(key, forward, capture_inputs=x)
    for step in (2, 7):
        for key, x in inputs.items():
            x.fill_(step)
            output = backend.replay(key, None)
            torch.testing.assert_close(output.next_token_logits, x * 3)
            torch.testing.assert_close(output.hidden_states, x + 2)
    torch.cuda.synchronize()


@pytest.mark.parametrize("padded", [False, True])
def test_dpa_idle_does_not_read_or_commit(setup, padded):
    model, _, _ = setup
    inputs = torch.zeros(4 if padded else 0, dtype=torch.int64, device="cuda")
    fb = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if padded else ForwardMode.IDLE,
        _original_forward_mode=ForwardMode.IDLE,
    )
    before = model.engram_hasher.history.clone()
    from contextlib import nullcontext

    graph = BreakableCUDAGraph()
    # Graph runners pad idle ranks; a truly empty batch uses the eager path.
    context = (
        BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream())
        if padded
        else nullcontext()
    )
    with context:
        ids = _prefetch_mooncake_engram(model, inputs, fb)
        outputs = [
            model.layers[layer].engram.embed(ids[:, i])
            for i, layer in enumerate((1, 14))
        ]
    if padded:
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(model.engram_hasher.history, before)
    for output in outputs:
        assert not torch.count_nonzero(output)
    assert all(model.layers[layer].engram.embed.lookup_count == 0 for layer in (1, 14))
