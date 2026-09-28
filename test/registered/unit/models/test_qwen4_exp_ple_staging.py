import gc
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from sglang.srt.arg_groups import memory_hook
from sglang.srt.arg_groups.fields.exec_ import ExecDllm, ExecGraph, ExecOverlap
from sglang.srt.models import qwen4_exp as qwen4
from sglang.srt.models.qwen4_exp_ple_rows import PageCacheRowSource
from sglang.srt.models.qwen4_exp_ple_staging import PleHostStaging
from sglang.srt.models.qwen4_exp_ple_table import _PAGE_SHIFT
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("row_bytes", [160, 320, 9000])
def test_rows_and_page_hints(tmp_path, row_bytes):
    table = np.random.default_rng(3).integers(0, 256, (100, row_bytes), dtype=np.uint8)
    path = tmp_path / "shard.bin"
    table.tofile(path)
    writer = np.memmap(path, mode="r+", dtype=np.uint8, shape=table.shape)
    source = PageCacheRowSource(str(path), row_bytes, writer)
    writer[1] = table[1] = 123
    ids = np.array([99, 1, 1, 0, 7], dtype=np.int64)
    out = np.empty((len(ids), row_bytes), dtype=np.uint8)
    with mock.patch("os.posix_fadvise") as advise:
        source.prefetch_rows(ids)
        source.fetch_rows(ids, out)
    np.testing.assert_array_equal(out, table[ids])
    page_size = 1 << _PAGE_SHIFT
    pages = set()
    for row in ids:
        start = int(row * row_bytes) // page_size
        end = int((row + 1) * row_bytes - 1) // page_size
        pages.update(range(start, end + 1))
    advised_pages = set()
    for call in advise.call_args_list:
        offset, length = call.args[1:3]
        advised_pages.update(
            range(offset // page_size, (offset + length - 1) // page_size + 1)
        )
    assert advised_pages == pages
    with pytest.raises(IndexError):
        source.fetch_rows(np.array([100]), out[:1])
    with pytest.raises(IndexError):
        source.fetch_rows(np.array([-1]), out[:1])
    source.fetch_rows(np.empty(0, dtype=np.int64), out[:0])
    source.close()
    source.close()
    writer._mmap.close()


def test_staging_shard_mask_and_buffer_lifetime(tmp_path):
    table = np.arange(24, dtype=np.uint8).reshape(8, 3)
    path = tmp_path / "shard.bin"
    table.tofile(path)
    mapping = np.memmap(path, mode="r", dtype=np.uint8, shape=table.shape)
    events = mock.Mock(side_effect=lambda **kw: mock.Mock())
    staging = PleHostStaging(
        PageCacheRowSource(str(path), 3, mapping),
        10,
        18,
        device_module=SimpleNamespace(
            Event=events, is_current_stream_capturing=lambda: False
        ),
        chunk_rows=2,
        pin_memory=False,
    )
    with pytest.raises(RuntimeError, match="no captured buffer"):
        staging.buffer(4, "cpu", graph=True)
    ids = np.array([10, 17, 9, 18, 12, 11, 13])
    with mock.patch(
        "sglang.srt.models.qwen4_exp_ple_staging.get_is_capture_mode", return_value=True
    ):
        with mock.patch.object(
            staging.device_module, "is_current_stream_capturing", return_value=True
        ):
            with pytest.raises(RuntimeError, match="allocation during capture"):
                staging.buffer(4, "cpu", graph=True)
        staging.begin(ids, "cpu", graph=True)
        with pytest.raises(RuntimeError):
            staging.begin(ids, "cpu")
        assert staging.pending_rows == len(ids)
        staging.discard()
        staging.begin(ids, "cpu", graph=True)
        result = staging.finish(lambda raw, start, end: None).clone()
        fixed = staging.buffer(len(ids), "cpu", graph=True)
        ptr = fixed.data_ptr()
        staging.begin(ids[::-1], "cpu", graph=True)
        staging.finish()
        assert staging.buffer(len(ids), "cpu", graph=True).data_ptr() == ptr
        with pytest.raises(RuntimeError):
            staging.finish()
        assert events.call_count == 2
        staging.close()
    expected = np.zeros((len(ids), 3), dtype=np.uint8)
    valid = (ids >= 10) & (ids < 18)
    expected[valid] = table[ids[valid] - 10]
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(fixed.numpy(), expected[::-1])
    mapping._mmap.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_page_hint_control_and_failure(tmp_path, caplog, monkeypatch, enabled):
    monkeypatch.setenv("SGLANG_QWEN4_PLE_FILE_PREFETCH", str(int(enabled)))
    path = tmp_path / "rows"
    path.write_bytes(bytes(32))
    mapping = np.memmap(path, mode="r", dtype=np.uint8, shape=(8, 4))
    source = PageCacheRowSource(str(path), 4, mapping)
    with mock.patch("os.posix_fadvise", side_effect=OSError("unsupported")) as advise:
        source.prefetch_rows(np.array([0]))
        source.prefetch_rows(np.array([1]))
    assert advise.call_count == int(enabled)
    assert caplog.text.count("page hints off") == int(enabled)
    source.close()
    mapping._mmap.close()


@pytest.mark.parametrize("staged", [False, True])
def test_replay_stages_padded_inputs_after_load(staged):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
    from sglang.srt.model_executor.runner import decode_cuda_graph_runner as module

    events = []
    runner = module.DecodeCudaGraphRunner.__new__(module.DecodeCudaGraphRunner)
    runner.bs = 4
    runner.captured_req_width = 1
    runner.capture_forward_mode = ForwardMode.DECODE
    runner._replay_graph_key = "graph"
    runner.buffers = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        req_pool_indices=torch.arange(4),
        out_cache_loc=torch.ones(4),
    )
    runner.buffer_registry = SimpleNamespace(
        get_slot=lambda name: SimpleNamespace(
            slice_for=lambda bs, nt: getattr(runner.buffers, name)[
                : bs if name == "req_pool_indices" else nt
            ]
        )
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=torch.tensor([11, 12]),
        spec_info=SimpleNamespace(draft_token_num=99),
    )

    def load(*_):
        events.append("load")
        runner.buffers.input_ids[:2].copy_(batch.input_ids)

    marker = object()

    def build(input_ids, view, *, ngram_size, ngram_eos_token_id):
        assert (ngram_size, ngram_eos_token_id) == (3, 0)
        assert view.batch_size == 4 and batch.batch_size == 2
        assert view.spec_info.draft_token_num == 1
        assert batch.spec_info.draft_token_num == 99
        assert view.input_ids.tolist() == [11, 12, 0, 0]
        assert len(view.req_pool_indices) == 4
        assert view.global_num_token_non_padded_cpu is None
        return marker

    def prepare(view, *, ple_batch, graph):
        assert graph and ple_batch is marker
        events.append("prepare")

    def replay(*_):
        events.append("replay")
        return PPProxyTensors({"hidden_states": torch.zeros(4, 2)})

    model = SimpleNamespace(
        model=SimpleNamespace(
            prepare_ple_rows=prepare, ple_ngram_size=3, ple_ngram_eos_token_id=0
        )
    )
    model.prepare_cuda_graph_replay = (
        qwen4.Qwen4ExpForConditionalGeneration.prepare_cuda_graph_replay.__get__(model)
    )
    runner.model_runner = SimpleNamespace(model=model, device_timer=None)
    model.needs_replay_prepare = staged
    runner.validate_model_support = lambda: None
    runner._init_model_hooks()
    assert (runner._replay_prepare_hook is not None) == staged
    runner.backend = SimpleNamespace(replay_session=nullcontext, replay=replay)
    runner.load_batch = load
    runner._replay_attn_backend = lambda: None
    runner._resolve_shared_read_ends = lambda *_: module.SharedReadEnds.PRE_REPLAY
    runner._publish_read_done = lambda **_: events.append("publish")
    with (
        mock.patch.object(module, "device_timer_ctx", return_value=nullcontext()),
        mock.patch.object(qwen4, "_prepare_ple_batch", build),
    ):
        runner.execute(batch)
    assert events == ["load"] + (["prepare"] if staged else []) + ["publish", "replay"]
    assert batch.input_ids.tolist() == [11, 12]


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize(
    "flag",
    [
        "enable-dp-attention",
        "enable-pdmux",
        "enable-two-batch-overlap",
        "dllm-algorithm",
        "SGLANG_RAGGED_VERIFY_MODE",
        "disable-prefill-cuda-graph",
        "cuda-graph-backend-decode",
        "enable-torch-compile",
        None,
    ],
)
def test_staging_rejects_unsupported_runners(staged, flag):
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    cls = DecodeCudaGraphRunner
    if flag == "disable-prefill-cuda-graph":
        cls = PrefillCudaGraphRunner
    runner = cls.__new__(cls)
    runner.ragged_verify_mode = flag == "SGLANG_RAGGED_VERIFY_MODE"
    execution = mock.Mock(overlap=ExecOverlap(), dllm=ExecDllm(), graph=ExecGraph())
    execution.overlap.enable_two_batch_overlap = flag == "enable-two-batch-overlap"
    execution.dllm.dllm_algorithm = "test" if flag == "dllm-algorithm" else None
    execution.graph.enable_torch_compile = flag == "enable-torch-compile"
    if flag == "cuda-graph-backend-decode":
        execution.graph.cuda_graph_config = mock.Mock(**{"decode.backend": "breakable"})
    with mock.patch.multiple(
        qwen4,
        is_dp_attention_enabled=lambda: flag == "enable-dp-attention",
        get_disagg=lambda: SimpleNamespace(enable_pdmux=flag == "enable-pdmux"),
        get_exec=lambda: execution,
    ):
        with (
            pytest.raises(ValueError, match=flag) if staged and flag else nullcontext()
        ):
            qwen4.Qwen4ExpForConditionalGeneration.validate_runner_support(
                SimpleNamespace(needs_replay_prepare=staged), runner
            )


@pytest.mark.parametrize("direct", [True, False, None])
def test_file_embedding_selects_staging(tmp_path, direct):
    cls = qwen4.Qwen4ExpPinnedHostEmbedding
    source = SimpleNamespace(**dict.fromkeys(cls._COPIED_ATTRIBUTES, 0))
    source.embedding_dim = 4
    source.shard_indices = SimpleNamespace(
        org_vocab_start_index=0, org_vocab_end_index=8
    )
    source.weight = torch.nn.Parameter(torch.empty(8, 4, dtype=torch.bfloat16))
    source.weight_scale = None
    source.quant_method = qwen4.UnquantizedEmbeddingMethod()
    with (
        mock.patch.object(qwen4, "_PLE_LOGGED_DEVICES", set()),
        mock.patch("torch.cuda.current_device", return_value=2),
        mock.patch.object(qwen4, "device_uses_host_page_tables", return_value=direct),
        mock.patch.object(qwen4, "make_ple_file_rss_trimmer", return_value=None),
    ):
        if direct is None:
            with pytest.raises(RuntimeError, match="--ple-offload-backend pinned"):
                cls(source, backend="file", table_dir=str(tmp_path))
            return
        embedding = cls(source, backend="file", table_dir=str(tmp_path))
    assert (embedding.host_staging is not None) == (direct is not True)
    if embedding.host_staging is not None:
        with pytest.raises(RuntimeError, match="0 staged, 2 requested"):
            embedding.gather_staged(rows=2, device=torch.device("cpu"))
        embedding.host_staging.close()
    if embedding._file_prefetcher is not None:
        embedding._file_prefetcher.close()


def test_eager_validates_before_warmup(monkeypatch):
    from sglang.srt.model_executor.runner import base_runner as br
    from sglang.srt.model_executor.runner import eager_runner as er

    def validate(runner):
        assert runner._eager_max_bs == 2
        assert runner._eager_registry is registry
        calls.append(("validate", runner))

    calls, registry = [], object()
    mr = mock.Mock(
        device="cpu",
        is_draft_worker=False,
        max_running_requests=2,
        max_total_num_tokens=8,
    )
    mr.spec_algorithm.is_speculative.return_value = False
    mr.spec_algorithm.is_none.return_value = True
    mr.model_config.is_encoder_decoder = False
    mr.model.validate_runner_support = validate
    parallel = SimpleNamespace(
        tp_size=1, dp_size=1, pp_size=1, attn_tp_size=1, attn_tp_rank=0
    )
    for module in (br, er):
        monkeypatch.setattr(module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(br, "get_disagg", lambda: SimpleNamespace(enable_pdmux=False))
    monkeypatch.setattr(
        br, "get_server_return_hidden_states_mode", lambda: br.CaptureHiddenMode.NULL
    )
    monkeypatch.setattr(br, "TboCudaGraphRunnerPlugin", lambda: None)
    for name, fn in dict(
        get_eager_max_batch_size=lambda bs: bs,
        max_prefill_buffer_tokens=lambda: 8,
        require_mlp_sync=lambda: False,
        build_eager_registry=lambda **kw: registry,
    ).items():
        monkeypatch.setattr(er, name, fn)
    execution = mock.Mock()
    execution.mamba.enable_mamba_extra_buffer = False
    monkeypatch.setattr(er, "get_exec", lambda: execution)
    monkeypatch.setattr(er.DllmConfig, "from_server_args", lambda _: None)
    monkeypatch.setattr(
        er.EagerRunner, "warmup", lambda runner: calls.append(("warmup", runner))
    )
    runner = er.EagerRunner(mr)
    assert calls == [("validate", runner), ("warmup", runner)]


@pytest.mark.parametrize("consumer_fails", [False, True])
def test_fetch_failure_clears_pending_batch(consumer_fails):
    source = mock.Mock(row_bytes=1)

    def fetch(ids, out):
        np.copyto(out, ids[:, None], casting="unsafe")

    source.fetch_rows.side_effect = fetch
    staging = PleHostStaging(
        source,
        0,
        8,
        chunk_rows=2,
        pin_memory=False,
        device_module=SimpleNamespace(Event=lambda **kw: mock.Mock()),
    )
    staging.begin(np.arange(4), "cpu")
    staging.pending_ids
    if not consumer_fails:
        source.fetch_rows.side_effect = OSError("read failed")

    def consume(*args):
        if consumer_fails:
            raise OSError("consumer failed")

    with pytest.raises(OSError, match="failed"):
        staging.finish(consume)
    assert staging._pending is None
    source.fetch_rows.side_effect = fetch
    staging.begin(np.arange(4), "cpu")
    result = torch.empty(4, 1, dtype=torch.uint8)
    staging.finish(lambda raw, start, end: result[start:end].copy_(raw))
    assert result[:, 0].tolist() == list(range(4))
    staging.close()


def test_discard_logs_fetch_failure(caplog):
    from concurrent.futures import Future

    staging = PleHostStaging(SimpleNamespace(close=lambda: None), 0, 8, chunk_rows=2)
    failed = Future()
    failed.set_exception(OSError("read failed"))
    staging._pending = np.arange(2), None, failed, False
    staging.discard()
    assert staging.pending_rows == 0
    assert "discarded batch failed" in caplog.text
    staging.close()


def test_file_backend_argument_checks():
    cfg = SimpleNamespace(
        ple_offload_backend="file",
        ple_offload_embedding=True,
        cpu_offload_gb=0,
        offload_group_size=0,
        device=None,
    )
    with mock.patch.object(memory_hook, "resolving_view", return_value=cfg):
        memory_hook.handle_offload_compatibility(cfg)
        cfg.device = "cuda"
        memory_hook.handle_offload_compatibility(cfg)
        cfg.device = "cpu"
        with pytest.raises(ValueError, match="--ple-offload-backend pinned"):
            memory_hook.handle_offload_compatibility(cfg)


def test_eager_derivations_verified_once():
    from sglang.test.qwen4_ple_utils import make_embedding

    staging = PleHostStaging(mock.Mock(), 0, 8)
    embedding = make_embedding(device="cpu")
    embedding.ngram_embedding = SimpleNamespace(host_staging=staging)
    embedding.compute_ngram_ids = mock.Mock(return_value=torch.tensor([1]))
    batch = SimpleNamespace(mode=qwen4.ForwardMode.DECODE, use_decode_fast_path=True)
    with (
        mock.patch.object(qwen4, "get_is_capture_mode", return_value=False),
        mock.patch.object(qwen4, "_ple_context_window", return_value=torch.ones(1, 3)),
        mock.patch(
            "sglang.kernels.ops.qwen4_ple.can_fuse_qwen4_ngram_hash", return_value=True
        ),
    ):
        for fused in [False, True, False, True]:
            embedding.enable_ple_fusion = fused
            staging._pending = np.array([1]), None, mock.Mock(), False
            embedding.verify_ids_once(batch)
    assert embedding.compute_ngram_ids.call_count == 2
    staging.close()


def test_staging_finalizer_closes_resources():
    staging = PleHostStaging(mock.Mock(), 0, 8)
    source, pool, ref = staging.source, staging._pool, weakref.ref(staging)
    del staging
    gc.collect()
    assert ref() is None
    source.close.assert_called_once()
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)


def test_later_layer_staging_failure_recovers():
    stages = [PleHostStaging(mock.Mock(), 0, 8) for _ in range(2)]
    model = mock.Mock(ple_ngram_size=3, _ple_host_contexts=torch.zeros(1, 3))
    model._ple_staged_layers = []
    for staging in stages:
        embedding = mock.Mock(host_hash_metadata=(), ngram_heads=1)
        embedding.ngram_embedding.host_staging = staging
        model._ple_staged_layers.append(embedding)
        staging.begin, staging.finish = mock.Mock(), mock.Mock()
        staging._contexts[1] = torch.zeros(1, 3)
    stages[1].finish.side_effect = OSError("read failed")
    with (
        mock.patch.object(qwen4, "_ple_context_window", return_value=torch.ones(1, 3)),
        mock.patch.object(qwen4, "get_req_to_token_pool"),
    ):

        def prepare():
            qwen4.Qwen4ExpModel.prepare_ple_rows(
                model,
                SimpleNamespace(input_ids=torch.ones(1)),
                ple_batch=SimpleNamespace(processed_tokens=1),
                graph=True,
            )

        with pytest.raises(OSError, match="read failed"):
            prepare()
        assert all(s._replay_check is None for s in stages)
        stages[1].finish.side_effect = None
        prepare()
        for staging in stages:
            assert staging._replay_check is not None
            staging.discard()
            staging.check_replay()
            contexts = model._ple_host_contexts.numpy()
            staging.expect_replay(contexts)
            with pytest.raises(RuntimeError, match="replay contexts differ"):
                staging.check_replay()
            staging._contexts[1].copy_(model._ple_host_contexts)
            staging.check_replay()
            staging.expect_replay(contexts)
            staging._contexts[1].zero_()
            staging.check_replay()
            staging.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
