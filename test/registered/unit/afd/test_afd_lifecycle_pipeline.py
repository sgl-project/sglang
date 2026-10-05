from sglang.test.afd.graph_fixtures import make_shape as planned_shape
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
import queue
import threading
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as config_mod
from sglang.srt.afd import connector as connector_mod
from sglang.srt.afd import contracts as contracts
from sglang.srt.afd import metadata as qwen
from sglang.srt.afd import pipeline as pipeline_mod
from sglang.srt.afd import role_graph as role_graph
from sglang.srt.afd import transport as afd_transport
from sglang.test.afd.graph_fixtures import role_service
from sglang.test.afd.pipeline_fixtures import (
    AttentionAdapter,
    DecodeBatch,
    RetainedTransport,
)


class FakeTransport(RetainedTransport):
    capturing = False

    def rejoin_streams(self):
        pass

    def __init__(
        self,
        role=contracts.AFDRole.ATTENTION,
        same_stream=False,
        *,
        stages=None,
        timeline=None,
    ):
        super().__init__(role=role)
        self._same_stream = same_stream
        self.events = []
        self.close_calls = 0
        self.exchange_calls = 0
        self._stages = stages
        self._timeline = timeline
        self._recv_count = 0

    def validate_invariants(self):
        if self._same_stream:
            raise RuntimeError("AFD_TRANSPORT_DISTINCT_STREAMS_REQUIRED")

    def begin_step(self, descriptor):
        self.events.append(("begin", descriptor.step_id))
        return descriptor

    def dispatch(self, tensor):
        self.events.append(("a2e", tensor.shape[0]))
        if self._timeline is not None:
            self._timeline.append(("send", *tensor.tag))

    def receive_return(self, buffer):
        self.events.append(("e2a", buffer.shape[0]))
        if self._stages is None:
            event = ("e2a_done", len(self.events))
        else:
            event = (
                "e2a_done",
                self._recv_count // self._stages,
                self._recv_count % self._stages,
            )
            self._recv_count += 1
        if self._timeline is not None:
            self._timeline.append(("post", *event[1:]))
        return event

    def wait(self, event):
        self.events.append(("wait", event))
        if self._timeline is not None:
            self._timeline.append(("wait", *event[1:]))

    def exchange_close(self, *, usage):
        self.exchange_calls += 1
        return {"peer": usage["status"]}

    def close(self):
        self.close_calls += 1
        return {"close_calls": self.close_calls}


def tensor_rows(rows, hidden=8):
    return torch.zeros((rows, hidden), dtype=torch.bfloat16)


class TraceAdapter(AttentionAdapter):
    num_layers = 48

    def __init__(self, trace):
        self.trace = trace

    def local_compute(self, *, layer, stage, hidden_states, residual, positions=None):
        self.trace.append(("local", layer, stage.index))
        hidden_states.tag = (layer, stage.index)
        return hidden_states, residual

    def finish_layer(self, *, layer, stage, ffn_output, residual):
        self.trace.append(("finish", layer, stage.index))
        return ffn_output, residual


@pytest.mark.parametrize("stages", [2])
def test_48_layer_qwen_callback_trace_is_stage_wise_wavefront(stages):
    """Each stage dispatches before the next stage's local compute starts."""

    cfg = config_mod.AFDConfig(
        stages=stages,
    )
    graph = role_service(role=contracts.AFDRole.ATTENTION, num_layers=48)[1]
    trace = []
    transport = FakeTransport(stages=stages, timeline=trace)
    connector = connector_mod.AFDConnector(
        transport=transport,
        graph_strategy=graph,
    )
    pipeline = pipeline_mod.AFDAttentionPipeline(
        adapter=TraceAdapter(trace),
        connector=connector,
        config=cfg,
        shape_factory=planned_shape,
    )
    output, residual = pipeline.execute(
        hidden_states=tensor_rows(stages * 2),
        residual=None,
        positions=tensor_rows(stages * 2, 1),
        forward_batch=DecodeBatch(),
    )
    assert output.shape[0] == stages * 2
    assert residual is None
    local = [item for item in trace if item[0] == "local"]
    finish = [item for item in trace if item[0] == "finish"]
    assert local == [
        ("local", layer, stage) for layer in range(48) for stage in range(stages)
    ]
    assert finish == [
        ("finish", layer, stage) for layer in range(48) for stage in range(stages)
    ]
    wire = [event[0] for event in transport.events[1:]]
    assert wire == (
        ["a2e", "e2a"] * stages
        + (["wait", "a2e", "e2a"] * stages) * 47
        + ["wait"] * stages
    )
    positions = {node: index for index, node in enumerate(trace)}
    # The wavefront this preserves: a stage's dispatch precedes the next stage's
    # local compute, so the FFN role computes one stage while this role runs the
    # next. The stream wait orders each return before its consumer.
    for layer in range(48):
        for stage in range(stages - 1):
            assert (
                positions[("send", layer, stage)]
                < positions[("local", layer, stage + 1)]
            )
    for layer in range(47):
        for stage in range(stages):
            assert positions[("post", layer, stage)] < positions[("wait", layer, stage)]
            assert (
                positions[("wait", layer, stage)] < positions[("finish", layer, stage)]
            )
            assert (
                positions[("finish", layer, stage)]
                < positions[("local", layer + 1, stage)]
            )
            assert (
                positions[("local", layer + 1, stage)]
                < positions[("send", layer + 1, stage)]
            )


def test_connector_rejects_aliased_a2e_e2a_streams():
    """The old optional-own-stream flag must not reappear as an aliasing route."""

    graph = role_service(role=contracts.AFDRole.ATTENTION, num_layers=1)[1]
    with pytest.raises(RuntimeError, match="DISTINCT_STREAMS_REQUIRED"):
        connector_mod.AFDConnector(
            transport=FakeTransport(same_stream=True),
            graph_strategy=graph,
        )


@pytest.mark.parametrize("failure", [False, True], ids=["success", "peer-timeout"])
def test_connector_close_and_final_usage_are_exactly_once(failure):
    class Transport(FakeTransport):
        def exchange_close(self, *, usage):
            result = super().exchange_close(usage=usage)
            if failure:
                raise RuntimeError("peer timed out")
            return result

    transport, emitted = Transport(), []
    connector = connector_mod.AFDConnector(
        transport=transport,
        graph_strategy=role_service(num_layers=1)[1],
        usage_emitter=emitted.append,
    )
    if failure:
        with pytest.raises(RuntimeError, match="AGGREGATE_CLOSE_FAILED") as first:
            connector.close()
        with pytest.raises(RuntimeError) as second:
            connector.close()
        assert second.value is first.value
        assert len(emitted) == 1
    else:
        first = connector.close()
        assert connector.close() is first
        assert first["event"] == "CLOSE"
        assert first["usage_final"] is True
        assert emitted == [first]
    assert transport.exchange_calls == transport.close_calls == 1


class FlashAttentionBackend:
    prefill_attention_backend_str = "fa3"
    decode_attention_backend_str = "fa3"

    def __init__(self):
        self.forward_metadata = object()
        self.capture_metadata = {"page_table": torch.zeros(32, dtype=torch.int32)}
        self.decode_cuda_graph_metadata = {"page_table": object()}
        self._sched_meta_buf = object()
        self.out_graph_calls = []
        self.metadata_values = []

    def init_forward_metadata_out_graph(self, forward_batch, in_capture):
        self.out_graph_calls.append((forward_batch.batch_size, in_capture))
        rows = forward_batch.num_token_non_padded_cpu
        self.metadata_values.append(
            (
                tuple(forward_batch.req_pool_indices.tolist()[:rows]),
                tuple(forward_batch.out_cache_loc.tolist()[:rows]),
            )
        )
        if forward_batch.batch_size not in self.decode_cuda_graph_metadata:
            self.decode_cuda_graph_metadata[forward_batch.batch_size] = (
                self.capture_metadata
            )
        self.forward_metadata = self.decode_cuda_graph_metadata[
            forward_batch.batch_size
        ]

    def init_forward_metadata_in_graph(self, forward_batch):
        del forward_batch


FlashAttentionBackend.__module__ = "sglang.srt.layers.attention.flashattention_backend"


@pytest.fixture
def fa_backend(monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType

    if importlib.util.find_spec("sgl_kernel") is None:
        kernel = ModuleType("sgl_kernel")

        def unexpected_kernel(*args, **kwargs):
            raise AssertionError("CUDA kernel called in CPU metadata test")

        kernel.merge_state_v2 = unexpected_kernel
        monkeypatch.setitem(sys.modules, "sgl_kernel", kernel)
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend as NativeFlashAttentionBackend,
    )

    # Only the device metadata generation boundary is fake. The snapshot
    # allocation, validation, refresh and release are the actual backend API.
    monkeypatch.setattr(
        FlashAttentionBackend,
        "create_graph_metadata_snapshot",
        NativeFlashAttentionBackend.create_graph_metadata_snapshot,
        raising=False,
    )
    return FlashAttentionBackend


def _forward_batch(rows, *, offset=0):
    values = list(range(offset + 1, offset + rows + 1))
    return SimpleNamespace(
        batch_size=rows,
        input_ids=torch.tensor(values),
        positions=torch.tensor(values),
        req_pool_indices=torch.tensor(values),
        seq_lens=torch.tensor(values),
        seq_lens_cpu=None,
        seq_lens_sum=sum(values),
        out_cache_loc=torch.tensor(values),
        num_token_non_padded_cpu=rows,
        forward_metadata_ready=False,
        forward_metadata_planned_bs=None,
        forward_metadata_planned_num_tokens=None,
    )


@pytest.mark.parametrize("version", ["fa3", "fa4"])
def test_fa3_fa4_snapshot_restore_and_cache_drift_guard(version, fa_backend):
    """A backend cache replacement must fall back before replaying stale pointers."""

    backend = fa_backend()
    backend.prefill_attention_backend_str = version
    backend.decode_attention_backend_str = version
    original = backend.forward_metadata
    batch = _forward_batch(2)
    stage = SimpleNamespace(forward_batch=batch, attention_metadata=None)
    guard = qwen.FlashAttentionMetadataGuard(
        backend=backend,
        stage=stage,
        bucket_rows=32,
    )
    guard.capture(batch)
    guard.activate_in_graph()
    assert stage.forward_batch.batch_size == 32
    assert stage.forward_batch.input_ids.tolist()[2:] == [0] * 30
    guard.restore()
    assert backend.forward_metadata is original
    assert stage.forward_batch is batch
    guard.prepare_replay(batch)
    guard.assert_stable()
    guard.restore()
    assert backend.out_graph_calls == [(32, True), (32, False)]
    distinct_batch = _forward_batch(2)
    guard.prepare_replay(distinct_batch)
    guard.restore()
    assert backend.out_graph_calls == [(32, True)] + [(32, False)] * 2
    guard.prepare_replay(distinct_batch)
    guard.restore()
    assert backend.out_graph_calls == [(32, True)] + [(32, False)] * 3
    peer_stage = SimpleNamespace(forward_batch=batch, attention_metadata=None)
    peer_guard = qwen.FlashAttentionMetadataGuard(
        backend=backend,
        stage=peer_stage,
        bucket_rows=32,
    )
    peer_guard.capture(batch)
    peer_guard.restore()
    assert backend.out_graph_calls == [(32, True)] + [(32, False)] * 4
    snapshot = guard._snapshot
    guard.close()
    guard.close()
    assert snapshot.metadata is None and not snapshot.is_valid()
    peer_guard.prepare_replay(batch)
    peer_guard.restore()
    backend.decode_cuda_graph_metadata = {"page_table": object()}
    with pytest.raises(contracts.AFDError, match="METADATA_CACHE_DRIFT"):
        peer_guard.assert_stable()


@pytest.mark.parametrize(
    "version,available,code",
    [
        (21600, True, "NCCL_CTA_CONFIG_UNSUPPORTED"),
        (21700, False, "NCCL_COMM_INIT_RANK_CONFIG_ABSENT"),
        (21700, True, None),
        (23000, True, None),
    ],
)
def test_the_channel_cap_binds_to_the_communicator_not_the_process(
    version, available, code
):
    from sglang.srt.distributed.device_communicators import pynccl_wrapper as wrapper

    captured = []
    library = object.__new__(wrapper.NCCLLibrary)
    library._funcs = (
        {"ncclCommInitRankConfig": lambda *args: captured.append(args[-1]._obj)}
        if available
        else {}
    )
    library.ncclGetRawVersion = lambda: version
    library.NCCL_CHECK = lambda result: None
    if code:
        with pytest.raises(RuntimeError, match=code):
            library.ncclCommInitRankConfig(
                2, wrapper.ncclUniqueId(), 0, min_ctas=8, max_ctas=8
            )
        assert not captured
    else:
        library.ncclCommInitRankConfig(
            2, wrapper.ncclUniqueId(), 0, min_ctas=8, max_ctas=8
        )
        assert len(captured) == 1
        assert captured[0].minCTAs == captured[0].maxCTAs == 8


def test_the_nccl_config_struct_preserves_upstream_abi_and_defaults():
    """Keep upstream's versioned ABI while pinning per-communicator CTAs."""
    import ctypes

    from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
        ncclConfig_t as struct,
    )

    value = struct.create()
    assert value.size == ctypes.sizeof(struct)
    assert value.magic == 0xCAFEBEEF
    assert value.version == 23000
    assert struct.minCTAs.offset == 24
    assert struct.maxCTAs.offset == 28
    assert struct.graphUsageMode.offset > struct.maxCTAs.offset
    for name, field_type in struct._fields_[3:]:
        assert getattr(value, name) == (
            None if field_type is ctypes.c_char_p else -(2**31)
        )
    value.minCTAs = value.maxCTAs = 8
    value.graphUsageMode = 1
    assert (value.minCTAs, value.maxCTAs, value.graphUsageMode) == (8, 8, 1)
    assert value.version == 23000  # header ABI, never the loaded library version


class BoundComm:
    def __init__(self):
        self.available = True
        self.group = object()
        self.comm = object()
        self.device = "cuda:0"
        self.rank = 0
        self.world_size = 2
        self.disabled = True

    @contextmanager
    def change_state(self, enable=None):
        old = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old


class BoundStream:
    cuda_stream = 17


class BoundCuda:
    @staticmethod
    def stream(stream):
        del stream
        return nullcontext()


def test_real_nccl_binding_scopes_enable_and_restores_sentinel():
    """Production PyNccl calls must run enabled on the bound stream, then restore."""

    comm = BoundComm()
    stream = BoundStream()
    binding = afd_transport._BoundNCCL(
        comm=comm,
        stream=stream,
        device="cuda:0",
    )
    observed = []
    binding.invoke(
        torch=SimpleNamespace(cuda=BoundCuda()),
        operation=lambda active: observed.append(active.disabled),
    )
    assert observed == [False]
    assert comm.disabled is True

    def fail_operation(active):
        assert active.disabled is False
        raise RuntimeError("operation failed")

    with pytest.raises(RuntimeError, match="operation failed"):
        binding.invoke(
            torch=SimpleNamespace(cuda=BoundCuda()),
            operation=fail_operation,
        )
    assert comm.disabled is True
    stream.cuda_stream = 18
    with pytest.raises(contracts.AFDError, match="NCCL_BINDING_DRIFT"):
        binding.invoke(
            torch=SimpleNamespace(cuda=BoundCuda()),
            operation=lambda active: None,
        )


class RecordingStore:
    """Minimal stand-in for the control store; records the close deadline."""

    def __init__(self):
        self.timeouts = []

    def set_timeout(self, value):
        self.timeouts.append(value)


class QueueControl:
    def __init__(self, *, inbound, outbound):
        self.inbound = inbound
        self.outbound = outbound
        self.send_count = 0
        self.store = RecordingStore()

    def send_obj(self, value, dst):
        del dst
        self.send_count += 1
        self.outbound.put(value)

    def recv_obj(self, src):
        del src
        return self.inbound.get(timeout=2)


class CloseTransport(FakeTransport):
    begin_step = afd_transport.AFDPairedP2PTransport.begin_step
    exchange_close = afd_transport.AFDPairedP2PTransport.exchange_close
    _retime_control_store = afd_transport.AFDPairedP2PTransport._retime_control_store

    def __init__(self, *, role, control):
        super().__init__(role=role)
        self._control = control
        self._peer_ranks = (1,) if role == contracts.AFDRole.FFN else (0,)
        self._peer_coordination_ranks = self._peer_ranks
        self._control_upstream = None
        self._control_followers = ()
        self._closed = False
        self._stats = {"steps": 0}
        self._forked_streams = []
        self._peer_close_usage = None
        self._close_exchange = None
        self._close_exchange_error = None
        self._close_timeout_seconds = 1


def test_close_lets_keyboard_interrupt_escape_instead_of_banking_it():
    """Ctrl-C during close must reach the interpreter, not become a close failure."""

    class InterruptingStrategy:
        role = contracts.AFDRole.ATTENTION
        retains_backing = False

        def close(self):
            raise KeyboardInterrupt

        def usage(self, *, status):
            del status
            return {}

    connector = connector_mod.AFDConnector(
        transport=CloseTransport(
            role=contracts.AFDRole.ATTENTION,
            control=QueueControl(inbound=queue.Queue(), outbound=queue.Queue()),
        ),
        graph_strategy=InterruptingStrategy(),
        usage_emitter=None,
    )
    with pytest.raises(KeyboardInterrupt):
        connector.close()
    # The interrupt must not be recorded as a component failure, so a later close
    # is still a first close rather than a replay of a banked RuntimeError.
    assert connector._close_receipt is None
    assert connector._close_error is None


def test_attention_ffn_close_ack_and_receipts_are_exactly_once():
    """The real exchange_close protocol must ACK before either final receipt returns."""

    attention_in = queue.Queue()
    ffn_in = queue.Queue()
    attention_control = QueueControl(
        inbound=attention_in,
        outbound=ffn_in,
    )
    ffn_control = QueueControl(
        inbound=ffn_in,
        outbound=attention_in,
    )
    attention_transport = CloseTransport(
        role=contracts.AFDRole.ATTENTION,
        control=attention_control,
    )
    ffn_transport = CloseTransport(
        role=contracts.AFDRole.FFN,
        control=ffn_control,
    )
    attention_emitted = []
    ffn_emitted = []
    attention = connector_mod.AFDConnector(
        transport=attention_transport,
        graph_strategy=role_service(role=contracts.AFDRole.ATTENTION, num_layers=1)[1],
        usage_emitter=attention_emitted.append,
    )
    ffn = connector_mod.AFDConnector(
        transport=ffn_transport,
        graph_strategy=role_service(role=contracts.AFDRole.FFN, num_layers=1)[1],
        usage_emitter=ffn_emitted.append,
    )
    receipts = {}

    def close_attention():
        receipts["attention"] = attention.close()

    def receive_and_close_ffn():
        descriptor = ffn_transport.begin_step(None)
        assert descriptor.kind == "CLOSE"
        receipts["ffn"] = ffn.close()

    threads = [
        threading.Thread(target=close_attention),
        threading.Thread(target=receive_and_close_ffn),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert receipts["attention"]["peer_graph"]["status"] == "CLOSED"
    assert receipts["ffn"]["peer_graph"]["status"] == "CLOSED"
    assert attention.close() is receipts["attention"]
    assert ffn.close() is receipts["ffn"]
    assert attention_control.send_count == 1
    assert ffn_control.send_count == 1
    assert attention_emitted == [receipts["attention"]]
    assert ffn_emitted == [receipts["ffn"]]


def test_close_timeout_is_bounded_and_replayed_without_second_send():
    """A missing ACK becomes one stable failure instead of another CLOSE frame."""

    class TimeoutControl:
        def __init__(self):
            self.send_count = 0
            self.store = RecordingStore()

        def send_obj(self, value, dst):
            del value, dst
            self.send_count += 1

        def recv_obj(self, src):
            del src
            raise TimeoutError("bounded store deadline")

    control = TimeoutControl()
    transport = CloseTransport(
        role=contracts.AFDRole.ATTENTION,
        control=control,
    )
    with pytest.raises(contracts.AFDError) as first:
        transport.exchange_close(usage={"status": "CLOSED"})
    with pytest.raises(contracts.AFDError) as second:
        transport.exchange_close(usage={"status": "CLOSED"})
    assert first.value.code == "AFD_TRANSPORT_CLOSE_ACK_TIMEOUT"
    assert second.value is first.value
    assert control.send_count == 1
    # The bound must be installed before the ACK wait, exactly once.
    assert control.store.timeouts == [timedelta(seconds=1)]


def test_role_graph_fa_metadata_stages_remain_distinct(fa_backend):
    """Whole-role staging must not let the last request overwrite every stage."""
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    backend = fa_backend()
    backend.capture_metadata = FlashAttentionMetadata(
        page_table=torch.zeros(4, dtype=torch.int64),
        cache_seqlens_int32=torch.zeros(4, dtype=torch.int64),
    )
    original_init = backend.init_forward_metadata_out_graph

    def update(batch, in_capture):
        original_init(batch, in_capture)
        backend.forward_metadata.page_table.copy_(batch.req_pool_indices)
        backend.forward_metadata.cache_seqlens_int32.copy_(batch.seq_lens)
        backend.forward_metadata.max_seq_len_k = max(batch.seq_lens.tolist())

    backend.init_forward_metadata_out_graph = update
    batches = [_forward_batch(1, offset=10), _forward_batch(1, offset=100)]
    stages = [
        SimpleNamespace(forward_batch=b, attention_metadata=None) for b in batches
    ]
    guards = [
        qwen.FlashAttentionMetadataGuard(backend=backend, stage=s, bucket_rows=4)
        for s in stages
    ]
    for guard, batch in zip(guards, batches):
        # Each role stage owns distinct static metadata.
        guard.capture(batch)
    for guard in guards:
        guard.activate_in_graph()
    captured = [stage.attention_metadata for stage in stages]
    assert [v.page_table.tolist()[0] for v in captured] == [11, 101]
    assert captured[0] is not captured[1]
    pointers = [id(v.page_table) for v in captured]
    for guard in guards:
        guard.restore()
    for guard, batch in zip(
        guards, [_forward_batch(1, offset=20), _forward_batch(1, offset=200)]
    ):
        guard.prepare_replay(batch)
    assert [v.page_table.tolist()[0] for v in captured] == [21, 201]
    assert [v.cache_seqlens_int32.tolist()[0] for v in captured] == [21, 201]
    assert [id(v.page_table) for v in captured] == pointers
    assert all(v.page_table.tolist()[1:] == [0, 0, 0] for v in captured)
    for guard in guards:
        guard.restore()
    backend.capture_metadata.page_table = torch.zeros(5, dtype=torch.int64)
    backend.init_forward_metadata_out_graph = original_init
    with pytest.raises(contracts.AFDError, match="FA_STAGE_METADATA_LAYOUT_CHANGED"):
        guards[0].prepare_replay(batches[0])
    assert not guards[0]._active
    assert captured[1].page_table.tolist() == [201, 0, 0, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
