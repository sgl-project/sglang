# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for prepared Sage communication and its SubBlock consumer."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed import sage_ulysses as adapter
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
    prepared_sage_attention as attention_backend,
)


@pytest.mark.parametrize("used", [64, 130])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_cake_receives_exact_tail_and_noncontiguous_routing(
    monkeypatch, used, heterogeneous
):
    global_sequence = 256
    blocks = (used + 63) // 64
    topk = max(1, blocks - 1)
    index = torch.arange(blocks, dtype=torch.int32)[::2][:topk]
    topk = index.numel()
    index = index.view(1, 1, 1, topk).expand(1, 2, blocks, topk).contiguous()
    monkeypatch.setattr(
        attention_backend,
        "SubBlockRouter",
        lambda **kwargs: SimpleNamespace(
            route=lambda *args, **kwargs: SimpleNamespace(index=index, topk=topk)
        ),
    )
    calls = []
    mask = torch.arange(blocks) % 2 == 0 if heterogeneous else None

    def kernel(q, k, v, qs, ks, vs, indices, count, **kwargs):
        assert q.shape == k.shape == (1, 2, used, 128)
        assert v.shape == (1, 2, 128, blocks * 64)
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert qs.shape == (1, 2, ((used + 127) // 128) * 4)
        assert ks.shape == (1, 2, blocks)
        assert kwargs["backend"] == "cake"
        assert kwargs["uniform_block_count"] is not (heterogeneous)
        assert kwargs["contiguous_block_indices"] is False
        if heterogeneous:
            counts = kwargs["q2k_block_nums"]
            assert counts.shape == (1, 2, blocks)
            assert counts[0, 0].tolist() == [
                topk if selected else blocks for selected in mask.tolist()
            ]
            assert indices[0, 0, 0, :topk].tolist() == index[0, 0, 0].tolist()
            if blocks > 1:
                assert indices[0, 0, 1].tolist() == list(range(blocks))
        else:
            assert kwargs["q2k_block_nums"] is None
        assert kwargs["block_sizes"].tolist() == [64] * (blocks - 1) + [
            used - 64 * (blocks - 1)
        ]
        out = kwargs["out"]
        assert (
            out.shape == q.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
        )
        calls.append((indices, count))
        out.fill_(3)
        return out

    monkeypatch.setattr(
        attention_backend, "load_bsa_attn_sm120_blk64_sage_fwd", lambda: kernel
    )
    schedule = SimpleNamespace(
        sparsity=0.75, n_k=1, n_q=1, min_seq_len=0, skip_first_steps=10
    )
    q = torch.zeros((1, global_sequence, 2, 128), dtype=torch.int8)
    v = torch.zeros((1, 128, 2, global_sequence), dtype=torch.float8_e4m3fn)
    operands = SimpleNamespace(
        q=q,
        k=q,
        v=v,
        q_scale=torch.ones((1, 2, 8)),
        k_scale=torch.ones((1, 2, 4)),
        v_scale=torch.ones((1, 2, 128)),
        used_sequence=used,
        logical_sequence=global_sequence,
        input_dtype=torch.bfloat16,
        layout="sage2_sm89_sm120",
    )
    out = attention_backend.PreparedSubBlockSageAttention(schedule, mask)(
        operands, softmax_scale=128**-0.5
    )
    assert torch.all(out[:, :used] == 3)
    assert torch.count_nonzero(out[:, used:]) == 0
    indices, count = calls[0]
    if heterogeneous:
        assert count == blocks
    else:
        assert count == topk and indices is index


def test_subblock_consumer_rejects_an_incompatible_quantized_layout():
    consumer = attention_backend.PreparedSubBlockSageAttention(
        SimpleNamespace(n_k=4, n_q=4)
    )
    with pytest.raises(ValueError, match="requires sage2_sm89_sm120"):
        consumer(SimpleNamespace(layout="sage2_sm90"), softmax_scale=128**-0.5)


@pytest.mark.parametrize("eligible", [True, False])
def test_prepared_forward_reuses_operands_and_gathers_once(monkeypatch, eligible):
    from sglang.multimodal_gen.runtime.distributed import sage_ulysses_route as routing

    monkeypatch.setattr(
        routing,
        "check_prepared_sage_route",
        lambda *args, **kw: SimpleNamespace(enabled=eligible, reason="test"),
    )
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    encoded = SimpleNamespace(
        q=torch.zeros((1, 256, 2, 128), dtype=torch.int8),
        k=torch.zeros((1, 256, 2, 128), dtype=torch.int8),
        v=torch.zeros((1, 128, 2, 256), dtype=torch.float8_e4m3fn),
        q_scale=torch.ones((1, 2, 8)),
        k_scale=torch.ones((1, 2, 4)),
        v_scale=torch.ones((1, 2, 128)),
        used_sequence=130,
        logical_sequence=256,
        input_dtype=torch.bfloat16,
    )
    calls = []

    def exchange(q_in, k_in, v_in):
        assert eligible and q_in is k_in is v_in is q
        calls.append("quantize+exchange")
        return encoded

    def compute(operands, *, softmax_scale):
        assert operands is encoded
        assert softmax_scale == 128**-0.5
        calls.append("compute")
        return torch.full_like(operands.q, 5, dtype=operands.input_dtype)

    def gather(output):
        calls.append("gather")
        return output.reshape(1, 64, 8, 128)

    monkeypatch.setattr(
        attention_backend, "PreparedSubBlockSageAttention", lambda *args: compute
    )
    impl = SimpleNamespace(
        schedule=SimpleNamespace(quantize_before_a2a=True, min_seq_len=0),
        layer_enabled=True,
        _step_enabled=lambda: True,
        softmax_scale=128**-0.5,
    )
    result = attention_backend.prepared_subblock_sage_attention_forward(
        q,
        q,
        q,
        impl=impl,
        cu_seqlens_host=(0, 130, 256),
        max_seqlen=130,
        ulysses_active=True,
        ring_active=False,
        query_block_mask=torch.ones(3, dtype=torch.bool),
        transport=SimpleNamespace(exchange=exchange, gather=gather),
    )
    if eligible:
        assert calls == ["quantize+exchange", "compute", "gather"]
        assert result.shape == q.shape and torch.all(result == 5)
    else:
        assert calls == [] and result is None


@pytest.mark.parametrize("gate", ["opt_in", "layer", "step", "length"])
def test_declines_before_transport_when_schedule_does_not_admit(monkeypatch, gate):
    impl = SimpleNamespace(
        schedule=SimpleNamespace(
            quantize_before_a2a=gate != "opt_in",
            min_seq_len=131 if gate == "length" else 0,
        ),
        layer_enabled=gate != "layer",
        _step_enabled=lambda: gate != "step",
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Schedule rejection must precede transport initialization")

    monkeypatch.setattr(adapter, "_get_transport", unexpected)
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    assert (
        attention_backend.prepared_subblock_sage_attention_forward(
            q,
            q,
            q,
            impl=impl,
            cu_seqlens_host=(0, 130, 256),
            max_seqlen=130,
            ulysses_active=True,
            ring_active=False,
        )
        is None
    )


@pytest.mark.parametrize("fail", [False, True])
def test_ulysses_accepts_an_independent_attention_consumer(monkeypatch, fail):
    calls = []
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    # The communication wrapper must pass the received record through without
    # inspecting attention-specific fields or creating a SubBlock consumer.
    operands = object()
    computed = torch.full((1, 256, 2, 128), 7, dtype=q.dtype)

    def unexpected(*args, **kwargs):
        pytest.fail("A supplied attention consumer must not invoke SubBlock")

    monkeypatch.setattr(attention_backend, "PreparedSubBlockSageAttention", unexpected)

    def exchange(query, key, value):
        assert query is key is value is q
        calls.append("exchange")
        return operands

    def attention(received):
        assert received is operands
        calls.append("attention")
        if fail:
            raise RuntimeError("attention failed")
        return computed

    def gather(output):
        assert output is computed
        calls.append("gather")
        return output.reshape(1, *q.shape)

    def get_transport(group, query, used):
        assert group == "group" and query is q and used == 130
        calls.append("prepare")
        return SimpleNamespace(exchange=exchange, gather=gather)

    monkeypatch.setattr(adapter, "_get_transport", get_transport)
    kwargs = dict(group="group", used_sequence=130, attention=attention)
    if fail:
        with pytest.raises(RuntimeError, match="attention failed"):
            adapter.prepared_sage_ulysses_forward(q, q, q, **kwargs)
        assert calls == ["prepare", "exchange", "attention"]
    else:
        result = adapter.prepared_sage_ulysses_forward(q, q, q, **kwargs)
        assert result.shape == q.shape and torch.all(result == 7)
        assert calls == ["prepare", "exchange", "attention", "gather"]


def test_workspace_and_received_storage_are_reused(monkeypatch):
    import flashinfer.comm
    import torch.distributed as dist

    calls = []
    encoded = object()

    class Communicator:
        def __init__(self, group, **kwargs):
            assert kwargs["backend"] == "nccl"
            assert kwargs["max_bytes"] == 3 * q.numel() * q.element_size()

        def prepare_qkv(self, shape, **kwargs):
            calls.append("prepare")
            assert shape == (1, *q.shape) and kwargs == {
                "quantization": "sage2",
                "used_sequence": 130,
            }
            return "workspace"

        def scatter_qkv(self, *args, workspace, out):
            assert workspace == "workspace" and out is (
                None if len(calls) == 1 else encoded
            )
            calls.append("scatter")
            return encoded

        def gather_heads(self, source, out):
            calls.append("gather")
            return out

        def close(self):
            calls.append("close")

    monkeypatch.setattr(flashinfer.comm, "UlyssesCommunicator", Communicator)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    transport = adapter._PreparedSageTransport("group", q, 130)
    assert transport.exchange(q, q, q) is encoded
    assert transport.exchange(q, q, q) is encoded
    assert transport.gather(q) is transport.gathered
    transport.close()
    assert calls == ["prepare", "scatter", "scatter", "gather", "close"]


def test_missing_prepared_api_fails_with_actionable_error(monkeypatch):
    import flashinfer.comm

    monkeypatch.setattr(flashinfer.comm, "UlyssesCommunicator", object)
    with pytest.raises(ImportError, match="#4924"):
        adapter._PreparedSageTransport("group", torch.zeros(64, 8, 128), 130)


@pytest.mark.parametrize("value", [False, True])
def test_schedule_requires_explicit_prepared_qkv_opt_in(monkeypatch, value):
    from sglang.multimodal_gen.runtime import server_args
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
        SubBlockSparseSchedule,
    )

    config = {"compute_mode": "sage_fp8"}
    if value:
        config["quantize_before_a2a"] = True
    monkeypatch.setattr(
        server_args,
        "get_global_server_args",
        lambda: SimpleNamespace(attention_backend_config=config),
    )
    assert SubBlockSparseSchedule.from_server_args().quantize_before_a2a is value


@pytest.mark.parametrize(
    "config",
    [
        {"quantize_before_a2a": "false"},
        {"quantize_before_a2a": 1},
        {"quantize_before_a2a": True, "compute_mode": "bf16"},
    ],
)
def test_invalid_prepared_qkv_configuration_is_rejected(monkeypatch, config):
    from sglang.multimodal_gen.runtime import server_args
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
        SubBlockSparseSchedule,
    )

    monkeypatch.setattr(
        server_args,
        "get_global_server_args",
        lambda: SimpleNamespace(attention_backend_config=config),
    )
    with pytest.raises(ValueError, match="quantize_before_a2a"):
        SubBlockSparseSchedule.from_server_args()


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "ring",
        "training",
        "capture",
        "architecture",
        "rocm",
        "head_dim",
        "dtype",
        "world",
        "group",
        "heads",
        "documents",
        "live_length",
        "long_sequence",
        "compile",
    ],
)
def test_geometry_guards_precede_collectives(monkeypatch, case):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.distributed import sage_ulysses_route as route

    monkeypatch.setattr(torch.version, "hip", "test" if case == "rocm" else None)
    monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_PREPARED_SAGE_QKV", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: (9, 0) if case == "architecture" else (12, 0),
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: case == "capture"
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: case == "compile")
    monkeypatch.setattr(
        route, "get_ulysses_parallel_world_size", lambda: 2 if case == "world" else 4
    )
    monkeypatch.setattr(
        route, "get_sp_group", lambda: SimpleNamespace(ulysses_group="group")
    )
    monkeypatch.setattr(
        route.dist, "get_world_size", lambda group: 3 if case == "group" else 4
    )
    total = 2**21 if case == "long_sequence" else 256
    used = total - 17 if case == "long_sequence" else 130
    q = SimpleNamespace(
        shape=(
            total // 4,
            10 if case == "heads" else 8,
            64 if case == "head_dim" else 128,
        ),
        dtype=torch.float32 if case == "dtype" else torch.bfloat16,
        device=torch.device("cuda", 0),
        is_cuda=True,
        requires_grad=False,
        ndim=3,
    )
    with nullcontext() if case == "training" else torch.inference_mode():
        result = route.check_prepared_sage_route(
            q,
            q,
            q,
            cu_seqlens_host=(0, total) if case == "documents" else (0, used, total),
            max_seqlen=used + 1 if case == "live_length" else used,
            ulysses_active=True,
            ring_active=case == "ring",
        )
    assert result.enabled is (case in ("valid", "long_sequence"))


def test_transport_cache_replaces_geometries_without_a_lifetime_limit(monkeypatch):
    calls = []
    created = []

    class Transport:
        def __init__(self, group, query, used):
            self.used = used
            self.closed = False
            calls.append(("prepare", used))
            created.append(self)

        def close(self):
            self.closed = True
            calls.append(("close", self.used))

    monkeypatch.setattr(adapter, "_TRANSPORTS", {})
    monkeypatch.setattr(adapter, "_PreparedSageTransport", Transport)
    q = torch.zeros(64, 8, 128)
    previous = None
    # More distinct live prefixes than the former eight-entry limit, even
    # though every call has the same padded tensor shape.
    for used in range(130, 142):
        current = adapter._get_transport("group", q, used)
        assert adapter._get_transport("group", q, used) is current
        previous = current
    assert len(adapter._TRANSPORTS) == 1
    assert all(transport.closed for transport in created[:-1])
    assert not previous.closed
    adapter.shutdown_prepared_sage_ulysses()
    assert not adapter._TRANSPORTS and previous.closed
    assert sum(kind == "prepare" for kind, _ in calls) == 12


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_transport_cache_replaces_changed_tensor_geometry(monkeypatch, change):
    calls = []

    class Transport:
        def __init__(self, *args):
            calls.append("prepare")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(adapter, "_TRANSPORTS", {})
    monkeypatch.setattr(adapter, "_PreparedSageTransport", Transport)
    q = torch.zeros(64, 8, 128, dtype=torch.bfloat16)
    first = adapter._get_transport("group", q, 130)
    changed = (
        torch.zeros(128, 8, 128, dtype=q.dtype)
        if change == "shape"
        else q.to(torch.float16)
    )
    second = adapter._get_transport("group", changed, 130)
    assert second is not first
    assert adapter._get_transport("group", changed, 130) is second
    assert calls == ["prepare", "close", "prepare"]
    adapter.shutdown_prepared_sage_ulysses()


def test_transport_replacement_is_scoped_to_group_and_device(monkeypatch):
    class Transport:
        def __init__(self, *args):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(adapter, "_TRANSPORTS", {})
    monkeypatch.setattr(adapter, "_PreparedSageTransport", Transport)
    q = torch.zeros(64, 8, 128)
    on_meta = q.to("meta")
    first = adapter._get_transport("first", q, 130)
    other_group = adapter._get_transport("second", q, 130)
    other_device = adapter._get_transport("first", on_meta, 130)
    replacement = adapter._get_transport("first", q, 131)
    assert first.closed and replacement is not first
    assert not other_group.closed and not other_device.closed
    assert adapter._get_transport("second", q, 130) is other_group
    assert adapter._get_transport("first", on_meta, 130) is other_device
    adapter.shutdown_prepared_sage_ulysses()
    assert other_group.closed and other_device.closed and replacement.closed


def test_failed_transport_replacement_can_be_retried(monkeypatch):
    fail = True
    closed = []

    class Transport:
        def __init__(self, group, query, used):
            self.used = used
            if used == 131 and fail:
                raise RuntimeError("prepare failed")

        def close(self):
            closed.append(self.used)

    monkeypatch.setattr(adapter, "_TRANSPORTS", {})
    monkeypatch.setattr(adapter, "_PreparedSageTransport", Transport)
    q = torch.zeros(64, 8, 128)
    adapter._get_transport("group", q, 130)
    with pytest.raises(RuntimeError, match="prepare failed"):
        adapter._get_transport("group", q, 131)
    assert closed == [130] and not adapter._TRANSPORTS
    fail = False
    recovered = adapter._get_transport("group", q, 131)
    assert adapter._get_transport("group", q, 131) is recovered
    adapter.shutdown_prepared_sage_ulysses()


@pytest.mark.parametrize("stage", ["prepare", "gather"])
def test_transport_initialization_closes_communicator_on_failure(monkeypatch, stage):
    import flashinfer.comm
    import torch.distributed as dist

    calls = []

    class Communicator:
        def __init__(self, *args, **kwargs):
            pass

        def prepare_qkv(self, *args, **kwargs):
            if stage == "prepare":
                raise RuntimeError("allocation failed")
            return "workspace"

        def close(self):
            calls.append("close")

    q = torch.zeros(64, 8, 128)
    monkeypatch.setattr(flashinfer.comm, "UlyssesCommunicator", Communicator)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    if stage == "gather":

        def fail_allocation(*args, **kwargs):
            raise RuntimeError("allocation failed")

        monkeypatch.setattr(torch, "empty", fail_allocation)
    with pytest.raises(RuntimeError, match="allocation failed"):
        adapter._PreparedSageTransport("group", q, 130)
    assert calls == ["close"]


def test_transport_close_releases_owned_buffers_and_is_idempotent(monkeypatch):
    import weakref

    import flashinfer.comm
    import torch.distributed as dist

    calls = []

    class Communicator:
        def __init__(self, *args, **kwargs):
            pass

        def prepare_qkv(self, *args, **kwargs):
            return torch.empty(1)

        def close(self):
            calls.append("close")

    monkeypatch.setattr(flashinfer.comm, "UlyssesCommunicator", Communicator)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    transport = adapter._PreparedSageTransport("group", torch.zeros(64, 8, 128), 130)
    transport.encoded = torch.empty(1)
    refs = [
        weakref.ref(getattr(transport, name))
        for name in ("workspace", "encoded", "gathered")
    ]
    transport.close()
    transport.close()
    assert calls == ["close"]
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize(
    "local,peer", [(True, True), (True, False), (False, True), (False, False)]
)
def test_hardware_admission_is_collective_and_cached(monkeypatch, local, peer):
    import torch.distributed as dist

    calls = []
    monkeypatch.setattr(adapter, "_HARDWARE_SUPPORT", {})
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(adapter, "_TRANSPORTS", {})
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (12, 0) if local else (9, 0)
    )

    def all_reduce(vote, *, op, group):
        assert op == dist.ReduceOp.MIN and group == "group"
        calls.append(int(vote[0]))
        vote.fill_(int(local and peer))

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    for _ in range(2):
        assert adapter._hardware_supported_by_all_ranks(
            "group", torch.device("cpu")
        ) is (local and peer)
    assert calls == [int(local)]
    adapter.shutdown_prepared_sage_ulysses()
    assert not adapter._HARDWARE_SUPPORT


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_prepared_exchange_zeros_only_rank_local_padding(monkeypatch, rank):
    import flashinfer.comm
    import torch.distributed as dist

    received = []

    class Communicator:
        def __init__(self, group, **kwargs):
            pass

        def prepare_qkv(self, *args, **kwargs):
            return "workspace"

        def scatter_qkv(self, *operands, **kwargs):
            received.extend(operand.clone() for operand in operands)
            return "encoded"

    monkeypatch.setattr(flashinfer.comm, "UlyssesCommunicator", Communicator)
    monkeypatch.setattr(dist, "get_rank", lambda group: rank)
    packed = torch.empty((64, 8, 3, 128), dtype=torch.bfloat16)
    q, k, v = packed.unbind(dim=2)
    for i, operand in enumerate((q, k, v), 1):
        operand.fill_(i)
    valid = max(0, min(64, 130 - rank * 64))
    transport = adapter._PreparedSageTransport("group", q, 130)
    assert transport.exchange(q, k, v) == "encoded"
    for i, (operand, sent) in enumerate(zip((q, k, v), received), 1):
        assert not operand.is_contiguous()
        assert torch.all(operand[:valid] == i)
        assert not operand[valid:].count_nonzero()
        torch.testing.assert_close(sent[0], operand, rtol=0, atol=0)


def test_subblock_routing_budget_declines_before_shared_exchange(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed import sage_ulysses_route

    monkeypatch.setattr(
        sage_ulysses_route,
        "check_prepared_sage_route",
        lambda *args, **kwargs: SimpleNamespace(enabled=True, reason=None),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Oversized sparse routing metadata must decline before exchange")

    total = 2**21
    q = SimpleNamespace(shape=(total // 4, 8, 128))
    impl = SimpleNamespace(
        schedule=SimpleNamespace(quantize_before_a2a=True, min_seq_len=0),
        layer_enabled=True,
        _step_enabled=lambda: True,
    )
    assert (
        attention_backend.prepared_subblock_sage_attention_forward(
            q,
            q,
            q,
            impl=impl,
            cu_seqlens_host=(0, total - 17, total),
            max_seqlen=total - 17,
            ulysses_active=True,
            ring_active=False,
            transport=SimpleNamespace(exchange=unexpected),
        )
        is None
    )
