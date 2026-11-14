# SPDX-License-Identifier: Apache-2.0
import faulthandler
import os
import sys
import time
import types

import pytest
import torch
import torch.multiprocessing as mp

HEAD_SIZE = 128
PRECISION_ENV = "SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION"
_MAX_REL_L2 = {"bf16": 0.02, "fp8": 0.1, "fp8_mixed": 0.3, "mxfp4": 0.4}


def _xpu_count():
    return torch.xpu.device_count() if torch.xpu.is_available() else 0


def _set_backend_config(monkeypatch, config):
    import addict

    from sglang.multimodal_gen.runtime.server_args import server_args

    monkeypatch.setattr(
        server_args,
        "_global_server_args",
        types.SimpleNamespace(attention_backend_config=addict.Dict(config)),
    )


@pytest.fixture(autouse=True)
def _empty_backend_config(monkeypatch):
    """With the ring on, XPUAttentionImpl reads sage_v3_bf16_blocks."""
    _set_backend_config(monkeypatch, {})


def _impl(*, num_kv_heads=40, head_size=HEAD_SIZE, causal=False):
    pytest.importorskip("sgl_kernel.flash_attn", reason="needs sgl-kernel-xpu")
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        XPUAttentionImpl,
    )

    return XPUAttentionImpl(
        num_heads=40,
        head_size=head_size,
        causal=causal,
        softmax_scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
    )


def _ring_module():
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        xpu_deepsymm_ring,
    )

    return xpu_deepsymm_ring


def _precision(query, impl, *, causal=False, is_cross_attention=False):
    return _ring_module().deepsymm_ring_precision(
        query,
        impl,
        ring_group=None,
        causal=causal,
        is_cross_attention=is_cross_attention,
    )


@pytest.fixture
def ring_precision(monkeypatch):
    def set_precision(precision, *, one_host=True):
        module = _ring_module()
        monkeypatch.setattr(module, "DEEPSYMM_RING_PRECISION", precision)
        monkeypatch.setattr(module, "_ring_group_on_one_host", lambda group: one_host)

    return set_precision


def test_off_by_default(monkeypatch):
    from sglang.multimodal_gen import envs

    monkeypatch.delenv(PRECISION_ENV, raising=False)
    precision = envs.SGLANG_DIFFUSION_XPU_DEEPSYMM_RING_PRECISION
    assert _ring_module()._parse_ring_precision(precision) is None


def test_unknown_precision_raises():
    with pytest.raises(ValueError, match=PRECISION_ENV):
        _ring_module()._parse_ring_precision("int8")


@pytest.mark.parametrize(
    "impl_kwargs, call_kwargs, device, dtype, expected",
    [
        ({}, {}, "xpu", torch.bfloat16, "mxfp4"),
        ({"num_kv_heads": 8}, {}, "xpu", torch.bfloat16, "bf16"),
        ({"head_size": 64}, {}, "xpu", torch.bfloat16, "bf16"),
        ({"head_size": 96}, {}, "xpu", torch.bfloat16, None),
        ({}, {"causal": True}, "xpu", torch.bfloat16, None),
        ({}, {"is_cross_attention": True}, "xpu", torch.bfloat16, None),
        ({}, {}, "xpu", torch.float16, None),
        ({}, {}, "cpu", torch.bfloat16, None),
    ],
)
def test_precision_rules(
    ring_precision, impl_kwargs, call_kwargs, device, dtype, expected
):
    if _xpu_count() < 1:
        pytest.skip("needs an XPU")
    ring_precision("mxfp4")
    impl = _impl(**impl_kwargs)
    q = torch.empty(1, 8, 40, impl.head_size, dtype=dtype, device=device)
    assert _precision(q, impl, **call_kwargs) == expected


@pytest.mark.parametrize(
    "precision, expected",
    [("mxfp4", "bf16"), ("fp8_mixed", "bf16"), ("fp8", "fp8"), ("bf16", "bf16")],
)
def test_batch_above_one_keeps_mx_precisions_off(ring_precision, precision, expected):
    if _xpu_count() < 1:
        pytest.skip("needs an XPU")
    ring_precision(precision)
    q = torch.empty(2, 8, 40, HEAD_SIZE, dtype=torch.bfloat16, device="xpu")
    assert _precision(q, _impl()) == expected


@pytest.mark.parametrize("impl_name", ["XPUAttentionImpl", "XPUSageAttention3Impl"])
@pytest.mark.parametrize(
    "prefix, expected",
    [("blocks.33.attn1.impl", "bf16"), ("blocks.330.attn1.impl", "mxfp4")],
)
def test_bf16_blocks_keep_the_bf16_ring_with_fa_and_sage_attn_3(
    monkeypatch, ring_precision, impl_name, prefix, expected
):
    """All self-attention goes over the ring, so sage_v3_bf16_blocks must keep
    those blocks off the MX ring with the default XPU backend (fa) as well as
    with sage_attn_3."""
    if _xpu_count() < 1:
        pytest.skip("needs an XPU")
    pytest.importorskip("sgl_kernel", reason="needs sgl-kernel-xpu")
    from sglang.multimodal_gen.runtime.layers.attention.backends import xpu_backend

    monkeypatch.setattr(xpu_backend, "_check_sage_v3_runs", lambda kind: None)
    _set_backend_config(monkeypatch, {"sage_v3_bf16_blocks": [0, 33]})
    ring_precision("mxfp4")
    impl_cls = {
        "XPUAttentionImpl": xpu_backend.XPUAttentionImpl,
        "XPUSageAttention3Impl": xpu_backend.XPUSageAttention3Impl,
    }[impl_name]
    impl = impl_cls(
        num_heads=40,
        head_size=HEAD_SIZE,
        causal=False,
        softmax_scale=HEAD_SIZE**-0.5,
        num_kv_heads=40,
        prefix=prefix,
    )
    q = torch.empty(1, 8, 40, HEAD_SIZE, dtype=torch.bfloat16, device="xpu")
    assert _precision(q, impl) == expected


def test_ring_across_hosts_uses_torch_ring(ring_precision):
    if _xpu_count() < 1:
        pytest.skip("needs an XPU")
    ring_precision("mxfp4", one_host=False)
    q = torch.empty(1, 8, 40, HEAD_SIZE, dtype=torch.bfloat16, device="xpu")
    assert _precision(q, _impl()) is None


@pytest.fixture
def fake_arenas(monkeypatch):
    """deep_symm stand-in: ring calls allocate an arena per shape, a clear frees all."""
    module = _ring_module()
    held, clears = set(), []
    monkeypatch.setattr(module, "_held_ring_shapes", set())
    monkeypatch.setattr(module, "_request_ring_shapes", set())

    def fake_forward(*, q, k, precision, **kwargs):
        held.add((tuple(q.shape), precision))
        return q

    def fake_clear():
        clears.append(frozenset(held))
        held.clear()

    monkeypatch.setattr(module, "_import_ring_attn_ipc_forward", lambda: fake_forward)
    ring_attn = types.SimpleNamespace(ring_attn_ipc_clear_cache=fake_clear)
    monkeypatch.setitem(
        sys.modules, "deep_symm", types.SimpleNamespace(ring_attn=ring_attn)
    )
    monkeypatch.setitem(sys.modules, "deep_symm.ring_attn", ring_attn)

    def serve(*shapes):
        module.begin_deepsymm_ring_request()
        for seq, precision in shapes:
            q = torch.empty(1, seq, 1, 8)
            module.deepsymm_ring_attn(q, q, q, ring_group=None, precision=precision)
        return set(held)

    return serve, clears


def test_new_shape_frees_previous_arenas_before_allocating(fake_arenas):
    serve, clears = fake_arenas
    a = [(720, "bf16"), (720, "mxfp4")]
    b = [(480, "bf16"), (480, "mxfp4")]
    assert serve(*a) == {((1, 720, 1, 8), "bf16"), ((1, 720, 1, 8), "mxfp4")}
    assert serve(*a) and clears == []
    # B's first ring call frees A's arenas before B allocates: never both sets.
    assert serve(*b) == {((1, 480, 1, 8), "bf16"), ((1, 480, 1, 8), "mxfp4")}
    assert len(clears) == 1
    # A repeated B keeps its arenas.
    serve(*b)
    assert len(clears) == 1


def test_shapes_used_earlier_in_the_request_do_not_trigger_a_clear(fake_arenas):
    serve, clears = fake_arenas
    # Within one request a second shape (e.g. the bf16 blocks next to mxfp4) is
    # allocated next to the first, not instead of it.
    serve((720, "bf16"), (720, "mxfp4"), (720, "bf16"))
    assert clears == []


def test_failed_ring_pass_clears_before_the_next_call(fake_arenas, monkeypatch):
    serve, clears = fake_arenas
    module = _ring_module()
    serve((720, "mxfp4"))
    working_forward = module._import_ring_attn_ipc_forward()
    fail = [True]

    def forward_failing_once(**kwargs):
        if fail.pop() if fail else False:
            raise RuntimeError("IPC relay timed out")
        return working_forward(**kwargs)

    monkeypatch.setattr(
        module, "_import_ring_attn_ipc_forward", lambda: forward_failing_once
    )
    with pytest.raises(RuntimeError):
        serve((720, "mxfp4"))
    assert clears == []
    # The same shape again: the poisoned relay is dropped before the retry.
    serve((720, "mxfp4"))
    assert len(clears) == 1
    serve((720, "mxfp4"))
    assert len(clears) == 1


def test_worker_begins_a_ring_request_before_each_forward(monkeypatch):
    """The release rule needs the worker to mark every request start before the
    forward; without it no later shape change would free the old arenas."""
    from unittest.mock import Mock

    import sglang.multimodal_gen.runtime.managers.gpu_worker as gpu_worker_module
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.platforms import current_platform

    events = []
    monkeypatch.setattr(
        gpu_worker_module,
        "begin_deepsymm_ring_request",
        lambda: events.append("begin"),
    )
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = True
    worker._realtime_sessions = types.SimpleNamespace(attach=Mock())
    worker._release_warmup_pool = Mock()
    worker._record_output_peak_memory = Mock()

    class Stop(Exception):
        pass

    def forward():
        events.append("forward")
        raise Stop

    for _ in range(2):
        with pytest.raises(Stop):
            worker._execute_forward_common(
                types.SimpleNamespace(is_warmup=False, extra={}),
                forward_fn=forward,
                log_reqs=[],
                return_req=False,
                save_output_paths=Mock(),
                error_context="request job",
                propagate_forward_errors=True,
            )
    assert events == ["begin", "forward", "begin", "forward"]


def test_teardown_forgets_ring_state_even_if_the_clear_fails(monkeypatch):
    module = _ring_module()
    monkeypatch.setattr(module, "_ring_group_on_one_host_cache", {123: True})
    monkeypatch.setattr(module, "_held_ring_shapes", {("s",)})
    monkeypatch.setattr(module, "_request_ring_shapes", {("s",)})

    def peer_gone():
        raise TimeoutError("peer did not answer")

    ring_attn = types.SimpleNamespace(ring_attn_ipc_clear_cache=peer_gone)
    monkeypatch.setitem(
        sys.modules, "deep_symm", types.SimpleNamespace(ring_attn=ring_attn)
    )
    monkeypatch.setitem(sys.modules, "deep_symm.ring_attn", ring_attn)
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.distributed.parallel_groups import (
        PROCESS_GROUP,
    )

    # Only the DeepSymm part of the real teardown runs: groups another test in this
    # process may have left registered are hidden from it.
    monkeypatch.setattr(parallel_state, "_clear_srt_tp_group", lambda: None)
    for name in (
        "_TP",
        "_SP",
        "_DP",
        "_CFG",
        "_PP",
        "_REPLICA",
        "_ENCODER_DP",
        "_VAE_DECODE",
        "_DIT",
        "_VAE",
        "_CFG_SP",
    ):
        monkeypatch.setattr(parallel_state, name, None)
    monkeypatch.setattr(PROCESS_GROUP, "ULYSSES_PG", None)
    monkeypatch.setattr(PROCESS_GROUP, "RING_PG", None)

    parallel_state.destroy_model_parallel()
    assert module._ring_group_on_one_host_cache == {}
    assert module._held_ring_shapes == set()
    assert module._request_ring_shapes == set()


def _reference(q, k, v):
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def _ring_worker(rank, world_size, port, seq_per_rank, num_heads):
    # Dump this rank's stack shortly before the parent's timeout if it hangs.
    faulthandler.dump_traceback_later(540, exit=False)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.xpu.set_device(rank)
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        destroy_model_parallel,
        get_sp_group,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        xpu_deepsymm_ring,
    )
    from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args

    set_global_server_args(_make_unit_server_args())
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world_size, ring_degree=world_size
    )
    generator = torch.Generator().manual_seed(1234)
    shape = (1, seq_per_rank * world_size, num_heads, HEAD_SIZE)
    q, k, v = (torch.randn(shape, generator=generator).bfloat16() for _ in range(3))
    local = slice(rank * seq_per_rank, (rank + 1) * seq_per_rank)
    expected = _reference(q[:, local], k, v)
    device = torch.device("xpu", rank)
    for precision, max_rel_l2 in _MAX_REL_L2.items():
        for _ in range(2):
            out = xpu_deepsymm_ring.deepsymm_ring_attn(
                q[:, local].to(device),
                k[:, local].to(device),
                v[:, local].to(device),
                ring_group=get_sp_group().ring_group,
                precision=precision,
            )
        assert out.shape == expected.shape and out.dtype == torch.bfloat16
        rel_l2 = ((out.cpu().float() - expected).norm() / expected.norm()).item()
        print(f"rank {rank} {precision}: rel_l2 {rel_l2:.4f}", flush=True)
        assert rel_l2 < max_rel_l2, (precision, rel_l2)

    # USPAttention must route to the same DeepSymm call, and to the torch ring when
    # off (stubbed: the torch ring itself is covered by the generic ring tests).
    # The backend is selected for the compute dtype; bf16 picks the XPU FA backend.
    torch.set_default_dtype(torch.bfloat16)
    layer = USPAttention(
        num_heads=num_heads,
        head_size=HEAD_SIZE,
        causal=False,
        required_attention_backend=AttentionBackendEnum.FA,
        prefix="blocks.0.attn1",
    )
    torch.set_default_dtype(torch.float32)
    local_qkv = [t[:, local].to(device) for t in (q, k, v)]
    direct = xpu_deepsymm_ring.deepsymm_ring_attn(
        *local_qkv, ring_group=get_sp_group().ring_group, precision="mxfp4"
    )
    from sglang.multimodal_gen.runtime.layers.attention import layer as layer_module

    torch_ring_calls = []

    def torch_ring(query, *args, **kwargs):
        torch_ring_calls.append(query.shape)
        return torch.zeros_like(query)

    layer_module.ring_attn = torch_ring
    for precision in ("mxfp4", None):
        xpu_deepsymm_ring.DEEPSYMM_RING_PRECISION = precision
        with torch.inference_mode(), set_forward_context(0, None, None):
            routed = layer(*local_qkv)
        if precision is not None:
            assert torch.equal(routed, direct), "USPAttention did not route to DeepSymm"
            assert not torch_ring_calls, "DeepSymm on still ran the torch ring"
            continue
        assert torch_ring_calls, "off did not take the torch ring"
    destroy_model_parallel()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("seq_per_rank, num_heads", [(2048, 8), (4725, 40)])
def test_two_rank_ring_matches_full_attention(seq_per_rank, num_heads):
    if _xpu_count() < 2:
        pytest.skip("needs two XPUs")
    pytest.importorskip("deep_symm.ring_attn", reason="needs deep_symm")
    from sglang.srt.utils.network import get_free_port_below_ephemeral

    context = mp.start_processes(
        _ring_worker,
        args=(2, get_free_port_below_ephemeral(), seq_per_rank, num_heads),
        nprocs=2,
        join=False,
    )
    # A rank stuck in a dead collective must fail the test, not hang the suite.
    deadline = time.monotonic() + 600
    try:
        while not context.join(timeout=5):
            assert time.monotonic() < deadline, "two-rank ring test timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
