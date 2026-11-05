# SPDX-License-Identifier: Apache-2.0
"""CPU counterexamples for native MLA admission, slot ownership and dispatch.

Kernel method bodies are extracted without importing CuTe on CPU. This checks
the changed branch and integer mapping, not CUDA compilation or arithmetic.
"""

import ast
import importlib
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.kernel_plugins.nvidia.kimi_k3.profile import native_config_reasons
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.kimi_k3_source_paths import source_definition

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def method(path, cls, name):
    return source_definition(path, cls, name)


def test_native_profile_admits_1m_dcp8_without_weakening_other_modes():
    config = dict(
        context_length=1048576,
        dcp_size=8,
        attn_cp_size=1,
        enable_prefill_cp=False,
        nnodes=1,
        dcp_replicate_q_proj=True,
    )
    assert native_config_reasons(SimpleNamespace(**config)) == []
    for key, value in (
        ("context_length", 1048577),
        ("dcp_size", 4),
        ("attn_cp_size", 8),
        ("enable_prefill_cp", True),
        ("enable_lmcache", True),
        ("nnodes", 2),
    ):
        assert native_config_reasons(SimpleNamespace(**(config | {key: value})))


def test_native_spec_is_explicit_and_refuses_an_unextended_framework():
    from sglang.kernels.plugin_kernels.nvidia.spec import Feature
    from sglang.srt.kernel_plugins.nvidia.kimi_k3 import spec

    try:
        with patch.dict(
            os.environ, {"SGLANG_NVIDIA_MEGAKERNEL_SLOTS": "k3_native_mla"}
        ):
            importlib.reload(spec)
            assert spec.NATIVE_MLA
            assert spec.SPEC.requires.resident_blocks == 148
            assert spec.SPEC.requires.server_args["decode_attention_backend"] == {
                "trtllm_mla",
                "cutedsl_mla",
            }
            assert Feature.KV_OFFLOAD in spec.SPEC.features
            assert Feature.SPECULATIVE_DECODING not in spec.SPEC.features
            assert "language_only" not in spec.SPEC.requires.hf_config
            config = SimpleNamespace(
                context_length=1048576,
                dcp_size=8,
                attn_cp_size=1,
                enable_prefill_cp=False,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full")
                ),
            )
            assert spec._config_disable_reasons(config, None) == []
            from sglang.srt.kernel_plugins.nvidia.binding import MegakernelBinding

            with patch.object(MegakernelBinding, "native_mla_api_version", 0):
                assert "API v1" in spec._config_disable_reasons(config, None)[0]
            config.decode_attention_backend = "cutedsl_mla"
            with patch.object(
                MegakernelBinding,
                "native_mla_supports_backend",
                None,
                create=True,
            ):
                assert any(
                    "backend capability" in reason
                    for reason in spec._config_disable_reasons(config, None)
                )
            with patch.object(
                MegakernelBinding,
                "native_mla_supports_backend",
                return_value=True,
                create=True,
            ):
                assert spec._config_disable_reasons(config, None) == []
            assert MegakernelBinding.native_mla_supports_backend("cutedsl_mla")
            assert spec._config_disable_reasons(config, None) == []
    finally:
        importlib.reload(spec)


def test_replicated_q_keeps_local_projection_and_validates_its_native_slice():
    """The model keeps both native layouts; the provider must use the local one."""
    from sglang.srt.kernel_plugins.nvidia.kimi_k3 import providers

    rank = 3
    # Each DCP shard has a distinct value so selecting the wrong rank is visible.
    qrep = (
        torch.repeat_interleave(torch.arange(8, dtype=torch.bfloat16), 2304)[:, None]
        .expand(-1, 1536)
        .contiguous()
    )
    krep = (
        torch.repeat_interleave(torch.arange(8, dtype=torch.bfloat16), 12)[
            :, None, None
        ]
        .expand(-1, 128, 512)
        .contiguous()
    )
    qb = qrep.narrow(0, rank * 2304, 2304).clone()
    wk = torch.empty(12, 512, 128, dtype=torch.bfloat16).transpose(1, 2)
    wk.copy_(krep.narrow(0, rank * 12, 12))
    wv = torch.empty(12, 128, 512, dtype=torch.bfloat16).transpose(1, 2)
    attention = SimpleNamespace(
        q_b_proj=SimpleNamespace(weight=qb),
        w_kc=wk,
        w_vc=wv,
        q_b_proj_qrep_weight=qrep,
        w_kc_qrep=krep,
        q_a_layernorm=SimpleNamespace(variance_epsilon=1e-5),
        kv_a_layernorm=object(),
        attn_mqa=SimpleNamespace(q_scale_float=1.0, k_scale_float=1.0),
        scaling=192.0**-0.5,
        w_scale=1.0,
    )
    identities = tuple(id(v) for v in (qb, wk, qrep, krep))
    pointers = tuple(v.data_ptr() for v in (qb, wk, qrep, krep))
    parallel = SimpleNamespace(attn_dcp_size=8, attn_dcp_rank=rank)
    with patch.object(providers, "get_parallel", return_value=parallel):
        proxy, _ = providers._mla_proxy(attention)
        assert proxy.q_b_proj is attention.q_b_proj
        assert proxy.W_UK_T.shape == (12, 128, 512)
        assert torch.equal(proxy.W_UK_T, wk)
        assert identities == tuple(id(v) for v in (qb, wk, qrep, krep))
        assert pointers == tuple(v.data_ptr() for v in (qb, wk, qrep, krep))
        # A full-head replacement is unsafe even if native replication is on.
        attention.q_b_proj = SimpleNamespace(weight=qrep)
        with pytest.raises(ValueError, match="original.*shard"):
            providers._mla_proxy(attention)
        attention.q_b_proj = proxy.q_b_proj
        qrep[rank * 2304, 0] = 99
        with pytest.raises(ValueError, match="differs from its local shard"):
            providers._mla_proxy(attention)
        qrep[rank * 2304, 0] = rank
        attention.w_kc_qrep = None
        with pytest.raises(ValueError, match="incomplete"):
            providers._mla_proxy(attention)
        # No replica is also a supported native fallback; it does not remove
        # the provider's explicit sharded projection/gather contract.
        attention.q_b_proj_qrep_weight = None
        providers._mla_proxy(attention)


def test_prepared_mla_binds_replicated_q_storage_after_backend_initialization():
    """A stable Parameter/shape can hide a new Q storage pointer before capture."""

    def to_cute(tensor, align=16):
        view = SimpleNamespace(
            pointer=tensor.data_ptr(),
            offset=tensor.storage_offset(),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            alignment=align,
        )
        return view

    scope = dict(
        torch=torch,
        cutlass=SimpleNamespace(Float32=float),
        core_for=lambda *args: "trtllm",
        to_cute=to_cute,
        layout=lambda tensor: (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        ),
        compile_core=lambda *args: (object(), object()),
    )
    definition = ast.ClassDef(
        name="PreparedMla",
        bases=[],
        keywords=[],
        body=[
            method("cutedsl/mla/prepared.py", "PreparedMla", name)
            for name in ("__init__", "prepare_cache")
        ],
        decorator_list=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
            "PreparedMla.weight_binding",
            "exec",
        ),
        scope,
    )
    rows, columns, rank = 2304, 1536, 3
    weight = torch.nn.Parameter(
        torch.empty(rows, columns, dtype=torch.bfloat16), requires_grad=False
    )
    original = weight.detach()  # Keep the old allocation distinct from the replica.
    identity, shape, stride = id(weight), tuple(weight.shape), tuple(weight.stride())
    attention = SimpleNamespace(
        q_b_proj=SimpleNamespace(weight=weight),
        q_a_layernorm=SimpleNamespace(weight=torch.ones(columns, dtype=torch.bfloat16)),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(512, dtype=torch.bfloat16)),
        W_UK_T=torch.empty(12, 128, 512, dtype=torch.bfloat16),
        W_UV=torch.empty(12, 512, 128, dtype=torch.bfloat16),
        _q_scale_inv=torch.ones(1),
        _k_scale_inv=torch.ones(1),
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        rms_norm_eps=1.0e-5,
        scale=192.0**-0.5,
    )
    core = scope["PreparedMla"](
        attention,
        {1: 3648},
        torch.empty(1, 12, 576, dtype=torch.uint8),
        native_mla=True,
        kv_shards=8,
        kv_rank=rank,
    )
    replica = torch.empty(8 * rows, columns, dtype=weight.dtype)
    local = replica.narrow(0, rank * rows, rows)
    weight.data = local
    assert (id(weight), tuple(weight.shape), tuple(weight.stride())) == (
        identity,
        shape,
        stride,
    )
    assert weight.data_ptr() != original.data_ptr()
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=148),
    ):
        assert core.prepare_cache(torch.empty(1, 64, 576, dtype=torch.uint8)) is core
    q_view = core.views[2]
    assert core.weights[2] is weight
    assert q_view.pointer == local.data_ptr()
    assert (
        q_view.pointer
        == replica.data_ptr() + rank * rows * columns * weight.element_size()
    )
    assert q_view.offset == rank * rows * columns
    assert (q_view.shape, q_view.stride, q_view.alignment) == (shape, stride, 16)


def test_query_insert_localizes_each_slot_to_exactly_one_dcp_rank():
    # Execute the actual insert method's pre-write slot selection on CPU.
    fn = method("cutedsl/mla/query.py", "MlaQuery", "insert")
    start = next(
        i
        for i, node in enumerate(fn.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "slot"
    )
    nodes = fn.body[start : start + 3]
    code = compile(
        ast.Module(body=nodes, type_ignores=[]),
        "MlaQuery.insert.slot_selection",
        "exec",
    )
    cutlass = SimpleNamespace(Int64=int, const_expr=bool)
    for slot in (-1, 0, 1, 7, 8, 63, 64, 511, 512, 1048575):
        owners = []
        for rank in range(8):
            ns = dict(
                cutlass=cutlass,
                self=SimpleNamespace(kv_shards=8, kv_rank=rank),
                valid=True,
                slots=[slot],
                tok0=0,
                j=0,
            )
            exec(code, ns)
            if ns["slot"] >= 0:
                owners.append((rank, ns["slot"]))
        assert owners == ([] if slot < 0 else [(slot % 8, slot // 8)])


def test_native_prepared_core_does_not_fall_through_or_insert_twice():
    fn = method("cutedsl/mla/prepared.py", "PreparedMla", "run")
    fn.decorator_list = []
    trace = []

    def record(name):
        return lambda *args, **kwargs: trace.append(name)

    scope = dict(
        torch=torch,
        stream=lambda device: None,
        to_cute=lambda x, *args: x,
        pad_requests=lambda batch, table, lens, totals: (table, lens, totals),
    )
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "PreparedMla.run", "exec"),
        scope,
    )
    core = SimpleNamespace(
        native_mla=True,
        device=torch.device("cpu"),
        kernels={(1, 1): (record("insert-query"), record("value"), None, None)},
        views=(None,) * 7,
        eps=None,
        query_scratch=torch.zeros(1, 12, 576, dtype=torch.uint8),
        latent=torch.zeros(1, 12, 512, dtype=torch.bfloat16),
        stock_core=record("unexpected-stock-core"),
        dcp=None,
    )
    args = dict(
        cache=torch.zeros(1, 64, 576, dtype=torch.uint8),
        slots=torch.tensor([8]),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        lengths=torch.tensor([1], dtype=torch.int32),
    )
    out = torch.empty(1, 1536)
    assert (
        scope["run"](
            core,
            torch.zeros(1, 3648),
            out,
            native_decode=record("native-attention"),
            **args,
        )
        is out
    )
    assert trace == ["insert-query", "native-attention", "value"]
    with pytest.raises(RuntimeError, match="no SGLang attention adapter"):
        scope["run"](core, torch.zeros(1, 3648), out, **args)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
