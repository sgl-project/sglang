# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU contracts for dormant DCP promotion; no CuTe import or CUDA qualification.

Execute the real polling bodies with small instruction shims: a missing peer must
reach the shared deadline helper before a launch can hang. Driver capacity and
workspace metadata shims verify host propagation, not GPU scheduling or numerics.
"""

from __future__ import annotations

import ast
import copy
import math
import sys
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.kimi_k3_source_paths import source_definition, source_path

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _definition(filename, *names):
    return copy.deepcopy(source_definition("cutedsl/mla/" + filename, *names))


def _function(filename, names, namespace=None):
    node = _definition(filename, *names)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {} if namespace is None else namespace
    exec(  # noqa: S102 - execute only the reviewed local AST bodies on CPU
        compile(
            ast.fix_missing_locations(module),
            str(source_path("cutedsl/mla/" + filename, *names)),
            "exec",
        ),
        namespace,
    )
    return namespace[node.name]


def _calls(node, name):
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and (
            isinstance(item.func, ast.Name)
            and item.func.id == name
            or isinstance(item.func, ast.Attribute)
            and item.func.attr == name
        )
    ]


@pytest.mark.parametrize("deadline", [False, True, 0, -1, 1.5, math.inf, 1 << 63])
def test_dcp_deadline_rejects_nonpositive_noninteger_and_overflow(deadline):
    require = _function("query.py", ("_require_deadline_ns",))
    with pytest.raises(ValueError, match="deadline_ns"):
        require(deadline)
    assert require(1) == 1


@pytest.mark.parametrize("site", ["query", "lse", "partial"])
def test_missing_remote_payload_reaches_shared_deadline(site):
    observed = []

    def start(deadline):
        observed.append(("start", deadline))
        return 17

    def check(started, polls, deadline):
        assert started == 17
        observed.append(("check", deadline))
        if polls == 2:
            raise TimeoutError("simulated device fail-stop")
        return polls + 1

    namespace = {
        "Int64": int,
        "Int32": int,
        "Uint32": int,
        "Float32": float,
        "cutlass": SimpleNamespace(range_constexpr=range),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: (0, 0, 0),
                shuffle_sync_bfly=lambda value, shift: value,
                sync_threads=lambda: None,
            ),
            math=SimpleNamespace(exp2=lambda value, **kwargs: 1.0),
        ),
        "wait_deadline_start": start,
        "wait_deadline_check": check,
        "published": lambda words, sentinel=0x80000000: all(
            w != sentinel for w in words
        ),
        "Q_SENT": 0xFFFFFFFF,
        "LSE_SENT": 0xFFFFFFFF,
        "EMPTY_LSE": -3e38,
        "PAIR_SENT": 0x80000000,
    }
    if site == "query":
        namespace["load_volatile"] = lambda source, **kwargs: [0xFFFFFFFF] * 4
        poll = _function("query.py", ("MlaQuery", "gathered"), namespace)
        owner = SimpleNamespace(
            grid=(8, 12, 1), batch=1, dcp=8, heads=12, cluster=8, deadline_ns=123
        )
        args = [None] * 11 + [SimpleNamespace(iterator=0)]

        def invoke():
            return poll(owner, args, 0)
    else:
        box, lbox, gate = object(), object(), object()
        namespace.update(
            at=lambda base, *coords: (base, coords),
            load_words=lambda *args, **kwargs: [0],
            load_volatile=lambda source, **kwargs: (
                [0xFFFFFFFF if site == "lse" else 0]
                if source[0] is lbox
                else [0x80000000] * 4
            ),
            unpack_bf16=lambda word: (0.0, 0.0),
            store_words=lambda *args: None,
            bits_float=lambda word: 0.0,
            warp_sum=lambda value: value,
            zeros=lambda count, dtype: [0.0] * count,
        )
        merge = _function("dcp.py", ("DcpValue", "merge"), namespace)
        args = [None] * 14
        args[3], args[4], args[9] = box, lbox, gate
        owner = SimpleNamespace(size=8, batch=1, deadline_ns=123)

        def invoke():
            return merge(owner, args, None, {}, None, 0)

    with pytest.raises(TimeoutError, match="fail-stop"):
        invoke()
    assert observed == [("start", 123)] + [("check", 123)] * 3


def test_all_dcp_remote_while_loops_and_compile_keys_retain_deadline():
    sites = [
        ("query.py", ("MlaQuery", "gathered"), 1),
        ("dcp.py", ("DcpValue", "merge"), 2),
    ]
    for filename, names, count in sites:
        node = _definition(filename, *names)
        loops = [item for item in ast.walk(node) if isinstance(item, ast.While)]
        assert len(loops) == count
        assert all(len(_calls(loop, "wait_deadline_check")) == 1 for loop in loops)
        assert len(_calls(node, "wait_deadline_start")) == count
    for filename, names in [
        ("query.py", ("MlaQuery", "__init__")),
        ("dcp.py", ("DcpBox", "__init__")),
        ("dcp.py", ("DcpValue", "__init__")),
        ("dcp.py", ("compile_value",)),
        ("prepared.py", ("compile_core",)),
        ("prepared.py", ("PreparedMla", "__init__")),
    ]:
        node = _definition(filename, *names)
        assert "deadline_ns" in {
            arg.arg for arg in node.args.args + node.args.kwonlyargs
        }
    core = _definition("prepared.py", "compile_core")
    assert any(isinstance(d, ast.Name) and d.id == "cache" for d in core.decorator_list)
    for callee in ("MlaQuery", "compile_value"):
        assert all(
            any(k.arg == "deadline_ns" for k in call.keywords)
            for call in _calls(core, callee)
        )
    prepared = _definition("prepared.py", "PreparedMla", "prepare_cache")
    assert "self.deadline_ns" in ast.unparse(prepared)
    assert "self.kv_shards" in ast.unparse(prepared)
    assert "self.kv_rank" in ast.unparse(prepared)


def test_native_query_default_does_not_require_dcp_deadline():
    require = _function("query.py", ("_require_deadline_ns",))
    init = _function(
        "query.py",
        ("MlaQuery", "__init__"),
        {"TILED_FROM": {12: 4, 6: 6}, "_require_deadline_ns": require},
    )
    native = SimpleNamespace()
    init(native, 12, 1, kv_shards=8, kv_rank=3)
    assert (native.deadline_ns, native.kv_shards, native.kv_rank) == (0, 8, 3)
    with pytest.raises(ValueError, match="deadline_ns"):
        init(SimpleNamespace(), 12, 1, dcp=8)
    with pytest.raises(ValueError, match="mailbox geometry"):
        init(SimpleNamespace(), 12, 1, dcp=8, deadline_ns=123)
    fused = SimpleNamespace()
    init(fused, 12, 1, dcp=8, kv_shards=8, kv_rank=3, deadline_ns=123)
    assert (fused.deadline_ns, fused.kv_shards, fused.kv_rank) == (123, 8, 3)


@pytest.mark.parametrize("capacity", [11, 12])
def test_query_residency_uses_compiled_cluster_capacity(capacity):
    calls = []
    configs = []

    def clusters(function, config):
        configs.append(config)
        return (capacity,)

    cuda = SimpleNamespace(
        CUfunction_attribute=SimpleNamespace(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=1
        ),
        CUlaunchAttributeID=SimpleNamespace(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION=2),
        CUlaunchConfig=SimpleNamespace,
        CUlaunchAttribute=lambda: SimpleNamespace(
            value=SimpleNamespace(clusterDim=SimpleNamespace())
        ),
        cuFuncGetAttribute=lambda *args: (4096,),
        cuOccupancyMaxActiveClusters=clusters,
    )
    check = _function(
        "prepared.py",
        ("_check_dcp_query_residency",),
        {
            "math": math,
            "cuda": cuda,
            "_checked": lambda values, *args: values,
            "check_residency": lambda *args: calls.append(args),
            "resolve_kernel": lambda *args: (None, "compiled-function"),
            "stream": lambda device: "device-stream",
        },
    )
    query = SimpleNamespace(grid=(8, 12, 1), cluster=8)
    if capacity < 12:
        with pytest.raises(RuntimeError, match="clusters cannot be co-resident"):
            check("compiled", query, 0)
    else:
        check("compiled", query, 0)
    assert calls == [("compiled", 0, 96, 256, "Kimi DCP query")]
    config = configs[0]
    assert (config.gridDimX, config.gridDimY, config.gridDimZ) == query.grid
    assert (config.blockDimX, config.blockDimY, config.blockDimZ) == (256, 1, 1)
    assert config.sharedMemBytes == 4096
    assert config.attrs[0].value.clusterDim.x == 8
    core = _definition("prepared.py", "compile_core")
    assert len(_calls(core, "_check_dcp_query_residency")) == 1
    value = _definition("dcp.py", "compile_value")
    assert len(_calls(value, "check_residency")) == 1


def test_workspace_preparation_shares_runtime_metadata_and_refuses_capture_growth(
    monkeypatch,
):
    metadata_calls, allocations = [], []
    capturing = [False]

    def sizing(batch, *args):
        metadata_calls.append((batch, args))
        return 4, batch * 100000

    md = SimpleNamespace(
        select_mla_decode_tilers=lambda *args, **kwargs: ((64, 64), None),
        get_mla_decode_fold_sq_factor=lambda *args: 1,
        get_num_sm=lambda device: 148,
        _get_split_kv_and_workspace_size=sizing,
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_mla",
        SimpleNamespace(mla_decode=md, get_num_sm=md.get_num_sm),
    )

    def empty(size, **kwargs):
        allocations.append(size)
        return SimpleNamespace(numel=lambda: size)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_capability=lambda device: (10, 3),
            is_current_stream_capturing=lambda: capturing[0],
        ),
        device=lambda device: device,
        int8="int8",
        float8_e4m3fn="fp8",
        empty=empty,
    )
    namespace = {"torch": torch, "_WORKSPACE": {}}
    for name in (
        "_decode_metadata",
        "_reserve_workspace",
        "fill_splits",
        "prepare_workspace",
    ):
        _function("dcp.py", (name,), namespace)
    workspace = namespace["prepare_workspace"]("cuda:0", 12, 8192, batches=(1, 8, 64))
    assert allocations == [6400000]
    assert [batch for batch, _ in metadata_calls] == [1, 8, 64]
    assert [args[-1] for _, args in metadata_calls] == [74, 9, 1]
    assert all(args[:5] == (1, 12, 512, 148, 8192) for _, args in metadata_calls)
    assert namespace["_reserve_workspace"]("cuda:0", 12, 1, 148) is workspace
    capturing[0] = True
    with pytest.raises(RuntimeError, match="before capture"):
        namespace["_reserve_workspace"]("cuda:0", 12, 6400001, 148)
    prepare = _definition("dcp.py", "prepare_workspace")
    runtime = _definition("dcp.py", "stock_decode")
    assert len(_calls(prepare, "_decode_metadata")) == 1
    assert len(_calls(runtime, "_decode_metadata")) == 1
    assert not _calls(prepare, "_get_compiled_mla_kernel")
    assert not _calls(prepare, "stock_decode")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
