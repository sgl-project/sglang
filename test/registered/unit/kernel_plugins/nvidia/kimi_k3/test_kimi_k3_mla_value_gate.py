# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU contracts for the opt-in FlashInfer MLA value gate; no CuTe or FlashInfer import.

The real PreparedMla bodies run with recording kernels: this checks the selection,
fallback, binding and the tensors handed to the runner, not GPU numerics.
"""

import ast
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.mla import flashinfer_value
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.kimi_k3_source_paths import source_definition

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SLOT = "k3_flashinfer_mla_value"


def _w_uv():
    return torch.empty(12, 128, 512, dtype=torch.bfloat16).transpose(1, 2)


def _flashinfer(prepare):
    module = SimpleNamespace(prepare_mla_value_gate=prepare)
    return {
        "flashinfer": types.ModuleType("flashinfer"),
        "flashinfer.cake_mla_value_gate": module,
    }


def _prepared_mla(names, scope):
    definition = ast.ClassDef(
        name="PreparedMla",
        bases=[],
        keywords=[],
        body=[
            source_definition("cutedsl/mla/prepared.py", "PreparedMla", name)
            for name in names
        ],
        decorator_list=[],
    )
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "PreparedMla", "exec"), scope)
    return scope["PreparedMla"]


def _gate(supported=True):
    """Patch the ops.attention device check; FlashInfer itself is a sys.modules stub."""
    from sglang.kernels.ops import attention as attention_ops

    return patch.object(
        attention_ops, "can_use_cake_mla_value_gate", return_value=supported
    )


def test_value_gate_is_opt_in_and_bucket_refusals_keep_mla_value():
    w_uv, calls = _w_uv(), []

    def prepare(weight, **kwargs):
        calls.append((weight, kwargs))
        if kwargs["max_rows"] == 3:
            raise ValueError("w_uv must start on a 16-byte boundary")
        return SimpleNamespace(program="program", arch="sm_103a")

    with (
        patch.object(flashinfer_value, "logger") as logger,
        patch.dict(flashinfer_value._REFUSED, clear=True),
        patch.dict(sys.modules, _flashinfer(prepare)),
    ):
        with (
            envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override("k3_native_mla"),
            _gate() as gate,
        ):
            assert flashinfer_value.prepare_value_gate(w_uv, rows=8, q=1) is None
        gate.assert_not_called()
        assert calls == []
        with envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override(f"k3_native_mla,{SLOT}"):
            with _gate():
                runner = flashinfer_value.prepare_value_gate(w_uv, rows=8, q=1)
                assert runner.program == "program"
                expected = dict(
                    max_rows=8,
                    sigmoid_precision="fp32",
                    rows_per_request=1,
                    enable_pdl=True,
                )
                assert calls == [(w_uv, expected)]
                # A weight the kernel refuses costs one bucket, not the device.
                assert flashinfer_value.prepare_value_gate(w_uv, rows=3, q=1) is None
                (message,), _ = logger.warning_once.call_args
                assert "using MlaValue" in message
                assert flashinfer_value.prepare_value_gate(w_uv, rows=8, q=1)
                assert len(calls) == 3 and flashinfer_value._REFUSED == {}


@pytest.mark.parametrize("refusal", ["unsupported", "missing", "build", "capture"])
def test_device_refusal_is_remembered_so_buckets_do_not_retry(refusal):
    """A failed JIT build or load must not be retried for every layer and bucket."""
    w_uv, calls = _w_uv(), []

    def prepare(weight, **kwargs):
        calls.append(kwargs["max_rows"])
        raise RuntimeError("ninja: build stopped")

    modules = _flashinfer(prepare)
    if refusal == "missing":
        modules = {"flashinfer.cake_mla_value_gate": None}
    capturing = refusal == "capture"
    with (
        patch.object(flashinfer_value, "logger") as logger,
        patch.dict(flashinfer_value._REFUSED, clear=True),
        patch.dict(sys.modules, modules),
        envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override(SLOT),
        _gate(refusal != "unsupported") as gate,
        patch.object(torch.cuda, "is_available", return_value=capturing),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=capturing),
    ):
        for rows in (1, 8, 64):
            assert flashinfer_value.prepare_value_gate(w_uv, rows=rows, q=1) is None
    if capturing:
        # Prepare refused under capture says nothing about the device.
        assert calls == [1, 8, 64] and flashinfer_value._REFUSED == {}
        assert logger.warning_once.call_count == 3
    else:
        assert gate.call_count == 1
        assert calls == ([1] if refusal == "build" else [])
        logger.warning.assert_called_once()
        (message,), _ = logger.warning.call_args
        assert "using MlaValue" in message


def _binding_scope(prepared):
    def prepare(weight, *, rows, q):
        prepared.append((weight.data_ptr(), rows, q))
        return ("runner", weight.data_ptr(), rows, q)

    return dict(
        torch=torch,
        cutlass=SimpleNamespace(Float32=float, Int32=int),
        core_for=lambda *args: "trtllm",
        to_cute=lambda tensor, *args: tensor,
        layout=lambda tensor: (tuple(tensor.shape), tuple(tensor.stride())),
        compile_core=lambda *args: (object(), object()),
        prepare_value_gate=prepare,
        _require_deadline_ns=lambda deadline: deadline,
        local_max=lambda length, size: length,
    )


def _attention():
    return SimpleNamespace(
        q_b_proj=SimpleNamespace(weight=torch.empty(2304, 1536, dtype=torch.bfloat16)),
        q_a_layernorm=SimpleNamespace(weight=torch.ones(1536, dtype=torch.bfloat16)),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(512, dtype=torch.bfloat16)),
        W_UK_T=torch.empty(12, 128, 512, dtype=torch.bfloat16),
        W_UV=torch.nn.Parameter(_w_uv(), requires_grad=False),
        _q_scale_inv=torch.ones(1),
        _k_scale_inv=torch.ones(1),
        _q_scale_float=1.0,
        _k_scale_float=1.0,
        rms_norm_eps=1.0e-5,
        scale=192.0**-0.5,
    )


@pytest.mark.parametrize("native", [False, True])
def test_each_bucket_prepares_once_on_the_bound_w_uv_and_rebinding_reprepares(native):
    prepared = []
    cls = _prepared_mla(("__init__", "prepare_cache"), _binding_scope(prepared))
    attention = _attention()
    pitches = {1: 3648, 8: 3648, 64: 3648}
    core = cls(attention, pitches, torch.empty(64, 12, 576), native_mla=native)
    kv = torch.empty(1, 64, 576, dtype=torch.uint8)
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=148),
    ):
        core.prepare_cache(kv)
        first = attention.W_UV.data_ptr()
        assert sorted(prepared) == [(first, rows, 1) for rows in pitches]
        assert core.value_gates == {
            (rows, 1): ("runner", first, rows, 1) for rows in pitches
        }
        # A backend may move the weight's storage after construction; binding again
        # must prepare against the new storage instead of keeping stale runners.
        attention.W_UV.data = _w_uv()
        second = attention.W_UV.data_ptr()
        assert second != first
        prepared.clear()
        core.prepare_cache(kv)
    assert sorted(prepared) == [(second, rows, 1) for rows in pitches]
    assert {runner[1] for runner in core.value_gates.values()} == {second}


def test_dcp_prepares_no_flashinfer_runner():
    prepared = []
    names = ("__init__", "prepare_cache", "prepare_dcp")
    cls = _prepared_mla(names, _binding_scope(prepared))
    box = SimpleNamespace(
        deadline_ns=7,
        size=8,
        rank=3,
        heads=12,
        cap=64,
        q=torch.zeros(2, 1),
        mc=lambda parity: 0,
        rows=torch.zeros(64, 1),
        rows_lse=torch.zeros(64, 1),
        partial=torch.zeros(2, 1),
        lse=torch.zeros(2, 1),
        peers=torch.zeros(1),
        lse_peers=torch.zeros(1),
    )
    core = cls(
        _attention(),
        {1: 3648, 8: 3648},
        torch.empty(64, 12, 576),
        dcp=box,
        kv_shards=8,
        kv_rank=3,
    )
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=148),
    ):
        core.prepare_cache(torch.empty(1, 64, 576, dtype=torch.uint8))
    assert prepared == [] and core.value_gates == {}
    assert sorted(core.dcp_args) == [1, 8]


class _Runner:
    def __init__(self, trace):
        self.trace, self.args = trace, None

    def launch(self, *args):
        self.trace.append("flashinfer-value")
        self.args = args


def _run(core_kwargs, **run_kwargs):
    fn = source_definition("cutedsl/mla/prepared.py", "PreparedMla", "run")
    pad = source_definition("cutedsl/mla/prepared.py", "pad_requests")

    def refuse(*args, **kwargs):
        raise AssertionError("run() must not prepare a runner")

    scope = dict(
        torch=torch,
        stream=lambda device: "stream",
        to_cute=lambda x, *args: x,
        prepare_value_gate=refuse,
    )
    module = ast.Module(body=[pad, fn], type_ignores=[])
    exec(compile(module, "PreparedMla.run", "exec"), scope)
    return scope["run"](SimpleNamespace(**core_kwargs), **run_kwargs)


def _core(trace, batch, runner, **kwargs):
    def record(name):
        return lambda *args, **kw: trace.append((name, args))

    return (
        dict(
            native_mla=False,
            device=torch.device("cpu"),
            kernels={(batch, 1): (record("query"), record("cute-value"), None, None)},
            value_gates={(batch, 1): runner},
            views=(None,) * 7,
            eps=None,
            query_scratch=torch.zeros(batch, 12, 576, dtype=torch.uint8),
            latent=torch.zeros(64, 12, 512, dtype=torch.bfloat16),
            stock_core=record("stock-core"),
            dcp=None,
        )
        | kwargs
    )


@pytest.mark.parametrize("native", [False, True])
def test_runner_reads_mla_value_tensors_with_the_same_padded_lengths(native):
    batch = 4
    projected = torch.arange(batch * 3648, dtype=torch.float32).view(batch, 3648)
    out = torch.empty(batch, 1536)
    # Two live requests; PIECEWISE padding adds two empty ones (length 0).
    lengths = torch.tensor([5, -1], dtype=torch.int32)
    run_kwargs = dict(
        cache=torch.zeros(1, 64, 576, dtype=torch.uint8),
        slots=torch.arange(batch),
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        lengths=lengths,
    )
    if native:
        run_kwargs["native_decode"] = lambda *args: None
    seen = {}
    for name, runner_factory in (("mla", lambda trace: None), ("fi", _Runner)):
        trace = []
        runner = runner_factory(trace)
        core = _core(trace, batch, runner, native_mla=native)
        assert _run(core, projected=projected, out=out, **run_kwargs) is out
        names = [item if isinstance(item, str) else item[0] for item in trace]
        if name == "mla":
            assert names[-1] == "cute-value"
            seen[name] = trace[-1][1]
        else:
            assert names[-1] == "flashinfer-value" and "cute-value" not in names
            seen[name] = runner.args
    latent, uv, gate, lens, mla_out, stream = seen["mla"]
    fi_latent, fi_gate, fi_lens, fi_out = seen["fi"]
    assert fi_out is out and mla_out is out
    assert fi_latent.data_ptr() == core["latent"].data_ptr()
    assert tuple(fi_latent.shape) == (batch, 12, 512)
    assert (fi_gate.data_ptr(), fi_gate.shape, fi_gate.stride()) == (
        gate.data_ptr(),
        gate.shape,
        gate.stride(),
    )
    assert fi_gate.data_ptr() == projected[:, 2112:].data_ptr()
    assert fi_gate.stride() == (3648, 1)
    assert fi_lens.tolist() == lens.tolist() == [5, -1, 0, 0]


def test_dcp_keeps_dcp_value_even_with_a_runner():
    batch, trace = 2, []
    runner = _Runner(trace)
    core = _core(trace, batch, runner, dcp=SimpleNamespace())
    core["dcp_args"] = {
        batch: (
            ("mailbox", "multicast", 3),
            (("rows", "lse"), ("partial", "lse"), ("peers", "lse_peers")),
            (0, 3),
        )
    }
    totals = torch.tensor([9, 9], dtype=torch.int32)
    out = torch.empty(batch, 1536)
    _run(
        core,
        projected=torch.zeros(batch, 3648),
        out=out,
        cache=torch.zeros(1, 64, 576, dtype=torch.uint8),
        slots=torch.arange(batch),
        block_table=torch.zeros(batch, 1, dtype=torch.int32),
        lengths=torch.tensor([4, 4], dtype=torch.int32),
        totals=totals,
    )
    names = [item if isinstance(item, str) else item[0] for item in trace]
    assert names == ["query", "stock-core", "cute-value"]
    assert runner.args is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
