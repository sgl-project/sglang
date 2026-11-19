"""Tuned cuBLASLt algorithm tables for FlashInfer bmm_fp8, on CPU: the shipped tables'
format and shapes, and when an entry runs, with the device and FlashInfer faked."""

import json
import os
import re
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm import bmm_fp8_cublaslt_algos as algos
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_GB300 = "NVIDIA GB300"
_GB300_TABLE = "device_name=NVIDIA_GB300.json"
_ENTRY_POINTS = algos._flashinfer_entry_points

# Fields of the public Qwen3.5-397B-A17B text config.
_QWEN3_5_397B = dict(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=256,
    linear_num_key_heads=16,
    linear_num_value_heads=64,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    shared_expert_intermediate_size=1024,
)


def _read(name):
    with open(os.path.join(algos._CONFIG_DIR, name)) as f:
        return json.load(f)


def _projections(cfg, tp):
    """(N, K) of each dense projection on one tensor-parallel rank."""
    hidden, head_dim = cfg["hidden_size"], cfg["head_dim"]
    q_heads = 2 * cfg["num_attention_heads"] // tp  # q and its output gate
    kv_heads = max(1, cfg["num_key_value_heads"] // tp)  # replicated when tp > kv heads
    key = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] // tp
    value = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"] // tp
    shared = cfg["shared_expert_intermediate_size"] // tp
    return {
        "self_attn.qkv_proj": ((q_heads + 2 * kv_heads) * head_dim, hidden),
        "self_attn.o_proj": (hidden, cfg["num_attention_heads"] * head_dim // tp),
        "linear_attn.in_proj_qkvz": (2 * key + 2 * value, hidden),
        "linear_attn.out_proj": (hidden, value),
        "shared_expert.gate_up_proj": (2 * shared, hidden),
        "shared_expert.down_proj": (hidden, shared),
    }


class TestShippedTables(CustomTestCase):
    def test_table_is_well_formed(self):
        for name in sorted(os.listdir(algos._CONFIG_DIR)):
            with self.subTest(name=name):
                match = re.fullmatch(r"device_name=(.+)\.json", name)
                self.assertIsNotNone(match, name)
                doc = _read(name)
                self.assertEqual(set(doc), {"meta", "configs"})
                meta = doc["meta"]
                self.assertEqual(meta["device"].replace(" ", "_"), match[1])
                self.assertIsInstance(meta["cublaslt"], int)
                self.assertGreaterEqual(meta["cublaslt"], 10000)
                self.assertRegex(meta["flashinfer"], r"^\d+\.\d+\.\d+\S*$")
                table = algos._parse_table(doc)
                self.assertTrue(table)
                self.assertTrue(all(min(key) > 0 for key in table))
                self.assertEqual(
                    len(table), sum(len(rows) for rows in doc["configs"].values())
                )

    def test_gb300_shapes_are_tp1_projections_of_a_public_checkpoint(self):
        table = algos._parse_table(_read(_GB300_TABLE))
        shapes = {(n, k) for n, k, _ in table}
        tp1 = set(_projections(_QWEN3_5_397B, 1).values())
        self.assertLessEqual(shapes, tp1, shapes - tp1)
        # No shape that exists only as a TP8 shard (for example the TP8 qkv_proj /
        # in_proj_qkvz shape a kernel plugin may pin with its own algorithms).
        tp8_only = set(_projections(_QWEN3_5_397B, 8).values()) - tp1
        self.assertIn((2560, 4096), tp8_only)
        self.assertEqual(shapes & tp8_only, set())


class _FakeFlashInfer:
    """FlashInfer's GEMM module on CPU: bmm_fp8_run_with_algo writes the stock product."""

    def __init__(self):
        self.algos = []
        self.refuse = False
        self.perturb = False

    def bmm_fp8_run_with_algo(self, a, b, out, x_scale, w_scale, workspace, algo, i):
        if self.refuse:
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")
        self.algos.append(bytes(algo[i * 64 : (i + 1) * 64].tolist()))
        out.copy_(_product(a[0], b[0], x_scale, w_scale).unsqueeze(0))
        if self.perturb:
            out.view(torch.int16)[0, 0, 0] ^= 1


def _product(a, w, x_scale, w_scale):
    return ((a.float() @ w.float()) * x_scale * w_scale).to(torch.bfloat16)


_KEY = (4096, 1024, 196)  # (N, K, M) of a GB300 entry


def _inputs(n=_KEY[0], k=_KEY[1], m=_KEY[2], dtype=torch.float8_e4m3fn):
    g = torch.Generator().manual_seed(0)
    a = (torch.randn(m, k, generator=g) * 3).to(dtype)
    w = (torch.randn(n, k, generator=g) * 0.05).to(dtype).t()
    return a, w, torch.tensor([0.0123]), torch.tensor([0.0021])


def _run(inputs, stock, out_dtype=torch.bfloat16):
    a, w, x_scale, w_scale = inputs
    return algos.run_with_tuned_algo(
        q_input=a,
        weight=w,
        x_scale=x_scale,
        weight_scale=w_scale,
        out_dtype=out_dtype,
        stock=stock,
    )


def _no_stock():
    raise AssertionError("the stock path ran")


def _fake_gemm_base(per_stream_workspace):
    """flashinfer.gemm.gemm_base with the attributes the tuned path reads."""
    module = SimpleNamespace(bmm_fp8_run_with_algo=object())

    class Runner:
        def forward(self, inputs, tactic=-1):
            return module.bmm_fp8_run_with_algo

    runner = Runner()
    gemm_base = types.ModuleType("flashinfer.gemm.gemm_base")
    gemm_base.get_gemm_module = lambda: SimpleNamespace(
        cublas_fp8_gemm_runner=lambda: runner
    )
    gemm_base.DEFAULT_WORKSPACE_SIZE = 1 << 20
    gemm_base.calls = []

    def get_buf(*args):
        gemm_base.calls.append(args)
        return "buffer"

    if per_stream_workspace:
        gemm_base._get_stream_cache_buf = get_buf
    else:
        gemm_base._get_cache_buf = get_buf
    return module, gemm_base


def _install(gemm_base):
    """A patch that installs the fake gemm_base as flashinfer.gemm.gemm_base."""
    flashinfer = types.ModuleType("flashinfer")
    gemm = types.ModuleType("flashinfer.gemm")
    gemm.gemm_base = gemm_base
    flashinfer.gemm = gemm
    return patch.dict(
        sys.modules,
        {
            "flashinfer": flashinfer,
            "flashinfer.gemm": gemm,
            "flashinfer.gemm.gemm_base": gemm_base,
        },
    )


class _Gb300Case(CustomTestCase):
    """The GB300 table with its versions found, the device and FlashInfer faked."""

    def setUp(self):
        meta = _read(_GB300_TABLE)["meta"]
        self.fake = _FakeFlashInfer()
        self.state = {
            "capturing": False,
            "versions": (meta["cublaslt"], meta["flashinfer"]),
            "info": [],
        }
        state = self.state
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in (
            ("_tables", {}),
            ("_validated", set()),
            ("_refused", set()),
            ("_logged", set()),
            ("_device_name", lambda device: state.get("name", _GB300)),
            ("_library_versions", lambda major: state["versions"]),
            (
                "_flashinfer_entry_points",
                lambda: (self.fake.bmm_fp8_run_with_algo, lambda device: None),
            ),
            ("log_info_on_rank0", lambda logger, msg: state["info"].append(msg)),
        ):
            stack.enter_context(patch.object(algos, name, value))
        stack.enter_context(
            patch.object(
                torch.cuda, "is_current_stream_capturing", lambda: state["capturing"]
            )
        )
        self.stack = stack


class TestWhenAnEntryRuns(_Gb300Case):
    def test_entry_runs_after_an_eager_byte_equal_check(self):
        inputs = _inputs()
        ref = _product(*inputs)
        stock_calls = []

        def stock():
            stock_calls.append(1)
            return ref.clone()

        got = _run(inputs, stock)
        self.assertIsNotNone(got)
        self.assertTrue(torch.equal(got.view(torch.int16), ref.view(torch.int16)))
        self.assertEqual(stock_calls, [1])
        table = algos._parse_table(_read(_GB300_TABLE))
        self.assertEqual(self.fake.algos, [table[_KEY]])
        # Checked once: later calls, also inside graph capture, skip the stock path.
        self.state["capturing"] = True
        got = _run(inputs, _no_stock)
        self.assertTrue(torch.equal(got.view(torch.int16), ref.view(torch.int16)))
        hits = [m for m in self.state["info"] if "in use for N=4096 K=1024" in m]
        self.assertEqual(len(hits), 1, self.state["info"])

    def test_first_call_inside_capture_runs_stock(self):
        self.state["capturing"] = True
        self.assertIsNone(_run(_inputs(), _no_stock))
        self.assertEqual(self.fake.algos, [])

    def test_mismatch_returns_stock_output_and_drops_the_shape(self):
        inputs = _inputs()
        ref = _product(*inputs)
        self.fake.perturb = True
        self.assertIs(_run(inputs, lambda: ref), ref)
        self.fake.perturb = False
        self.assertIsNone(_run(inputs, _no_stock))

    def test_refused_algorithm_runs_stock(self):
        self.fake.refuse = True
        self.assertIsNone(_run(_inputs(), _no_stock))
        self.fake.refuse = False
        self.assertIsNone(_run(_inputs(), _no_stock))


class TestCallsOutsideTheTable(_Gb300Case):
    def test_calls_outside_the_table_run_stock(self):
        a, w, x_scale, w_scale = _inputs()
        cublaslt, flashinfer = self.state["versions"]
        bf16 = torch.bfloat16
        cases = {
            "other device": (dict(name="NVIDIA B300 SXM6 AC"), _inputs(), bf16),
            "other cuBLASLt": (
                dict(versions=(cublaslt + 1, flashinfer)),
                _inputs(),
                bf16,
            ),
            "other FlashInfer": (
                dict(versions=(cublaslt, flashinfer + ".post1")),
                _inputs(),
                bf16,
            ),
            "cuBLASLt not loaded": (dict(versions=(None, flashinfer)), _inputs(), bf16),
            "row count not in table": ({}, _inputs(m=_KEY[2] + 1), bf16),
            "shape not in table": ({}, _inputs(n=_KEY[0] + 8), bf16),
            "e5m2 input": ({}, _inputs(dtype=torch.float8_e5m2), bf16),
            "fp16 output": ({}, _inputs(), torch.float16),
            "row-major weight": ({}, (a, w.contiguous(), x_scale, w_scale), bf16),
        }
        found = dict(self.state)
        for name, (state, inputs, out_dtype) in cases.items():
            with self.subTest(name):
                self.state.clear()
                self.state.update(found, **state)
                for cache in (algos._tables, algos._validated, algos._refused):
                    cache.clear()
                self.fake.algos.clear()
                self.assertIsNone(_run(inputs, _no_stock, out_dtype))
                self.assertIsNone(_run(inputs, _no_stock, out_dtype))
                self.assertEqual(self.fake.algos, [])

    def test_bad_meta_is_not_loaded(self):
        doc = _read(_GB300_TABLE)
        doc["meta"]["cublaslt"] = str(doc["meta"]["cublaslt"])
        tmp = self.stack.enter_context(tempfile.TemporaryDirectory())
        with open(os.path.join(tmp, _GB300_TABLE), "w") as f:
            json.dump(doc, f)
        self.stack.enter_context(patch.object(algos, "_CONFIG_DIR", tmp))
        self.assertIsNone(_run(_inputs(), _no_stock))
        self.assertEqual(algos._tables, {None: None})


class TestFlashInferEntryPoints(_Gb300Case):
    def test_flashinfer_without_the_stock_workspace_loads_no_table(self):
        _, gemm_base = _fake_gemm_base(per_stream_workspace=False)
        with (
            _install(gemm_base),
            patch.object(algos, "_flashinfer_entry_points", _ENTRY_POINTS),
        ):
            self.assertIsNone(_run(_inputs(), _no_stock))
        self.assertEqual(algos._tables, {None: None})
        self.assertEqual(gemm_base.calls, [])
        info = self.state["info"]
        self.assertTrue(any("does not expose" in m for m in info), info)

    def test_a_tuned_hit_leaves_flashinfers_runner_unchanged(self):
        """A caller that runs FlashInfer's cuBLAS FP8 runner with its own algorithm
        keeps it: the tuned path only reads FlashInfer, it patches nothing there."""
        module, gemm_base = _fake_gemm_base(per_stream_workspace=True)
        module.bmm_fp8_run_with_algo = self.fake.bmm_fp8_run_with_algo
        runner = gemm_base.get_gemm_module().cublas_fp8_gemm_runner()
        before = (
            runner.forward.__func__,
            gemm_base.get_gemm_module,
            vars(module).copy(),
        )
        inputs = _inputs()
        with (
            _install(gemm_base),
            patch.object(algos, "_flashinfer_entry_points", _ENTRY_POINTS),
        ):
            self.assertIsNotNone(_run(inputs, lambda: _product(*inputs)))
        self.assertEqual(len(self.fake.algos), 1)
        after = (
            gemm_base.get_gemm_module().cublas_fp8_gemm_runner().forward.__func__,
            gemm_base.get_gemm_module,
            vars(module),
        )
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
