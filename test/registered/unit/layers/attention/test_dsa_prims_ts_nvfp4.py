"""prims_ts over the 352 B/token NVFP4 DSA pool, on CPU: the Prims-TS launcher's
plan and run arguments (FlashInfer stubbed), the pool row size and the server
argument resolution."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import overrides
from sglang.srt.arg_groups.overrides import (
    _PRIMS_TS_NVFP4_UNSUPPORTED,
    _prims_ts_nvfp4_backends,
)
from sglang.srt.layers.attention.dsa.prims_ts import (
    _EXACT_PLAN_MAX_ROWS,
    PrimsTSSparseMLA,
)
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FP4 = object()  # stands in for torch.float4_e2m1fn_x2, which needs CUDA here


class _FakeWrapper:
    """Records plan() and run() of flashinfer's BatchSparseMLADecodePagedTSWrapper."""

    instances = []

    def __init__(self):
        self.plan_args = None
        self.runs = []
        _FakeWrapper.instances.append(self)

    def plan(self, *args, **kwargs):
        self.plan_args = (args, kwargs)

    def run(self, q, kv, indices, lengths, **kwargs):
        self.runs.append(dict(q=q, kv=kv, indices=indices, lengths=lengths, **kwargs))
        kwargs["out"].fill_(1)


def _fake_flashinfer():
    module = ModuleType("flashinfer.attention.prims_ts")
    module.BatchSparseMLADecodePagedTSWrapper = _FakeWrapper
    return {
        "flashinfer": ModuleType("flashinfer"),
        "flashinfer.attention": ModuleType("flashinfer.attention"),
        "flashinfer.attention.prims_ts": module,
    }


class _NoCapture(unittest.TestCase):
    """CPU-only torch raises from torch.cuda.is_current_stream_capturing()."""

    def setUp(self):
        capturing = patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        )
        capturing.start()
        self.addCleanup(capturing.stop)


class TestPrimsTSLauncher(_NoCapture):
    def setUp(self):
        super().setUp()
        _FakeWrapper.instances.clear()
        self.modules = patch.dict(sys.modules, _fake_flashinfer())
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def _run(self, runner, rows, cache):
        q = torch.zeros((rows, 4, 576), dtype=torch.float8_e4m3fn)
        indices = torch.full((rows, 2048), -1, dtype=torch.int32)
        lengths = torch.full((rows,), 2048, dtype=torch.int32)
        out, lse = runner.forward(
            q=q,
            kv_cache=cache,
            indices=indices,
            lengths=lengths,
            sm_scale=0.5,
            kv_scale=1.0,
        )
        return out, lse, indices, lengths

    def test_nvfp4_plans_the_352_byte_format_and_passes_ids_in_place(self):
        runner = PrimsTSSparseMLA(torch.device("cpu", 0), "nvfp4.fp8rope")
        cache = torch.zeros((128, 1, 352), dtype=torch.uint8)
        out, _, indices, lengths = self._run(runner, 3, cache)

        (wrapper,) = _FakeWrapper.instances
        args, kwargs = wrapper.plan_args
        self.assertEqual(args[1:], (3, 4))
        self.assertEqual(kwargs["kv_cache_format"], "nvfp4.fp8rope")
        self.assertEqual(kwargs["q_dtype"], torch.float8_e4m3fn)
        self.assertEqual((kwargs["max_q_len"], kwargs["packed_query"]), (1, False))
        (run,) = wrapper.runs
        self.assertEqual(run["kv"].shape, (128, 352))
        self.assertEqual(run["kv"].dtype, torch.uint8)
        self.assertEqual(run["kv"].data_ptr(), cache.data_ptr())
        # The indexer's ids and the lengths are read in place, not rewritten.
        self.assertIs(run["indices"], indices)
        self.assertIs(run["lengths"], lengths)
        self.assertEqual((run["softmax_scale"], run["output_scale"]), (0.5, 1.0))
        self.assertFalse(run["validate"])
        self.assertEqual(out.shape, (3, 4, 512))

    def test_plans_are_reused_and_large_eager_batches_pack(self):
        runner = PrimsTSSparseMLA(torch.device("cpu", 0), "nvfp4.fp8rope")
        cache = torch.zeros((64, 1, 352), dtype=torch.uint8)
        self._run(runner, 3, cache)
        self._run(runner, 3, cache)
        self.assertEqual(len(_FakeWrapper.instances), 1)

        rows = _EXACT_PLAN_MAX_ROWS + 1
        self._run(runner, rows, cache)
        self._run(runner, rows + 7, cache)  # same power-of-two capacity
        self.assertEqual(len(_FakeWrapper.instances), 2)
        packed = _FakeWrapper.instances[1]
        _, kwargs = packed.plan_args
        self.assertEqual((kwargs["max_q_len"], kwargs["packed_query"]), (2048, True))
        self.assertEqual(packed.runs[-1]["qo_indptr"].tolist(), [0, rows + 7])

    def test_native_format_keeps_the_fp8_view(self):
        runner = PrimsTSSparseMLA(torch.device("cpu", 0))
        cache = torch.zeros((64, 1, 576), dtype=torch.uint8)
        self._run(runner, 2, cache)
        (wrapper,) = _FakeWrapper.instances
        self.assertEqual(wrapper.plan_args[1]["kv_cache_format"], "native")
        self.assertEqual(wrapper.runs[0]["kv"].dtype, torch.float8_e4m3fn)
        self.assertEqual(wrapper.runs[0]["kv"].shape, (64, 576))

    def test_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            PrimsTSSparseMLA(torch.device("cpu", 0), "nvfp4")


class TestPrimsTSNvfp4Pool(unittest.TestCase):
    def _cache_dim(self, prefill, decode, dtype=FP4):
        exec_ = SimpleNamespace(
            kernel=SimpleNamespace(
                dsa_prefill_backend=prefill, dsa_decode_backend=decode
            )
        )
        model_config = SimpleNamespace(
            hf_config=None, kv_lora_rank=512, qk_rope_head_dim=64
        )
        with (
            patch.object(kv_cache_configurator, "get_exec", return_value=exec_),
            patch.object(
                kv_cache_configurator,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch.object(kv_cache_configurator, "is_deepseek_dsa", lambda _: True),
            patch.object(
                kv_cache_configurator, "is_float4_e2m1fn_x2", lambda d: d is FP4
            ),
        ):
            return kv_cache_configurator.calculate_mla_kv_cache_dim(
                model_config=model_config, kv_cache_dtype=dtype
            )

    def test_nvfp4_rows_are_352_bytes(self):
        self.assertEqual(self._cache_dim("prims_ts", "prims_ts"), 352)
        # FP8 prims_ts keeps the raw 576 B layout.
        self.assertEqual(
            self._cache_dim("prims_ts", "prims_ts", torch.float8_e4m3fn), 576
        )


class TestPrimsTSNvfp4Arguments(unittest.TestCase):
    def _view(self, **overrides):
        values = dict(
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            attention_backend="dsa",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            disable_radix_cache=True,
            speculative_algorithm=None,
            dllm_algorithm=None,
            disaggregation_mode="null",
            dcp_size=1,
            attn_cp_size=1,
        )
        for name in _PRIMS_TS_NVFP4_UNSUPPORTED:
            values[name] = False
        values.update(overrides)
        return SimpleNamespace(**values)

    def _hf(self, arch="GlmMoeDsaForCausalLM"):
        return SimpleNamespace(architectures=[arch])

    def _resolve(self, view, hf=None, major=10):
        platform = SimpleNamespace(is_hip=False)
        with patch.object(overrides, "get_platform", return_value=platform):
            return _prims_ts_nvfp4_backends(view, hf or self._hf(), major)

    def test_tested_configuration_pins_prims_ts(self):
        self.assertEqual(
            self._resolve(self._view()),
            {"dsa_prefill_backend": "prims_ts", "dsa_decode_backend": "prims_ts"},
        )
        explicit = self._view(
            dsa_prefill_backend="prims_ts", dsa_decode_backend="prims_ts"
        )
        self.assertEqual(self._resolve(explicit), {})

    def test_rejects_other_backends_and_unsupported_modes(self):
        for view in (
            self._view(dsa_decode_backend="flashmla_kv"),
            self._view(disable_radix_cache=False),
            self._view(speculative_algorithm="EAGLE"),
            self._view(dcp_size=2),
            self._view(enable_hisparse=True),
        ):
            with self.assertRaises(ValueError):
                self._resolve(view)
        with self.assertRaises(ValueError):
            self._resolve(self._view(), major=9)
        with self.assertRaises(ValueError):
            self._resolve(self._view(), hf=self._hf("DeepseekV32ForCausalLM"))


if __name__ == "__main__":
    unittest.main()
