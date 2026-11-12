"""prims_ts over the 352 B/token NVFP4 DSA pool, on CPU: the shared Prims-TS
launcher's plan and run arguments (FlashInfer stubbed), the per-forward lengths and
causal indices of the NVFP4 route, its dispatch, and the pool/argument checks."""

import sys
import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import overrides
from sglang.srt.arg_groups.overrides import (
    _check_dsa_backend_constraints,
    _prims_ts_nvfp4_backends,
)
from sglang.srt.layers.attention.dsa import prims_ts
from sglang.srt.layers.attention.dsa.prims_ts import (
    _EXACT_PLAN_MAX_ROWS,
    PrimsTSSparseMLA,
    make_prims_ts_causal_indices,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
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


class TestPrimsTSCausalIndices(unittest.TestCase):
    def test_causal_rows_take_physical_slots_up_to_their_length(self):
        table = torch.tensor([[91, 37, 19, 0], [103, 82, 5, 6]], dtype=torch.int32)
        got = make_prims_ts_causal_indices(
            table,
            torch.tensor([0, 1, 1], dtype=torch.int32),
            torch.tensor([1, 2, 3], dtype=torch.int32),
            4,  # one DP padding row
        )
        want = torch.full((4, 2048), -1, dtype=torch.int32)
        want[0, :1] = torch.tensor([91])
        want[1, :2] = torch.tensor([103, 82])
        want[2, :3] = torch.tensor([103, 82, 5])
        self.assertTrue(torch.equal(got, want))

    def test_rejects_causal_length_beyond_topk(self):
        table = torch.zeros((1, 4096), dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            make_prims_ts_causal_indices(
                table,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([2049], dtype=torch.int32),
                1,
            )


def _nvfp4_backend():
    backend = SimpleNamespace(
        device="cpu",
        is_dsa_prims_nvfp4=True,
        _prims_ts_live_rows=0,
        _prims_ts_lengths={},
        _prims_ts_graph_rows=set(),
    )
    backend._prims_ts_nvfp4_lengths = MethodType(
        DeepseekSparseAttnBackend._prims_ts_nvfp4_lengths, backend
    )
    return backend


class TestPrimsTSNvfp4Lengths(_NoCapture):
    def test_live_rows_get_full_width_and_padding_rows_zero(self):
        backend = _nvfp4_backend()
        lengths = backend._prims_ts_nvfp4_lengths(4, 3, for_graph=True)
        self.assertEqual(lengths.tolist(), [2048, 2048, 2048, 0])

    def test_unchanged_live_count_does_not_rewrite_and_keeps_the_address(self):
        backend = _nvfp4_backend()
        lengths = backend._prims_ts_nvfp4_lengths(4, 3, for_graph=True)
        lengths[0] = 7  # a rewrite would restore 2048
        again = backend._prims_ts_nvfp4_lengths(4, 3)
        self.assertIs(again, lengths)
        self.assertEqual(again[0].item(), 7)
        # A new live count (graph replay with more padding) updates in place.
        self.assertIs(backend._prims_ts_nvfp4_lengths(4, 1), lengths)
        self.assertEqual(lengths.tolist(), [2048, 0, 0, 0])

    def test_eager_shapes_are_evicted_but_graph_shapes_kept(self):
        backend = _nvfp4_backend()
        graph = backend._prims_ts_nvfp4_lengths(8, 8, for_graph=True)
        backend._prims_ts_nvfp4_lengths(300, 300)
        backend._prims_ts_nvfp4_lengths(500, 499)
        self.assertEqual(set(backend._prims_ts_lengths), {8, 500})
        self.assertIs(backend._prims_ts_lengths[8][0], graph)

    def test_rejects_invalid_live_count(self):
        backend = _nvfp4_backend()
        with self.assertRaises(ValueError):
            backend._prims_ts_nvfp4_lengths(4, 5)


class TestPrimsTSNvfp4Forward(_NoCapture):
    def test_casts_q_and_hands_the_pool_ids_and_lengths_to_the_launcher(self):
        backend = _nvfp4_backend()
        backend._prims_ts_live_rows = 2
        cache = torch.zeros((64, 1, 352), dtype=torch.uint8)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _layer_id: cache
        )
        casts = []
        backend._cast_prims_ts_query = lambda n, r, out: casts.append((n, r, out))
        calls = []

        def launch(**kwargs):
            calls.append(kwargs)
            q = kwargs["q"]
            return torch.zeros((q.shape[0], q.shape[1], 512)), None

        backend._prims_ts = SimpleNamespace(forward=launch)
        layer = SimpleNamespace(
            v_head_dim=512, head_dim=576, tp_q_head_num=4, scaling=0.25, layer_id=3
        )
        q_nope = torch.zeros((3, 4, 512), dtype=torch.bfloat16)
        q_rope = torch.zeros((3, 4, 64), dtype=torch.bfloat16)
        ids = torch.zeros((3, 2048), dtype=torch.int32)
        out = DeepseekSparseAttnBackend._forward_prims_ts_nvfp4(
            backend, q_nope, q_rope, layer, ids
        )

        self.assertEqual(out.shape, (3, 4, 512))
        (call,) = calls
        self.assertIs(call["indices"], ids)
        self.assertIs(call["kv_cache"], cache)
        self.assertEqual(call["lengths"].tolist(), [2048, 2048, 0])
        self.assertEqual((call["sm_scale"], call["kv_scale"]), (0.25, 1.0))
        self.assertEqual(call["q"].dtype, torch.float8_e4m3fn)
        (cast,) = casts
        self.assertIs(cast[2], call["q"])


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

    def test_nvfp4_pool_cannot_be_shared_with_other_backends(self):
        for backends in (("prims_ts", "flashmla_kv"), ("flashmla_sparse", "trtllm")):
            with self.assertRaisesRegex(ValueError, "prefill and decode"):
                self._cache_dim(*backends)


class TestPrimsTSNvfp4Arguments(unittest.TestCase):
    def _view(self, **overrides):
        values = dict(
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            attention_backend="dsa",
            page_size=64,
            disable_radix_cache=True,
            disable_prefill_cuda_graph=True,
        )
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
            self._view(disable_prefill_cuda_graph=False),
            self._view(page_size=1),
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

    def test_backend_constraints(self):
        for dtype in ("fp8_e4m3", "nvfp4"):
            _check_dsa_backend_constraints(dtype, "prims_ts", "prims_ts", hip=False)
        with self.assertRaisesRegex(ValueError, "fp8_e4m3 or nvfp4"):
            _check_dsa_backend_constraints("auto", "prims_ts", "prims_ts", hip=False)
        with self.assertRaises(ValueError):
            _check_dsa_backend_constraints("nvfp4", "prims_ts", "prims_ts", hip=True)
        with self.assertRaisesRegex(ValueError, "read only by prims_ts"):
            _check_dsa_backend_constraints(
                "nvfp4", "flashmla_sparse", "trtllm", hip=False
            )

    def test_module_exports(self):
        self.assertEqual(prims_ts.PRIMS_TS_KV_FORMATS["nvfp4.fp8rope"], 352)


if __name__ == "__main__":
    unittest.main()
