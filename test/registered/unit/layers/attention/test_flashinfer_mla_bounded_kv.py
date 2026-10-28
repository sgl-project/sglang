"""FP8 MLA row planning and dispatch without FlashInfer or a CUDA device."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_backend():
    # Only GPU dependency boundaries are stubbed. Execute the complete backend
    # module so the tests exercise the real metadata builders and forward paths.
    dependencies = {
        "sglang.kernels.ops.attention.utils": {"assert_buffer_fits": Mock()},
        "sglang.srt.layers.attention.base_attn_backend": {"AttentionBackend": object},
        "sglang.srt.layers.dcp": {
            "DecodeContextParallelMetadata": object,
            "update_local_kv_lens_for_dcp": Mock(),
        },
        "sglang.srt.layers.dcp.planner": {"plan_dcp_decode_metadata": Mock()},
        "sglang.srt.mem_cache.memory_pool": {"KVWriteLoc": Mock()},
        "sglang.srt.model_executor.forward_batch_info": {
            "ForwardBatch": object,
            "ForwardMode": Mock(),
        },
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": {
            "is_in_breakable_cuda_graph": lambda: False,
        },
        "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph": {
            "is_in_tc_piecewise_cuda_graph": lambda: False,
        },
        "sglang.srt.speculative.spec_info": {
            "SpecInput": object,
            "SpecInputType": Mock(),
        },
        "sglang.srt.speculative.spec_utils": {
            "draft_kv_indices_buffer_width": Mock(),
            "draft_kv_indices_used_len": Mock(),
            "generate_draft_decode_kv_indices": Mock(),
        },
    }
    stubs = {}
    for name, attrs in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        stubs[name] = module
    path = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/attention/flashinfer_mla_backend.py"
    )
    spec = importlib.util.spec_from_file_location("_mla_bounded_kv_test", path)
    backend = importlib.util.module_from_spec(spec)
    stubs[spec.name] = backend
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(backend)
    return backend


class TestFlashInferMLABoundedKV(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_backend()

    def setUp(self):
        self.parallel = SimpleNamespace(dcp_enabled=False)
        self.enterContext(
            patch.object(self.module, "get_parallel", return_value=self.parallel)
        )
        self.enterContext(
            patch.object(
                self.module,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(flashinfer_mla_disable_ragged=True)
                ),
            )
        )

    def _backend(self, dtype=torch.float8_e4m3fn, page_size=1, rows=None):
        backend = self.module.FlashInferMLAAttnBackend.__new__(
            self.module.FlashInferMLAAttnBackend
        )
        buffers = [torch.arange(256 * 8).remainder(31).view(256, 1, 8).to(dtype)]
        backend.token_to_kv_pool = SimpleNamespace(
            dtype=dtype,
            size=256 - page_size,
            page_size=page_size,
            get_key_buffer=lambda layer_id: buffers[layer_id],
        )
        reader_type = self.module._Fp8KvRowReader
        backend.fp8_kv_reader = (
            reader_type(backend.token_to_kv_pool)
            if reader_type.enabled(backend.token_to_kv_pool, torch.bfloat16)
            else None
        )
        backend.skip_prefill = False
        backend.causal = True
        backend.decode_wrapper = Mock()
        backend.prefill_wrapper_paged = Mock()
        backend.prefill_wrapper_ragged = Mock()
        backend.prefill_wrapper_verify = Mock()
        backend.decode_cuda_graph_metadata = {}
        backend.kv_indptr = torch.zeros(5, dtype=torch.int32)
        backend.q_indptr_decode = torch.arange(5, dtype=torch.int32)
        backend.qo_indptr = torch.zeros(5, dtype=torch.int32)
        backend.cuda_graph_kv_indices = torch.zeros(32, dtype=torch.int32)
        if rows is None:
            rows = torch.tensor([7, 2, 7, 0, 19], dtype=torch.int32)
        backend.kv_index_translator = SimpleNamespace(
            pack_read_stream=lambda *args, out, **kwargs: out.copy_(rows) is not None,
            needs_read_translate=False,
        )
        config = SimpleNamespace(
            num_attention_heads=1,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            scaling=0.5,
        )
        runner = SimpleNamespace(model_config=config, dtype=torch.bfloat16)
        with patch.object(
            self.module,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
        ):
            backend.indices_updater_decode = (
                self.module.FlashInferMLAIndicesUpdaterDecode(runner, backend)
            )
            backend.indices_updater_prefill = (
                self.module.FlashInferMLAIndicesUpdaterPrefill(runner, backend)
            )
        return backend

    def _batch(self, *, decode=False, verify=False, draft=False, spec_info=None):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: decode,
                is_decode=lambda: decode,
                is_target_verify=lambda: verify,
                is_draft_extend_v2=lambda: draft,
            ),
            spec_info=spec_info,
            seq_lens=torch.tensor([3, 2], dtype=torch.int32),
            seq_lens_sum=5,
            batch_size=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            kv_loc_plan=object(),
            extend_prefix_lens=torch.tensor([1, 0], dtype=torch.int32),
            extend_prefix_lens_cpu=[1, 0],
            seq_lens_cpu=None,
            attn_dcp_metadata=None,
            attn_attend_prefix_cache=None,
        )

    def test_decode_remaps_only_eager_nonspeculative_plans(self):
        backend = self._backend()
        updater = backend.indices_updater_decode
        rows = torch.tensor([7, 2, 7, 0, 19], dtype=torch.int32)
        # Isolate the CUDA allocation boundary while retaining real plan logic.
        original_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)

        with patch.object(torch, "empty", side_effect=cpu_empty):
            backend.init_forward_metadata(self._batch(decode=True))
        torch.testing.assert_close(backend.forward_metadata.kv_rows, rows)
        torch.testing.assert_close(
            backend.decode_wrapper.plan.call_args.args[2],
            torch.arange(5, dtype=torch.int32),
        )
        backend.decode_wrapper.run.side_effect = lambda q, qr, k, kr, **kw: q
        layer = SimpleNamespace(layer_id=0, tp_q_head_num=1, v_head_dim=4, head_dim=8)
        backend.forward_decode(
            torch.zeros(2, 1, 8, dtype=torch.bfloat16),
            None,
            None,
            layer,
            self._batch(decode=True),
        )
        kv_inputs = backend.decode_wrapper.run.call_args.args[2:4]
        expected = backend.token_to_kv_pool.get_key_buffer(0).to(torch.bfloat16)[
            rows.long()
        ]
        torch.testing.assert_close(
            torch.cat(kv_inputs, dim=-1), expected, rtol=0, atol=0
        )
        spec = SimpleNamespace(
            kv_indices=rows, kv_indptr=torch.tensor([0, 3, 5], dtype=torch.int32)
        )
        result = updater.update(
            torch.tensor([3, 2]),
            5,
            backend.decode_wrapper,
            spec_info=spec,
            plan=object(),
            req_pool_indices=torch.tensor([0, 1]),
            compact_fp8_kv=True,
        )
        self.assertIsNone(result)
        self.assertIs(backend.decode_wrapper.plan.call_args.args[2], rows)

    def test_compact_memory_boundary_preserves_plan_and_forward_rows(self):
        for decode in (False, True):
            for count in (170, 171):
                with self.subTest(decode=decode, count=count):
                    rows = torch.arange(count, dtype=torch.int32).remainder(23)
                    backend = self._backend(rows=rows)
                    batch = self._batch(decode=decode)
                    batch.seq_lens = torch.tensor([count - 2, 2], dtype=torch.int32)
                    batch.seq_lens_sum = count
                    original_empty = torch.empty

                    def cpu_empty(*args, **kwargs):
                        kwargs["device"] = "cpu"
                        return original_empty(*args, **kwargs)

                    with patch.object(torch, "empty", side_effect=cpu_empty):
                        backend.init_forward_metadata(batch)
                    compact = backend.forward_metadata.kv_rows
                    wrapper = (
                        backend.decode_wrapper
                        if decode
                        else backend.prefill_wrapper_paged
                    )
                    plan_ids = wrapper.plan.call_args.args[2]
                    if count == 170:
                        torch.testing.assert_close(compact, rows)
                        torch.testing.assert_close(
                            plan_ids, torch.arange(count, dtype=torch.int32)
                        )
                    else:
                        self.assertIsNone(compact)
                        torch.testing.assert_close(plan_ids, rows)
                    kv = backend._mla_kv_buffer(
                        SimpleNamespace(layer_id=0), compact, torch.bfloat16
                    )
                    full = backend.token_to_kv_pool.get_key_buffer(0).to(torch.bfloat16)
                    torch.testing.assert_close(
                        kv[plan_ids.long()], full[rows.long()], rtol=0, atol=0
                    )

    def test_dtype_and_capacity_guards(self):
        reader_type = self.module._Fp8KvRowReader
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            self.assertTrue(
                reader_type.enabled(SimpleNamespace(dtype=dtype), torch.bfloat16)
            )
            self.assertFalse(reader_type.enabled(SimpleNamespace(dtype=dtype), dtype))
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.uint8):
            self.assertFalse(
                reader_type.enabled(SimpleNamespace(dtype=dtype), torch.bfloat16)
            )
        reader = self._backend(page_size=64).fp8_kv_reader
        self.assertTrue(reader.use_compact(170))
        self.assertFalse(reader.use_compact(171))
        self.assertFalse(reader.use_compact(256))
        reader.pool_rows = 255
        self.assertTrue(reader.use_compact(170))
        self.assertFalse(reader.use_compact(171))
        reader.pool_rows = 0
        self.assertFalse(reader.use_compact(1))

    def test_prefill_fallbacks_keep_physical_ids(self):
        for kind in (
            "bf16",
            "oversized",
            "dcp",
            "dcp_global",
            "draft",
            "spec",
            "verify",
        ):
            with self.subTest(kind=kind):
                backend = self._backend(
                    dtype=torch.bfloat16 if kind == "bf16" else torch.float8_e4m3fn
                )
                if kind == "oversized":
                    backend.fp8_kv_reader.pool_rows = 4
                batch = self._batch(
                    draft=kind == "draft",
                    spec_info=object() if kind == "spec" else None,
                    verify=kind == "verify",
                )
                if kind == "dcp":
                    batch.attn_dcp_metadata = SimpleNamespace(
                        dcp_kv_indptr=None, dcp_kv_indices=None
                    )
                self.parallel.dcp_enabled = kind == "dcp_global"
                if kind == "verify":
                    # Verify's input owns its own plan, outside the compact path.
                    backend.indices_updater_prefill = Mock()
                backend.init_forward_metadata(batch)
                self.assertIsNone(backend.forward_metadata.kv_rows)
                if kind != "verify":
                    torch.testing.assert_close(
                        backend.prefill_wrapper_paged.plan.call_args.args[2],
                        torch.tensor([7, 2, 7, 0, 19], dtype=torch.int32),
                    )
                else:
                    self.assertFalse(
                        backend.indices_updater_prefill.update.call_args.kwargs.get(
                            "compact_fp8_kv", False
                        )
                    )
        self.parallel.dcp_enabled = False

    def test_graph_dispatch_requires_registered_wrapper_and_nonspeculative_batch(self):
        for kind in (
            "graph",
            "eager",
            "spec",
            "draft",
            "dcp",
            "sm80_e4m3",
            "sm80_e5m2",
        ):
            with self.subTest(kind=kind):
                with patch.object(
                    self.module,
                    "use_fp8_e4b15_for_e4m3fn",
                    return_value=kind.startswith("sm80"),
                ):
                    backend = self._backend(
                        dtype=torch.float8_e5m2
                        if kind == "sm80_e5m2"
                        else torch.float8_e4m3fn
                    )
                wrapper = Mock()
                wrapper.run.side_effect = lambda q, qr, k, kr, **kw: q
                backend.forward_metadata = self.module.DecodeMetadata(wrapper)
                if kind != "eager":
                    backend.decode_cuda_graph_metadata[2] = wrapper
                backend.skip_prefill = kind == "draft"
                self.parallel.dcp_enabled = kind == "dcp"
                batch = self._batch(
                    decode=True, spec_info=object() if kind == "spec" else None
                )
                layer = SimpleNamespace(
                    layer_id=0, tp_q_head_num=1, v_head_dim=4, head_dim=8
                )
                with patch.object(
                    backend.fp8_kv_reader,
                    "refresh",
                    return_value=backend.token_to_kv_pool.get_key_buffer(0).to(
                        torch.bfloat16
                    ),
                ) as refresh:
                    backend.forward_decode(
                        torch.zeros(2, 1, 8, dtype=torch.bfloat16),
                        None,
                        None,
                        layer,
                        batch,
                    )
                self.assertEqual(
                    refresh.call_count, int(kind in ("graph", "sm80_e5m2"))
                )
                if kind in ("graph", "sm80_e5m2"):
                    self.assertIs(
                        refresh.call_args.args[1], backend.cuda_graph_kv_indices
                    )
                    self.assertEqual(
                        refresh.call_args.args[2].data_ptr(),
                        backend.kv_indptr[2:3].data_ptr(),
                    )
        self.parallel.dcp_enabled = False


if __name__ == "__main__":
    unittest.main()
