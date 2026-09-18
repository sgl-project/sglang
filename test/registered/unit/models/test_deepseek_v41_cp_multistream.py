"""Scheduling tests for DeepSeek-V4.1 CP multi-stream prepare."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.attention.deepseek_v4_backend as dsv4_backend
import sglang.srt.layers.attention.dsv4.compressor as dsv4_compressor
import sglang.srt.layers.attention.dsv4.compressor_v2 as dsv4_compressor_v2
import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeStream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_stream(self, other):
        self.log.append(f"wait:{self.name}:{other.name}")

    def wait_event(self, event):
        self.log.append(f"wait_event:{self.name}:{event}")

    def record_event(self):
        event = f"event:{self.name}:{len(self.log)}"
        self.log.append(f"record:{event}")
        return event


class _FakeLowRatioLayer:
    def __init__(self, log, current_stream):
        self.log = log
        self.current_stream = current_stream
        self.alt_streams = [
            _FakeStream("kv", log),
            _FakeStream("unused", log),
            _FakeStream("sources", log),
        ]
        self.fuse_wqa_wkv = False
        self.compressor = object()
        self.indexer = object()
        self.layer_id = 3

    def _compute_q_a(self, x, qkv_a=None):
        self.log.append("compute:q_a")
        return x, x

    def _compute_q_b(self, q, positions, q_out=None):
        self.log.append(f"compute:q_b:{self.current_stream().name}")
        return q

    def _materialize_cp_swa_k(self, x, forward_batch, qkv_a=None):
        self.log.append(f"gather:swa:{self.current_stream().name}")
        return torch.full((2, 1), 11.0)

    def _store_cp_swa_k(self, kv, forward_batch, attn_backend):
        self.log.append(f"consume:swa:{self.current_stream().name}")


class _FakeLowRatioBackend:
    def __init__(self, log, current_stream):
        self.log = log
        self.current_stream = current_stream
        self.x_global = None

    def forward_low_ratio_sources(self, **kwargs):
        self.log.append(f"consume:sources:{self.current_stream().name}")
        self.x_global = kwargs["precomputed_x_global"]


class TestDeepseekV41CPMultiStream(CustomTestCase):
    def test_replay_copy_pads_aliased_cp_global_prefix(self):
        backing = torch.arange(8, dtype=torch.int32)
        dsv4_backend._copy_tensor_allowing_storage_alias(
            backing, backing[:4], pad_value=0
        )
        self.assertEqual(backing.tolist(), [0, 1, 2, 3, 0, 0, 0, 0])

    def test_cp_vision_inputs_are_prepared_before_sharding(self):
        image_id = deepseek_v4.MM_PAD_SHIFT_VALUE
        ids = torch.tensor([7, image_id, 8])
        embeddings = torch.ones(3, 4)
        model = SimpleNamespace(
            vision=object(),
            config=SimpleNamespace(image_token_id=99),
            _prepare_mm_embeddings=Mock(return_value=embeddings),
        )
        mode = SimpleNamespace(
            is_decode=lambda: False,
            is_target_verify=lambda: False,
            is_decode_or_idle=lambda: False,
        )
        batch = SimpleNamespace(forward_mode=mode, mm_inputs=[object()])
        result_ids, result_embeddings = (
            deepseek_v4.DeepseekV4ForCausalLM.prepare_language_model_inputs(
                model, ids, batch
            )
        )
        model._prepare_mm_embeddings.assert_called_once_with(ids, batch)
        self.assertEqual(result_ids.tolist(), [7, 99, 8])
        self.assertIs(result_embeddings, embeddings)

    def test_bcg_cp_swa_store_uses_live_global_prefix(self):
        pool = Mock()
        layer = SimpleNamespace(
            layer_id=3, kv_norm=SimpleNamespace(weight=1), eps=1e-6, freqs_cis=2
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(late_layer_tail=None),
            get_swa_out_cache_loc=lambda batch: torch.arange(2),
        )
        batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=2),
            positions=torch.arange(4),
        )
        with patch.object(deepseek_v4, "get_token_to_kv_pool", return_value=pool):
            deepseek_v4.MQALayer._store_cp_swa_k(
                layer, torch.ones(4, 2), batch, backend
            )
        kwargs = pool.set_swa_key_buffer_radix_fused_norm_rope.call_args.kwargs
        self.assertEqual(kwargs["kv"].shape[0], 2)
        self.assertEqual(kwargs["positions"].shape[0], 2)
        self.assertEqual(kwargs["swa_loc"].shape[0], 2)

    def test_eager_cp_prefill_selects_prepare_by_ratio(self):
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate

        class Selected(Exception):
            pass

        mode = SimpleNamespace(
            is_extend=lambda: True,
            is_decode=lambda: False,
            is_target_verify=lambda: False,
            is_decode_or_idle=lambda: False,
        )
        batch = SimpleNamespace(forward_mode=mode)
        x = torch.zeros((1024, 4))  # Larger than the capture-time BS limit.
        for ratio, method_name in (
            (0, "_forward_prepare"),
            (1, "_forward_prepare_low_ratio_multi_stream"),
        ):
            with self.subTest(ratio=ratio):
                selected = Mock(side_effect=Selected)
                layer = SimpleNamespace(
                    is_dsv41=True,
                    dsa_enable_prefill_cp=True,
                    alt_streams=[object(), object(), object()],
                    _multi_stream_bs_limit=128,
                    compress_ratio=ratio,
                    compressor=object() if ratio == 1 else None,
                    _kernel_num_heads=lambda _: 1,
                    n_local_heads=1,
                    head_dim=128,
                    _local_attn_sink=lambda _: None,
                    **{method_name: selected},
                )
                with (
                    patch.object(deepseek_v4, "_is_cuda", True),
                    patch.object(deepseek_v4, "_is_hip", False),
                    patch.object(deepseek_v4, "_is_npu", False),
                    patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=True),
                    patch.object(
                        deepseek_v4, "get_is_capture_mode", return_value=False
                    ),
                    patch.object(
                        deepseek_v4,
                        "get_attn_tp_context",
                        return_value=SimpleNamespace(input_scattered=False),
                    ),
                    patch.object(
                        deepseek_v4,
                        "get_attn_backend",
                        return_value=SimpleNamespace(low_ratio_prefill_graph=False),
                    ),
                    patch.object(
                        deepseek_v4,
                        "get_platform",
                        return_value=SimpleNamespace(is_blackwell=True),
                    ),
                    patch.object(
                        deepseek_v4.envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP,
                        "get",
                        return_value=True,
                    ),
                    patch.object(env_gate, "is_unified_kv_triton", return_value=False),
                ):
                    with self.assertRaises(Selected):
                        deepseek_v4.MQALayer.forward(
                            layer, x, torch.arange(1024), batch
                        )
                selected.assert_called_once()

    def test_collectives_finish_before_low_ratio_worker_streams(self):
        log = []
        main_stream = _FakeStream("main", log)
        active_stream = main_stream

        @contextmanager
        def use_stream(stream):
            nonlocal active_stream
            previous = active_stream
            active_stream = stream
            try:
                yield
            finally:
                active_stream = previous

        def current_stream():
            return active_stream

        def gather_x(value, forward_batch, stream=None):
            log.append(f"gather:x:{current_stream().name}")
            return value + 10

        layer = _FakeLowRatioLayer(log, current_stream)
        backend = _FakeLowRatioBackend(log, current_stream)
        forward_batch = SimpleNamespace(encoder_swa_replay=False)
        x = torch.full((2, 1), 2.0)

        with (
            patch.object(torch.cuda, "current_stream", side_effect=current_stream),
            patch.object(torch.cuda, "stream", side_effect=use_stream),
            patch.object(
                deepseek_v4, "cp_materialize_global_token_order", side_effect=gather_x
            ),
        ):
            output = deepseek_v4.MQALayer._forward_prepare_low_ratio_cp_multi_stream(
                layer,
                x,
                torch.arange(2),
                forward_batch,
                backend,
            )

        self.assertEqual(
            [event for event in log if event.startswith("gather:")],
            ["gather:swa:main", "gather:x:main"],
        )
        last_gather = log.index("gather:x:main")
        self.assertGreater(log.index("consume:swa:kv"), last_gather)
        self.assertGreater(log.index("consume:sources:sources"), last_gather)
        torch.testing.assert_close(backend.x_global, torch.full((2, 1), 12.0))
        torch.testing.assert_close(output, x)

    def test_bcg_cp_indexer_runs_after_captured_worker_join(self):
        log = []
        main_stream = _FakeStream("main", log)
        layer = _FakeLowRatioLayer(log, lambda: main_stream)
        layer.compress_ratio = 1
        indexer_buffers = {"q": torch.empty(2, 1, 1), "w": torch.empty(2, 1)}
        local_pos = torch.arange(2)
        backend = SimpleNamespace(
            low_ratio_prefill_graph=True,
            forward_metadata=SimpleNamespace(low_ratio_pos_i64=local_pos),
            _cp_indexer_projection_buffers=Mock(return_value=indexer_buffers),
            _low_ratio_index_topk_prefill_graph=Mock(
                side_effect=lambda *args: log.append("capture:indexer")
            ),
        )
        with (
            patch.object(torch.cuda, "current_stream", return_value=main_stream),
            patch.object(deepseek_v4, "is_in_breakable_cuda_graph", return_value=True),
            patch.object(
                deepseek_v4,
                "cp_materialize_global_token_order",
                side_effect=lambda x, batch, stream: x,
            ),
            patch.object(
                deepseek_v4,
                "deepseek_v4_low_ratio_cp_begin",
                side_effect=lambda *args: log.append("capture:begin"),
            ),
            patch.object(
                deepseek_v4,
                "deepseek_v4_low_ratio_cp_finish",
                side_effect=lambda *args: log.append("capture:join"),
            ),
        ):
            deepseek_v4.MQALayer._forward_prepare_low_ratio_cp_multi_stream(
                layer,
                torch.ones(2, 1),
                local_pos,
                SimpleNamespace(encoder_swa_replay=False),
                backend,
            )
        self.assertLess(log.index("capture:begin"), log.index("compute:q_b:main"))
        self.assertLess(log.index("compute:q_b:main"), log.index("capture:join"))
        self.assertLess(log.index("capture:join"), log.index("capture:indexer"))
        backend._low_ratio_index_topk_prefill_graph.assert_called_once_with(
            layer, local_pos, indexer_buffers["q"], indexer_buffers["w"]
        )

    def test_backend_reuses_precomputed_global_hidden_states(self):
        precomputed = torch.full((2, 1), 7.0)

        class Backend:
            forward_metadata = SimpleNamespace(
                late_layer_tail=SimpleNamespace(
                    local_lens_cpu=[2],
                    req_global=torch.zeros(2, dtype=torch.int64),
                    pos_global=torch.arange(2, dtype=torch.int64),
                )
            )

            def __init__(self):
                self.compressor_input = None

            def _low_ratio_in_prefill_graph(self):
                return False

            def _low_ratio_compress_torch(self, layer, x, req, pos):
                self.compressor_input = x

        backend = Backend()
        layer = SimpleNamespace(compressor=object(), indexer=None)
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=2)
        )

        with patch.object(
            dsv4_backend,
            "cp_materialize_global_token_order",
            side_effect=AssertionError("unexpected second CP gather"),
        ):
            dsv4_backend.DeepseekV4AttnBackend._forward_low_ratio_sources_cp(
                backend,
                layer=layer,
                x=torch.empty(2, 1),
                q_lora=torch.empty(2, 1),
                positions=torch.arange(2),
                forward_batch=forward_batch,
                run_compressor=True,
                run_indexer=False,
                precomputed_x_global=precomputed,
            )

        self.assertEqual(backend.compressor_input.data_ptr(), precomputed.data_ptr())
        torch.testing.assert_close(backend.compressor_input, precomputed)

    def test_bcg_cp_compressor_uses_bucket_rows_and_global_metadata(self):
        precomputed = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        req = torch.tensor([7, 7, 0, 0, 0, 0], dtype=torch.int64)
        positions = torch.tensor([4, 5, 0, 0], dtype=torch.int64)

        class Backend:
            forward_metadata = SimpleNamespace(
                late_layer_tail=None,
                low_ratio_req_indices=req,
                core_metadata=SimpleNamespace(raw_out_loc=torch.tensor([8, 9, 0, 0])),
            )

            def _low_ratio_in_prefill_graph(self):
                return True

            def _low_ratio_compress_torch(self, layer, x, req_rows, pos_rows):
                self.compressor_input = x
                self.req_rows = req_rows
                self.pos_rows = pos_rows

        backend = Backend()
        layer = SimpleNamespace(compressor=object(), indexer=None)
        batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=2), positions=positions
        )
        with patch.object(
            dsv4_backend,
            "cp_materialize_global_token_order",
            side_effect=AssertionError("unexpected second CP gather"),
        ):
            dsv4_backend.DeepseekV4AttnBackend._forward_low_ratio_sources_cp(
                backend,
                layer=layer,
                x=torch.empty(2, 1),
                q_lora=None,
                positions=torch.arange(2),
                forward_batch=batch,
                run_compressor=True,
                run_indexer=False,
                precomputed_x_global=precomputed,
            )

        torch.testing.assert_close(backend.compressor_input, precomputed[:4])
        torch.testing.assert_close(backend.req_rows, req[:4])
        torch.testing.assert_close(backend.pos_rows, positions)

    def test_bcg_cp_c2_pair_ring_ignores_bucket_padding(self):
        kv = torch.tensor([[4.0], [5.0], [0.0], [0.0]])
        score = torch.ones_like(kv)
        req = torch.tensor([3, 3, 0, 0])
        pos = torch.tensor([5, 6, 0, 0])
        pad = torch.tensor([False, False, True, True])

        class State:
            ring_size = 2

            def translate_from_req_position_to_state_loc(self, req_rows, positions):
                return positions

            def get_state_by_state_loc(self, locations):
                return dsv4_backend.KVAndScore.from_kv_score(
                    kv=torch.full_like(kv, 9.0), score=torch.zeros_like(score)
                )

            def set_state_by_state_loc(self, locations, value):
                self.write_positions = locations

        state = State()
        backend = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                get_attention_compress_states=lambda layer_id: state
            )
        )
        partner_kv, _ = dsv4_backend.DeepseekV4AttnBackend._low_ratio_pair_partners(
            backend, layer_id=1, kv=kv, score=score, req=req, pos=pos, pad=pad
        )
        self.assertEqual(partner_kv[0].item(), 9.0)  # Previous chunk's token 4.
        self.assertEqual(partner_kv[1].item(), kv[0].item())
        self.assertEqual(state.write_positions.tolist(), [5, 6, -1, -1])

    def test_bcg_cp_captured_sources_overlap_q_b_and_join_before_indexer(self):
        log = []
        main_stream = _FakeStream("main", log)
        active_stream = main_stream

        @contextmanager
        def use_stream(stream):
            nonlocal active_stream
            previous = active_stream
            active_stream = stream
            try:
                yield
            finally:
                active_stream = previous

        def current_stream():
            return active_stream

        layer = _FakeLowRatioLayer(log, current_stream)
        backend = _FakeLowRatioBackend(log, current_stream)
        forward_batch = SimpleNamespace(encoder_swa_replay=False)
        x = torch.zeros((2, 1))
        positions = torch.arange(2)

        with (
            patch.object(torch.cuda, "current_stream", side_effect=current_stream),
            patch.object(torch.cuda, "stream", side_effect=use_stream),
            patch.object(
                deepseek_v4,
                "get_tc_piecewise_forward_context",
                return_value=SimpleNamespace(forward_batch=forward_batch),
            ),
            patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
        ):
            deepseek_v4.deepseek_v4_low_ratio_cp_begin(
                layer, x, x, positions, x, x, None
            )
            log.append("compute:q_b:main")
            deepseek_v4.deepseek_v4_low_ratio_cp_finish(layer)
            log.append("capture:indexer")

        self.assertLess(
            log.index("consume:sources:sources"), log.index("compute:q_b:main")
        )
        self.assertLess(log.index("consume:swa:kv"), log.index("compute:q_b:main"))
        self.assertLess(log.index("compute:q_b:main"), log.index("wait:main:sources"))
        self.assertLess(log.index("wait:main:sources"), log.index("capture:indexer"))
        self.assertLess(log.index("wait:main:kv"), log.index("capture:indexer"))

    def test_bcg_cp_static_indexer_projects_bucket_rows(self):
        log = []
        main_stream = _FakeStream("main", log)
        active_stream = main_stream

        @contextmanager
        def use_stream(stream):
            nonlocal active_stream
            old_stream = active_stream
            active_stream = stream
            try:
                yield
            finally:
                active_stream = old_stream

        class Indexer:
            def queries(self, q_lora, freqs):
                return q_lora[:, None, :] + 1

            def head_weights(self, x):
                return x[:, :1] + 2

        layer = _FakeLowRatioLayer(log, lambda: active_stream)
        layer.compressor = None
        layer.indexer = Indexer()
        layer.freqs_cis = torch.zeros(3)
        backend = _FakeLowRatioBackend(log, lambda: active_stream)
        batch = SimpleNamespace(encoder_swa_replay=False)
        x = torch.ones(3, 2)
        q_lora = torch.full((3, 2), 3.0)
        bufs = {"q": torch.full((3, 1, 2), 9.0), "w": torch.full((3, 1), 9.0)}
        with (
            patch.object(
                torch.cuda, "current_stream", side_effect=lambda: active_stream
            ),
            patch.object(torch.cuda, "stream", side_effect=use_stream),
            patch.object(
                deepseek_v4,
                "get_tc_piecewise_forward_context",
                return_value=SimpleNamespace(forward_batch=batch),
            ),
            patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
        ):
            deepseek_v4.deepseek_v4_low_ratio_cp_begin(
                layer, x, q_lora, torch.arange(3), x, x, bufs
            )
            deepseek_v4.deepseek_v4_low_ratio_cp_finish(layer)

        torch.testing.assert_close(bufs["q"], torch.full((3, 1, 2), 4.0))
        torch.testing.assert_close(bufs["w"], torch.full((3, 1), 3.0))
        self.assertFalse(any("consume:indexer" in event for event in log))


class _FakeGenericCompressor:
    compute_kv_score = dsv4_compressor.Compressor.compute_kv_score

    def __init__(self, name, value, log):
        self.name = name
        self.value = value
        self.log = log

    def _compute_wkv_gate(self, x):
        self.log.append(f"project:{self.name}")
        return torch.full((x.shape[0], 1), self.value)


class _FakeGenericIndexer:
    def __init__(self, compressor, log):
        self.compressor = compressor
        self.log = log
        self.precomputed_kv_score = None

    def __call__(self, **kwargs):
        self.log.append("consume:indexer")
        self.precomputed_kv_score = kwargs["precomputed_kv_score"]


class _FakeGenericBackend:
    def __init__(self, log):
        self.log = log
        self.swa_k = None
        self.compressor_kv_score = None

    def forward_core_compressor(
        self,
        x,
        forward_batch,
        layer_id,
        compressor,
        precomputed_kv_score=None,
    ):
        self.log.append("consume:core")
        self.compressor_kv_score = precomputed_kv_score


class _FakeGenericLayer:
    def __init__(self, log, current_stream):
        self.log = log
        self.current_stream = current_stream
        self.alt_streams = [
            _FakeStream("kv", log),
            _FakeStream("core", log),
            _FakeStream("indexer", log),
        ]
        self.fuse_wqa_wkv = False
        self.dsa_enable_prefill_cp = True
        self.layer_id = 7
        self.indexer = _FakeGenericIndexer(
            _FakeGenericCompressor("indexer", 2, log), log
        )
        self.compressor = _FakeGenericCompressor("core", 3, log)
        self.used_fused_kv_store = False

    def _compute_q_a(self, x, qkv_a=None):
        self.log.append("compute:q_a")
        return x, x

    def _compute_q_b(self, q, positions, q_out=None):
        self.log.append("compute:q_b")
        return q

    def _materialize_cp_swa_k(self, x, forward_batch, qkv_a=None):
        self.log.append("project:swa")
        kv = torch.full((x.shape[0], 1), 1.0)
        return deepseek_v4.cp_materialize_global_token_order(
            kv, forward_batch, self.current_stream()
        )

    def _store_cp_swa_k(self, kv, forward_batch, attn_backend):
        self.log.append("consume:swa")
        attn_backend.swa_k = kv

    def _compute_kv_to_cache(
        self, x, positions, forward_batch, attn_backend, qkv_a=None
    ):
        self.log.append("consume:fused_swa")
        self.used_fused_kv_store = True


class TestDeepseekV4GenericCPMultiStream(CustomTestCase):
    def _run_prepare(self, use_cp):
        log = []
        main_stream = _FakeStream("main", log)
        active_stream = main_stream

        @contextmanager
        def use_stream(stream):
            nonlocal active_stream
            previous = active_stream
            active_stream = stream
            try:
                yield
            finally:
                active_stream = previous

        def current_stream():
            return active_stream

        def gather(value, forward_batch, stream=None):
            name = {1: "swa", 2: "indexer", 3: "core"}[int(value[0, 0])]
            log.append(f"gather:{name}:{current_stream().name}")
            return value + 10

        layer = _FakeGenericLayer(log, current_stream)
        backend = _FakeGenericBackend(log)
        forward_batch = SimpleNamespace()
        x = torch.ones(2, 1)
        positions = torch.arange(2)
        with (
            patch.object(torch.cuda, "current_stream", side_effect=current_stream),
            patch.object(torch.cuda, "stream", side_effect=use_stream),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=use_cp),
            patch.object(dsv4_compressor, "dsa_use_prefill_cp", return_value=use_cp),
            patch.object(
                deepseek_v4, "cp_materialize_global_token_order", side_effect=gather
            ),
            patch.object(
                dsv4_compressor,
                "cp_materialize_global_token_order",
                side_effect=gather,
            ),
        ):
            output = deepseek_v4.MQALayer._forward_prepare_multi_stream(
                layer, x, positions, forward_batch, backend
            )
        return log, layer, backend, output

    def test_cp_collectives_finish_before_worker_stream_consumers(self):
        log, layer, backend, output = self._run_prepare(use_cp=True)

        gather_events = [event for event in log if event.startswith("gather:")]
        self.assertEqual(
            gather_events,
            ["gather:swa:main", "gather:indexer:main", "gather:core:main"],
        )
        last_gather = max(log.index(event) for event in gather_events)
        first_consumer = min(
            log.index(event)
            for event in ("consume:indexer", "consume:swa", "consume:core")
        )
        self.assertLess(last_gather, first_consumer)
        self.assertFalse(layer.used_fused_kv_store)
        torch.testing.assert_close(backend.swa_k, torch.full((2, 1), 11.0))
        torch.testing.assert_close(
            layer.indexer.precomputed_kv_score, torch.full((2, 1), 12.0)
        )
        torch.testing.assert_close(
            backend.compressor_kv_score, torch.full((2, 1), 13.0)
        )
        torch.testing.assert_close(output, torch.ones(2, 1))

    def test_non_cp_keeps_existing_worker_owned_prepare(self):
        log, layer, backend, _ = self._run_prepare(use_cp=False)

        self.assertFalse(any(event.startswith("gather:") for event in log))
        self.assertTrue(layer.used_fused_kv_store)
        self.assertIsNone(backend.swa_k)
        self.assertIsNone(layer.indexer.precomputed_kv_score)
        self.assertIsNone(backend.compressor_kv_score)

    def test_unified_compressor_consumes_precomputed_score(self):
        precomputed = torch.ones(2, 4)

        class TokenPool:
            uniform_fp8 = False

            @staticmethod
            def get_index_k_page_size(ratio):
                return 64

            @staticmethod
            def get_index_k_with_scale_buffer(layer_id):
                return torch.empty(1, dtype=torch.uint8)

        class Backend:
            token_to_kv_pool = TokenPool()
            enable_deepseek_v4_fp4_indexer = False

            def __init__(self):
                self.received_kv_score = None

            @staticmethod
            def _get_out_loc(ratio):
                return torch.zeros(1, dtype=torch.int32)

            def _forward_compress_all_in_one(self, **kwargs):
                self.received_kv_score = kwargs["kv_score_input"]

        compressor = SimpleNamespace(
            is_in_indexer=True,
            ratio=4,
            head_dim=128,
            rotate=True,
            ape=torch.empty(4, 256),
            norm=SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6),
            freqs_cis=torch.empty(1),
            get_state_pool=lambda backend: SimpleNamespace(
                kv_score_buffer=SimpleNamespace(kv_score=torch.empty(1))
            ),
            compute_kv_score=lambda *args: self.fail(
                "precomputed score must bypass compute_kv_score"
            ),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False)
        )
        backend = Backend()

        dsv4_compressor_v2.CompressorBackendMixin.forward_unified(
            backend,
            torch.empty(2, 1),
            forward_batch,
            layer_id=0,
            compressor=compressor,
            precomputed_kv_score=precomputed,
        )
        self.assertIs(backend.received_kv_score, precomputed)


if __name__ == "__main__":
    unittest.main()
