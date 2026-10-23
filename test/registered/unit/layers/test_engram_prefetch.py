"""Real host tables exercise bounded prefetch, slot reuse and graph replay."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.engram_prefetch import EngramPrefetcher
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.engram_utils import make_engram_hasher, make_engram_layers
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-large")


class TestEngramPrefetch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        publish_build_topology(
            tp_size=1,
            disable_custom_all_reduce=True,
            fp8_gemm_runner_backend="triton",
        )
        from sglang.srt.layers.quantization import fp8_utils

        backend = patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", None)
        backend.start()
        cls.addClassCleanup(backend.stop)
        fp8_utils.initialize_fp8_gemm_config()
        cls.rendezvous = tempfile.NamedTemporaryFile()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file://" + cls.rendezvous.name,
        )
        initialize_model_parallel()

    @classmethod
    def tearDownClass(cls):
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()
        reset_context()

    def test_lookup_projection_and_window_reuse(self):
        for layout in ("shared", "per_rank"):
            layers = make_engram_layers(layout)
            for window in (0, 1, 2):
                prefetch = EngramPrefetcher(layers, 128, window)
                for tokens in (259, 1, 129, 0):
                    hashes = torch.randint(0, 1024, (tokens, 3, 4), device="cuda")
                    source = torch.randn(
                        tokens, 2, 64, dtype=torch.bfloat16, device="cuda"
                    )
                    expected = source
                    for layer in layers:
                        expected = layer(expected, hashes[:, layer.layer_hash_index])
                    prefetch.begin(hashes, None)
                    actual = source
                    for layer in layers:
                        actual = layer(
                            actual, hashes[:, layer.layer_hash_index], prefetch=prefetch
                        )
                    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
                    self.assertLessEqual(
                        prefetch.buffer_bytes, (window + 2) * 128 * 4 * 32 * 2
                    )

    def test_exact_rows_delayed_consumers_and_exception_recovery(self):
        layers = make_engram_layers()
        prefetch = EngramPrefetcher(layers, 128, 1)
        hashes = torch.randint(0, 1024, (387, 3, 4), device="cuda")
        hashes[0] = 0
        expected = [layer.embed(hashes[:, layer.layer_hash_index]) for layer in layers]
        with torch.cuda.stream(prefetch.stream):
            torch.cuda._sleep(500000)
        prefetch.begin(hashes, None)
        for layer, reference in zip(layers, expected):
            pieces = []
            for block in range(prefetch.num_chunks):
                with prefetch.chunk(layer.layer_hash_index, block) as (rows, values):
                    torch.cuda._sleep(500000)
                    pieces.append(values.clone())
            prefetch.end_layer(layer.layer_hash_index)
            torch.testing.assert_close(torch.cat(pieces), reference, atol=0, rtol=0)
        prefetch.begin(hashes, None)
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with prefetch.chunk(0, 0):
                raise RuntimeError("consumer failed")
        prefetch.begin(hashes, None)
        for layer in layers:
            for block in range(prefetch.num_chunks):
                with prefetch.chunk(layer.layer_hash_index, block):
                    pass
            prefetch.end_layer(layer.layer_hash_index)

    def test_reserved_rows_and_finish_clear_forward_state(self):
        from sglang.srt.layers.engram_prefetch import EngramPrefetchLayout

        layers = make_engram_layers(layer_ids=(0,))
        prefetch = EngramPrefetcher(layers, 128, 0)
        hashes = torch.randint(0, 1024, (260, 1, 4), device="cuda")
        source = torch.randn(131, 2, 64, dtype=torch.bfloat16, device="cuda")
        expected = layers[0](source, hashes[:131, 0])
        prefetch.begin(hashes, None)
        # Model a rank's reserved gather slot containing fewer activations.
        prefetch.layout = EngramPrefetchLayout((260,), 0, 131, 128)
        actual = layers[0](source, hashes[:131, 0], prefetch=prefetch)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertEqual(prefetch.num_chunks, 0)
        self.assertEqual(prefetch.order, ())
        self.assertIsNone(prefetch.layout)
        prefetch.begin(hashes, None, layer_ids=())
        self.assertEqual(prefetch.num_chunks, 0)

    def test_empty_shard_returns_zero_without_host_reads(self):
        layers = make_engram_layers(rows=0, layer_ids=(0,))
        prefetch = EngramPrefetcher(layers, 128, 0)
        hashes = torch.zeros(129, 1, 4, dtype=torch.int64, device="cuda")
        source = torch.randn(129, 2, 64, dtype=torch.bfloat16, device="cuda")
        prefetch.begin(hashes, None)
        actual = layers[0](source, hashes[:, 0], prefetch=prefetch)
        torch.testing.assert_close(actual, source, atol=0, rtol=0)

    def test_verify_commits_only_accepted_history_before_next_decode(self):
        from sglang.kernels.ops.embeddings.engram_hash import MODE_VERIFY
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        layers = make_engram_layers()
        hasher = make_engram_hasher()
        prefetch = EngramPrefetcher(layers, 128, 1)
        slots = torch.tensor([0, 1], device="cuda")
        hasher.history[:2].copy_(torch.tensor([[7], [8]], device="cuda"))
        saved = hasher.history.clone()
        ids = torch.tensor([1, 2, 3, 4, 5, 6], device="cuda")
        positions = torch.tensor([2, 3, 4, 2, 3, 4], device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            req_pool_indices=slots,
            positions=positions,
            spec_info=SimpleNamespace(draft_token_num=3),
        )
        hashes = hasher(ids, batch)
        reference, _ = hasher._torch_hash_ids(
            ids, positions, MODE_VERIFY, saved[:2], 6, 3, None, None
        )
        torch.testing.assert_close(hashes, reference, atol=0, rtol=0)
        torch.testing.assert_close(hasher.history, saved, atol=0, rtol=0)
        source = torch.randn(6, 2, 64, device="cuda", dtype=torch.bfloat16)
        expected = source
        for layer in layers:
            expected = layer(expected, hashes[:, layer.layer_hash_index])
        prefetch.begin(hashes, batch)
        actual = source
        for layer in layers:
            actual = layer(
                actual, hashes[:, layer.layer_hash_index], batch, prefetch=prefetch
            )
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        # Request 0 rejects all drafts; request 1 accepts its first draft.
        hasher.commit_after_verify(
            ids.view(2, 3), slots, torch.tensor([1, 2], device="cuda")
        )
        torch.testing.assert_close(
            hasher.history[:2],
            torch.tensor([[1], [5]], device="cuda", dtype=torch.int32),
            atol=0,
            rtol=0,
        )
        decode_ids = torch.tensor([9, 10], device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=slots,
            positions=torch.tensor([5, 5], device="cuda"),
            out_cache_loc=torch.tensor([1, 2], device="cuda"),
        )
        decode_hashes = hasher(decode_ids, batch)
        torch.testing.assert_close(
            hasher.history[:2], decode_ids[:, None].to(torch.int32), atol=0, rtol=0
        )
        prefetch.begin(decode_hashes, batch)
        expected, actual = source[:2], source[:2]
        for layer in layers:
            expected = layer(expected, decode_hashes[:, layer.layer_hash_index])
            actual = layer(
                actual,
                decode_hashes[:, layer.layer_hash_index],
                batch,
                prefetch=prefetch,
            )
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_h20_fp8_projection_matches_synchronous_path(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        for block_quant in (False, True):
            with self.subTest(block_quant=block_quant):
                layers = make_engram_layers(layer_ids=(0,))
                layer = layers[0]
                quant = (
                    Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        weight_block_size=[32, 32],
                        scale_fmt="ue8m0",
                    )
                    if block_quant
                    else Fp8Config()
                )
                with torch.device("cuda"):
                    layer.wkv = ReplicatedLinear(
                        128,
                        192,
                        bias=False,
                        params_dtype=torch.bfloat16,
                        quant_config=quant,
                    )
                weight = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16) * (
                    0.5 if block_quant else 0.01
                )
                layer.wkv.weight.data.copy_(weight.to(layer.wkv.weight.dtype))
                if block_quant:
                    layer.wkv.weight_scale_inv.data.fill_(2**-5)
                layer.wkv.quant_method.process_weights_after_loading(layer.wkv)
                prefetch = EngramPrefetcher(layers, 128, 0)
                hashes = torch.randint(0, 1024, (259, 1, 4), device="cuda")
                source = torch.randn(259, 2, 64, dtype=torch.bfloat16, device="cuda")
                expected = layer(source, hashes[:, 0])
                prefetch.begin(hashes, None)
                actual = layer(source, hashes[:, 0], prefetch=prefetch)
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_breakable_hooks_read_live_batch_and_replay_new_rows(self):
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            set_tc_piecewise_forward_context,
        )
        from sglang.srt.models.deepseek_v4 import (
            bcg_deepseek_v4_engram_apply,
            bcg_deepseek_v4_engram_prefetch,
        )

        layers = make_engram_layers()
        prefetch = EngramPrefetcher(layers, 128, 1)
        hashes = torch.randint(0, 1024, (259, 3, 4), device="cuda")
        source = torch.randn(259, 2, 64, dtype=torch.bfloat16, device="cuda")

        def run():
            live_hashes = bcg_deepseek_v4_engram_prefetch(
                hashes, prefetch, (0, 1, 2), False
            )
            value = source + 0
            for layer in layers:
                value = bcg_deepseek_v4_engram_apply(
                    value,
                    live_hashes[:, layer.layer_hash_index],
                    layer,
                    prefetch,
                    False,
                )
                value = value + 0
            return value

        with set_tc_piecewise_forward_context(SimpleNamespace(), [], None, [], []):
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = BreakableCUDAGraph()
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with BreakableCUDAGraphCapture(graph, stream=capture_stream):
                actual = run()
            torch.cuda.current_stream().wait_stream(capture_stream)
        for _ in range(2):
            hashes.random_(0, 1024)
            with set_tc_piecewise_forward_context(SimpleNamespace(), [], None, [], []):
                graph.replay()
            expected = source
            for layer in layers:
                expected = layer(expected, hashes[:, layer.layer_hash_index])
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_model_loop_preserves_history_and_restarts_after_tail_selection(self):
        from unittest.mock import patch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4 as dsv4

        layers = make_engram_layers()
        model = dsv4.DeepseekV4Model.__new__(dsv4.DeepseekV4Model)
        torch.nn.Module.__init__(model)

        class Block(torch.nn.Module):
            boundary_mhc_fusions = False
            hc_cfg = None

            def __init__(self, engram=None):
                super().__init__()
                self.engram = engram

            def forward(self, hidden_states, **kwargs):
                return hidden_states

        model.layers = torch.nn.ModuleList(
            [
                Block(layers[0]),
                Block(),
                Block(layers[1]),
                Block(),
                Block(layers[2]),
            ]
        )
        model.start_layer, model.end_layer = 0, 5
        model.pp_group = SimpleNamespace(world_size=1)
        model.config = SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0)
        model.decoder_replay_tail = None
        model.tail_writes_source_kv = False
        model.engram_prefetch = None
        hasher = make_engram_hasher()
        model.engram_hasher = hasher
        ids = torch.tensor([1, 2, 3, 4, 5, 6], device="cuda")
        positions = torch.tensor([0, 1, 2, 0, 1, 2], device="cuda")
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            positions=positions,
            input_ids=ids,
            req_pool_indices=torch.tensor([0, 1], device="cuda"),
            extend_seq_lens=torch.tensor([3, 3], device="cuda"),
            extend_seq_lens_cpu=[3, 3],
            extend_start_loc=torch.tensor([0, 3], device="cuda"),
            engram_history=None,
            out_cache_loc=torch.arange(1, 7, device="cuda"),
            residual_stream=None,
            capture_hidden_mode=None,
            return_logprob=False,
        )
        tail_indices = torch.tensor([2, 5], device="cuda")
        tail = SimpleNamespace(
            scatter_indices=None,
            extend_seq_lens_cpu=(1, 1),
            rows=lambda value: value[tail_indices],
            positions=positions[[2, 5]],
        )
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            enter_late_layer_tail=lambda *args: object(),
            exit_late_layer_tail=lambda *args: None,
        )
        source = torch.randn(6, 2, 64, device="cuda", dtype=torch.bfloat16)
        prefetch = EngramPrefetcher(layers, 128, 1)
        for boundary in (None, 2):
            model.late_layer_start = boundary
            hasher.history.zero_()
            model.engram_prefetch = None
            with patch.object(dsv4, "get_attn_backend", return_value=backend):
                expected, _, _ = model._forward_layers_hc_pre_from_prev(
                    positions, source, batch, ids, ids, False, []
                )
            expected_history = hasher.history.clone()
            hasher.history.zero_()
            model.engram_prefetch = prefetch
            with patch.object(dsv4, "get_attn_backend", return_value=backend):
                actual, _, _ = model._forward_layers_hc_pre_from_prev(
                    positions, source, batch, ids, ids, False, []
                )
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(hasher.history, expected_history, atol=0, rtol=0)
            self.assertIsNone(prefetch.hashes)
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
                BreakableCUDAGraph,
                BreakableCUDAGraphCapture,
            )
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
                enable_breakable_cuda_graph,
            )
            from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
                set_tc_piecewise_forward_context,
            )

            hasher.history.zero_()
            torch.cuda.synchronize()
            graph = BreakableCUDAGraph()
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with (
                patch.object(dsv4, "get_attn_backend", return_value=backend),
                enable_breakable_cuda_graph(),
                set_tc_piecewise_forward_context(batch, [], None, [], []),
                BreakableCUDAGraphCapture(graph, stream=capture_stream),
            ):
                graph_output, _, _ = model._forward_layers_hc_pre_from_prev(
                    positions, source, batch, ids, ids, False, []
                )
            torch.cuda.current_stream().wait_stream(capture_stream)
            hasher.history.zero_()
            with (
                patch.object(dsv4, "get_attn_backend", return_value=backend),
                set_tc_piecewise_forward_context(batch, [], None, [], []),
            ):
                graph.replay()
            torch.testing.assert_close(graph_output, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(hasher.history, expected_history, atol=0, rtol=0)

    def test_graph_replay_uses_new_ids_and_releases_slots(self):
        layers = make_engram_layers()
        prefetch = EngramPrefetcher(layers, 128, 1)
        hashes = torch.randint(0, 1024, (259, 3, 4), device="cuda")
        source = torch.randn(259, 2, 64, dtype=torch.bfloat16, device="cuda")

        def run():
            prefetch.begin(hashes, None)
            value = source
            for layer in layers:
                value = layer(
                    value, hashes[:, layer.layer_hash_index], prefetch=prefetch
                )
            return value

        for _ in range(3):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for _ in range(3):
            hashes.random_(0, 1024)
            graph.replay()
            expected = source
            for layer in layers:
                expected = layer(expected, hashes[:, layer.layer_hash_index])
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
