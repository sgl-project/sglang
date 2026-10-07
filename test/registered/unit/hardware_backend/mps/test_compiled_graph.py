"""Live weights and dependent asynchronous launches for direct MLX export."""

import gc
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=30, suite="stage-a-unit-test-mps")


@unittest.skipUnless(
    torch.backends.mps.is_available() and importlib.util.find_spec("mlx") is not None,
    "Requires Torch MPS and MLX",
)
class TestCompiledGraph(CustomTestCase):
    def test_weighted_rms_fusion_preserves_intermediate_casts(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Norm(torch.nn.Module):
            def __init__(self, cast_before_weight):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(128, device="mps", dtype=torch.bfloat16)
                )
                self.cast_before_weight = cast_before_weight

            def forward(self, value):
                x = value.float()
                normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                if self.cast_before_weight:
                    normalized = normalized.to(value.dtype)
                return (normalized * self.weight).to(value.dtype)

        inputs = (torch.randn(3, 128, device="mps", dtype=torch.bfloat16),)
        for cast_before_weight in (False, True):
            with self.subTest(cast_before_weight=cast_before_weight):
                model = Norm(cast_before_weight)
                graph = CompiledMlxGraph(model=model, example_inputs=inputs)
                self.addCleanup(graph.close)
                self.assertEqual(len(graph.weighted_norms), int(not cast_before_weight))
                torch.mps.synchronize()
                actual = mlx_to_torch(graph.launch(graph.bind(inputs))[0])
                torch.testing.assert_close(actual, model(*inputs))

    def test_whole_forward_preserves_logits_and_checks_attention_and_metadata(self):
        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_region import (
            UnsupportedMlxRegion,
            discover_decode_region,
            static_metadata_matches,
        )
        from sglang.srt.hardware_backend.mps.compiled_runner import _DecodeModule
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.runtime_context import get_context
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vocab_size=17)
                self.embed = torch.nn.Embedding(
                    17, 64, device="mps", dtype=torch.bfloat16
                )
                self.layers = torch.nn.ModuleList(
                    [RadixAttention(1, 64, 0.125, 1, i) for i in range(2)]
                )
                self.lm_head = torch.nn.Linear(
                    64, 17, device="mps", dtype=torch.bfloat16
                )
                self.logits_processor = LogitsProcessor(self.config, logit_scale=1.75)
                self.repeat_attention = False
                self.read_sum = False
                self.mutate_metadata = False

            def forward(self, input_ids, forward_batch, positions):
                if self.mutate_metadata:
                    forward_batch.return_logprob = True
                hidden = self.embed(input_ids) + positions[:, None].to(torch.bfloat16)
                if self.read_sum:
                    hidden = hidden + forward_batch.seq_lens_sum
                for layer in self.layers:
                    hidden = layer(hidden, hidden, hidden, forward_batch)
                if self.repeat_attention:
                    hidden = self.layers[-1](hidden, hidden, hidden, forward_batch)
                return self.logits_processor(
                    input_ids, hidden, self.lm_head, forward_batch
                )

        with get_context().override_server_args(
            device="mps", tp_size=1, pp_size=1, enable_fp32_lm_head=True
        ):
            model = Model().eval()
            pools = tuple(
                torch.zeros(8, 1, 64, device="mps", dtype=torch.bfloat16)
                for _ in range(4)
            )
            table = torch.tensor([[1, 2, 3, 4]], device="mps", dtype=torch.int32)
            wrapper = _DecodeModule(
                model=model,
                req_pool=SimpleNamespace(req_to_token=table),
                kv_pool=SimpleNamespace(
                    get_kv_buffer=lambda i: pools[2 * i : 2 * i + 2]
                ),
                region=discover_decode_region(model),
            )
            inputs = tuple(
                torch.tensor([value], device="mps", dtype=torch.int64)
                for value in (3, 0, 0, 1, 1)
            )
            with forward_context(ForwardContext(attn_backend=wrapper.attention)):
                graph = CompiledMlxGraph(
                    model=wrapper, example_inputs=inputs, attention=radix_decode
                )
                self.addCleanup(graph.close)
                for token, position in ((3, 0), (4, 1)):
                    current = (
                        torch.full_like(inputs[0], token),
                        torch.full_like(inputs[1], position),
                        *inputs[2:],
                    )
                    expected = wrapper(*current)[0]
                    torch.mps.synchronize()
                    actual = mlx_to_torch(graph.launch(graph.bind(current))[0])
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                    self.assertEqual(actual.dtype, torch.float32)
                model.repeat_attention = True
                with self.assertRaisesRegex(UnsupportedMlxRegion, "invocation order"):
                    CompiledMlxGraph(
                        model=wrapper, example_inputs=inputs, attention=radix_decode
                    )
                model.repeat_attention = False
                model.read_sum = True
                wrapper(*inputs)
                model.mutate_metadata = True
                with self.assertRaisesRegex(
                    UnsupportedMlxRegion, "mutate batch metadata"
                ):
                    CompiledMlxGraph(
                        model=wrapper, example_inputs=inputs, attention=radix_decode
                    )
            batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=1,
                input_ids=inputs[0],
                req_pool_indices=inputs[2],
                seq_lens=inputs[3],
                out_cache_loc=inputs[4],
                seq_lens_sum=1,
            )
            self.assertIn("seq_lens_sum", wrapper.static_reads)
            self.assertFalse(static_metadata_matches(batch, wrapper.static_reads))

    def test_silu_does_not_round_sigmoid_before_multiplication(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        inputs = (torch.linspace(-8, 8, 4096, device="mps", dtype=torch.bfloat16),)
        model = torch.nn.SiLU()
        graph = CompiledMlxGraph(model=model, example_inputs=inputs)
        self.addCleanup(graph.close)
        torch.mps.synchronize()
        actual = mlx_to_torch(graph.launch(graph.bind(inputs))[0])
        torch.testing.assert_close(actual, model(*inputs), rtol=0, atol=0)

    def test_structural_admission_rejects_ambiguous_or_reused_state(self):
        from sglang.srt.hardware_backend.mps.compiled_region import (
            UnsupportedMlxRegion,
            discover_decode_region,
        )
        from sglang.srt.layers.radix_attention import RadixAttention

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.ModuleDict(
                    {
                        "blocks": torch.nn.ModuleList(
                            [
                                RadixAttention(
                                    num_heads=2,
                                    head_dim=64,
                                    scaling=0.125,
                                    num_kv_heads=1,
                                    layer_id=i,
                                )
                                for i in range(2)
                            ]
                        )
                    }
                )

            def forward(self, input_ids, forward_batch, position_ids):
                return input_ids

        model = Model()
        region = discover_decode_region(model)
        self.assertEqual(region.position_arg, "position_ids")
        self.assertEqual(tuple(x.layer_id for x in region.layers), (0, 1))
        model.transformer["blocks"][1].layer_id = 0
        with self.assertRaisesRegex(UnsupportedMlxRegion, "unique"):
            discover_decode_region(model)
        model = Model()
        model.other = Model().transformer
        with self.assertRaisesRegex(UnsupportedMlxRegion, "one decoder stack"):
            discover_decode_region(model)
        model = Model()
        model.transformer["blocks"][1] = model.transformer["blocks"][0]
        with self.assertRaisesRegex(UnsupportedMlxRegion, "unique"):
            discover_decode_region(model)

        class UnknownArgument(Model):
            def forward(self, input_ids, positions, forward_batch, secret_state):
                return input_ids

        with self.assertRaisesRegex(UnsupportedMlxRegion, "secret_state"):
            discover_decode_region(UnknownArgument())

    def test_dense_activation_and_layernorm_lowerings_match_torch(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64, device="mps", dtype=dtype)

            def forward(self, x):
                original = x
                x = self.norm(x)
                return (
                    torch.nn.functional.gelu(x),
                    torch.nn.functional.gelu(x, approximate="tanh"),
                    torch.relu(x),
                    torch.tanh(x),
                    *torch.native_layer_norm(
                        original, (64,), self.norm.weight, self.norm.bias, self.norm.eps
                    ),
                )

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = Model(dtype).eval()
                inputs = (torch.randn(3, 64, device="mps", dtype=dtype),)
                graph = CompiledMlxGraph(model=model, example_inputs=inputs)
                self.addCleanup(graph.close)
                torch.mps.synchronize()
                outputs = graph.launch(graph.bind(inputs))
                for actual, expected in zip(outputs, model(*inputs)):
                    torch.testing.assert_close(mlx_to_torch(actual), expected)

    def test_custom_metal_lowering_survives_export_and_dependent_compile(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        @torch.library.custom_op(
            "sglang_test_compiled::square_plus_one", mutates_args=()
        )
        def custom(x: torch.Tensor) -> torch.Tensor:
            return x.square() + 1

        @custom.register_fake
        def fake(x):
            return torch.empty_like(x)

        class Model(torch.nn.Module):
            def forward(self, x):
                return custom(x) * 0.5

        kernel = mx.fast.metal_kernel(
            name="test_compiled_square_plus_one",
            input_names=["x"],
            output_names=["out"],
            source="uint i = thread_position_in_grid.x; out[i] = x[i] * x[i] + 1;",
        )

        def lower(x):
            return kernel(
                inputs=[x],
                grid=(x.size, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[x.shape],
                output_dtypes=[x.dtype],
            )[0]

        model = Model().eval()
        inputs = (torch.randn(2, 32, device="mps"),)
        with self.assertRaisesRegex(ValueError, "No direct MLX lowering"):
            CompiledMlxGraph(model=model, example_inputs=inputs)
        torch.mps.synchronize()
        graph = CompiledMlxGraph(
            model=model,
            example_inputs=inputs,
            custom_lowerings={
                torch.ops.sglang_test_compiled.square_plus_one.default: lower
            },
        )
        self.addCleanup(graph.close)
        arrays = graph.bind(inputs)
        first = graph.launch(arrays)
        second = graph.launch(first)
        torch.testing.assert_close(mlx_to_torch(second[0]), model(model(*inputs)))

    def test_host_alias_retains_offset_storage_and_live_updates(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import host_alias

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source = torch.arange(24, dtype=dtype, device="mps").reshape(6, 4)
                view = source[2:5]
                torch.mps.synchronize()
                alias = host_alias(view)
                source.add_(7)
                torch.mps.synchronize()
                torch.testing.assert_close(alias, view.cpu())
                expected = alias.clone()
                del source, view
                gc.collect()
                torch.testing.assert_close(alias, expected)

    def test_async_requires_compiled_local_mps_execution(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs(model_path="dummy").mps_execution_backend, "eager")
        enabled = ServerArgs(
            model_path="dummy",
            device="mps",
            mps_execution_backend="mlx-compiled",
        )
        enabled.resolve_once()
        for option in (
            {"enable_hierarchical_cache": True},
            {"enable_lmcache": True},
            {"enable_session_radix_cache": True},
        ):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(ValueError, "local"),
            ):
                ServerArgs(
                    model_path="dummy",
                    device="mps",
                    mps_execution_backend="mlx-compiled",
                    **option,
                ).resolve_once()
        with self.assertRaisesRegex(ValueError, "standard Torch MPS"):
            ServerArgs(
                model_path="dummy",
                device="cpu",
                mps_execution_backend="mlx-compiled",
            ).resolve_once()

    def test_normalization_fusion_preserves_cast_boundary(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Norm(torch.nn.Module):
            def forward(self, value):
                x = value.float()
                return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(
                    value.dtype
                )

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = (torch.randn(2, 128, device="mps", dtype=dtype),)
                model = Norm()
                graph = CompiledMlxGraph(model=model, example_inputs=inputs)
                self.assertEqual(len(graph.norms), 1)
                torch.mps.synchronize()
                result = graph.launch(graph.bind(inputs))
                torch.testing.assert_close(mlx_to_torch(result[0]), model(*inputs))
                graph.close()

    def test_mutation_and_changed_attribute_metadata_fail_closed(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph

        class Mutating(torch.nn.Module):
            def forward(self, x):
                return x.add_(1)

        with self.assertRaisesRegex(ValueError, "mutate"):
            CompiledMlxGraph(
                model=Mutating(), example_inputs=(torch.ones(4, device="mps"),)
            )
        model = torch.nn.Linear(4, 4, device="mps").eval()
        inputs = (torch.ones(1, 4, device="mps"),)
        graph = CompiledMlxGraph(model=model, example_inputs=inputs)
        model.weight = torch.nn.Parameter(torch.ones(8, 4, device="mps"))
        with self.assertRaisesRegex(ValueError, "attribute metadata"):
            graph.bind(inputs)
        graph.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            graph.bind(inputs)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            graph.launch(())

    def test_enqueue_precedes_current_wait_and_stale_results_are_discarded(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.hardware_backend.mps.compiled_runner import (
            CompiledMlxRunner,
            _batch_inputs,
        )
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 4, device="mps")

            def forward(self, ids, positions, requests, lengths, locations):
                logits = self.embedding(ids)
                kv = logits.reshape(1, -1, 1, 4)
                return logits, kv, kv

        model = Decoder().eval()
        table = torch.zeros(4, 32, device="mps", dtype=torch.int32)
        table[1, 1] = 5
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([1], device="mps"),
            positions=torch.tensor([1], device="mps"),
            req_pool_indices=torch.tensor([1], device="mps"),
            seq_lens=torch.tensor([2], device="mps"),
            seq_lens_sum=2,
            out_cache_loc=torch.tensor([5], device="mps"),
        )
        batch.sampling_info = SimpleNamespace(
            is_all_greedy=True,
            grammars=None,
            grammar_mask=None,
            has_custom_logit_processor=False,
            logit_bias=None,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            penalizer_orchestrator=None,
        )
        graph = CompiledMlxGraph(model=model, example_inputs=_batch_inputs(batch))
        self.addCleanup(graph.close)
        runner = object.__new__(CompiledMlxRunner)
        runner.model_runner = SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(context_len=32),
            req_to_token_pool=SimpleNamespace(
                req_to_token=table, req_generation=torch.zeros(4, dtype=torch.int64)
            ),
        )
        runner._max_batch_size = 4
        runner._unsupported_reason = None
        runner._unsupported_batches = {}
        runner._model_structure = tuple(
            (name, id(module))
            for name, module in model.named_modules(remove_duplicate=False)
        )
        model.static_reads = {}
        runner._fallback_reasons = set()
        runner._layers = ()
        runner._async = False
        runner._pending = None
        runner.execution_count = runner.prefetch_count = 0
        runner.prefetch_hits = runner.prefetch_discards = 0
        events = []
        launch, evaluate = graph.launch, mx.eval

        torch.mps.synchronize()
        synchronous = graph.launch(graph.bind(_batch_inputs(batch)))
        runner._prefetch(
            graph=graph,
            arrays=graph.bind(_batch_inputs(batch)),
            outputs=synchronous,
            batch=batch,
            metadata=runner._metadata(batch),
            weights=runner._weights(),
        )
        self.assertIsNone(runner._pending)
        runner.enable_lookahead()

        def record_launch(*args, **kwargs):
            events.append("launch")
            return launch(*args, **kwargs)

        def record_wait(*args, **kwargs):
            events.append("wait")
            return evaluate(*args, **kwargs)

        with (
            patch.object(runner, "_graph", return_value=graph),
            patch.object(
                runner,
                "_validate_metadata",
                side_effect=lambda _: torch.mps.synchronize(),
            ),
            patch.object(graph, "launch", side_effect=record_launch),
            patch.object(mx, "eval", side_effect=record_wait),
        ):
            result = runner.execute(batch)
        self.assertEqual(events[:3], ["launch", "launch", "wait"])
        torch.testing.assert_close(
            result.next_token_logits, model.embedding(batch.input_ids)
        )
        runner.drain()
        pending = runner._pending
        runner.model_runner.model_config.context_len = 2
        runner._pending = None
        count = runner.prefetch_count
        runner._prefetch(
            graph=graph,
            arrays=graph.bind(_batch_inputs(batch)),
            outputs=pending.outputs,
            batch=batch,
            metadata=runner._metadata(batch),
            weights=runner._weights(),
        )
        self.assertIsNone(runner._pending)
        self.assertEqual(runner.prefetch_count, count)
        runner.model_runner.model_config.context_len = 32
        runner._pending = pending
        batch.input_ids = torch.tensor(pending.tokens.tolist(), device="mps")
        batch.positions = batch.positions + 1
        batch.seq_lens = batch.seq_lens + 1
        torch.mps.synchronize()
        metadata, weights = runner._metadata(batch), runner._weights()
        self.assertIs(
            runner._consume(
                graph=graph, batch=batch, metadata=metadata, weights=weights
            ),
            pending.outputs,
        )
        self.assertEqual(runner.prefetch_hits, 1)
        for changed in ("generation", "length", "weights", "tokens", "prefix_slot"):
            with self.subTest(changed=changed):
                runner._pending = pending
                candidate, stamp = metadata, weights
                if changed == "generation":
                    candidate = (*metadata[:3], (1,))
                elif changed == "length":
                    candidate = (metadata[0], (4,), *metadata[2:])
                elif changed == "weights":
                    stamp = ()
                elif changed == "tokens":
                    batch.input_ids = (batch.input_ids + 1) % 4
                else:
                    table[1, 1] = 6
                torch.mps.synchronize()
                self.assertIsNone(
                    runner._consume(
                        graph=graph, batch=batch, metadata=candidate, weights=stamp
                    )
                )
                batch.input_ids = torch.tensor(pending.tokens.tolist(), device="mps")
        self.assertEqual(runner.prefetch_discards, 5)
        runner._pending = pending
        batch.forward_mode = ForwardMode.EXTEND
        self.assertFalse(runner.can_run_graph(batch))
        self.assertIsNone(runner._pending)

    def test_embedding_unbounded_slice_and_live_nonpersistent_buffer(self):
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Embedding(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 8, device="mps")
                self.register_buffer(
                    "offset", torch.ones(8, device="mps"), persistent=False
                )

            def forward(self, ids):
                result = torch.ops.aten.alias.default(self.embedding(ids)[:, :])
                return result + self.offset

        model = Embedding().eval()
        inputs = (torch.tensor([1, 3], device="mps"),)
        with torch.no_grad():
            graph = CompiledMlxGraph(model=model, example_inputs=inputs)
            for replacement in (False, True):
                if replacement:
                    model.offset = torch.full((8,), 3.0, device="mps")
                torch.mps.synchronize()
                actual = graph.launch(graph.bind(inputs))
                torch.testing.assert_close(mlx_to_torch(actual[0]), model(*inputs))
            graph.close()

    def test_aliased_module_bindings_follow_submodule_replacement(self):
        """Exported alias paths must survive enumeration and subtree replacement."""
        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        class Branch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, device="mps")
                self.register_buffer(
                    "offset", torch.randn(4, device="mps"), persistent=False
                )

            def forward(self, x):
                return self.linear(x) + self.offset

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = Branch()
                self.right = self.left

            def forward(self, x):
                return self.right(x)

        model = Model().eval()
        inputs = (torch.randn(2, 4, device="mps"),)
        with torch.no_grad():
            graph = CompiledMlxGraph(model=model, example_inputs=inputs)
            self.addCleanup(graph.close)
            for replace in (False, True):
                with self.subTest(replace=replace):
                    if replace:
                        model.right = Branch()
                    torch.mps.synchronize()
                    output = graph.launch(graph.bind(inputs))
                    torch.testing.assert_close(mlx_to_torch(output[0]), model(*inputs))

    def test_compiled_live_weights_and_dependent_launch(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mps.compiled_graph import CompiledMlxGraph
        from sglang.srt.utils.tensor_bridge import mlx_to_torch

        model = (
            torch.nn.Sequential(
                torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32)
            )
            .to("mps")
            .eval()
        )
        inputs = (torch.randn(2, 32, device="mps"),)
        with torch.no_grad():
            graph = CompiledMlxGraph(model=model, example_inputs=inputs)
            torch.mps.synchronize()
            flat = graph.bind(inputs)
            first = graph.launch(flat)
            following = list(flat)
            user_index = next(
                i
                for i, (kind, _) in enumerate(graph.bindings)
                if kind.name == "USER_INPUT"
            )
            following[user_index] = first[0]
            second = graph.launch(tuple(following))
            # Both dependent calls were submitted before any host materialization.
            mx.eval(*second)
            torch.testing.assert_close(
                mlx_to_torch(second[0]), model(model(*inputs)), atol=1e-5, rtol=1e-4
            )
            model[0].weight.add_(0.1)
            torch.mps.synchronize()
            updated = graph.launch(graph.bind(inputs))
            torch.testing.assert_close(
                mlx_to_torch(updated[0]), model(*inputs), atol=1e-5, rtol=1e-4
            )
            model[0].weight = torch.nn.Parameter(model[0].weight * 0.5)
            torch.mps.synchronize()
            replaced = graph.launch(graph.bind(inputs))
            exported = mlx_to_torch(replaced[0])
            expected = model(*inputs).clone()
            graph.close()
            torch.testing.assert_close(exported, expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
