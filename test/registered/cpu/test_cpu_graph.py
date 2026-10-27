import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.layer_boundary.residual.stream as stream_mod
import sglang.srt.model_executor.cpu_graph_runner as cpu_graph_runner_module
import sglang.srt.model_executor.model_runner_components.cuda_graph_setup as graph_setup
from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_prefill_cuda_graph_batch_sizes,
    handle_cuda_graph_config,
)
from sglang.srt.arg_groups.memory_hook import handle_gpu_memory_settings
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.rotary_embedding.utils import rotary_embedding_cpu
from sglang.srt.model_executor.cpu_graph_runner import (
    CPUGraphRunner,
    _mark_dynamic,
    cpu_graph_patch_boundary_classes,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
    default_prefill_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_cpu_ids_by_node, kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    intel_amx_benchmark,
    is_in_ci,
    popen_launch_server,
)

register_cpu_ci(est_time=315, suite="stage-a-tp-test-cpu-intel")


def _make_cpu_graph_runner():
    runner = CPUGraphRunner.__new__(CPUGraphRunner)
    runner.capture_num_tokens = [128, 256, 390]
    runner.disable_padding = False
    runner.prefill_max_num_tokens = 390
    runner.prefill_max_bs = 4
    runner.prefill_graphs = {390: object()}
    runner.prefill_graphs_bs1 = {}
    runner.prefill_dynamic_graph = None
    runner.prefill_dynamic_graph_bs1 = None
    runner.enable_torch_compile = False
    return runner


def _make_cpu_prefill_forward_batch(num_tokens=6, batch_size=2):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        global_forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=torch.arange(num_tokens, dtype=torch.int64),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
        mrope_positions=torch.arange(3 * num_tokens, dtype=torch.int64).reshape(
            3, num_tokens
        ),
        mm_inputs=None,
    )


class _MiniTransformerBlock(torch.nn.Module):
    """Minimal transformer block incorporating RoPE, linears, MLP, and layer boundary."""

    def __init__(self):
        super().__init__()
        self.head_size = 128
        self.cos_sin_cache = torch.randn(2048, 128, dtype=torch.bfloat16)
        self.q_proj = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
        self.out_proj = torch.nn.Linear(2048, 2048, bias=False, dtype=torch.bfloat16)
        self.mlp_gate = torch.nn.Linear(2048, 4096, bias=False, dtype=torch.bfloat16)
        self.mlp_down = torch.nn.Linear(4096, 2048, bias=False, dtype=torch.bfloat16)

    def forward(self, input_ids, positions, forward_batch):
        # Access active forward context
        _ = get_forward_context()
        q = self.q_proj(input_ids)
        k = self.k_proj(input_ids)
        v = self.v_proj(input_ids)
        q, k = rotary_embedding_cpu(
            positions, q, k, self.head_size, self.cos_sin_cache, is_neox=True
        )
        attn = q * k + v
        hidden = self.out_proj(attn)

        # Layer boundary / residual stream compatibility
        contrib = stream_mod.Contribution(hidden, hidden)
        decl = stream_mod.DeclaredSum("group", complete_sum=lambda x: x + 1.0)
        normed = decl.complete(contrib.value)

        # Non-linear activation & MLP projection
        gate = torch.relu(self.mlp_gate(normed))
        return self.mlp_down(gate) + normed


class TestCPUGraph(CustomTestCase):
    def _resolve_cpu_graph_args(self, **overrides):
        args = ServerArgs(model_path="dummy", device="cpu", **overrides)
        args._model_config = SimpleNamespace(
            is_multimodal=False,
            is_generation=True,
            is_encoder_decoder=False,
            is_mrope=False,
            is_deepseek_v4=False,
            attention_arch=None,
            is_mla=False,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,
            num_hidden_layers=32,
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        )
        handle_cuda_graph_config(args)
        with (
            patch(
                "sglang.srt.arg_groups.memory_hook.use_mla_backend", return_value=False
            ),
            patch(
                "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
                return_value=None,
            ),
        ):
            handle_gpu_memory_settings(args)
        return resolved_view(args)

    def test_cpu_prefill_backend_defaults_to_disabled(self):
        self.assertEqual(default_prefill_backend(), Backend.DISABLED)

    def test_cpu_graph_backend_requires_intel_amx_for_both_phases(self):
        mr = ModelRunner.__new__(ModelRunner)
        mr.device = "cpu"
        mr.prefill_attention_backend_str = "intel_amx"
        mr.decode_attention_backend_str = "intel_amx"
        self.assertTrue(mr._uses_cpu_graph_attention_backend())

        mr.decode_attention_backend_str = "torch_native"
        self.assertFalse(mr._uses_cpu_graph_attention_backend())

        mr.decode_attention_backend_str = "intel_amx"
        mr.prefill_attention_backend_str = "torch_native"
        self.assertFalse(mr._uses_cpu_graph_attention_backend())

    def test_cpu_graph_boundary_patch_is_reentrant_and_restores(self):
        import msgspec.structs

        from sglang.srt.layers.layer_boundary import exit as exit_mod
        from sglang.srt.layers.layer_boundary import output as output_mod
        from sglang.srt.layers.layer_boundary import prepare as prepare_mod
        from sglang.srt.layers.layer_boundary import stage as stage_mod
        from sglang.srt.layers.layer_boundary.fusions import allreduce as allreduce_mod
        from sglang.srt.layers.layer_boundary.fusions import cutedsl as cutedsl_mod
        from sglang.srt.layers.layer_boundary.residual import access as access_mod

        patched_refs = [
            (stream_mod, "Contribution", "_DynamoCompatibleContribution"),
            (stream_mod, "DeclaredSum", "_DynamoCompatibleDeclaredSum"),
            (prepare_mod, "DeclaredSum", "_DynamoCompatibleDeclaredSum"),
            (stage_mod, "DeclaredSum", "_DynamoCompatibleDeclaredSum"),
            (stream_mod, "OwedOutput", "_DynamoCompatibleOwedOutput"),
            (access_mod, "OwedOutput", "_DynamoCompatibleOwedOutput"),
            (exit_mod, "ExitDecision", "_DynamoCompatibleExitDecision"),
            (stream_mod, "UnreducedOutput", "_DynamoCompatibleUnreducedOutput"),
            (prepare_mod, "UnreducedOutput", "_DynamoCompatibleUnreducedOutput"),
            (access_mod, "UnreducedOutput", "_DynamoCompatibleUnreducedOutput"),
            (exit_mod, "UnreducedOutput", "_DynamoCompatibleUnreducedOutput"),
            (output_mod, "UnreducedOutput", "_DynamoCompatibleUnreducedOutput"),
            (allreduce_mod, "UnreducedOutput", "_DynamoCompatibleUnreducedOutput"),
            (cutedsl_mod, "UnreducedOutput", "_DynamoCompatibleUnreducedOutput"),
        ]
        original_refs = [
            (module, name, getattr(module, name)) for module, name, _ in patched_refs
        ]
        original_replace = msgspec.structs.replace

        with cpu_graph_runner_module.cpu_graph_patch_boundary_classes():
            capture_refs = [getattr(module, name) for module, name, _ in patched_refs]
            for (module, name, expected_name), actual in zip(
                patched_refs, capture_refs
            ):
                self.assertIs(
                    actual,
                    getattr(cpu_graph_runner_module, expected_name),
                    f"unexpected patched class for {module.__name__}.{name}",
                )

            with cpu_graph_runner_module.cpu_graph_patch_boundary_classes():
                nested_refs = [
                    getattr(module, name) for module, name, _ in patched_refs
                ]
                self.assertEqual(capture_refs, nested_refs)

        self.assertIs(msgspec.structs.replace, original_replace)
        for module, name, original in original_refs:
            self.assertIs(getattr(module, name), original)

        with cpu_graph_runner_module.cpu_graph_patch_boundary_classes():
            replay_refs = [getattr(module, name) for module, name, _ in patched_refs]
            self.assertEqual(capture_refs, replay_refs)

        self.assertIs(msgspec.structs.replace, original_replace)
        for module, name, original in original_refs:
            self.assertIs(getattr(module, name), original)

    def test_prefill_graph_dispatches_inner_language_model(self):
        class ToyLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Identity()])

            def forward(self, input_ids, positions, forward_batch, input_embeds=None):
                return input_embeds if input_embeds is not None else input_ids

        class ToyOuterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = ToyLanguageModel()
                self.pp_group = SimpleNamespace(is_last_rank=True)

            def forward(self, input_ids, positions, forward_batch):
                return self.model(
                    input_ids=None,
                    positions=positions,
                    forward_batch=forward_batch,
                    input_embeds=input_ids,
                )

        runner = CPUGraphRunner.__new__(CPUGraphRunner)
        runner.is_generation = True
        runner.model_runner = SimpleNamespace(model=ToyOuterModel())
        inner_model = runner._get_prefill_graph_model()
        original_forward = inner_model.forward
        dispatcher = cpu_graph_runner_module._install_prefill_forward_dispatcher(
            inner_model
        )
        compiled_calls = []

        def compiled_forward(*args, **kwargs):
            compiled_calls.append((args, kwargs))
            return kwargs["input_embeds"] + 1

        input_ids = torch.ones(2)
        positions = torch.zeros(2)
        forward_batch = SimpleNamespace()
        eager_output = runner.model_runner.model(input_ids, positions, forward_batch)
        with cpu_graph_runner_module._use_prefill_forward(dispatcher, compiled_forward):
            output = runner.model_runner.model(input_ids, positions, forward_batch)

        self.assertTrue(torch.equal(eager_output, input_ids))
        self.assertTrue(torch.equal(output, torch.ones(2) + 1))
        self.assertEqual(len(compiled_calls), 1)
        self.assertIs(runner.model_runner.model.model, inner_model)
        self.assertIs(inner_model.forward, dispatcher)
        self.assertIs(dispatcher.original_forward.__self__, original_forward.__self__)
        self.assertIs(dispatcher.original_forward.__func__, original_forward.__func__)

    def test_prefill_dispatcher_is_stable_for_decode_compile(self):
        class ToyLanguageModel(torch.nn.Module):
            def forward(self, input_ids, positions, forward_batch):
                return input_ids.sin() + positions

        class ToyOuterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = ToyLanguageModel()

            def forward(self, input_ids, positions, forward_batch):
                return self.model(input_ids, positions, forward_batch)

        model = ToyOuterModel()
        cpu_graph_runner_module._install_prefill_forward_dispatcher(model.model)
        compile_count = 0

        def backend(graph_module, example_inputs):
            nonlocal compile_count
            compile_count += 1
            return graph_module.forward

        compiled = torch.compile(model, backend=backend, dynamic=True)
        for num_tokens in (4, 7, 9, 4):
            input_ids = torch.arange(num_tokens, dtype=torch.float32)
            output = compiled(
                input_ids,
                torch.ones(num_tokens),
                SimpleNamespace(),
            )
            self.assertTrue(torch.allclose(output, input_ids.sin() + 1))

        self.assertEqual(compile_count, 1)

    def test_cpu_graph_transformer_layer_has_no_graph_break(self):
        model = _MiniTransformerBlock()

        num_tokens = 8
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.randn(num_tokens, 2048, dtype=torch.bfloat16),
            positions=torch.arange(num_tokens, dtype=torch.int64),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int64),
            out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens_sum=num_tokens,
        )

        with cpu_graph_patch_boundary_classes():
            with forward_context(ForwardContext(attn_backend=None), forward_batch=fb):
                explanation = torch._dynamo.explain(model)(
                    fb.input_ids, fb.positions, fb
                )

        self.assertEqual(
            explanation.graph_count,
            1,
            f"Expected exactly 1 compiled graph, got {explanation.graph_count}. "
            f"Break reasons: {explanation.break_reasons}",
        )
        self.assertEqual(
            len(explanation.break_reasons),
            0,
            f"Unexpected graph break(s) detected: {explanation.break_reasons}",
        )

    def test_cpu_graph_dynamic_shapes_have_no_unexpected_recompilation(self):
        model = _MiniTransformerBlock()

        compile_count = [0]

        def count_backend(graph_module, example_inputs):
            compile_count[0] += 1
            return graph_module.forward

        with cpu_graph_patch_boundary_classes():
            # A. Prefill dynamic token sequence length changes (8 -> 16 -> 24 -> 8)
            torch._dynamo.reset()
            compile_count[0] = 0
            compiled_prefill = torch.compile(model, backend=count_backend, dynamic=True)
            for n in (8, 16, 24, 8):
                fb = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=torch.randn(n, 2048, dtype=torch.bfloat16),
                    positions=torch.arange(n, dtype=torch.int64),
                    seq_lens=torch.tensor([n], dtype=torch.int64),
                    out_cache_loc=torch.arange(n, dtype=torch.int64),
                    req_pool_indices=torch.tensor([0], dtype=torch.int64),
                    seq_lens_sum=n,
                )
                _mark_dynamic(fb.input_ids, fb.positions, max_value=64)
                with forward_context(
                    ForwardContext(attn_backend=None), forward_batch=fb
                ):
                    _ = compiled_prefill(fb.input_ids, fb.positions, fb)
                self.assertEqual(
                    compile_count[0],
                    1,
                    f"Unexpected recompile on prefill token count {n}: count={compile_count[0]}",
                )

            # B. Decode dynamic batch size changes (2 -> 3 -> 4 -> 2)
            torch._dynamo.reset()
            compile_count[0] = 0
            compiled_decode = torch.compile(model, backend=count_backend, dynamic=True)
            for bs in (2, 3, 4, 2):
                fb = ForwardBatch(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=bs,
                    input_ids=torch.randn(bs, 2048, dtype=torch.bfloat16),
                    positions=torch.zeros(bs, dtype=torch.int64),
                    seq_lens=torch.ones(bs, dtype=torch.int64),
                    out_cache_loc=torch.arange(bs, dtype=torch.int64),
                    req_pool_indices=torch.arange(bs, dtype=torch.int64),
                    seq_lens_sum=bs,
                )
                _mark_dynamic(fb.input_ids, fb.positions, max_value=16)
                with forward_context(
                    ForwardContext(attn_backend=None), forward_batch=fb
                ):
                    _ = compiled_decode(fb.input_ids, fb.positions, fb)
                self.assertEqual(
                    compile_count[0],
                    1,
                    f"Unexpected recompile on decode batch size {bs}: count={compile_count[0]}",
                )

    def test_cpu_graph_default_decode_buckets_are_generated(self):
        args = self._resolve_cpu_graph_args()

        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 32)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 32)
        self.assertEqual(args.torch_compile_max_bs, 32)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

    def test_cpu_prefill_graph_requires_explicit_full_backend(self):
        args = self._resolve_cpu_graph_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.FULL, max_bs=512)
            )
        )

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.FULL)
        self.assertEqual(
            args.cuda_graph_config.prefill.bs,
            generate_prefill_cuda_graph_batch_sizes(
                args.cuda_graph_config.prefill.max_bs
            ),
        )

    def test_cpu_decode_graph_size_options_have_expected_precedence(self):
        args = self._resolve_cpu_graph_args(torch_compile_max_bs=4)
        self.assertEqual(args.torch_compile_max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 4)

        args = self._resolve_cpu_graph_args(
            cuda_graph_bs_decode=[1, 2, 4],
        )
        self.assertEqual(args.torch_compile_max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.bs, [1, 2, 4])

        # Explicit decode.max_bs overrides default torch_compile_max_bs (32)
        args = self._resolve_cpu_graph_args(
            cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(max_bs=4))
        )
        self.assertEqual(args.torch_compile_max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 4)

        args = self._resolve_cpu_graph_args(
            cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(max_bs=64))
        )
        self.assertEqual(args.torch_compile_max_bs, 64)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 64)
        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 64)

    def test_fused_experts_cpu_fake_dispatch_omits_optional_activation(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        cpu_graph_runner_module.register_fake_ops(1)
        hidden = torch.randn(2, 64, dtype=torch.bfloat16)
        w1 = torch.randn(4, 64, 128, dtype=torch.bfloat16)
        w2 = torch.randn(4, 128, 64, dtype=torch.bfloat16)
        topk_weights = torch.ones(2, 2, dtype=torch.float32)
        topk_ids = torch.zeros(2, 2, dtype=torch.int32)

        with FakeTensorMode(allow_non_fake_inputs=True):
            out = torch.ops.sgl_kernel.fused_experts_cpu(
                hidden,
                w1,
                w2,
                topk_weights,
                topk_ids,
                False,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                True,
            )
            self.assertEqual(out.shape, hidden.shape)
            self.assertEqual(out.dtype, torch.bfloat16)

    def test_cpu_graph_rejects_unsupported_backends(self):
        model_runner = SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            _uses_cpu_graph_attention_backend=lambda: True,
        )
        with (
            patch.object(
                graph_setup,
                "get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=True)
                ),
            ),
            patch.object(
                graph_setup,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=CudaGraphConfig(
                            prefill=PhaseConfig(backend=Backend.BREAKABLE),
                            decode=PhaseConfig(backend=Backend.FULL),
                        )
                    )
                ),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "Unsupported CPU prefill graph backend"
            ):
                graph_setup.capture_prefill_graph(
                    model_runner=model_runner,
                    eager_runner=object(),
                )

    def test_cpu_graph_prefill_only_operation(self):
        model_runner = SimpleNamespace(
            device="cpu",
            is_generation=True,
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
            _uses_cpu_graph_attention_backend=lambda: True,
            prefill_cuda_graph_runner=None,
        )
        with (
            patch.object(
                graph_setup,
                "get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=True)
                ),
            ),
            patch.object(
                graph_setup,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(
                        cuda_graph_config=CudaGraphConfig(
                            prefill=PhaseConfig(backend=Backend.FULL, max_bs=128),
                            decode=PhaseConfig(backend=Backend.DISABLED),
                        )
                    )
                ),
            ),
            patch.object(
                graph_setup, "CPUGraphRunner", return_value=object()
            ) as mock_runner_cls,
        ):
            prefill_cap = graph_setup.capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=object(),
            )
            self.assertIsNotNone(prefill_cap.runner)
            model_runner.prefill_cuda_graph_runner = prefill_cap.runner

            decode_cap = graph_setup.capture_decode_graph(
                model_runner=model_runner,
                prefill_runner=prefill_cap.runner,
            )
            # When decode is disabled, decode runner should be None
            self.assertIsNone(decode_cap.runner)

    def test_cpu_graph_compile_backend_can_be_customized(self):
        from sglang.srt.environ import envs
        from sglang.srt.model_executor.cpu_graph_runner import patch_model

        class ToyModel(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        model = ToyModel()
        dummy_tp = SimpleNamespace(ca_comm=None)
        inp = torch.tensor([2.0])

        for backend_name in ("inductor", "eager"):
            with envs.SGLANG_CPU_GRAPH_COMPILE_BACKEND.override(backend_name):
                with patch_model(
                    model, enable_compile=True, tp_group=dummy_tp, dynamic=False
                ) as compiled_fwd:
                    out = compiled_fwd(inp)
                    self.assertEqual(out.item(), 3.0)

    def test_cpu_prefill_graph_requires_intel_amx_and_torch_compile(self):
        eager_runner = object()
        model_runner = SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            prefill_attention_backend_str="intel_amx",
            decode_attention_backend_str="intel_amx",
            _uses_cpu_graph_attention_backend=lambda: True,
        )

        with (
            patch.object(graph_setup, "CPUGraphRunner", return_value=eager_runner),
            patch.object(
                graph_setup,
                "get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=True)
                ),
            ),
        ):
            capture = graph_setup.capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=object(),
            )

        self.assertIs(capture.runner, eager_runner)

        model_runner._uses_cpu_graph_attention_backend = lambda: False
        with (
            patch.object(graph_setup, "CPUGraphRunner", return_value=eager_runner),
            patch.object(
                graph_setup,
                "get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=True)
                ),
            ),
        ):
            capture = graph_setup.capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=eager_runner,
            )

        self.assertIsNone(capture.runner)

    def test_encoder_decoder_decode_graph_reuses_cpu_runner(self):
        cpu_graph_runner = CPUGraphRunner.__new__(CPUGraphRunner)
        model_runner = SimpleNamespace(
            device="cpu",
            is_generation=True,
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
            server_args=SimpleNamespace(
                model_impl="auto",
                disaggregation_mode="null",
            ),
            model_config=SimpleNamespace(is_encoder_decoder=True),
        )

        with (
            patch.object(
                graph_setup,
                "get_flags",
                return_value=SimpleNamespace(
                    capture=SimpleNamespace(enable_torch_compile=True)
                ),
            ),
            patch.object(
                graph_setup,
                "get_model",
                return_value=SimpleNamespace(model_impl="auto"),
            ),
        ):
            capture = graph_setup.capture_decode_graph(
                model_runner=model_runner,
                prefill_runner=cpu_graph_runner,
            )

        self.assertIs(capture.runner, cpu_graph_runner)

    def test_prefill_padding_bucket_uses_ratio_and_token_limits(self):
        runner = _make_cpu_graph_runner()

        self.assertEqual(runner._get_prefill_padding_bucket(325), 390)
        self.assertEqual(runner._get_prefill_padding_bucket(324), None)
        self.assertEqual(runner._get_prefill_padding_bucket(100), 128)

        runner.disable_padding = True
        self.assertIsNone(runner._get_prefill_padding_bucket(325))

    def test_prefill_dummy_batch_splits_tokens_and_uses_kernel_dtypes(self):
        runner = _make_cpu_graph_runner()
        runner.device = torch.device("cpu")
        runner.model_runner = SimpleNamespace(
            model=SimpleNamespace(is_mrope_enabled=False),
            model_config=SimpleNamespace(model_is_mrope=False),
            spec_algorithm=object(),
        )

        batch = runner._build_prefill_dummy_forward_batch(num_tokens=7, bs=3)

        self.assertEqual(batch.batch_size, 3)
        self.assertEqual(batch.input_ids.shape, (7,))
        self.assertEqual(batch.extend_seq_lens.tolist(), [2, 2, 3])
        self.assertEqual(batch.extend_start_loc.tolist(), [0, 2, 4])
        self.assertEqual(batch.positions.tolist(), [0, 1, 0, 1, 0, 1, 2])
        self.assertEqual(batch.extend_seq_lens.dtype, torch.int32)
        self.assertEqual(batch.extend_start_loc.dtype, torch.int32)

    def test_dynamic_prefill_axes_use_configured_token_bound(self):
        runner = CPUGraphRunner.__new__(CPUGraphRunner)
        runner.prefill_max_bs = 4
        runner.dynamic_prefill_max_tokens = 1024
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.zeros(2, dtype=torch.int64),
            seq_lens=torch.zeros(2, dtype=torch.int64),
            extend_seq_lens=torch.zeros(2, dtype=torch.int32),
            extend_prefix_lens=torch.zeros(2, dtype=torch.int32),
            extend_start_loc=torch.zeros(2, dtype=torch.int32),
            input_ids=torch.zeros(512, dtype=torch.int64),
            positions=torch.zeros(512, dtype=torch.int64),
            out_cache_loc=torch.zeros(512, dtype=torch.int64),
            mrope_positions=None,
        )

        with patch.object(cpu_graph_runner_module, "_mark_dynamic") as mark_dynamic:
            runner._mark_dynamic_token_axis(forward_batch)

        mark_dynamic.assert_called_once_with(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch.out_cache_loc,
            max_value=1024,
        )

    def test_dynamic_decode_axes_use_request_pool_bound(self):
        runner = CPUGraphRunner.__new__(CPUGraphRunner)
        runner.dynamic_decode_max_bs = 8
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
            mrope_positions=torch.zeros(3, 4, dtype=torch.int64),
        )

        with patch.object(
            cpu_graph_runner_module, "_mark_dynamic_axis"
        ) as mark_dynamic_axis:
            runner._mark_dynamic_decode_inputs(forward_batch)

        self.assertEqual(mark_dynamic_axis.call_count, 3)
        for call in mark_dynamic_axis.call_args_list:
            self.assertEqual(call.kwargs, {"max_value": 8})

    def test_prefill_padding_trims_full_hidden_states(self):
        runner = _make_cpu_graph_runner()
        runner.capture_num_tokens = [8]
        runner.prefill_max_num_tokens = 8
        runner.prefill_graphs = {8: object()}
        forward_batch = _make_cpu_prefill_forward_batch()
        captured_batch = []

        def run_prefill_once(compiled_fn, batch):
            del compiled_fn
            captured_batch.append(batch)
            return LogitsProcessorOutput(
                next_token_logits=torch.zeros(2, 4),
                hidden_states=torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3),
            )

        with patch.object(
            CPUGraphRunner, "_run_prefill_once", side_effect=run_prefill_once
        ):
            output = runner._execute_prefill_graph(forward_batch)

        self.assertEqual(output.hidden_states.shape, (6, 3))
        self.assertEqual(captured_batch[0].input_ids.shape, (8,))
        self.assertEqual(captured_batch[0].positions.shape, (8,))
        self.assertEqual(captured_batch[0].out_cache_loc.shape, (8,))
        self.assertTrue(
            torch.equal(captured_batch[0].input_ids[:6], forward_batch.input_ids)
        )
        self.assertTrue(torch.equal(captured_batch[0].input_ids[6:], torch.zeros(2)))
        self.assertEqual(forward_batch.input_ids.shape, (6,))

    def test_dynamic_decode_graph_is_selected_for_uncaptured_batch(self):
        runner = CPUGraphRunner.__new__(CPUGraphRunner)
        runner.is_encoder_decoder = False
        runner.graphs = {2: object()}
        runner.graphs_cross = {}
        runner.disable_padding = True
        runner.max_bs = 2
        runner.enable_dynamic_graph = True
        runner.decode_dynamic_graphs = {True: object()}

        forward_batch = SimpleNamespace(
            batch_size=3,
            forward_mode=ForwardMode.DECODE,
        )

        self.assertTrue(runner.can_run_graph(forward_batch))

    def test_encoder_decoder_without_prefill_skips_decode_rewarm(self):
        runner = CPUGraphRunner.__new__(CPUGraphRunner)
        runner.is_generation = True
        runner.enable_dynamic_graph = True
        runner.enable_torch_compile = True
        runner.is_encoder_decoder = True
        runner.capture_num_tokens = []
        runner.decode_dynamic_graphs = {}
        runner.model_runner = SimpleNamespace(req_to_token_pool=SimpleNamespace())

        with (
            patch.object(
                runner,
                "_compile_decode_dynamic_graph",
                side_effect=lambda skip_cross_attention: skip_cross_attention,
            ) as compile_graph,
            patch.object(runner, "_rewarm_decode_dynamic_graphs") as rewarm,
        ):
            runner._compile_decode_dynamic_graphs()

        self.assertEqual(compile_graph.call_count, 2)
        self.assertEqual(
            [call.kwargs for call in compile_graph.call_args_list],
            [
                {"skip_cross_attention": True},
                {"skip_cross_attention": False},
            ],
        )
        rewarm.assert_not_called()

    def test_dynamic_prefill_graph_is_selected_for_uncaptured_tokens(self):
        runner = _make_cpu_graph_runner()
        runner.prefill_dynamic_graph = object()
        runner.prefill_dynamic_batch_template = None
        runner.enable_torch_compile = False
        forward_batch = _make_cpu_prefill_forward_batch(num_tokens=129)
        selected = []

        def run_prefill_once(compiled_fn, batch):
            selected.append((compiled_fn, batch))
            return object()

        with patch.object(runner, "_run_prefill_once", side_effect=run_prefill_once):
            runner._execute_prefill_graph(forward_batch)

        self.assertIs(selected[0][0], runner.prefill_dynamic_graph)
        self.assertIs(selected[0][1], forward_batch)

    def test_prefill_padding_bucket_is_preferred_over_dynamic_fallback(self):
        runner = _make_cpu_graph_runner()
        padded_graph = object()
        dynamic_graph = object()
        runner.prefill_graphs = {256: padded_graph}
        runner.prefill_dynamic_graph = dynamic_graph
        runner.prefill_dynamic_batch_template = None
        runner.enable_torch_compile = False

        # 240 tokens can be padded to 256 (diff 16 <= 64 and <= 0.20)
        forward_batch = _make_cpu_prefill_forward_batch(num_tokens=240)
        selected = []

        def run_prefill_once(compiled_fn, batch):
            selected.append((compiled_fn, batch))
            return object()

        with patch.object(runner, "_run_prefill_once", side_effect=run_prefill_once):
            runner._execute_prefill_graph(forward_batch)

        # Should select the padded static bucket, NOT dynamic_graph
        self.assertIs(selected[0][0], padded_graph)
        self.assertEqual(selected[0][1].input_ids.shape[0], 256)

    @intel_amx_benchmark(
        extra_args=[
            "--batch-size",
            "1",
            "--mem-fraction-static",
            "0.05",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "2",
            "--cuda-graph-bs-decode",
            "1",
            "2",
            "--cuda-graph-backend-prefill",
            "full",
            "--cuda-graph-bs-prefill",
            "128",
        ],
        min_throughput=7,
    )
    def test_latency_torch_compile_cpu(self):
        return DEFAULT_MODEL_NAME_FOR_TEST

    def test_mmlu_torch_compile_cpu(self):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        env = copy.deepcopy(os.environ)
        env["SGLANG_CPU_OMP_THREADS_BIND"] = "all"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--disable-overlap-schedule",
                "--enable-torch-compile",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "--cuda-graph-backend-prefill",
                "full",
                "--cuda-graph-bs-prefill",
                "128",
                "--tp",
                f"{n_numa_node}",
            ],
            env=env,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_sgl_eval(args)
            if is_in_ci():
                self.assertGreater(metrics["score"], 0.45)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
