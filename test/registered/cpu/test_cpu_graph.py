"""
Usage:
python3 -m unittest test_cpu_graph.TestCPUGraph.test_mmlu_torch_compile_cpu
"""

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.cpu_graph_runner as cpu_graph_runner_module
import sglang.srt.model_executor.model_runner_components.cuda_graph_setup as graph_setup
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
    check_cpu_graph_backend,
    default_prefill_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_cpu_ids_by_node, kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    intel_amx_benchmark,
    is_in_ci,
    popen_launch_server,
)

register_cpu_ci(est_time=10, suite="base-b-test-cpu")


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


class TestCPUGraph(CustomTestCase):

    def _resolve_cpu_graph_args(self, **overrides):
        args = ServerArgs(model_path="dummy", device="cpu", **overrides)
        args._parse_cuda_graph_config()
        model_config = SimpleNamespace(is_multimodal=False)
        args.model_config = model_config
        with patch.object(args, "use_mla_backend", return_value=False), patch.object(
            args, "get_model_config", return_value=model_config
        ):
            args._handle_gpu_memory_settings(None)
        return args

    def test_cpu_prefill_backend_defaults_to_full(self):
        self.assertEqual(default_prefill_backend(), Backend.FULL)

    def test_cpu_graph_backend_requires_intel_amx_for_both_phases(self):
        model_runner = SimpleNamespace(
            device="cpu",
            prefill_attention_backend_str="intel_amx",
            decode_attention_backend_str="intel_amx",
        )
        with patch(
            "sglang.srt.model_executor.cuda_graph_config.check_cuda_graph_backend",
            side_effect=lambda phase, backend: backend == Backend.FULL,
        ), patch(
            "sglang.srt.runtime_context.get_flags",
            return_value=SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=True)
            ),
        ):
            self.assertTrue(check_cpu_graph_backend(model_runner, "decode"))

        model_runner.decode_attention_backend_str = "torch_native"
        with patch(
            "sglang.srt.model_executor.cuda_graph_config.check_cuda_graph_backend",
            side_effect=lambda phase, backend: backend == Backend.FULL,
        ), patch(
            "sglang.srt.runtime_context.get_flags",
            return_value=SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=True)
            ),
        ):
            self.assertFalse(check_cpu_graph_backend(model_runner, "decode"))

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

    def test_cpu_graph_default_decode_and_prefill_buckets_are_generated(self):
        args = self._resolve_cpu_graph_args()

        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 32)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 32)
        self.assertEqual(args.torch_compile_max_bs, 32)
        self.assertEqual(
            args.cuda_graph_config.prefill.bs,
            args._generate_prefill_cuda_graph_batch_sizes(
                args.cuda_graph_config.prefill.max_bs
            ),
        )

    def test_cpu_decode_graph_size_options_have_expected_precedence(self):
        args = self._resolve_cpu_graph_args(torch_compile_max_bs=4)
        self.assertEqual(args.torch_compile_max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 4)

        args = self._resolve_cpu_graph_args(
            cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(max_bs=4))
        )
        self.assertEqual(args.torch_compile_max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 4)

        args = self._resolve_cpu_graph_args(
            torch_compile_max_bs=4,
            cuda_graph_max_bs_decode=8,
        )
        self.assertEqual(args.torch_compile_max_bs, 8)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 8)
        self.assertEqual(args.cuda_graph_config.decode.bs[-1], 8)

        args = self._resolve_cpu_graph_args(
            cuda_graph_max_bs_decode=8,
            cuda_graph_bs_decode=[1, 2, 4],
        )
        self.assertEqual(args.torch_compile_max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.max_bs, 4)
        self.assertEqual(args.cuda_graph_config.decode.bs, [1, 2, 4])

    def test_cpu_prefill_graph_requires_intel_amx_and_torch_compile(self):
        eager_runner = object()
        model_runner = SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            prefill_attention_backend_str="intel_amx",
            decode_attention_backend_str="intel_amx",
        )

        with patch(
            "sglang.srt.model_executor.cuda_graph_config.check_cuda_graph_backend",
            side_effect=lambda phase, backend: backend == graph_setup.Backend.FULL,
        ), patch.object(
            graph_setup, "CPUGraphRunner", return_value=eager_runner
        ), patch(
            "sglang.srt.runtime_context.get_flags",
            return_value=SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=True)
            ),
        ):
            capture = graph_setup.capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=object(),
            )

        self.assertIs(capture.runner, eager_runner)

        model_runner.prefill_attention_backend_str = "torch_native"
        with patch(
            "sglang.srt.model_executor.cuda_graph_config.check_cuda_graph_backend",
            side_effect=lambda phase, backend: backend == graph_setup.Backend.FULL,
        ), patch(
            "sglang.srt.runtime_context.get_flags",
            return_value=SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=True)
            ),
        ):
            capture = graph_setup.capture_prefill_graph(
                model_runner=model_runner,
                eager_runner=eager_runner,
            )

        self.assertIsNone(capture.runner)

    def test_cpu_prefill_graph_rejects_unsupported_backend(self):
        model_runner = SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            prefill_attention_backend_str="intel_amx",
            decode_attention_backend_str="intel_amx",
        )

        with patch(
            "sglang.srt.model_executor.cuda_graph_config.check_cuda_graph_backend",
            return_value=False,
        ), patch(
            "sglang.srt.runtime_context.get_flags",
            return_value=SimpleNamespace(
                capture=SimpleNamespace(enable_torch_compile=True)
            ),
        ):
            with self.assertRaisesRegex(ValueError, "only supports the 'full'"):
                graph_setup.capture_prefill_graph(
                    model_runner=model_runner,
                    eager_runner=object(),
                )

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

        with patch.object(
            graph_setup,
            "check_cpu_graph_backend",
            return_value=True,
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

    @intel_amx_benchmark(
        extra_args=[
            "--batch-size",
            "1",
            "--mem-fraction-static",
            "0.05",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "2",
            "--cuda-graph-bs",
            "2",
            "--cuda-graph-bs-prefill",
            "128",
            "--cuda-graph-backend-prefill",
            "full",
        ],
        min_throughput=7,
    )
    def test_latency_torch_compile_cpu(self):
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST

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
                "--cuda-graph-bs",
                "2",
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

            metrics = run_eval(args)
            if is_in_ci():
                self.assertGreater(metrics["score"], 0.45)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
