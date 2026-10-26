"""Executor-owned routing bounds survive retained callbacks and padded CPU replay."""

import importlib
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopK, fused_topk_torch_native
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

breakable_graph = importlib.import_module(
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph"
)
breakable_backend = importlib.import_module(
    "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend"
)


def _batch(size, live, *, prepared=True):
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=size,
        input_ids=torch.arange(size),
        req_pool_indices=torch.arange(size),
        seq_lens=torch.ones(size, dtype=torch.int32),
        out_cache_loc=torch.arange(size),
        seq_lens_sum=size,
        positions=torch.arange(size),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    batch.global_num_token_non_padded = torch.tensor(live, dtype=torch.int32)
    batch.global_num_token_non_padded_cpu = live
    batch.num_token_non_padded = batch.global_num_token_non_padded if prepared else None
    batch.global_dp_buffer_len = size
    batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    return batch


def _attention_backend():
    return SimpleNamespace(
        init_forward_metadata=lambda batch: None,
        init_forward_metadata_out_graph=lambda batch, **kwargs: None,
        init_forward_metadata_in_graph=lambda batch: None,
        shared_read_ends=lambda mode: SharedReadEnds.UNKNOWN,
    )


class _RoutingModel:
    shared_experts_config = SharedExpertsConfig()

    def __init__(self, *, proxy_output):
        self.topk = TopK(
            2,
            layer_id=0,
            is_draft=True,
            custom_routing_function=fused_topk_torch_native,
        )
        self.proxy_output = proxy_output
        self.seen = []
        self.fail = False

    def forward(self, input_ids, positions, batch):
        rows = len(input_ids)
        bound, _ = self.topk.routing_inputs()
        ids = self.topk.forward_native(
            torch.ones((rows, 2)),
            torch.tensor([[9.0, 8.0, 1.0, 0.0]]).repeat(rows, 1),
        ).topk_ids
        self.seen.append(
            (batch, bound, get_forward_context().attn_backend, ids.clone())
        )
        if self.fail:
            raise RuntimeError("replayed model failed")
        if self.proxy_output:
            return PPProxyTensors({"routing_ids": ids})
        return LogitsProcessorOutput(next_token_logits=ids.float())


class _CpuStream:
    def wait_stream(self, other):
        pass


class _NoDeviceGraph:
    """Only accelerator graph recording/replay is replaced; eager breaks remain real."""

    def capture_begin(self, **kwargs):
        pass

    def capture_end(self):
        pass

    def replay(self):
        pass


class TestGraphRoutingContext(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = get_context().override_server_args(
            enable_eplb=False, ep_dispatch_algorithm=None, debug_cuda_graph=True
        )
        config.install()
        self.addCleanup(config.restore)
        initialize_moe_config()
        topology = get_parallel().override(
            tp_size=4,
            moe_ep_size=4,
            attn_dp_size=1,
            attn_cp_size=1,
            attn_tp_size=4,
            attn_tp_rank=0,
            tp_group=SimpleNamespace(barrier=lambda: None, world_size=4),
        )
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        for name in ("_is_cuda", "_is_hip", "_use_aiter"):
            scope = patch.object(topk_module, name, False)
            scope.start()
            self.addCleanup(scope.stop)

    def _model_runner(self, *, device, model):
        runner = ModelRunner.__new__(ModelRunner)
        runner.device = device
        runner.attn_backend = _attention_backend()
        runner.hisparse_coordinator = None
        runner.device_timer = None
        runner.is_draft_worker = False
        runner.spec_algorithm = SpeculativeAlgorithm.NONE
        runner.capture_tail_hooks = []
        runner.capture_head_hooks = []
        runner.canary_manager = None
        runner.model = model
        return runner

    def _debug_runner(self):
        model = _RoutingModel(proxy_output=True)
        model_runner = self._model_runner(device="cuda", model=model)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = model_runner
        runner.captured_req_width = 1
        runner.ragged_verify_mode = False
        runner.enable_pdmux = False
        runner.pp_size = 1
        runner.dllm_uses_input_embeds = False
        runner.in_graph_metadata_prep_done = None
        runner.is_dllm = False
        static = _batch(4, 4)
        runner.attn_backend = captured_backend = _attention_backend()
        runner.capture_prepare = lambda *args, **kwargs: (
            static,
            captured_backend,
            None,
        )
        runner.tbo_plugin = SimpleNamespace(
            capture_one_batch_size=lambda *args, **kw: None
        )
        runner.deepep_adapter = SimpleNamespace(capture=lambda **kwargs: None)
        stream = _CpuStream()
        runner.device_module = SimpleNamespace(
            Stream=_CpuStream,
            current_stream=lambda *args: stream,
            synchronize=lambda: None,
            is_current_stream_capturing=lambda: False,
        )
        runner.backend = BreakableCudaGraphBackend(runner, debug_eager=True)
        runner.can_run_graph = lambda batch: True

        def load_batch(batch, pp_proxy_tensors):
            # Substitute only the accelerator input copies/shape selection.
            static.num_token_non_padded.copy_(batch.global_num_token_non_padded)
            runner.raw_bs = runner.raw_num_token = batch.batch_size
            runner.bs = 4
            runner._replay_graph_key = runner._make_graph_key(4)

        runner.load_batch = load_batch
        model_runner.decode_cuda_graph_runner = runner
        for target, name, replacement in (
            (torch.cuda, "CUDAGraph", _NoDeviceGraph),
            (breakable_graph, "get_device_module", lambda: runner.device_module),
            (breakable_backend, "graph_pool_capture_scope", nullcontext),
            (breakable_backend, "graph_pool_replay_scope", nullcontext),
        ):
            scope = patch.object(target, name, replacement)
            scope.start()
            self.addCleanup(scope.stop)
        with envs.SGLANG_ENABLE_GRAPH_POOL_PRECARVE.override(False):
            runner.capture_one_shape(4, model.forward)
        self.assertEqual(
            len(runner.backend._graphs[runner._make_graph_key(4)]._break_fns), 1
        )
        return model_runner, model, static, captured_backend

    def _cpu_runner(self):
        model = _RoutingModel(proxy_output=False)
        model_runner = self._model_runner(device="cpu", model=model)
        runner = CPUGraphRunner.__new__(CPUGraphRunner)
        runner.model_runner = model_runner
        runner.is_encoder_decoder = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_bs = [4]
        runner.captured_req_width = 1
        runner.seq_len_fill_value = 1
        runner.graphs = {4: model.forward}
        runner.graphs_cross = {}
        static = _batch(4, 4)
        runner.captured_forward_batches = {4: static}
        runner.can_run_graph = lambda batch: True
        model_runner.decode_cuda_graph_runner = runner
        return model_runner, model, static, model_runner.attn_backend

    def _capture_cpu_runner(self):
        model_runner, model, _, backend = self._cpu_runner()
        runner = model_runner.decode_cuda_graph_runner
        runner.capture_bs = [1, 4]
        runner.capture_forward_mode = ForwardMode.DECODE
        runner.captured_forward_batches = {}
        runner.captured_forward_batches_cross = {}
        runner.input_ids = torch.arange(4)
        runner.req_pool_indices = torch.arange(4)
        runner.seq_lens = torch.ones(4, dtype=torch.int32)
        runner.out_cache_loc = torch.arange(4)
        runner.positions = torch.arange(4)
        runner.mrope_positions = torch.zeros((3, 4), dtype=torch.int64)
        runner.num_token_non_padded = torch.zeros(1, dtype=torch.int64)
        model_runner.kv_index_translator = SimpleNamespace(
            bind_runner_slots=lambda batch: None
        )
        backend.init_forward_metadata_capture_cpu_graph = lambda *args: None
        for size in reversed(runner.capture_bs):
            runner.graphs[size], _ = runner.capture_one_batch_size(
                size, model.forward, skip_cross_attention=True
            )
        return runner, model, backend

    def test_cpu_capture_without_tracking_keeps_live_routes(self):
        """Padded CPU replays keep every live route when count tracking is disabled."""
        with get_parallel().override(
            tp_size=1,
            moe_ep_size=1,
            attn_tp_size=1,
            tp_group=SimpleNamespace(barrier=lambda: None, world_size=1),
        ):
            self.assertFalse(enable_num_token_non_padded())
            runner, model, backend = self._capture_cpu_runner()
            for size in (3, 4, 2, 1):
                with self.subTest(size=size):
                    incoming = _batch(size, size, prepared=False)
                    incoming.global_num_token_non_padded = None
                    with forward_context(
                        ForwardContext(attn_backend=backend), forward_batch=incoming
                    ):
                        eager = model.forward(
                            incoming.input_ids, incoming.positions, incoming
                        )
                    output = runner.execute(incoming)
                    torch.testing.assert_close(
                        output.next_token_logits, eager.next_token_logits
                    )
                    prepared, bound, _, ids = model.seen[-1]
                    self.assertIsNone(bound)
                    self.assertTrue((ids >= 0).all())
                    self.assertEqual(prepared.batch_size, 4 if size > 1 else 1)
                    self.assertFalse(has_forward_context())
            for captured in runner.captured_forward_batches.values():
                self.assertIsNone(captured.num_token_non_padded)

    def _assert_replay(self, model_runner, model, static, backend):
        persistent_bound = static.num_token_non_padded
        for live in (3, 1, 0):
            incoming = _batch(live, live, prepared=False)
            model_runner._forward_raw(incoming, None)
            batch, bound, active_backend, ids = model.seen[-1]
            self.assertIs(batch, static)
            self.assertIs(bound, persistent_bound)
            self.assertEqual(int(bound), live)
            self.assertIs(active_backend, backend)
            self.assertEqual(ids[:live].tolist(), [[0, 1]] * live)
            self.assertEqual(ids[live:].tolist(), [[-1, -1]] * (4 - live))
            self.assertIsNone(incoming.num_token_non_padded)
            self.assertFalse(has_forward_context())

    def test_debug_saved_forward_rebinds_static_batch(self):
        self._assert_replay(*self._debug_runner())

    def test_cpu_replay_rebinds_actual_prepared_batch(self):
        self._assert_replay(*self._cpu_runner())

    def _assert_failure_restores_parent(self, model_runner, model, static, backend):
        parent = _batch(1, 1)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=parent
        ):
            previous = get_forward_context()
            model.fail = True
            with self.assertRaisesRegex(RuntimeError, "replayed model failed"):
                model_runner._forward_raw(_batch(3, 3, prepared=False), None)
            self.assertIs(model.seen[-1][0], static)
            self.assertIs(model.seen[-1][1], static.num_token_non_padded)
            self.assertIs(model.seen[-1][2], backend)
            self.assertIs(get_forward_context(), previous)
            self.assertIs(
                previous.moe_num_token_non_padded(), parent.num_token_non_padded
            )
        self.assertFalse(has_forward_context())

    def test_debug_saved_forward_failure_restores_parent(self):
        self._assert_failure_restores_parent(*self._debug_runner())

    def test_cpu_replay_failure_restores_parent(self):
        self._assert_failure_restores_parent(*self._cpu_runner())


if __name__ == "__main__":
    unittest.main()
