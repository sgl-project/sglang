"""CPU tests for refresh-sharing ownership, eligibility, and native fallback."""

import unittest
from array import array
from contextlib import contextmanager, nullcontext
from copy import copy
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.dllm import refresh_pair_serving as module
from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.refresh_pair_serving import (
    PairGraph,
    RefreshPairServing,
    discard_incompatible_prefetch,
    eligible_request,
    finish_paired_step,
)
from sglang.srt.layers.attention.flashinfer_block_pair_backend import (
    FlashInferBlockPairBackend,
    bind_owned_cache_view,
    split_pair_batch,
)
from sglang.srt.layers.attention.graph_utils import padded_extend_real_tokens
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.kv_loc_plan import IdSpace, KVLocPlan
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class StaticTranslator:
    is_translating = False

    def space(self, kind):
        return IdSpace(key="owned-test-pool")

    def plan(self, **kwargs):
        return KVLocPlan(source=self, **kwargs)


def source_batch(n=4, block=2):
    prefix = [4 * (i + 1) for i in range(n)]
    seq = torch.tensor(prefix, dtype=torch.int32) + block
    return NS(
        dllm_prompt_mask=None,
        batch_size=n,
        forward_mode=ForwardMode.DLLM_EXTEND,
        input_ids=torch.arange(1, n * block + 1),
        positions=torch.tensor([p + j for p in prefix for j in range(block)]),
        out_cache_loc=torch.arange(100, 100 + n * block),
        req_pool_indices=torch.arange(n, dtype=torch.int32),
        seq_lens=seq,
        seq_lens_cpu=seq.clone(),
        seq_lens_sum=int(seq.sum()),
        extend_seq_lens=torch.full((n,), block, dtype=torch.int32),
        extend_seq_lens_cpu=[block] * n,
        extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32),
        extend_prefix_lens_cpu=prefix,
        extend_logprob_start_lens_cpu=[0] * n,
        extend_start_loc=torch.arange(n, dtype=torch.int32) * block,
        extend_num_tokens=n * block,
        spec_info=None,
        encoder_lens=None,
        dllm_block_pair_size=None,
    )


def algorithm(block=2, threshold=0.9):
    algo = object.__new__(LowConfidence)
    algo.block_size, algo.mask_id, algo.threshold, algo.fdfo = (
        block,
        99,
        threshold,
        True,
    )
    return algo


def adapter(block=2):
    allocator = NS(
        alloc=Mock(side_effect=lambda n: torch.arange(1000, 1000 + n)), free=Mock()
    )
    runner = NS(
        kv_index_translator=StaticTranslator(),
        token_to_kv_pool_allocator=allocator,
        prepare_dummy_forward_batch=lambda b: b,
        model_config=NS(context_len=32768),
        decode_cuda_graph_capture_bs=(1, 2, 4),
        decode_cuda_graph_runner=object(),
        forward_pass_id=0,
        device_timer=None,
    )
    result = RefreshPairServing(runner, algorithm(block))
    result.graph = FakeGraph(runner, None)
    return result


def requests(source, block=2):
    result = []
    for i in range(source.batch_size):
        p = source.extend_prefix_lens_cpu[i]
        ids = source.input_ids.view(-1, block)[i].tolist()
        result.append(
            NS(
                sampling_params=NS(
                    ignore_eos=True,
                    stop=[],
                    stop_regex=None,
                    stop_token_ids=None,
                    max_new_tokens=256,
                ),
                return_logprob=False,
                dllm_algo_state=None,
                dllm_incomplete_ids=array("q", ids),
                extend_end=p + block,
                origin_input_ids=array("q", [7] * p),
                output_ids=array("q"),
                dllm_config=NS(block_size=block),
                dllm_pair_pending=None,
                full_untruncated_fill_ids=array("q", [7] * p + ids),
            )
        )
    return result


def native_request(sampling=None):
    config = DllmConfig(
        "LowConfidence", {}, 16, 151669, 1, first_done_first_out_mode=True
    )
    req = Req(
        "native-pair",
        "",
        array("q", [7] * 128),
        sampling
        if sampling is not None
        else SamplingParams(temperature=0, max_new_tokens=128),
        dllm_config=config,
        eos_token_ids={151643, 151645},
        vocab_size=152064,
    )
    req.prefix_len = 128
    req.extend_end = 144
    req.dllm_pair_context_limit = 1024
    req.dllm_incomplete_ids = array("q", [8] * 16)
    req.full_untruncated_fill_ids = req.origin_input_ids + req.dllm_incomplete_ids
    return req


class TestCacheViews(CustomTestCase):
    def test_new_plan_does_not_reuse_parent_write_window(self):
        source = source_batch(2, 16)
        translator = StaticTranslator()
        bind_owned_cache_view(source, translator)
        parent = source.kv_loc_plan
        old_slots = parent.write_physical.clone()
        source.out_cache_loc = torch.arange(2000, 2032)
        bind_owned_cache_view(source, translator)
        self.assertIsNot(source.kv_loc_plan, parent)
        self.assertTrue(torch.equal(parent.write_physical, old_slots))
        self.assertEqual(
            source.kv_loc_plan.write_physical.tolist(), list(range(2000, 2032))
        )
        self.assertIs(source.out_cache_loc, source.kv_loc_plan.write_physical)
        self.assertTrue(source.out_cache_loc_is_physical)

    def test_translated_pool_is_rejected_before_binding(self):
        source = source_batch(2)
        translator = StaticTranslator()
        translator.is_translating = True
        with self.assertRaisesRegex(ValueError, "static KV pool"):
            bind_owned_cache_view(source, translator)

    def test_prompt_mask_follows_request_rows(self):
        for n in (1, 2, 4):
            source = source_batch(n, 2)
            source.dllm_prompt_mask = torch.zeros(n, 2, dtype=torch.bool)
            source.dllm_prompt_mask[0, 0] = True
            a = adapter(2)
            with a.scratch_batch(source) as combined:
                current, future = split_pair_batch(combined, 2)
                self.assertTrue(
                    torch.equal(current.dllm_prompt_mask, source.dllm_prompt_mask)
                )
                self.assertFalse(future.dllm_prompt_mask.any())
                self.assertEqual(future.dllm_prompt_mask.shape, (n, 2))


class FakeGraph:
    def __init__(self, runner, combined):
        pass

    def __call__(self, combined):
        logits = torch.zeros(combined.input_ids.numel(), 100)
        logits[:, 8] = 20
        return NS(full_logits=logits)


class AdapterTests(CustomTestCase):
    def test_recapture_discards_predictions_in_current_and_later_batches(self):
        a = adapter()
        a.qualified = True
        a.runner.decode_cuda_graph_runner = object()
        for _ in range(2):
            source = source_batch(1)
            req = requests(source)[0]
            req.dllm_pair_pending = (4, [1, 2])
            req.dllm_pair_applied_offset = 4
            req.dllm_pair_future_width = 1
            req.dllm_incomplete_ids = array("q")
            self.assertIsNone(a.run(source, [req]))
            self.assertEqual(source.input_ids.tolist(), [99, 99])
            self.assertIsNone(req.dllm_pair_pending)
        self.assertTrue(a.disabled)
        self.assertIsNone(a.graph)
        a.runner.token_to_kv_pool_allocator.alloc.assert_not_called()

    def test_replay_records_forward_span_and_gpu_time(self):
        a = adapter()
        a.qualified = True
        entered = []

        @contextmanager
        def timer(metadata):
            entered.append(metadata)
            yield
            entered.append("finished")

        a.runner.device_timer = NS(wrap=timer)
        graph = a.graph

        def replay(batch):
            self.assertEqual(entered, [{"category": "dllm_extend"}])
            return graph(batch)

        a.graph = replay
        source = source_batch(1)
        with patch.object(module, "profile_range", return_value=nullcontext()) as span:
            a.run(source, requests(source))
        self.assertEqual(entered[-1], "finished")
        self.assertEqual(a.runner.forward_pass_id, 1)
        self.assertIn("DLLM_EXTEND", span.call_args.args[0])

    def test_numerical_gate_rejects_changed_confidence_decisions(self):
        a = adapter()
        with a.scratch_batch(source_batch(1)) as combined:
            reference = torch.zeros(2, 100)
            reference[:, 8] = 6.782
            candidate = torch.cat([reference, reference.clone()])
            candidate[2:, 8] += 0.02
            # Logits meet tolerance and greedy IDs agree, but .9 decisions differ.
            torch.testing.assert_close(candidate[2:], reference, atol=0.01, rtol=0.01)
            a.runner.forward = Mock(
                side_effect=[
                    NS(can_run_graph=True, logits_output=NS(full_logits=reference))
                    for _ in range(2)
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "confidence decisions"):
                a.qualify(combined, candidate)

    def test_numerical_gate_rejects_nonfinite_logits(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                a = adapter()
                with a.scratch_batch(source_batch(1)) as combined:
                    candidate = torch.zeros(4, 100)
                    candidate[0, 0] = value
                    a.runner.forward = Mock(
                        return_value=NS(
                            can_run_graph=True,
                            logits_output=NS(full_logits=torch.zeros(2, 100)),
                        )
                    )
                    with self.assertRaisesRegex(RuntimeError, "qualification failed"):
                        a.qualify(combined, candidate)

    def test_failed_numerical_gate_disables_pairing_and_uses_native_path(self):
        source = source_batch(1)
        a = adapter()
        a.qualify = Mock(
            side_effect=module.RefreshPairNumericalError("qualification failed")
        )
        reqs = requests(source)
        for _ in range(2):
            self.assertIsNone(a.run(source, reqs))
            self.assertIsNone(a.graph)
            self.assertTrue(a.disabled)
            self.assertIsNone(reqs[0].dllm_pair_pending)
            self.assertIsNone(a.scratch)
        a.qualify.assert_called_once()
        a.runner.token_to_kv_pool_allocator.free.assert_called_once()

    def test_qualified_graph_reused_and_future_ids_preserved(self):
        source = source_batch(1)
        a = adapter()
        a.qualify = Mock()
        reqs = requests(source)
        graph = a.graph
        for _ in range(2):
            output = a.run(source, reqs)
            self.assertEqual(output.full_logits.shape, (2, 100))
            self.assertEqual(reqs[0].dllm_pair_pending, (6, [8, 8]))
            self.assertEqual(reqs[0].dllm_pair_future_width, 1)
            self.assertIs(a.graph, graph)
        a.qualify.assert_called_once()
        self.assertEqual(a.runner.token_to_kv_pool_allocator.free.call_count, 2)

    def test_larger_batch_uses_native_path_without_scratch(self):
        for n in (2, 4):
            source = source_batch(n)
            a = adapter()
            self.assertIsNone(a.run(source, requests(source)))
            a.runner.token_to_kv_pool_allocator.alloc.assert_not_called()

    def test_pending_n1_prediction_is_discarded_before_larger_batch(self):
        source = source_batch(2)
        a = adapter()
        reqs = requests(source)
        req = reqs[0]
        req.dllm_pair_pending = (4, [1, 2])
        req.dllm_pair_applied_offset = 4
        req.dllm_pair_future_width = 1
        req.dllm_incomplete_ids = array("q")
        before_other = source.input_ids[2:].clone()
        self.assertIsNone(a.run(source, reqs))
        self.assertEqual(source.input_ids[:2].tolist(), [99, 99])
        self.assertTrue(torch.equal(before_other, source.input_ids[2:]))
        self.assertIsNone(req.dllm_pair_pending)
        self.assertEqual(req.full_untruncated_fill_ids[-2:].tolist(), [99, 99])
        a.runner.token_to_kv_pool_allocator.alloc.assert_not_called()

    def test_missing_n1_capture_and_unresolved_block_use_native_path(self):
        for kind in ("capture", "unresolved"):
            source = source_batch(1)
            a = adapter()
            if kind == "capture":
                a.runner.decode_cuda_graph_capture_bs = (2, 4)
            else:
                source.input_ids[0] = 99
            self.assertIsNone(a.run(source, requests(source)))
            a.runner.token_to_kv_pool_allocator.alloc.assert_not_called()

    def test_allocation_exhaustion_and_replay_failure_release_scratch(self):
        source = source_batch(1)
        a = adapter()
        a.runner.token_to_kv_pool_allocator.alloc.side_effect = None
        a.runner.token_to_kv_pool_allocator.alloc.return_value = None
        self.assertIsNone(a.run(source, requests(source)))
        a.runner.token_to_kv_pool_allocator.free.assert_not_called()
        a = adapter()
        a.graph = Mock(side_effect=RuntimeError("replay failed"))
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            a.run(source, requests(source))
        a.runner.token_to_kv_pool_allocator.free.assert_called_once()
        self.assertIsNone(a.scratch)

    def test_replay_copies_inputs_into_owned_buffers(self):
        source = source_batch(1)
        a = adapter()
        with a.scratch_batch(source) as first:
            graph = object.__new__(PairGraph)
            graph.batch = copy(first)
            graph.runner = a.runner
            for name, value in vars(first).items():
                if isinstance(value, torch.Tensor):
                    setattr(graph.batch, name, value.clone())
            graph.backend = NS(init_forward_metadata_out_graph=Mock())
            graph.graph, graph.output = NS(replay=Mock()), object()
            pointers = {
                k: v.data_ptr()
                for k, v in vars(graph.batch).items()
                if isinstance(v, torch.Tensor)
            }
        source.positions += 2
        source.dllm_prompt_mask = torch.zeros((1, 2), dtype=torch.bool)
        with a.scratch_batch(source) as second:
            self.assertIs(graph(second), graph.output)
            self.assertEqual(graph.batch.positions.tolist(), second.positions.tolist())
            self.assertFalse(graph.batch.dllm_prompt_mask.any())
        self.assertEqual(
            pointers,
            {
                k: v.data_ptr()
                for k, v in vars(graph.batch).items()
                if isinstance(v, torch.Tensor)
            },
        )
        graph.graph.replay.assert_called_once()

    def test_prompt_mask_layout_is_stable_across_requests(self):
        source = source_batch(1)
        a = adapter()
        with a.scratch_batch(source) as implicit:
            self.assertEqual(
                implicit.dllm_prompt_mask.tolist(), [[True, True, False, False]]
            )
        source.dllm_prompt_mask = torch.tensor([[True, False]])
        with a.scratch_batch(source) as explicit:
            self.assertEqual(
                explicit.dllm_prompt_mask.tolist(), [[True, False, False, False]]
            )

    def test_native_completion_retains_committed_ids(self):
        source = source_batch(1)
        a = adapter()
        output = NS(full_logits=torch.zeros(2, 100))
        result = finish_paired_step(a.algorithm, source, output, None)
        self.assertEqual(result[1:4], ([[1, 2]], [2], [None]))
        source.input_ids[0] = 99
        with self.assertRaisesRegex(RuntimeError, "complete block"):
            finish_paired_step(a.algorithm, source, output, None)

    def test_next_attention_reads_refreshed_kv_with_no_future_leak(self):
        a = adapter()
        with a.scratch_batch(source_batch(1)) as combined:
            current, future = split_pair_batch(combined, 2)
            seen = []
            stored = {}

            class Attention:
                def __init__(self, index):
                    self.index = index

                def forward_extend(self, q, k, v, layer, part, save_kv_cache=True):
                    seen.append((self.index, part.positions.tolist()))
                    if self.index == 0:
                        stored["k"], stored["v"] = k.clone(), v.clone()
                        return q + v
                    # Use the KV written by the preceding attention operation.
                    return q + stored["k"].sum(0) + stored["v"].sum(0) + v

            backend = object.__new__(FlashInferBlockPairBackend)
            backend.block_size = 2
            backend.parts = [current, future]
            backend.attn_backend_list = [Attention(0), Attention(1)]
            q = torch.arange(8).reshape(4, 2).float()
            k, v = q + 10, q + 20
            output = backend.forward_extend(q, k, v, None, combined)
            expected = torch.cat(
                (q[:2] + v[:2], q[2:] + k[:2].sum(0) + v[:2].sum(0) + v[2:])
            )
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            self.assertEqual([x[0] for x in seen], [0, 1])
            changed = v.clone()
            changed[2:] += 1000
            repeat = backend.forward_extend(q, k, changed, None, combined)
            torch.testing.assert_close(output[:2], repeat[:2], atol=0, rtol=0)


class StartupTests(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        self.adapter = adapter()
        self.runner = self.adapter.runner
        self.runner.dtype = self.runner.kv_cache_dtype = torch.bfloat16
        self.runner.model_config.quantization = None
        self.runner.model_config.hf_config = NS(architectures=["SDARForCausalLM"])
        self.runner.sliding_window_size = None
        self.runner.page_size = 1
        self.runner.canary_manager = self.runner.msprobe_debugger = None
        self.runner.capture_head_hooks = self.runner.capture_tail_hooks = []
        self.runner.spec_algorithm = NS(is_none=lambda: True)
        backend = object.__new__(FlashInferAttnBackend)
        backend.dllm_config = NS(algorithm="LowConfidence")
        backend.use_paged = backend.prefill_uses_dequant_workspace = False
        self.runner.attn_backend = backend
        self.execution = NS(
            features=NS(enable_memory_saver=False),
            offload=NS(cpu_offload_gb=0, offload_group_size=-1),
            graph=NS(debug_cuda_graph=False),
        )
        for target, value in (
            ("sglang.srt.runtime_context.get_parallel", NS(tp_size=1, pp_size=1)),
            ("sglang.srt.runtime_context.get_exec", self.execution),
        ):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("sglang.srt.debug_utils.dumper.dumper", NS(may_enable=False))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.adapter.capture = Mock()

    def test_supported_configuration_captures_before_requests(self):
        self.adapter.initialize()
        self.adapter.capture.assert_called_once()
        self.assertFalse(self.adapter.qualified)

    def test_incompatible_configurations_fail_before_capture(self):
        cases = (
            (self.runner, "dtype", torch.float16, "BF16"),
            (self.runner, "kv_cache_dtype", torch.float16, "BF16"),
            (self.runner, "attn_backend", object(), "FlashInfer"),
            (self.runner, "spec_algorithm", NS(is_none=lambda: False), "speculative"),
            (self.runner.attn_backend, "use_paged", True, "ragged"),
            (self.runner.model_config, "quantization", "fp8", "unquantized"),
            (self.runner, "decode_cuda_graph_capture_bs", (2, 4), "N1 CUDA graph"),
            (self.execution.features, "enable_memory_saver", True, "resident"),
            (self.execution.offload, "cpu_offload_gb", 1, "resident"),
            (self.runner, "canary_manager", object(), "debug"),
            (self.runner, "msprobe_debugger", object(), "debug"),
            (self.execution.graph, "debug_cuda_graph", True, "debug"),
            (self.runner, "capture_tail_hooks", [lambda: None], "hooks"),
        )
        for owner, name, value, message in cases:
            with self.subTest(setting=name), patch.object(owner, name, value):
                with self.assertRaisesRegex(ValueError, message):
                    self.adapter.initialize()
        self.adapter.capture.assert_not_called()

    def test_capture_failure_prevents_readiness(self):
        self.adapter.capture.side_effect = torch.OutOfMemoryError("capture allocation")
        with self.assertRaisesRegex(torch.OutOfMemoryError, "capture allocation"):
            self.adapter.initialize()
        self.assertFalse(self.adapter.qualified)

    def test_startup_capture_uses_only_reserved_padding_slots(self):
        self.runner.device = "cpu"
        self.runner.spec_algorithm = None
        with patch.object(module, "PairGraph") as create:
            RefreshPairServing.capture(self.adapter)
        batch = create.call_args.args[1]
        self.assertEqual(batch.req_pool_indices.tolist(), [0])
        self.assertEqual(batch.out_cache_loc.tolist(), [0, 0, 0, 0])
        self.assertEqual(batch.input_ids.tolist(), [0, 0, 99, 99])
        current, future = split_pair_batch(batch, 2)
        self.assertEqual(current.extend_prefix_lens_cpu, [0])
        self.assertEqual(future.extend_prefix_lens_cpu, [2])
        self.runner.token_to_kv_pool_allocator.alloc.assert_not_called()
        self.runner.token_to_kv_pool_allocator.free.assert_not_called()


class TestNativeRequest(CustomTestCase):
    def test_native_request_fields_and_prefetch_commit(self):
        req = native_request()
        self.assertFalse(hasattr(req, "extend_range"))
        self.assertTrue(hasattr(req, "extend_end"))
        self.assertTrue(eligible_request(req, 16, 151669))
        req.dllm_pair_pending = (144, [9] * 16)
        req.output_ids.extend([8] * 16)
        req.dllm_incomplete_ids = array("q")
        req.prefix_len = 144
        req.extend_end = 160
        req._init_fill_ids_for_dllm()
        self.assertEqual(req.full_untruncated_fill_ids[144:].tolist(), [9] * 16)
        self.assertEqual(req.dllm_pair_applied_offset, 144)
        self.assertTrue(eligible_request(req, 16, 151669))


def request(ids=None):
    return NS(
        sampling_params=NS(
            ignore_eos=False,
            stop=[],
            stop_token_ids=None,
            stop_regex=None,
            max_new_tokens=128,
        ),
        eos_token_ids={151643, 151645},
        tokenizer=NS(eos_token_id=151643, additional_stop_token_ids={151646}),
        return_logprob=False,
        dllm_algo_state=None,
        dllm_incomplete_ids=array("q", ids or [7] * 16),
        extend_end=48,
        origin_input_ids=array("q", [3] * 32),
        output_ids=array("q"),
        dllm_pair_context_limit=4096,
        dllm_pair_applied_offset=None,
        finished_reason=None,
        finished_len=None,
        full_untruncated_fill_ids=array("q", [3] * 32 + (ids or [7] * 16)),
    )


class TestNaturalStop(CustomTestCase):
    def test_request_specific_graph_state_uses_native_execution(self):
        for name, value in (
            ("return_hidden_states", True),
            ("return_hidden_states", "last"),
            ("lora_id", "adapter-a"),
            ("input_embeds", [[0.0]]),
            ("return_logprob", True),
        ):
            with self.subTest(name=name, value=value):
                req = request()
                setattr(req, name, value)
                self.assertFalse(eligible_request(req, 16, 99))

    def test_nonterminal_block_can_pair(self):
        self.assertTrue(eligible_request(request(), 16, 99))

    def test_all_stock_eos_sources_disable_lookahead(self):
        for token in (151643, 151645, 151646):
            for position in (0, 7, 15):
                with self.subTest(token=token, position=position):
                    ids = [7] * 16
                    ids[position] = token
                    req = request(ids)
                    self.assertFalse(eligible_request(req, 16, 99))
                    req.output_ids.extend(ids)
                    self.assertTrue(Req._check_token_based_finish(req, ids))
                    self.assertEqual(req.finished_len, position + 1)

    def test_prompt_eos_does_not_finish_partial_prompt_block(self):
        req = request([151643] * 9 + [7] * 7)
        req.origin_input_ids.extend([3] * 9)
        self.assertTrue(eligible_request(req, 16, 99))
        self.assertFalse(Req._check_token_based_finish(req, [7] * 7))

    def test_prefetched_terminal_block_is_not_paired_again(self):
        req = request()
        req.dllm_incomplete_ids = array("q")
        req.dllm_pair_applied_offset = 32
        req.full_untruncated_fill_ids[-1] = 151645
        self.assertFalse(eligible_request(req, 16, 99))

    def test_unknown_stop_metadata_falls_back(self):
        req = request()
        req.eos_token_ids = None
        req.tokenizer = None
        self.assertFalse(eligible_request(req, 16, 99))
        req.tokenizer = NS(eos_token_id=None, additional_stop_token_ids=None)
        self.assertFalse(eligible_request(req, 16, 99))

    def test_explicit_stop_modes_keep_the_ordinary_path(self):
        for name, value in [
            ("stop", ["end"]),
            ("stop_regex", ["end.*"]),
            ("stop_token_ids", {17}),
        ]:
            req = request()
            setattr(req.sampling_params, name, value)
            self.assertFalse(eligible_request(req, 16, 99))

    def test_forced_output_policy_is_unchanged(self):
        req = request([151643] * 16)
        req.sampling_params.ignore_eos = True
        self.assertTrue(eligible_request(req, 16, 99))


class TestPairState(CustomTestCase):
    def test_qualification_uses_production_graph_with_request_row_order(self):
        block, n = 2, 2
        combined = NS(
            dllm_prompt_mask=None,
            batch_size=n,
            forward_mode=NS(is_dllm_extend=lambda: True),
            extend_seq_lens_cpu=[4, 4],
            extend_prefix_lens_cpu=[4, 8],
            spec_info=None,
            encoder_lens=None,
            input_ids=torch.tensor([1, 2, 99, 99, 3, 4, 99, 99]),
            positions=torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
            out_cache_loc=torch.arange(8),
            seq_lens=torch.tensor([8, 12]),
            seq_lens_cpu=torch.tensor([8, 12]),
            seq_lens_sum=20,
            extend_prefix_lens=torch.tensor([4, 8]),
            extend_seq_lens=torch.tensor([4, 4]),
            dllm_block_pair_size=block,
        )
        logits = torch.arange(8 * 7, dtype=torch.float32).reshape(8, 7)
        refs = [
            logits.reshape(n, 2, block, 7)[:, i].reshape(n * block, 7).contiguous()
            for i in (0, 1)
        ]
        forward = Mock(
            side_effect=[
                NS(can_run_graph=True, logits_output=NS(full_logits=x)) for x in refs
            ]
        )
        adapter = module.RefreshPairServing(
            NS(forward=forward), NS(fdfo=True, block_size=block)
        )
        with patch.object(LowConfidence, "step", return_value=None):
            adapter.qualify(combined, logits)
        calls = forward.call_args_list
        self.assertEqual(calls[0].args[0].input_ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(calls[1].args[0].input_ids.tolist(), [99, 99, 99, 99])
        self.assertEqual(calls[1].args[0].extend_prefix_lens_cpu, [6, 10])
        self.assertTrue(all(c.args[0].dllm_block_pair_size is None for c in calls))
        forward.side_effect = [NS(can_run_graph=False)]
        with self.assertRaisesRegex(RuntimeError, "stock CUDA graph"):
            adapter.qualify(combined, logits)
        forward.side_effect = [
            NS(can_run_graph=True, logits_output=NS(full_logits=refs[0] + 100))
        ]
        with self.assertRaisesRegex(RuntimeError, "qualification failed"):
            adapter.qualify(combined, logits)

    def request(self):
        return NS(
            sampling_params=NS(
                ignore_eos=True,
                stop=[],
                stop_regex=None,
                stop_token_ids=None,
                max_new_tokens=64,
            ),
            return_logprob=False,
            dllm_algo_state=None,
            dllm_incomplete_ids=array("q", [7] * 16),
            extend_end=48,
            origin_input_ids=array("q", [3] * 32),
            output_ids=array("q"),
            dllm_pair_context_limit=128,
            dllm_config=NS(block_size=16),
            dllm_pair_pending=None,
            full_untruncated_fill_ids=array("q", [3] * 32 + [7] * 16),
        )

    def test_pair_batch_matches_stock_graph_index_contract(self):
        block, n = 2, 2
        runner = NS(
            kv_index_translator=StaticTranslator(),
            token_to_kv_pool_allocator=NS(
                alloc=lambda count: torch.arange(100, 100 + count, dtype=torch.int64)
            ),
            prepare_dummy_forward_batch=lambda batch: batch,
        )
        adapter = module.RefreshPairServing(
            runner, NS(fdfo=True, block_size=block, mask_id=99)
        )
        source = source_batch(n, block)
        source.out_cache_loc = torch.tensor([20, 21, 30, 31], dtype=torch.int64)
        source.positions = source.positions.to(torch.int32)
        paired = adapter.paired_batch(source)
        self.assertEqual(paired.out_cache_loc.dtype, torch.int64)
        self.assertEqual(paired.positions.dtype, torch.int64)
        self.assertEqual(source.positions.dtype, torch.int32)
        self.assertEqual(
            paired.out_cache_loc.reshape(n, 4)[:, :2].tolist(), [[20, 21], [30, 31]]
        )
        self.assertEqual(
            paired.out_cache_loc.reshape(n, 4)[:, 2:].tolist(), [[100, 101], [102, 103]]
        )
        self.assertEqual(paired.positions.tolist(), [4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(paired.input_ids.tolist(), [1, 2, 99, 99, 3, 4, 99, 99])
        self.assertEqual(source.out_cache_loc.tolist(), [20, 21, 30, 31])
        self.assertEqual(paired.seq_lens.tolist(), [8, 12])

    def test_scratch_is_released_on_success_and_failure(self):
        allocator = NS(free=Mock())
        adapter = module.RefreshPairServing(
            NS(token_to_kv_pool_allocator=allocator), NS(fdfo=True, block_size=16)
        )
        slots = torch.arange(4, dtype=torch.int64).reshape(2, 2)

        def create(source):
            adapter.scratch = slots
            return "batch"

        adapter.paired_batch = create
        with adapter.scratch_batch(None) as batch:
            self.assertEqual(batch, "batch")
            allocator.free.assert_not_called()
        self.assertIsNone(adapter.scratch)
        self.assertTrue(
            torch.equal(allocator.free.call_args.args[0], slots.reshape(-1))
        )
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            with adapter.scratch_batch(None):
                raise RuntimeError("model failed")
        self.assertIsNone(adapter.scratch)
        self.assertEqual(allocator.free.call_count, 2)

    def test_scratch_exhaustion_uses_no_unowned_slots(self):
        allocator = NS(alloc=Mock(return_value=None), free=Mock())
        adapter = module.RefreshPairServing(
            NS(token_to_kv_pool_allocator=allocator), NS(fdfo=True, block_size=16)
        )
        source = NS(batch_size=2)
        with adapter.scratch_batch(source) as batch:
            self.assertIsNone(batch)
        allocator.alloc.assert_called_once_with(32)
        allocator.free.assert_not_called()
        self.assertIsNone(adapter.scratch)

    def test_pending_waits_for_commit_then_applies_idempotently(self):
        req = self.request()
        req.dllm_pair_pending = (48, [9] * 16)
        self.assertFalse(module.apply_prefetch(req))
        req.output_ids = array("q", [7] * 16)
        req.full_untruncated_fill_ids.extend([99] * 16)
        self.assertTrue(module.apply_prefetch(req))
        before = req.full_untruncated_fill_ids[:]
        self.assertTrue(module.apply_prefetch(req))
        self.assertEqual(req.full_untruncated_fill_ids, before)
        self.assertEqual(list(before[48:]), [9] * 16)

    def test_stale_prediction_fails_instead_of_silent_reuse(self):
        req = self.request()
        req.dllm_pair_pending = (32, [9] * 16)
        req.output_ids = array("q", [7] * 16)
        with self.assertRaises(RuntimeError):
            module.apply_prefetch(req)

    def test_terminal_block_and_stop_requests_do_not_pair(self):
        req = self.request()
        self.assertTrue(module.eligible_request(req, 16, 99))
        req.sampling_params.max_new_tokens = 16
        self.assertFalse(module.eligible_request(req, 16, 99))
        req.sampling_params.max_new_tokens = 64
        req.sampling_params.ignore_eos = False
        self.assertFalse(module.eligible_request(req, 16, 99))
        req.sampling_params.ignore_eos = True
        req.sampling_params.stop = ["end"]
        self.assertFalse(module.eligible_request(req, 16, 99))

    def test_unresolved_or_context_limited_requests_do_not_pair(self):
        req = self.request()
        req.dllm_incomplete_ids[3] = 99
        self.assertFalse(module.eligible_request(req, 16, 99))
        req.dllm_incomplete_ids[3] = 7
        req.dllm_pair_context_limit = 63
        self.assertFalse(module.eligible_request(req, 16, 99))

    def test_fully_predicted_next_block_can_pair_again(self):
        req = self.request()
        req.dllm_incomplete_ids = array("q")
        req.dllm_pair_applied_offset = 32
        self.assertTrue(module.eligible_request(req, 16, 99))
        req.full_untruncated_fill_ids[40] = 99
        self.assertFalse(module.eligible_request(req, 16, 99))


class PrefetchWidthTests(CustomTestCase):
    def case(self, width=2, selected=0, future_width=1):
        rows = torch.arange(10, 10 + width * 2).view(width, 2)
        source = NS(batch_size=width, input_ids=rows.reshape(-1).clone())
        reqs = []
        for index, row in enumerate(rows):
            applied = index == selected
            reqs.append(
                NS(
                    extend_end=6,
                    dllm_pair_applied_offset=4 if applied else None,
                    dllm_pair_future_width=future_width if applied else None,
                    dllm_pair_pending=(4, row.tolist()) if applied else None,
                    full_untruncated_fill_ids=array("q", [1, 2, 3, 4] + row.tolist()),
                    output_ids=array("q", [3, 4]),
                    dllm_incomplete_ids=array("q"),
                )
            )
        return source, reqs

    def test_mixed_future_restores_mask_and_preserves_other_requests(self):
        for width in (2, 4):
            for selected in range(width):
                with self.subTest(width=width, selected=selected):
                    source, reqs = self.case(width, selected)
                    before = source.input_ids.clone()
                    self.assertEqual(
                        discard_incompatible_prefetch(source, reqs, 2, 99, [1, 2, 4]), 1
                    )
                    expected = before.view(width, 2)
                    expected[selected] = 99
                    self.assertTrue(torch.equal(source.input_ids, expected.reshape(-1)))
                    self.assertEqual(
                        reqs[selected].full_untruncated_fill_ids.tolist(),
                        [1, 2, 3, 4, 99, 99],
                    )
                    self.assertEqual(reqs[selected].output_ids.tolist(), [3, 4])
                    self.assertIsNone(reqs[selected].dllm_pair_pending)
                    self.assertIsNone(reqs[selected].dllm_pair_applied_offset)
                    self.assertIsNone(reqs[selected].dllm_pair_future_width)
                    self.assertEqual(
                        discard_incompatible_prefetch(
                            source, reqs, 2, 99, [1, 2, 4, 8]
                        ),
                        0,
                    )

    def test_matching_unpadded_width_keeps_predictions(self):
        for width in (1, 2, 4, 8):
            source, reqs = self.case(width, future_width=width)
            before = source.input_ids.clone()
            self.assertEqual(
                discard_incompatible_prefetch(source, reqs, 2, 99, [1, 2, 4, 8]), 0
            )
            self.assertTrue(torch.equal(before, source.input_ids))
            self.assertIsNotNone(reqs[0].dllm_pair_pending)

    def test_missing_or_padded_capture_uses_ordinary_step(self):
        for captures in ((), (2, 4)):
            source, reqs = self.case(1, future_width=1)
            self.assertEqual(
                discard_incompatible_prefetch(source, reqs, 2, 99, captures), 1
            )
            self.assertEqual(source.input_ids.tolist(), [99, 99])

    def test_missing_certificate_uses_ordinary_step(self):
        source, reqs = self.case(1, future_width=1)
        del reqs[0].dllm_pair_future_width
        self.assertEqual(discard_incompatible_prefetch(source, reqs, 2, 99, [1]), 1)

    def test_cannot_reset_an_already_started_denoising_step(self):
        source, reqs = self.case()
        before = source.input_ids.clone()
        reqs[0].dllm_incomplete_ids = array("q", [10, 99])
        with self.assertRaisesRegex(RuntimeError, "ordinary denoising started"):
            discard_incompatible_prefetch(source, reqs, 2, 99, [1, 2])
        self.assertTrue(torch.equal(before, source.input_ids))

    def test_unapplied_future_is_not_discarded_early(self):
        source, reqs = self.case()
        reqs[0].dllm_pair_applied_offset = None
        before = source.input_ids.clone()
        self.assertEqual(discard_incompatible_prefetch(source, reqs, 2, 99, [1, 2]), 0)
        self.assertTrue(torch.equal(before, source.input_ids))
        self.assertIsNotNone(reqs[0].dllm_pair_pending)


class RealTokenExtentTests(CustomTestCase):
    def check_layout(self):
        n, block = 2, 16
        source = source_batch(n, block)
        # Use aligned prefixes and the same row-count fields as production.
        source.forward_mode = ForwardMode.DLLM_EXTEND
        source.extend_prefix_lens_cpu = [64, 128]
        source.extend_prefix_lens = torch.tensor([64, 128])
        source.seq_lens_cpu = source.extend_prefix_lens + block
        source.seq_lens = source.seq_lens_cpu.clone()
        source.seq_lens_sum = int(source.seq_lens.sum())
        source.global_num_token_non_padded_cpu = n * block
        source.global_num_token_non_padded = torch.tensor(n * block, dtype=torch.int32)
        source.num_token_non_padded = source.global_num_token_non_padded.clone()
        source.global_num_tokens_cpu = [n * block]
        source.global_num_tokens_gpu = torch.tensor([n * block])
        runner = NS(
            kv_index_translator=StaticTranslator(),
            token_to_kv_pool_allocator=NS(
                alloc=lambda count: torch.arange(1000, 1000 + count)
            ),
            prepare_dummy_forward_batch=lambda batch: batch,
        )
        adapter = RefreshPairServing(runner, algorithm(block))
        combined = adapter.paired_batch(source)
        views = [combined, *split_pair_batch(combined, block)]
        for view in views:
            count = view.input_ids.numel()
            self.assertIsNone(padded_extend_real_tokens(torch.empty(count, 8), view))
            self.assertEqual(view.global_num_token_non_padded_cpu, count)
            self.assertEqual(view.global_num_token_non_padded.item(), count)
            self.assertEqual(view.num_token_non_padded.item(), count)
            self.assertEqual(view.global_num_tokens_cpu, [count])
            self.assertEqual(view.global_num_tokens_gpu.item(), count)
            self.assertEqual(view.out_cache_loc.numel(), count)
            self.assertNotEqual(
                view.global_num_token_non_padded.data_ptr(),
                source.global_num_token_non_padded.data_ptr(),
            )
        self.assertEqual(source.global_num_token_non_padded_cpu, n * block)
        self.assertEqual(source.global_num_token_non_padded.item(), n * block)
        self.assertEqual(source.global_num_tokens_cpu, [n * block])

    def test_uniform_pair(self):
        self.check_layout()


class LifecycleTests(CustomTestCase):
    def request(self, **params):
        sampling = SamplingParams(temperature=0, max_new_tokens=128, **params)
        tokenizer = NS(encode=lambda text, add_special_tokens=False: [7])
        sampling.normalize(tokenizer)
        return native_request(sampling)

    def test_retraction_discards_pending_predictions(self):
        for already_applied in (False, True):
            with self.subTest(already_applied=already_applied):
                req = self.request()
                req.output_ids.extend([8] * 16)
                req.dllm_incomplete_ids = array("q")
                req.dllm_initialized = True
                req.dllm_pair_pending = (144, [9] * 16)
                req.dllm_pair_applied_offset = 144 if already_applied else None
                req.dllm_pair_future_width = 1
                req.reset_for_retract()
                self.assertIsNone(req.dllm_pair_pending)
                self.assertIsNone(req.dllm_pair_applied_offset)
                self.assertIsNone(req.dllm_pair_future_width)
                req._init_fill_ids_for_dllm()
                req.determine_dllm_phase()
                self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_PREFILL)
                self.assertEqual(req.output_ids.tolist(), [8] * 16)
                self.assertEqual(
                    req.full_untruncated_fill_ids[144:].tolist(), [151669] * 16
                )

    def test_normalized_stop_strings_use_fallback(self):
        for params in ({"stop": ["END"]}, {"stop_regex": ["END"]}):
            with self.subTest(params=params):
                req = self.request(**params)
                self.assertIsNone(req.sampling_params.stop)
                self.assertIsNone(req.sampling_params.stop_regex)
                self.assertFalse(eligible_request(req, 16, 151669))


if __name__ == "__main__":
    unittest.main()
