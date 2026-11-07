import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.arg_groups.fields.exec_ import ExecDllm
from sglang.srt.arg_groups.fields.mm import Mm
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import _dllm_attention_backend
from sglang.srt.dllm.algorithm import get_algorithm_cls
from sglang.srt.dllm.algorithm.hierarchy_block import HierarchyBlock
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.fast_dvlm import _flatten_weight_name
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MASK = 99
VOCAB = 128
HIGH = 20.0
LOW = 0.5


def _algorithm(block_size=8, sub_block_size=4, threshold=0.9):
    config = DllmConfig(
        algorithm="HierarchyBlock",
        algorithm_config={"sub_block_size": sub_block_size, "threshold": threshold},
        block_size=block_size,
        mask_id=MASK,
        max_running_requests=4,
        carries_block_state=True,
    )
    return HierarchyBlock(config)


def _batch(rows, prompt_mask=None):
    input_ids = torch.tensor(rows, dtype=torch.long)
    return SimpleNamespace(
        batch_size=input_ids.shape[0],
        input_ids=input_ids.view(-1).clone(),
        dllm_prompt_mask=prompt_mask,
    )


class _Model:
    def __init__(self):
        self.causal = True

    def set_block_attention_causal(self, causal):
        self.causal = causal


class _OracleRunner:
    """Logits at block position p peak at targets[:, p + 1] (token shift).

    Under bidirectional attention the peaks move to bidirectional_targets
    when given, so tests can tell which attention produced a token.
    """

    def __init__(
        self, targets, peaks, bidirectional_targets=None, fail_on_forward=None
    ):
        self.targets = torch.tensor(targets)
        self.bidirectional_targets = torch.tensor(
            targets if bidirectional_targets is None else bidirectional_targets
        )
        self.peaks = torch.as_tensor(peaks, dtype=torch.float32)
        self.model = _Model()
        self.forwards = []
        self.fail_on_forward = fail_on_forward

    def forward(self, forward_batch, pp_proxy_tensors=None):
        if self.fail_on_forward == len(self.forwards):
            raise RuntimeError("forward failed")
        self.forwards.append((self.model.causal, forward_batch.input_ids.clone()))
        targets = self.targets if self.model.causal else self.bidirectional_targets
        batch_size, width = targets.shape
        block_size = width - 1
        logits = torch.zeros(batch_size, block_size, VOCAB)
        peak = self.peaks.expand(batch_size, block_size)
        logits.scatter_(-1, targets[:, 1:].unsqueeze(-1), peak.unsqueeze(-1))
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits.view(-1, VOCAB)),
            can_run_graph=False,
        )


class TestHierarchyBlockRun(unittest.TestCase):
    def test_registered(self):
        self.assertIs(get_algorithm_cls("HierarchyBlock"), HierarchyBlock)

    def test_sub_blocks_resolve_in_order_and_kv_is_written_causally(self):
        targets = [[10, 11, 12, 13, 14, 15, 16, 17, 18]]
        runner = _OracleRunner(targets, HIGH)
        batch = _batch([[MASK] * 8])

        _, next_ids, _, states, _ = _algorithm().run(runner, batch, [10])

        causal_flags = [causal for causal, _ in runner.forwards]
        self.assertEqual(causal_flags, [False, False, True])
        # Sub-block 1 stays masked until sub-block 0 is resolved.
        self.assertEqual(runner.forwards[1][1][4:].tolist(), [MASK] * 4)
        self.assertEqual(next_ids[0].tolist(), targets[0][:8])
        self.assertEqual(states, [18])
        self.assertTrue(runner.model.causal)

    def test_low_confidence_still_unmasks_one_token_per_step(self):
        targets = [[10, 11, 12, 13, 14, 15, 16, 17, 18]]
        runner = _OracleRunner(targets, LOW)
        batch = _batch([[MASK] * 8])

        _, next_ids, _, _, _ = _algorithm().run(runner, batch, [10])

        # Seven masks after seeding: seven denoise forwards, then one causal.
        self.assertEqual(len(runner.forwards), 8)
        self.assertEqual([c for c, _ in runner.forwards].count(True), 1)
        self.assertEqual(next_ids[0].tolist(), targets[0][:8])

    def test_prompt_only_block_still_returns_carried_state(self):
        targets = [[1, 2, 3, 4, 5, 6, 7, 8, 42]]
        runner = _OracleRunner(targets, HIGH)
        batch = _batch([targets[0][:8]])

        _, next_ids, _, states, _ = _algorithm().run(runner, batch, [None])

        self.assertEqual([c for c, _ in runner.forwards], [True])
        self.assertEqual(next_ids, [])
        self.assertEqual(states, [42])

    def test_requests_in_a_batch_keep_their_own_state(self):
        targets = [
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [1, 2, 3, 20, 21, 22, 23, 24, 25],
        ]
        runner = _OracleRunner(targets, [[HIGH] * 8, [LOW] * 8])
        batch = _batch([[MASK] * 8, [1, 2, 3] + [MASK] * 5])

        _, next_ids, _, states, _ = _algorithm().run(runner, batch, [10, 77])

        self.assertEqual(next_ids[0].tolist(), targets[0][:8])
        # Position 0 holds a prompt token, so the carried token is not used.
        self.assertEqual(next_ids[1].tolist(), targets[1][3:8])
        self.assertEqual(states, [18, 25])
        # One causal forward seeds after the prompt tail, one writes the KV.
        self.assertEqual([c for c, _ in runner.forwards].count(True), 2)
        self.assertTrue(runner.forwards[0][0])
        self.assertTrue(runner.forwards[-1][0])

    def test_first_token_after_prompt_tail_is_predicted_causally(self):
        causal = [[1, 2, 3, 20, 21, 22, 23, 24, 25]]
        bidirectional = [[1, 2, 3, 60, 21, 22, 23, 24, 25]]
        runner = _OracleRunner(causal, HIGH, bidirectional_targets=bidirectional)
        batch = _batch([[1, 2, 3] + [MASK] * 5])

        _, next_ids, _, _, _ = _algorithm().run(runner, batch, [None])

        self.assertEqual(next_ids[0].tolist(), causal[0][3:8])

    def test_literal_mask_in_prompt_is_not_denoised(self):
        targets = [[5, MASK, 6, 30, 31, 32, 33, 34, 35]]
        runner = _OracleRunner(targets, HIGH)
        prompt_mask = torch.tensor([[True, True, True] + [False] * 5])
        batch = _batch([[5, MASK, 6] + [MASK] * 5], prompt_mask=prompt_mask)

        _, next_ids, _, _, _ = _algorithm().run(runner, batch, [None])

        self.assertEqual(batch.input_ids[1].item(), MASK)
        self.assertEqual(next_ids[0].tolist(), targets[0][3:8])

    def test_predicted_mask_token_is_never_emitted(self):
        """A model whose top prediction is the mask token must not stall a step
        or leave a mask token in the output."""
        targets = [[10] + [MASK] * 8]
        runner = _OracleRunner(targets, HIGH)
        batch = _batch([[MASK] * 8])

        _, next_ids, _, _, _ = _algorithm().run(runner, batch, [10])

        self.assertNotIn(MASK, next_ids[0].tolist())
        self.assertLessEqual(len(runner.forwards), 8)

    def test_attention_is_causal_again_after_a_failed_denoise_forward(self):
        runner = _OracleRunner([[10] * 9], HIGH, fail_on_forward=0)
        with self.assertRaises(RuntimeError):
            _algorithm().run(runner, _batch([[MASK] * 8]), [10])
        self.assertTrue(runner.model.causal)

    def test_sub_block_size_must_divide_block_size(self):
        with self.assertRaises(ValueError):
            _algorithm(block_size=8, sub_block_size=3)

    def test_empty_batch_does_not_read_logits(self):
        runner = SimpleNamespace(
            model=_Model(),
            forward=lambda fb, pp_proxy_tensors=None: SimpleNamespace(
                logits_output=None, can_run_graph=False
            ),
        )
        batch = SimpleNamespace(
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.long),
            dllm_prompt_mask=None,
        )

        _, next_ids, _, states, _ = _algorithm().run(runner, batch, [])

        self.assertEqual((next_ids, states), ([], []))


class TestFastDVLMWeightNames(unittest.TestCase):
    def test_nested_checkpoint_names_map_to_qwen2_5_vl_params(self):
        """The Fast-dVLM release nests weights under model.language_model. and
        model.visual.; Qwen2.5-VL's loader keys on the flat names."""
        cases = {
            "model.language_model.layers.0.mlp.gate_proj.weight": "model.layers.0.mlp.gate_proj.weight",
            "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.visual.blocks.0.attn.qkv.weight": "visual.blocks.0.attn.qkv.weight",
            "lm_head.weight": "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight": "model.layers.0.self_attn.q_proj.weight",
        }
        for name, expected in cases.items():
            self.assertEqual(_flatten_weight_name(name), expected)


class TestHierarchyBlockLaunch(unittest.TestCase):
    def _server_args(self, **overrides):
        args = dict(
            device="cuda",
            # Upstream defaults, so a changed default cannot hide behind the test.
            dllm_fdfo=ExecDllm().dllm_fdfo,
            enable_multimodal=Mm().enable_multimodal,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.FULL),
                prefill=PhaseConfig(backend=Backend.FULL),
            ),
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_cuda_graphs_are_disabled_in_both_phases(self):
        """Graphs bake the attention mask in at capture, but denoising and the
        final KV write need different masks in the same block."""
        server_args = self._server_args()

        HierarchyBlock.configure_server_args(server_args)

        resolved = resolving_view(server_args).cuda_graph_config
        self.assertEqual(resolved.decode.backend, Backend.DISABLED)
        self.assertEqual(resolved.prefill.backend, Backend.DISABLED)

    def test_multimodal_is_off_unless_explicitly_requested(self):
        """The checkpoint carries a vision sub-config, so it would otherwise be
        served as multimodal, which needs an image processor and image support
        on the dLLM path that do not exist yet."""
        server_args = self._server_args()
        HierarchyBlock.configure_server_args(server_args)
        self.assertFalse(resolving_view(server_args).enable_multimodal)
        with self.assertRaises(ValueError):
            HierarchyBlock.configure_server_args(
                self._server_args(enable_multimodal=True)
            )

    def test_default_launch_runs_blocks_synchronously(self):
        server_args = self._server_args()
        HierarchyBlock.configure_server_args(server_args)
        self.assertFalse(resolving_view(server_args).dllm_fdfo)

    def test_rejects_non_cuda_devices(self):
        with self.assertRaises(ValueError):
            HierarchyBlock.configure_server_args(self._server_args(device="cpu"))

    def test_only_backends_that_read_attn_type_per_forward_are_kept(self):
        def view(backend):
            return SimpleNamespace(
                dllm_algorithm="HierarchyBlock",
                attention_backend=backend,
                prefill_attention_backend=None,
                decode_attention_backend=None,
                cuda_graph_config=CudaGraphConfig(),
            )

        for backend in ("flashinfer", "triton", "fa3"):
            self.assertEqual(_dllm_attention_backend(view(backend)), {})
        self.assertEqual(
            _dllm_attention_backend(view("trtllm_mha")),
            {
                "attention_backend": "flashinfer",
                "prefill_attention_backend": "flashinfer",
                "decode_attention_backend": "flashinfer",
            },
        )


def _req(prompt, algorithm="HierarchyBlock", carries=True):
    sampling_params = SamplingParams(max_new_tokens=16, temperature=0)
    sampling_params.normalize(None)
    return Req(
        rid="hierarchy-block",
        origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=sampling_params,
        vocab_size=VOCAB,
        dllm_config=DllmConfig(algorithm, {}, 4, MASK, 4, carries_block_state=carries),
    )


def _cache_with(tokens):
    cache = RadixCache.create_simulated(page_size=4)
    cache.insert(
        InsertParams(key=RadixKey(array("q", tokens)), value=torch.arange(len(tokens)))
    )
    return cache


class TestHierarchyBlockPrefixReuse(unittest.TestCase):
    def test_new_request_recomputes_its_last_cached_block(self):
        """The carried token comes from the causal forward over the last
        committed block, so a new request cannot take that block from cache."""
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req = _req(prompt)

        req.init_next_round_input(_cache_with(prompt))

        self.assertEqual(req.prefix_len, 4)
        self.assertTrue(req.is_dllm_prefill())

    def test_request_with_carried_state_reuses_the_whole_prefix(self):
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req = _req(prompt)
        req.dllm_algo_state = 42  # e.g. kept across a retraction

        req.init_next_round_input(_cache_with(prompt))

        self.assertEqual(req.prefix_len, 8)
        self.assertFalse(req.is_dllm_prefill())

    def test_algorithms_without_carried_state_keep_upstream_reuse(self):
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req = _req(prompt, algorithm="LowConfidence", carries=False)

        req.init_next_round_input(_cache_with(prompt))

        self.assertEqual(req.prefix_len, 8)


class TestHierarchyBlockResultProcessing(unittest.TestCase):
    def _scheduler(self, carries):
        return SimpleNamespace(
            dllm_config=SimpleNamespace(
                block_size=4,
                requires_separate_context_encoding=False,
                first_done_first_out_mode=False,
                carries_block_state=carries,
            ),
            token_to_kv_pool_allocator=Mock(),
            metrics_reporter=SimpleNamespace(
                num_generated_tokens=0, report_prefill_stats=Mock()
            ),
            output_streamer=Mock(),
            _store_dllm_block_state=SchedulerDllmMixin._store_dllm_block_state,
            _finish_dllm_request_if_needed=Mock(),
        )

    def _batch(self, reqs):
        return SimpleNamespace(
            reqs=reqs,
            batch_size=lambda: len(reqs),
            forward_mode=ForwardMode.DLLM_EXTEND,
            prefill_stats=None,
            dp_cooperation_info=None,
            return_logprob=False,
        )

    def _result(self, next_token_ids, states):
        return SimpleNamespace(
            copy_done=None,
            next_token_ids=next_token_ids,
            dllm_algo_state=states,
            accept_length_per_req_cpu=None,
            can_run_cuda_graph=False,
        )

    def _staged(self, prompt):
        req = _req(prompt)
        req.init_next_round_input()
        req.extend_end = len(req.full_untruncated_fill_ids)
        return req

    def test_prompt_only_block_stores_state_without_emitting(self):
        reqs = [self._staged([1, 2, 3, 4]), self._staged([5, 6, 7, 8])]

        SchedulerDllmMixin.process_batch_result_dllm(
            self._scheduler(True), self._batch(reqs), self._result([], [40, 41])
        )

        self.assertEqual([r.dllm_algo_state for r in reqs], [40, 41])
        self.assertEqual([list(r.output_ids) for r in reqs], [[], []])

    def test_generated_block_stores_state_and_emits(self):
        req = self._staged([1, 2])
        tokens = torch.tensor([20, 21])

        SchedulerDllmMixin.process_batch_result_dllm(
            self._scheduler(True), self._batch([req]), self._result([tokens], [22])
        )

        self.assertEqual(req.dllm_algo_state, 22)
        self.assertEqual(list(req.output_ids), [20, 21])

    def test_other_algorithms_leave_request_state_alone(self):
        req = self._staged([1, 2])

        SchedulerDllmMixin.process_batch_result_dllm(
            self._scheduler(False),
            self._batch([req]),
            self._result([torch.tensor([20, 21])], None),
        )

        self.assertIsNone(req.dllm_algo_state)

    def test_worker_passes_carried_state_into_sync_runs(self):
        def worker(carries):
            algorithm = Mock(fdfo=False, carries_block_state=carries)
            algorithm.run.return_value = (None, [], None, None, False)
            return SimpleNamespace(dllm_algorithm=algorithm, model_runner=Mock())

        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(dllm_algo_state=5),
                SimpleNamespace(dllm_algo_state=None),
            ]
        )
        for carries, expected in ((True, [5, None]), (False, None)):
            tp_worker = worker(carries)
            TpModelWorker._forward_batch_generation_dllm(tp_worker, Mock(), batch)
            self.assertEqual(tp_worker.dllm_algorithm.run.call_args.args[2], expected)


if __name__ == "__main__":
    unittest.main()
