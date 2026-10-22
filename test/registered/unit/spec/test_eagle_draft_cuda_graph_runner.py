"""Regression test for EAGLE draft CUDA graph seq_lens_sum padding.

This is a white-box unit test: it constructs a bare ``EAGLEDraftCudaGraphRunner``
via ``__new__`` and stubs the fields/methods that ``execute()`` touches, so the
padding bookkeeping can be exercised on CPU without a captured CUDA graph. It is
therefore coupled to ``execute()``'s internals and may need updating when that
method changes -- that coupling is intentional and the price of testing the
padding path in isolation.

The contract under test is the invariant ``seq_lens_sum == seq_lens.sum()``:
while the runner pads ``raw_bs`` up to a captured ``bs`` with fake rows, the
draft attention backends read ``seq_lens_sum`` to size/slice draft kv_indices,
so it must stay consistent with the padded ``seq_lens`` they are handed -- and
the raw value must be restored once replay finishes.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_worker_v2
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.watermarking.synthid import (
    SynthIDConfig,
    SynthIDSparseOutput,
    SynthIDState,
    SynthIDWatermarkProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CAPTURE_BS = 4
SEQ_LEN_FILL_VALUE = 1
NUM_STEPS = 3


class _RecordingDraftBackend:
    """Records ``(batch_size, seq_lens_sum, seq_lens)`` at every point the runner
    asks for replay metadata, so a test can check they stay consistent.

    The same recorder is also invoked from the stubbed graph replay, so both the
    metadata-build and the graph-replay phases are observed.
    """

    def __init__(self):
        self.observations = []

    def init_forward_metadata_out_graph(self, forward_batch):
        self.observe("metadata_build", forward_batch)

    def observe(self, phase, forward_batch):
        seq_lens = forward_batch.seq_lens
        self.observations.append(
            SimpleNamespace(
                phase=phase,
                batch_size=forward_batch.batch_size,
                seq_lens_sum=forward_batch.seq_lens_sum,
                seq_lens=None if seq_lens is None else seq_lens.clone(),
            )
        )


class TestEagleDraftCudaGraphRunner(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.enterContext(envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False))

    def _build_fused_proposal_worker(self):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.device = "cpu"
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.hot_token_id = None
        worker.index_share_for_mtp_iteration = False
        worker.seed_dsa_topk_from_draft_extend = False
        worker._topk1_parents_prealloc = torch.tensor([[-1, 0]])
        worker._topk1_score_indices_prealloc = torch.tensor([[0, 1]])
        worker.plan_stream = None
        worker.plan_stream_ctx = nullcontext()
        worker.cuda_graph_runner_for_draft_extend = None
        worker.draft_attn_backend = SimpleNamespace(attn_backends=[object()])
        worker.draft_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DummyMTP"]),
                model_is_mrope=False,
            ),
            canary_manager=None,
            forward=lambda batch: SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.arange(8, dtype=torch.float32)
                    .expand(batch.input_ids.shape[0], 8)
                    .clone(),
                    hidden_states=torch.zeros(batch.input_ids.shape[0], 2),
                )
            ),
        )
        probs = torch.zeros(1, 8)
        probs[:, 2] = 1
        worker._apply_synthid_to_draft_logits = Mock(
            return_value=(
                None,
                (probs, None, torch.tensor([1.0]), torch.tensor([2])),
            )
        )
        config = SynthIDConfig(keys=(1, 2), top_k=3, context_history_size=4)
        processor = SynthIDWatermarkProcessor(config, "cpu", num_slots=1)
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=SimpleNamespace(synthid_watermark=processor)
            )
        )
        info = SamplingBatchInfo(
            temperatures=torch.ones(1, 1),
            top_ps=torch.ones(1),
            top_ks=torch.tensor([TOP_K_ALL]),
            min_ps=torch.zeros(1),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            vocab_size=8,
            synthid_state=SynthIDState.for_initialized_slots(
                torch.tensor([0]), config=config
            ),
            synthid_force_dense_draft_probs=True,
        )
        return worker, info

    def test_fused_proposal_survives_top1_draft_dispatch(self):
        """A supplied fused sample must win over the raw-logit argmax shortcut."""
        worker, info = self._build_fused_proposal_worker()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            moe_num_token_non_padded=None,
            batch_size=1,
            out_cache_loc=torch.arange(2),
            positions=torch.zeros(1, dtype=torch.int64),
            spec_info=EagleDraftInput(
                topk_p=torch.ones(1, 1),
                topk_index=torch.tensor([[0]]),
                hidden_states=torch.zeros(1, 2),
                draft_probs=torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]]),
            ),
            sampling_info=info,
        )
        with (
            get_context().override_server_args(speculative_use_rejection_sampling=True),
            patch.object(eagle_worker_v2, "_is_cuda", False),
            patch.object(eagle_worker_v2, "_is_hip", False),
            torch.compiler.set_stance("force_eager"),
        ):
            _, _, tokens, _, _ = worker.draft_forward(batch)
        self.assertEqual(tokens.tolist(), [[0, 2]])

    def test_fused_proposal_survives_top1_draft_extend_dispatch(self):
        """Draft-extend must not replace a fused sample with unmarked argmax."""
        worker, info = self._build_fused_proposal_worker()
        batch = SimpleNamespace(
            seq_lens=torch.tensor([4]),
            sampling_info=info,
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=torch.zeros(3, 2)),
            accept_lens=torch.tensor([1]),
            new_seq_lens=torch.tensor([5]),
            next_token_ids=torch.tensor([3]),
            next_draft_input=SimpleNamespace(),
            prepared_draft_extend_inputs=None,
        )
        with (
            get_context().override_server_args(speculative_use_rejection_sampling=True),
            patch.object(eagle_worker_v2, "_is_hip", False),
            patch.object(
                eagle_worker_v2,
                "prepare_for_draft_extend",
                return_value=SimpleNamespace(input_ids=torch.arange(3)),
            ),
        ):
            worker._draft_extend_for_decode(batch, result)
        self.assertEqual(result.next_draft_input.topk_index.tolist(), [[2]])
        self.assertEqual(result.next_draft_input.topk_p.tolist(), [[1.0]])

    def test_dense_capture_and_sparse_eager_draft_layouts(self):
        """Dense capture proposals must not shrink to the eager candidate cap."""
        for force_dense in (False, True):
            with self.subTest(force_dense=force_dense):
                config = SynthIDConfig(
                    keys=(1, 2), ngram_len=18, top_k=3, context_history_size=4
                )
                processor = SynthIDWatermarkProcessor(config, "cpu", num_slots=1)
                worker = EagleDraftWorker.__new__(EagleDraftWorker)
                worker.topk = 1
                worker.speculative_num_steps = 2
                worker.hot_token_id = None
                worker.index_share_for_mtp_iteration = False
                worker.seed_dsa_topk_from_draft_extend = False
                worker._topk1_parents_prealloc = torch.tensor([[-1, 0]])
                worker._topk1_score_indices_prealloc = torch.tensor([[0, 1]])
                worker.target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        sampler=SimpleNamespace(synthid_watermark=processor)
                    )
                )
                worker.draft_runner = SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(architectures=["DummyMTP"]),
                        model_is_mrope=False,
                    ),
                    canary_manager=None,
                    forward=Mock(
                        side_effect=lambda _: SimpleNamespace(
                            logits_output=SimpleNamespace(
                                next_token_logits=torch.arange(
                                    8, dtype=torch.float32
                                ).reshape(1, 8),
                                hidden_states=torch.zeros(1, 2),
                            )
                        )
                    ),
                )
                worker.draft_attn_backend = SimpleNamespace(attn_backends=[object()])
                initial_dense = torch.zeros(1, 8)
                initial_dense[:, 7] = 1.0
                initial_probs, initial_ids = initial_dense, None
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    moe_num_token_non_padded=None,
                    batch_size=1,
                    out_cache_loc=torch.arange(2),
                    positions=torch.zeros(1, dtype=torch.int64),
                    spec_info=EagleDraftInput(
                        topk_p=torch.ones(1, 1),
                        topk_index=torch.tensor([[7]]),
                        hidden_states=torch.zeros(1, 2),
                        draft_probs=initial_probs,
                        draft_prob_token_ids=initial_ids,
                    ),
                    sampling_info=SamplingBatchInfo(
                        temperatures=torch.ones(1, 1),
                        top_ps=torch.ones(1),
                        top_ks=torch.tensor([TOP_K_ALL]),
                        min_ps=torch.zeros(1),
                        is_all_greedy=False,
                        is_any_greedy=False,
                        need_top_k_sampling=False,
                        need_top_p_sampling=False,
                        need_min_p_sampling=False,
                        vocab_size=8,
                        synthid_state=SynthIDState.for_initialized_slots(
                            torch.tensor([0]), config=config
                        ),
                        synthid_force_dense_draft_probs=force_dense,
                    ),
                )
                worker._apply_synthid_to_draft_logits = Mock(
                    return_value=(
                        None,
                        (
                            initial_dense.clone(),
                            None,
                            torch.ones(1),
                            torch.tensor([7]),
                        ),
                    )
                )
                if not force_dense:
                    (
                        batch.spec_info.draft_probs,
                        batch.spec_info.draft_prob_token_ids,
                    ) = worker._ensure_sparse_algo5_draft_probs(
                        initial_dense, None, batch.sampling_info
                    )
                with (
                    get_context().override_server_args(
                        speculative_use_rejection_sampling=True
                    ),
                    patch.object(eagle_worker_v2, "_is_cuda", False),
                    torch.compiler.set_stance("force_eager"),
                ):
                    _, _, tokens, probs, ids = worker.draft_forward(batch)
                self.assertEqual(tokens.tolist(), [[7, 7]])
                self.assertEqual(tuple(probs.shape), (1, 2, 8 if force_dense else 3))
                if force_dense:
                    self.assertIsNone(ids)
                    dense = probs
                else:
                    self.assertEqual(tuple(ids.shape), (1, 2, 3))
                    dense = torch.zeros(1, 2, 8).scatter_(-1, ids, probs)
                torch.testing.assert_close(
                    dense, initial_dense[:, None, :].expand(1, 2, 8)
                )

    @patch(
        "sglang.srt.speculative.eagle_info.get_spec",
        return_value=SimpleNamespace(speculative_use_rejection_sampling=True),
    )
    def test_algorithm5_idle_input_uses_sparse_probability_layout(self, _get_spec):
        info = EagleDraftInput.create_idle_input(
            device=torch.device("cpu"),
            hidden_size=8,
            dtype=torch.float32,
            topk=1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            vocab_size=256,
            synthid_config=SimpleNamespace(
                top_k=128,
            ),
        )
        self.assertEqual(tuple(info.draft_probs.shape), (0, 128))
        self.assertEqual(tuple(info.draft_prob_token_ids.shape), (0, 128))

    def _build_runner(self, backend, graph_output=None):
        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        runner.synthid_mixed = False
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.buffers = SimpleNamespace(
            seq_lens=torch.empty(CAPTURE_BS, dtype=torch.int32),
            out_cache_loc=torch.empty(CAPTURE_BS * NUM_STEPS, dtype=torch.int64),
            positions=torch.empty(CAPTURE_BS, dtype=torch.int64),
            rids_int=None,
            bootstrap_room_ids_int=None,
            topk_p=torch.empty(CAPTURE_BS, 1, dtype=torch.float32),
            topk_index=torch.empty(CAPTURE_BS, 1, dtype=torch.int64),
            draft_probs=None,
            draft_prob_token_ids=None,
            hidden_states=torch.empty(CAPTURE_BS, 2, dtype=torch.float32),
            req_pool_indices=torch.empty(CAPTURE_BS, dtype=torch.int32),
            seq_lens_cpu=torch.empty(CAPTURE_BS, dtype=torch.int32),
            dsa_seed_topk=None,
        )
        runner.capture_bs = [1, CAPTURE_BS]
        runner.captured_req_width = 1
        runner.speculative_num_steps = NUM_STEPS
        runner.seq_len_fill_value = SEQ_LEN_FILL_VALUE
        runner.require_mlp_tp_gather = False
        runner.require_gathered_buffer = False
        runner.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=8, model_is_mrope=False),
            server_args=SimpleNamespace(speculative_use_rejection_sampling=False),
            device_timer=None,
            draft_attn_backend=backend,
        )
        runner.draft_attn_backend = backend
        runner._postprocess_output_to_raw_bs = lambda out, raw_bs: out

        def replay_graph_stub(shape_key, forward_batch):
            # Observe again during graph replay to catch a premature restore.
            backend.observe("graph_replay", forward_batch)
            return graph_output

        runner._replay_graph = replay_graph_stub
        return runner

    def _build_forward_batch(self, seq_lens, seq_lens_sum):
        raw_bs = len(seq_lens)
        return SimpleNamespace(
            batch_size=raw_bs,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens_sum=seq_lens_sum,
            out_cache_loc=torch.arange(raw_bs * NUM_STEPS, dtype=torch.int64),
            positions=torch.arange(raw_bs, dtype=torch.int64),
            req_pool_indices=torch.arange(raw_bs, dtype=torch.int32),
            rids_int=None,
            bootstrap_room_ids_int=None,
            sampling_info=None,
            spec_info=SimpleNamespace(
                topk_p=torch.ones(raw_bs, 1, dtype=torch.float32),
                topk_index=torch.zeros(raw_bs, 1, dtype=torch.int64),
                draft_probs=None,
                draft_prob_token_ids=None,
                hidden_states=torch.zeros(raw_bs, 2, dtype=torch.float32),
            ),
        )

    def _execute_and_observe(self, seq_lens, seq_lens_sum):
        backend = _RecordingDraftBackend()
        runner = self._build_runner(backend)
        forward_batch = self._build_forward_batch(seq_lens, seq_lens_sum)
        runner.execute(forward_batch)
        # Both the metadata build and the graph replay must have been observed.
        self.assertEqual(
            [observation.phase for observation in backend.observations],
            ["metadata_build", "graph_replay"],
        )
        return backend, forward_batch

    def test_padded_replay_keeps_seq_lens_sum_consistent(self):
        # raw_bs=2 < CAPTURE_BS=4: the batch is padded with fake rows of length
        # SEQ_LEN_FILL_VALUE, so every observation must see seq_lens_sum equal to
        # the sum of the padded seq_lens it was handed.
        raw_seq_lens = [10, 11]
        num_fake_rows = CAPTURE_BS - len(raw_seq_lens)
        expected_padded_seq_lens = raw_seq_lens + [SEQ_LEN_FILL_VALUE] * num_fake_rows
        expected_padded_sum = sum(expected_padded_seq_lens)
        backend, forward_batch = self._execute_and_observe(
            raw_seq_lens, sum(raw_seq_lens)
        )

        for observation in backend.observations:
            # Padding actually happened (guards against "sum matches an
            # un-padded seq_lens" false negatives).
            self.assertEqual(observation.batch_size, CAPTURE_BS, msg=observation.phase)
            self.assertEqual(
                observation.seq_lens.tolist(),
                expected_padded_seq_lens,
                msg=observation.phase,
            )
            # The invariant the fix maintains, plus the independently-derived
            # expected value.
            self.assertEqual(
                observation.seq_lens_sum,
                int(observation.seq_lens.sum()),
                msg=observation.phase,
            )
            self.assertEqual(
                observation.seq_lens_sum, expected_padded_sum, msg=observation.phase
            )

        # The raw batch shape is restored once replay finishes.
        self.assertEqual(forward_batch.batch_size, len(raw_seq_lens))
        self.assertEqual(forward_batch.seq_lens_sum, sum(raw_seq_lens))

    def test_unpadded_replay_leaves_seq_lens_sum_untouched(self):
        # raw_bs == CAPTURE_BS: no fake rows, so the padding branch is skipped
        # and seq_lens_sum must pass through unchanged.
        seq_lens = [3, 4, 5, 6]
        backend, forward_batch = self._execute_and_observe(seq_lens, sum(seq_lens))

        for observation in backend.observations:
            self.assertEqual(
                observation.seq_lens_sum, sum(seq_lens), msg=observation.phase
            )
            self.assertEqual(
                observation.seq_lens_sum,
                int(observation.seq_lens.sum()),
                msg=observation.phase,
            )
        self.assertEqual(forward_batch.seq_lens_sum, sum(seq_lens))

    def test_none_seq_lens_sum_is_preserved(self):
        # seq_lens_sum may be intentionally absent; padding must keep it None
        # rather than coerce it into an int.
        backend, forward_batch = self._execute_and_observe([10, 11], None)

        for observation in backend.observations:
            self.assertIsNone(observation.seq_lens_sum, msg=observation.phase)
        self.assertIsNone(forward_batch.seq_lens_sum)

    def _build_synthid_graph_gate(self, *, sparse: bool):
        runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
        runner.synthid_mixed = False
        runner.synthid_watermark = object()
        runner.synthid_graph_supported = True
        runner.sparse_synthid_draft_probs = sparse
        runner.captured_req_width = 1
        runner.require_mlp_tp_gather = False
        runner.require_mlp_sync = False
        runner.disable_padding = True
        runner.backend = SimpleNamespace(can_run=lambda *_args: True)
        runner.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_use_rejection_sampling=True)
        )
        return runner

    @staticmethod
    def _build_synthid_forward_batch(*, sparse: bool, idle: bool = False):
        batch_size = 0 if idle else 2
        draft_probs = (
            None
            if idle
            else torch.zeros(
                batch_size,
                1024 if sparse else 4096,
                dtype=torch.float32,
            )
        )
        draft_prob_token_ids = (
            torch.zeros(batch_size, 1024, dtype=torch.int64)
            if sparse and not idle
            else None
        )
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: idle),
            sampling_info=SimpleNamespace(
                is_all_greedy=False,
                sampling_seed=None,
                need_top_k_sampling=False,
                need_top_p_sampling=False,
                need_min_p_sampling=False,
                synthid_state=object(),
            ),
            spec_info=SimpleNamespace(
                num_tokens_per_req=1,
                draft_probs=draft_probs,
                draft_prob_token_ids=draft_prob_token_ids,
            ),
            batch_size=batch_size,
            can_run_dp_cuda_graph=True,
        )

    def test_synthid_graph_gate_accepts_matching_probability_layout(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                runner = self._build_synthid_graph_gate(sparse=sparse)
                batch = self._build_synthid_forward_batch(sparse=sparse)
                self.assertTrue(runner.can_run_graph(batch))

    def test_synthid_graph_gate_rejects_incompatible_batches(self):
        runner = self._build_synthid_graph_gate(sparse=True)
        for flag in (
            "missing_draft_probs",
            "sampling_seed",
            "need_top_k_sampling",
            "need_top_p_sampling",
            "need_min_p_sampling",
        ):
            with self.subTest(flag=flag):
                batch = self._build_synthid_forward_batch(sparse=True)
                if flag == "missing_draft_probs":
                    batch.spec_info.draft_probs = None
                elif flag == "sampling_seed":
                    batch.sampling_info.sampling_seed = torch.tensor([42, 43])
                else:
                    setattr(batch.sampling_info, flag, True)
                self.assertFalse(runner.can_run_graph(batch))

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.get_spec",
        return_value=SimpleNamespace(speculative_use_rejection_sampling=True),
    )
    def test_greedy_synthid_draft_uses_unwatermarked_argmax(self, _get_spec):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            synthid_state=object(),
            is_all_greedy=True,
            need_top_k_sampling=True,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            top_ks=torch.tensor([1]),
            top_ps=torch.tensor([1.0]),
            min_ps=torch.tensor([0.0]),
        )
        proposal_probs, proposal_tokens, sampled_probs, sampled_tokens = (
            worker._sample_draft_logits(
                torch.tensor([[0.3, 0.5, 0.2]]).log(),
                sampling_info,
            )
        )

        self.assertEqual(sampled_tokens.item(), 1)
        self.assertEqual(sampled_probs.item(), 1.0)
        self.assertIsNone(proposal_tokens)
        torch.testing.assert_close(
            proposal_probs,
            torch.tensor([[0.0, 1.0, 0.0]]),
        )

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.get_spec",
        return_value=SimpleNamespace(speculative_use_rejection_sampling=True),
    )
    def test_synthid_draft_does_not_refilter_watermarked_probs(self, _get_spec):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            synthid_state=object(),
            is_all_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=True,
            top_ks=torch.tensor([1 << 30]),
            top_ps=torch.tensor([1.0]),
            min_ps=torch.tensor([0.5]),
        )
        sparse_output = SynthIDSparseOutput(
            scores=torch.tensor([[0.9, 0.1]]).log(),
            token_ids=torch.tensor([[13, 11]], dtype=torch.int64),
        )

        proposal_probs, proposal_tokens, _, _ = worker._sample_draft_logits(
            torch.empty((1, 0)),
            sampling_info,
            synthid_sparse_output=sparse_output,
        )

        self.assertEqual(proposal_tokens.tolist(), [[11, 13]])
        torch.testing.assert_close(proposal_probs, torch.tensor([[0.1, 0.9]]))

    def test_synthid_idle_graph_requires_supported_shape(self):
        batch = self._build_synthid_forward_batch(sparse=True, idle=True)
        for supported in (False, True):
            with self.subTest(supported=supported):
                runner = self._build_synthid_graph_gate(sparse=True)
                runner.synthid_graph_supported = supported
                self.assertEqual(runner.can_run_graph(batch), supported)

    def test_seeded_draft_dispatch_preserves_seed_position_and_step(self):
        """The SynthID dispatch must not discard upstream's per-request RNG keys."""
        from sglang.srt.speculative.spec_utils import fast_sample

        def hash_coordinates(seed, position, columns):
            state = (
                seed.to(torch.int64)[:, None]
                ^ (position.to(torch.int64)[:, None] * 0x9E3779B9)
                ^ (columns.to(torch.int64)[None, :] * 0x85EBCA6B)
            )
            state = state ^ (state >> 16)
            return ((state * 0x7FEB352D) & 0xFFFFFFFF).to(torch.uint32)

        seeds = torch.arange(16, dtype=torch.int64)
        positions = torch.arange(20, 36, dtype=torch.int64)
        probs = torch.full((16, 2), 0.5)
        sparse_tokens = torch.tensor([[13, 11]]).expand(16, -1)
        self.enterContext(
            patch(
                "sglang.kernels.ops.sampling.murmur_hash.murmur_hash32",
                side_effect=hash_coordinates,
            )
        )
        expected_probs, expected_indices = fast_sample(
            probs, sampling_seed=seeds, positions=positions, draft_step=2
        )
        for enabled in (False, True):
            with (
                self.subTest(watermarked=enabled),
                get_context().override_server_args(
                    speculative_use_rejection_sampling=True
                ),
                torch.random.fork_rng(devices=[]),
            ):
                worker = EagleDraftWorker.__new__(EagleDraftWorker)
                worker.topk = 1
                info = SimpleNamespace(
                    synthid_state=object() if enabled else None,
                    sampling_seed=seeds,
                    temperatures=torch.ones(16, 1),
                    top_ks=torch.full((16,), TOP_K_ALL),
                    is_all_greedy=False,
                )
                sparse = (
                    SynthIDSparseOutput(scores=probs.log(), token_ids=sparse_tokens)
                    if enabled
                    else None
                )
                expected_tokens = (
                    sparse_tokens.sort(dim=-1).values.gather(1, expected_indices)
                    if enabled
                    else expected_indices
                )
                for rng_seed in (7, 99):
                    torch.manual_seed(rng_seed)
                    _, _, actual_probs, actual_tokens = worker._sample_draft_logits(
                        probs.log(),
                        info,
                        synthid_sparse_output=sparse,
                        positions=positions,
                        draft_step=2,
                    )
                    torch.testing.assert_close(actual_probs, expected_probs)
                    torch.testing.assert_close(actual_tokens, expected_tokens)

    def test_execute_leaves_graph_output_ownership_to_caller(self):
        backend = _RecordingDraftBackend()
        pooled_probs = torch.ones(CAPTURE_BS, NUM_STEPS, 8)
        graph_output = (None, None, None, pooled_probs)
        runner = self._build_runner(backend, graph_output=graph_output)
        forward_batch = self._build_forward_batch([3, 4, 5, 6], 18)

        output = runner.execute(forward_batch)

        self.assertIs(output[-1], pooled_probs)


if __name__ == "__main__":
    unittest.main()
