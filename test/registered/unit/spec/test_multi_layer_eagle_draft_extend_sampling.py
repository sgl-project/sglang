"""Who draws multi-layer EAGLE's rejection-sampling proposal on graph replay.

Runs the production ``_draft_extend_for_decode`` over the real draft-extend
graph runners. Only the boundaries are stubbed: graph capture records each
step function and replay re-runs it, the draft model is a deterministic
function of each request's last token, and the Triton kernels have torch
references.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu.graph_runner.multi_layer_eagle_draft_extend_npu_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendNpuGraphRunner,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
    OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_utils import (
    fill_draft_extend_prepare_buffers_native,
)
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import sample_draft_proposal
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 11
HIDDEN = 4
NUM_STEPS = 3
WIDTH = NUM_STEPS + 1
CAPTURE_BS = (2, 4)

RUNNER = "sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner"
WORKER = "sglang.srt.speculative.multi_layer_eagle_worker_v2"


def _rotate_input_ids(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    """rotate_input_ids: shift each request's window left by one and write
    the new token at its selected row."""
    for i in range(extend_seq_lens.shape[0]):
        start, length = int(extend_start_loc[i]), int(extend_seq_lens[i])
        window = input_ids[start : start + length]
        window[:-1] = window[1:].clone()
        last = int(select_index[i]) if select_index is not None else start + length - 1
        input_ids[last] = topk_index.view(-1)[i]
    return input_ids


def _murmur_hash32(seeds, positions, cols):
    """murmur_hash32's contract: one uint32 per (row seed, row position, column)."""
    keys = (
        seeds.to(torch.int64)[:, None] * 1000003
        + positions.to(torch.int64)[:, None] * 7919
        + cols[None, :] * 104729
    )
    return torch.remainder(keys, 2**32)


def _wide_row_softmax(logits, temperatures, out):
    out.copy_(torch.softmax(logits / temperatures, dim=-1))
    return out


class _DraftModel:
    """Each step's draft model: logits peaked at a function of the request's
    last token, so a step's proposal decides the next step's logits."""

    shared_experts_config = None

    def __init__(self, step, calls):
        self.step, self.calls = step, calls

    def forward(self, input_ids, positions, forward_batch):
        last = input_ids[forward_batch.spec_info.select_index].clone()
        target = (last * 3 + self.step + 1) % VOCAB
        offset = (torch.arange(VOCAB)[None, :] - target[:, None]) % VOCAB
        logits = -0.5 * offset.to(torch.float32)
        self.calls.append((self.step, last, logits.clone()))
        return LogitsProcessorOutput(next_token_logits=logits, hidden_states=None)


class _GraphBoundary:
    """Capture records the step function and replay re-runs it. Per-step
    graphs share one memory pool, so a replay writes its top-k into storage
    the next step's replay overwrites."""

    def __init__(self, pool):
        self.pool = pool
        self.graphs = {}

    def capture_one(self, shape_key, fn, capture_inputs=None, post_warmup_hook=None):
        self.graphs[shape_key] = fn

    def replay(self, shape_key, forward_batch):
        out = self.graphs[shape_key]()
        if isinstance(out, list):  # one graph for all steps: outputs coexist
            return out
        for name in ("topk_p", "topk_index"):
            value = getattr(out, name)
            slot = self.pool.setdefault(name, torch.empty((8, 1), dtype=value.dtype))
            slot[: value.shape[0]].copy_(value)
            setattr(out, name, slot[: value.shape[0]])
        return out


class _AttnBackend:
    supports_draft_extend_metadata_staging = False

    def draft_extend_metadata_captured_in_graph(self):
        return True

    def init_forward_metadata_in_graph(self, forward_batch):
        pass

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        pass


def _step_runner(eagle_worker, step):
    """A per-step runner with the fields its __init__ would read off a model
    runner and attention backend."""
    runner = object.__new__(MultiLayerEagleDraftExtendCudaGraphRunner)
    runner.step = step
    runner.eagle_worker = eagle_worker
    runner.model_runner = eagle_worker.mtp_model_runner(step)
    runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    runner.device = "cpu"
    runner.num_dp_ranks = 1
    runner.enable_torch_compile = False
    runner.disable_padding = False
    runner.require_gathered_buffer = False
    runner.require_mlp_tp_gather = False
    runner.require_mlp_sync = False
    runner.require_attn_tp_gather = False
    runner.topk = 1
    runner.attn_backend = eagle_worker.draft_extend_attn_backend_list[step]
    runner.metadata_captured_in_graph = True
    runner.prune_draft_extend_logits = True
    runner.capture_bs = list(CAPTURE_BS)
    runner.max_bs = max(CAPTURE_BS)
    runner.num_front_tokens = 0
    runner.captured_req_width = WIDTH
    runner.extend_seq_lens_cpu = [WIDTH] * runner.max_bs
    runner.seq_len_fill_value = 1
    runner.deepep_adapter = SimpleNamespace(
        capture=lambda **_: None, replay=lambda: None
    )
    runner.capture = lambda: [
        runner.capture_one_shape(bs, None) for bs in reversed(CAPTURE_BS)
    ]
    return runner


class _FakeStepRunners:
    def _create_runner(self, step):
        return _step_runner(self.eagle_worker, step)

    def _cuda_graph_disabled(self):
        return False


class _PerStepGraphs(
    _FakeStepRunners, MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
):
    pass


class _PerStepGraphsWithoutProposal(_PerStepGraphs):
    captures_proposal = False


class _OneGraph(
    _FakeStepRunners, OneGraphMultiLayerEagleMultiStepDraftExtendCudaGraphRunner
):
    pass


def _make_worker(runner_cls, *, rejection_sampling, deterministic=False):
    worker = object.__new__(MultiLayerEagleDraftWorker)
    worker.calls = []
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        vocab_size=VOCAB,
        spec_hidden_size=HIDDEN,
        dtype=torch.float32,
    )
    worker.draft_runner_list = [
        SimpleNamespace(
            model=_DraftModel(step, worker.calls),
            model_config=model_config,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            is_draft_worker=False,
            device_timer=None,
            kv_index_translator=SimpleNamespace(bind_runner_slots=lambda fb: None),
        )
        for step in range(NUM_STEPS)
    ]
    worker.draft_extend_attn_backend_list = [_AttnBackend() for _ in range(NUM_STEPS)]
    worker.device = "cpu"
    worker.gpu_id = 0
    worker.topk = 1
    worker.speculative_num_steps = NUM_STEPS
    worker.speculative_num_draft_tokens = NUM_STEPS + 1
    worker.use_rejection_sampling = rejection_sampling
    worker.chain_mtp_hidden_states = False
    worker.draft_extend_num_front_tokens = 0
    worker.plan_stream = None
    worker.plan_stream_ctx = contextlib.nullcontext()
    worker._compute_boundary_kv_locs_positions = lambda batch: (None, None)
    worker._apply_deferred_mamba_init_to_draft_pools = lambda forward_batch: None
    worker._fill_boundary_kv_front_and_update_stash = lambda *args: None
    pool = {}
    exec_config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic),
        mamba=SimpleNamespace(enable_mamba_extra_buffer=False),
    )
    with (
        patch(f"{RUNNER}.resolve_decode_backend", lambda runner: _GraphBoundary(pool)),
        patch(f"{RUNNER}.get_exec", lambda: exec_config),
    ):
        worker.cuda_graph_runner_for_draft_extend = runner_cls(worker)
    return worker


def _decode_round(worker, *, tokens, accept_lens, temperatures, sampling_seed=None):
    """One _draft_extend_for_decode call; returns the next draft input."""
    raw_bs = accept_lens.shape[0]
    num_tokens = raw_bs * WIDTH
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        batch_size=raw_bs,
        input_ids=tokens.clone(),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        out_cache_loc=torch.zeros(num_tokens, dtype=torch.int64),
        seq_lens=torch.full((raw_bs,), 16, dtype=torch.int32),
        seq_lens_cpu=None,
        seq_lens_sum=16 * raw_bs,
        req_pool_indices=torch.arange(raw_bs, dtype=torch.int64),
        mamba_track_indices=None,
        out_cache_loc_virtual=None,
        kv_loc_plan=None,
        kv_loc_cols=None,
        spec_info=SimpleNamespace(
            hidden_states=torch.zeros((num_tokens, HIDDEN)),
            num_tokens_per_req=WIDTH,
        ),
        sampling_info=SimpleNamespace(
            temperatures=temperatures, sampling_seed=sampling_seed
        ),
    )
    batch_result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.zeros((num_tokens, HIDDEN))),
        next_token_ids=tokens.clone(),
        accept_lens=accept_lens,
        next_draft_input=SimpleNamespace(),
        can_run_cuda_graph=True,
    )
    with patch(f"{WORKER}.prepare_for_draft_extend", lambda *a, **kw: forward_batch):
        worker._draft_extend_for_decode(SimpleNamespace(), batch_result)
    return batch_result.next_draft_input


class TestDraftExtendProposalOwnership(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(model_path="dummy")
        override.install()
        self.addCleanup(override.restore)
        for target, value in (
            (
                f"{RUNNER}.fill_draft_extend_prepare_buffers",
                fill_draft_extend_prepare_buffers_native,
            ),
            (f"{RUNNER}.wide_row_softmax_triton", _wide_row_softmax),
            (f"{RUNNER}.rotate_input_ids", _rotate_input_ids),
            (f"{RUNNER}.get_available_gpu_memory", lambda *a: 0.0),
            (f"{WORKER}.rotate_input_ids", _rotate_input_ids),
            (
                "sglang.srt.speculative.spec_utils.row_argmax",
                lambda x: x.argmax(dim=-1),
            ),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch(f"{WORKER}.sample_draft_proposal", wraps=sample_draft_proposal)
        self.eager_proposal = patcher.start()
        self.addCleanup(patcher.stop)
        torch.manual_seed(0)

    def _round(self, worker, raw_bs=3, sampling_seed=None, generator=None):
        tokens = torch.randint(0, VOCAB, (raw_bs * WIDTH,), generator=generator)
        accept_lens = torch.tensor([2, 1, 4, 3][:raw_bs], dtype=torch.int32)
        temperatures = torch.tensor([[0.7], [1.0], [1.3], [0.9]][:raw_bs])
        worker.calls.clear()
        out = _decode_round(
            worker,
            tokens=tokens,
            accept_lens=accept_lens,
            temperatures=temperatures,
            sampling_seed=sampling_seed,
        )
        return out, temperatures, list(worker.calls)

    def _check_chain(self, out, calls, raw_bs=3):
        """Each step's returned proposal is the token the next step reads."""
        self.assertEqual([step for step, _, _ in calls], list(range(NUM_STEPS)))
        self.assertEqual(out.topk_index.shape, (raw_bs, NUM_STEPS))
        for step in range(NUM_STEPS - 1):
            self.assertEqual(
                calls[step + 1][1][:raw_bs].tolist(), out.topk_index[:, step].tolist()
            )

    def _check_proposal(self, worker, out, temperatures, calls, raw_bs=3):
        """q is softmax(logits / T) per step, X ~ q, q(X) = q[X], and q is the
        worker's own copy, not the runner's buffer."""
        cgr = worker.cuda_graph_runner_for_draft_extend
        self.assertEqual(out.draft_probs.shape, (raw_bs, NUM_STEPS, VOCAB))
        for step, _, logits in calls:
            expected = torch.softmax(logits[:raw_bs] / temperatures, dim=-1)
            torch.testing.assert_close(out.draft_probs[:, step], expected)
        torch.testing.assert_close(
            out.topk_p,
            out.draft_probs.gather(2, out.topk_index.unsqueeze(-1)).squeeze(-1),
            rtol=0,
            atol=0,
        )
        if cgr.buffers.draft_probs is not None:
            self.assertNotEqual(
                out.draft_probs.data_ptr(), cgr.buffers.draft_probs.data_ptr()
            )

    def _check_survives_next_round(self, worker, out):
        held = (out.topk_index.clone(), out.topk_p.clone(), out.draft_probs.clone())
        self._round(worker)
        torch.testing.assert_close(out.topk_index, held[0], rtol=0, atol=0)
        torch.testing.assert_close(out.topk_p, held[1], rtol=0, atol=0)
        torch.testing.assert_close(out.draft_probs, held[2], rtol=0, atol=0)

    def test_graphs_without_proposal_resample_in_the_worker(self):
        """(samples_in_graph, rotates_in_graph) = (False, False)."""
        worker = _make_worker(_PerStepGraphsWithoutProposal, rejection_sampling=True)
        cgr = worker.cuda_graph_runner_for_draft_extend
        self.assertFalse(cgr.samples_in_graph)
        self.assertIsNone(cgr.buffers.draft_probs)
        out, temperatures, calls = self._round(worker)
        self.assertEqual(self.eager_proposal.call_count, NUM_STEPS)
        self._check_chain(out, calls)
        self._check_proposal(worker, out, temperatures, calls)
        self._check_survives_next_round(worker, out)

    def test_per_step_graphs_sample_in_graph(self):
        """(True, False): the worker rotates and clones each step's top-k."""
        worker = _make_worker(_PerStepGraphs, rejection_sampling=True)
        cgr = worker.cuda_graph_runner_for_draft_extend
        self.assertTrue(cgr.samples_in_graph)
        self.assertFalse(cgr.rotates_in_graph)
        out, temperatures, calls = self._round(worker)
        self.eager_proposal.assert_not_called()
        self._check_chain(out, calls)
        self._check_proposal(worker, out, temperatures, calls)
        self._check_survives_next_round(worker, out)

    def test_one_graph_samples_and_rotates_in_graph(self):
        """(True, True)."""
        worker = _make_worker(_OneGraph, rejection_sampling=True)
        cgr = worker.cuda_graph_runner_for_draft_extend
        self.assertTrue(cgr.samples_in_graph)
        self.assertTrue(cgr.rotates_in_graph)
        out, temperatures, calls = self._round(worker)
        self.eager_proposal.assert_not_called()
        self._check_chain(out, calls)
        self._check_proposal(worker, out, temperatures, calls)
        self._check_survives_next_round(worker, out)

    def test_seeded_proposals_match_the_worker_side_draws(self):
        """Under deterministic inference the per-step graphs draw the seeded
        proposal the worker draws when its graphs cannot: the same seeds,
        positions and draft steps, so the same tokens whatever the RNG state."""
        seeds = torch.tensor([11, 22, 33])
        hashed = MagicMock(side_effect=_murmur_hash32)
        drawn = {}
        with patch("sglang.kernels.ops.sampling.murmur_hash.murmur_hash32", hashed):
            for runner_cls in (_PerStepGraphsWithoutProposal, _PerStepGraphs):
                worker = _make_worker(
                    runner_cls, rejection_sampling=True, deterministic=True
                )
                cgr = worker.cuda_graph_runner_for_draft_extend
                self.assertEqual(
                    cgr.buffers.sampling_seed is not None, cgr.samples_in_graph
                )
                rounds = []
                for rng_seed in (1, 2):
                    torch.manual_seed(rng_seed)
                    hashed.reset_mock()
                    out, temperatures, calls = self._round(
                        worker,
                        sampling_seed=seeds,
                        generator=torch.Generator().manual_seed(0),
                    )
                    self._check_chain(out, calls)
                    self._check_proposal(worker, out, temperatures, calls)
                    rounds.append(out.topk_index)
                    # Each step's draw is keyed by the request's seed and
                    # position, and by the step through the hash columns.
                    self.assertEqual(hashed.call_count, NUM_STEPS)
                    for step, call in enumerate(hashed.call_args_list):
                        salted, positions, cols = call.args
                        self.assertEqual(
                            (salted[:3].to(torch.int64) - seeds).unique().numel(), 1
                        )
                        self.assertEqual(positions[:3].tolist(), [16, 16, 16])
                        self.assertEqual(cols[0].item(), step * VOCAB)
                self.assertEqual(rounds[0].tolist(), rounds[1].tolist())
                drawn[runner_cls] = rounds[0]
        self.assertEqual(self.eager_proposal.call_count, 2 * NUM_STEPS)
        self.assertEqual(
            drawn[_PerStepGraphsWithoutProposal].tolist(),
            drawn[_PerStepGraphs].tolist(),
        )

    def test_rejection_sampling_off_takes_the_top_token(self):
        worker = _make_worker(_PerStepGraphs, rejection_sampling=False)
        cgr = worker.cuda_graph_runner_for_draft_extend
        self.assertFalse(cgr.samples_in_graph)
        self.assertIsNone(cgr.buffers.temperatures)
        out, _, calls = self._round(worker)
        self.eager_proposal.assert_not_called()
        self.assertIsNone(out.draft_probs)
        self._check_chain(out, calls)
        for step, _, logits in calls:
            self.assertEqual(
                out.topk_index[:, step].tolist(), logits[:3].argmax(dim=-1).tolist()
            )

    def test_samples_in_graph_needs_captured_graphs(self):
        worker = _make_worker(_PerStepGraphs, rejection_sampling=True)
        with patch.object(_PerStepGraphs, "_cuda_graph_disabled", lambda self: True):
            disabled = _PerStepGraphs(worker)
        self.assertIsNone(disabled.buffers)
        self.assertFalse(disabled.samples_in_graph)

    def test_npu_graphs_leave_the_proposal_to_the_worker(self):
        self.assertFalse(
            MultiLayerEagleMultiStepDraftExtendNpuGraphRunner.captures_proposal
        )
        self.assertTrue(
            MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.captures_proposal
        )


if __name__ == "__main__":
    unittest.main()
