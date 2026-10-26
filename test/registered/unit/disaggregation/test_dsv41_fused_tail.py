"""SGLANG_DSV41_FUSED_TAIL: decode tails fed through the DSpark verify steps."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.utils import dsv41_encoder_prefix_len
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.srt.speculative.dspark_components.dspark_verify import (
    build_accept_outs,
    force_verify_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Mixin = SchedulerDisaggregationDecodeMixin
WIDTH = 6


def _fed_req(prompt_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(1000, 20000, (prompt_len,), generator=g).tolist()
    start = dsv41_encoder_prefix_len(prompt_len, 128)
    return SimpleNamespace(
        origin_input_ids=prompt,
        dsv41_feed_floor=start,
        dsv41_feed_next=start,
        dsv41_feed_prompt_left=prompt_len - 1 - start,
    )


def _decode_batch(reqs):
    return SimpleNamespace(
        reqs=reqs,
        forward_mode=ForwardMode.DECODE,
        device="cpu",
        dsv41_forced_tokens=None,
        dsv41_swa_floor=None,
    )


def _sched():
    return SimpleNamespace(model_worker=SimpleNamespace(verify_num_draft_tokens=WIDTH))


class TestFeedSchedule(CustomTestCase):
    def test_steps_force_exactly_the_prompt_after_the_start(self):
        req = _fed_req(1000)
        start = req.dsv41_feed_floor
        self.assertEqual(start, 872)
        normal = SimpleNamespace(dsv41_feed_next=None)
        sched = _sched()
        row0 = start  # position of verify row 0 (holding the prompt token there)
        steps = 0
        while req.dsv41_feed_next is not None:
            batch = _decode_batch([normal, req])
            Mixin._set_dsv41_feed_inputs(sched, batch)
            forced, floor = batch.dsv41_forced_tokens, batch.dsv41_swa_floor
            steps += 1
            self.assertEqual(forced.shape, (2, WIDTH))
            self.assertEqual(forced[0].tolist(), [-1] * WIDTH)
            self.assertEqual(floor.tolist(), [0, start])
            # Row j sits at row0 + j and is forced to the prompt token after it.
            for j in range(WIDTH):
                want = req.origin_input_ids[row0 + j + 1] if row0 + j + 1 < 1000 else -1
                self.assertEqual(forced[1, j].item(), want, (steps, j))
            row0 += WIDTH
        # 128 tail tokens, six per step; the last row of the last step sits at the
        # final prompt position and samples the first output token.
        self.assertEqual(steps, 22)
        self.assertEqual(row0 - WIDTH, 998)

        batch = _decode_batch([normal, req])
        Mixin._set_dsv41_feed_inputs(sched, batch)
        self.assertIsNone(batch.dsv41_forced_tokens)
        self.assertIsNone(batch.dsv41_swa_floor)

    def test_only_decode_steps_get_feed_inputs(self):
        req = _fed_req(1000)
        batch = _decode_batch([req])
        batch.forward_mode = ForwardMode.EXTEND
        Mixin._set_dsv41_feed_inputs(_sched(), batch)
        self.assertIsNone(batch.dsv41_forced_tokens)
        self.assertEqual(req.dsv41_feed_next, 872)


class TestForceVerifyRows(CustomTestCase):
    def test_forced_rows_pick_their_token_and_others_stay_bitwise(self):
        torch.manual_seed(0)
        bs, vocab = 3, 50
        logits = torch.randn(bs * WIDTH + 2, vocab)  # graph padding rows at the end
        original = logits.clone()
        forced = torch.full((bs, WIDTH), -1, dtype=torch.int64)
        forced[1] = torch.tensor([7, 8, 9, 10, 11, 12])
        forced[2, :2] = torch.tensor([3, 49])
        force_verify_rows(logits, forced)

        flat = forced.reshape(-1)
        for row in range(bs * WIDTH + 2):
            if row < flat.numel() and flat[row] >= 0:
                token = flat[row].item()
                self.assertEqual(logits[row].argmax().item(), token)
                self.assertEqual(logits[row, token].item(), 0.0)
                self.assertTrue(torch.isinf(logits[row]).sum() == vocab - 1)
                # Any temperature keeps it one-hot.
                probs = torch.softmax(logits[row] / 0.7, dim=-1)
                self.assertEqual(probs[token].item(), 1.0)
            else:
                self.assertTrue(torch.equal(logits[row], original[row]), row)


class TestSimulatedAcceptance(CustomTestCase):
    """SGLANG_SIMULATE_ACC_LEN replaces every request's accept length. A fed tail
    must keep its served accept: _set_dsv41_feed_inputs advances the feed by the
    whole verify width, assuming every forced row is accepted."""

    VOCAB = 64

    def _accept(self, forced, drafts, simulated_len, sampling_info=None):
        bs = forced.shape[0]
        torch.manual_seed(1)
        logits = torch.randn(bs * WIDTH, self.VOCAB)
        original = logits.clone()
        candidates = torch.cat([torch.zeros(bs, 1, dtype=torch.long), drafts], dim=1)
        force_verify_rows(logits, forced)
        correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
            candidates=candidates, target_logits=logits, verify_num_draft_tokens=WIDTH
        )
        accept = build_accept_outs(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=torch.full((bs,), 1000),
            draft_tokens=drafts,
            gamma=WIDTH - 1,
            verify_num_draft_tokens=WIDTH,
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
            sync_site=None,
            simulated_correct_len=torch.tensor([simulated_len - 1]),
            target_logits=logits,
            simulate_bonus_sampling_info=sampling_info,
            forced_rows=(forced >= 0).any(dim=1),
        )
        return accept, original

    def test_sampled_simulated_bonus_spares_fed_rows(self):
        # SGLANG_SIMULATE_ACC_GREEDY draws the bonus at the simulated row (a Triton
        # sampler; stubbed here to a sentinel). The fed row keeps its served bonus.
        prompt = [7, 8, 9, 10, 11, 12]
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1] = torch.tensor(prompt)
        drafts = torch.tensor([[1, 2, 3, 4, 5], prompt[:5]])
        sentinel = self.VOCAB - 1
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "sample_simulated_bonus",
            lambda *, greedy_bonus, **_: torch.full_like(greedy_bonus, sentinel),
        ):
            accept, _ = self._accept(forced, drafts, 2, sampling_info=object())
        self.assertEqual(accept.out_tokens[0, :2].tolist(), [1, sentinel])
        self.assertEqual(accept.out_tokens[1].tolist(), prompt)

    def test_mid_feed_commits_the_whole_forced_width(self):
        prompt = [7, 8, 9, 10, 11, 12]
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1] = torch.tensor(prompt)
        drafts = torch.tensor([[1, 2, 3, 4, 5], prompt[:5]])
        accept, _ = self._accept(forced, drafts, simulated_len=2)
        # The ordinary request takes the simulated length; the fed one all six.
        self.assertEqual(accept.commit_lens.tolist(), [2, 6])
        self.assertEqual(accept.out_tokens[1].tolist(), prompt)
        self.assertEqual(accept.new_seq_lens.tolist(), [1002, 1006])

    def test_last_feed_step_ends_at_the_served_bonus(self):
        # Two prompt tokens left, then real rows: the draft after the prompt is
        # chosen wrong, so the served accept stops right after the prompt and the
        # first output token is the target's argmax at the last prompt position.
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1, :2] = torch.tensor([7, 8])
        torch.manual_seed(1)
        first_out = torch.randn(2 * WIDTH, self.VOCAB)[WIDTH + 2].argmax().item()
        drafts = torch.tensor(
            [[1, 2, 3, 4, 5], [7, 8, (first_out + 1) % self.VOCAB, 30, 31]]
        )
        accept, original = self._accept(forced, drafts, simulated_len=5)
        self.assertEqual(original[WIDTH + 2].argmax().item(), first_out)
        self.assertEqual(accept.commit_lens.tolist(), [5, 3])
        self.assertEqual(accept.out_tokens[1, :3].tolist(), [7, 8, first_out])


def _processor():
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        # Main snapshots forward-pass metrics before resolving the tokens.
        metrics_reporter=SimpleNamespace(
            snapshot_spec_decode_metrics=lambda *a, **k: None
        ),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        beam_coordinator=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _live_req(rid, prompt_left=0):
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
        return_logprob=True,
    )
    req.kv.kv_committed_len = 0
    req.dsv41_feed_prompt_left = prompt_left
    return req


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = False
        self.forward_mode = ForwardMode.DECODE
        self.spec_algorithm = SimpleNamespace(
            is_none=lambda: False, is_dflash=lambda: False
        )


class TestFedResults(CustomTestCase):
    def test_prompt_tokens_commit_to_kv_but_never_reach_output(self):
        # Three requests: an ordinary one (3 accepted), one mid-feed (6 prompt
        # tokens), one finishing its feed (2 prompt tokens, then 2 outputs).
        normal, feeding, finishing = (
            _live_req("n"),
            _live_req("f", prompt_left=10),
            _live_req("e", prompt_left=2),
        )
        batch = _Batch([normal, feeding, finishing])
        stride = WIDTH
        tokens = [0] * (3 * stride)
        tokens[0:3] = [101, 102, 103]
        tokens[stride : stride + 6] = [201, 202, 203, 204, 205, 206]
        tokens[2 * stride : 2 * stride + 4] = [301, 302, 303, 304]
        result = GenerationBatchResult(
            next_token_ids=torch.tensor(tokens, dtype=torch.long),
            accept_lens=torch.tensor([3, 6, 4], dtype=torch.long),
            speculative_num_draft_tokens=WIDTH,
        )
        with envs.SGLANG_DSV41_FUSED_TAIL.override(True):
            out = _processor()._resolve_spec_v2_tokens(result, batch)

        self.assertEqual(out, [[101, 102, 103], [], [303, 304]])
        self.assertEqual([r.kv.kv_committed_len for r in batch.reqs], [3, 6, 4])
        self.assertEqual(feeding.dsv41_feed_prompt_left, 4)
        self.assertEqual(finishing.dsv41_feed_prompt_left, 0)
        self.assertEqual(result.num_fed_only_reqs, 1)
        # Generated tokens = correct drafts + one per request = real outputs.
        self.assertEqual(result.get_num_generated_tokens(3), 3 + 0 + 2)
        # Spec stats skip fed verifies; the normal request counts as usual.
        self.assertEqual([r.spec_verify_ct for r in batch.reqs], [1, 0, 0])

        # Output logprobs skip the fed prompt rows like the tokens do.
        row_logprobs = [[-0.1 * (10 * i + j) for j in range(stride)] for i in range(3)]
        _processor()._apply_decode_logprobs(
            req=finishing,
            i=2,
            batch=batch,
            next_token_id=out[2],
            next_token_logprobs=row_logprobs,
            logits_output=SimpleNamespace(),
            fed_per_req=result.num_fed_per_req,
        )
        self.assertEqual(finishing.logprob.output_token_logprobs_idx, [303, 304])
        self.assertEqual(
            finishing.logprob.output_token_logprobs_val, row_logprobs[2][2:4]
        )


def _waiting_req(prompt_len, **fields):
    start = dsv41_encoder_prefix_len(prompt_len, 128)
    values = dict(
        origin_input_ids=list(range(prompt_len)),
        dsv41_tail_start=start,
        kv=SimpleNamespace(kv_allocated_len=start),
        output_ids=[],
        return_logprob=False,
        grammar=None,
        time_stats=MagicMock(),
        dsv41_feed_floor=None,
        dsv41_feed_next=None,
        dsv41_feed_prompt_left=0,
    )
    values.update(fields)
    return SimpleNamespace(**values)


def _admit_sched(
    waiting, running_size, graph_bs=(1, 2, 4, 8, 16, 32, 64), max_running=64
):
    running = MagicMock()
    running.is_empty.return_value = running_size == 0
    running.batch_size.return_value = running_size
    return SimpleNamespace(
        waiting_queue=list(waiting),
        req_to_token_pool=MagicMock(size=64),
        max_running_requests=max_running,
        tp_worker=SimpleNamespace(get_decode_cuda_graph_bs=lambda: list(graph_bs)),
        token_to_kv_pool_allocator=MagicMock(
            page_size=256, available_size=MagicMock(return_value=1 << 20)
        ),
        tree_cache=None,
        model_config=None,
        enable_overlap=False,
        spec_algorithm=None,
        ngram_embedding_manager=MagicMock(),
        future_map=MagicMock(),
    ), running


@patch("sglang.srt.disaggregation.decode.set_time_batch")
@patch("sglang.srt.disaggregation.decode.ScheduleBatch")
class TestAdmission(CustomTestCase):
    def test_eligible_tails_join_the_running_batch(self, schedule_batch, _):
        fed = _waiting_req(1000)
        constrained = _waiting_req(1000, grammar=object())
        short = _waiting_req(100)  # nothing transferred
        sched, running = _admit_sched([constrained, fed, short], running_size=4)
        with envs.SGLANG_DSV41_FUSED_TAIL.override(True):
            out = Mixin._admit_dsv41_fed_tails(sched, running)
        self.assertIs(out, running)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], [fed])
        new_batch = schedule_batch.init_new.return_value
        new_batch.prepare_for_prebuilt.assert_called_once()
        new_batch.process_prebuilt.assert_called_once_with(sched.future_map)
        running.merge_batch.assert_called_once_with(new_batch)
        self.assertEqual(sched.waiting_queue, [constrained, short])
        self.assertEqual((fed.prefix_len, fed.extend_end), (0, 872))
        self.assertEqual((fed.dsv41_feed_floor, fed.dsv41_feed_next), (872, 872))
        self.assertEqual(fed.dsv41_feed_prompt_left, 127)
        self.assertIsNone(fed.dsv41_tail_start)

    def test_idle_rank_or_flag_off_keeps_the_extend(self, schedule_batch, _):
        for running_size, flag in ((0, True), (4, False)):
            sched, running = _admit_sched([_waiting_req(1000)], running_size)
            with envs.SGLANG_DSV41_FUSED_TAIL.override(flag):
                self.assertIs(Mixin._admit_dsv41_fed_tails(sched, running), running)
            self.assertEqual(len(sched.waiting_queue), 1)
        schedule_batch.init_new.assert_not_called()

    def _admit(self, running_size, **limits):
        waiting = [_waiting_req(1000) for _ in range(5)]
        sched, running = _admit_sched(waiting, running_size, **limits)
        with envs.SGLANG_DSV41_FUSED_TAIL.override(True):
            Mixin._admit_dsv41_fed_tails(sched, running)
        return waiting, sched

    def test_feeding_stops_at_the_largest_decode_graph(self, schedule_batch, _):
        # Graphs up to bs 8, 5 running: 3 slots stay graphed. The other two
        # tails stay queued for the extend, in arrival order.
        waiting, sched = self._admit(5, graph_bs=(1, 2, 4, 8))
        self.assertEqual(schedule_batch.init_new.call_args.args[0], waiting[:3])
        self.assertEqual(sched.waiting_queue, waiting[3:])

    def test_full_graph_batch_feeds_nothing(self, schedule_batch, _):
        for running_size in (8, 9):
            waiting, sched = self._admit(running_size, graph_bs=(1, 2, 4, 8))
            self.assertEqual(sched.waiting_queue, waiting)
        schedule_batch.init_new.assert_not_called()

    def test_running_cap_below_the_graph_still_binds(self, schedule_batch, _):
        # max_running_requests 6 < graph bs 8: 5 running leaves 1 slot.
        waiting, _ = self._admit(5, graph_bs=(1, 2, 4, 8), max_running=6)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], waiting[:1])

    def test_no_decode_graphs_leaves_only_the_running_cap(self, schedule_batch, _):
        waiting, sched = self._admit(5, graph_bs=())
        self.assertEqual(schedule_batch.init_new.call_args.args[0], waiting)
        self.assertEqual(sched.waiting_queue, [])


if __name__ == "__main__":
    unittest.main()
