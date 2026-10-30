"""SGLang's post-verify graphs (SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH).

On CPU: the switch alone turns them on for every verify of a setup they can run,
only batches a graph can reproduce take them, a KV pool with c128 draft states never
reaches them, and with them on the graphs and the eager verify draw the forced
acceptance lengths from one rank-identical stream; without them the eager verify
draws from the global CPU RNG as before. On a GPU, at batch sizes up to 64:
every round's outputs, eager warm-ups and graph replays alike, must equal bitwise
what SGLang's eager sampling gives on the same inputs, greedy and with forced
acceptance, and stay unchanged through the next round; rounds that run eagerly in
between draw from the same forced-length generator. With the NVIDIA megakernel's
fused chain acceptance, eagle_sample's tail (greedy check, injected acceptance,
accept lengths) must leave every output bitwise as the unfused tail does, greedy,
compact top-k and dense sampled, at batch sizes up to 64, and the graphs must still
match eager sampling."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import (
    eagle_post_verify_graphs,
    eagle_utils,
    eagle_worker_common,
    eagle_worker_v2,
    spec_utils,
)
from sglang.srt.speculative.eagle_post_verify_graphs import (
    EaglePostVerifyGraphs,
    simulated_acc_len_generator,
)
from sglang.srt.speculative.eagle_utils import compact_verify_max_top_k, eagle_sample
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

TOKENS = 7
VOCAB = 257
BATCH_SIZES = (1, 3, 16, 64)
FUSED_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
# FlashInfer's top-k reads each logits row in 16-byte vectors
SAMPLED_VOCAB = 512
TOP_K = 20
# eager warm-ups, the capture of each slot, then replays of both slots
ROUNDS = 2 * eagle_post_verify_graphs._NUM_WARMUP + 4


def _target_worker():
    return SimpleNamespace(
        model_runner=SimpleNamespace(watermark_state=None),
        random_seed=11,
        device="cpu",
    )


def _worker(*, switch, cuda=True, topk=1, dp_attention=False):
    """An EAGLE worker after init_attention_backends, with
    SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH set to switch, on a CUDA host with a
    chain draft and no DP attention unless the arguments say otherwise."""
    worker = eagle_worker_v2.EAGLEWorkerV2.__new__(eagle_worker_v2.EAGLEWorkerV2)
    worker.__dict__.update(
        topk=topk,
        _target_worker=_target_worker(),
        _draft_worker=SimpleNamespace(init_attention_backends=lambda: None),
        # the rest of what verify hands run_eagle_verify
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        plan_stream=None,
        plan_stream_ctx=None,
        speculative_num_draft_tokens=7,
        device="cpu",
    )
    parallel = SimpleNamespace(attn_dp_enabled=dp_attention)
    with (
        envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(switch),
        patch.object(eagle_worker_v2, "_is_cuda", cuda),
        patch.object(eagle_worker_v2, "get_parallel", return_value=parallel),
    ):
        worker.init_attention_backends()
    return worker


def _verify_kwargs(worker):
    """What the worker's verify hands run_eagle_verify."""
    with patch.object(eagle_worker_v2, "run_eagle_verify") as run_eagle_verify:
        worker.verify(batch=None)
    return run_eagle_verify.call_args.kwargs


def _eager_length(generator):
    """The forced acceptance length SGLang's eager verify draws for one round."""
    correct = torch.zeros(1, dtype=torch.int32)
    spec_utils.generate_simulated_accept_index(
        accept_index=torch.zeros((1, 7), dtype=torch.int32),
        predict=torch.zeros(7, dtype=torch.int32),
        num_correct_drafts=correct,
        candidates=None,
        target_predict=None,
        bs=1,
        spec_steps=6,
        simulate_acc_len=4.8,
        simulate_acc_method="match-expected",
        simulate_acc_token_mode="fixed",
        generator=generator,
    )
    return int(correct) + 1


@contextmanager
def _graph_capture(stream):
    # one rank: nothing collective to set up for the capture
    yield SimpleNamespace(stream=stream)


def _round(bs, generator, vocab=VOCAB):
    """A chain verify round whose drafts follow the target's argmax for a random
    prefix of each request."""
    logits = torch.randn(bs * TOKENS, vocab, device="cuda", generator=generator)
    target = logits.argmax(-1).view(bs, TOKENS)
    drafts = torch.randint(0, vocab, (bs, TOKENS), device="cuda", generator=generator)
    keep = torch.randint(0, TOKENS, (bs, 1), device="cuda", generator=generator)
    cols = torch.arange(TOKENS - 1, device="cuda")
    drafts[:, 1:] = torch.where(cols < keep, target[:, :-1], drafts[:, 1:])
    next_token = torch.arange(1, TOKENS + 1, device="cuda")
    next_token[-1] = -1
    verify_input = SimpleNamespace(
        topk=1,
        tree_topk=1,
        draft_probs=None,
        draft_token=drafts.flatten(),
        draft_token_num=TOKENS,
        max_tree_depth=TOKENS,
        retrieve_index=torch.arange(bs * TOKENS, device="cuda").view(bs, TOKENS),
        retrieve_next_token=next_token.expand(bs, TOKENS).contiguous(),
        retrieve_next_sibling=torch.full((bs, TOKENS), -1, device="cuda"),
    )
    batch = SimpleNamespace(
        device="cuda",
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=torch.randint(16, 512, (bs,), device="cuda", generator=generator),
        req_pool_indices=torch.randperm(bs, device="cuda", generator=generator),
        sampling_info=SimpleNamespace(
            is_all_greedy=True,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            sampling_seed=None,
            has_watermark_candidates=False,
            temperatures=None,
            top_ks=None,
            top_ps=None,
        ),
        has_grammar=False,
        hisparse_coordinator=None,
        mamba_track_indices=None,
        tree_cache=SimpleNamespace(page_size=64),
        return_logprob=False,
    )
    return batch, verify_input, logits


def _eager(batch, verify_input, logits, sim_rng):
    predict, accept_lens, accept_index = eagle_sample(
        verify_input,
        batch,
        SimpleNamespace(
            next_token_logits=logits.clone(), next_token_logits_vocab_shard=None
        ),
        simulated_acc_len_rng=sim_rng,
    )
    new_seq_lens, bonus_tokens, _, _ = prepare_verify_commit_outputs(
        predict, accept_index, accept_lens, batch.seq_lens, num_draft_tokens=TOKENS
    )
    return predict, accept_lens, accept_index, new_seq_lens, bonus_tokens


class TestPostVerifyGraphs(CustomTestCase):
    def test_the_switch_turns_the_post_verify_graphs_on_for_every_verify(self):
        """Off, the default, the verify samples and commits eagerly as before; on,
        every verify hands its batch to SGLang's post-verify graphs. A setup they
        cannot run keeps them off, and the forced lengths on the global RNG: no CUDA,
        a tree draft, or DP attention, whose ranks would capture at different times."""
        off = _worker(switch=False)
        self.assertIsNone(off._post_verify_graphs)
        self.assertIsNone(_verify_kwargs(off)["post_verify_graphs"])
        on = _worker(switch=True)
        self.assertIsInstance(on._post_verify_graphs, EaglePostVerifyGraphs)
        self.assertIs(_verify_kwargs(on)["post_verify_graphs"], on._post_verify_graphs)
        for setup in (dict(cuda=False), dict(topk=2), dict(dp_attention=True)):
            with self.subTest(**setup):
                worker = _worker(switch=True, **setup)
                self.assertIsNone(worker._post_verify_graphs)
                self.assertIsNone(worker._simulated_acc_len_rng)

    def test_a_kv_pool_that_clears_c128_draft_states_keeps_the_eager_verify(self):
        """The post-verify graphs skip the eager verify's cleanup of rejected c128
        draft states (DeepSeek-V4), so a KV pool that has one never reaches them;
        any other pool hands the round to them."""

        class _EagerSample(Exception):
            pass

        class _EagerHooks:
            @property
            def post_verify(self):
                raise _EagerSample

        for clears_c128 in (False, True):
            kvcache = (
                object.__new__(DeepSeekV4TokenToKVPool)
                if clears_c128
                else SimpleNamespace()
            )
            graphs = SimpleNamespace(run=MagicMock(return_value=None))
            batch = SimpleNamespace(
                spec_info=SimpleNamespace(),
                seq_lens=torch.zeros(1),
                input_ids=None,
                out_cache_loc=None,
                has_grammar=False,
            )
            target_worker = SimpleNamespace(
                forward_batch_generation=lambda **_: SimpleNamespace(
                    logits_output=SimpleNamespace(
                        next_token_logits=torch.zeros(7, 8),
                        next_token_logits_vocab_shard=None,
                    )
                ),
                model_runner=SimpleNamespace(
                    watermark_state=None,
                    attn_backend=SimpleNamespace(verify_mask=None),
                ),
            )
            with (
                self.subTest(clears_c128=clears_c128),
                patch.object(eagle_worker_common, "record_stream_for_v2_verify"),
                patch.object(eagle_worker_common, "record_stream_each"),
                patch.object(
                    eagle_worker_common,
                    "eagle_prepare_for_verify",
                    return_value=(SimpleNamespace(), True),
                ),
                self.assertRaises(_EagerSample),
            ):
                eagle_worker_common.run_eagle_verify(
                    batch,
                    speculative_runtime_hooks=_EagerHooks(),
                    target_worker=target_worker,
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=SimpleNamespace(
                        get_kvcache=lambda: kvcache
                    ),
                    plan_stream=None,
                    plan_stream_ctx=nullcontext(),
                    topk=1,
                    num_draft_tokens=7,
                    device="cpu",
                    metadata_ready_pre_pad=False,
                    finalize_tree_path=True,
                    post_verify_graphs=graphs,
                )
            self.assertEqual(graphs.run.called, not clears_c128)

    def test_only_batches_a_graph_can_replay_take_the_post_verify_graphs(self):
        """A batch whose sampling a captured graph cannot reproduce (grammar, penalties,
        seeds, ...) or whose verify ran eagerly must take SGLang's eager path."""
        graphs = EaglePostVerifyGraphs(
            target_worker=_target_worker(),
            simulated_acc_len_rng=simulated_acc_len_generator(0),
        )
        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros((7, 8)),
            next_token_logits_vocab_shard=None,
        )

        def key(can_run_cuda_graph=True, topk=1, **changes):
            info = dict(
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
                sampling_seed=None,
                need_min_p_sampling=False,
                has_watermark_candidates=False,
                is_all_greedy=False,
                need_top_k_sampling=True,
                need_top_p_sampling=True,
            )
            fields = dict(
                has_grammar=False,
                hisparse_coordinator=None,
                return_logprob=False,
            )
            for name, value in changes.items():
                (info if name in info else fields)[name] = value
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                seq_lens=torch.zeros(1),
                sampling_info=SimpleNamespace(**info),
                mamba_track_indices=None,
                **fields,
            )
            verify_input = SimpleNamespace(
                topk=topk, draft_probs=None, draft_token_num=7, max_tree_depth=7
            )
            return graphs._key(batch, verify_input, logits_output, can_run_cuda_graph)

        self.assertIsNotNone(key())
        for change in (
            dict(can_run_cuda_graph=False),
            dict(topk=2),
            dict(has_grammar=True),
            dict(hisparse_coordinator=object()),
            dict(acc_additive_penalties=torch.zeros(1)),
            dict(acc_scaling_penalties=torch.zeros(1)),
            dict(logit_bias=torch.zeros(1)),
            dict(sampling_seed=torch.zeros(1)),
            dict(need_min_p_sampling=True),
        ):
            with self.subTest(change=sorted(change)):
                self.assertIsNone(key(**change))

    def test_forced_lengths_are_one_stream_with_the_graphs_and_unchanged_without(self):
        """With the post-verify graphs on, they and the eager verify draw each round's
        forced length from the worker's one generator, seeded from the server seed:
        every rank forces the same lengths, whichever path each round took and
        whatever else used its global RNG. Without them the eager verify draws from
        the global CPU RNG exactly as before."""

        def lengths(paths, global_seed):
            worker = _worker(switch=True)
            rng = worker._simulated_acc_len_rng
            self.assertIs(_verify_kwargs(worker)["simulated_acc_len_rng"], rng)
            graphs = worker._post_verify_graphs._graphs
            # the draw a graph round makes, without its pinned ring on a CPU host
            graphs._sim_host = [torch.zeros(1, dtype=torch.int64) for _ in range(4)]
            graphs._sim_done = [None] * 4
            torch.manual_seed(global_seed)  # this rank's global RNG
            drawn = []
            for path in paths:
                torch.rand(3)  # other uses of the global RNG
                if path == "graph":
                    drawn.append(int(graphs._draw_sim_len(7)[0][0]))
                else:
                    drawn.append(_eager_length(rng))
            return drawn

        with (
            patch.object(spec_utils, "SIMULATE_ACC_LEN", 4.8),
            patch.object(spec_utils, "SIMULATE_ACC_METHOD", "match-expected"),
        ):
            all_graph = lengths(["graph"] * 48, global_seed=1)
            all_eager = lengths(["eager"] * 48, global_seed=2)
            mixed = lengths(["graph", "eager", "eager"] * 16, global_seed=3)
        self.assertEqual(set(all_graph), {4, 5})
        self.assertEqual(all_eager, all_graph)
        self.assertEqual(mixed, all_graph)
        server_seeded = simulated_acc_len_generator(11)
        self.assertEqual(all_graph, [_eager_length(server_seeded) for _ in range(48)])

        self.assertIsNone(
            _verify_kwargs(_worker(switch=False))["simulated_acc_len_rng"]
        )
        torch.manual_seed(3)
        drawn = [_eager_length(None) for _ in range(32)]
        state = torch.get_rng_state()
        torch.manual_seed(3)
        weight = 4.8 - 4
        probs = torch.tensor([1.0 - weight, weight])
        before = [4 + int(torch.multinomial(probs, num_samples=1)) for _ in range(32)]
        self.assertEqual(drawn, before)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_compact_top_k_sampling_takes_only_batches_of_finite_top_k(self):
        """With SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING a verify samples on the
        top-k support when every request has a finite top-k within the compact
        limit, in the post-verify graphs (keyed apart from the dense branch, whose
        sampling differs) and the eager verify alike; without it nothing changes."""

        def batch(*top_ks):
            reqs = [
                SimpleNamespace(sampling_params=SimpleNamespace(top_k=k))
                for k in top_ks
            ]
            return SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY, reqs=reqs)

        def verify_kwargs(verify_batch):
            worker = _worker(switch=True)
            with patch.object(eagle_worker_v2, "run_eagle_verify") as run:
                worker.verify(batch=verify_batch)
            kwargs = run.call_args.kwargs
            return kwargs["uno_target_max_top_k"], kwargs["sparse_bonus_argmax"]

        limit = SimpleNamespace(_SPARSE_TOP_K_LIMIT=128)
        with patch.dict("sys.modules", {"sglang.srt.speculative.uno_utils": limit}):
            with patch.object(eagle_utils, "_COMPACT_VERIFY_SAMPLING", True):
                self.assertEqual(compact_verify_max_top_k(batch(20, 64)), 64)
                for dense in ((20, 1 << 30), (20, 1), (129,), ()):
                    with self.subTest(top_ks=dense):
                        self.assertIsNone(compact_verify_max_top_k(batch(*dense)))
                self.assertEqual(verify_kwargs(batch(20)), (20, True))
            self.assertIsNone(compact_verify_max_top_k(batch(20, 64)))
            self.assertEqual(verify_kwargs(batch(20)), (None, False))

        graphs = EaglePostVerifyGraphs(
            target_worker=_target_worker(),
            simulated_acc_len_rng=simulated_acc_len_generator(0),
        )
        verify_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.zeros(1),
            sampling_info=SimpleNamespace(
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
                sampling_seed=None,
                need_min_p_sampling=False,
                has_watermark_candidates=False,
                is_all_greedy=False,
                need_top_k_sampling=True,
                need_top_p_sampling=False,
            ),
            mamba_track_indices=None,
            has_grammar=False,
            hisparse_coordinator=None,
            return_logprob=False,
        )
        verify_input = SimpleNamespace(
            topk=1, draft_probs=None, draft_token_num=7, max_tree_depth=7
        )
        logits = SimpleNamespace(
            next_token_logits=torch.zeros((7, 8)),
            next_token_logits_vocab_shard=None,
        )
        keys = {
            graphs._key(verify_batch, verify_input, logits, True, max_top_k)
            for max_top_k in (20, None)
        }
        self.assertEqual(len(keys), 2)

    def test_a_packed_verify_result_is_one_buffer_holding_the_stock_values(self):
        """With SGLANG_Q35MF_EDGE_PACKED_VERIFY_BCAST a sampled verify result goes to
        the TP group in one broadcast: predict, accept_index and num_correct_drafts
        must be views of one buffer, in order, starting from the values the
        unpacked tensors start from."""
        kw = dict(rows=14, bs=2, depth=7, device="cpu")
        stock = eagle_utils._verify_outputs(packed=False, **kw)
        *views, buffer = eagle_utils._verify_outputs(packed=True, **kw)
        self.assertIsNone(stock[3])
        for got, want in zip(views[:2], stock[:2]):
            self.assertTrue(torch.equal(got, want))
        offset = 0
        for view in views:
            self.assertEqual(view.data_ptr(), buffer.data_ptr() + 4 * offset)
            offset += view.numel()
        self.assertEqual(offset, buffer.numel())

    def test_one_launch_load_stages_what_the_per_dtype_copies_stage(self):
        """With SGLANG_Q35MF_MKH_N3C a round's inputs and its forced length go to
        one copy plan; without it, or when the plan refuses the layout, the
        per-dtype copies stage the same values."""
        inputs = {
            "draft_tokens": torch.arange(14),
            "seq_lens": torch.tensor([17, 40]),
            "temperatures": torch.rand(2, 1),
        }
        sim_len = torch.tensor([3])
        for plan_mode in ("off", "accepts", "refuses"):
            calls = []

            def plan(dsts, srcs, accepts=plan_mode == "accepts"):
                calls.append(len(dsts))
                for dst, src in zip(dsts, srcs) if accepts else ():
                    dst.copy_(src)
                return accepts

            staged = {name: torch.zeros_like(src) for name, src in inputs.items()}
            staged["sim_accept_len"] = torch.zeros(1, dtype=torch.int64)
            slot = SimpleNamespace(
                staged=staged, load_plan=None if plan_mode == "off" else plan
            )
            eagle_post_verify_graphs._PostVerifyGraphs._load(slot, inputs, sim_len)
            with self.subTest(plan=plan_mode):
                self.assertEqual(calls, [] if plan_mode == "off" else [4])
                for name, src in inputs.items():
                    self.assertTrue(torch.equal(staged[name], src), name)
                self.assertEqual(int(staged["sim_accept_len"]), 3)

    def test_lengths_taken_from_a_slot_outlive_the_next_round(self):
        """Prepared lengths remain live for the next round; replaying their
        slot before that round finishes is rejected."""
        graphs = _prepared_length_graphs("cpu")
        with patch.object(torch.cuda, "stream", lambda stream: nullcontext()):
            _check_prepared_lengths(self, graphs, device="cpu")


def _prepared_length_graphs(device):
    """_PostVerifyGraphs whose body only moves the lengths (+3), and on a CPU host a capture
    that keeps the first outputs and rewrites them in place on each replay."""

    def body(slot):
        new_seq_lens = slot.staged["seq_lens"] + 3
        return (None, None, None, new_seq_lens, None, None, None)

    def commit(slot, outputs):
        # the side-stream graph needs some work of its own to capture
        slot.commit_ct.add_(1)

    graphs = eagle_post_verify_graphs._PostVerifyGraphs(
        body=body,
        commit=commit,
        sim_rng=None,
        protect_prepared_seq_lens=True,
    )
    if device == "cpu":

        def capture(slot):
            graphs._main_done = graphs._commit_done = MagicMock()
            graphs._side_stream = MagicMock()
            outputs = body(slot)

            def replay():
                outputs[3].copy_(body(slot)[3])

            slot.graph = SimpleNamespace(replay=replay)
            slot.commit_graph = SimpleNamespace(replay=lambda: None)
            slot.outputs = outputs

        graphs._capture = capture
    return graphs


def _check_prepared_lengths(test, graphs, *, device):
    def run(seq_lens):
        inputs = {
            "seq_lens": seq_lens,
            "req_pool_indices": torch.arange(2, device=device),
        }
        return graphs.run(
            key=0,
            inputs=inputs,
            make_slot=lambda staged: SimpleNamespace(
                num_requests=2,
                commit_ct=torch.zeros(1, dtype=torch.int64, device=device),
            ),
            sim_max_len=7,
        )

    with patch.object(spec_utils, "SIMULATE_ACC_LEN", -1.0):
        lengths = torch.tensor([100, 200], device=device)
        # eager warm-ups, then each slot's capture: every round takes the last one's lengths
        for _ in range(2 * eagle_post_verify_graphs._NUM_WARMUP + 2):
            lengths = run(lengths)[3]
        round_n = run(lengths)[3]
        want_n = round_n.clone()
        round_n1 = run(round_n)[3]
        test.assertTrue(
            torch.equal(round_n, want_n), "round N's lengths changed in N+1"
        )
        test.assertNotEqual(round_n1.data_ptr(), round_n.data_ptr())
        round_n2 = run(round_n1)[3]
        test.assertEqual(round_n2.data_ptr(), round_n.data_ptr())
        test.assertTrue(torch.equal(round_n2, want_n + 6))
        with test.assertRaisesRegex(RuntimeError, "holds this batch's lengths"):
            run(round_n1)  # round N+1's slot is next: a batch still on its lengths


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestPostVerifyGraphsMatchEager(CustomTestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def _check(self, *, sim_acc_len, fused=False):
        # the worker's forced lengths, drawn by its graphs and its eager rounds alike;
        # the all-eager reference draws the same stream from its own generator
        worker_rng = simulated_acc_len_generator(0)
        reference_rng = simulated_acc_len_generator(0)
        graphs = EaglePostVerifyGraphs(
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(watermark_state=None)
            ),
            simulated_acc_len_rng=worker_rng,
        )
        generator = torch.Generator(device="cuda").manual_seed(0)
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            patch.object(spec_utils, "SIMULATE_ACC_LEN", sim_acc_len),
            patch.object(spec_utils, "SIMULATE_ACC_METHOD", "match-expected"),
            patch.object(spec_utils, "SIMULATE_ACC_TOKEN_MODE", "real-draft-token"),
            patch.object(spec_utils, "commit_mamba_states_after_verify"),
            patch(
                "sglang.srt.distributed.parallel_state.graph_capture", _graph_capture
            ),
            patch.object(eagle_utils, "_fused_accept_enabled", return_value=fused),
        ):
            for bs in BATCH_SIZES:
                # the verify graph's logits buffer: one per batch size, refilled
                logits_buf = torch.empty(bs * TOKENS, VOCAB, device="cuda")
                previous = None
                # every third round runs eagerly, as when its verify ran without a graph
                for i in range(ROUNDS * 3 // 2):
                    batch, verify_input, logits = _round(bs, generator)
                    logits_buf.copy_(logits)
                    with patch.object(
                        eagle_utils, "_fused_accept_enabled", return_value=False
                    ):
                        want = _eager(batch, verify_input, logits, reference_rng)
                    graphed = i % 3 != 2
                    got = graphs.run(
                        batch=batch,
                        verify_input=verify_input,
                        logits_output=SimpleNamespace(
                            next_token_logits=logits_buf,
                            next_token_logits_vocab_shard=None,
                        ),
                        can_run_cuda_graph=graphed,
                    )
                    graphs.join()
                    if graphed:
                        self.assertIsNotNone(got)
                    else:
                        self.assertIsNone(got)
                        got = _eager(batch, verify_input, logits, worker_rng)
                    rows = want[2].clamp(min=0).long()
                    with self.subTest(bs=bs, round=i):
                        self.assertTrue(torch.equal(got[0][rows], want[0][rows]))
                        for name, g, w in zip(
                            ("accept_lens", "accept_index", "new_seq_lens", "bonus"),
                            got[1:5],
                            want[1:5],
                        ):
                            self.assertTrue(torch.equal(g, w), name)
                        # the host may still read a round's outputs in the next round
                        if previous is not None:
                            for kept, snapshot in zip(*previous):
                                self.assertTrue(torch.equal(kept, snapshot))
                    previous = (got[:5], [t.clone() for t in got[:5]])

    def test_greedy(self):
        self._check(sim_acc_len=-1.0)

    def test_forced_acceptance(self):
        self._check(sim_acc_len=4.8)

    def test_forced_acceptance_with_one_launch_load(self):
        # SGLANG_Q35MF_MKH_N3C: inputs and the pinned forced length in one launch
        with patch.object(eagle_post_verify_graphs, "_ONE_LAUNCH_LOAD", True):
            self._check(sim_acc_len=4.8)

    def test_fused_accept_greedy(self):
        self._check(sim_acc_len=-1.0, fused=True)

    def test_fused_accept_forced_acceptance(self):
        self._check(sim_acc_len=4.8, fused=True)

    def test_lengths_taken_from_a_slot_outlive_the_next_round(self):
        """As the CPU test, with captured CUDA graphs replaying into the slots' outputs."""
        graphs = _prepared_length_graphs("cuda")
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            patch(
                "sglang.srt.distributed.parallel_state.graph_capture", _graph_capture
            ),
        ):
            _check_prepared_lengths(self, graphs, device="cuda")

    def test_the_bonus_in_int64_is_the_int32_bonus_cast(self):
        """The prepared path writes the bonus in its consumer's int64 dtype
        without changing any value from the regular int32 output."""
        generator = torch.Generator(device="cuda").manual_seed(2)
        for bs in BATCH_SIZES:
            batch, verify_input, logits = _round(bs, generator)
            predict, accept_lens, accept_index, _, _ = _eager(
                batch, verify_input, logits, simulated_acc_len_generator(0)
            )
            outputs = {
                dtype: prepare_verify_commit_outputs(
                    predict,
                    accept_index,
                    accept_lens,
                    batch.seq_lens,
                    num_draft_tokens=TOKENS,
                    bonus_dtype=dtype,
                )
                for dtype in (torch.int32, torch.int64)
            }
            with self.subTest(bs=bs):
                narrow, wide = outputs[torch.int32][1], outputs[torch.int64][1]
                self.assertEqual(wide.dtype, torch.int64)
                self.assertTrue(torch.equal(wide, narrow.to(torch.int64)))
                self.assertIs(wide.to(torch.int64), wide)
                for a, b in zip(outputs[torch.int32][0:1], outputs[torch.int64][0:1]):
                    self.assertTrue(torch.equal(a, b))


def _sampled(batch, *, generator):
    """The round's batch with T0.6 / top-k / top-p sampling on every request."""
    bs = len(batch.seq_lens)
    info = SimpleNamespace(**vars(batch.sampling_info))
    info.is_all_greedy = False
    info.need_top_k_sampling = True
    info.need_top_p_sampling = True
    info.temperatures = torch.full((bs, 1), 0.6, device="cuda")
    info.top_ks = torch.randint(2, TOP_K + 1, (bs,), device="cuda", generator=generator)
    info.top_ps = torch.full((bs,), 0.95, device="cuda")
    return SimpleNamespace(**{**vars(batch), "sampling_info": info})


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFusedAcceptMatchesUnfused(CustomTestCase):
    """Every output of the NVIDIA megakernel's fused chain acceptance and of the
    commit outputs built on it (predict at every node, accept lengths and index,
    sequence lengths, bonus, draft-extend inputs, commit steps) equals the unfused
    tail's."""

    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def _run(self, batch, verify_input, logits, *, fused, max_top_k, sim_len, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with patch.object(eagle_utils, "_fused_accept_enabled", return_value=fused):
            predict, accept_lens, accept_index = eagle_sample(
                verify_input,
                batch,
                SimpleNamespace(
                    next_token_logits=logits.clone(),
                    next_token_logits_vocab_shard=None,
                ),
                uno_target_max_top_k=max_top_k,
                simulated_acc_len=sim_len,
                sparse_bonus_argmax=max_top_k is not None,
            )
        commit = prepare_verify_commit_outputs(
            predict,
            accept_index,
            accept_lens,
            batch.seq_lens,
            num_draft_tokens=TOKENS,
            mamba_track_interval=64,
        )
        new_seq_lens, bonus, (last_steps, track_steps), extend_inputs = commit
        return dict(
            predict=predict,
            accept_lens=accept_lens,
            accept_index=accept_index,
            new_seq_lens=new_seq_lens,
            bonus=bonus,
            last_steps=last_steps,
            track_steps=track_steps,
            correct=extend_inputs[0],
            select=extend_inputs[1],
            ids=extend_inputs[2],
        )

    def _compare(self, *, sim_acc_len, token_mode, packed):
        generator = torch.Generator(device="cuda").manual_seed(1)
        parallel = SimpleNamespace(
            tp_group=SimpleNamespace(world_size=1),
            attn_tp_group=SimpleNamespace(world_size=1),
        )
        with (
            patch.object(spec_utils, "SIMULATE_ACC_LEN", sim_acc_len),
            patch.object(spec_utils, "SIMULATE_ACC_TOKEN_MODE", token_mode),
            patch.object(eagle_utils, "_PACKED_VERIFY_BCAST", packed),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        ):
            for bs in FUSED_BATCH_SIZES:
                for sampling in ("greedy", "compact", "dense"):
                    for length in (1, 3, TOKENS) if sim_acc_len > 0 else (None,):
                        batch, verify_input, logits = _round(
                            bs, generator, vocab=SAMPLED_VOCAB
                        )
                        if sampling != "greedy":
                            batch = _sampled(batch, generator=generator)
                        max_top_k = TOP_K if sampling == "compact" else None
                        sim_len = (
                            None
                            if length is None
                            else torch.tensor([length], device="cuda")
                        )
                        kw = dict(max_top_k=max_top_k, sim_len=sim_len, seed=bs)
                        want = self._run(batch, verify_input, logits, fused=False, **kw)
                        got = self._run(batch, verify_input, logits, fused=True, **kw)
                        with self.subTest(bs=bs, sampling=sampling, length=length):
                            for name, w in want.items():
                                g = got[name]
                                if w is None:
                                    self.assertIsNone(g, name)
                                    continue
                                self.assertEqual(g.dtype, w.dtype, name)
                                self.assertTrue(torch.equal(g, w), name)

    def test_greedy_check_and_sampled_targets(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                self._compare(sim_acc_len=-1.0, token_mode="fixed", packed=packed)

    def test_forced_acceptance_of_real_draft_tokens(self):
        for packed in (False, True):
            with self.subTest(packed=packed):
                self._compare(
                    sim_acc_len=4.8, token_mode="real-draft-token", packed=packed
                )

    def test_forced_acceptance_of_fixed_tokens(self):
        self._compare(sim_acc_len=4.8, token_mode="fixed", packed=True)


if __name__ == "__main__":
    unittest.main()
