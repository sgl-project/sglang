"""CPU contracts of SGLang's post-verify graphs
(SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH): the switch alone turns them on for
every verify of a setup they can run, only batches a graph can reproduce take them,
a KV pool with c128 draft states never reaches them, and with them on the graphs and
the eager verify draw the forced acceptance lengths from one rank-identical stream;
without them the eager verify draws from the global CPU RNG as before."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import eagle_worker_common, eagle_worker_v2, spec_utils
from sglang.srt.speculative.eagle_post_verify_graphs import (
    EaglePostVerifyGraphs,
    simulated_acc_len_generator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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

        for clears_c128 in (False, True):
            kvcache = SimpleNamespace()
            if clears_c128:
                kvcache.clear_unaccepted_c128_draft_states = print
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
                    logits_output=SimpleNamespace(next_token_logits=torch.zeros(7, 8))
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
                patch.object(
                    eagle_worker_common, "eagle_sample", side_effect=_EagerSample
                ),
                self.assertRaises(_EagerSample),
            ):
                eagle_worker_common.run_eagle_verify(
                    batch,
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
        logits_output = SimpleNamespace(next_token_logits=torch.zeros((7, 8)))

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
            fields = dict(has_grammar=False, hisparse_coordinator=None)
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


if __name__ == "__main__":
    unittest.main()
