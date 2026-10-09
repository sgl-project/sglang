import gc
import sys
import unittest
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace as NS
from unittest import TestCase, skipUnless
from unittest.mock import DEFAULT, Mock, patch

import torch

from sglang.srt.sampling.verify_graph import VerifySamplingBuffers
from sglang.srt.sampling.verify_probs import build_verify_target_probs
from sglang.srt.speculative import eagle_verify_graph
from sglang.srt.speculative.eagle_verify_graph import (
    EagleVerifyEpilogue,
    install_eagle_verify_epilogue,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class ProposalProbabilityTest(TestCase):
    def test_draft_prob_snapshot_covers_padded_and_exact_batches(self):
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )

        for raw_bs in (2, 4):
            probs = torch.full((4, 3, 8), 0.125)
            owners = (object(), object(), object())
            out = EAGLEDraftCudaGraphRunner._snapshot_draft_probs(
                (*owners, probs), raw_bs
            )
            self.assertEqual(out[:3], owners)
            self.assertEqual(out[3].shape, (raw_bs, 3, 8))
            probs.zero_()
            torch.testing.assert_close(out[3], torch.full_like(out[3], 0.125))
        self.assertIsNone(
            EAGLEDraftCudaGraphRunner._snapshot_draft_probs((*owners, None), 4)[3]
        )

    def test_rejection_proposal_requires_full_shape(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative import eagle_utils

        batch = NS(
            device="cpu",
            forward_mode=ForwardMode.TARGET_VERIFY,
            seq_lens=torch.tensor([10, 20]),
            sampling_info=NS(
                acc_additive_penalties=None,
                acc_scaling_penalties=None,
                logit_bias=None,
                is_all_greedy=False,
            ),
        )
        verify = NS(
            draft_token=torch.zeros(6, dtype=torch.long),
            draft_token_num=3,
            max_tree_depth=3,
        )
        logits = NS(next_token_logits=torch.zeros(6, 8))
        spec = NS(
            speculative_use_rejection_sampling=True,
            speculative_use_block_verification=False,
        )
        with (
            patch.multiple(
                eagle_utils, _is_cpu=False, _is_hip=False, _is_xpu=False, _is_npu=False
            ),
            patch.dict(
                sys.modules,
                {"sgl_kernel": NS(top_k_renorm_prob=Mock(), top_p_renorm_prob=Mock())},
            ),
            patch.object(
                eagle_utils,
                "get_spec",
                return_value=spec,
            ),
            patch("sglang.srt.utils.async_probe.sanitize_nan_logits"),
            patch.object(
                eagle_utils,
                "build_verify_target_probs",
                side_effect=lambda **kwargs: torch.softmax(
                    kwargs["next_token_logits"], dim=-1
                ).view(2, 3, 8),
            ),
            patch.object(
                eagle_utils,
                "_can_use_sparse_uno_tree_target_sampling",
                return_value=False,
            ),
            patch.object(
                eagle_utils, "_verify_coins", side_effect=RuntimeError("shape accepted")
            ) as sample,
        ):
            for shape in (None, (1, 2, 8), (2, 1, 8), (2, 2, 7), (2, 2, 8)):
                with self.subTest(shape=shape):
                    verify.draft_probs = None if shape is None else torch.zeros(shape)
                    error, message = (
                        (RuntimeError, "shape accepted")
                        if shape == (2, 2, 8)
                        else (ValueError, "distribution with shape")
                    )
                    with self.assertRaisesRegex(error, message):
                        eagle_utils.eagle_sample(
                            verify,
                            batch,
                            logits,
                        )
            self.assertEqual(sample.call_count, 1)


class InstallationTest(TestCase):
    def test_shared_staging_preserves_storage_and_resets_padding(self):
        buffers = VerifySamplingBuffers(4, 3, 8, "cpu", with_draft=True)
        pointers = {
            name: value.data_ptr()
            for name, value in vars(buffers).items()
            if isinstance(value, torch.Tensor)
        }
        for bs in (4, 1, 3, 2):
            info = NS(
                temperatures=torch.full((bs, 1), 0.7),
                top_ks=torch.tensor([1] + [7] * (bs - 1)),
                top_ps=torch.full((bs,), 0.8),
            )
            buffers.stage(info, bs)
            view = buffers.for_batch(bs, is_all_greedy=False)
            torch.testing.assert_close(view.temperatures, info.temperatures)
            self.assertEqual(
                buffers.greedy_mask[:bs].tolist(), [True] + [False] * (bs - 1)
            )
            self.assertEqual(buffers.top_ks[bs:].tolist(), [1] * (4 - bs))
            self.assertTrue(buffers.greedy_mask[bs:].all())
            self.assertEqual(
                pointers, {name: getattr(buffers, name).data_ptr() for name in pointers}
            )
        with self.assertRaisesRegex(ValueError, "one value per request"):
            buffers.stage(info, 1)

    def test_proposal_staging_rejects_broadcastable_shapes(self):
        epilogue = EagleVerifyEpilogue.__new__(EagleVerifyEpilogue)
        epilogue.width, epilogue.vocab_size = 3, 8
        epilogue.sampling = True
        epilogue.info = VerifySamplingBuffers(4, 3, 8, "cpu", with_draft=True)
        epilogue.armed = torch.zeros(4, dtype=torch.int32)
        batch = NS(reqs=[NS(), NS()], has_grammar=False)
        # Invalid proposals must not broadcast over a missing batch or draft axis.
        for shape in ((1, 2, 8), (2, 1, 8), (2, 2, 7)):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "Expected draft proposal shape"),
            ):
                epilogue.arm(batch, None, torch.ones(shape))
        self.assertFalse(epilogue.armed.any())
        self.assertFalse(epilogue.info.draft_distribution.any())
        with self.assertRaisesRegex(RuntimeError, "requires a draft distribution"):
            epilogue.arm(batch, None)
        probs = torch.full((2, 2, 8), 0.125)
        epilogue.arm(batch, None, probs)
        probs.zero_()
        torch.testing.assert_close(
            epilogue.info.draft_distribution[:2], torch.full((2, 2, 8), 0.125)
        )
        self.assertEqual(epilogue.armed.tolist(), [1, 1, 0, 0])

    def test_verify_dispatch_commits_once(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative import eagle_worker_common as common

        results = (
            torch.arange(6, dtype=torch.int32),
            torch.tensor([3]),
            torch.tensor([[0, 1, 2, -1, -1, -1]]),
        )
        batch = NS(
            spec_info=NS(draft_probs=None),
            seq_lens=torch.tensor([10]),
            input_ids=None,
            out_cache_loc=None,
            has_grammar=False,
            return_logprob=False,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        output = NS(
            logits_output=NS(next_token_logits=None),
            routed_experts_output=None,
            indexer_topk_output=None,
        )
        for installed, eligible, captured in (
            (False, False, True),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        ):
            epilogue = Mock(
                prepare=Mock(return_value=eligible),
                read=Mock(
                    return_value=(*results, torch.tensor([2]), torch.tensor([13]))
                ),
            )
            target = NS(
                model_runner=NS(spec_verify_epilogue=epilogue if installed else None),
                forward_batch_generation=Mock(return_value=output),
            )
            with patch.multiple(
                common,
                record_stream_for_v2_verify=DEFAULT,
                record_stream_each=DEFAULT,
                eagle_prepare_for_verify=DEFAULT,
                eagle_sample=DEFAULT,
                commit_mamba_states_after_verify=DEFAULT,
                fill_bonus_tokens_func=DEFAULT,
            ) as mocks:
                mocks["eagle_prepare_for_verify"].return_value = (NS(), captured)
                mocks["eagle_sample"].return_value = results
                mocks["fill_bonus_tokens_func"].side_effect = (
                    lambda tokens, lens, bonus, *args: bonus.fill_(2)
                )
                result = common.run_eagle_verify(
                    batch,
                    target_worker=target,
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=NS(get_kvcache=lambda: NS()),
                    plan_stream=None,
                    plan_stream_ctx=nullcontext(),
                    topk=1,
                    num_draft_tokens=6,
                    device="cpu",
                    metadata_ready_pre_pad=False,
                    finalize_tree_path=True,
                )
                folded = installed and eligible and captured
                self.assertEqual(epilogue.arm.call_count, int(folded))
                self.assertEqual(epilogue.read.call_count, int(folded))
                self.assertEqual(mocks["eagle_sample"].call_count, int(not folded))
                self.assertEqual(
                    mocks["commit_mamba_states_after_verify"].call_count,
                    int(not folded),
                )
                self.assertEqual(
                    mocks["fill_bonus_tokens_func"].call_count, int(not folded)
                )
                self.assertEqual(result.new_seq_lens.item(), 13)
                self.assertEqual(result.next_draft_input.bonus_tokens.item(), 2)

    def test_static_chain_and_state_commit_capabilities(self):
        from sglang.srt.mem_cache.memory_pool import MambaPool

        pool = MambaPool.__new__(MambaPool)
        pool.replayssm_spec_fold = pool.replayssm_is_kda = True
        target = NS(
            model_runner=NS(
                device="cuda",
                req_to_token_pool=NS(mamba_pool=pool),
                capture_tail_hooks=[],
                spec_verify_epilogue=None,
                token_to_kv_pool=NS(),
                model_config=NS(),
            )
        )
        args = NS(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            speculative_num_draft_tokens=6,
            speculative_adaptive=False,
            speculative_use_rejection_sampling=False,
            enable_linear_replayssm_spec=True,
            disable_cuda_graph=False,
            enable_dp_attention=False,
            enable_two_batch_overlap=False,
            enable_pdmux=False,
            pp_size=1,
        )
        config = NS(decode=NS(backend="full", bs=[1, 2, 4]))
        with (
            patch(
                "sglang.srt.speculative.eagle_verify_graph.mambaish_config",
                return_value=NS(),
            ) as recurrent_config,
            patch(
                "sglang.srt.speculative.eagle_verify_graph.resolved_view",
                return_value=args,
            ),
            patch(
                "sglang.srt.speculative.eagle_verify_graph.get_exec",
                return_value=NS(graph=NS(cuda_graph_config=config)),
            ),
            patch(
                "sglang.srt.speculative.eagle_verify_graph.EagleVerifyEpilogue"
            ) as ctor,
        ):
            for owner, field, value in (
                (args, "speculative_eagle_topk", 2),
                (args, "speculative_num_draft_tokens", 5),
                (args, "speculative_adaptive", True),
                (args, "enable_linear_replayssm_spec", False),
                (args, "enable_dp_attention", True),
                (args, "enable_two_batch_overlap", True),
                (args, "enable_pdmux", True),
                (args, "pp_size", 2),
                (config.decode, "bs", []),
                (config.decode, "backend", "breakable"),
                (pool, "replayssm_is_kda", False),
                (target.model_runner, "device", "cpu"),
                (target.model_runner.req_to_token_pool, "mamba_pool", NS()),
                (
                    target.model_runner,
                    "token_to_kv_pool",
                    NS(clear_unaccepted_c128_draft_states=Mock()),
                ),
            ):
                before = getattr(owner, field)
                setattr(owner, field, value)
                install_eagle_verify_epilogue(target, args)
                ctor.assert_not_called()
                setattr(owner, field, before)
            install_eagle_verify_epilogue(target, args)
            self.assertEqual(ctor.call_args.args, (target, 6, 4))
            self.assertFalse(ctor.call_args.kwargs["rejection_sampling"])
            self.assertIs(
                ctor.call_args.kwargs["commit"].func,
                eagle_verify_graph.commit_mamba_states_after_verify,
            )
            args.speculative_use_rejection_sampling = True
            ctor.reset_mock()
            target.model_runner.capture_tail_hooks.clear()
            install_eagle_verify_epilogue(target, args)
            self.assertTrue(ctor.call_args.kwargs["rejection_sampling"])
            args.speculative_use_rejection_sampling = False
            self.assertIs(target.model_runner.spec_verify_epilogue, ctor.return_value)
            self.assertEqual(
                target.model_runner.capture_tail_hooks, [ctor.return_value.capture_hook]
            )
            # Ordinary full-attention models require no recurrent-state commit.
            from transformers import LlamaConfig

            from sglang.srt.configs.hybrid_arch import mambaish_config

            target.model_runner.model_config = NS(
                hf_config=LlamaConfig(), linear_attn_registry_result=None
            )
            recurrent_config.side_effect = mambaish_config
            args.enable_linear_replayssm_spec = False
            target.model_runner.req_to_token_pool = NS()
            target.model_runner.capture_tail_hooks.clear()
            ctor.reset_mock()
            install_eagle_verify_epilogue(target, args)
            ctor.assert_called_once_with(
                target, 6, 4, rejection_sampling=False, commit=None
            )


@skipUnless(torch.cuda.is_available(), "requires CUDA")
class EagleVerifyGraphTest(TestCase):
    def test_kda_commit_snapshots_and_disabled_replay(self):
        from sglang.srt.configs import KimiLinearConfig
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_context
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        override = get_context().override_server_args(
            mamba_track_interval=64, _mamba_cache_chunk_size=64
        )
        override.install()
        self.addCleanup(override.restore)
        width, layers, slots, heads, dim, cap = 6, 2, 10, 2, 32, 4
        state = NS(
            temporal=torch.randn(layers, slots, heads, dim, dim, device="cuda"),
            replayssm_rawv=torch.randn(layers, slots, heads, width, dim, device="cuda"),
            replayssm_rawk=torch.randn(layers, slots, heads, width, dim, device="cuda"),
            replayssm_g=-torch.rand(layers, slots, heads, width, dim, device="cuda"),
            replayssm_beta=torch.rand(layers, slots, heads, width, device="cuda"),
            conv=[torch.randn(layers, slots, dim, 3, device="cuda")],
            intermediate_conv_window=[
                torch.randn(layers, slots, width, dim, 3, device="cuda")
            ],
        )
        mapping = torch.arange(slots, device="cuda")
        pool = NS(
            mamba_pool=NS(replayssm_spec_fold=True, replayssm_is_kda=True),
            get_mamba_indices=lambda ids: mapping[ids],
            get_speculative_mamba2_params_all_layers=lambda: state,
        )
        logits = torch.zeros(cap * width, 129, device="cuda")
        target = NS(
            model_runner=NS(
                req_to_token_pool=pool,
                device="cuda",
                page_size=1,
                model=NS(forward=lambda *args: NS(next_token_logits=logits.clone())),
                model_config=NS(hf_config=KimiLinearConfig(), vocab_size=129),
            )
        )
        epilogue = EagleVerifyEpilogue(
            target, width, cap, commit=partial(commit_mamba_states_after_verify, target)
        )
        batch = NS(
            input_ids=torch.arange(width, device="cuda").repeat(cap),
            batch_size=cap,
            seq_lens=torch.full((cap,), 63, device="cuda"),
            req_pool_indices=torch.arange(1, cap + 1, device="cuda"),
            mamba_track_indices=torch.arange(5, 5 + cap, device="cuda"),
            tree_cache=NS(page_size=1),
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        runner = NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=False)

        def run():
            epilogue.capture_hook(
                runner, NS(next_token_logits=logits.clone()), batch, cap * width
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        initial = state.temporal.clone(), state.conv[0].clone()
        for seq in (57, 59, 63, 64):
            for accepted in range(1, width + 1):
                with self.subTest(seq=seq, accepted=accepted):
                    bs = 1 + accepted % cap
                    batch.seq_lens.copy_(seq + torch.arange(cap, device="cuda"))
                    batch.req_pool_indices.copy_(
                        torch.arange(1, cap + 1, device="cuda")
                    )
                    # Padding deliberately aliases a live state slot; it must not write.
                    batch.req_pool_indices[bs:].fill_(1)
                    accepted_per_req = [
                        1 + (accepted + row - 1) % width for row in range(cap)
                    ]
                    logits.fill_(-100)
                    logits[:, 100] = 100
                    for row, count in enumerate(accepted_per_req):
                        for i in range(count - 1):
                            logits[row * width + i, 100] = -100
                            logits[row * width + i, i + 1] = 100
                    state.temporal.copy_(initial[0])
                    state.conv[0].copy_(initial[1])
                    epilogue.armed.zero_()
                    epilogue.armed[:bs].fill_(1)
                    graph.replay()
                    actual = state.temporal.clone(), state.conv[0].clone()
                    self.assertEqual(
                        epilogue.accept_lens[:bs].tolist(), accepted_per_req[:bs]
                    )
                    state.temporal.copy_(initial[0])
                    state.conv[0].copy_(initial[1])
                    live = NS(
                        forward_mode=batch.forward_mode,
                        seq_lens=batch.seq_lens[:bs],
                        req_pool_indices=batch.req_pool_indices[:bs],
                        mamba_track_indices=batch.mamba_track_indices[:bs],
                        tree_cache=batch.tree_cache,
                    )
                    commit_mamba_states_after_verify(
                        target,
                        live,
                        epilogue.accept_lens[:bs],
                        epilogue.accept_index[:bs],
                        width,
                    )
                    for got, expected in zip(
                        actual, (state.temporal, state.conv[0]), strict=True
                    ):
                        torch.testing.assert_close(got, expected, atol=0, rtol=0)
                    for row, count in enumerate(accepted_per_req[:bs]):
                        if (seq + row) // 64 == (seq + row + count) // 64:
                            torch.testing.assert_close(
                                actual[0][:, 5 + row], initial[0][:, 5 + row]
                            )
                            torch.testing.assert_close(
                                actual[1][:, 5 + row], initial[1][:, 5 + row]
                            )
                    epilogue.armed.zero_()
                    graph.replay()
                    torch.testing.assert_close(
                        state.temporal, actual[0], atol=0, rtol=0
                    )
                    torch.testing.assert_close(state.conv[0], actual[1], atol=0, rtol=0)

    def test_sparse_distribution(self):
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

        torch.manual_seed(123)
        for vocab in (129, 163840):
            logits = torch.randn(6, vocab, device="cuda")
            # Cutoff ties overflow the sparse buffer, including a uniform row.
            logits[1].zero_()
            logits[2, :100] = 10
            logits[3].fill_(-float("inf"))
            logits[3, 7] = 0
            for k in (1, 7, 50, 64):
                for p in (0.1, 0.95, 1.0):
                    with self.subTest(vocab=vocab, k=k, p=p):
                        info = NS(
                            temperatures=torch.tensor([[0.7]], device="cuda"),
                            top_ks=torch.tensor([k], dtype=torch.int32, device="cuda"),
                            top_ps=torch.tensor([p], device="cuda"),
                            need_top_k_sampling=True,
                            need_top_p_sampling=True,
                        )
                        dense = torch.softmax(logits / 0.7, dim=-1)
                        dense = top_k_renorm_prob(dense, info.top_ks.repeat(6))
                        dense = top_p_renorm_prob(dense, info.top_ps.repeat(6))
                        actual = build_verify_target_probs(
                            next_token_logits=logits,
                            sampling_info=info,
                            draft_token_num=6,
                            bs=1,
                            max_top_k=64,
                            sparse_top_k_mode="threshold",
                        )[0]
                        torch.testing.assert_close(actual, dense, atol=2e-6, rtol=2e-5)
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(
                            actual.sum(-1), torch.ones(6, device="cuda")
                        )

    def _sampling_epilogue(self, logits, width, max_bs=1, rejection=False):
        from sglang.srt.runtime_context import get_context, get_parallel

        self.enterContext(get_parallel().override(tp_group=NS(world_size=1)))
        override = get_context().override_server_args(
            speculative_use_rejection_sampling=rejection,
        )
        override.install()
        self.addCleanup(override.restore)
        model = NS(
            forward=lambda ids, *args: NS(
                next_token_logits=logits[: ids.numel()].clone()
            )
        )
        target = NS(
            model_runner=NS(
                model=model,
                device="cuda",
                model_config=NS(vocab_size=logits.shape[1]),
                page_size=1,
            )
        )
        return EagleVerifyEpilogue(
            target,
            width,
            max_bs,
            rejection_sampling=rejection,
            commit=lambda *args: eagle_verify_graph.commit_mamba_states_after_verify(
                target, *args
            ),
        )

    def _capture_sampling_graphs(self, epilogue, batch, capacities):
        runner = NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=False)
        graphs = {}
        self.captured_logits = {}
        for cap in capacities:
            cb = NS(
                input_ids=batch.input_ids[: cap * epilogue.width],
                positions=batch.positions[: cap * epilogue.width],
                seq_lens=batch.seq_lens[:cap],
                req_pool_indices=batch.req_pool_indices[:cap],
                mamba_track_indices=(
                    batch.mamba_track_indices[:cap]
                    if batch.mamba_track_indices is not None
                    else None
                ),
                forward_mode=batch.forward_mode,
                batch_size=cap,
            )

            def run(cb=cb, cap=cap):
                out = epilogue.target_worker.model_runner.model.forward(
                    cb.input_ids, cb.positions, cb
                )
                epilogue.capture_hook(runner, out, cb, cap * epilogue.width)
                self.captured_logits[
                    cap, epilogue.sampling_top_k if epilogue.sampling else 0
                ] = out.next_token_logits

            for limit in (0, 64, epilogue.vocab_size):
                epilogue.sampling = limit > 0
                epilogue.sampling_top_k = limit
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graphs[cap, limit] = graph
        return graphs

    def test_sampling_graph_replay_and_fallback(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.eagle_utils import eagle_sample

        width, vocab = 6, 129
        logits = torch.randn(width, vocab, device="cuda")
        epilogue = self._sampling_epilogue(logits, width)
        batch = NS(
            input_ids=torch.arange(width, device="cuda"),
            positions=torch.arange(width, device="cuda"),
            seq_lens=torch.tensor([63], device="cuda"),
            req_pool_indices=torch.tensor([1], device="cuda"),
            mamba_track_indices=torch.tensor([2], device="cuda"),
            batch_size=1,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        info = NS(
            temperatures=torch.ones((1, 1), device="cuda"),
            top_ks=torch.tensor([50], dtype=torch.int32, device="cuda"),
            top_ps=torch.tensor([0.95], device="cuda"),
            is_all_greedy=False,
            sampling_seed=None,
            has_custom_logit_processor=False,
        )
        request = NS(sampling_params=NS(top_k=50))
        live = NS(
            **vars(batch),
            sampling_info=info,
            reqs=[request],
            return_logprob=False,
            has_grammar=False,
        )
        self.assertTrue(epilogue.prepare(live))
        epilogue.arm(live, None)
        committed = torch.zeros(1, device="cuda", dtype=torch.int32)

        def commit(target, b, lengths, indices, width):
            committed.add_(lengths)

        with (
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=NS(world_size=1),
            ),
            patch(
                "sglang.srt.speculative.eagle_verify_graph.commit_mamba_states_after_verify",
                side_effect=commit,
            ),
        ):
            graphs = self._capture_sampling_graphs(epilogue, batch, (1,))
            for k, p in ((50, 0.95), (7, 0.6), (1, 1.0), (vocab, 0.9), (64, 1.0)):
                live.return_logprob = k == 7
                enabled = k in (7, 64)
                info.sampling_seed = (
                    torch.tensor([12345], device="cuda") if enabled else None
                )
                info.need_min_p_sampling = enabled
                info.min_ps = torch.tensor([0.3], device="cuda")
                info.acc_additive_penalties = (
                    torch.randn(1, vocab, device="cuda") if enabled else None
                )
                info.acc_scaling_penalties = (
                    torch.full((1, vocab), 1.2, device="cuda") if enabled else None
                )
                info.logit_bias = (
                    torch.randn(1, vocab, device="cuda") if enabled else None
                )
                info.need_top_k_sampling = info.need_top_p_sampling = True
                request.sampling_params.top_k = k
                info.top_ks.fill_(k)
                info.top_ps.fill_(p)
                info.is_all_greedy = k == 1
                epilogue.prepare(live)
                epilogue.arm(live, None)
                graph = graphs[1, epilogue.sampling_top_k if epilogue.sampling else 0]
                epilogue.info.is_all_greedy = info.is_all_greedy
                logits.normal_()
                candidates = NS(
                    draft_token=batch.input_ids,
                    draft_token_num=width,
                    max_tree_depth=width,
                    tree_topk=1,
                    retrieve_index=epilogue.retrieve_index,
                    retrieve_next_token=epilogue.retrieve_next_token,
                    retrieve_next_sibling=epilogue.retrieve_next_sibling,
                )
                reference_batch = NS(
                    device="cuda",
                    forward_mode=batch.forward_mode,
                    seq_lens=batch.seq_lens,
                    sampling_info=info,
                )
                torch.cuda.manual_seed(45)
                graph.replay()
                actual = epilogue.read(live)
                torch.cuda.manual_seed(45)
                reference_output = NS(next_token_logits=logits.clone())
                reference = eagle_sample(candidates, reference_batch, reference_output)
                if live.return_logprob:
                    from sglang.srt.layers.logprob_processor import (
                        compute_spec_logprobs,
                    )

                    reference_batch.top_logprobs_nums = [3]
                    reference_batch.token_ids_logprobs = [[0, 1]]
                    graph_output = NS(
                        next_token_logits=self.captured_logits[
                            1, epilogue.sampling_top_k
                        ].clone()
                    )
                    compute_spec_logprobs(
                        reference_batch,
                        graph_output,
                        actual.predict,
                        accept_index=actual.accept_index,
                    )
                    compute_spec_logprobs(
                        reference_batch,
                        reference_output,
                        reference[0],
                        accept_index=reference[2],
                    )
                    for name in (
                        "next_token_logprobs",
                        "next_token_top_logprobs_val",
                        "next_token_top_logprobs_idx",
                        "next_token_token_ids_logprobs_val",
                        "next_token_token_ids_logprobs_idx",
                    ):
                        torch.testing.assert_close(
                            getattr(graph_output, name), getattr(reference_output, name)
                        )
                for got, expected in zip(actual[:3], reference, strict=True):
                    torch.testing.assert_close(got, expected, atol=0, rtol=0)
                saved = [t.clone() for t in actual]
                graph.replay()
                for got, expected in zip(actual, saved, strict=True):
                    torch.testing.assert_close(got, expected)

            info.sampling_seed = None
            info.need_min_p_sampling = False
            info.acc_additive_penalties = info.acc_scaling_penalties = (
                info.logit_bias
            ) = None
            import xgrammar as xg

            from sglang.kernels.ops.grammar.bitmask_ops import (
                apply_token_bitmask_inplace_triton,
            )
            from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
            from sglang.srt.speculative.spec_utils import generate_token_bitmask

            tokenizer = xg.TokenizerInfo(
                [bytes([i]) for i in range(vocab)], stop_token_ids=128
            )
            ctx = xg.GrammarCompiler(tokenizer).compile_grammar(
                'root ::= "{\\"x\\":" ("1" | "2") "}"'
            )
            live.has_grammar = True
            batch.input_ids.copy_(torch.tensor([58, 49, 125, 128, 0, 0], device="cuda"))
            for _ in range(20):
                grammar = XGrammarGrammar(
                    xg.GrammarMatcher(ctx, max_rollback_tokens=200), vocab, ctx, 128
                )
                for token in b'{"x":':
                    grammar.accept_token(token)
                before = grammar.accepted_tokens.copy()
                request.grammar = grammar
                expected_mask, _ = generate_token_bitmask(
                    [request],
                    epilogue.retrieve_next_token.cpu(),
                    epilogue.retrieve_next_sibling.cpu(),
                    batch.input_ids.cpu().view(1, width),
                    vocab,
                )
                self.assertTrue(epilogue.prepare(live))
                barrier = Mock()
                epilogue.arm(live, barrier)
                barrier.assert_called_once_with()
                torch.cuda.manual_seed(45)
                graph.replay()
                actual = epilogue.read(live)
                masked = logits.clone()
                apply_token_bitmask_inplace_triton(masked, expected_mask.cuda())
                torch.cuda.manual_seed(45)
                reference = eagle_sample(
                    candidates, reference_batch, NS(next_token_logits=masked)
                )
                for got, expected in zip(actual[:3], reference, strict=True):
                    torch.testing.assert_close(got, expected, atol=0, rtol=0)
                self.assertEqual(grammar.accepted_tokens, before)
            live.has_grammar = False
            for field, value in (("has_custom_logit_processor", True),):
                setattr(info, field, value)
                self.assertFalse(epilogue.prepare(live))
                epilogue.info.acc_additive_penalties.fill_(7)
                epilogue.info.acc_scaling_penalties.fill_(2)
                epilogue.info.logit_bias.fill_(3)
                before = committed.clone()
                rng = torch.cuda.get_rng_state()
                graphs[1, 0].replay()
                torch.testing.assert_close(committed, before)
                torch.testing.assert_close(self.captured_logits[1, 0], logits)
                self.assertTrue(torch.equal(torch.cuda.get_rng_state(), rng))
                setattr(info, field, False)
        # xgrammar matchers free native state at GC time; collect while the
        # grammar builder thread is done so teardown cannot land inside a
        # later test's CUDA graph capture.
        gc.collect()

    def test_rejection_sampling_graph(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.eagle_utils import eagle_sample

        width, max_bs, vocab = 6, 4, 129
        logits = torch.randn(max_bs * width, vocab, device="cuda")
        self.assertIsNone(
            self._sampling_epilogue(logits, width).info.draft_distribution
        )
        epilogue = self._sampling_epilogue(logits, width, max_bs, rejection=True)
        self.assertEqual(
            tuple(epilogue.info.draft_distribution.shape), (max_bs, width - 1, vocab)
        )
        self.assertFalse(epilogue.info.draft_distribution.any())
        batch = NS(
            input_ids=torch.arange(max_bs * width, device="cuda") % vocab,
            positions=torch.arange(max_bs * width, device="cuda"),
            seq_lens=torch.arange(max_bs, device="cuda") + 63,
            req_pool_indices=torch.arange(max_bs, device="cuda"),
            mamba_track_indices=None,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        committed = torch.zeros(max_bs, dtype=torch.int32, device="cuda")

        def commit(target, b, lengths, *args):
            committed[: lengths.numel()].copy_(lengths)

        draft_probs = torch.softmax(
            torch.randn(max_bs, width - 1, vocab, device="cuda"), dim=-1
        )
        with (
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=NS(world_size=1),
            ),
            patch(
                "sglang.srt.speculative.eagle_verify_graph.commit_mamba_states_after_verify",
                side_effect=commit,
            ),
        ):
            graphs = self._capture_sampling_graphs(epilogue, batch, (4,))
            for bs, ks, ps in (
                (1, [50], [0.95]),
                (1, [vocab], [0.9]),
                (2, [50, 1], [0.95, 1.0]),
                (1, [1], [1.0]),
            ):
                info = NS(
                    temperatures=torch.ones((bs, 1), device="cuda"),
                    top_ks=torch.tensor(ks, dtype=torch.int32, device="cuda"),
                    top_ps=torch.tensor(ps, device="cuda"),
                    is_all_greedy=all(k == 1 for k in ks),
                    sampling_seed=None,
                    has_custom_logit_processor=False,
                )
                live = NS(
                    reqs=[NS(sampling_params=NS(top_k=k)) for k in ks],
                    sampling_info=info,
                    forward_mode=batch.forward_mode,
                    return_logprob=False,
                    has_grammar=False,
                )
                self.assertTrue(epilogue.prepare(live))
                if epilogue.sampling:
                    with self.assertRaisesRegex(
                        RuntimeError, "requires a draft distribution"
                    ):
                        epilogue.arm(live, None)
                    epilogue.arm(live, None, draft_probs=draft_probs[:bs])
                    torch.testing.assert_close(
                        epilogue.info.draft_distribution[:bs],
                        draft_probs[:bs],
                        atol=0,
                        rtol=0,
                    )
                else:
                    epilogue.arm(live, None)
                limit = epilogue.sampling_top_k if epilogue.sampling else 0
                ref_info = epilogue.info.for_batch(4, is_all_greedy=info.is_all_greedy)
                ref_batch = NS(
                    device="cuda",
                    forward_mode=batch.forward_mode,
                    seq_lens=batch.seq_lens[:4],
                    sampling_info=ref_info,
                )
                candidates = NS(
                    draft_token=batch.input_ids[: 4 * width],
                    draft_token_num=width,
                    max_tree_depth=width,
                    tree_topk=1,
                    retrieve_index=epilogue.retrieve_index[:4],
                    retrieve_next_token=epilogue.retrieve_next_token[:4],
                    retrieve_next_sibling=epilogue.retrieve_next_sibling[:4],
                    draft_probs=epilogue.info.draft_distribution[:4],
                )
                logits.normal_()
                torch.cuda.manual_seed(45)
                graphs[4, limit].replay()
                actual = epilogue.read(live)
                torch.cuda.manual_seed(45)
                expected = eagle_sample(
                    candidates,
                    ref_batch,
                    NS(next_token_logits=logits[: 4 * width].clone()),
                )
                for got, want in zip(
                    actual[:3],
                    (expected[0][: bs * width], expected[1][:bs], expected[2][:bs]),
                    strict=True,
                ):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                expected_bonus = expected[0][
                    expected[2][:bs]
                    .gather(1, (expected[1][:bs] - 1).long()[:, None])
                    .flatten()
                ]
                torch.testing.assert_close(actual.bonus_tokens, expected_bonus)
                torch.testing.assert_close(
                    actual.new_seq_lens, batch.seq_lens[:bs] + actual.accept_lens
                )
                torch.testing.assert_close(committed[:bs], actual[1])
                self.assertTrue((committed[bs:] == 0).all())

    def test_batched_sampling_grammar_and_graph_padding(self):
        import xgrammar as xg

        from sglang.kernels.ops.grammar.bitmask_ops import (
            apply_token_bitmask_inplace_triton,
        )
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.eagle_utils import eagle_sample
        from sglang.srt.speculative.spec_utils import generate_token_bitmask

        width, max_bs, vocab = 6, 4, 129
        logits = torch.randn(max_bs * width, vocab, device="cuda")
        epilogue = self._sampling_epilogue(logits, width, max_bs)
        # Full-attention targets have no recurrent state commit callback.
        epilogue.commit = None
        batch = NS(
            input_ids=torch.tensor([0, 49, 128, 0, 0, 0], device="cuda").repeat(max_bs),
            positions=torch.arange(max_bs * width, device="cuda"),
            seq_lens=torch.arange(max_bs, device="cuda") + 63,
            req_pool_indices=torch.arange(max_bs, device="cuda"),
            mamba_track_indices=None,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        committed = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        tokenizer = xg.TokenizerInfo(
            [bytes([i]) for i in range(vocab)], stop_token_ids=128
        )
        compiler = xg.GrammarCompiler(tokenizer)
        contexts = [
            compiler.compile_grammar(text)
            for text in ('root ::= "1" | "2"', 'root ::= "7" | "8"')
        ]

        def commit(target, b, lengths, *args):
            committed[: lengths.numel()].copy_(lengths)

        with (
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=NS(world_size=1),
            ),
            patch(
                "sglang.srt.speculative.eagle_verify_graph.commit_mamba_states_after_verify",
                side_effect=commit,
            ),
        ):
            graphs = self._capture_sampling_graphs(epilogue, batch, (1, 2, 4))
            for bs, grammar_enabled, greedy in (
                (4, True, False),
                (1, False, True),
                (3, True, False),
                (2, True, True),
                (1, True, False),
                (4, False, False),
            ) * 3:
                cap = next(n for n in (1, 2, 4) if n >= bs)
                ks = [1] * bs if greedy else [50, 1, vocab, 7][:bs]
                reqs = []
                for i, k in enumerate(ks):
                    ctx = contexts[(i // 2) % 2]
                    grammar = (
                        XGrammarGrammar(xg.GrammarMatcher(ctx), vocab, ctx, 128)
                        if grammar_enabled and i % 2 == 0
                        else None
                    )
                    reqs.append(NS(grammar=grammar, sampling_params=NS(top_k=k)))
                info = NS(
                    temperatures=torch.tensor(
                        [[0.7], [1.0], [1.3], [0.8]][:bs], device="cuda"
                    ),
                    top_ks=torch.tensor(ks, dtype=torch.int32, device="cuda"),
                    top_ps=torch.tensor([0.95, 1.0, 0.8, 0.9][:bs], device="cuda"),
                    is_all_greedy=greedy,
                    sampling_seed=None,
                    has_custom_logit_processor=False,
                )
                live = NS(
                    reqs=reqs,
                    sampling_info=info,
                    forward_mode=batch.forward_mode,
                    return_logprob=False,
                    has_grammar=grammar_enabled,
                )
                mask, _ = generate_token_bitmask(
                    reqs,
                    epilogue.retrieve_next_token[:bs].cpu(),
                    epilogue.retrieve_next_sibling[:bs].cpu(),
                    batch.input_ids[: bs * width].view(bs, width).cpu(),
                    vocab,
                )
                before = [
                    req.grammar.accepted_tokens.copy() if req.grammar else None
                    for req in reqs
                ]
                self.assertTrue(epilogue.prepare(live))
                epilogue.arm(live, None)
                limit = epilogue.sampling_top_k if epilogue.sampling else 0
                torch.cuda.manual_seed(45)
                graphs[cap, limit].replay()
                actual = epilogue.read(live)
                ref_info = epilogue.info.for_batch(cap, is_all_greedy=greedy)
                ref_logits = logits[: cap * width].clone()
                if mask is not None:
                    apply_token_bitmask_inplace_triton(
                        ref_logits[: bs * width], mask.cuda()
                    )
                verify = NS(
                    draft_token=batch.input_ids[: cap * width],
                    draft_token_num=width,
                    max_tree_depth=width,
                    tree_topk=1,
                    retrieve_index=epilogue.retrieve_index[:cap],
                    retrieve_next_token=epilogue.retrieve_next_token[:cap],
                    retrieve_next_sibling=epilogue.retrieve_next_sibling[:cap],
                )
                ref_batch = NS(
                    device="cuda",
                    forward_mode=batch.forward_mode,
                    seq_lens=batch.seq_lens[:cap],
                    sampling_info=ref_info,
                )
                torch.cuda.manual_seed(45)
                expected = eagle_sample(
                    verify, ref_batch, NS(next_token_logits=ref_logits)
                )
                for got, want in zip(
                    actual[:3],
                    (expected[0][: bs * width], expected[1][:bs], expected[2][:bs]),
                    strict=True,
                ):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                expected_bonus = expected[0][
                    expected[2][:bs]
                    .gather(1, (expected[1][:bs] - 1).long()[:, None])
                    .flatten()
                ]
                torch.testing.assert_close(actual.bonus_tokens, expected_bonus)
                torch.testing.assert_close(
                    actual.new_seq_lens, batch.seq_lens[:bs] + actual.accept_lens
                )
                self.assertTrue((committed == 0).all())
                self.assertTrue((committed[bs:cap] == 0).all())
                self.assertEqual(
                    [
                        req.grammar.accepted_tokens if req.grammar else None
                        for req in reqs
                    ],
                    before,
                )
        gc.collect()


if __name__ == "__main__":
    unittest.main()
