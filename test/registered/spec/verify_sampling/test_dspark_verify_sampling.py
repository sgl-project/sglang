import builtins
import gc
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.sampling.verify_graph import verify_logits_adjustments_are_noop
from sglang.srt.speculative.dspark_components.dspark_commit_graph import (
    DSparkCompactVerifyEpilogue,
    DSparkStaticVerifyEpilogue,
)
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class VerifySamplingTest(TestCase):
    def test_moe_keeps_native_epilogue_and_single_capture(self):
        from sglang.srt.speculative.dspark_components import (
            dspark_worker_v2 as worker_module,
        )

        for moe in (True, False):
            for compact in (True, False):
                with self.subTest(moe=moe, compact=compact):
                    runner = NS(
                        model_config=NS(
                            hf_text_config=NS(model_type="deepseek_v41"), vocab_size=129
                        ),
                        capture_tail_hooks=[],
                        spec_verify_epilogue=None,
                        tp_rank=0,
                        pp_size=1,
                    )
                    draft = NS(
                        sample_from_anchor=False,
                        uses_own_vocab_modules=True,
                        markov_head=None,
                    )
                    bundle = NS(
                        draft_worker=Mock(),
                        draft_model=draft,
                        draft_model_runner=NS(model_config=NS(hf_config=NS())),
                        resolved_attention_backend="trtllm_mha",
                    )
                    planner = NS(
                        is_compact_mode=compact,
                        mode_value="compact" if compact else "static",
                    )
                    with (
                        patch.multiple(
                            worker_module,
                            get_schedule=lambda: NS(page_size=64),
                            get_parallel=lambda: NS(
                                enable_dp_attention=False,
                                attn_dp_enabled=False,
                                tp_rank=0,
                                pp_size=1,
                                tp_group=Mock(),
                                pp_group=NS(is_last_rank=True),
                            ),
                            get_disagg=lambda: NS(disaggregation_mode="null"),
                            get_exec=lambda: NS(
                                graph=NS(
                                    cuda_graph_config=NS(
                                        decode=NS(backend="full", bs=[1, 2])
                                    )
                                )
                            ),
                            get_spec=lambda: NS(
                                speculative_draft_window_size=None,
                                speculative_num_draft_tokens=3,
                            ),
                            draft_is_deepseek_v4=lambda moe=moe: moe,
                            draft_pp_context=nullcontext,
                            build_draft_tp_worker=Mock(return_value=bundle),
                            mambaish_config=lambda _: None,
                            resolve_runtime_config=lambda **kwargs: NS(
                                gamma=2, verify_num_draft_tokens=3, mask_token_id=1
                            ),
                            get_dp_tp_group=Mock(),
                            SpecTpSync=Mock(),
                            make_draft_block_spec_info=Mock(),
                            DSparkVerifyPlanner=Mock(return_value=planner),
                            TargetHiddenKvInjector=Mock(),
                            DraftBlockProposer=Mock(),
                            TargetVerifyExecutor=Mock(),
                            DsparkStepObservers=Mock(),
                            is_cuda=lambda: True,
                        ),
                        patch.object(
                            worker_module.envs.SGLANG_SIMULATE_ACC_LEN,
                            "get",
                            return_value=0,
                        ),
                        patch.dict(
                            worker_module.os.environ,
                            {"SGLANG_DSPARK_XGRAMMAR_HOST_CALLBACK": "0"},
                        ),
                    ):
                        worker = worker_module.DSparkWorkerV2(
                            NS(),
                            0,
                            1234,
                            NS(model_runner=runner, device="cpu", random_seed=0),
                        )
                    expected = (
                        worker_module.DsparkVerifyEpilogue
                        if moe
                        else DSparkCompactVerifyEpilogue
                        if compact
                        else DSparkStaticVerifyEpilogue
                    )
                    self.assertIs(type(worker._verify_epilogue), expected)
                    self.assertEqual(len(runner.capture_tail_hooks), 1)
                    self.assertIs(
                        runner.spec_verify_epilogue,
                        None if moe else worker._verify_epilogue,
                    )
                    graph = NS(
                        model_runner=runner,
                        ragged_verify_mode=False,
                        _capture_one_shape=Mock(),
                    )
                    DecodeCudaGraphRunner.capture_one_shape(graph, 1, Mock())
                    self.assertEqual(
                        graph._capture_one_shape.call_count, 1 if moe else 3
                    )

    def test_mamba_commit_keeps_cpu_sequence_mirror(self):
        from sglang.srt.speculative.dspark_components import dspark_worker_v2

        backend = Mock()
        target = NS(model_runner=NS(attn_backend=backend, page_size=64, model=None))
        worker = NS(_need_mamba_verify_commit=True, target_worker=target)
        epilogue = NS(
            folds_mamba_commit=True,
            target_worker=target,
            commit_mamba=lambda **kwargs: (
                dspark_worker_v2.DSparkWorkerV2._commit_target_mamba_states_after_verify(
                    worker, **kwargs
                )
            ),
        )
        batch = NS(
            mamba_track_indices=torch.tensor([3, 4]),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([63, 64]),
        )
        with (
            patch.object(
                dspark_worker_v2, "get_spec", return_value=NS(speculative_eagle_topk=1)
            ),
            patch.object(dspark_worker_v2, "mamba_track_grid", return_value=64),
            patch.object(dspark_worker_v2, "_is_npu", False),
        ):
            for mirror in (None, batch.seq_lens):
                batch.seq_lens_cpu = mirror
                DSparkStaticVerifyEpilogue._commit_mamba(
                    epilogue, batch, torch.tensor([2, 1])
                )
                committed = backend.update_mamba_state_after_mtp_verify.call_args.kwargs
                self.assertEqual(
                    committed["last_correct_step_indices"].tolist(), [1, 0]
                )
                self.assertEqual(committed["mamba_steps_to_track"].tolist(), [0, -1])

    def tearDown(self):
        # CUDA callbacks cannot free pinned buffers from previous graph fixtures.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()

    @skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_result_copy_survives_next_graph_buffer_write(self):
        from sglang.srt.managers.utils import GenerationBatchResult

        for epilogue_type in (DSparkStaticVerifyEpilogue, DSparkCompactVerifyEpilogue):
            with self.subTest(epilogue=epilogue_type.__name__):
                epilogue = epilogue_type(
                    tp_sync=SpecTpSync(NS(world_size=1)),
                    max_bs=1,
                    verify_num_draft_tokens=8,
                    vocab_size=129,
                    device="cuda",
                )
                epilogue.out_tokens_buf.fill_(11)
                epilogue.commit_lens_buf.fill_(2)
                accept = epilogue.read_accept(1)
                saved = {
                    name: getattr(accept, name).clone()
                    for name in accept.__struct_fields__
                }
                for name in ("correct_len", "bonus", "cap_trim_lens", "new_seq_lens"):
                    getattr(epilogue, f"{name}_buf").fill_(123)
                for name, value in saved.items():
                    torch.testing.assert_close(getattr(accept, name), value)
                result = GenerationBatchResult(
                    next_token_ids=accept.out_tokens.reshape(-1),
                    accept_lens=accept.commit_lens,
                    copy_done=torch.cuda.Event(),
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(100_000_000)
                    result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
                epilogue.out_tokens_buf.fill_(22)
                epilogue.commit_lens_buf.fill_(5)
                result.copy_done.synchronize()
                self.assertEqual(result.next_token_ids.tolist(), [11] * 8)
                self.assertEqual(result.accept_lens.tolist(), [2])

    @skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_confidence_copy_survives_next_publication(self):
        from sglang.srt.managers.overlap_utils import ConfidenceRelay

        relay = ConfidenceRelay(
            device=torch.device("cuda"),
            req_pool_size=1,
            pool=NS(req_generation=torch.tensor([1])),
        )
        indices = torch.tensor([0], device="cuda")
        relay.scatter(indices, torch.full((1, 7), 0.25, device="cuda"))
        published = torch.cuda.Event()
        published.record()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
        relay.issue_ring_copy(stream=stream, publish_ready=published)
        relay.scatter(indices, torch.full((1, 7), 0.75, device="cuda"))
        relay.copy_done[0].synchronize()
        self.assertEqual(relay.conf_ring[0, 0].tolist(), [0.25] * 7)

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_replayssm_commit_matches_eager(self):
        for compact, max_bs in (
            (True, 1),
            (True, 2),
            (True, 3),
            (False, 1),
            (False, 2),
        ):
            with self.subTest(compact=compact, max_bs=max_bs):
                self._check_replayssm_commit_matches_eager(compact, max_bs)

    def _check_replayssm_commit_matches_eager(self, compact, max_bs):
        from sglang.srt.configs import KimiLinearConfig
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_context
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        override = get_context().override_server_args(
            mamba_track_interval=64, _mamba_cache_chunk_size=64
        )
        override.install()
        self.addCleanup(override.restore)
        width, layers, slots, heads, dim = 8, 2, 2 * max_bs + 3, 2, 32
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
        mamba = MambaPool.__new__(MambaPool)
        mamba.replayssm_spec_fold = mamba.replayssm_is_kda = True
        pool = NS(
            mamba_pool=mamba,
            # Scheduler mappings may already differ from the captured verify metadata.
            get_mamba_indices=Mock(
                side_effect=AssertionError("late shared mapping read")
            ),
            get_speculative_mamba2_params_all_layers=lambda: state,
        )
        batch = NS(
            input_ids=torch.arange(width, device="cuda").repeat(max_bs),
            batch_size=max_bs,
            seq_lens=torch.full((max_bs,), 63, device="cuda"),
            seq_lens_cpu=None,
            req_pool_indices=torch.arange(1, max_bs + 1, device="cuda"),
            mamba_track_indices=torch.arange(max_bs + 2, 2 * max_bs + 2, device="cuda"),
            tree_cache=NS(page_size=1),
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
        backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        backend.linear_attn_backend = NS(
            req_to_token_pool=pool,
            _translate_mamba_indices=lambda ids: ids,
            forward_metadata=NS(mamba_cache_indices=batch.req_pool_indices),
        )
        target = NS(
            model_runner=NS(
                req_to_token_pool=pool,
                attn_backend=backend,
                page_size=1,
                model=None,
                model_config=NS(hf_config=KimiLinearConfig()),
            )
        )
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._target_worker = target
        worker._need_mamba_verify_commit = True
        epilogue_type = (
            DSparkCompactVerifyEpilogue if compact else DSparkStaticVerifyEpilogue
        )
        epilogue = epilogue_type(
            tp_sync=SpecTpSync(NS(world_size=1)),
            max_bs=max_bs,
            verify_num_draft_tokens=width,
            vocab_size=129,
            device="cuda",
            target_worker=target,
            commit_mamba=worker._commit_target_mamba_states_after_verify,
            commit_ctx=NS(
                resolve_pool=lambda: MHATokenToKVPool.__new__(MHATokenToKVPool),
                draft_model=NS(write_target_hidden_kv=Mock()),
            ),
        )
        self.assertTrue(epilogue.folds_mamba_commit)
        logits = torch.zeros(max_bs * width, 129, device="cuda")
        hidden = torch.zeros(max_bs * width, 4, device="cuda")
        lens = torch.full((max_bs,), width, device="cuda")
        runner = NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=compact)
        epilogue.draft_tokens_buf.copy_(
            batch.input_ids.view(max_bs, width)[:, 1:].reshape(-1)
        )

        def verify(bucket):
            packed_logits = logits[: max_bs * bucket]
            if compact:
                packed_logits = torch.cat(
                    [
                        logits[row * width : row * width + bucket]
                        for row in range(max_bs)
                    ]
                )
                batch.input_ids = torch.arange(bucket, device="cuda").repeat(max_bs)
            epilogue.capture_hook(
                runner,
                NS(
                    next_token_logits=packed_logits.clone(),
                    hidden_states=hidden[: max_bs * bucket],
                ),
                batch,
                max_bs * bucket,
            )

        graphs = {}
        for bucket in range(1, width + 1) if compact else (width,):
            lens.fill_(bucket)
            epilogue.begin_step(lens, armed=False)
            verify(bucket)
            graphs[bucket] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[bucket]):
                verify(bucket)
        initial = state.temporal.clone(), state.conv[0].clone()
        for seq in (55, 59, 63, 64):
            for bucket in graphs:
                for accepted in range(1, bucket + 1):
                    batch.seq_lens.copy_(seq + torch.arange(max_bs, device="cuda"))
                    batch.req_pool_indices.copy_(
                        1 + accepted % 2 + torch.arange(max_bs, device="cuda")
                    )
                    logits.fill_(-100)
                    logits[:, 100] = 100
                    counts = torch.tensor(
                        [(accepted + row - 1) % bucket + 1 for row in range(max_bs)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    for row, count in enumerate(counts.tolist()):
                        for i in range(count - 1):
                            logits[row * width + i, i + 1] = 200
                    state.temporal.copy_(initial[0])
                    state.conv[0].copy_(initial[1])
                    lens.fill_(bucket)
                    # Exercise BS1 padded to the BS2 graph as well as full BS2.
                    if max_bs > 1 and accepted % 2 == 0:
                        lens[-1] = 0
                        counts[-1] = 0
                    epilogue.begin_step(lens, armed=compact)
                    if not compact:
                        epilogue.stage_sampling(
                            bs=max_bs,
                            sampling_info=None,
                            draft_block=None,
                            grammar_mask=None,
                        )
                    graphs[bucket].replay()
                    actual = state.temporal.clone(), state.conv[0].clone()
                    torch.testing.assert_close(
                        torch.minimum(epilogue.commit_lens_buf, lens.to(torch.int32)),
                        counts,
                        atol=0,
                        rtol=0,
                    )
                    state.temporal.copy_(initial[0])
                    state.conv[0].copy_(initial[1])
                    worker._commit_target_mamba_states_after_verify(
                        batch=batch,
                        seq_lens_pre_verify=batch.seq_lens,
                        seq_lens_post_verify=batch.seq_lens + counts,
                        commit_lens=counts,
                    )
                    for got, expected in zip(
                        actual, (state.temporal, state.conv[0]), strict=True
                    ):
                        torch.testing.assert_close(got, expected, atol=0, rtol=0)
                    epilogue.begin_step(lens, armed=False)
                    graphs[bucket].replay()
                    torch.testing.assert_close(
                        state.temporal, actual[0], atol=0, rtol=0
                    )
                    torch.testing.assert_close(state.conv[0], actual[1], atol=0, rtol=0)

    def test_host_callback_requires_target(self):
        for max_bs in (1, 2):
            epilogue = DSparkStaticVerifyEpilogue(
                tp_sync=SpecTpSync(NS(world_size=1)),
                max_bs=max_bs,
                verify_num_draft_tokens=8,
                vocab_size=129,
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "require a target model"):
                epilogue.enable_grammar_host_callback(None)

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_host_callback_masks_acceptance(self):
        for sampled in (False, True):
            for compact in (False, True):
                with self.subTest(sampled=sampled, compact=compact):
                    self._host_callback_masks_acceptance(sampled, compact)

    def _host_callback_masks_acceptance(self, sampled, compact):
        # Dispose previous fixtures' pinned buffers outside the CUDA callback thread.
        gc.collect()
        import xgrammar as xg
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        stride, vocab = 8, 129
        tokenizer = xg.TokenizerInfo(
            [bytes([i]) for i in range(vocab)], stop_token_ids=128
        )
        compiler = xg.GrammarCompiler(tokenizer)
        contexts = [compiler.compile_grammar(f'root ::= "{c}"') for c in "ab"]
        epilogue_class = (
            DSparkCompactVerifyEpilogue if compact else DSparkStaticVerifyEpilogue
        )
        epilogue = epilogue_class(
            tp_sync=SpecTpSync(NS(world_size=1)),
            max_bs=1,
            verify_num_draft_tokens=stride,
            vocab_size=vocab,
            device="cuda",
        )
        ids = torch.zeros(stride, dtype=torch.int64, device="cuda")
        positions = torch.arange(stride, device="cuda")
        seq_lens = torch.tensor([40], device="cuda")
        verify_lens = torch.tensor([stride], device="cuda")
        logits = torch.zeros(stride, vocab, device="cuda")
        logits[:, 90] = 1000  # Invalid under either grammar.
        model = NS(forward=lambda ids, positions, batch: logits[: ids.numel()].clone())
        epilogue.enable_grammar_host_callback(model)
        host = epilogue.host_grammar
        batch = NS(forward_mode=NS(is_target_verify=lambda: True))
        info = (
            NS(
                is_all_greedy=False,
                is_any_greedy=False,
                temperatures=torch.ones((1, 1), device="cuda"),
                top_ks=torch.tensor([50], device="cuda"),
                top_ps=torch.tensor([0.95], device="cuda"),
                need_top_k_sampling=True,
                need_top_p_sampling=True,
            )
            if sampled
            else None
        )
        draft = NS(
            temperatures=torch.ones(1, device="cuda"),
            greedy_mask=torch.zeros(1, dtype=torch.bool, device="cuda"),
            corrected_logits=torch.zeros(1, stride - 1, vocab, device="cuda"),
        )

        def stage():
            epilogue.begin_step(None, armed=False)
            epilogue.stage_sampling(
                bs=1,
                sampling_info=info,
                draft_block=draft,
                grammar_mask=None,
                max_top_k=50,
            )
            epilogue.draft_tokens_buf.copy_(ids[1:])
            if compact:
                epilogue.begin_step(verify_lens, armed=True)

        def verify(length):
            output = model.forward(ids[:length], positions[:length], batch)
            if compact:
                epilogue.capture_hook(
                    NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=True),
                    LogitsProcessorOutput(
                        next_token_logits=output, hidden_states=output
                    ),
                    NS(
                        input_ids=ids[:length],
                        seq_lens=seq_lens,
                        req_pool_indices=seq_lens,
                        batch_size=1,
                    ),
                    length,
                )
            else:
                epilogue.strided_logits = output
                epilogue._accept(ids, seq_lens, verify_lens, 1)
            return epilogue.strided_logits

        stage()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            verify(stride)
        torch.cuda.current_stream().wait_stream(stream)
        graphs = {}
        for length in (1, 2, 4, 8) if compact else (stride,):
            verify_lens.fill_(length)
            stage()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = verify(length)
            graphs[length] = graph, output
        for step in range(10):
            length = (8, 2, 4, 1)[step % 4] if compact else stride
            graph, epilogue.strided_logits = graphs[length]
            verify_lens.fill_(length)
            ctx = contexts[step % 2]
            grammar = XGrammarGrammar(
                xg.GrammarMatcher(ctx, max_rollback_tokens=200), vocab, ctx, 128
            )
            stage()
            barrier = Mock()
            host.prepare(grammar, barrier)
            barrier.assert_called_once_with()
            graph.replay()
            host.finish()
            self.assertEqual(epilogue.correct_len_buf.item(), 0)
            self.assertEqual(epilogue.bonus_buf.item(), ord("ab"[step % 2]))
            self.assertEqual(grammar.accepted_tokens, [])
        stage()
        graph.replay()
        self.assertEqual(epilogue.bonus_buf.item(), 90)
        self.assertTrue((epilogue.sampling_buffers.vocab_mask == -1).all())

    def test_forced_budget_overrides_verify_all(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._is_verify_all = True
        planner._budget_planner = NS(forced_budget_frac=None)
        self.assertTrue(planner.is_verify_all)
        planner.set_forced_budget_frac(0.25)
        self.assertFalse(planner.is_verify_all)
        planner.set_forced_budget_frac(None)
        self.assertTrue(planner.is_verify_all)
        planner._is_verify_all = False
        self.assertFalse(planner.is_verify_all)

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_batched_compact_host_callback(self):
        gc.collect()
        import xgrammar as xg
        from sglang.srt.constrained.grammar_graph import GrammarHostCallback
        from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
        from sglang.srt.speculative.spec_utils import traverse_tree

        bs, width, vocab = 3, 4, 129
        compiler = xg.GrammarCompiler(
            xg.TokenizerInfo([bytes([i]) for i in range(vocab)], stop_token_ids=128)
        )
        contexts = [compiler.compile_grammar(f'root ::= "{c}"+') for c in "ab"]
        mask = torch.full(
            (bs * width, (vocab + 31) // 32), -1, dtype=torch.int32, device="cuda"
        )
        host = GrammarHostCallback(mask, width)
        ids = torch.zeros(bs * width, dtype=torch.long, device="cuda")
        lengths = torch.full((bs,), width, device="cuda")
        batch = NS(
            batch_size=bs,
            forward_mode=NS(is_target_verify=lambda: True),
            spec_info=NS(ragged_verify_layout=NS(verify_lens=lengths)),
        )
        model = NS(forward=lambda *args: None)
        host.bind(model)
        model.forward(ids, ids, batch)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model.forward(ids, ids, batch)
            output = mask.clone()
        for lens in ([1, 4, 2], [3, 1, 0], [2, 0, 0]):
            grammars = [
                XGrammarGrammar(
                    xg.GrammarMatcher(ctx, max_rollback_tokens=200), vocab, ctx, 128
                )
                for ctx in contexts
            ]
            grammars.append(None)
            chains = [
                [0] + [ord("abc"[row])] * (length - 1) if length else []
                for row, length in enumerate(lens)
            ]
            packed = torch.tensor(sum(chains, []), device="cuda")
            ids[: packed.numel()].copy_(packed)
            lengths.copy_(torch.tensor(lens, device="cuda"))
            host.prepare(grammars, None)
            graph.replay()
            host.finish()
            expected = torch.full_like(host.mask_cpu, -1)
            for row, grammar in enumerate(grammars):
                if grammar is not None:
                    tokens = torch.tensor(chains[row] + [0] * (width - lens[row]))
                    traverse_tree(
                        host.next_token,
                        host.next_sibling,
                        tokens,
                        grammar,
                        expected[row * width : (row + 1) * width],
                        vocab_size=vocab,
                    )
                    self.assertEqual(grammar.accepted_tokens, [])
            torch.testing.assert_close(output.cpu(), expected)

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_compact_sampling_matches_eager(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.runner_utils.capture_owner import (
            collect_full_cuda_graph_owners,
        )
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            accept_draft_tokens,
        )

        stride, vocab = 8, 129
        epilogue = DSparkCompactVerifyEpilogue(
            tp_sync=SpecTpSync(NS(world_size=1)),
            max_bs=1,
            verify_num_draft_tokens=stride,
            vocab_size=vocab,
            device="cuda",
        )
        self.assertFalse(epilogue.folds_commit)
        logits = torch.randn(stride, vocab, device="cuda")
        hidden = torch.randn(stride, 4, device="cuda")
        ids = torch.randint(vocab, (stride,), device="cuda")
        seq_lens = torch.tensor([40], device="cuda")
        lengths = torch.tensor([stride], device="cuda")
        info = NS(
            is_all_greedy=False,
            is_any_greedy=False,
            temperatures=torch.tensor([[0.7]], device="cuda"),
            top_ks=torch.tensor([50], device="cuda"),
            top_ps=torch.tensor([0.95], device="cuda"),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
        )
        draft = NS(
            temperatures=info.temperatures.flatten(),
            greedy_mask=torch.tensor([False], device="cuda"),
            corrected_logits=torch.randn(1, stride - 1, vocab, device="cuda"),
        )
        runner = NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=True)

        def stage(top_k):
            epilogue.stage_sampling(
                bs=1,
                sampling_info=info,
                draft_block=draft,
                grammar_mask=None,
                max_top_k=top_k,
            )
            epilogue.draft_tokens_buf.copy_(ids[1:])
            epilogue.begin_step(lengths, armed=True)

        def verify(bucket):
            epilogue.capture_hook(
                runner,
                LogitsProcessorOutput(
                    next_token_logits=logits[:bucket], hidden_states=hidden[:bucket]
                ),
                NS(
                    input_ids=ids[:bucket],
                    seq_lens=seq_lens,
                    req_pool_indices=seq_lens,
                    batch_size=1,
                ),
                bucket,
            )

        owners = []
        for top_k in (64, vocab):
            info.top_ks.fill_(50 if top_k == 64 else vocab)
            stage(top_k)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                verify(stride)
            torch.cuda.current_stream().wait_stream(stream)
            graphs = {}
            for bucket in range(1, stride + 1):
                lengths.fill_(bucket)
                stage(top_k)
                graph = torch.cuda.CUDAGraph()
                with (
                    collect_full_cuda_graph_owners() as retained,
                    torch.cuda.graph(graph),
                ):
                    verify(bucket)
                owners.extend(retained)
                graphs[bucket] = graph
            for bucket in (8, 2, 7, 1, 4, 6, 3, 5):
                for length in {1, bucket}:
                    lengths.fill_(length)
                    logits.normal_()
                    draft.corrected_logits.normal_()
                    stage(top_k)
                    torch.cuda.manual_seed(123)
                    graphs[bucket].replay()
                    actual = [
                        t.clone()
                        for t in (
                            epilogue.correct_len_buf,
                            epilogue.bonus_buf,
                            epilogue.cap_trim_lens_buf,
                        )
                    ]
                    torch.cuda.manual_seed(123)
                    expected = accept_draft_tokens(
                        candidates=ids.view(1, stride),
                        target_logits=epilogue.strided_logits,
                        draft_block=draft,
                        sampling_info=info,
                        draft_input=NS(max_top_k=top_k, uniform_top_k_value=None),
                        gamma=stride - 1,
                        verify_num_draft_tokens=stride,
                        cutoff_layout=NS(verify_lens=lengths),
                    )
                    for got, want in zip(actual, expected, strict=True):
                        torch.testing.assert_close(
                            got, want.to(got.dtype), rtol=0, atol=0
                        )
                    self.assertLess(epilogue.correct_len_buf.item(), length)
                    torch.testing.assert_close(
                        epilogue.strided_hidden[:length], hidden[:length]
                    )
                    self.assertTrue((epilogue.strided_hidden[length:] == 0).all())
            draft.corrected_logits.fill_(-float("inf"))
            draft.corrected_logits.scatter_(2, ids[None, 1:, None], 0)
            for length in range(1, stride + 1):
                lengths.fill_(length)
                logits.fill_(-float("inf"))
                predicted = torch.cat((ids[1:], ids[:1]))
                logits.scatter_(1, predicted[:, None], 0)
                stage(top_k)
                graphs[length].replay()
                self.assertEqual(epilogue.correct_len_buf.item(), length - 1)
                self.assertEqual(
                    epilogue.bonus_buf.item(), predicted[length - 1].item()
                )
                self.assertEqual(epilogue.commit_lens_buf.item(), length)
                self.assertEqual(epilogue.new_seq_lens_buf.item(), 40 + length)

                logits[0].fill_(-float("inf"))
                rejected_bonus = (ids[1].item() + 1) % vocab
                logits[0, rejected_bonus] = 0
                graphs[length].replay()
                self.assertEqual(epilogue.correct_len_buf.item(), 0)
                self.assertEqual(epilogue.bonus_buf.item(), rejected_bonus)
                self.assertEqual(epilogue.commit_lens_buf.item(), 1)
        self.assertTrue(owners)

    def test_staging_and_fallback_reset(self):
        for max_bs in (2, 3):
            with self.subTest(max_bs=max_bs):
                self._check_staging_and_fallback_reset(max_bs)

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_batched_compact_sampling_and_commit(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            accept_draft_tokens,
        )

        bs, width, vocab = 3, 8, 129
        hidden = torch.randn(bs * width, 4, device="cuda")
        written = torch.zeros_like(hidden)
        written_positions = torch.zeros(bs * width, dtype=torch.long, device="cuda")
        # A sink slot checks that masked graph padding cannot overwrite live KV.
        storage = torch.zeros(bs * width + 1, 4, device="cuda")
        positions = torch.zeros(bs * width + 1, dtype=torch.long, device="cuda")

        def write(**kw):
            locations = kw["cache_loc"]
            mask = kw["commit_lens"].bool()
            destination = torch.where(mask, locations, bs * width)
            storage[destination] = kw["target_hidden"]
            positions[destination] = kw["positions"]
            written.copy_(storage[:-1])
            written_positions.copy_(positions[:-1])

        epilogue = DSparkCompactVerifyEpilogue(
            tp_sync=SpecTpSync(NS(world_size=1)),
            max_bs=bs,
            verify_num_draft_tokens=width,
            vocab_size=vocab,
            device="cuda",
            commit_ctx=NS(
                resolve_pool=lambda: MHATokenToKVPool.__new__(MHATokenToKVPool),
                draft_model=NS(write_target_hidden_kv=write),
            ),
        )
        seq_lens = torch.tensor([40, 80, 120], device="cuda")
        locations = torch.arange(bs * width, device="cuda").view(bs, width)
        epilogue.prepare(
            NS(
                verify_cache_loc_2d=locations,
                positions_2d=seq_lens[:, None] + torch.arange(width, device="cuda"),
            ),
            bs,
        )
        candidates = torch.randint(vocab, (bs, width), device="cuda")
        packed_ids = torch.zeros(bs * width, dtype=torch.long, device="cuda")
        logits = torch.randn(bs * width, vocab, device="cuda")
        lengths = torch.full((bs,), width, device="cuda")
        info = NS(
            is_all_greedy=False,
            is_any_greedy=True,
            temperatures=torch.tensor([[1.0], [0.7], [0.9]], device="cuda"),
            top_ks=torch.tensor([1, 50, 50], device="cuda"),
            top_ps=torch.tensor([1.0, 0.95, 0.8], device="cuda"),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
        )
        draft = NS(
            temperatures=info.temperatures.flatten(),
            greedy_mask=torch.tensor([True, False, False], device="cuda"),
            corrected_logits=torch.randn(bs, width - 1, vocab, device="cuda"),
        )
        batch = NS(batch_size=bs, input_ids=packed_ids, seq_lens=seq_lens)
        runner = NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=True)
        epilogue.stage_sampling(
            bs=bs,
            sampling_info=info,
            draft_block=draft,
            grammar_mask=None,
            max_top_k=50,
        )
        epilogue.draft_tokens_buf.copy_(candidates[:, 1:].reshape(-1))
        epilogue.begin_step(lengths, armed=True)

        def verify():
            epilogue.capture_hook(
                runner,
                NS(next_token_logits=logits.clone(), hidden_states=hidden),
                batch,
                hidden.shape[0],
            )

        verify()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify()
        for lens in ([1, 7, 4], [8, 1, 2], [3, 0, 5], [2, 0, 0]):
            with self.subTest(lengths=lens):
                lengths.copy_(torch.tensor(lens, device="cuda"))
                packed = torch.cat(
                    [candidates[row, :length] for row, length in enumerate(lens)]
                )
                packed_ids[: packed.numel()].copy_(packed)
                epilogue.begin_step(lengths, armed=True)
                storage.zero_()
                positions.zero_()
                torch.cuda.manual_seed(123)
                graph.replay()
                actual = [
                    buf.clone()
                    for buf in (
                        epilogue.correct_len_buf,
                        epilogue.bonus_buf,
                        epilogue.cap_trim_lens_buf,
                    )
                ]
                torch.cuda.manual_seed(123)
                expected = accept_draft_tokens(
                    candidates=candidates,
                    target_logits=epilogue.strided_logits,
                    draft_block=draft,
                    sampling_info=info,
                    draft_input=NS(max_top_k=64, uniform_top_k_value=None),
                    gamma=width - 1,
                    verify_num_draft_tokens=width,
                    cutoff_layout=NS(verify_lens=lengths.clamp_min(1)),
                )
                for got, want in zip(actual, expected, strict=True):
                    torch.testing.assert_close(
                        got[lengths > 0],
                        want[lengths > 0].to(got.dtype),
                        atol=0,
                        rtol=0,
                    )
                expected_hidden = torch.zeros_like(written)
                expected_positions = torch.zeros_like(written_positions)
                start = 0
                for row, length in enumerate(lens):
                    count = min(length, epilogue.commit_lens_buf[row].item())
                    expected_hidden[row * width : row * width + count] = hidden[
                        start : start + count
                    ]
                    expected_positions[row * width : row * width + count] = seq_lens[
                        row
                    ] + torch.arange(count, device="cuda")
                    start += length
                torch.testing.assert_close(written, expected_hidden)
                torch.testing.assert_close(written_positions, expected_positions)
                epilogue.begin_step(lengths, armed=False)
                graph.replay()
                torch.testing.assert_close(written, expected_hidden)

    def _check_staging_and_fallback_reset(self, max_bs):
        epilogue = DSparkStaticVerifyEpilogue(
            tp_sync=SpecTpSync(NS(world_size=1)),
            max_bs=max_bs,
            verify_num_draft_tokens=4,
            vocab_size=65,
            device="cpu",
        )
        buffers = epilogue.sampling_buffers
        buffers.temperatures.fill_(2)
        buffers.top_ks.fill_(7)
        buffers.top_ps.fill_(0.5)
        buffers.greedy_mask.fill_(False)
        pointers = {
            name: value.data_ptr()
            for name, value in vars(buffers).items()
            if isinstance(value, torch.Tensor)
        }
        info = NS(
            is_all_greedy=False,
            top_ks=torch.tensor([1, 50]),
            top_ps=torch.tensor([1.0, 0.95]),
        )
        draft = NS(
            temperatures=torch.tensor([1.0, 0.6]),
            greedy_mask=torch.tensor([True, False]),
            corrected_logits=torch.randn(2, 3, 65, dtype=torch.bfloat16),
        )
        mask = NS(vocab_mask=torch.zeros((8, 3), dtype=torch.int32))
        epilogue.stage_sampling(
            bs=2, sampling_info=info, draft_block=draft, grammar_mask=mask
        )
        epilogue.begin_step(torch.tensor([4, 2]), armed=True)
        self.assertTrue(epilogue.sampling)
        self.assertEqual(buffers.top_ks.tolist(), [1, 50, 1][:max_bs])
        self.assertEqual(buffers.greedy_mask.tolist(), [True, False, True][:max_bs])
        self.assertTrue((buffers.temperatures[2:] == 1).all())
        self.assertTrue((buffers.top_ps[2:] == 1).all())
        self.assertEqual(epilogue.verify_lens_buf.tolist(), [4, 2, 0][:max_bs])
        torch.testing.assert_close(
            buffers.draft_distribution[:2], draft.corrected_logits.float()
        )
        self.assertTrue((buffers.vocab_mask[:8] == 0).all())
        self.assertTrue((buffers.vocab_mask[8:] == -1).all())

        # A reused slot without a grammar must not inherit its predecessor's mask.
        epilogue.stage_sampling(
            bs=1, sampling_info=None, draft_block=None, grammar_mask=None
        )
        self.assertFalse(epilogue.sampling)
        self.assertTrue((buffers.vocab_mask == -1).all())
        epilogue.stage_sampling(
            bs=2, sampling_info=info, draft_block=draft, grammar_mask=mask
        )
        epilogue.begin_step(None, armed=False)
        self.assertFalse(epilogue.sampling)
        self.assertTrue((buffers.vocab_mask == -1).all())
        self.assertEqual(
            pointers,
            {
                name: value.data_ptr()
                for name, value in vars(buffers).items()
                if isinstance(value, torch.Tensor)
            },
        )

    def test_uncaptured_sampling_shape_is_refused(self):
        # 64, not 50: sampling_top_k is the NORMALISED graph-shape field, and the
        # epilogue maps any raw request with 0 < top_k <= 64 to
        # min(64, vocab_size) (dspark_commit_graph.py). A raw top_k of 50 -- the
        # common case, and Kimi-K3's -- arrives here as 64, so 50 is not a state
        # the runtime can produce and testing it would prove nothing.
        epilogue = NS(sampling=True, sampling_top_k=64, vocab_size=129)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(spec_verify_epilogue=epilogue)
        runner._captured_spec_top_ks = {0}
        # The check precedes every other admission test, so a bare batch is never
        # inspected: an uncaptured 64 is refused here rather than reaching replay
        # and raising KeyError.
        self.assertFalse(runner.can_run_graph(NS()))
        # Greedy is captured, so admission continues past the check and trips on
        # the stub batch. Reaching that proves the check let it through.
        epilogue.sampling = False
        with self.assertRaises(AttributeError):
            runner.can_run_graph(NS())

    def test_capture_records_every_variant_it_captured(self):
        epilogue = NS(sampling=False, sampling_top_k=128, vocab_size=128)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(spec_verify_epilogue=epilogue)
        runner.ragged_verify_mode = False
        with patch.object(runner, "_capture_one_shape"):
            runner.capture_one_shape(1, None)
        # Admission checks membership of this set, so it must match the capture
        # loop exactly or a captured graph becomes unreachable -- or worse, an
        # uncaptured one reachable.
        self.assertEqual(runner._captured_spec_top_ks, {128, 64, 0})

    def test_graph_variant_reaches_the_attention_metadata_key(self):
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )

        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        # A backend nobody publishes to keys exactly as it did before.
        self.assertEqual(backend._cg_metadata_key(4), (4, 0))

        epilogue = NS(sampling=True, sampling_top_k=64, vocab_size=128)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(spec_verify_epilogue=epilogue)
        runner._publish_graph_variant(backend)
        self.assertEqual(backend._cg_metadata_key(4), (4, 64))
        wrappers = [object()]
        backend.decode_cuda_graph_metadata = {(4, 64): wrappers}
        self.assertIs(
            backend.get_cuda_graph_decode_wrappers(bs=4, num_tokens=4), wrappers
        )

        # Greedy and sampling at one batch size must not collide: that collision
        # is what let a later capture overwrite an earlier graph's wrappers.
        epilogue.sampling = False
        runner._publish_graph_variant(backend)
        self.assertNotEqual(backend._cg_metadata_key(4), (4, 64))
        self.assertEqual(backend._cg_metadata_key(4), (4, 0))

    def test_graph_variant_reaches_a_wrapped_backends_child(self):
        """A wrapper owns no captured metadata; its child does.

        HybridAttnBackend forwards init_forward_metadata_out_graph to whichever
        child the forward mode selects, and that child holds
        decode_cuda_graph_metadata. Publishing the variant only onto the wrapper
        leaves every graph sharing the (bs, 0) key inside the child, which is
        the collision this patch exists to remove -- so the fix would silently
        do nothing for exactly the models that use a hybrid backend.
        """
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        prefill = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        decode = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        wrapper = HybridAttnBackend.__new__(HybridAttnBackend)
        wrapper.prefill_backend = prefill
        wrapper.decode_backend = decode

        epilogue = NS(sampling=True, sampling_top_k=64, vocab_size=128)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(spec_verify_epilogue=epilogue)
        runner._publish_graph_variant(wrapper)

        self.assertEqual(wrapper.cuda_graph_variant, 64)
        for child in (prefill, decode):
            self.assertEqual(child._cg_metadata_key(4), (4, 64))

        # And the children follow the wrapper back to greedy, so the two graphs
        # cannot land on one key.
        epilogue.sampling = False
        runner._publish_graph_variant(wrapper)
        for child in (prefill, decode):
            self.assertEqual(child._cg_metadata_key(4), (4, 0))

    def test_graph_variant_targets_are_resolved_once(self):
        """The traversal must not run on every replay.

        _publish_graph_variant sits on the replay path (load_batch), i.e. every
        decode step, while the backend topology is built during initialization
        and never changes. Walking vars() each time is pure waste, so targets
        are resolved once per root and cached.
        """
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        child = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        wrapper = HybridAttnBackend.__new__(HybridAttnBackend)
        wrapper.decode_backend = child

        epilogue = NS(sampling=True, sampling_top_k=64, vocab_size=128)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(spec_verify_epilogue=epilogue)

        calls = []
        real_vars = builtins.vars

        def counting_vars(obj):
            calls.append(obj)
            return real_vars(obj)

        with patch.object(builtins, "vars", counting_vars):
            runner._publish_graph_variant(wrapper)
            first = len(calls)
            self.assertGreater(first, 0, "the first publication must walk the tree")
            for _ in range(20):
                runner._publish_graph_variant(wrapper)
            self.assertEqual(
                len(calls), first, "replays must reuse the resolved targets"
            )

        # Cached or not, the variant still reaches the child and still changes.
        self.assertEqual(child._cg_metadata_key(4), (4, 64))
        epilogue.sampling = False
        runner._publish_graph_variant(wrapper)
        self.assertEqual(child._cg_metadata_key(4), (4, 0))

        # A different root resolves its own targets rather than reusing these.
        other_child = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        other = HybridAttnBackend.__new__(HybridAttnBackend)
        other.decode_backend = other_child
        epilogue.sampling = True
        runner._publish_graph_variant(other)
        self.assertEqual(other_child._cg_metadata_key(4), (4, 64))

    def test_graph_variant_publication_survives_a_backend_cycle(self):
        """Children are found by type, so a self-referencing graph must terminate."""
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferAttnBackend,
        )
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        child = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        wrapper = HybridAttnBackend.__new__(HybridAttnBackend)
        wrapper.decode_backend = child
        child.parent_backend = wrapper  # a cycle; traversal must not hang
        wrapper.siblings = [child]  # and containers are walked too

        epilogue = NS(sampling=True, sampling_top_k=64, vocab_size=128)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(spec_verify_epilogue=epilogue)
        runner._publish_graph_variant(wrapper)
        self.assertEqual(child._cg_metadata_key(4), (4, 64))

    def test_graph_variants_and_unsupported_adjustments(self):
        epilogue = NS(sampling=False, sampling_top_k=128, vocab_size=128)
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(spec_verify_epilogue=epilogue)
        runner.ragged_verify_mode = False
        greedy_key = runner._make_graph_key(
            16, variant_label="lora", attention_variant="dense"
        )
        with patch.object(runner, "_capture_one_shape") as capture:
            keys = []
            capture.side_effect = lambda *args: keys.append(runner._make_graph_key(16))
            runner.capture_one_shape(1, None)
        self.assertEqual([key.spec_sampling_top_k for key in keys], [128, 64, 0])
        self.assertFalse(epilogue.sampling)
        epilogue.sampling = True
        epilogue.sampling_top_k = 64
        self.assertNotEqual(
            greedy_key,
            runner._make_graph_key(16, variant_label="lora", attention_variant="dense"),
        )
        with (
            patch.object(
                runner, "_capture_one_shape", side_effect=RuntimeError("capture failed")
            ),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            runner.capture_one_shape(1, None)
        self.assertFalse(epilogue.sampling)

        info = NS(has_custom_logit_processor=False)
        self.assertTrue(verify_logits_adjustments_are_noop(info))
        for field in (
            "acc_additive_penalties",
            "acc_scaling_penalties",
            "logit_bias",
            "grammar_mask",
        ):
            with self.subTest(field=field):
                setattr(info, field, torch.ones(1))
                self.assertFalse(verify_logits_adjustments_are_noop(info))
                self.assertEqual(
                    verify_logits_adjustments_are_noop(info, allow_grammar=True),
                    field == "grammar_mask",
                )
                setattr(info, field, None)

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_compact_graph_variants_share_layout(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = True
        runner.max_bs = 1
        runner.captured_req_width = 8
        runner.capture_num_tokens = list(range(1, 9))
        runner.device = "cuda"
        runner._captured_ragged_layouts = {}
        layout = runner._capture_ragged_verify_layout(8)
        graphs = []
        outputs = [torch.empty(3, dtype=torch.int32, device="cuda") for _ in range(3)]
        for output in outputs:
            self.assertIs(runner._capture_ragged_verify_layout(8), layout)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output[:1].copy_(layout.verify_lens)
                output[1:].copy_(layout.qo_indptr_device)
            graphs.append(graph)
        self.assertIsNot(runner._capture_ragged_verify_layout(4), layout)
        for length in (8, 2, 7):
            live = NS(
                bs=1,
                cap=8,
                verify_lens=torch.tensor([length], dtype=torch.int32, device="cuda"),
                qo_indptr_device=torch.tensor(
                    [0, length], dtype=torch.int32, device="cuda"
                ),
            )
            runner._stage_ragged_verify_layout(live, 8)
            for graph, output in zip(graphs, outputs, strict=True):
                graph.replay()
                self.assertEqual(output.tolist(), [length, 0, length])

    def test_compact_graph_buckets(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.model_runner = NS(
            spec_verify_epilogue=None,
            spec_algorithm=NS(is_dspark=lambda: True),
            server_args=NS(linear_attn_verify_backend="nv_cutedsl"),
        )
        get_exec = patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.get_exec",
            return_value=NS(mamba=runner.model_runner.server_args),
        )
        get_exec.start()
        self.addCleanup(get_exec.stop)
        runner.ragged_verify_mode = True
        runner.capture_bs = [1]
        runner.max_bs = 1
        for width in (5, 8, 16):
            runner.captured_req_width = width
            runner.capture_num_tokens = runner._build_ragged_verify_token_buckets()
            expected = sorted(set(range(1, min(width, 8) + 1)) | {width})
            self.assertEqual(runner.capture_num_tokens, expected)
            with patch.object(runner, "_capture_one_shape") as capture:
                runner.capture_one_shape(1, None)
            self.assertEqual(
                [call.kwargs["num_tokens"] for call in capture.call_args_list],
                list(reversed(expected)),
            )
            for tokens in expected:
                self.assertEqual(runner._ragged_capture_slots(tokens), 1)
        runner.capture_bs = [1, 2, 4]
        expected = sorted(
            {bs * width for bs in runner.capture_bs for width in (*range(1, 9), 16)}
        )
        self.assertEqual(runner._build_ragged_verify_token_buckets(), expected)
        with patch(
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner.envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get",
            return_value=True,
        ):
            self.assertEqual(runner._build_ragged_verify_token_buckets(), [16, 32, 64])
        runner.capture_num_tokens = expected
        with patch.object(runner, "_capture_one_shape") as capture:
            for bs in reversed(runner.capture_bs):
                runner.capture_one_shape(bs, None)
        self.assertEqual(
            [call.kwargs["num_tokens"] for call in capture.call_args_list],
            list(reversed(expected)),
        )
        runner.capture_bs = [1]
        runner.model_runner.server_args.linear_attn_verify_backend = "triton"
        self.assertEqual(runner._build_ragged_verify_token_buckets(), [16])

        epilogue = NS(sampling=False, sampling_top_k=129, vocab_size=129)
        runner.model_runner.spec_verify_epilogue = epilogue
        runner.capture_num_tokens = list(range(1, 9))
        keys = []
        with patch.object(runner, "_capture_one_shape") as capture:
            capture.side_effect = lambda *args, **kwargs: keys.append(
                (
                    kwargs["num_tokens"],
                    runner._make_graph_key(kwargs["num_tokens"]).spec_sampling_top_k,
                )
            )
            runner.capture_one_shape(1, None)
        self.assertEqual(keys, [(n, k) for k in (129, 64, 0) for n in range(8, 0, -1)])
        self.assertFalse(epilogue.sampling)

    def test_compact_cutedsl_dispatch(self):
        backend = NS(
            kernel_dispatcher=NS(verify_backend=NS(is_nv_cutedsl=lambda: True))
        )
        args = dict(
            layer=NS(
                bias=None,
                lower_bound=-5.0,
                num_q_heads=1,
                num_k_heads=1,
                num_v_heads=1,
                head_q_dim=128,
                head_k_dim=128,
                head_v_dim=128,
                q_dim=128,
                k_dim=128,
                v_dim=128,
                conv_weights=torch.empty(384, 4),
            ),
            mixed_qkv=torch.empty(3, 384, dtype=torch.bfloat16),
            a=torch.empty(1, 3, 1, 128, dtype=torch.bfloat16),
            b=torch.empty(1, 3, 1, dtype=torch.bfloat16),
            draft_token_num=16,
            ragged_layout=NS(bs=1),
            conv_states=torch.empty(2, 3, 384),
            ssm_states=torch.empty(2, 1, 128, 128),
            intermediate_state_cache=None,
            intermediate_conv_window_cache=torch.empty(1, 16, 3, 384),
            retrieve_parent_token=None,
            replayssm_rawv=torch.empty(2, 1, 32, 128),
        )

        def eligible():
            return KDAAttnBackend._can_run_dspark_cutedsl_mtp(backend, **args)

        module = "sglang.srt.layers.attention.linear.kda_backend"
        with (
            patch(f"{module}.is_cuda", return_value=True),
            patch(f"{module}.importlib.util.find_spec", return_value=object()),
            patch("torch.cuda.get_device_capability", return_value=(10, 3)),
        ):
            for tokens in (1, 2, 3, 5, 8):
                args["mixed_qkv"] = torch.empty(tokens, 384, dtype=torch.bfloat16)
                self.assertTrue(eligible())
            for tokens in (9, 16):
                args["mixed_qkv"] = torch.empty(tokens, 384, dtype=torch.bfloat16)
                self.assertFalse(eligible())
            args["mixed_qkv"] = torch.empty(3, 384, dtype=torch.bfloat16)
            args["ragged_layout"] = NS(bs=2)
            self.assertTrue(eligible())
            args["draft_token_num"] = 8
            args["mixed_qkv"] = torch.empty(16, 384, dtype=torch.bfloat16)
            self.assertTrue(eligible())
            args["draft_token_num"] = 16
            args["ragged_layout"] = NS(bs=1)
            args["retrieve_parent_token"] = torch.empty(1)
            self.assertFalse(eligible())
            args["retrieve_parent_token"] = None
            args["ragged_layout"] = None
            self.assertFalse(eligible())
            args["draft_token_num"] = 3
            self.assertTrue(eligible())

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_cutedsl_packed_replay(self):
        for bs in (1, 2, 3):
            with self.subTest(bs=bs):
                self._cutedsl_packed_replay(bs)

    def _cutedsl_packed_replay(self, bs):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires Blackwell")
        torch.manual_seed(42)
        width = 8
        layer = NS(
            num_v_heads=1,
            head_q_dim=128,
            head_k_dim=128,
            head_v_dim=128,
            q_dim=128,
            k_dim=128,
            v_dim=128,
            conv_weights=torch.randn(384, 4, device="cuda") * 0.1,
            A_log=torch.randn(1, device="cuda"),
            dt_bias=torch.randn(128, device="cuda"),
            lower_bound=-5.0,
        )
        args = dict(
            layer=layer,
            mixed_qkv=torch.randn(bs * width, 384, device="cuda", dtype=torch.bfloat16),
            a=torch.randn(1, bs * width, 1, 128, device="cuda", dtype=torch.bfloat16),
            b=torch.randn(1, bs * width, 1, device="cuda", dtype=torch.bfloat16),
            conv_states=torch.randn(
                bs + 1, 3, 384, device="cuda", dtype=torch.bfloat16
            ),
            ssm_states=torch.randn(bs + 1, 1, 128, 128, device="cuda"),
            intermediate_state_cache=None,
            intermediate_conv_window_cache=torch.zeros(
                bs, 16, 3, 384, device="cuda", dtype=torch.bfloat16
            ),
            intermediate_state_indices=torch.arange(
                bs, device="cuda", dtype=torch.int32
            ),
            cache_indices=torch.arange(1, bs + 1, device="cuda", dtype=torch.int32),
            query_start_loc=torch.arange(bs + 1, device="cuda", dtype=torch.int32)
            * width,
            max_verify_tokens=width,
            replayssm_rawv=torch.zeros(
                bs + 1, 1, 32, 128, device="cuda", dtype=torch.bfloat16
            ),
            replayssm_rawk=torch.zeros(
                bs + 1, 1, 32, 128, device="cuda", dtype=torch.bfloat16
            ),
            replayssm_g=torch.zeros(bs + 1, 1, 32, 128, device="cuda"),
            replayssm_beta=torch.zeros(bs + 1, 1, 32, device="cuda"),
        )
        run = KDAAttnBackend._run_dspark_cutedsl_mtp
        if bs > 1:
            layer._k3_onorm_gate = torch.randn(
                bs * width, 128, device="cuda", dtype=torch.bfloat16
            )
            layer._k3_fused_decode_args = (None,) * 5 + (
                torch.ones(128, device="cuda"),
                1e-6,
            )
        persistent = {
            name: args[name].clone() for name in ("conv_states", "ssm_states")
        }
        run(**args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(**args)
        for step in range(9):
            lengths = [(step + row * 3) % 9 for row in range(bs)]
            with self.subTest(lengths=lengths):
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1] + length)
                args["query_start_loc"].copy_(
                    torch.tensor(offsets, device="cuda", dtype=torch.int32)
                )
                # A nonempty padding slot must not zero the following request.
                args["cache_indices"][0] = -1 if step == 4 else 1
                graph.replay()
                torch.testing.assert_close(
                    output[:, offsets[-1] :], torch.zeros_like(output[:, offsets[-1] :])
                )
                for row, length in enumerate(lengths):
                    if not length:
                        continue
                    start, end = offsets[row : row + 2]
                    if row == 0 and step == 4:
                        torch.testing.assert_close(
                            output[:, start:end], torch.zeros_like(output[:, start:end])
                        )
                        continue
                    self._compare_cutedsl_row(
                        args, output[:, start:end], row, start, end
                    )
                for name, initial in persistent.items():
                    torch.testing.assert_close(args[name], initial)

    def _compare_cutedsl_row(self, args, output, row, start, end):
        reference_args = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in args.items()
        }
        length = end - start
        if hasattr(args["layer"], "_k3_onorm_gate"):
            reference_args["layer"] = NS(**vars(args["layer"]))
            reference_args["layer"]._k3_onorm_gate = (
                args["layer"]._k3_onorm_gate[start:end].clone()
            )
        reference_args["max_verify_tokens"] = None
        reference_args["mixed_qkv"] = args["mixed_qkv"][start:end].clone()
        reference_args["a"] = args["a"][:, start:end].clone()
        reference_args["b"] = args["b"][:, start:end].clone()
        reference_args["cache_indices"] = args["cache_indices"][row : row + 1].clone()
        reference_args["intermediate_state_indices"] = args[
            "intermediate_state_indices"
        ][row : row + 1].clone()
        reference_args["query_start_loc"] = torch.tensor(
            [0, length], device="cuda", dtype=torch.int32
        )
        reference = KDAAttnBackend._run_dspark_cutedsl_mtp(**reference_args)
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(
            args["intermediate_conv_window_cache"][row, :length],
            reference_args["intermediate_conv_window_cache"][row, :length],
        )
        for name in (
            "replayssm_rawv",
            "replayssm_rawk",
            "replayssm_g",
            "replayssm_beta",
        ):
            torch.testing.assert_close(
                args[name][row + 1, :, :length],
                reference_args[name][row + 1, :, :length],
            )

    @skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_static_sampling_and_xgrammar_replay(self):
        for stride, vocab, top_k in ((5, 128, 64), (16, 163840, 163840)):
            with self.subTest(stride=stride, vocab=vocab, top_k=top_k):
                self._static_sampling_and_xgrammar_replay(stride, vocab, top_k)

        self._static_sampling_and_xgrammar_replay(5, 128, 64, commit_hidden=False)

    def _static_sampling_and_xgrammar_replay(
        self, stride, vocab, top_k, commit_hidden=True
    ):
        import xgrammar as xgr
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.model_executor.runner_utils.capture_owner import (
            collect_full_cuda_graph_owners,
        )
        from sglang.srt.speculative.dflash_utils import (
            _get_or_create_chain_verify_buffers,
        )
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            accept_draft_tokens,
        )

        bs = 2
        hidden = torch.randn(bs * stride, 8, device="cuda")
        committed = torch.zeros_like(hidden)
        pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        epilogue = DSparkStaticVerifyEpilogue(
            tp_sync=SpecTpSync(NS(world_size=1)),
            max_bs=bs,
            verify_num_draft_tokens=stride,
            vocab_size=vocab,
            device="cuda",
            commit_ctx=NS(
                resolve_pool=lambda: pool,
                draft_model=NS(
                    write_target_hidden_kv=lambda **kw: committed.copy_(
                        kw["target_hidden"]
                    )
                ),
            ),
        )
        if not commit_hidden:
            epilogue.commit_ctx = None
        self.assertEqual(epilogue.folds_commit, commit_hidden)
        logits = torch.randn(bs * stride, vocab, device="cuda")
        candidates = torch.randint(vocab, (bs, stride), device="cuda")
        seq_lens = torch.tensor([40, 80], device="cuda")
        window = NS(
            verify_cache_loc_2d=torch.arange(bs * stride, device="cuda").view(
                bs, stride
            ),
            positions_2d=seq_lens[:, None] + torch.arange(stride, device="cuda"),
        )
        info = NS(
            is_all_greedy=False,
            is_any_greedy=True,
            temperatures=torch.tensor([[1.0], [0.7]], device="cuda"),
            top_ks=torch.tensor([1, 50], dtype=torch.int32, device="cuda"),
            top_ps=torch.tensor([1.0, 0.95], device="cuda"),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=True,
            min_ps=torch.tensor([0.0, 0.3], device="cuda"),
        )
        draft = NS(
            temperatures=info.temperatures.flatten(),
            greedy_mask=torch.tensor([True, False], device="cuda"),
            corrected_logits=torch.randn(bs, stride - 1, vocab, device="cuda"),
        )
        epilogue.prepare(window, bs=bs)
        epilogue.stage_sampling(
            bs=bs,
            sampling_info=info,
            draft_block=draft,
            grammar_mask=None,
            max_top_k=top_k,
        )
        self.assertEqual(epilogue.sampling_top_k, top_k)
        epilogue.draft_tokens_buf.copy_(candidates[:, 1:].reshape(-1))
        runner = NS(model_runner=NS(is_draft_worker=False), ragged_verify_mode=False)
        batch = NS(input_ids=candidates.flatten(), seq_lens=seq_lens, batch_size=bs)

        def verify():
            epilogue.capture_hook(
                runner,
                LogitsProcessorOutput(
                    next_token_logits=logits.clone(), hidden_states=hidden.clone()
                ),
                batch,
                bs * stride,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                verify()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with collect_full_cuda_graph_owners() as owners, torch.cuda.graph(graph):
            verify()
        self.assertTrue(owners)
        # Growing eager scratch must not invalidate pointers retained by the graph.
        _get_or_create_chain_verify_buffers(
            bs=32, draft_token_num=32, device=torch.device("cuda")
        )
        for seed in (123, 456):
            torch.cuda.manual_seed(seed)
            graph.replay()
            actual = [t.clone() for t in (epilogue.correct_len_buf, epilogue.bonus_buf)]
            torch.cuda.manual_seed(seed)
            expected = accept_draft_tokens(
                candidates=candidates,
                target_logits=logits,
                draft_block=draft,
                sampling_info=info,
                draft_input=NS(max_top_k=top_k, uniform_top_k_value=None),
                gamma=stride - 1,
                verify_num_draft_tokens=stride,
                cutoff_layout=None,
            )
            for got, want in zip(actual, expected[:2], strict=True):
                torch.testing.assert_close(got, want.to(got.dtype), rtol=0, atol=0)
            torch.testing.assert_close(
                committed, hidden if commit_hidden else torch.zeros_like(hidden)
            )

        # Point-mass proposals exercise full acceptance and rejection at position zero.
        candidates[:, 1:] = torch.arange(1, stride, device="cuda")
        epilogue.draft_tokens_buf.copy_(candidates[:, 1:].reshape(-1))
        draft.greedy_mask.fill_(False)
        draft.corrected_logits.fill_(-float("inf"))
        draft.corrected_logits.scatter_(2, candidates[:, 1:, None], 0)
        logits.fill_(-float("inf"))
        predicted = torch.cat(
            (candidates[:, 1:], torch.full((bs, 1), 7, device="cuda")), dim=1
        )
        logits.scatter_(1, predicted.reshape(-1, 1), 0)
        epilogue.stage_sampling(
            bs=bs,
            sampling_info=info,
            draft_block=draft,
            grammar_mask=None,
            max_top_k=top_k,
        )
        graph.replay()
        self.assertTrue((epilogue.correct_len_buf == stride - 1).all())
        self.assertTrue((epilogue.bonus_buf == 7).all())
        logits.view(bs, stride, vocab)[:, 0].fill_(-float("inf"))
        logits.view(bs, stride, vocab)[:, 0, 9] = 0
        graph.replay()
        self.assertTrue((epilogue.correct_len_buf == 0).all())
        self.assertTrue((epilogue.bonus_buf == 9).all())
        logits.normal_()
        draft.corrected_logits.normal_()

        tokens = [f"t{i}".encode() for i in range(vocab)]
        tokens[7] = b"a"
        tokenizer = xgr.TokenizerInfo(tokens, stop_token_ids=[127])
        compiled = xgr.GrammarCompiler(tokenizer).compile_grammar(
            xgr.Grammar.from_regex("a")
        )
        mask = xgr.allocate_token_bitmask(bs * stride, vocab)
        for row in range(bs * stride):
            xgr.GrammarMatcher(compiled).fill_next_token_bitmask(mask, row)
        epilogue.prepare(window, bs=bs)
        epilogue.stage_sampling(
            bs=bs,
            sampling_info=info,
            draft_block=draft,
            grammar_mask=NS(vocab_mask=mask.cuda()),
            max_top_k=top_k,
        )
        graph.replay()
        self.assertTrue((epilogue.bonus_buf == 7).all())
        epilogue.prepare(window, bs=bs)
        epilogue.stage_sampling(
            bs=bs,
            sampling_info=info,
            draft_block=draft,
            grammar_mask=None,
            max_top_k=top_k,
        )
        self.assertTrue((epilogue.sampling_buffers.vocab_mask == -1).all())
        graph.replay()
        torch.testing.assert_close(
            committed, hidden if commit_hidden else torch.zeros_like(hidden)
        )

        epilogue.prepare(
            NS(
                verify_cache_loc_2d=window.verify_cache_loc_2d[:1],
                positions_2d=window.positions_2d[:1],
            ),
            bs=1,
        )
        epilogue.stage_sampling(
            bs=1,
            sampling_info=NS(
                is_all_greedy=False, top_ks=info.top_ks[:1], top_ps=info.top_ps[:1]
            ),
            draft_block=NS(
                temperatures=draft.temperatures[:1],
                greedy_mask=draft.greedy_mask[:1],
                corrected_logits=draft.corrected_logits[:1],
            ),
            grammar_mask=None,
            max_top_k=top_k,
        )
        graph.replay()
        self.assertEqual(epilogue.correct_len_buf[1].item(), 0)
        self.assertEqual(epilogue.verify_lens_buf.tolist(), [stride, 0])


if __name__ == "__main__":
    main()
