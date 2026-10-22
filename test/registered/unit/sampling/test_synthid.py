"""Algorithm 5 goldens, request-state lifecycle and ordinary/speculative consumers."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import (
    OutputLogprobProcessor,
    compute_spec_logprobs,
)
from sglang.srt.layers.sampler import Sampler, sampling_from_probs_torch
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.proposal_relay import RequestProposalRelay
from sglang.srt.sampling.watermarking.synthid_request import (
    apply_mixed_logits,
    is_enabled,
    is_mixed,
)
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.watermarking.synthid import (
    SynthIDConfig,
    SynthIDSparseOutput,
    SynthIDState,
    SynthIDWatermarkProcessor,
    compute_context_hash_torch,
    compute_g_values_torch,
    compute_synthid_key_iv,
    watermark_topk_scores_torch,
)
from sglang.srt.watermarking.synthid.speculative import (
    apply_synthid_to_draft_logits,
    apply_synthid_to_target,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _case(output_ids=((),), *, config=None, enabled=None, generations=None):
    config = config or SynthIDConfig(
        keys=(1, 2), ngram_len=3, top_k=3, context_history_size=4
    )
    count = len(output_ids)
    state = SynthIDState.from_output_ids(
        output_ids,
        config=config,
        device="cpu",
        slot_indices=torch.arange(1, count + 1),
        generations=torch.tensor(generations or [1] * count),
    )
    if enabled is not None:
        state.request_enabled_host = enabled
        state.request_enabled = torch.tensor(enabled)
        state.commit_mask &= state.request_enabled
    return SynthIDWatermarkProcessor(config, "cpu", count + 1), state


def _info(state, *, vocab=4, **overrides):
    count = len(state)
    values = dict(
        temperatures=torch.ones(count, 1),
        top_ks=torch.full((count,), TOP_K_ALL),
        top_ps=torch.ones(count),
        min_ps=torch.zeros(count),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        vocab_size=vocab,
        device="cpu",
        synthid_state=state,
        return_sampling_masks=[False] * count,
    )
    values.update(overrides)
    masks = values["return_sampling_masks"]
    values["sampling_mask_batch_indices"] = (
        SamplingBatchInfo._make_sampling_mask_batch_indices(masks, "cpu")
    )
    values["sampling_mask_top_ks"] = [
        values["top_ks"][i].item() if requested else 1
        for i, requested in enumerate(masks)
    ]
    return SamplingBatchInfo(**values)


def _sampler(processor, *, ordinary=True, fused=False):
    sampler = Sampler.__new__(Sampler)
    sampler.synthid_watermark = processor
    sampler.use_ordinary_synthid_sampling = ordinary
    sampler.use_fused_synthid_sampling = fused
    sampler.rl_on_policy_target = None
    sampler.use_ascend_backend = False
    sampler.use_log_softmax_logprob = False
    sampler.enable_deterministic = False
    sampler.output_logprob_processor = OutputLogprobProcessor()
    sampler.sampling_mask_max_tokens = 4096
    sampler.cp_sync_group = None
    sampler._sync_token_ids_across_tp = Mock()
    return sampler


def _sample(sampler, info, logits, *, logprobs=True):
    output = SimpleNamespace(next_token_logits=logits.clone())
    rows, vocab = logits.shape
    tokens = sampler.forward(
        output,
        info,
        logprobs,
        [vocab] * rows,
        [list(range(vocab)) for _ in range(rows)],
        torch.zeros(rows, dtype=torch.int64),
    )
    return tokens, output


class TestSynthID(CustomTestCase):
    def setUp(self):
        overrides = ExitStack()
        self.addCleanup(overrides.close)
        overrides.enter_context(
            get_context().override_server_args(
                sampling_backend="pytorch",
                speculative_use_rejection_sampling=True,
            )
        )

    def test_external_hash_and_algorithm5_goldens(self):
        keys = torch.tensor([1, 2])
        iv = compute_synthid_key_iv([1, 2])
        self.assertEqual(iv, 5312871951099227430)
        hashes = compute_context_hash_torch(torch.tensor([[0, 0], [0, 2]]), iv)
        self.assertEqual(hashes.tolist(), [2393652284050965700, -3324819341964999906])
        self.assertEqual(
            compute_g_values_torch(
                hashes, torch.tensor([[0, 1, 2]] * 2), keys
            ).tolist(),
            [[[1, 0], [0, 0], [1, 0]], [[1, 0], [1, 1], [0, 0]]],
        )
        transformed = watermark_topk_scores_torch(
            torch.tensor([[1.0, 0.0, -0.5]], dtype=torch.float64),
            torch.tensor([[1, 0, 2]]),
            hashes[1:],
            keys,
            torch.tensor([True]),
        )
        torch.testing.assert_close(
            transformed,
            torch.tensor(
                [[-0.08367551842228967, -2.594303613204244, -5.189915007942694]],
                dtype=torch.float64,
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            SynthIDConfig(keys=(1, 1))

    def test_repeat_skip_and_bounded_history_projection(self):
        for skip in (False, True):
            config = SynthIDConfig(
                keys=(1, 2),
                ngram_len=3,
                top_k=3,
                context_history_size=4,
                skip_first_ngram_calls=skip,
            )
            processor, state = _case(config=config)
            pool = processor.state_pool
            for step, token in enumerate((7, 8, 7, 8)):
                _, mask = pool.current_hashes_and_mask(state)
                self.assertEqual(mask.item(), not skip or step >= 2)
                pool.commit(state, torch.tensor([token]))
            self.assertFalse(pool.current_hashes_and_mask(state)[1].item())
        processor, state = _case(([7],))
        hashes, mask = processor.state_pool.project_linear_speculative(
            state, torch.tensor([[7, 8, 9]])
        )
        expected = compute_context_hash_torch(
            torch.tensor([[0, 7], [7, 8], [8, 9]]), processor.config.hash_iv
        )
        torch.testing.assert_close(hashes, expected)
        self.assertEqual(mask.tolist(), [True] * 3)
        processor.commit(state, torch.tensor([[8, 9, 0]]), torch.tensor([2]))
        self.assertEqual(processor.state_pool.context[1].tolist(), [8, 9])
        self.assertEqual(processor.state_pool.num_calls[1].item(), 3)
        config = SynthIDConfig(
            keys=(1, 2), ngram_len=3, top_k=3, context_history_size=1
        )
        processor, state = _case(config=config)
        pool = processor.state_pool
        pool.initialize_pending(state)
        pool.context[1] = torch.tensor([1, 2])
        pool.context_history[1, 0] = compute_context_hash_torch(
            torch.tensor([[2, 3]]), config.hash_iv
        )[0]
        self.assertEqual(
            pool.project_linear_speculative(state, torch.tensor([[99, 3]]))[1].tolist(),
            [True, True],
        )

    def test_reused_slot_and_cache_flush_do_not_inherit_history(self):
        processor, state = _case(([1],))
        processor.commit(state, torch.tensor([2]))
        _, reused = _case(generations=[2])
        processor.state_pool.current_hashes_and_mask(reused)
        self.assertEqual(processor.state_pool.context[1].tolist(), [0, 0])
        self.assertEqual(processor.state_pool.generations[1].item(), 2)
        processor.commit(reused, torch.tensor([7]))
        pool = ReqToTokenPool.__new__(ReqToTokenPool)
        pool._alloc_size, pool.free_slots = 2, []
        pool.req_generation = torch.tensor([0, 2])
        pool._aux_cache = None
        pool.clear()
        processor.invalidate_on_cache_flush()
        self.assertEqual(pool.alloc_rows(1), [1])
        self.assertEqual(pool.req_generation[1].item(), 3)
        _, fresh = _case(generations=[3])
        processor.state_pool.current_hashes_and_mask(fresh)
        self.assertEqual(processor.state_pool.context[1].tolist(), [0, 0])
        self.assertEqual(processor.state_pool.num_calls[1].item(), 0)

    def test_chunked_prefill_policy_and_row_lifecycle(self):
        requests = [SimpleNamespace(output_ids=[7]), SimpleNamespace(output_ids=[8, 9])]
        config = SynthIDConfig(
            keys=(1, 2), ngram_len=3, top_k=2, context_history_size=4
        )
        state = SynthIDState.from_requests(
            reqs=requests,
            config=config,
            device="cpu",
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor([1, 2], dtype=torch.int32),
            req_generation=torch.tensor([0, 3, 7]),
            chunked_req=requests[0],
            pin_memory=False,
        )
        self.assertEqual(state.generations.tolist(), [3, 7])
        self.assertEqual(state.commit_mask.tolist(), [False, True])
        processor = SynthIDWatermarkProcessor(config, "cpu", num_slots=3)
        processor.commit(state, torch.tensor([11, 12]))
        self.assertEqual(processor.state_pool.num_calls[[1, 2]].tolist(), [1, 3])
        processor, state = _case(([], []), enabled=[True, False])
        state.filter_batch(torch.tensor([1]), [1])
        self.assertFalse(is_enabled(state))
        _, on = _case(enabled=[True])
        state.merge_batch(on)
        self.assertEqual(state.request_enabled_host, [False, True])
        self.assertTrue(is_mixed(state))

    def test_mixed_target_and_draft_keep_off_full_vocabulary(self):
        for draft, repeats in ((True, 1), (False, 3)):
            with self.subTest(draft=draft):
                processor, state = _case(([], []), enabled=[True, False])
                raw = torch.tensor([[4.0, 3.0, 2.0, 1.0]] * (2 * repeats))
                temperatures = torch.tensor([[0.5], [2.0]])
                actual, expected = raw.clone(), raw.clone()
                kwargs = (
                    {} if draft else {"verify_tokens": torch.tensor([[0, 1, 2]] * 2)}
                )
                processor.apply(expected, temperatures, state, **kwargs)
                apply_mixed_logits(
                    processor=processor,
                    logits=actual,
                    temperatures=temperatures,
                    state=state,
                    draft=draft,
                    **kwargs,
                )
                torch.testing.assert_close(actual[:repeats], expected[:repeats])
                torch.testing.assert_close(actual[repeats:], raw[repeats:] / 2)
                self.assertTrue(torch.isfinite(actual[repeats:]).all())
                processor.commit(state, torch.tensor([0, 3]))
                self.assertEqual(
                    processor.state_pool.num_calls[[1, 2]].tolist(), [1, 0]
                )

    def test_speculative_filters_apply_before_watermark_and_repeat_per_request(self):
        for draft in (False, True):
            for filter_kind in ("top_k", "top_p", "min_p"):
                with self.subTest(draft=draft, filter=filter_kind):
                    processor, state = _case(([], []))
                    kwargs = {
                        "top_k": dict(
                            top_ks=torch.tensor([1, TOP_K_ALL]),
                            need_top_k_sampling=True,
                            is_any_greedy=True,
                        ),
                        "top_p": dict(
                            top_ps=torch.tensor([0.5, 1.0]), need_top_p_sampling=True
                        ),
                        "min_p": dict(
                            min_ps=torch.tensor([0.5, 0.0]), need_min_p_sampling=True
                        ),
                    }[filter_kind]
                    info = _info(state, vocab=2, **kwargs)
                    repeats = 1 if draft else 2
                    logits = torch.tensor([[0.8, 0.2]] * (2 * repeats)).log()
                    if draft:
                        output, _ = apply_synthid_to_draft_logits(
                            processor=processor,
                            logits=logits,
                            sampling_info=info,
                        )
                    else:
                        output, _, _, _ = apply_synthid_to_target(
                            processor=processor,
                            logits=logits,
                            sampling_info=info,
                            candidates=torch.zeros(2, 2, dtype=torch.int64),
                            tree_topk=1,
                            return_logprob=False,
                        )
                    probs = torch.zeros_like(logits).scatter_(
                        1, output.token_ids, output.scores.softmax(-1)
                    )
                    torch.testing.assert_close(
                        probs[:repeats], torch.tensor([[1.0, 0.0]] * repeats)
                    )
                    self.assertTrue((probs[repeats:] > 0).all())

    def test_ordinary_mixed_sampling_and_logprobs_keep_off_law(self):
        raw = torch.tensor([[4.0, 3.0, 2.0, 1.0]] * 2)
        for mode in ("default", "original", "watermarked"):
            with self.subTest(mode=mode):
                processor, state = _case(([], []), enabled=[True, False])
                info = _info(state, temperatures=torch.tensor([[0.5], [2.0]]))
                expected = raw.clone()
                processor.apply(expected, info.temperatures, state)
                with (
                    patch(
                        "sglang.srt.layers.sampler.SGLANG_RETURN_ORIGINAL_LOGPROB",
                        mode == "original",
                    ),
                    envs.SGLANG_RETURN_WATERMARKED_LOGPROB.override(
                        mode == "watermarked"
                    ),
                    patch(
                        "sglang.srt.layers.sampler.sampling_from_probs_torch",
                        return_value=torch.tensor([0, 3]),
                    ) as sample,
                ):
                    tokens, output = _sample(_sampler(processor), info, raw)
                probs = sample.call_args.args[0]
                torch.testing.assert_close(probs[0], expected[0].softmax(-1))
                torch.testing.assert_close(probs[1], (raw[1] / 2).softmax(-1))
                self.assertEqual(tokens.tolist(), [0, 3])
                reference = raw[1] if mode == "original" else raw[1] / 2
                torch.testing.assert_close(
                    output.next_token_logprobs[1], reference.log_softmax(-1)[3]
                )
                self.assertEqual(
                    processor.state_pool.num_calls[[1, 2]].tolist(), [1, 0]
                )

    def test_mixed_seeded_off_rows_keep_baseline_candidate_noise_pairing(self):
        """An ON neighbour must not change OFF seeded draws or candidate order."""
        for logits, seed, top_k, hashes in (
            ([0.0, -46.0], 1871062016, TOP_K_ALL, [1229680045, 4294967295]),
            ([0.0, 2.0, 1.0], 0, 2, [2167721464, 10027521, 2423355346]),
        ):
            with self.subTest(seed=seed, top_k=top_k):
                outputs = []
                masks = []
                mask_scores = []
                for enabled in ([False, False], [True, False]):
                    processor, state = _case(
                        ([], []),
                        config=SynthIDConfig(keys=(1, 2), top_k=len(logits)),
                        enabled=enabled,
                    )
                    info = _info(
                        state,
                        vocab=len(logits),
                        top_ks=torch.full((2,), top_k),
                        need_top_k_sampling=top_k != TOP_K_ALL,
                        sampling_seed=torch.full((2,), seed),
                        return_sampling_masks=[False, True],
                    )
                    consumer = _sampler(processor)
                    consumer.enable_deterministic = True
                    with patch(
                        "sglang.srt.layers.sampler.murmur_hash32",
                        return_value=torch.tensor([hashes] * 2, dtype=torch.uint32),
                    ):
                        tokens, output = _sample(
                            consumer, info, torch.tensor([logits] * 2), logprobs=False
                        )
                    outputs.append(tokens)
                    captured = output.sampling_mask_output
                    masks.append(captured.token_ids[0, :captured.lengths[0]].tolist())
                    mask_scores.append(captured.selected_logprobs[0].item())
                self.assertEqual(outputs[0][1].item(), outputs[1][1].item())
                self.assertEqual(masks[0], masks[1])
                self.assertEqual(mask_scores[0], mask_scores[1])
                if top_k == TOP_K_ALL:
                    self.assertEqual(outputs[1][0].item(), 0)

    def test_mixed_off_logprobs_do_not_depend_on_neighbour_policy(self):
        """Reporting flags and a greedy OFF row must retain ordinary scores."""
        for mode in ("default", "original", "watermarked"):
            for greedy in (False, True, "all"):
                for deterministic in (False, True):
                    with self.subTest(
                        mode=mode, greedy=greedy, deterministic=deterministic
                    ):
                        scores = []
                        for enabled in ([False, False], [True, False]):
                            processor, state = _case(([], []), enabled=enabled)
                            info = _info(
                                state,
                                vocab=3,
                                temperatures=torch.tensor([[1.0], [0.5]]),
                                top_ks=torch.tensor(
                                    [1 if greedy == "all" else 2, 1 if greedy else 2]
                                ),
                                need_top_k_sampling=True,
                                is_any_greedy=bool(greedy),
                                is_all_greedy=greedy == "all",
                            )
                            consumer = _sampler(processor)
                            consumer.enable_deterministic = deterministic
                            with (
                                patch(
                                    "sglang.srt.layers.sampler.SGLANG_RETURN_ORIGINAL_LOGPROB",
                                    mode == "original",
                                ),
                                envs.SGLANG_RETURN_WATERMARKED_LOGPROB.override(
                                    mode == "watermarked"
                                ),
                            ):
                                _, output = _sample(
                                    consumer, info, torch.tensor([[1.0, 0.0, -1.0]] * 2)
                                )
                            scores.append(output.next_token_token_ids_logprobs_val[1])
                        torch.testing.assert_close(scores[1], scores[0], rtol=0, atol=0)

    def test_mixed_speculative_off_rows_preserve_target_and_draft_filter_laws(self):
        """OFF target filters stay sequential; OFF rejection drafts stay unfiltered."""
        for draft in (False, True):
            for kind in ("joint", "min_p", "greedy"):
                with self.subTest(draft=draft, kind=kind):
                    processor, state = _case(([], []), enabled=[True, False])
                    probabilities = (
                        [0.8, 0.2, 1e-30] if kind == "min_p" else [0.5, 0.3, 0.2]
                    )
                    info = _info(
                        state,
                        vocab=3,
                        top_ks=torch.tensor(
                            [2, 1 if kind == "greedy" else 2]
                            if kind != "min_p"
                            else [TOP_K_ALL, TOP_K_ALL]
                        ),
                        top_ps=torch.full((2,), 0.55 if kind == "joint" else 1.0),
                        min_ps=torch.full((2,), 0.5 if kind == "min_p" else 0.0),
                        need_top_k_sampling=kind != "min_p",
                        need_top_p_sampling=kind == "joint",
                        need_min_p_sampling=kind == "min_p",
                        is_any_greedy=kind == "greedy",
                    )
                    repeats = 1 if draft else 2
                    logits = torch.tensor([probabilities] * (2 * repeats)).log()
                    if draft:
                        apply_synthid_to_draft_logits(
                            processor=processor, logits=logits, sampling_info=info
                        )
                    else:
                        apply_synthid_to_target(
                            processor=processor,
                            logits=logits,
                            sampling_info=info,
                            candidates=torch.zeros(2, 2, dtype=torch.int64),
                            tree_topk=1,
                            return_logprob=False,
                        )
                    expected = (
                        torch.tensor(probabilities)
                        if draft or kind == "min_p"
                        else torch.tensor([1.0, 0.0, 0.0])
                    )
                    torch.testing.assert_close(
                        logits.softmax(-1)[repeats:], expected.expand(repeats, -1)
                    )

    def test_all_greedy_mixed_draft_preserves_off_argmax_proposal(self):
        from sglang.srt.watermarking.synthid.speculative import (
            sample_synthid_draft_logits,
        )

        processor, state = _case(([], []), enabled=[True, False])
        info = _info(
            state,
            vocab=3,
            temperatures=torch.tensor([[1.0], [2.0]]),
            top_ks=torch.ones(2, dtype=torch.int32),
            need_top_k_sampling=True,
            is_all_greedy=True,
            is_any_greedy=True,
        )
        raw = torch.tensor([[2.0, 1.0, 0.0]] * 2)
        sparse, fused = apply_synthid_to_draft_logits(
            processor=processor, logits=raw, sampling_info=info
        )
        with patch(
            "sglang.srt.speculative.spec_utils.fast_sample",
            return_value=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int64)),
        ):
            probs, _, _, _ = sample_synthid_draft_logits(
                logits=raw,
                sampling_info=info,
                sparse_output=sparse,
                fused_draft_sample=fused,
                use_rejection_sampling=True,
                topk=1,
            )
        torch.testing.assert_close(probs[0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(probs[1], torch.tensor([1.0, 0.0, 0.0]))

    def test_ordinary_filter_order_caps_and_greedy_ties(self):
        for cap, requested, support in (
            (3, 2, 2),
            (3, TOP_K_ALL, 3),
            (16, TOP_K_ALL, 4),
        ):
            with self.subTest(cap=cap, requested=requested):
                config = SynthIDConfig(
                    keys=(1, 2), top_k=cap, skip_first_ngram_calls=True
                )
                processor, state = _case(config=config)
                info = _info(
                    state,
                    top_ks=torch.tensor([requested]),
                    need_top_k_sampling=requested != TOP_K_ALL,
                )
                sampler = _sampler(processor)
                sampler._sample_from_probs = Mock(wraps=sampler._sample_from_probs)
                raw = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
                _sample(sampler, info, raw)
                expected = raw.exp()
                expected[:, support:] = 0
                expected /= expected.sum(-1, keepdim=True)
                torch.testing.assert_close(
                    sampler._sample_from_probs.call_args.args[0], expected
                )
        processor, state = _case()
        info = _info(
            state,
            vocab=3,
            temperatures=torch.tensor([[0.5]]),
            top_ps=torch.tensor([0.6]),
            need_top_p_sampling=True,
        )
        sampler = _sampler(processor)
        sampler._sample_from_probs = Mock(wraps=sampler._sample_from_probs)
        _sample(sampler, info, torch.tensor([[0.5, 0.3, 0.2]]).log())
        torch.testing.assert_close(
            sampler._sample_from_probs.call_args.args[0],
            torch.tensor([[1.0, 0.0, 0.0]]),
        )
        processor, state = _case(([], []), enabled=[True, False])
        info = _info(
            state,
            vocab=8,
            top_ks=torch.ones(2, dtype=torch.int32),
            is_all_greedy=True,
            is_any_greedy=True,
            need_top_k_sampling=True,
        )
        with patch(
            "sglang.srt.layers.sampler.forward_mixed",
            side_effect=AssertionError("greedy entered stochastic dispatch"),
        ):
            tokens, _ = _sample(
                _sampler(processor), info, torch.zeros(2, 8), logprobs=False
            )
        self.assertEqual(tokens.tolist(), [0, 0])

    def test_sampling_masks_and_speculative_logprobs_use_the_correct_law(self):
        processor, state = _case(config=SynthIDConfig(keys=(9,), top_k=3))
        info = _info(
            state,
            vocab=3,
            top_ps=torch.tensor([0.6]),
            need_top_p_sampling=True,
            return_sampling_masks=[True],
        )
        sampler = _sampler(processor)
        sampler._sample_from_probs = Mock(wraps=sampler._sample_from_probs)
        tokens, output = _sample(sampler, info, torch.tensor([[0.5, 0.3, 0.2]]).log())
        captured = output.sampling_mask_output
        self.assertEqual(set(captured.token_ids[0, :captured.lengths[0]].tolist()), {0, 1})
        probs = sampler._sample_from_probs.call_args.args[0]
        self.assertAlmostEqual(
            captured.selected_logprobs[0].item(),
            probs[0, tokens[0]].log().item(),
            places=6,
        )
        raw = torch.tensor([[1.0, 0.0]])
        for original in (False, True):
            output = SimpleNamespace(
                next_token_logits=raw.flip(1), next_token_logits_before_watermark=raw
            )
            batch = SimpleNamespace(
                seq_lens=torch.tensor([1]),
                top_logprobs_nums=[],
                token_ids_logprobs=[],
                sampling_info=SimpleNamespace(
                    synthid_state=state,
                    is_all_greedy=False,
                    temperatures=torch.tensor([[2.0]]),
                ),
            )
            with envs.SGLANG_RETURN_ORIGINAL_LOGPROB.override(original):
                compute_spec_logprobs(
                    batch, output, torch.tensor([0]), accept_index=torch.tensor([[0]])
                )
            torch.testing.assert_close(
                output.next_token_logprobs[0, 0],
                (raw / (1 if original else 2)).log_softmax(-1)[0, 0],
            )

    def test_all_off_speculative_logprobs_do_not_require_watermarked_logits(self):
        """An all-OFF batch must report baseline logprobs without a saved copy."""
        _, state = _case(enabled=[False])
        raw = torch.tensor([[1.0, 0.0, -1.0]])
        batch = SimpleNamespace(
            seq_lens=torch.tensor([1]),
            top_logprobs_nums=[],
            token_ids_logprobs=[],
            sampling_info=_info(state, vocab=3, temperatures=torch.tensor([[2.0]])),
        )
        for original in (False, True):
            with self.subTest(original=original):
                output = SimpleNamespace(
                    next_token_logits=raw,
                    next_token_logits_before_watermark=None,
                )
                with envs.SGLANG_RETURN_ORIGINAL_LOGPROB.override(original):
                    compute_spec_logprobs(
                        batch, output, torch.tensor([1]), accept_index=torch.tensor([[0]])
                    )
                expected = (raw / (1 if original else 2)).log_softmax(-1)[0, 1]
                torch.testing.assert_close(output.next_token_logprobs[0, 0], expected)

    def test_fused_sampling_gate_seed_and_zero_probability_boundary(self):
        for ordinary, fused, seed, expected in (
            (True, True, None, 7),
            (False, False, None, 11),
            (True, True, torch.tensor([42]), 1),
        ):
            with self.subTest(ordinary=ordinary, seed=seed):
                processor = Mock()
                processor.supports_fused_sampling.return_value = True
                processor.supports_wide_sampling.return_value = True
                processor.sample_stochastic_and_commit.return_value = torch.tensor(
                    [7], dtype=torch.int32
                )
                processor.apply.return_value = SynthIDSparseOutput(
                    scores=torch.tensor([[0.0, -1.0]]),
                    token_ids=torch.tensor([[11, 13]]),
                )
                _, state = _case()
                info = _info(state, vocab=16, sampling_seed=seed)
                with patch(
                    "sglang.srt.layers.sampler.sampling_from_probs_torch",
                    return_value=torch.tensor([1 if seed is not None else 0]),
                ):
                    tokens, _ = _sample(
                        _sampler(processor, ordinary=ordinary, fused=fused),
                        info,
                        torch.zeros(1, 16),
                        logprobs=False,
                    )
                self.assertEqual(tokens.item(), expected)
                self.assertEqual(
                    processor.sample_stochastic_and_commit.called,
                    ordinary and seed is None,
                )
                if seed is not None:
                    self.assertTrue(
                        processor.apply.call_args.kwargs["scatter_to_logits"]
                    )
        with patch(
            "sglang.srt.layers.sampler.murmur_hash32",
            return_value=torch.tensor([[0, 2**32 - 1]], dtype=torch.uint32),
        ):
            for enabled, expected in ((False, 0), (True, 0)):
                sampled = sampling_from_probs_torch(
                    torch.tensor([[1.0, 0.0]]),
                    sampling_seed=torch.tensor([42]),
                    positions=torch.tensor([10]),
                    synthid_applied=enabled,
                )
                self.assertEqual(sampled.item(), expected)

    def test_proposal_relay_policy_transitions_and_nonoverlap_merge(self):
        relay = RequestProposalRelay(slots=3, candidate_count=2, device="cpu")
        relay.store(
            torch.tensor([0]),
            torch.tensor([[0.7, 0.3]]),
            torch.tensor([[0, 1]]),
            [True],
        )
        relay.store(
            torch.tensor([1]), torch.tensor([[0.4, 0.3, 0.2, 0.1]]), None, [False]
        )
        _, mixed = _case(([], []), enabled=[True, False])
        probs, tokens = relay.resolve(torch.tensor([0, 1]), mixed)
        self.assertIsNone(tokens)
        torch.testing.assert_close(
            probs, torch.tensor([[0.7, 0.3, 0.0, 0.0], [0.4, 0.3, 0.2, 0.1]])
        )
        relay.store(torch.tensor([0, 1]), probs, None, [True, False])
        with patch(
            "torch.topk", side_effect=AssertionError("dense relay must not re-sort")
        ):
            actual, tokens = relay.resolve(torch.tensor([1, 0]), mixed)
        torch.testing.assert_close(actual, probs.flip(0))
        self.assertIsNone(tokens)
        for enabled, slot, expected in (
            (True, 0, [[0.7, 0.3]]),
            (False, 1, [[0.4, 0.3, 0.2, 0.1]]),
        ):
            _, state = _case(enabled=[enabled])
            actual, tokens = relay.resolve(torch.tensor([slot]), state)
            torch.testing.assert_close(actual, torch.tensor(expected))
            self.assertEqual(tokens is not None, enabled)
        for reverse in (False, True):
            sparse = EagleDraftInput(
                topk_p=torch.ones(1, 1),
                topk_index=torch.zeros(1, 1, dtype=torch.int64),
                bonus_tokens=torch.zeros(1, dtype=torch.int64),
                draft_probs=torch.tensor([[0.7, 0.3]]),
                draft_prob_token_ids=torch.tensor([[0, 2]]),
            )
            dense = EagleDraftInput(
                topk_p=torch.ones(1, 1),
                topk_index=torch.zeros(1, 1, dtype=torch.int64),
                bonus_tokens=torch.zeros(1, dtype=torch.int64),
                draft_probs=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            )
            left, right = (dense, sparse) if reverse else (sparse, dense)
            left.merge_batch(right)
            self.assertIsNone(left.draft_prob_token_ids)
            expected = torch.tensor([[0.7, 0.0, 0.3, 0.0], [0.1, 0.2, 0.3, 0.4]])
            torch.testing.assert_close(
                left.draft_probs, expected.flip(0) if reverse else expected
            )

    def test_disabled_verification_keeps_release_base_behavior(self):
        info = SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=False,
            temperatures=torch.ones(1, 1),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=True,
            min_ps=torch.tensor([0.5]),
            sampling_seed=None,
        )
        batch = SimpleNamespace(
            device="cpu",
            seq_lens=torch.tensor([1]),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            sampling_info=info,
        )
        verify = SimpleNamespace(
            draft_token=torch.tensor([0]),
            draft_token_num=1,
            max_tree_depth=1,
            retrieve_index=torch.tensor([[0]]),
            retrieve_next_token=torch.tensor([[-1]]),
            retrieve_next_sibling=torch.tensor([[-1]]),
            draft_probs=None,
            draft_prob_token_ids=None,
            tree_topk=1,
        )
        kernel = Mock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "sgl_kernel": SimpleNamespace(
                        top_k_renorm_prob=Mock(),
                        top_p_renorm_prob=Mock(),
                        tree_speculative_sampling_target_only=kernel,
                    )
                },
            ),
            patch.multiple(
                eagle_utils, _is_cpu=False, _is_hip=False, _is_xpu=False
            ),
            get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            output = SimpleNamespace(next_token_logits=torch.tensor([[0.8, 0.2]]).log())
            with get_context().override_server_args(
                speculative_use_rejection_sampling=False
            ):
                eagle_utils.eagle_sample(verify, batch, output)
            torch.testing.assert_close(
                kernel.call_args.kwargs["target_probs"], torch.tensor([[[0.8, 0.2]]])
            )
            with get_context().override_server_args(
                speculative_use_rejection_sampling=True
            ):
                with self.assertRaisesRegex(ValueError, "target-vocab draft proposal"):
                    eagle_utils.eagle_sample(verify, batch, output)
        with patch(
            "sglang.kernels.ops.sampling.murmur_hash.murmur_hash32"
        ) as hash_columns:
            hash_columns.side_effect = (
                lambda seeds, positions, columns: columns.unsqueeze(0).to(torch.uint32)
            )
            eagle_utils._seeded_verify_coins(
                sampling_seed=torch.tensor([42]),
                seq_lens=torch.tensor([10]),
                draft_token_num=4,
                device="cpu",
            )
            self.assertEqual(hash_columns.call_args.args[2].tolist(), [0, 1, 2, 3, 4])

    def test_zero_step_skip_flag_changes_only_for_synthid(self):
        for enabled in (False, True):
            for skip in (False, True):
                with self.subTest(enabled=enabled, skip=skip):
                    draft = SimpleNamespace(
                        draft_tp_context=Mock(return_value=nullcontext()),
                        draft_runner=SimpleNamespace(tp_group=None),
                        _draft_extend_for_decode=Mock(),
                        draft_owns_attention=False,
                    )
                    worker = SimpleNamespace(
                        speculative_num_steps=0,
                        activate_step_by_batch=Mock(),
                        target_worker=SimpleNamespace(
                            model_runner=SimpleNamespace(
                                sampler=SimpleNamespace(
                                    synthid_watermark=object() if enabled else None
                                )
                            )
                        ),
                        draft_worker=draft,
                        verify=Mock(),
                        _stub_skipped_draft_extend=Mock(),
                        _build_trivial_verify_input=Mock(
                            return_value=SimpleNamespace(
                                is_verify_input=lambda: True, kv_loc_plan=None
                            )
                        ),
                    )
                    batch = SimpleNamespace(
                        forward_mode=SimpleNamespace(is_extend=lambda: False),
                        is_extend_in_batch=False,
                        seq_lens=torch.tensor([1]),
                        spec_info=EagleDraftInput(),
                    )
                    with (
                        get_parallel().override(pp_size=1),
                        envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.override(skip),
                        patch(
                            "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
                            side_effect=nullcontext,
                        ),
                        patch(
                            "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
                            side_effect=nullcontext,
                        ),
                        patch(
                            "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
                            side_effect=nullcontext,
                        ),
                    ):
                        EAGLEWorkerV2.forward_batch_generation(worker, batch)
                    self.assertEqual(
                        worker._stub_skipped_draft_extend.called, skip and not enabled
                    )
                    self.assertEqual(
                        draft._draft_extend_for_decode.called,
                        not (skip and not enabled),
                    )

    def test_missing_dense_relay_does_not_reuse_stale_probabilities(self):
        relay = FutureMap.__new__(FutureMap)
        relay.spec_algo = SpeculativeAlgorithm.EAGLE
        relay.request_proposal_relay = None
        relay.device, relay.req_pool_size, relay._forward_buf_initialized = (
            "cpu",
            2,
            True,
        )
        relay.need_topk, relay.need_hidden_states = True, False
        relay.output_tokens_buf = torch.zeros(2, dtype=torch.int64)
        relay.topk_p_buf, relay.topk_index_buf = torch.ones(2, 1), torch.zeros(
            2, 1, dtype=torch.int64
        )
        relay.draft_probs_buf = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
        relay.draft_prob_token_ids_buf = relay.dsa_topk_indices_buf = None
        payload = RelayPayload(
            bonus_tokens=torch.tensor([3]),
            topk_p=torch.ones(1, 1),
            topk_index=torch.zeros(1, 1, dtype=torch.int64),
        )
        indices = torch.tensor([0])
        relay.stash(indices, payload)
        draft = SimpleNamespace(
            future_indices=indices,
            draft_probs=None,
            draft_prob_token_ids=None,
            future_dsa_topk_indices_available=False,
        )
        with (
            patch.object(torch.cpu, "current_stream", return_value=None, create=True),
            patch.object(torch.Tensor, "record_stream", new=lambda *_: None),
            patch(
                "sglang.srt.managers.overlap_utils.gather_spec_extras",
                return_value=(
                    torch.ones(1, 1),
                    torch.zeros(1, 1, dtype=torch.int64),
                    torch.tensor([3]),
                    None,
                ),
            ),
        ):
            relay._resolve_spec_extras(SimpleNamespace(spec_info=draft))
        torch.testing.assert_close(draft.draft_probs, torch.tensor([[0.8, 0.2]]))
        relay.draft_prob_token_ids_buf = torch.zeros(2, 2, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly became empty"):
            relay.stash(indices, payload)


if __name__ == "__main__":
    unittest.main()
