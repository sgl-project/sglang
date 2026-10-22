"""SynthID kernel parity, proposal ordering and live mixed-policy graph replay."""

import unittest
from functools import partial
from types import SimpleNamespace

import torch

from sglang.kernels.ops.sampling.synthid.triton import (
    chain_synthid_speculative_sampling_triton,
    sample_synthid_zero_step_triton,
    watermark_topk_probs_and_sample_triton,
    watermark_topk_scores_triton,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.layers.sampler import sampling_from_probs_torch
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.synthid_request import apply_mixed_logits
from sglang.srt.watermarking.synthid import (
    REFERENCE_SYNTHID_KEYS,
    SynthIDConfig,
    SynthIDState,
    SynthIDWatermarkProcessor,
    watermark_topk_scores_torch,
)
from sglang.srt.watermarking.synthid.sampling import filter_synthid_probs
from sglang.srt.watermarking.synthid.speculative import (
    apply_synthid_to_draft_logits,
    apply_synthid_to_target,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _processor(device, *, top_k=40, skip_first=False, commit_mask=(True, True)):
    config = SynthIDConfig(
        keys=REFERENCE_SYNTHID_KEYS[:4],
        top_k=top_k,
        context_history_size=4,
        skip_first_ngram_calls=skip_first,
    )
    state = SynthIDState.from_output_ids(
        [[11, 12], [21, 22, 23]],
        config=config,
        device=device,
        slot_indices=torch.tensor([1, 5], device=device),
        generations=torch.ones(2, dtype=torch.int64, device=device),
        commit_mask=torch.tensor(commit_mask, device=device),
    )
    return SynthIDWatermarkProcessor(config, device, num_slots=8), state


class TestSynthIDKernels(CustomTestCase):
    def test_mixed_off_target_and_draft_match_cuda_baseline_laws(self):
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

        for kind in ("joint", "min_p", "greedy"):
            with self.subTest(kind=kind), get_context().override_server_args(
                sampling_backend="flashinfer", speculative_use_rejection_sampling=True
            ):
                config = SynthIDConfig(keys=(1, 2), top_k=3)
                processor = SynthIDWatermarkProcessor(config, "cuda", num_slots=2)
                state = SynthIDState.from_output_ids(
                    [[], []],
                    config=config,
                    device="cuda",
                    slot_indices=torch.arange(2, device="cuda"),
                    generations=torch.ones(2, dtype=torch.int64, device="cuda"),
                )
                state.request_enabled_host = [True, False]
                state.request_enabled = torch.tensor([True, False], device="cuda")
                state.commit_mask &= state.request_enabled
                info = SamplingBatchInfo(
                    temperatures=torch.tensor([[0.5], [2.0]], device="cuda"),
                    top_ks=torch.tensor(
                        (
                            [2, 1 if kind == "greedy" else 2]
                            if kind != "min_p"
                            else [TOP_K_ALL, TOP_K_ALL]
                        ),
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    top_ps=torch.full(
                        (2,), 0.55 if kind == "joint" else 1.0, device="cuda"
                    ),
                    min_ps=torch.full(
                        (2,), 0.5 if kind == "min_p" else 0.0, device="cuda"
                    ),
                    is_all_greedy=False,
                    is_any_greedy=kind == "greedy",
                    need_top_k_sampling=kind != "min_p",
                    need_top_p_sampling=kind == "joint",
                    need_min_p_sampling=kind == "min_p",
                    vocab_size=3,
                    device="cuda",
                    synthid_state=state,
                )
                raw = torch.tensor([[0.5, 0.3, 0.2]] * 2, device="cuda").log()
                expected = (raw / info.temperatures).softmax(-1)
                target_expected = expected.clone()
                if info.need_top_k_sampling:
                    target_expected = top_k_renorm_prob(target_expected, info.top_ks)
                if info.need_top_p_sampling:
                    target_expected = top_p_renorm_prob(target_expected, info.top_ps)
                target = raw.repeat_interleave(2, dim=0)
                apply_synthid_to_target(
                    processor=processor,
                    logits=target,
                    sampling_info=info,
                    candidates=torch.zeros(2, 2, dtype=torch.int64, device="cuda"),
                    tree_topk=1,
                    return_logprob=False,
                )
                torch.testing.assert_close(
                    target.softmax(-1)[2:], target_expected[1].expand(2, -1)
                )
                draft = raw.clone()
                apply_synthid_to_draft_logits(
                    processor=processor, logits=draft, sampling_info=info
                )
                torch.testing.assert_close(draft.softmax(-1)[1], expected[1])

    def assert_state_equal(self, reference, actual):
        for field in (
            "context",
            "context_history",
            "history_cursor",
            "num_calls",
            "generations",
        ):
            torch.testing.assert_close(
                getattr(actual.state_pool, field).cpu(),
                getattr(reference.state_pool, field),
            )

    def test_algorithm5_and_fused_temperature_match_torch(self):
        for rows, top_k, depth in (
            (1, 3, 2),
            (8, 40, 30),
            (64, 64, 30),
            (8, 1024, 30),
            (1, 1024, 64),
            (1, 65536, 30),
        ):
            with self.subTest(rows=rows, top_k=top_k, depth=depth):
                torch.manual_seed(7)
                scores = torch.randn(rows, top_k, device="cuda")
                tokens = torch.randint(0, 200_000, scores.shape, device="cuda")
                hashes = torch.arange(rows, device="cuda", dtype=torch.int64)
                keys = torch.arange(1, depth + 1, device="cuda")
                mask = torch.arange(rows, device="cuda") % 3 != 1
                temperatures = torch.linspace(0.5, 1.3, rows, device="cuda")[:, None]
                scaled = scores / temperatures
                expected = watermark_topk_scores_torch(
                    scaled, tokens, hashes, keys, mask
                ).softmax(dim=-1)
                actual = watermark_topk_scores_triton(
                    scaled, tokens, hashes, keys, mask
                ).softmax(dim=-1)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                noise = torch.empty_like(scores).exponential_()
                probs, sampled, selected_probs = watermark_topk_probs_and_sample_triton(
                    scores if top_k <= 1024 else scaled,
                    tokens,
                    hashes,
                    keys,
                    mask,
                    noise,
                    temperatures=temperatures if top_k <= 1024 else None,
                    rows_per_request=1,
                )
                selected = (expected / noise).argmax(dim=-1, keepdim=True)
                torch.testing.assert_close(probs, expected, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(sampled, tokens.gather(1, selected).view(-1))
                torch.testing.assert_close(
                    selected_probs,
                    expected.gather(1, selected).view(-1),
                    rtol=2e-4,
                    atol=2e-5,
                )

    def test_sort_keeps_candidate_noise_pairing_in_eager_and_graph(self):
        for top_k, offset in (
            (1, 0),
            (40, 0),
            (512, 0),
            (1024, 0),
            (512, (1 << 31) - 4096),
        ):
            for sample in (False, True):
                with self.subTest(top_k=top_k, offset=offset, sample=sample):
                    scores = torch.randn(8, top_k, device="cuda")
                    tokens = (
                        torch.stack(
                            [
                                torch.randperm(4096, device="cuda")[:top_k]
                                for _ in range(8)
                            ]
                        )
                        + offset
                    )
                    inputs = (
                        scores,
                        tokens,
                        torch.arange(8, device="cuda"),
                        torch.tensor(REFERENCE_SYNTHID_KEYS[:30], device="cuda"),
                        torch.arange(8, device="cuda") % 3 != 1,
                        torch.empty_like(scores).exponential_() if sample else None,
                    )
                    kwargs = dict(
                        temperatures=torch.tensor([[0.5], [1.3]], device="cuda"),
                        rows_per_request=4,
                    )
                    expected = watermark_topk_probs_and_sample_triton(*inputs, **kwargs)
                    sorted_tokens, order = tokens.sort(dim=-1)
                    storage = torch.empty_like(tokens)
                    original = tokens.clone()
                    graph = torch.cuda.CUDAGraph()
                    for captured in (False, True):
                        if captured:
                            with torch.cuda.graph(graph):
                                actual = watermark_topk_probs_and_sample_triton(
                                    *inputs, **kwargs, sorted_tokens=storage
                                )
                            graph.replay()
                        else:
                            actual = watermark_topk_probs_and_sample_triton(
                                *inputs, **kwargs, sorted_tokens=storage
                            )
                        torch.testing.assert_close(
                            storage, sorted_tokens, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual[0], expected[0].gather(1, order), rtol=0, atol=0
                        )
                        torch.testing.assert_close(tokens, original, rtol=0, atol=0)
                        if sample:
                            torch.testing.assert_close(
                                actual[1:], expected[1:], rtol=0, atol=0
                            )

    def test_projection_sampling_and_commit_match_independent_cpu_state(self):
        for skip_first in (False, True):
            for stage in ("current", "draft", "verify", "sample", "wide", "commit"):
                with self.subTest(skip_first=skip_first, stage=stage):
                    top_k = 65536 if stage == "wide" else 40
                    reference, cpu_state = _processor(
                        "cpu", top_k=top_k, skip_first=skip_first
                    )
                    actual, gpu_state = _processor(
                        "cuda", top_k=top_k, skip_first=skip_first
                    )
                    vocab = 70_000 if stage == "wide" else 4096
                    rows = 6 if stage == "verify" else 2
                    # Unique logits avoid CPU/CUDA top-k tie-order differences.
                    logits = torch.stack(
                        [
                            torch.randperm(vocab, device="cuda").float() / vocab
                            for _ in range(rows)
                        ]
                    )
                    temperatures = torch.tensor([[0.5], [1.3]], device="cuda")
                    prior = torch.tensor([[101, 102], [201, 202]], device="cuda")
                    kwargs = {"scatter_to_logits": False}
                    if stage == "verify":
                        kwargs["verify_tokens"] = torch.cat(
                            (prior, prior[:, :1]), dim=1
                        )
                    if stage == "draft":
                        kwargs["prior_draft_tokens"] = prior
                    if stage == "current":
                        cpu = reference._draft_hashes_and_mask(cpu_state, None)
                        gpu = actual._draft_hashes_and_mask(gpu_state, None)
                        torch.testing.assert_close(tuple(t.cpu() for t in gpu), cpu)
                    elif stage == "commit":
                        for tokens, lengths, mask in (
                            (torch.tensor([101, 201]), None, [False, True]),
                            (
                                torch.tensor([[101, 102, 103], [201, 202, 0]]),
                                torch.tensor([3, 2]),
                                [True, True],
                            ),
                        ):
                            cpu_state.commit_mask = torch.tensor(mask)
                            gpu_state.commit_mask = torch.tensor(mask, device="cuda")
                            reference.commit(cpu_state, tokens, lengths)
                            actual.commit(
                                gpu_state,
                                tokens.cuda(),
                                None if lengths is None else lengths.cuda(),
                            )
                            self.assert_state_equal(reference, actual)
                    else:
                        cpu_kwargs = {
                            k: v.cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in kwargs.items()
                        }
                        apply = "apply_draft" if stage == "draft" else "apply"
                        expected = getattr(reference, apply)(
                            logits.cpu(), temperatures.cpu(), cpu_state, **cpu_kwargs
                        )
                        if stage in ("sample", "wide"):
                            # Unsorted top-k order differs by device; noise belongs to token IDs.
                            vocab_noise = torch.empty_like(logits).exponential_()
                            gpu_candidates = logits.topk(top_k, sorted=False).indices
                            noise = vocab_noise.gather(1, gpu_candidates)
                            cpu_noise = vocab_noise.cpu().gather(1, expected.token_ids)
                            selected = (expected.scores.softmax(-1) / cpu_noise).argmax(
                                -1, keepdim=True
                            )
                            tokens = (
                                expected.token_ids.gather(1, selected).view(-1).int()
                            )
                            reference.commit(cpu_state, tokens)
                            method = (
                                actual.sample_wide_stochastic_and_commit
                                if stage == "wide"
                                else actual.sample_stochastic_and_commit
                            )
                            sampled = method(
                                logits, temperatures, gpu_state, exponential_noise=noise
                            )
                            torch.testing.assert_close(sampled.cpu(), tokens)
                            self.assert_state_equal(reference, actual)
                        else:
                            output = getattr(actual, apply)(
                                logits, temperatures, gpu_state, **kwargs
                            )
                            gpu_tokens, gpu_order = output.token_ids.cpu().sort(-1)
                            cpu_tokens, cpu_order = expected.token_ids.sort(-1)
                            torch.testing.assert_close(gpu_tokens, cpu_tokens)
                            torch.testing.assert_close(
                                output.scores.softmax(-1).cpu().gather(1, gpu_order),
                                expected.scores.softmax(-1).gather(1, cpu_order),
                                rtol=2e-4,
                                atol=2e-5,
                            )

    def test_filters_precede_watermark_for_both_backends(self):
        for backend in ("pytorch", "flashinfer"):
            for top_k, top_p, min_p in (
                (2, 1.0, 0.0),
                (TOP_K_ALL, 0.75, 0.0),
                (TOP_K_ALL, 1.0, 0.5),
                (2, 0.55, 0.0),
                (2, 0.55, 0.5),
            ):
                with self.subTest(
                    backend=backend, top_k=top_k, top_p=top_p, min_p=min_p
                ):
                    config = SynthIDConfig(keys=(9,), top_k=3)
                    processor = SynthIDWatermarkProcessor(config, "cuda", num_slots=1)
                    state = SynthIDState.from_output_ids(
                        [[]],
                        config=config,
                        device="cuda",
                        slot_indices=torch.tensor([0], device="cuda"),
                        generations=torch.tensor([1], device="cuda"),
                    )
                    info = SimpleNamespace(
                        top_ks=torch.tensor([top_k], device="cuda", dtype=torch.int32),
                        top_ps=torch.tensor([top_p], device="cuda"),
                        min_ps=torch.tensor([min_p], device="cuda"),
                        need_min_p_sampling=min_p > 0,
                    )
                    logits = torch.tensor([[0.5, 0.3, 0.2]], device="cuda").log()
                    with get_context().override_server_args(sampling_backend=backend):
                        processor.apply(
                            logits,
                            torch.ones(1, 1, device="cuda"),
                            state,
                            filter_probs=partial(
                                filter_synthid_probs, sampling_info=info
                            ),
                        )
                    expected = (
                        [1.0, 0.0, 0.0]
                        if backend == "flashinfer" and min_p > 0 and top_p == 0.55
                        else [5 / 8, 3 / 8, 0.0]
                    )
                    torch.testing.assert_close(
                        logits.softmax(-1),
                        torch.tensor([expected], device="cuda"),
                        rtol=2e-5,
                        atol=2e-6,
                    )
                    sampled = sampling_from_probs_torch(logits.softmax(-1))
                    self.assertGreater(logits.softmax(-1)[0, sampled.item()].item(), 0)

    def test_mixed_graph_replays_live_state_masks_and_padding(self):
        config = SynthIDConfig(keys=REFERENCE_SYNTHID_KEYS[:3], top_k=64)
        processor = SynthIDWatermarkProcessor(config, "cuda", num_slots=4)
        initial = SynthIDState.from_output_ids(
            [[i + 1, i + 2, i + 3, i + 4] for i in range(4)],
            config=config,
            device="cuda",
            slot_indices=torch.arange(4, device="cuda"),
            generations=torch.ones(4, dtype=torch.int64, device="cuda"),
        )
        processor.state_pool.initialize_pending(initial)
        processor.capture_mixed_target_graphs(
            capture_bs=[2, 4], rows_per_request=4, vocab_size=1024
        )
        for flags in ([True, False, True, True], [False, True, False], [True, False]):
            size = len(flags)
            processor.commit(initial, torch.arange(4, device="cuda") + 17)
            state = SynthIDState.for_initialized_slots(
                initial.slot_indices[:size].flip(0), config=config
            )
            state.request_enabled_host = flags
            state.request_enabled = torch.tensor(flags, device="cuda")
            temperatures = torch.tensor([[0.5], [1.3], [0.8], [1.0]], device="cuda")[
                :size
            ]
            candidates = torch.randint(0, 1024, (size, 4), device="cuda")
            raw = torch.randn(size * 4, 1024, device="cuda")
            expected, actual = raw.clone(), raw.clone()
            apply_mixed_logits(
                processor=processor,
                logits=expected,
                temperatures=temperatures,
                state=state,
                verify_tokens=candidates,
            )
            self.assertTrue(
                processor.try_apply_mixed_target(
                    logits=actual,
                    temperatures=temperatures,
                    state=state,
                    verify_tokens=candidates,
                )
            )
            torch.testing.assert_close(
                actual.softmax(-1), expected.softmax(-1), rtol=2e-4, atol=2e-5
            )
            off = ~state.request_enabled.repeat_interleave(4)
            torch.testing.assert_close(
                actual[off],
                (raw / temperatures.repeat_interleave(4, dim=0))[off],
                rtol=0,
                atol=0,
            )
            draft = raw[:size].clone()
            apply_mixed_logits(
                processor=processor,
                logits=draft,
                temperatures=temperatures,
                state=state,
                draft=True,
            )
            rng = torch.cuda.get_rng_state()
            probs, selected_probs, tokens = processor.try_sample_mixed_draft(
                logits=raw[:size],
                temperatures=temperatures,
                state=state,
                prior_draft_tokens=None,
            )
            self.assertFalse(torch.equal(rng, torch.cuda.get_rng_state()))
            torch.testing.assert_close(probs, draft.softmax(-1), rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(
                selected_probs, probs.gather(1, tokens), rtol=0, atol=0
            )

    def test_sparse_rejection_matches_dense_for_accept_and_reject(self):
        candidates = torch.tensor([[0, 1, 3, 5], [0, 1, 3, 5]], device="cuda")
        indices = torch.arange(8, device="cuda").view(2, 4)
        target_tokens = (
            torch.tensor([1, 3, 5], device="cuda").expand(2, 4, 3).contiguous()
        )
        target = (
            torch.tensor([0.2, 0.3, 0.5], device="cuda").expand(2, 4, 3).contiguous()
        )
        draft_tokens = target_tokens[:, :3].contiguous()
        draft = (
            torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]], device="cuda")[:, None]
            .expand(2, 3, 3)
            .contiguous()
        )
        uniforms = torch.full((2, 3), 0.95, device="cuda")
        final_uniforms = torch.full((2,), 0.5, device="cuda")
        outputs = [
            (
                torch.zeros(8, dtype=torch.int32, device="cuda"),
                torch.full((2, 4), -1, dtype=torch.int32, device="cuda"),
                torch.empty(2, dtype=torch.int32, device="cuda"),
            )
            for _ in range(2)
        ]
        chain_speculative_sampling_triton(
            predicts=outputs[0][0],
            accept_index=outputs[0][1],
            accept_token_num=outputs[0][2],
            candidates=candidates,
            retrive_index=indices,
            retrive_next_token=None,
            retrive_next_sibling=None,
            uniform_samples=uniforms,
            uniform_samples_for_final_sampling=final_uniforms,
            target_probs=torch.zeros(2, 4, 8, device="cuda").scatter_(
                2, target_tokens, target
            ),
            draft_probs=torch.zeros(2, 3, 8, device="cuda").scatter_(
                2, draft_tokens, draft
            ),
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        chain_synthid_speculative_sampling_triton(
            predicts=outputs[1][0],
            accept_index=outputs[1][1],
            accept_token_num=outputs[1][2],
            candidates=candidates,
            retrieve_index=indices,
            uniform_samples=uniforms,
            uniform_samples_for_final_sampling=final_uniforms,
            target_probs=target,
            target_tokens=target_tokens,
            draft_probs=draft,
            draft_tokens=draft_tokens,
        )
        torch.testing.assert_close(outputs[1], outputs[0])
        self.assertGreater(outputs[1][2][0].item(), outputs[1][2][1].item())
        sample_synthid_zero_step_triton(
            predicts=outputs[1][0],
            accept_index=outputs[1][1][:, :1].contiguous(),
            accept_token_num=outputs[1][2],
            retrieve_index=torch.arange(2, device="cuda")[:, None],
            uniform_samples_for_final_sampling=final_uniforms,
            target_probs=torch.nn.functional.one_hot(
                torch.tensor([3, 6], device="cuda"),
                num_classes=8,
            ).float()[:, None],
        )
        self.assertEqual(outputs[1][0][:2].tolist(), [3, 6])
        self.assertEqual(outputs[1][2].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
