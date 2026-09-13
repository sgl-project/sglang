import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    _make_forward_batch,
    build_dense_attention_fixture,
    dense_attention_reference_with_custom_mask,
    dense_fixture_inputs,
    make_dense_random_inputs,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    prepare_dense_runner_inputs,
    run_dense_attention_case,
    run_dense_forward,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    _prepare_spec_verify_batch,
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_dense_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=13, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferSWAAttentionBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_swa_no_prefix_input_config_cases(
        "flashinfer"
    ) + make_swa_prefix_input_config_cases("flashinfer")
    # Paged-only prefill has no ragged pass / custom prefix mask, so the kernel
    # must enforce the window; the long case puts tokens past the window.
    PAGED_MODE_CASES = CASES + (
        DenseAttentionCase(
            name="swa_extend_no_prefix_above_window_long",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(6, 8, 12),
            sliding_window_size=4,
        ),
    )
    # Above-window decode case requires the `extend_window` reference rule
    # (window+1 keys), not the `min_seq_len_window` rule — FlashInfer's
    # decode metadata uses `clamp(seq_lens, max=window+1)` per
    # `flashinfer_backend.py:1031`. See `_SWA_DECODE_EXTEND_WINDOW` in
    # `common/attention_methods/dense_attention.py`.
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_within_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(1, 2, 3),
            sliding_window_size=4,
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_above_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(7, 8, 9),
            sliding_window_size=4,
        ),
    )
    # NOTE: a `runner_split_op_swa_extend_prefix_within_window` clone of the
    # triton SWA test fails on flashinfer (~0.21 max diff). FlashInfer's
    # prefill-split path does not handle SWA prefix the same way as triton;
    # the projected EXTEND covers the prefix path through the unsplit kernel
    # which does match the reference. Investigate before adding split_op
    # prefix to flashinfer SWA.
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_swa_extend_no_prefix_window_edges",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(0, 0, 0),
                extend_lens=(3, 4, 5),
                sliding_window_size=4,
            ),
            16,
        ),
    )
    SPEC_VERIFY_CASES = (
        (
            DenseAttentionCase(
                name="runner_dflash_verify_swa_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_dflash_verify_swa_window_edges",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                # Straddle the window: one request below, one at, one above.
                prefix_lens=(1, 4, 9),
                extend_lens=(3, 3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
    )
    SPEC_VERIFY_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_dflash_verify_swa_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
    )

    EAGLE_VERIFY_CASES = tuple(
        DenseAttentionCase(
            name=f"eagle_swa_{prefix_lens}_{draft_tokens}_{window}",
            backend="flashinfer",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=prefix_lens,
            extend_lens=(draft_tokens,) * len(prefix_lens),
            sliding_window_size=window,
        )
        for prefix_lens, draft_tokens, window in (
            ((2,), 3, 4),
            ((3, 4, 5), 3, 4),
            ((9, 2), 3, 4),
            ((2, 9), 3, 4),
            ((12, 7), 3, 4),
            ((0, 9), 3, 4),
            ((9, 2), 7, 2),
            ((1220, 325), 8, 1023),
            ((325, 1220), 8, 1023),
        )
    )

    def test_eagle_swa_tree_generation(self):
        from sglang.srt.batch_overlap.two_batch_overlap import split_spec_info
        from sglang.srt.layers.attention.verify_mask import VerifyMask
        from sglang.srt.runtime_context import get_context
        from sglang.srt.speculative.eagle_utils import (
            TreeMaskMode,
            build_tree_kernel_efficient,
        )
        from sglang.srt.speculative.eagle_worker_common import build_eagle_verify_input

        runner = SimpleNamespace(
            attn_backend=SimpleNamespace(verify_mask=None, max_context_len=2048),
            sliding_window_size=4,
            prefill_attention_backend_str="flashinfer",
            decode_attention_backend_str="triton",
        )
        for lengths, host_lens, mode in (
            ((12, 7), True, "prefill"),
            ((2, 9), True, "prefill"),
            ((1, 4), True, "prefill"),
            ((1, 1), True, "prefill"),
            ((12, 7), False, "prefill"),
            ((12, 7), True, "decode"),
            ((1, 1), True, "decode"),
            ((1, 4), True, "decode"),
        ):
            with self.subTest(lengths=lengths, host_lens=host_lens, mode=mode):
                override = get_context().override_server_args(
                    speculative_attention_mode=mode
                )
                override.install()
                self.addCleanup(override.restore)
                bs, draft = len(lengths), 8
                runner.attn_backend.verify_mask = (
                    VerifyMask(
                        buffer=torch.zeros(
                            bs * draft * (2048 + draft),
                            dtype=torch.uint8,
                            device="cuda",
                        ),
                        mode=TreeMaskMode.FULL_MASK,
                        max_bs=bs,
                    )
                    if mode == "decode"
                    else None
                )
                cpu = torch.tensor(lengths)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    seq_lens=cpu.cuda(),
                    seq_lens_cpu=cpu if host_lens else None,
                    seq_lens_sum=sum(lengths) if host_lens else None,
                )
                bonus = torch.arange(bs, device="cuda")
                parents = torch.tensor(
                    [[0, 0, 1, 4, 0, 5, 0, 0, 0]] * bs, device="cuda"
                )
                indices = torch.tensor([[0, 1, 4, 8, 5, 20, 12]] * bs, device="cuda")
                tokens = torch.arange(bs * (draft - 1), device="cuda").view(bs, -1)
                expected = build_tree_kernel_efficient(
                    bonus,
                    parents,
                    indices,
                    tokens,
                    batch.seq_lens,
                    sum(lengths),
                    4,
                    3,
                    draft,
                )
                with patch(
                    "sglang.srt.speculative.eagle_worker_common.build_tree_kernel_efficient",
                    wraps=build_tree_kernel_efficient,
                ) as build:
                    actual = build_eagle_verify_input(
                        batch,
                        SimpleNamespace(bonus_tokens=bonus),
                        parents,
                        indices,
                        tokens,
                        None,
                        target_worker=SimpleNamespace(model_runner=runner),
                        topk=4,
                        num_steps=3,
                        num_draft_tokens=draft,
                        tree_mask_mode=TreeMaskMode.FULL_MASK,
                        device="cuda",
                    )
                torch.testing.assert_close(
                    actual.custom_mask[: expected[0].numel()],
                    expected[0],
                    check_dtype=False,
                )
                for attr, value in zip(
                    (
                        "positions",
                        "retrieve_index",
                        "retrieve_next_token",
                        "retrieve_next_sibling",
                        "draft_token",
                    ),
                    expected[1:],
                ):
                    torch.testing.assert_close(getattr(actual, attr), value)
                full_masks, offset = [], 0
                for request, prefix in enumerate(lengths):
                    size = draft * (prefix + draft)
                    full = expected[0][offset : offset + size].view(draft, -1)
                    pos = expected[1].view(bs, draft)[request]
                    key_pos = torch.cat((torch.arange(prefix, device="cuda"), pos))
                    bounded = full & (key_pos[None, :] >= pos[:, None] - 4)
                    full_masks.append(bounded[:, max(0, prefix - 4) :].flatten())
                    offset += size
                compact = torch.cat(full_masks)
                if host_lens:
                    self.assertEqual(actual.swa_custom_mask.numel(), compact.numel())
                torch.testing.assert_close(
                    actual.swa_custom_mask[: compact.numel()],
                    compact,
                    check_dtype=False,
                )
                self.assertEqual(build.call_count, 1 if max(lengths) <= 4 else 2)
                if max(lengths) + 3 <= 4:
                    self.assertEqual(
                        actual.swa_custom_mask.data_ptr(), actual.custom_mask.data_ptr()
                    )
                else:
                    self.assertIsNot(actual.swa_custom_mask, actual.custom_mask)
                actual.seq_lens_cpu = cpu
                for i, expected_mask in enumerate(full_masks):
                    child = split_spec_info(
                        actual, i, i + 1, i * draft, (i + 1) * draft
                    )
                    torch.testing.assert_close(
                        child.swa_custom_mask, expected_mask, check_dtype=False
                    )

    def test_full_verify_mask_padding_contract(self):
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPVerifyInput

        for verify_cls in (EagleVerifyInput, FrozenKVMTPVerifyInput):
            for size in (5, 29):
                with self.subTest(verify_cls=verify_cls.__name__, size=size):
                    info = verify_cls.create_idle_input(1, 2, 3, "cuda")
                    original = torch.arange(size, device="cuda") % 2 == 0
                    info.custom_mask = original
                    info.swa_custom_mask = torch.zeros(
                        21, device="cuda", dtype=torch.bool
                    )
                    indices, _, _, mask = info.generate_attn_arg_prefill(
                        torch.tensor([0], device="cuda"),
                        torch.tensor([4], device="cuda"),
                        4,
                        torch.arange(32, device="cuda").view(1, -1),
                    )
                    torch.testing.assert_close(
                        indices, torch.arange(7, device="cuda", dtype=torch.int32)
                    )
                    expected = torch.cat(
                        (
                            original,
                            torch.ones(
                                max(0, 21 - size), device="cuda", dtype=torch.bool
                            ),
                        )
                    )
                    torch.testing.assert_close(mask, expected)
                    self.assertIs(mask, info.custom_mask)
                    if size >= 21:
                        self.assertIs(mask, original)

    def test_eagle_swa_verify(self):
        for case in self.EAGLE_VERIFY_CASES:
            for topk in (1, 2) if case.extend_lens[0] == 3 else (1,):
                with self.subTest(case=case.name, topk=topk):
                    run_dense_spec_verify_case(
                        self,
                        case,
                        topk=topk,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        max_context_len=2048,
                    )

    def test_eagle_swa_verify_graph_metadata(self):
        for case in self.EAGLE_VERIFY_CASES:
            for topk in (1, 2) if case.extend_lens[0] == 3 else (1,):
                with self.subTest(case=case.name, topk=topk):
                    run_dense_spec_verify_cuda_graph_case(
                        self,
                        case,
                        topk=topk,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        max_context_len=2048,
                    )

    @torch.no_grad()
    def test_eagle_swa_and_full_real_graph_replay(self):
        # An irregular, multi-level tree: siblings can have equal positions.
        parents = (-1, 0, 0, 1, 2, 1, 5, 3)
        draft = len(parents)
        tree = torch.zeros(draft, draft, dtype=torch.bool)
        depths = []
        for query in range(draft):
            node = query
            while node >= 0:
                tree[query, node] = True
                node = parents[node]
            depths.append(int(tree[query].sum()) - 1)
        case = DenseAttentionCase(
            name="eagle_swa_and_full_real_graph",
            backend="flashinfer",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(1, 1, 1),
            extend_lens=(draft,) * 3,
            sliding_window_size=4,
        )
        fixture = build_dense_attention_fixture(
            self,
            case,
            head_dim=self.HEAD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            max_context_len=2048,
            disable_cuda_graph=False,
        )
        backend = fixture.backend
        batch = fixture.forward_batch
        inputs = dense_fixture_inputs(fixture)
        _prepare_spec_verify_batch(
            case, batch, topk=1, spec_kind="eagle", device="cuda"
        )
        batch.spec_info.positions = None  # Production's dummy capture input.
        batch.spec_info.swa_custom_mask = None
        backend.init_cuda_graph_state(max_bs=3, max_num_tokens=3 * draft)
        self.assertIsNone(backend.cuda_graph_swa_custom_mask)
        backend._create_prefill_wrappers(3, use_custom_mask=False)  # DFlash path.
        self.assertIsNone(backend.cuda_graph_swa_custom_mask)

        def forward_both():
            outputs = []
            for window in (case.sliding_window_size, -1):
                fixture.actual_module.attn.sliding_window_size = window
                outputs.append(run_dense_forward(fixture, batch, inputs))
            return outputs

        with forward_context(ForwardContext(attn_backend=backend)):
            backend.init_forward_metadata_out_graph(batch, in_capture=True)
            backend.init_forward_metadata_in_graph(batch)
            for _ in range(3):
                forward_both()
            backend.on_after_cuda_graph_warmup()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = forward_both()
            mask_ptrs = (
                backend.cuda_graph_swa_custom_mask.data_ptr(),
                backend.cuda_graph_custom_mask.data_ptr(),
            )
            self.assertNotEqual(*mask_ptrs)

            for prefix_lens in ((12, 7), (2, 9), (9,), (325, 1220), (1, 4, 9)):
                with self.subTest(prefix_lens=prefix_lens):
                    real_bs = len(prefix_lens)
                    replay_case = replace(
                        case, prefix_lens=prefix_lens + (1,) * (3 - real_bs)
                    )
                    replay_inputs = make_dense_random_inputs(
                        replay_case,
                        fixture,
                        dtype=inputs["input_hidden"].dtype,
                        device="cuda",
                    )
                    replay_batch = _make_forward_batch(
                        replay_case, fixture.runner, max_context_len=2048, device="cuda"
                    )
                    _prepare_spec_verify_batch(
                        replay_case,
                        replay_batch,
                        topk=1,
                        spec_kind="eagle",
                        device="cuda",
                    )
                    full_masks, swa_masks, positions = [], [], []
                    for prefix in replay_case.prefix_lens:
                        full = torch.cat(
                            (torch.ones(draft, prefix, dtype=torch.bool), tree), dim=1
                        )
                        swa = full.clone()
                        key_positions = list(range(prefix)) + [
                            prefix + d for d in depths
                        ]
                        for query, depth in enumerate(depths):
                            for key, pos in enumerate(key_positions):
                                if pos < prefix + depth - case.sliding_window_size:
                                    swa[query, key] = False
                        full_masks.append(full.cuda())
                        swa_masks.append(swa.cuda())
                        positions.extend(prefix + d for d in depths)
                    # Real verify inputs omit padded requests from the mask and positions.
                    original_mask = torch.cat(
                        [m.flatten() for m in full_masks[:real_bs]]
                    )
                    replay_batch.spec_info.custom_mask = original_mask
                    replay_batch.spec_info.swa_custom_mask = torch.cat(
                        [
                            m[:, max(0, p - case.sliding_window_size) :].flatten()
                            for p, m in zip(prefix_lens, swa_masks[:real_bs])
                        ]
                    )
                    replay_batch.spec_info.positions = torch.tensor(
                        positions[: real_bs * draft], device="cuda"
                    )
                    prepare_dense_runner_inputs(
                        fixture,
                        replay_case,
                        replay_batch,
                        replay_inputs,
                        max_context_len=2048,
                    )
                    expected = [
                        dense_attention_reference_with_custom_mask(
                            fixture.reference_module,
                            replace(replay_case, sliding_window_size=None),
                            replay_inputs["prefix_hidden"],
                            replay_inputs["input_hidden"],
                            masks,
                        )
                        for masks in (swa_masks, full_masks)
                    ]
                    inputs["input_hidden"].copy_(replay_inputs["input_hidden"])
                    batch.out_cache_loc.copy_(replay_batch.out_cache_loc)
                    if real_bs == 1:
                        replay_batch.seq_lens_cpu = (
                            None  # Exercise the device-only fallback.
                        )
                    backend.init_forward_metadata_out_graph(replay_batch)
                    graph.replay()
                    graph.replay()
                    torch.cuda.synchronize()
                    for output, reference in zip(actual, expected):
                        torch.testing.assert_close(
                            output[: real_bs * draft],
                            reference[: real_bs * draft],
                            atol=DENSE_ATOL,
                            rtol=DENSE_RTOL,
                        )
                    torch.testing.assert_close(
                        original_mask,
                        torch.cat([m.flatten() for m in full_masks[:real_bs]]),
                    )
                    self.assertEqual(
                        mask_ptrs,
                        (
                            backend.cuda_graph_swa_custom_mask.data_ptr(),
                            backend.cuda_graph_custom_mask.data_ptr(),
                        ),
                    )

    def test_projected_swa_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_projected_swa_attention_cases_paged_mode(self):
        for case in self.PAGED_MODE_CASES:
            with self.subTest(case=case.name, backend=case.backend, mode="paged"):
                with envs.SGLANG_FLASHINFER_USE_PAGED.override(True):
                    run_dense_attention_case(
                        self,
                        case,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                    )

    # Layout-robustness. See dense/test_triton.py for the full rationale.
    # The default `shuffled_pages` is already exercised by
    # test_projected_swa_attention_cases on the existing case list.
    # This method opts into the more aggressive interleaved_pages +
    # non_monotonic_extend on within-window extend + decode.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_swa_extend_below_window",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(10,),
            sliding_window_size=12,
        ),
        DenseAttentionCase(
            name="layout_swa_decode_within_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(8, 10),
            sliding_window_size=12,
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(
                        self,
                        case,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_dense_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                    )

    def test_runner_mode_spec_verify_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )


if __name__ == "__main__":
    unittest.main()
