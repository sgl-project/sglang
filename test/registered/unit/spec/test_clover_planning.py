# SPDX-License-Identifier: Apache-2.0
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterPrefill,
    fast_prefill_plan,
)
from sglang.srt.layers.attention.linear.clover_backend import CloverGDNAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.clover_draft import CloverPreviewFlashInferBackend
from sglang.srt.speculative.clover_info import CloverPreviewInput
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestCloverPlanning(CustomTestCase):
    def test_verify_preplan_marker_and_shape_change(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.clover_info import CloverVerifyInput

        verify = CloverVerifyInput.__new__(CloverVerifyInput)
        for mode, ready in [
            (ForwardMode.TARGET_VERIFY, True),
            (ForwardMode.IDLE, False),
        ]:
            fb = ForwardBatch.__new__(ForwardBatch)
            fb.batch_size = 1
            fb.input_ids = torch.empty(16, dtype=torch.int64)
            with patch(
                "sglang.srt.speculative.clover_info.DFlashVerifyInput.prepare_for_verify",
                return_value=(fb, True),
            ):
                result, graph = verify.prepare_for_verify(
                    SimpleNamespace(forward_mode=mode), None
                )
            self.assertIs(result, fb)
            self.assertTrue(graph)
            self.assertEqual(fb.needs_forward_metadata_init(), not ready)
            fb.input_ids = torch.empty(32, dtype=torch.int64)
            self.assertTrue(fb.needs_forward_metadata_init())

    def test_capture_planner_preserves_existing_routes(self):
        # Install only after capture's ordinary plan initialized the wrapper.
        preview, base = CloverPreviewFlashInferBackend, FlashInferAttnBackend
        verify, extend = ForwardMode.TARGET_VERIFY, ForwardMode.DLLM_EXTEND
        draft = SpecInputType.CLOVER_DRAFT
        cases = [
            (preview, verify, SpecInputType.CLOVER_VERIFY, True, "fa2", None, True),
            (base, verify, SpecInputType.DFLASH_VERIFY, True, "fa2", None, True),
            (base, verify, SpecInputType.EAGLE_VERIFY, True, "fa2", None, False),
            (preview, extend, draft, True, "fa2", None, True),
            (preview, extend, draft, False, "fa2", None, False),
            (preview, extend, None, True, "fa2", None, False),
            (preview, extend, draft, True, "fa3", None, False),
            (preview, extend, draft, True, "fa2", "swa", False),
        ]
        for backend_type, mode, kind, paged, impl, dispatch, fast in cases:
            with self.subTest(
                backend=backend_type.__name__,
                mode=mode,
                kind=kind,
                paged=paged,
                impl=impl,
                dispatch=dispatch,
            ):
                backend = backend_type.__new__(backend_type)
                regular = Mock()
                wrapper = SimpleNamespace(begin_forward=regular)
                backend.prefill_cuda_graph_metadata = {1: [wrapper]}
                backend.draft_extend_cuda_graph_metadata = {1: [wrapper]}
                backend.prefill_backend = impl
                backend.dispatch_reason = dispatch
                backend.use_paged = paged
                backend.dllm_config = SimpleNamespace(block_size=16)
                backend.use_sliding_window_kv_pool = False
                backend._prepare_cuda_graph_metadata = Mock()

                def update(*args, **kwargs):
                    # Missing preview metadata would fast-plan a normal extend.
                    if mode.is_dllm_extend():
                        self.assertIs(
                            kwargs["spec_info"],
                            (
                                spec
                                if backend_type is CloverPreviewFlashInferBackend
                                else None
                            ),
                        )
                    wrapper.begin_forward()

                backend.indices_updater_prefill = SimpleNamespace(update=update)
                spec = (
                    None
                    if kind is None
                    else SimpleNamespace(
                        spec_input_type=kind,
                        custom_mask=None,
                        ragged_verify_layout=None,
                    )
                )
                if kind == SpecInputType.CLOVER_DRAFT:
                    spec = CloverPreviewInput(16)
                batch = SimpleNamespace(
                    batch_size=1,
                    forward_mode=mode,
                    spec_info=spec,
                    req_pool_indices=torch.tensor([0]),
                    seq_lens=torch.tensor([101]),
                    seq_lens_cpu=torch.tensor([101]),
                    seq_lens_sum=101,
                    encoder_lens=None,
                    positions=torch.arange(16),
                    kv_loc_plan=object(),
                    out_cache_loc=None,
                )
                if (
                    backend_type is CloverPreviewFlashInferBackend
                    and mode.is_dllm_extend()
                    and spec is None
                ):
                    with self.assertRaisesRegex(TypeError, "CloverPreviewInput"):
                        backend.init_forward_metadata_out_graph(batch, in_capture=True)
                    continue
                backend.init_forward_metadata_out_graph(batch, in_capture=True)
                self.assertEqual(regular.call_count, 1)
                if fast and (
                    not mode.is_dllm_extend()
                    or backend_type is CloverPreviewFlashInferBackend
                ):
                    self.assertIs(wrapper.begin_forward.func, fast_prefill_plan)
                else:
                    self.assertIs(wrapper.begin_forward, regular)

    def test_preview_worker_and_capture_share_width_contract(self):
        from sglang.srt.speculative.clover_draft import CloverDraftCudaGraphRunner
        from sglang.srt.speculative.clover_worker import CloverWorker

        # Worker and capture batches must agree when a capture bucket pads.
        width, bs = 16, 2
        worker = CloverWorker.__new__(CloverWorker)
        worker.block_size = width
        worker.device = "cpu"
        worker.model_runner = SimpleNamespace(kv_index_translator=None)
        prefixes = torch.arange(bs) * 7 + 7
        batch = SimpleNamespace(
            batch_size=lambda: bs,
            seq_lens=prefixes,
            seq_lens_cpu=prefixes,
            req_pool_indices=torch.arange(bs),
            sampling_info=None,
            spec_algorithm=None,
        )
        fb = worker._preview_batch(
            batch,
            torch.zeros(bs, width, dtype=torch.int64),
            (prefixes[:, None] + torch.arange(width)).reshape(-1),
            torch.arange(bs * width),
            SimpleNamespace(bind=lambda *args: None),
        )
        runner = CloverDraftCudaGraphRunner.__new__(CloverDraftCudaGraphRunner)
        runner.captured_req_width = width
        padded_bs = 4 if bs == 3 else bs
        for spec in (fb.spec_info, runner.get_spec_info(padded_bs * width)):
            self.assertIsInstance(spec, CloverPreviewInput)
            self.assertTrue(spec.is_draft_input())
            self.assertEqual(spec.num_tokens_per_req, width)
            self.assertEqual(spec.draft_token_num, width)
            self.assertEqual(spec.num_tokens_for_logprob_per_req, width)
            self.assertIsNone(spec.hidden_states)
        self.assertTrue(fb.forward_mode.is_dllm_extend())
        self.assertEqual(fb.seq_lens_cpu.tolist(), (prefixes + width).tolist())
        self.assertEqual(fb.seq_lens_sum, int(prefixes.sum()) + bs * width)
        self.assertEqual(fb.extend_seq_lens_cpu, [width] * bs)
        self.assertEqual(
            fb.extend_start_loc.tolist(), list(range(0, bs * width, width))
        )

    def test_preview_host_layout_uses_batch_speculative_width(self):
        # Preview KV lengths include the block, and host CSR must refresh per call.
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.attn_backend = SimpleNamespace(
            kv_index_translator=SimpleNamespace(
                reads_are_translated=False,
                pack_read_stream=lambda plan, **kwargs: kwargs["out"].zero_(),
            ),
            dq_paged_kernel_lens=None,
        )
        updater.num_qo_heads = 16
        updater.num_kv_heads = 2
        updater.head_dim = 128
        updater.q_data_type = updater.data_type = torch.bfloat16
        updater.kv_last_page_len = torch.ones(2, dtype=torch.int32)
        snapshots = []

        def capture(wrapper, qo, kv, *args, **kwargs):
            torch.testing.assert_close(qo, kwargs["qo_indptr_host"])
            torch.testing.assert_close(kv, kwargs["kv_indptr_host"])
            snapshots.append((qo.tolist(), kv.tolist(), kwargs["max_q_len"]))

        with patch(
            "sglang.srt.layers.attention.flashinfer_backend.fast_prefill_plan",
            side_effect=capture,
        ) as fast:
            wrapper = SimpleNamespace(
                begin_forward=partial(fast, None),
                is_cuda_graph_enabled=True,
                _custom_mask_buf=None,
            )
            for width, lengths in [(16, [31]), (16, [105]), (32, [43, 59])]:
                seq_lens = torch.tensor(lengths, dtype=torch.int32)
                bs = len(lengths)
                updater.call_begin_forward(
                    None,
                    wrapper,
                    torch.arange(bs),
                    seq_lens,
                    sum(lengths),
                    seq_lens,
                    seq_lens - width,
                    None,
                    torch.zeros(bs + 1, dtype=torch.int32),
                    torch.zeros(bs + 1, dtype=torch.int32),
                    False,
                    CloverPreviewInput(width),
                    seq_lens_cpu=seq_lens,
                    custom_kv_indices=torch.zeros(sum(lengths), dtype=torch.int32),
                    plan=object(),
                )
        self.assertEqual(
            snapshots,
            [
                ([0, 16], [0, 31], 16),
                ([0, 16], [0, 105], 16),
                ([0, 32, 64], [0, 43, 102], 32),
            ],
        )


def make_backend(device="cpu", block_size=4):
    backend = object.__new__(CloverGDNAttnBackend)
    backend.device = device
    backend.pad_slot_id = -1
    backend.topk = 1
    backend._fused_state_indices_ok = False
    backend._graph_seq_len_fill_value = 1
    mapping = torch.tensor([0, 2, 1, 3], device=device, dtype=torch.int32)
    physical = torch.tensor([0, 3, 2, 1], device=device, dtype=torch.int32)
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda reqs: mapping[reqs.long()],
        translate_mamba_indices=lambda slots: physical[slots.long()],
        mamba_pool=SimpleNamespace(size=4, enable_linear_replayssm=False),
    )
    for name in (
        "state_indices_list",
        "query_start_loc_list",
        "retrieve_next_token_list",
        "retrieve_next_sibling_list",
        "retrieve_parent_token_list",
    ):
        setattr(backend, name, [])
    backend.init_cuda_graph_state(3, 3 * block_size)
    return backend, mapping, physical


def test_replay_refreshes_reordered_restored_and_padded_state_slots():
    block_size = 32
    backend, mapping, physical = make_backend(block_size=block_size)
    captured = backend._capture_metadata(
        3, torch.tensor([1, 2, 3]), ForwardMode.DLLM_EXTEND, None
    )
    indices, offsets = captured.mamba_cache_indices, captured.query_start_loc
    pointers = indices.data_ptr(), offsets.data_ptr()
    assert indices.tolist() == [2, 3, 1]
    assert offsets.tolist() == [i * block_size for i in range(4)]

    # Reorder, early finish, then restore a radix prefix into different physical
    # state storage. The same captured views must observe all three updates.
    cases = [
        ([2, 1, 3], [132, 68, 36], 0, [3, 2, 1]),
        ([3, 1, 2], [68, 36, 1], 1, [1, 2, -1]),
        ([1, 3, 2], [132, 1, 1], 2, [2, -1, -1]),
    ]
    for reqs, lengths, padding, expected in cases:
        replay = backend._replay_metadata(
            3,
            torch.tensor(reqs),
            ForwardMode.DLLM_EXTEND,
            None,
            torch.tensor(lengths),
            num_padding=padding,
        )
        assert indices.tolist() == expected
        assert pointers == (
            replay.mamba_cache_indices.data_ptr(),
            replay.query_start_loc.data_ptr(),
        )
        # Padded rows retain block-wide offsets and the -1 state sentinel masks them.
        assert offsets.tolist() == [i * block_size for i in range(4)]
    mapping[1] = 1
    physical[1] = 2
    replay = backend._replay_metadata(
        3,
        torch.tensor([1, 2, 3]),
        ForwardMode.DLLM_EXTEND,
        None,
        torch.tensor([260, 132, 68]),
        num_padding=0,
        mamba_track_indices=torch.tensor([1, 2, 3]),
    )
    assert indices.tolist() == [2, 2, 1]
    assert replay.mamba_track_indices.tolist() == [2, 2, 1]
    assert indices.data_ptr() == pointers[0]
    # Each graph bucket owns its views, so replaying bs=1 cannot corrupt bs=3.
    backend._capture_metadata(1, torch.tensor([3]), ForwardMode.DLLM_EXTEND, None)
    assert indices.tolist() == [2, 2, 1]


@pytest.mark.parametrize(
    "mode,causal", [(ForwardMode.EXTEND, True), (ForwardMode.DLLM_EXTEND, False)]
)
def test_flashinfer_context_and_preview_have_distinct_attention_masks(
    monkeypatch, mode, causal
):
    from unittest.mock import Mock

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.layers.radix_attention import AttentionType

    backend = object.__new__(FlashInferAttnBackend)
    backend.is_dllm_model = True
    backend.prefill_uses_dequant_workspace = False
    backend.kv_cache_quant_method = SimpleNamespace(needs_global_scale=lambda: False)
    backend._get_wrapper_idx = lambda layer: 0
    wrapper = SimpleNamespace(forward=Mock(return_value=torch.zeros(4, 1, 8)))
    backend.prefill_wrapper_ragged = wrapper
    backend.forward_metadata = SimpleNamespace(
        prefill_wrappers=[wrapper],
        use_ragged=False,
        extend_no_prefix=True,
        multi_item_params=None,
        swa_out_cache_loc=None,
    )
    backend.token_to_kv_pool = Mock()
    from sglang.srt.layers import radix_attention
    from sglang.srt.models.clover_moe import CloverAttentionDecoderLayer

    qkv = torch.zeros(4, 8)
    model_layer = object.__new__(CloverAttentionDecoderLayer)
    torch.nn.Module.__init__(model_layer)
    model_layer.attn = radix_attention.RadixAttention(
        1, 8, 8**-0.5, num_kv_heads=1, layer_id=0
    )
    model_layer.attn_preview = radix_attention.RadixAttention(
        1, 8, 8**-0.5, num_kv_heads=1, layer_id=0, attn_type=AttentionType.ENCODER_ONLY
    )
    model_layer.attn_output_gate = False
    model_layer._prepare_qkv_gate = lambda **kw: (qkv, qkv, qkv, None)
    model_layer.o_proj = lambda output: (output, None)
    monkeypatch.setattr(radix_attention, "get_attn_backend", lambda: backend)
    forward_batch = SimpleNamespace(
        forward_mode=mode,
        global_num_token_non_padded_cpu=4,
        out_cache_loc=torch.arange(4),
        out_cache_loc_is_physical=False,
        swa_out_cache_loc=None,
    )
    model_layer.self_attention(None, qkv, forward_batch)
    assert model_layer.attn.attn_type == AttentionType.DECODER
    assert model_layer.attn_preview.attn_type == AttentionType.ENCODER_ONLY
    assert wrapper.forward.call_args.kwargs["causal"] is causal


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
