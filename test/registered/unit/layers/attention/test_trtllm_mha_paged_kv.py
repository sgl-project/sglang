"""trtllm_mha's paged K/V views keep the pool's slot stride.

``_reshape_paged_kv_cache`` turns a per-layer ``[slots, ...]`` buffer into the
``[pages, heads, page_size, head_dim]`` layout the trtllm kernels take. Its
nvfp4 callers also hand it block-scale rows that arrive FLAT
(``[slots, heads * head_dim // 16]``), so the row has to be shaped before the
pages are split: shaping it afterwards, or splitting with ``.view()``,
re-derives the size-1 page dimension's stride from the row at
``page_size == 1``. On a strided buffer (the unified pool's slots are a whole
entry apart, wider than one row) that stride is then wrong.

An HND pool hands over page-major ``[pages, heads, page_size, head_dim]``
buffers instead, which are already the kernel layout: they must reach the
kernel as themselves, head for head and token for token.

CPU-only.

    python -m pytest test/registered/unit/layers/attention/test_trtllm_mha_paged_kv.py -v
"""

import contextlib
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mha_backend
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    TRTLLMMHAMetadata,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _strided_rows(num_slots, row_shape, slot_stride):
    # Distinct values, so equal values mean equal addresses.
    backing = torch.arange(num_slots * slot_stride, dtype=torch.float32)
    return backing.as_strided(
        (num_slots, *row_shape), (slot_stride, *torch.empty(row_shape).stride())
    )


def _fake_runner(*, req_to_token_row_width, kv_cache_dtype=torch.bfloat16, model=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            context_len=64,
            hidden_size=32,
            num_attention_heads=4,
            head_dim=8,
            v_head_dim=8,
        ),
        prefill_attention_backend_str="trtllm_mha",
        decode_attention_backend_str="trtllm_mha",
        kv_cache_dtype=kv_cache_dtype,
        dtype=torch.bfloat16,
        page_size=16,
        req_to_token_pool=SimpleNamespace(
            row_width=req_to_token_row_width,
            req_to_token=torch.zeros((2, req_to_token_row_width)),
        ),
        device="cpu",
        is_draft_worker=False,
        max_running_requests=4,
        model=torch.nn.Identity() if model is None else model,
    )


def _construct_backend(runner, *, configured=0, xqa=False, native_fp4=False):
    """Build a TRTLLMHAAttnBackend on CPU; returns it and the mocked
    ``_init_full_attn_kv_splits``."""
    access = (
        SimpleNamespace(
            kind=trtllm_mha_backend.KVCacheAttentionAccessKind.NATIVE_FP4,
            scale_recipe="nvfp4",
        )
        if native_fp4
        else SimpleNamespace(kind=trtllm_mha_backend.KVCacheAttentionAccessKind.PLAIN)
    )

    def parent_init(backend, model_runner, *args, **kwargs):
        backend.req_to_token_pool = model_runner.req_to_token_pool
        backend.max_kv_len = model_runner.req_to_token_pool.row_width
        backend.kv_cache_quant_method = SimpleNamespace(
            resolve_attention_access=lambda *args: access
        )
        backend.runner_ids = SimpleNamespace(is_translating=False)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            trtllm_mha_backend.envs.SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS.override(
                configured
            )
        )
        stack.enter_context(
            patch.object(
                trtllm_mha_backend.FlashInferAttnBackend, "__init__", parent_init
            )
        )
        stack.enter_context(
            patch.object(
                trtllm_mha_backend,
                "get_platform",
                return_value=SimpleNamespace(is_sm90=xqa, is_sm120=False),
            )
        )
        stack.enter_context(
            patch.object(
                trtllm_mha_backend,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_eagle_topk=1, speculative_num_draft_tokens=None
                ),
            )
        )
        stack.enter_context(
            patch.object(
                trtllm_mha_backend,
                "get_buffer",
                side_effect=lambda *args: torch.ones(1),
            )
        )
        stack.enter_context(
            patch.object(
                trtllm_mha_backend, "resolve_window_kv_pool", return_value=None
            )
        )
        stack.enter_context(
            patch.object(
                trtllm_mha_backend,
                "make_persistent_multi_ctas_kv_counter_buffer",
                return_value=None,
            )
        )
        if native_fp4:
            stack.enter_context(
                patch.object(
                    trtllm_mha_backend,
                    "get_schedule",
                    return_value=SimpleNamespace(max_prefill_tokens=64),
                )
            )
            stack.enter_context(
                patch.object(
                    trtllm_mha_backend,
                    "max_prefill_buffer_tokens",
                    return_value=64,
                )
            )
            stack.enter_context(
                patch.object(
                    trtllm_mha_backend,
                    "max_speculative_num_draft_tokens",
                    return_value=1,
                )
            )
            stack.enter_context(
                patch.object(
                    trtllm_mha_backend,
                    "get_exec",
                    return_value=SimpleNamespace(
                        graph=SimpleNamespace(
                            cuda_graph_config=SimpleNamespace(
                                decode=SimpleNamespace(max_bs=4)
                            )
                        )
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    trtllm_mha_backend,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_size=1),
                )
            )
        initialize = stack.enter_context(
            patch.object(TRTLLMHAAttnBackend, "_init_full_attn_kv_splits")
        )
        return TRTLLMHAAttnBackend(runner), initialize


class TestTRTLLMHAPagedKV(unittest.TestCase):
    def test_graph_variants_keep_decode_prefix_sums_and_verify_row_map_alive(self):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.device = "cpu"
        backend.max_num_pages = 4
        backend.speculative_num_draft_tokens = 3
        backend.use_sliding_window_kv_pool = False
        backend._swa_kv_pool = None
        backend.expand_encoder_only_verify = True
        backend.page_size = 1
        backend.req_to_token = torch.zeros((4, 4), dtype=torch.int32)
        backend._swa_full_to_swa_mapping = None
        backend.runner_ids = SimpleNamespace(is_translating=False)
        backend.init_cuda_graph_state(4, 12)

        def build(num_tokens, forward_mode):
            backend._build_cuda_graph_metadata(
                2, num_tokens, forward_mode, None, "cpu", ragged_verify_layout=None
            )

        build(2, ForwardMode.DECODE)
        captured = backend.decode_cuda_graph_metadata[2]
        build(2, ForwardMode.DECODE)

        def update(**kwargs):
            lengths = kwargs["seq_lens"].to(torch.int32)
            kwargs["cache_seqlens"].copy_(lengths)
            kwargs["cu_seqlens_k"][1:].copy_(lengths.cumsum(0))

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.update_trtllm_mha_graph_metadata",
            side_effect=update,
        ):
            backend._apply_cuda_graph_metadata(
                2, torch.tensor([0, 1]), torch.tensor([7, 11]), ForwardMode.DECODE, None
            )
        self.assertEqual(
            captured.cu_seqlens_q.data_ptr(),
            backend.decode_cuda_graph_metadata[2].cu_seqlens_q.data_ptr(),
        )
        torch.testing.assert_close(
            captured.cu_seqlens_k, torch.tensor([0, 7, 18], dtype=torch.int32)
        )

        build(6, ForwardMode.TARGET_VERIFY)
        captured_map = backend.target_verify_metadata[2].encoder_row_map
        build(6, ForwardMode.TARGET_VERIFY)
        self.assertEqual(
            captured_map.data_ptr(),
            backend.target_verify_metadata[2].encoder_row_map.data_ptr(),
        )
        torch.testing.assert_close(captured_map, torch.tensor([0, 0, 0, 1, 1, 1]))

    def test_reshape_keeps_slot_stride_for_packed_and_flat_rows(self):
        H, D, pages = 4, 32, 3
        slot_stride = 3 * H * D
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        # 16 is the smallest page size trtllm_mha runs at; at 1, `.view()`
        # would give the size-1 page dim the row stride.
        for page_size in (1, 16):
            backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
            backend.page_size = page_size
            # (row shape handed in, head_dim argument): nvfp4 packed K/V rows,
            # then the flat block-scale rows.
            for row_shape, head_dim in (
                ((H, D // 2), D // 2),
                ((H * D // 16,), D // 16),
            ):
                with self.subTest(page_size=page_size, row_shape=row_shape):
                    k = _strided_rows(pages * page_size, row_shape, slot_stride)
                    v = _strided_rows(pages * page_size, row_shape, slot_stride)
                    outs = backend._reshape_paged_kv_cache(k, v, layer, head_dim)
                    for flat, out in zip((k, v), outs):
                        self.assertEqual(
                            tuple(out.shape), (pages, H, page_size, head_dim)
                        )
                        self.assertEqual(out.stride(0), page_size * slot_stride)
                        self.assertEqual(out.stride(2), slot_stride)
                        for token in range(pages * page_size):
                            page, slot = divmod(token, page_size)
                            self.assertTrue(
                                torch.equal(
                                    out[page, :, slot],
                                    flat[token].reshape(H, head_dim),
                                )
                            )

    def test_reshape_hands_the_hnd_pool_buffer_through(self):
        H, D, pages, page_size = 4, 32, 3, 16
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.page_size = page_size
        layer = SimpleNamespace(tp_k_head_num=H, tp_v_head_num=H)
        # Distinct values, so equal values mean equal (page, head, token).
        k = torch.arange(pages * H * page_size * D, dtype=torch.float32).view(
            pages, H, page_size, D
        )
        v = k + 0.5
        outs = backend._reshape_paged_kv_cache(k, v, layer, D)
        for buf, out in zip((k, v), outs):
            self.assertEqual(out.shape, buf.shape)
            self.assertEqual(out.stride(), buf.stride())
            self.assertEqual(out.data_ptr(), buf.data_ptr())
            self.assertTrue(torch.equal(out, buf))


class TestTRTLLMHAPageTableWidth(CustomTestCase):
    def test_page_table_spans_the_request_row(self):
        # A spec verify reads past the model context, up to the row width; a
        # narrower page table would spill its fill into the next request's row.
        backend, _ = _construct_backend(_fake_runner(req_to_token_row_width=72))
        self.assertEqual(backend.max_num_pages, 5)  # ceil(72 / 16), not 64 / 16


class TestTRTLLMHAFullAttnKVSplit(CustomTestCase):
    """SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS scopes the SM count trtllm-gen
    decode reads, which sets its KV splits per (request, KV head) to
    sm_count // (batch * kv_heads) and its grid to that many CTAs."""

    SMS, KV_HEADS, SPLITS = 148, 4, 8

    def test_constructor_keeps_main_kv_split_configuration(self):
        runner = _fake_runner(req_to_token_row_width=64)
        for configured, xqa, expected in ((0, False, 0), (8, False, 8), (8, True, 0)):
            with self.subTest(configured=configured, xqa=xqa):
                backend, initialize = _construct_backend(
                    runner, configured=configured, xqa=xqa
                )
                self.assertEqual(backend._full_attn_kv_splits, expected)
                if expected > 1:
                    initialize.assert_called_once_with(v_head_dim=8)
                else:
                    initialize.assert_not_called()

    def _backend(self, *, min_bs=12, max_split_ctas=1 << 20):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend._full_attn_kv_splits = self.SPLITS
        backend._full_attn_kv_split_min_bs = min_bs
        backend._device_sm_count = self.SMS
        backend._full_attn_kv_split_max_ctas = max_split_ctas
        return backend

    def _flashinfer(self, decode):
        return patch.object(
            trtllm_mha_backend,
            "flashinfer",
            SimpleNamespace(decode=decode),
            create=True,
        )

    def _kv_cache(self):
        k_cache = torch.empty(2, self.KV_HEADS, 16, 8)  # HND pages
        return (k_cache, k_cache)

    def _scope_sm_count(self, *, window_left, batch_size, **backend_kwargs):
        backend = self._backend(**backend_kwargs)
        decode = SimpleNamespace(get_device_sm_count=lambda device: self.SMS)
        with self._flashinfer(decode):
            with backend._full_attn_kv_split_scope(
                window_left=window_left,
                batch_size=batch_size,
                kv_cache=self._kv_cache(),
            ):
                seen = decode.get_device_sm_count(None)
            self.assertEqual(decode.get_device_sm_count(None), self.SMS)
        return seen

    def test_verify_launches_inside_the_scope_sized_by_requests(self):
        """A fixed-q-len verify launches q_len_per_req query rows per request and
        FlashInfer counts requests, so the launch must run inside the scope and
        size it by query rows // q_len_per_req."""
        backend = self._backend()
        backend.decode_seq_len_splits = 1
        backend.q_data_type = torch.bfloat16
        backend.max_kv_len = 1024
        backend.workspace_buffer = backend._multi_ctas_kv_counter_buffer = None
        seen = []
        decode = SimpleNamespace(get_device_sm_count=lambda device: self.SMS)

        def launch(**kwargs):
            seen.append(decode.get_device_sm_count(None))
            return kwargs["query"]

        decode.trtllm_batch_decode_with_kv_cache = launch
        bs, q_len_per_req = 32, 8
        with self._flashinfer(decode):
            backend._run_fixed_q_len_decode(
                torch.empty(bs * q_len_per_req, 16, 8),
                self._kv_cache(),
                torch.zeros(bs, 2, dtype=torch.int32),
                torch.ones(bs, dtype=torch.int32),
                bmm1_scale=1.0,
                bmm2_scale=1.0,
                window_left=-1,
                sinks=None,
                q_len_per_req=q_len_per_req,
            )
        self.assertEqual(seen, [self.SPLITS * self.KV_HEADS * bs])

    def test_splits_only_full_attention_batches_flashinfer_runs_unsplit(self):
        # Full attention at bs 32: 4 * 32 request-heads exceed the SMs, so
        # FlashInfer would run one CTA each; the scope makes it 8.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=32),
            self.SPLITS * self.KV_HEADS * 32,
        )
        # Sliding-window layers, batches under MIN_BS, and batches FlashInfer
        # already splits as finely keep the device SM count.
        self.assertEqual(self._scope_sm_count(window_left=127, batch_size=32), self.SMS)
        self.assertEqual(self._scope_sm_count(window_left=-1, batch_size=8), self.SMS)
        # At bs 4 FlashInfer already gives 148 // 16 = 9 splits.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=4, min_bs=1), self.SMS
        )

    def test_split_ctas_stay_within_the_workspace_and_counters(self):
        # 4 * 32 base CTAs and room for 640 split CTAs: 5 splits, not 8.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=32, max_split_ctas=640),
            5 * self.KV_HEADS * 32,
        )
        # Room for under 2 splits: no override.
        self.assertEqual(
            self._scope_sm_count(window_left=-1, batch_size=32, max_split_ctas=255),
            self.SMS,
        )

    def test_scope_restores_the_sm_count_after_an_error(self):
        decode = SimpleNamespace(get_device_sm_count=lambda device: self.SMS)
        with self._flashinfer(decode):
            with self.assertRaises(RuntimeError):
                with trtllm_mha_backend._trtllm_decode_sm_count(1024):
                    raise RuntimeError("kernel launch failed")
            self.assertEqual(decode.get_device_sm_count(None), self.SMS)


class _AttnTypeModule(torch.nn.Module):
    def __init__(self, attn_type: AttentionType):
        super().__init__()
        self.attn_type = attn_type


def _paged_gqa_single_token_decode(
    *,
    query: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: torch.Tensor | None = None,
    q_len_per_req: int = 1,
    **kwargs,
) -> torch.Tensor:
    """Pure-PyTorch reference for ``trtllm_batch_decode_with_kv_cache`` at
    ``q_len_per_req == 1`` with optional per-query-head attention sinks."""
    assert q_len_per_req == 1
    k_cache, v_cache = kv_cache
    bs, num_q_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[1]
    assert num_q_heads % num_kv_heads == 0
    gqa_group = num_q_heads // num_kv_heads

    outputs = []
    for row in range(bs):
        seq_len = int(seq_lens[row].item())
        pages = block_tables[row].long()
        # [pages, kv_heads, page_size, head_dim] -> [kv_heads, tokens, head_dim]
        k = (
            k_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)[:, :seq_len]
            .repeat_interleave(gqa_group, dim=0)
            .float()
        )
        v = (
            v_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)[:, :seq_len]
            .repeat_interleave(gqa_group, dim=0)
            .float()
        )
        q_row = query[row].float()
        scores = torch.einsum("hd,htd->ht", q_row, k) * float(bmm1_scale)
        if sinks is not None:
            logits = torch.cat([scores, sinks.float().unsqueeze(-1)], dim=-1)
            probs = torch.softmax(logits, dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        out_row = torch.einsum("ht,htd->hd", probs, v) * float(bmm2_scale)
        outputs.append(out_row.to(query.dtype))
    return torch.stack(outputs, dim=0)


class TestTRTLLMHAEncoderOnlyVerifyFold(CustomTestCase):
    def test_expand_encoder_only_verify_selection(self):
        encoder_model = torch.nn.Sequential(
            _AttnTypeModule(AttentionType.DECODER),
            _AttnTypeModule(AttentionType.ENCODER_ONLY),
        )
        decoder_model = torch.nn.Sequential(_AttnTypeModule(AttentionType.DECODER))

        cases = [
            # GenMHA with BF16 or FP8 KV cache folds ENCODER_ONLY verify.
            (encoder_model, torch.bfloat16, False, False, False),
            (encoder_model, torch.float8_e4m3fn, False, False, False),
            # XQA and SM100 native NVFP4 keep the expanded single-token path.
            (encoder_model, torch.bfloat16, True, False, True),
            (encoder_model, torch.float8_e4m3fn, True, False, True),
            (encoder_model, torch.bfloat16, False, True, True),
            # Models without ENCODER_ONLY layers never expand.
            (decoder_model, torch.bfloat16, False, False, False),
            (decoder_model, torch.float8_e4m3fn, False, False, False),
            (decoder_model, torch.bfloat16, True, False, False),
            (decoder_model, torch.bfloat16, False, True, False),
        ]
        for model, kv_dtype, xqa, native_fp4, expected in cases:
            with self.subTest(
                has_encoder=model is encoder_model,
                kv_dtype=kv_dtype,
                xqa=xqa,
                native_fp4=native_fp4,
            ):
                runner = _fake_runner(
                    req_to_token_row_width=64,
                    kv_cache_dtype=kv_dtype,
                    model=model,
                )
                backend, _ = _construct_backend(runner, xqa=xqa, native_fp4=native_fp4)
                self.assertEqual(backend.expand_encoder_only_verify, expected)

    def test_folded_path_skips_expanded_metadata_buffers(self):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.device = "cpu"
        backend.max_num_pages = 4
        backend.speculative_num_draft_tokens = 7
        backend.use_sliding_window_kv_pool = False
        backend._swa_kv_pool = None
        backend.expand_encoder_only_verify = False
        backend.page_size = 16
        backend.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
        backend._swa_full_to_swa_mapping = None
        backend.runner_ids = SimpleNamespace(is_translating=False)
        backend.init_cuda_graph_state(4, 28)

        self.assertNotIn("encoder_cache_seqlens", backend.target_verify_metadata)
        self.assertNotIn("encoder_page_table", backend.target_verify_metadata)
        metadata = backend._build_cuda_graph_metadata(
            2, 14, ForwardMode.TARGET_VERIFY, None, "cpu", ragged_verify_layout=None
        )
        self.assertIsNone(metadata.encoder_cache_seqlens)
        self.assertIsNone(metadata.encoder_page_table)
        self.assertIsNone(metadata.encoder_row_map)
        self.assertFalse(
            backend._needs_encoder_only_expand(ForwardMode.TARGET_VERIFY, metadata)
        )

    def _run_encoder_verify(
        self,
        *,
        expand: bool,
        q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        window: int,
        kv_heads: int,
        group: int,
        head_dim: int,
        page_size: int,
        sinks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        bs = int(seq_lens.shape[0])
        num_q_heads = kv_heads * group
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.uses_trtllm_gen_native_fp4 = False
        backend.prefill_uses_native_fp4 = False
        backend.decode_uses_native_fp4 = False
        backend.is_xqa_impl = False
        backend.use_fmha_v2 = False
        backend.data_type = torch.float32
        backend.q_data_type = torch.float32
        backend.page_size = page_size
        backend.max_kv_len = page_table.shape[1] * page_size
        backend.workspace_buffer = None
        backend._multi_ctas_kv_counter_buffer = None
        backend.expand_encoder_only_verify = expand
        backend.token_to_kv_pool = SimpleNamespace(
            layer_router=SimpleNamespace(
                read=lambda layer_id: SimpleNamespace(kv=lambda: kv_cache)
            )
        )

        row_map = torch.arange(bs * window) // window
        backend.forward_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=seq_lens,
            max_seq_len_q=window,
            page_table=page_table,
            is_ragged_verify=False,
            encoder_cache_seqlens=seq_lens[row_map] if expand else None,
            encoder_page_table=page_table[row_map] if expand else None,
        )
        layer = SimpleNamespace(
            layer_id=0,
            attn_type=AttentionType.ENCODER_ONLY,
            tp_q_head_num=num_q_heads,
            tp_k_head_num=kv_heads,
            tp_v_head_num=kv_heads,
            head_dim=head_dim,
            scaling=1.0 / math.sqrt(head_dim),
            sliding_window_size=-1,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
        )

        captured = {}

        def fake_decode(**kwargs):
            captured.update(kwargs)
            return _paged_gqa_single_token_decode(**kwargs)

        decode_ns = SimpleNamespace(trtllm_batch_decode_with_kv_cache=fake_decode)
        with patch.object(
            trtllm_mha_backend,
            "flashinfer",
            SimpleNamespace(decode=decode_ns),
            create=True,
        ):
            out = backend.forward_extend(
                q,
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
                sinks=sinks,
            )
        return out, captured

    def test_gqa_fold_matches_expanded_reference(self):
        page_size = 8
        head_dim = 16
        cases = [
            # (bs, window, kv_heads, group, use_sink, non_contiguous)
            (1, 7, 1, 2, False, False),
            (1, 7, 2, 4, True, False),
            (1, 7, 4, 8, True, True),
            (4, 7, 1, 2, False, False),
            (4, 7, 2, 4, False, True),
            (4, 7, 2, 4, True, False),
            (4, 7, 4, 8, True, True),
            (4, 1, 2, 4, True, False),
        ]
        for seed, (bs, window, kv_heads, group, use_sink, non_contiguous) in enumerate(
            cases, start=1
        ):
            with self.subTest(
                bs=bs,
                window=window,
                kv_heads=kv_heads,
                group=group,
                use_sink=use_sink,
                non_contiguous=non_contiguous,
            ):
                gen = torch.Generator().manual_seed(seed)
                num_q_heads = kv_heads * group
                q_dim = num_q_heads * head_dim
                seq_lens_list = [10, 23, 7, 19][:bs]
                seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
                pages_per_req = math.ceil(max(seq_lens_list) / page_size)
                num_pages = bs * pages_per_req + 1

                k_cache = torch.randn(
                    (num_pages, kv_heads, page_size, head_dim),
                    generator=gen,
                    dtype=torch.float32,
                )
                v_cache = torch.randn(
                    (num_pages, kv_heads, page_size, head_dim),
                    generator=gen,
                    dtype=torch.float32,
                )
                page_table = torch.arange(
                    1, 1 + bs * pages_per_req, dtype=torch.int32
                ).view(bs, pages_per_req)

                if non_contiguous:
                    q_backing = torch.randn(
                        (q_dim, bs * window), generator=gen, dtype=torch.float32
                    )
                    q = q_backing.transpose(0, 1)
                    self.assertFalse(q.is_contiguous())
                else:
                    q = torch.randn(
                        (bs * window, q_dim), generator=gen, dtype=torch.float32
                    )

                if use_sink:
                    if non_contiguous:
                        sink_backing = torch.randn(
                            (num_q_heads * 2,), generator=gen, dtype=torch.float32
                        )
                        sinks = sink_backing[::2]
                        self.assertFalse(sinks.is_contiguous())
                    else:
                        sinks = torch.randn(
                            (num_q_heads,), generator=gen, dtype=torch.float32
                        )
                else:
                    sinks = None

                folded_out, folded_call = self._run_encoder_verify(
                    expand=False,
                    q=q,
                    kv_cache=(k_cache, v_cache),
                    page_table=page_table,
                    seq_lens=seq_lens,
                    window=window,
                    kv_heads=kv_heads,
                    group=group,
                    head_dim=head_dim,
                    page_size=page_size,
                    sinks=sinks,
                )
                expanded_out, expanded_call = self._run_encoder_verify(
                    expand=True,
                    q=q,
                    kv_cache=(k_cache, v_cache),
                    page_table=page_table,
                    seq_lens=seq_lens,
                    window=window,
                    kv_heads=kv_heads,
                    group=group,
                    head_dim=head_dim,
                    page_size=page_size,
                    sinks=sinks,
                )

                self.assertEqual(
                    tuple(folded_call["query"].shape),
                    (bs, window * num_q_heads, head_dim),
                )
                self.assertEqual(
                    tuple(folded_call["block_tables"].shape), (bs, pages_per_req)
                )
                self.assertEqual(tuple(folded_call["seq_lens"].shape), (bs,))
                self.assertEqual(folded_call["q_len_per_req"], 1)
                self.assertEqual(
                    tuple(expanded_call["query"].shape),
                    (bs * window, num_q_heads, head_dim),
                )
                if use_sink:
                    self.assertEqual(
                        tuple(folded_call["sinks"].shape), (window * num_q_heads,)
                    )
                else:
                    self.assertIsNone(folded_call["sinks"])

                self.assertEqual(tuple(folded_out.shape), (bs * window, q_dim))
                torch.testing.assert_close(
                    folded_out, expanded_out, atol=1e-6, rtol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
