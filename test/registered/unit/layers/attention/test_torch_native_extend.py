"""CPU numerical coverage for prefix-aware torch-native extend attention."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import torch_native_backend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference_attention(
    *,
    query,
    key,
    value,
    req_to_token,
    req_indices,
    prefix_lens,
    extend_lens,
    seq_lens,
    encoder_lens,
    causal,
    window,
    cross_attention,
    scale,
):
    expected = []
    start_q = 0
    for i, extend_len in enumerate(extend_lens):
        encoder_len = encoder_lens[i]
        kv_start = 0 if cross_attention else encoder_len
        kv_end = encoder_len if cross_attention else encoder_len + seq_lens[i]
        token_ids = req_to_token[req_indices[i], kv_start:kv_end]
        q = query[start_q : start_q + extend_len].transpose(0, 1)
        k = key[token_ids].float().transpose(0, 1)
        v = value[token_ids].float().transpose(0, 1)
        k = k.repeat_interleave(query.shape[1] // key.shape[1], dim=0)
        v = v.repeat_interleave(query.shape[1] // key.shape[1], dim=0)
        logits = q @ k.transpose(-1, -2) * scale
        if causal:
            # Enumerate visible positions independently of the backend mask.
            visible = torch.tensor(
                [
                    [
                        k_pos <= prefix_lens[i] + q_pos
                        and (window is None or k_pos >= prefix_lens[i] + q_pos - window)
                        for k_pos in range(len(token_ids))
                    ]
                    for q_pos in range(extend_len)
                ],
                dtype=torch.bool,
            ).reshape(extend_len, len(token_ids))
            logits = logits.masked_fill(~visible, -torch.inf)
        expected.append((logits.softmax(dim=-1) @ v).transpose(0, 1))
        start_q += extend_len

    return torch.cat(expected)


def _set_cpu_metadata(batch, cpu_metadata):
    if cpu_metadata:
        batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
        batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.extend_prefix_lens_cpu = batch.extend_prefix_lens.tolist()
        batch.extend_seq_lens_cpu = batch.extend_seq_lens.tolist()
        batch.encoder_lens_cpu = (
            batch.encoder_lens.tolist() if batch.encoder_lens is not None else None
        )
        if cpu_metadata == "partial":
            batch.extend_prefix_lens_cpu = None
        elif cpu_metadata == "unpadded_encoder":
            batch.encoder_lens_cpu = batch.encoder_lens[:-1].tolist()
        elif cpu_metadata == "stale":
            batch.req_pool_indices_cpu = torch.full_like(batch.req_pool_indices, 1000)
            batch.seq_lens_cpu = torch.zeros_like(batch.seq_lens)
            batch.extend_prefix_lens_cpu = [0] * batch.batch_size
            batch.extend_seq_lens_cpu = [0] * batch.batch_size
            batch.encoder_lens_cpu = [0] * batch.batch_size


def _run_attention(
    *,
    query,
    key,
    value,
    req_to_token,
    batch,
    causal,
    window,
    cross_attention,
    public_forward,
    profile_metadata,
    scale,
):
    backend = object.__new__(TorchNativeAttnBackend)
    output = query.new_empty(query.shape[0], query.shape[1], value.shape[2])
    with patch.object(
        torch_native_backend,
        "scaled_dot_product_attention",
        wraps=scaled_dot_product_attention,
    ) as sdpa:
        if public_forward:
            backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
            backend.token_to_kv_pool = SimpleNamespace(
                get_key_buffer=lambda _: key, get_value_buffer=lambda _: value
            )
            layer = SimpleNamespace(
                layer_id=0,
                tp_q_head_num=query.shape[1],
                tp_k_head_num=key.shape[1],
                qk_head_dim=query.shape[2],
                v_head_dim=value.shape[2],
                scaling=scale,
                is_cross_attention=cross_attention,
                attn_type=AttentionType.DECODER
                if causal
                else AttentionType.ENCODER_ONLY,
                sliding_window_size=window,
            )
            with (
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                if profile_metadata
                else nullcontext()
            ) as profiler:
                output = backend.forward_extend(
                    q=query.flatten(1),
                    k=None,
                    v=None,
                    layer=layer,
                    forward_batch=batch,
                    save_kv_cache=False,
                ).view_as(output)
        else:
            backend._run_sdpa_forward_extend(
                query=query,
                output=output,
                k_cache=key,
                v_cache=value,
                req_to_token=req_to_token,
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                extend_prefix_lens=batch.extend_prefix_lens,
                extend_seq_lens=batch.extend_seq_lens,
                encoder_lens=batch.encoder_lens,
                scaling=scale,
                enable_gqa=query.shape[1] != key.shape[1],
                causal=causal,
                is_cross_attn=cross_attention,
                sliding_window_size=window,
            )

    scalar_reads = None
    if profile_metadata:
        scalar_reads = sum(
            event.count
            for event in profiler.key_averages()
            if event.key == "aten::_local_scalar_dense"
        )
    return (
        output,
        [call.args[0].shape[-2] for call in sdpa.call_args_list],
        scalar_reads,
    )


class TestTorchNativeExtend(CustomTestCase):
    def _check_attention(
        self,
        *,
        prefix_lens,
        extend_lens,
        causal=True,
        window=None,
        encoder_lens=None,
        cross_attention=False,
        num_kv_heads=2,
        cache_dtype=torch.float32,
        public_forward=False,
        cpu_metadata=False,
        forward_mode=ForwardMode.EXTEND,
        profile_metadata=False,
    ):
        generator = torch.Generator().manual_seed(42)
        batch_size = len(prefix_lens)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        has_encoder = encoder_lens is not None
        encoder_lens = encoder_lens or [0] * batch_size
        cache_lens = [s + e for s, e in zip(seq_lens, encoder_lens)]
        num_heads, qk_dim, v_dim = 4, 8, 6
        query = torch.randn(sum(extend_lens), num_heads, qk_dim, generator=generator)
        key = torch.randn(
            sum(cache_lens), num_kv_heads, qk_dim, generator=generator
        ).to(cache_dtype)
        value = torch.randn(
            sum(cache_lens), num_kv_heads, v_dim, generator=generator
        ).to(cache_dtype)
        slots = torch.randperm(sum(cache_lens), generator=generator)
        req_indices = torch.arange(batch_size - 1, -1, -1)
        req_to_token = torch.zeros(batch_size, max(cache_lens), dtype=torch.int64)
        start = 0
        for req_idx, length in zip(req_indices, cache_lens):
            req_to_token[req_idx, :length] = slots[start : start + length]
            start += length

        batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=torch.zeros(sum(extend_lens), dtype=torch.int64),
            seq_lens_sum=sum(seq_lens),
            req_pool_indices=req_indices,
            seq_lens=torch.tensor(seq_lens),
            extend_prefix_lens=torch.tensor(prefix_lens),
            extend_seq_lens=torch.tensor(extend_lens),
            encoder_lens=torch.tensor(encoder_lens) if has_encoder else None,
            out_cache_loc=None,
            encoder_out_cache_loc=None,
        )
        _set_cpu_metadata(batch, cpu_metadata)
        options = dict(
            query=query,
            key=key,
            value=value,
            req_to_token=req_to_token,
            causal=causal,
            window=window,
            cross_attention=cross_attention,
            scale=0.3,
        )
        expected = _reference_attention(
            **options,
            req_indices=req_indices,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
        )
        output, query_lengths, scalar_reads = _run_attention(
            **options,
            batch=batch,
            public_forward=public_forward,
            profile_metadata=profile_metadata,
        )

        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
        # Cached prefix queries must not be recomputed and discarded by SDPA.
        self.assertEqual(query_lengths, list(extend_lens))
        return scalar_reads

    def test_cpu_metadata_avoids_scalar_reads(self):
        """Existing host mirrors remove device scalar reads from public extend."""
        options = dict(
            prefix_lens=(0, 5, 8),
            extend_lens=(3, 4, 1),
            public_forward=True,
            profile_metadata=True,
        )
        self.assertGreater(self._check_attention(**options), 0)
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self._check_attention(
                        **options, cpu_metadata=True, forward_mode=mode
                    ),
                    0,
                )

    def test_optional_cpu_metadata(self):
        """Missing mirrors and an unpadded encoder mirror retain correct slicing."""
        for mirrors in (True, "partial", "unpadded_encoder"):
            for cross_attention in (False, True):
                with self.subTest(mirrors=mirrors, cross_attention=cross_attention):
                    self._check_attention(
                        prefix_lens=(2, 5),
                        extend_lens=(3, 1),
                        encoder_lens=(7, 4),
                        causal=not cross_attention,
                        cross_attention=cross_attention,
                        public_forward=True,
                        cpu_metadata=mirrors,
                    )

    def test_speculative_modes_ignore_stale_cpu_metadata(self):
        """Speculative device metadata can change independently of host mirrors."""
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                self._check_attention(
                    prefix_lens=(2, 5),
                    extend_lens=(3, 1),
                    public_forward=True,
                    cpu_metadata="stale",
                    forward_mode=mode,
                )

    def test_causal_prefix_and_ragged_batch(self):
        """A rectangular causal mask must align each query after its own prefix."""
        self._check_attention(prefix_lens=(0, 3, 7), extend_lens=(4, 2, 1))

    def test_zero_extend_rows(self):
        """Padding rows contribute no queries and preserve later request offsets."""
        for public_forward in (False, True):
            for window in (None, 2):
                with self.subTest(public_forward=public_forward, window=window):
                    self._check_attention(
                        prefix_lens=(0, 5, 3),
                        extend_lens=(0, 2, 0),
                        window=window,
                        public_forward=public_forward,
                        cpu_metadata=public_forward,
                    )

    def test_sliding_window_prefix(self):
        """Window edges use absolute query positions even when the prefix is omitted."""
        for window in (0, 2, 20):
            with self.subTest(window=window):
                self._check_attention(
                    prefix_lens=(0, 5, 8), extend_lens=(3, 4, 1), window=window
                )

    def test_noncausal_prefix(self):
        """Noncausal queries retain access to future keys after removing padding."""
        for public_forward in (False, True):
            with self.subTest(public_forward=public_forward):
                self._check_attention(
                    prefix_lens=(0, 4),
                    extend_lens=(3, 2),
                    causal=False,
                    public_forward=public_forward,
                    cpu_metadata=public_forward,
                )

    def test_encoder_decoder_self_attention(self):
        """Decoder self attention must skip encoder cache slots for each request."""
        self._check_attention(
            prefix_lens=(2, 5),
            extend_lens=(3, 1),
            encoder_lens=(7, 4),
            public_forward=True,
        )

    def test_cross_attention(self):
        """Decoder queries attend to encoder keys of an independent sequence length."""
        self._check_attention(
            prefix_lens=(3, 2),
            extend_lens=(2, 4),
            encoder_lens=(7, 3),
            causal=False,
            cross_attention=True,
            public_forward=True,
        )

    def test_cache_dtype_conversion(self):
        """Query and cache dtypes may differ; conversion must preserve the output."""
        self._check_attention(
            prefix_lens=(6,),
            extend_lens=(2,),
            num_kv_heads=4,
            cache_dtype=torch.float16,
        )


if __name__ == "__main__":
    unittest.main()
