"""Sequence-length buckets for the partition count of aiter paged_attention_ragged decode.

``AiterAttnBackend`` launches the kernel with fewer partitions for batches of short
requests, through one decode graph per bucket and a matching count in eager mode.
Partitions past a request's length exit early, so any count that covers the longest
request gives the same output; a count that does not drops KV silently.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_RUNNABLE = is_hip()
if _RUNNABLE:
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
    from sglang.srt.layers.attention.aiter_backend import (
        _AITER_PARTITION_SIZE_ROCM,
        AiterAttnBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        _set_capture_attention_variant,
    )

_DEVICE = "cuda"
_NUM_Q_HEADS = 32
_NUM_KV_HEADS = 8
_HEAD_DIM = 128


def _make_backend(context_len=131072, **flags):
    backend = object.__new__(AiterAttnBackend)
    backend.use_mla = flags.get("use_mla", False)
    backend.use_triton_unified_attention = flags.get("unified", False)
    backend.kv_cache_is_vectorized_5d = flags.get("vec5d", False)
    backend.max_context_len = context_len
    backend.max_num_partitions = -(-context_len // _AITER_PARTITION_SIZE_ROCM)
    backend._decode_partition_buckets = backend._build_decode_partition_buckets()
    return backend


def _variants(backend, mode=None, width=1):
    return backend.get_decode_graph_variants(mode or ForwardMode.DECODE, width)


@unittest.skipUnless(_RUNNABLE, "requires HIP with AITER")
class TestDecodePartitionBucketTable(CustomTestCase):
    def test_bucket_table(self):
        self.assertEqual(
            _make_backend(131072)._decode_partition_buckets,
            (("p16", 4096, 16), ("p128", 32768, 128), ("pmax", 131072, 512)),
        )
        self.assertEqual(
            _make_backend(40960)._decode_partition_buckets,
            (("p16", 4096, 16), ("pmax", 40960, 160)),
        )
        # No variants: too few partitions, other decode kernels, other modes, switch off.
        backends = [_make_backend(16384)] + [
            _make_backend(**{flag: True}) for flag in ("use_mla", "unified", "vec5d")
        ]
        with envs.SGLANG_AITER_DECODE_PARTITION_BUCKETS.override(False):
            backends.append(_make_backend())
        for backend in backends:
            self.assertIsNone(_variants(backend))
        for mode, width in (
            (ForwardMode.TARGET_VERIFY, 1),
            (ForwardMode.DLLM_EXTEND, 1),
            (ForwardMode.DECODE, 2),
        ):
            self.assertIsNone(_variants(_make_backend(), mode, width))

    def test_boundaries(self):
        backend = _make_backend()
        variants = _variants(backend)
        self.assertEqual(variants.capture_labels, ("p16", "p128", "pmax"))
        for max_len, label in (
            (1, "p16"),
            (4096, "p16"),
            (4097, "p128"),
            (32768, "p128"),
            (32769, "pmax"),
        ):
            batch = SimpleNamespace(seq_lens_cpu=torch.tensor([1, max_len, 7]))
            self.assertEqual(variants.select(batch), label, max_len)
        # Without a usable host copy of seq_lens, fall back to the full count.
        for seq_lens_cpu in (
            None,
            torch.tensor([]),
            torch.tensor([10], device=_DEVICE),
        ):
            self.assertEqual(
                variants.select(SimpleNamespace(seq_lens_cpu=seq_lens_cpu)), "pmax"
            )
        for max_len in (None, 0):
            self.assertEqual(backend._partitions_for_max_seq_len(max_len), 512)


class _FakeKVPool:
    def __init__(self, k_cache, v_cache):
        self.kv = (k_cache, v_cache)
        self.start_layer = 0
        self.layer_transfer_counter = None
        self.layer_router = PlainPoolRouter(self, "full")

    def get_kv_buffer(self, _layer_id):
        return self.kv


@unittest.skipUnless(_RUNNABLE, "requires HIP with AITER")
class TestDecodePartitionBucketMetadata(CustomTestCase):
    """Eager and capture metadata, then the real kernel through forward_decode."""

    def setUp(self):
        torch.manual_seed(0)

    def tearDown(self):
        _set_capture_attention_variant(None)

    def _make_runnable_backend(self, seq_lens, kv_dtype):
        bs, max_len = len(seq_lens), max(seq_lens)
        backend = _make_backend()
        backend.use_sliding_window_kv_pool = False
        backend.decode_uses_page_table = False
        backend._kv_indices_scratch = None
        backend.kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=_DEVICE)
        backend.kv_last_page_len = torch.ones(bs, dtype=torch.int32, device=_DEVICE)
        # Each request's KV is scattered so that reading past its range is wrong.
        slots = torch.randperm(sum(seq_lens), device=_DEVICE).to(torch.int32)
        backend.req_to_token = torch.zeros(
            bs, max_len, dtype=torch.int32, device=_DEVICE
        )
        for i, chunk in enumerate(slots.split(seq_lens)):
            backend.req_to_token[i, : len(chunk)] = chunk
        k, v = torch.randn(2, len(slots), _NUM_KV_HEADS, _HEAD_DIM, device=_DEVICE).to(
            kv_dtype
        )
        backend.kv_cache_dtype = kv_dtype
        backend.token_to_kv_pool = _FakeKVPool(k, v)
        backend.input_dtype = torch.bfloat16
        backend.k_scale = backend.v_scale = torch.ones(1, device=_DEVICE)
        rows = bs * _NUM_Q_HEADS * backend.max_num_partitions
        backend.workspace_buffer = torch.empty(
            rows * (_HEAD_DIM + 2) * 4, dtype=torch.uint8, device=_DEVICE
        )
        return backend

    def _forward_batch(self, seq_lens, spec_info=None):
        lens = torch.tensor(seq_lens, dtype=torch.int32)
        return SimpleNamespace(
            batch_size=len(seq_lens),
            forward_mode=ForwardMode.DECODE,
            spec_info=spec_info,
            seq_lens=lens.to(_DEVICE),
            seq_lens_cpu=lens,
            seq_lens_sum=int(lens.sum()),
            req_pool_indices=torch.arange(len(seq_lens), device=_DEVICE),
        )

    def _decode(self, backend, forward_batch, num_partitions):
        backend.forward_metadata.decode_num_partitions = num_partitions
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=_NUM_Q_HEADS,
            tp_k_head_num=_NUM_KV_HEADS,
            tp_v_head_num=_NUM_KV_HEADS,
            qk_head_dim=_HEAD_DIM,
            v_head_dim=_HEAD_DIM,
            k_scale=None,
            v_scale=None,
            sliding_window_size=-1,
            logit_cap=0.0,
            scaling=_HEAD_DIM**-0.5,
        )
        gen = torch.Generator(_DEVICE).manual_seed(1)
        q = torch.randn(
            forward_batch.batch_size,
            _NUM_Q_HEADS * _HEAD_DIM,
            device=_DEVICE,
            dtype=torch.bfloat16,
            generator=gen,
        )
        return backend.forward_decode(
            q, None, None, layer, forward_batch, save_kv_cache=False
        )

    def test_eager_decode_matches_max_partitions(self):
        cases = (
            ([1, 4095, 4096], 16),
            ([4097, 1], 128),
            ([32768, 300, 4097], 128),
            ([32769, 5], 512),
        )
        for kv_dtype in (torch.bfloat16, fp8_dtype):
            for seq_lens, expected in cases:
                with self.subTest(kv_dtype=str(kv_dtype), max_len=max(seq_lens)):
                    backend = self._make_runnable_backend(seq_lens, kv_dtype)
                    forward_batch = self._forward_batch(seq_lens)
                    backend.init_forward_metadata(forward_batch)
                    num_partitions = backend.forward_metadata.decode_num_partitions
                    self.assertEqual(num_partitions, expected)
                    out = self._decode(backend, forward_batch, num_partitions)
                    ref = self._decode(backend, forward_batch, None)
                    self.assertTrue(torch.equal(out, ref))

    def test_too_few_partitions_drop_kv(self):
        # Too few partitions must change the output; also the only check that
        # forward_decode hands decode_num_partitions to the kernel.
        seq_lens = [4097, 100]
        backend = self._make_runnable_backend(seq_lens, torch.bfloat16)
        forward_batch = self._forward_batch(seq_lens)
        backend.init_forward_metadata(forward_batch)
        out = self._decode(backend, forward_batch, 16)
        ref = self._decode(backend, forward_batch, None)
        self.assertFalse(torch.equal(out[0], ref[0]))

    def test_eager_speculative_decode_keeps_max_partitions(self):
        seq_lens = [10, 20]
        backend = self._make_runnable_backend(seq_lens, torch.bfloat16)
        spec_info = SimpleNamespace(
            kv_indptr=torch.tensor([0, 10, 30], dtype=torch.int32, device=_DEVICE),
            kv_indices=torch.arange(30, dtype=torch.int32, device=_DEVICE),
        )
        backend.init_forward_metadata(self._forward_batch(seq_lens, spec_info))
        self.assertIsNone(backend.forward_metadata.decode_num_partitions)

    def test_capture_uses_the_variant_partitions(self):
        backend = _make_backend()
        backend.use_sliding_window_kv_pool = False

        def apply_cuda_graph_metadata(**_kwargs):
            backend.forward_metadata = SimpleNamespace(decode_num_partitions=None)

        backend._apply_cuda_graph_metadata = apply_cuda_graph_metadata
        forward_batch = self._forward_batch([1, 1])
        for label, expected in (
            ("p16", 16),
            ("p128", 128),
            ("pmax", 512),
            (None, None),
        ):
            _set_capture_attention_variant(label)
            backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
            self.assertEqual(
                backend.forward_metadata.decode_num_partitions, expected, label
            )
        _set_capture_attention_variant("p16")
        backend.init_forward_metadata_out_graph(forward_batch)
        self.assertIsNone(backend.forward_metadata.decode_num_partitions)
        _set_capture_attention_variant("dense")
        with self.assertRaises(ValueError):
            backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)


if __name__ == "__main__":
    unittest.main()
