"""DSA prefill attention captured in the breakable prefill CUDA graph: host-side gates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_prefill_cuda_graph_batch_sizes,
)
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _layerwise_kv_load_configured,
    _min_bitwise_prefill_graph_bucket,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeLoadCounter:
    def __init__(self, num_layers, consumer_index):
        self.num_layers = num_layers
        self.consumer_index = consumer_index
        self.waited = []

    def wait_until(self, layer_id):
        self.waited.append(layer_id)


def _backend(*, min_rows=96, seq_lens_rows=4096, use_mha=False, counter=None):
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend._prefill_graph_seq_lens = (
        None if seq_lens_rows is None else torch.zeros(seq_lens_rows, dtype=torch.int32)
    )
    backend._prefill_graph_min_rows = min_rows
    backend.use_mha = use_mha
    backend.token_to_kv_pool = SimpleNamespace(layer_transfer_counter=counter)
    return backend


class TestBitwiseBucketFloor(CustomTestCase):
    def test_default_buckets_on_gb300(self):
        buckets = generate_prefill_cuda_graph_batch_sizes(8192)
        self.assertEqual(
            _min_bitwise_prefill_graph_bucket(buckets=buckets, num_sms=152), 96
        )

    def test_every_real_row_count_from_the_floor_up_exceeds_half_the_sms(self):
        # A bucket b takes batches of (previous bucket, b] real rows; the sparse
        # MLA kernel keeps one CTA per row only above SMs / 2 rows.
        buckets = generate_prefill_cuda_graph_batch_sizes(8192)
        for num_sms in (16, 78, 132, 148, 152, 170, 304):
            floor = _min_bitwise_prefill_graph_bucket(buckets=buckets, num_sms=num_sms)
            idx = buckets.index(floor)
            self.assertGreater(buckets[idx - 1] + 1, num_sms / 2, num_sms)
            # The floor is the smallest such bucket.
            self.assertLessEqual(buckets[idx - 2] + 1, num_sms / 2, num_sms)

    def test_no_bucket_qualifies(self):
        self.assertEqual(
            _min_bitwise_prefill_graph_bucket(buckets=[16, 32, 64], num_sms=152), 65
        )


class TestPrefillGraphSeqLens(CustomTestCase):
    def test_padding_rows_attend_to_one_slot(self):
        backend = _backend(seq_lens_rows=8)
        backend._refresh_prefill_graph_seq_lens(
            torch.tensor([5, 0, 3], dtype=torch.int32)
        )
        # Attention-TP padding (0) and the bucket's padded rows read length 1.
        self.assertEqual(
            backend._prefill_graph_seq_lens.tolist(), [5, 1, 3, 1, 1, 1, 1, 1]
        )

    def test_batch_wider_than_every_bucket_leaves_the_buffer(self):
        backend = _backend(seq_lens_rows=4)
        backend._prefill_graph_seq_lens.fill_(7)
        backend._refresh_prefill_graph_seq_lens(torch.ones(5, dtype=torch.int32))
        self.assertEqual(backend._prefill_graph_seq_lens.tolist(), [7, 7, 7, 7])

    def test_hicache_load_back_waits_for_every_layer(self):
        in_flight = _FakeLoadCounter(num_layers=3, consumer_index=0)
        _backend(counter=in_flight)._wait_for_layerwise_kv_load()
        self.assertEqual(in_flight.waited, [0, 1, 2])

        idle = _FakeLoadCounter(num_layers=3, consumer_index=-1)
        _backend(counter=idle)._wait_for_layerwise_kv_load()
        self.assertEqual(idle.waited, [])

        _backend(counter=None)._wait_for_layerwise_kv_load()


class TestCanRunPrefillAttentionInGraph(CustomTestCase):
    def _can_run(
        self, backend, *, mode=ForwardMode.EXTEND, rows=128, bcg=True, ctx=True
    ):
        backend._is_in_breakable_cuda_graph = lambda: bcg
        backend._get_tc_piecewise_forward_context = lambda: object() if ctx else None
        return backend.can_run_prefill_attention_in_graph(
            SimpleNamespace(forward_mode=mode), num_rows=rows
        )

    def test_extend_in_the_prefill_graph(self):
        self.assertTrue(self._can_run(_backend()))

    def test_declines(self):
        cases = {
            "path off at init": dict(backend=_backend(seq_lens_rows=None)),
            "bucket below the bitwise floor": dict(backend=_backend(), rows=80),
            "MHA prefill": dict(backend=_backend(use_mha=True)),
            "target verify": dict(backend=_backend(), mode=ForwardMode.TARGET_VERIFY),
            "decode": dict(backend=_backend(), mode=ForwardMode.DECODE),
            "outside a breakable graph": dict(backend=_backend(), bcg=False),
            # The decode breakable backend has no tc_piecewise forward context.
            "decode breakable backend": dict(backend=_backend(), ctx=False),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertFalse(self._can_run(**kwargs))


class TestLayerwiseKvLoadGate(CustomTestCase):
    def _configured(self, *, disagg_backup="none", **memory):
        fields = dict(
            enable_flexkv=False,
            enable_unified_cache_external_linker=False,
            radix_cache_backend=None,
        )
        fields.update(memory)
        with (
            patch.object(dsa_backend, "get_memory", lambda: SimpleNamespace(**fields)),
            patch.object(
                dsa_backend,
                "get_disagg",
                lambda: SimpleNamespace(
                    disaggregation_decode_retraction_backup=disagg_backup
                ),
            ),
        ):
            return _layerwise_kv_load_configured()

    def test_per_layer_loaders_keep_the_eager_break(self):
        self.assertTrue(self._configured(enable_flexkv=True))
        self.assertTrue(self._configured(enable_unified_cache_external_linker=True))
        self.assertTrue(self._configured(radix_cache_backend="plugin"))
        self.assertTrue(self._configured(disagg_backup="host_pool"))

    def test_hicache_alone_keeps_the_captured_path(self):
        # HiCache loads are waited for in full by _wait_for_layerwise_kv_load.
        self.assertFalse(self._configured())


class TestForwardAbsorbCoreGate(CustomTestCase):
    def _backend_for(self, extra_args, topk=torch.zeros(128, 4, dtype=torch.int32)):
        backend = _backend()
        backend.can_run_prefill_attention_in_graph = lambda fb, num_rows: True
        with patch(
            "sglang.srt.models.deepseek_common.attention_forward_methods."
            "forward_mla.get_attn_backend",
            return_value=backend,
        ):
            return (
                backend,
                DeepseekMLAForwardMixin._dsa_prefill_attention_in_graph_backend(
                    SimpleNamespace(), SimpleNamespace(), topk, extra_args
                ),
            )

    def test_fused_rope_fp8_path_only(self):
        fused = dict(cos_sin_cache=torch.zeros(1), is_neox=False, llama_4_scaling=None)
        backend, got = self._backend_for(fused)
        self.assertIs(got, backend)
        for name, extra_args in {
            "no fused RoPE": {},
            "llama 4 scaling": dict(fused, llama_4_scaling=torch.ones(1)),
            "attention sink": dict(fused, attn_sink=torch.zeros(1)),
        }.items():
            with self.subTest(name):
                self.assertIsNone(self._backend_for(extra_args)[1])
        self.assertIsNone(self._backend_for(fused, topk=None)[1])


if __name__ == "__main__":
    unittest.main()
