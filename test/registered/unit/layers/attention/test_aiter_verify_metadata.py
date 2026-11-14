"""AITER backend workspace allocation and unified-verify metadata, built on CPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BATCH_SIZE = 2
_VERIFY_TOKENS_PER_REQ = 17
_SEQ_LENS = torch.tensor([32, 64], dtype=torch.int32)


class _NoopKernel:
    def __getitem__(self, _grid):
        return lambda *_args, **_kwargs: None


def _fail_on_host_sync(_tensor):
    raise AssertionError("tensor.item() forces a device-to-host sync")


def _metadata(backend):
    return {
        name: value.tolist() if isinstance(value, torch.Tensor) else value
        for name, value in vars(backend.forward_metadata).items()
    }


def _verify_batch():
    return SimpleNamespace(
        batch_size=_BATCH_SIZE,
        input_ids=torch.zeros(_BATCH_SIZE * _VERIFY_TOKENS_PER_REQ),
        req_pool_indices=torch.arange(_BATCH_SIZE, dtype=torch.int32),
        seq_lens=_SEQ_LENS.clone(),
        seq_lens_cpu=_SEQ_LENS.clone(),
        seq_lens_sum=int(_SEQ_LENS.sum()),
        encoder_lens=None,
        spec_info=SimpleNamespace(draft_token_num=_VERIFY_TOKENS_PER_REQ - 1),
        forward_mode=ForwardMode.TARGET_VERIFY,
        out_cache_loc=None,
    )


def _unified_verify_backend():
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.device = torch.device("cpu")
    backend.use_mla = False
    backend._use_unified_verify = True
    backend.use_sliding_window_kv_pool = False
    backend.use_muse_attention_paths = False
    backend.muse_reuse_verify_lengths = False
    backend.page_size = 1
    backend.max_context_len = 128
    backend.num_draft_tokens = _VERIFY_TOKENS_PER_REQ
    backend.qo_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.kv_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.cuda_graph_kv_indices = torch.zeros(128, dtype=torch.int32)
    backend.cuda_graph_kv_last_page_len = torch.zeros(_BATCH_SIZE, dtype=torch.int32)
    backend.req_to_token = torch.zeros((_BATCH_SIZE, 128), dtype=torch.int32)
    backend.cuda_graph_page_table = torch.zeros(
        (_BATCH_SIZE, backend.max_context_len), dtype=torch.int32
    )

    def build_verify_unified_metadata(
        bs,
        _seq_lens,
        _req_pool_indices,
        draft_num,
        page_table_dest=None,
        swa_page_table_dest=None,
    ):
        page_table = (
            page_table_dest
            if page_table_dest is not None
            else torch.zeros((bs, 8), dtype=torch.int32)
        )
        qo_indptr = torch.arange(0, (bs + 1) * draft_num, draft_num, dtype=torch.int32)
        return page_table, qo_indptr, draft_num, swa_page_table_dest

    backend._build_verify_unified_metadata = build_verify_unified_metadata
    return backend


class TestAiterUnifiedVerifyWithoutHostSeqLens(CustomTestCase):
    """A backend that opts out of the host seq_lens mirror (needs_cpu_seq_lens=False)
    receives seq_lens_cpu=None on verify batches; unified verify sizes its metadata
    from the page table, so the metadata must match and no .item() may run."""

    def setUp(self):
        for target, replacement in (
            ("reset_verify_attn_plan_cache", lambda: None),
            ("create_flashinfer_kv_indices_triton", _NoopKernel()),
        ):
            patcher = patch.object(aiter_backend, target, replacement, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _assert_same_metadata_without_host_seq_lens(self, backend, init):
        forward_batch = _verify_batch()
        init(forward_batch)
        expected = _metadata(backend)
        self.assertEqual(backend.forward_metadata.max_q_len, _VERIFY_TOKENS_PER_REQ)

        forward_batch.seq_lens_cpu = None
        forward_batch.seq_lens_sum = None
        with patch.object(torch.Tensor, "item", _fail_on_host_sync):
            init(forward_batch)
        self.assertEqual(_metadata(backend), expected)

    def test_eager_init_skips_host_max(self):
        backend = _unified_verify_backend()
        self._assert_same_metadata_without_host_seq_lens(
            backend, backend.init_forward_metadata
        )

    def test_graph_replay_skips_device_max(self):
        backend = _unified_verify_backend()
        self._assert_same_metadata_without_host_seq_lens(
            backend,
            lambda fb: backend.init_forward_metadata_out_graph(fb, in_capture=False),
        )


class _MHAPool:
    start_layer = 0
    layer_num = 1
    requires_physical_write_loc = False

    def __init__(self, kv_cache_layout):
        self.kv_cache_layout = kv_cache_layout
        self.layer_router = PlainPoolRouter(self, "full")

    def get_value_buffer(self, _layer_id):
        return torch.empty(1, 1, 8, 128)


def _init_mha_backend(kv_cache_layout):
    pool = _MHAPool(kv_cache_layout)
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        is_multimodal=False,
        num_attention_heads=32,
        head_dim=128,
        get_num_kv_heads=lambda _tp_size: 8,
        attention_arch=AttentionArch.MHA,
        context_len=4096,
    )
    model_runner = SimpleNamespace(
        model_config=model_config,
        device=torch.device("cpu"),
        kv_cache_dtype=torch.bfloat16,
        req_to_token_pool=SimpleNamespace(req_to_token=None, size=_BATCH_SIZE),
        runner_ids=None,
        token_to_kv_pool=pool,
        is_draft_worker=False,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
    )
    spec = SimpleNamespace(
        speculative_num_draft_tokens=None, speculative_num_steps=None
    )
    parallel = SimpleNamespace(attn_tp_size=1, attn_dcp_size=1)
    exec_config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False)
    )
    with (
        patch.object(
            aiter_backend, "get_schedule", lambda: SimpleNamespace(page_size=1)
        ),
        patch.object(aiter_backend, "get_exec", lambda: exec_config),
        patch.object(aiter_backend, "get_spec", lambda: spec),
        patch.object(aiter_backend, "get_parallel", lambda: parallel),
    ):
        return aiter_backend.AiterAttnBackend(model_runner, skip_prefill=True)


class TestAiterDecodeWorkspace(CustomTestCase):
    """Only paged_attention_ragged reads workspace_buffer; a SHUFFLE 5D pool decodes
    through pa_decode_gluon, so it must not allocate it."""

    def test_allocated_only_for_paged_ragged_decode(self):
        allocated = {
            layout: "workspace_buffer" in vars(_init_mha_backend(layout))
            for layout in ("nhd", "vectorized_5d")
        }
        self.assertEqual(allocated, {"nhd": True, "vectorized_5d": False})


if __name__ == "__main__":
    unittest.main()
