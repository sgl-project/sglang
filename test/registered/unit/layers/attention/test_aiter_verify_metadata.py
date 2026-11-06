"""AITER verify metadata: unified verify without host seq_lens, and the DCP verify shard."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.dcp import layout as dcp_layout
from sglang.srt.layers.dcp import planner as dcp_planner
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


_DCP_SIZE = 8
_DRAFT_TOKENS = 8
_DCP_MAX_CONTEXT_LEN = 128
# The second request fills the context, so its asm shard meets the static graph bound.
_DCP_SEQ_LENS = torch.tensor([32, _DCP_MAX_CONTEXT_LEN], dtype=torch.int32)


def _rank0_shard_lens(kv_lens):
    # Tokens are dealt round-robin from rank 0, so rank 0 holds ceil(len / DCP).
    return ((kv_lens + _DCP_SIZE - 1) // _DCP_SIZE).tolist()


def _dcp_verify_backend(asm_cprr):
    """An MLA backend at DCP8 rank 0, with the cuda graph buffers it allocates."""
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.device = torch.device("cpu")
    backend.use_mla = True
    backend.use_mla_dcp_asm = False
    backend._use_unified_verify = False
    backend.use_sliding_window_kv_pool = False
    backend.decode_uses_page_table = False
    backend.skip_prefill = True
    backend.page_size = 1
    backend.dcp_world_size = _DCP_SIZE
    backend.num_draft_tokens = _DRAFT_TOKENS
    backend.cprr_kernel_heads = 96 if asm_cprr else 0
    backend.max_context_len = _DCP_MAX_CONTEXT_LEN
    backend.max_split_per_batch = 32
    backend.cprr_fast_mode = True
    backend.cprr_intra_batch_mode = False
    backend.qo_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.kv_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.req_to_token = torch.zeros(
        (_BATCH_SIZE, _DCP_MAX_CONTEXT_LEN + _DRAFT_TOKENS), dtype=torch.int32
    )
    backend.runner_ids = SimpleNamespace(
        full_v2p_table=None, needs_read_translate=False
    )
    backend._asm_cprr_supports_verify_shape = lambda _q_len: asm_cprr
    # The persistent-kernel work plan is aiter's; these tests check the KV shard.
    backend.make_mla_decode_meta_data_buffer = lambda *_a, **_k: (torch.zeros(1),) * 6
    backend.make_mla_meta_data = lambda *_args, **_kwargs: None
    backend.init_cuda_graph_state(
        max_bs=_BATCH_SIZE, max_num_tokens=_BATCH_SIZE * _DRAFT_TOKENS
    )
    return backend


class TestAiterDcpVerifyGraphReplay(CustomTestCase):
    """Graph replay must plan each rank's DCP verify shard from the lengths eager
    plans: the asm cp verify reads the draft tokens from the cache, so its shard
    spans them, and that shard must still fit the static graph buffers."""

    def setUp(self):
        parallel = SimpleNamespace(
            dcp_enabled=True, dcp_size=_DCP_SIZE, dcp_rank=0, attn_dcp_rank=0
        )
        for module, target, replacement in (
            (aiter_backend, "create_flashinfer_kv_indices_triton", _NoopKernel()),
            (aiter_backend, "create_mla_kv_page_table_for_dcp", _NoopKernel()),
            (aiter_backend, "compact_dcp_verify_token_table_to_ragged", _NoopKernel()),
            (dcp_planner, "update_kv_lens_and_indices", _NoopKernel()),
            # The asm cp route runs only with the persistent MLA kernels.
            (aiter_backend, "_use_mla_ps_kernel", True),
            (aiter_backend, "get_parallel", lambda: parallel),
            (dcp_planner, "get_parallel", lambda: parallel),
            (dcp_layout, "get_parallel", lambda: parallel),
        ):
            patcher = patch.object(module, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _replay_shard_lens(self, asm_cprr):
        backend = _dcp_verify_backend(asm_cprr)
        seq_lens = _DCP_SEQ_LENS.clone()
        backend._apply_cuda_graph_metadata(
            bs=_BATCH_SIZE,
            req_pool_indices=torch.arange(_BATCH_SIZE, dtype=torch.int32),
            seq_lens=seq_lens,
            seq_lens_sum=int(_DCP_SEQ_LENS.sum()),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=None,
            seq_lens_cpu=None,
            verify_tokens_per_req=_DRAFT_TOKENS,
        )
        # The planner localizes its KV lengths in place; the batch's must survive.
        self.assertEqual(seq_lens.tolist(), _DCP_SEQ_LENS.tolist())

        metadata = backend.forward_metadata
        shard_lens = metadata.local_kv_lens.view(_BATCH_SIZE, _DRAFT_TOKENS)[:, 0]
        self.assertLessEqual(
            int(shard_lens.max()), metadata.verify_token_table.shape[1]
        )
        if asm_cprr:
            self.assertLessEqual(
                int(metadata.cprr_kv_indptr[-1]), metadata.cprr_kv_indices.numel()
            )
        return shard_lens.tolist()

    def test_asm_cp_verify_shard_spans_the_draft_tokens(self):
        self.assertEqual(
            self._replay_shard_lens(asm_cprr=True),
            _rank0_shard_lens(_DCP_SEQ_LENS + _DRAFT_TOKENS),
        )

    def test_gluon_verify_shard_stays_the_prefix(self):
        # Gluon attends the draft tokens from k_window, on one rank only.
        self.assertEqual(
            self._replay_shard_lens(asm_cprr=False), _rank0_shard_lens(_DCP_SEQ_LENS)
        )


if __name__ == "__main__":
    unittest.main()
