"""AITER verify: unified-verify metadata without host seq_lens, and DCP head padding."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend
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


_CPRR_QUERY = dict(
    cp_world_size=8, fast_mode=True, intra_batch_mode=False, v_head_dim=512
)


def _cprr_kernel_heads(gathered_heads, q_len):
    return aiter_backend._asm_cprr_kernel_heads(
        gathered_heads,
        lambda heads: aiter_backend._asm_cprr_shape_verdict(
            heads, q_len, ask_aiter=True, **_CPRR_QUERY
        )[0],
    )


class _ShapeSupport:
    """aiter's get_mla_decode_shape_support, answering from a fixed verdict table."""

    def __init__(self, native_shapes, supported=True, arch="gfx950", reasons=()):
        self.native_shapes, self.supported, self.calls = native_shapes, supported, []
        self.arch, self.reasons = arch, reasons

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        native = (kwargs["num_heads"], kwargs["max_seqlen_qo"]) in self.native_shapes
        return SimpleNamespace(
            arch=self.arch,
            native=native,
            fold_factor=1 if native else 6,
            seqlen_fold=1,
            supported=self.supported,
            reasons=self.reasons,
        )


def _failing_query(**_kwargs):
    raise RuntimeError("module_mla_metadata failed to build")


def _dcp_backend(*, use_mla=True, dcp_world_size=8, num_draft_tokens=8, verify="asm"):
    # Kimi-K3 at TP8: 12 heads a rank, v_head_dim 128, kv_lora_rank 512.
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.use_mla = use_mla
    backend.dcp_world_size = dcp_world_size
    backend.num_head = 12
    backend.num_draft_tokens = num_draft_tokens
    backend.mla_verify_backend = verify
    backend.kv_cache_dtype = aiter_backend.fp8_dtype
    backend.cprr_fast_mode = True
    backend.cprr_intra_batch_mode = False
    model_config = SimpleNamespace(v_head_dim=128)
    if use_mla:
        model_config.kv_lora_rank = 512
    backend._init_cprr_verify(model_config=model_config)
    return backend


class TestAiterCprrKernelHeads(CustomTestCase):
    """The DCP verify pads its gathered heads only where aiter would fold them."""

    def setUp(self):
        patcher = patch.object(aiter_backend, "_use_mla_ps_kernel", True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _query(self, query):
        patcher = patch.object(aiter_backend, "get_mla_decode_shape_support", query)
        patcher.start()
        self.addCleanup(patcher.stop)
        return query

    def test_aiter_verdict_decides_the_padding(self):
        # Kimi-K3 at DCP8 gathers 96 heads and verifies 8 tokens a request.
        for native_shapes, expected in (({(96, 8), (128, 8)}, 96), ({(128, 8)}, 128)):
            support = _ShapeSupport(native_shapes)
            with patch.object(aiter_backend, "get_mla_decode_shape_support", support):
                self.assertEqual(_cprr_kernel_heads(96, 8), expected)
            # The query describes the call the verify makes: fp8, LSE, round robin.
            self.assertLessEqual(
                {
                    "q_dtype": aiter_backend.fp8_dtype,
                    "kv_dtype": aiter_backend.fp8_dtype,
                    "return_lse": True,
                    "cp_round_robin": True,
                    **_CPRR_QUERY,
                }.items(),
                support.calls[0].items(),
            )

    def test_init_asks_with_the_latent_width_and_verify_length(self):
        # The absorbed verify writes the 512-wide latent, not the 128-wide
        # v_head_dim, and aiter's asm route only exists for 512.
        support = self._query(_ShapeSupport({(96, 8), (128, 8), (96, 1)}))
        backend = _dcp_backend(num_draft_tokens=8)
        self.assertEqual(backend.cprr_kernel_heads, 96)
        self.assertEqual(support.calls[0]["v_head_dim"], 512)
        self.assertEqual(support.calls[0]["max_seqlen_qo"], 8)
        # Target verify asks on every step; aiter is asked once.
        for _ in range(2):
            self.assertTrue(backend._asm_cprr_supports_verify_shape(8))
        self.assertEqual(len(support.calls), 1)
        # Without speculative decoding the verify window is one token.
        self.assertEqual(_dcp_backend(num_draft_tokens=None).cprr_kernel_heads, 96)
        self.assertEqual(support.calls[-1]["max_seqlen_qo"], 1)

    def test_init_asks_aiter_only_on_the_asm_verify_route(self):
        support = self._query(_ShapeSupport({(96, 8), (128, 8)}))
        for use_mla, dcp_world_size in ((True, 1), (False, 8)):
            backend = _dcp_backend(use_mla=use_mla, dcp_world_size=dcp_world_size)
            self.assertEqual(backend.cprr_kernel_heads, 0)
        # A Gluon verify keeps the copied table, which has no 96-head 8-token row.
        self.assertEqual(_dcp_backend(verify="gluon").cprr_kernel_heads, 128)
        self.assertEqual(support.calls, [])

    def test_failed_query_falls_back_to_the_table(self):
        self._query(_failing_query)
        with self.assertLogs(aiter_backend.logger, "WARNING"):
            self.assertEqual(_dcp_backend(num_draft_tokens=6).cprr_kernel_heads, 96)
        self.assertEqual(_dcp_backend(num_draft_tokens=8).cprr_kernel_heads, 128)

    def test_table_decides_where_aiter_cannot(self):
        # aiter cannot confirm a kernel on gfx950 (supported=None), or has no query.
        unknown = _ShapeSupport({(96, 6), (96, 8), (128, 6), (128, 8)}, supported=None)
        for query in (unknown, None):
            with patch.object(aiter_backend, "get_mla_decode_shape_support", query):
                self.assertEqual(_cprr_kernel_heads(96, 6), 96)
                self.assertEqual(_cprr_kernel_heads(96, 8), 128)

    def test_unconfirmed_kernel_off_gfx950_is_not_native(self):
        # The copied table describes gfx950 kernels only.
        unknown = _ShapeSupport(
            {(96, 6), (96, 8), (128, 6), (128, 8)}, supported=None, arch="gfx1250"
        )
        with patch.object(aiter_backend, "get_mla_decode_shape_support", unknown):
            self.assertEqual(_cprr_kernel_heads(96, 6), 0)

    def test_table_never_overrides_a_fold(self):
        # The table says 96 heads are native at q_len 6, but aiter's planner folds
        # them: running 96 unpadded would fold on one side only.
        folds_96 = _ShapeSupport({(128, 6)}, supported=None)
        with patch.object(aiter_backend, "get_mla_decode_shape_support", folds_96):
            self.assertEqual(_cprr_kernel_heads(96, 6), 128)

    def test_unsupported_shape_stays_on_gluon_with_aiter_reason(self):
        reason = "asm no_kernel on gfx950: q=fp8 kv=fp8 heads=96 qlen=8"
        self._query(
            _ShapeSupport({(96, 8), (128, 8)}, supported=False, reasons=(reason,))
        )
        backend = _dcp_backend(num_draft_tokens=8)
        self.assertEqual(backend.cprr_kernel_heads, 0)
        with self.assertLogs(aiter_backend.logger, "INFO") as logs:
            self.assertFalse(backend._asm_ps_supports_decode_and_verify())
        self.assertIn(reason, logs.output[0])


if __name__ == "__main__":
    unittest.main()
