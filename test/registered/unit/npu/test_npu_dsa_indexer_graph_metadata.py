"""CPU-only tests for the INT8 DSA indexer metadata handle on AscendAttnBackend.

The vendor kernel binds a metadata handle built from the same length tensors it
is launched with; a decode graph replays the captured handle storage, so the
backend must refresh it in place outside the graph. torch.ops.custom is stubbed;
no NPU operator runs here.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (  # noqa: E402
    AscendAttnBackend,
    ForwardMetadata,
    _quant_indexer_metadata_static_kwargs,
    _uses_int8_quant_indexer,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import (  # noqa: E402
    NPUMLATokenToKVPool,
)
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (  # noqa: E402
    DSANPUIndexerMixin,
    _resolve_int8_indexer_metadata,
)
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402

INDEX_TOPK = 7
HEAD_DIM = 128
NUM_DRAFT_TOKENS = 3
PAGE_SIZE = 4
MAX_PAGES = 8
HANDLE_SIZE = 1024
# Handle kwargs the kernel call takes as query / key shape, not as literals.
_METADATA_ONLY_KWARGS = frozenset({"num_heads_q", "num_heads_k", "head_dim"})


class _FakeMetadataOp:
    """Returns a fresh handle per call, filled with the 1-based call index."""

    def __init__(self, shape=(HANDLE_SIZE,)):
        self.calls = []
        self.shape = shape

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full(self.shape, len(self.calls), dtype=torch.int32)


def _backend(*, use_int8=True):
    backend = object.__new__(AscendAttnBackend)
    backend.device = "cpu"
    backend.page_size = PAGE_SIZE
    backend.speculative_step_id = 0
    backend.speculative_num_draft_tokens = NUM_DRAFT_TOKENS
    backend.speculative_step_offset_npu = torch.tensor(1)
    backend.is_hybrid_swa = False
    backend.use_sliding_window_kv_pool = False
    backend.use_mla = False
    backend.use_dsa = False
    backend.dsa_dcp_target = False
    backend.dense_mla_dcp_target = False
    backend.dcp_size = 1
    backend.enable_sparsity_driven_kv_offload = False
    backend.q_head_num_padding = None
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(4 * 64).view(4, 64)
    )
    backend.req_to_token = backend.req_to_token_pool.req_to_token
    backend.graph_metadata = {
        "block_tables": torch.zeros(4, MAX_PAGES, dtype=torch.int32)
    }
    backend.use_int8_quant_indexer = use_int8
    backend.quant_indexer_metadata_kwargs = (
        _quant_indexer_metadata_static_kwargs(index_topk=INDEX_TOPK, head_dim=HEAD_DIM)
        if use_int8
        else None
    )
    return backend


def _eager_batch(
    *, mode, seq_lens, extend_seq_lens=None, spec_info=None, draft_token_num=0
):
    bs = len(seq_lens)
    if spec_info is None and draft_token_num:
        spec_info = SimpleNamespace(draft_token_num=draft_token_num)
    extend = (
        torch.tensor(extend_seq_lens, dtype=torch.int32)
        if extend_seq_lens is not None
        else None
    )
    return SimpleNamespace(
        batch_size=bs,
        forward_mode=mode,
        spec_info=spec_info,
        spec_algorithm=None,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        req_pool_indices=torch.arange(bs),
        extend_seq_lens=extend,
        extend_seq_lens_cpu=extend_seq_lens
        if extend_seq_lens is not None
        else [1] * bs,
        extend_prefix_lens_cpu=[0] * bs,
        out_cache_loc=None,
    )


class TestInt8IndexerResolution(CustomTestCase):
    def test_metadata_kwargs_pin_the_vendor_abi(self):
        # External literals from the vendor op and the 910B probe; num_heads_q
        # is the padded width, not the model's indexer head count.
        self.assertEqual(
            _quant_indexer_metadata_static_kwargs(
                index_topk=INDEX_TOPK, head_dim=HEAD_DIM
            ),
            {
                "layout_key": "PA_BSND",
                "sparse_count": INDEX_TOPK,
                "sparse_mode": 3,
                "layout_query": "TND",
                "cmp_ratio": 1,
                "key_quant_mode": 0,
                "query_quant_mode": 0,
                "num_heads_q": 64,
                "num_heads_k": 1,
                "head_dim": HEAD_DIM,
            },
        )

    def test_flag_follows_the_indexer_predicate(self):
        def pool(*, scale, dtype, cls=NPUMLATokenToKVPool):
            p = object.__new__(cls)
            p.index_k_scale_buffer = scale
            p.indexer_kv_dtype = dtype
            return p

        scale = torch.zeros(1)
        self.assertTrue(_uses_int8_quant_indexer(pool(scale=scale, dtype=torch.int8)))
        self.assertFalse(
            _uses_int8_quant_indexer(pool(scale=scale, dtype=torch.float8_e4m3fn))
        )
        self.assertFalse(_uses_int8_quant_indexer(pool(scale=None, dtype=torch.int8)))
        # A non-MLA pool (e.g. the DSV4 pool this backend's subclass uses)
        # never takes the DSA vendor path, even with matching attributes.
        self.assertFalse(
            _uses_int8_quant_indexer(
                SimpleNamespace(index_k_scale_buffer=scale, indexer_kv_dtype=torch.int8)
            )
        )


class TestEagerHandle(CustomTestCase):
    def _run(self, backend, batch):
        # Init stores the lengths; the first indexer layer builds from them.
        op = _FakeMetadataOp()
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            custom_ops.npu_quant_lightning_indexer_metadata.side_effect = op
            backend.init_forward_metadata(batch)
            _resolve_int8_indexer_metadata(backend)
        return backend.forward_metadata, op

    def _assert_built_from(self, fm, op, *, cu_q_lens, kv_lens):
        self.assertEqual(len(op.calls), 1)
        kwargs = op.calls[0]
        self.assertIs(kwargs["actual_seq_lengths_query"], fm.quant_indexer_cu_q_lens)
        self.assertIs(kwargs["actual_seq_lengths_key"], fm.quant_indexer_kv_lens)
        self.assertEqual(kwargs["num_heads_q"], 64)
        self.assertEqual(kwargs["cmp_ratio"], 1)
        for name in ("quant_indexer_cu_q_lens", "quant_indexer_kv_lens"):
            self.assertEqual(getattr(fm, name).dtype, torch.int32, name)
        self.assertEqual(fm.quant_indexer_cu_q_lens.tolist(), cu_q_lens)
        self.assertEqual(fm.quant_indexer_kv_lens.tolist(), kv_lens)
        self.assertEqual(fm.quant_indexer_metadata.tolist(), [1] * HANDLE_SIZE)

    def test_spec_decode_kv_lens_carry_the_step_offset(self):
        backend = _backend()
        backend.speculative_step_id = 1
        batch = _eager_batch(
            mode=ForwardMode.DECODE, seq_lens=[5, 9], spec_info=SimpleNamespace()
        )
        fm, op = self._run(backend, batch)
        # seq_lens_cpu_int (+ step_id + 1) wins over the raw device seq_lens.
        self._assert_built_from(fm, op, cu_q_lens=[1, 2], kv_lens=[7, 11])
        self.assertIs(fm.quant_indexer_cu_q_lens, fm.actual_seq_lengths_q)

    def test_target_verify_uses_draft_q_and_offset_kv(self):
        backend = _backend()
        batch = _eager_batch(
            mode=ForwardMode.TARGET_VERIFY,
            seq_lens=[5, 9],
            draft_token_num=NUM_DRAFT_TOKENS,
        )
        fm, op = self._run(backend, batch)
        self._assert_built_from(fm, op, cu_q_lens=[3, 6], kv_lens=[8, 12])

    def test_prefill_uses_extend_cumsum_and_batch_seq_lens(self):
        backend = _backend()
        batch = _eager_batch(
            mode=ForwardMode.EXTEND, seq_lens=[5, 9], extend_seq_lens=[2, 4]
        )
        fm, op = self._run(backend, batch)
        self._assert_built_from(fm, op, cu_q_lens=[2, 6], kv_lens=[5, 9])
        # The prefill q lens must not leak into actual_seq_lengths_q, which
        # forward_sparse branches on.
        self.assertIsNone(fm.actual_seq_lengths_q)

    def test_kv_lens_fall_back_to_device_seq_lens_without_cpu_int(self):
        backend = _backend()
        backend.forward_metadata = ForwardMetadata(
            actual_seq_lengths_q=torch.tensor([1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        )
        batch = _eager_batch(mode=ForwardMode.DECODE, seq_lens=[1, 1])
        op = _FakeMetadataOp()
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            custom_ops.npu_quant_lightning_indexer_metadata.side_effect = op
            backend._init_quant_indexer_metadata(batch)
        self.assertEqual(
            backend.forward_metadata.quant_indexer_kv_lens.tolist(), [4, 6]
        )

    def test_idle_and_disabled_build_nothing(self):
        for backend, mode in (
            (_backend(), ForwardMode.IDLE),
            (_backend(use_int8=False), ForwardMode.DECODE),
        ):
            backend.forward_metadata = ForwardMetadata()
            batch = _eager_batch(mode=mode, seq_lens=[3])
            with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
                backend._init_quant_indexer_metadata(batch)
            custom_ops.npu_quant_lightning_indexer_metadata.assert_not_called()
            fm = backend.forward_metadata
            self.assertIsNone(fm.quant_indexer_metadata)
            self.assertIsNone(fm.quant_indexer_cu_q_lens)
            self.assertIsNone(fm.quant_indexer_kv_lens)


class TestGraphHandle(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.runtime_context import get_context

        scope = get_context().override_server_args()
        scope.install()
        self.addCleanup(scope.restore)

    def _capture(self, backend, *, mode, bs):
        # Capture fills seq_lens with 0, as the graph runner does.
        batch = _eager_batch(mode=mode, seq_lens=[0] * bs)
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        return backend.forward_metadata

    def _capture_and_replay(self, *, mode, bs, live_seq_lens, op):
        backend = _backend()
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            custom_ops.npu_quant_lightning_indexer_metadata.side_effect = op
            captured = self._capture(backend, mode=mode, bs=bs)
            captured_handle = captured.quant_indexer_metadata
            captured_kv_lens = captured.quant_indexer_kv_lens
            backend.init_forward_metadata_out_graph(
                _eager_batch(mode=mode, seq_lens=live_seq_lens)
            )
        return backend.forward_metadata, captured_handle, captured_kv_lens

    def test_capture_binds_the_zero_handle_without_building(self):
        """Capture must never run the metadata op: its internal sync is illegal
        under graph capture and crashed the 910B decode-graph capture."""
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            backend = _backend()
            with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
                fm = self._capture(backend, mode=mode, bs=2)
            custom_ops.npu_quant_lightning_indexer_metadata.assert_not_called()
            handle = fm.quant_indexer_metadata
            self.assertEqual(tuple(handle.shape), (HANDLE_SIZE,), mode)
            self.assertEqual(handle.dtype, torch.int32, mode)
            self.assertEqual(handle.tolist(), [0] * HANDLE_SIZE, mode)

    def test_replay_refreshes_the_captured_storage_in_place(self):
        op = _FakeMetadataOp()
        fm, handle, kv_lens = self._capture_and_replay(
            mode=ForwardMode.DECODE, bs=3, live_seq_lens=[6, 9, 0], op=op
        )
        # The captured kernel binds these objects; replay must not rebind them.
        self.assertIs(fm.quant_indexer_metadata, handle)
        self.assertIs(fm.quant_indexer_kv_lens, kv_lens)
        self.assertEqual(len(op.calls), 1)
        self.assertEqual(handle.tolist(), [1] * HANDLE_SIZE)
        replay_kwargs = op.calls[-1]
        self.assertIs(
            replay_kwargs["actual_seq_lengths_query"], fm.quant_indexer_cu_q_lens
        )
        self.assertIs(replay_kwargs["actual_seq_lengths_key"], kv_lens)
        # The zero-length padded row is clamped, as DSV4 does for this op.
        self.assertEqual(kv_lens.tolist(), [6, 9, 1])
        self.assertEqual(fm.quant_indexer_cu_q_lens.tolist(), [1, 2, 3])

    def test_target_verify_replay_reads_offset_kv(self):
        op = _FakeMetadataOp()
        fm, _, kv_lens = self._capture_and_replay(
            mode=ForwardMode.TARGET_VERIFY, bs=2, live_seq_lens=[5, 9], op=op
        )
        self.assertEqual(kv_lens.tolist(), [8, 12])
        self.assertEqual(fm.quant_indexer_cu_q_lens.tolist(), [3, 6])

    def test_replay_shape_mismatch_raises(self):
        op = _FakeMetadataOp(shape=(HANDLE_SIZE + 1,))
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            self._capture_and_replay(
                mode=ForwardMode.DECODE, bs=2, live_seq_lens=[6, 9], op=op
            )

    def test_replay_keeps_batch_bucket_storage_independent(self):
        backend = _backend()
        op = _FakeMetadataOp()
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            custom_ops.npu_quant_lightning_indexer_metadata.side_effect = op
            first = self._capture(backend, mode=ForwardMode.DECODE, bs=1)
            second = self._capture(backend, mode=ForwardMode.DECODE, bs=2)
            first_ptr = first.quant_indexer_metadata.data_ptr()
            second_ptr = second.quant_indexer_metadata.data_ptr()
            self.assertNotEqual(first_ptr, second_ptr)
            for lengths in ([5], [7, 9], [11]):
                batch = _eager_batch(mode=ForwardMode.DECODE, seq_lens=lengths)
                batch.seq_lens_cpu = None
                backend.init_forward_metadata_out_graph(batch)
            self.assertEqual(first.quant_indexer_kv_lens.tolist(), [11])
            self.assertEqual(second.quant_indexer_kv_lens.tolist(), [7, 9])
            self.assertEqual(first.quant_indexer_metadata.data_ptr(), first_ptr)
            self.assertEqual(second.quant_indexer_metadata.data_ptr(), second_ptr)
            self.assertEqual(len(op.calls), 3)
            self.assertTrue(torch.all(first.quant_indexer_metadata == 3))
            self.assertTrue(torch.all(second.quant_indexer_metadata == 2))

    def test_draft_extend_capture_builds_no_handle(self):
        backend = _backend()
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            metadata = backend._init_cuda_graph_metadata(
                2, ForwardMode.DRAFT_EXTEND_V2, torch.zeros(2, dtype=torch.int64)
            )
        custom_ops.npu_quant_lightning_indexer_metadata.assert_not_called()
        self.assertIsNone(metadata.quant_indexer_metadata)


class TestInForwardBuild(CustomTestCase):
    def test_first_layer_builds_once_and_later_layers_reuse(self):
        # The op rejects a decode-shape build outside the model forward on
        # 910B, so init must leave the build to the indexer layers.
        backend = _backend()
        batch = _eager_batch(mode=ForwardMode.DECODE, seq_lens=[5, 9])
        op = _FakeMetadataOp()
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            custom_ops.npu_quant_lightning_indexer_metadata.side_effect = op
            backend.init_forward_metadata(batch)
            fm = backend.forward_metadata
            self.assertEqual(op.calls, [])
            self.assertIsNone(fm.quant_indexer_metadata)
            self.assertEqual(fm.quant_indexer_kv_lens.tolist(), [5, 9])
            handles = [_resolve_int8_indexer_metadata(backend) for _ in range(3)]
        self.assertEqual(len(op.calls), 1)
        kwargs = op.calls[0]
        self.assertIs(kwargs["actual_seq_lengths_query"], fm.quant_indexer_cu_q_lens)
        self.assertIs(kwargs["actual_seq_lengths_key"], fm.quant_indexer_kv_lens)
        self.assertIs(fm.quant_indexer_metadata, handles[0])
        for handle in handles[1:]:
            self.assertIs(handle, handles[0])

    def test_bound_handle_is_returned_without_a_build(self):
        # A graph-bound buffer must never be rebuilt in the forward, which
        # capture forbids; replay refreshes it in place instead.
        bound = torch.zeros(HANDLE_SIZE, dtype=torch.int32)
        backend = _backend()
        backend.forward_metadata = ForwardMetadata(quant_indexer_metadata=bound)
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            resolved = _resolve_int8_indexer_metadata(backend)
        custom_ops.npu_quant_lightning_indexer_metadata.assert_not_called()
        self.assertIs(resolved, bound)


class TestIndexerReadsBackendHandle(CustomTestCase):
    def _call(self, *, metadata):
        indexer = SimpleNamespace(n_heads=32, head_dim=HEAD_DIM, index_topk=INDEX_TOPK)
        cu_q_lens = torch.tensor([1, 2], dtype=torch.int32)
        kv_lens = torch.tensor([10, 20], dtype=torch.int32)
        with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
            custom_ops.npu_quant_lightning_indexer.return_value = (
                torch.zeros(2, INDEX_TOPK, dtype=torch.int32),
                None,
            )
            DSANPUIndexerMixin._npu_vendor_quant_indexer(
                indexer,
                query=torch.zeros(2, 32, HEAD_DIM, dtype=torch.int8),
                query_scale=torch.ones(2, 32, dtype=torch.float16),
                key=torch.zeros(1, PAGE_SIZE, 1, HEAD_DIM, dtype=torch.int8),
                key_scale=torch.ones(1, PAGE_SIZE, 1, dtype=torch.float16),
                weights=torch.zeros(2, 32, dtype=torch.bfloat16),
                block_table=torch.zeros(2, 1, dtype=torch.int32),
                metadata=metadata,
                cu_q_lens=cu_q_lens,
                kv_lens=kv_lens,
            )
        return custom_ops, cu_q_lens, kv_lens

    def test_kernel_gets_the_backend_handle_and_its_lengths(self):
        handle = torch.zeros(HANDLE_SIZE, dtype=torch.int32)
        custom_ops, cu_q_lens, kv_lens = self._call(metadata=handle)
        custom_ops.npu_quant_lightning_indexer_metadata.assert_not_called()
        kwargs = custom_ops.npu_quant_lightning_indexer.call_args.kwargs
        self.assertIs(kwargs["metadata"], handle)
        self.assertIs(kwargs["actual_seq_lengths_query"], cu_q_lens)
        self.assertIs(kwargs["actual_seq_lengths_key"], kv_lens)
        self.assertEqual(kwargs["cmp_ratio"], 1)
        self.assertEqual(tuple(kwargs["query"].shape), (2, 64, HEAD_DIM))

    def test_missing_handle_raises(self):
        with self.assertRaises(RuntimeError):
            self._call(metadata=None)

    def test_kernel_literals_match_the_handle_kwargs(self):
        # The handle and the kernel call must agree on these literals; head
        # counts and head_dim reach the kernel as the query shape instead.
        static = _quant_indexer_metadata_static_kwargs(
            index_topk=INDEX_TOPK, head_dim=HEAD_DIM
        )
        shared = static.keys() - _METADATA_ONLY_KWARGS
        self.assertEqual(
            shared,
            {
                "layout_query",
                "layout_key",
                "cmp_ratio",
                "sparse_mode",
                "query_quant_mode",
                "key_quant_mode",
                "sparse_count",
            },
        )
        custom_ops, _, _ = self._call(
            metadata=torch.zeros(HANDLE_SIZE, dtype=torch.int32)
        )
        kernel_kwargs = custom_ops.npu_quant_lightning_indexer.call_args.kwargs
        for key in sorted(shared):
            self.assertEqual(kernel_kwargs[key], static[key], key)
        query = kernel_kwargs["query"]
        self.assertEqual(query.shape[1], static["num_heads_q"])
        self.assertEqual(query.shape[2], static["head_dim"])


if __name__ == "__main__":
    unittest.main()
