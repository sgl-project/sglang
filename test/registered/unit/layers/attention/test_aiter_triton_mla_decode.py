"""When the aiter backend sends MLA decode to aiter's Triton mla_decode_fwd on gfx1250."""

import os
import sys
import types
import unittest
import warnings
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention import aiter_triton_mla as atm
from sglang.srt.mem_cache.kv.router import KVLayerRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MLA_MODULE = "aiter.ops.triton.attention.mla"
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_PAGE_SIZE = 64
_HEADS = 8
_KERNEL_OUT = 3.0


def _fake_aiter(mla_decode_fwd):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "aiter",
            "aiter.ops",
            "aiter.ops.triton",
            "aiter.ops.triton.attention",
        )
    }
    modules[_MLA_MODULE] = types.ModuleType(_MLA_MODULE)
    modules[_MLA_MODULE].mla_decode_fwd = mla_decode_fwd
    return modules


def _metadata(bs, triton):
    return aiter_backend.ForwardMetadata(
        kv_indptr=None,
        kv_indices=None,
        qo_indptr=torch.arange(bs + 1, dtype=torch.int32),
        kv_last_page_len=None,
        max_q_len=1,
        max_kv_len=100,
        triton_mla_block_tables=(
            torch.arange(bs, dtype=torch.int32)[:, None] + 1 if triton else None
        ),
        triton_mla_seqused_k=torch.full((bs,), 100, dtype=torch.int32),
    )


def _backend(kv_dtype=torch.bfloat16, bs=1, triton=True):
    """An AiterAttnBackend holding only what MLA decode reads."""
    kv_buffer = torch.zeros((1024, 1, _KV_LORA_RANK + _ROPE_DIM), dtype=kv_dtype)
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    if triton:
        backend.use_aiter_triton_mla_decode = True
    backend.use_mla = True
    backend.input_dtype = torch.bfloat16
    backend.page_size = _PAGE_SIZE
    backend.dcp_world_size = 1
    backend.use_mla_dcp_asm = False
    pool = types.SimpleNamespace(start_layer=0, layer_transfer_counter=None)
    view = types.SimpleNamespace(get_key_buffer=lambda view_layer_id: kv_buffer)
    pool.layer_router = KVLayerRouter.from_layer_maps(
        pool, [("full", view, {lid: lid for lid in range(8)})]
    )
    backend.token_to_kv_pool = pool
    backend.device = torch.device("cpu")
    backend.kv_cache_dtype = kv_dtype
    backend.k_scale = backend.v_scale = torch.tensor([1.0])
    backend.use_mla_auto_kv_splits = False
    backend.forward_metadata = _metadata(bs, triton)
    return backend


def _layer(k_scale=None):
    return types.SimpleNamespace(
        layer_id=0,
        qk_head_dim=_KV_LORA_RANK + _ROPE_DIM,
        v_head_dim=_KV_LORA_RANK,
        tp_q_head_num=_HEADS,
        scaling=1.0,
        k_scale=None if k_scale is None else torch.tensor(k_scale),
        v_scale=None,
        logit_cap=0.0,
    )


def _q(layer, n_tokens):
    return torch.zeros((n_tokens, _HEADS * layer.qk_head_dim), dtype=torch.bfloat16)


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["out"].fill_(_KERNEL_OUT)


class TestSupportsAiterTritonMlaDecode(CustomTestCase):
    def _supports(self, gfx1250=True, **overrides):
        kwargs = dict(
            input_dtype=torch.bfloat16,
            dcp_world_size=1,
            qk_rope_head_dim=_ROPE_DIM,
        )
        kwargs.update(overrides)
        with mock.patch.object(atm, "is_gfx1250_supported", return_value=gfx1250):
            return atm.supports_aiter_triton_mla_decode(**kwargs)

    def test_gfx1250_bf16_runs_aiter_triton_mla(self):
        self.assertTrue(self._supports())

    def test_settings_the_kernel_cannot_serve_decline(self):
        """fp16 activations would be run as bf16, DCP decode runs
        _forward_decode_dcp, which needs an LSE this kernel does not return, and a
        0 rope dim is a kernel shape error."""
        for kwargs in (
            dict(dcp_world_size=2),
            dict(input_dtype=torch.float16),
            dict(qk_rope_head_dim=0),
            dict(gfx1250=False),
        ):
            with self.subTest(**kwargs):
                self.assertFalse(self._supports(**kwargs))


class TestResolveAiterMlaDecodeBackend(CustomTestCase):
    def _resolve(self, value, gfx1250=True, **overrides):
        kwargs = dict(
            use_mla=True,
            input_dtype=torch.bfloat16,
            dcp_world_size=1,
            qk_rope_head_dim=_ROPE_DIM,
        )
        kwargs.update(overrides)
        with mock.patch.object(atm, "is_gfx1250_supported", return_value=gfx1250):
            return atm.resolve_aiter_mla_decode_backend(value, **kwargs)

    def _env_value(self, **env):
        names = (
            "SGLANG_AITER_MLA_DECODE_BACKEND",
            "SGLANG_AITER_MLA_DCP_DECODE_BACKEND",
        )
        with mock.patch.dict(os.environ):
            for name in names:
                os.environ.pop(name, None)
            os.environ.update(env)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return envs.SGLANG_AITER_MLA_DECODE_BACKEND.get()

    def test_unset_env_runs_aiter_triton_mla_on_gfx1250(self):
        self.assertEqual(self._resolve(self._env_value()), "triton")

    def test_unset_env_keeps_asm_off_gfx1250(self):
        for kwargs in (dict(gfx1250=False), dict(use_mla=False)):
            with self.subTest(**kwargs):
                self.assertEqual(self._resolve(self._env_value(), **kwargs), "asm")

    def test_auto_keeps_asm_where_the_triton_kernel_cannot_serve(self):
        """The cases supports_aiter_triton_mla_decode declines keep the
        existing dispatch, as before the env had a triton value."""
        for kwargs in (
            dict(dcp_world_size=2),
            dict(input_dtype=torch.float16),
            dict(qk_rope_head_dim=0),
        ):
            with self.subTest(**kwargs):
                self.assertEqual(self._resolve("auto", **kwargs), "asm")

    def test_explicit_triton_and_the_deprecated_name(self):
        self.assertEqual(self._resolve("triton"), "triton")
        self.assertEqual(self._resolve("TRITON"), "triton")
        value = self._env_value(SGLANG_AITER_MLA_DCP_DECODE_BACKEND="triton")
        self.assertEqual(self._resolve(value), "triton")

    def test_triton_the_kernel_cannot_serve_raises(self):
        for kwargs in (
            dict(gfx1250=False),
            dict(gfx1250=False, use_mla=False),
            dict(dcp_world_size=2),
            dict(input_dtype=torch.float16),
            dict(qk_rope_head_dim=0),
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "=triton"):
                    self._resolve("triton", **kwargs)

    def test_explicit_asm_and_gluon_are_honored_on_gfx1250(self):
        """Explicit values are not second-guessed, including where auto would
        pick triton and the DCP ASM decode."""
        for value in ("asm", "gluon", "ASM", "Gluon"):
            for kwargs in (
                dict(),
                dict(dcp_world_size=2),
                dict(input_dtype=torch.float16),
                dict(qk_rope_head_dim=0),
                dict(use_mla=False),
            ):
                with self.subTest(value=value, **kwargs):
                    self.assertEqual(self._resolve(value, **kwargs), value.lower())

    def test_other_archs_keep_asm_and_gluon(self):
        for value in ("asm", "gluon", "ASM", "Gluon"):
            for kwargs in (
                dict(gfx1250=False),
                dict(gfx1250=False, dcp_world_size=2),
                dict(gfx1250=False, use_mla=False),
            ):
                with self.subTest(value=value, **kwargs):
                    self.assertEqual(self._resolve(value, **kwargs), value.lower())

    def test_invalid_value_raises(self):
        for gfx1250 in (True, False):
            with self.subTest(gfx1250=gfx1250):
                with self.assertRaisesRegex(ValueError, "must be 'auto'"):
                    self._resolve("aiter", gfx1250=gfx1250)


class TestTritonMlaBlockSize(CustomTestCase):
    def test_block_is_the_largest_power_of_2_dividing_the_page_up_to_128(self):
        for page_size, block in (
            (1, 1),
            (3, 1),
            (12, 4),
            (16, 16),
            (64, 64),
            (96, 32),
            (128, 128),
            (256, 128),
            (1024, 128),
        ):
            with self.subTest(page_size=page_size):
                self.assertEqual(atm.triton_mla_block_size(page_size), block)


class TestTritonMlaBlockTables(CustomTestCase):
    def test_pages_larger_than_the_block_list_every_sub_block(self):
        """A page of page_size // block blocks expands to that many consecutive
        block ids, the rows the (-1, block) KV view reads."""
        for page_size in (96, 256):
            block = atm.triton_mla_block_size(page_size)
            ratio = page_size // block
            pages = [4, 1]
            req_to_token = torch.cat(
                [p * page_size + torch.arange(page_size) for p in pages]
            ).to(torch.int32)[None]
            with self.subTest(page_size=page_size):
                tables = atm.triton_mla_block_tables(
                    req_to_token=req_to_token,
                    req_pool_indices=torch.tensor([0]),
                    max_kv_len=page_size + 1,
                    page_size=page_size,
                )
                expected = [p * ratio + i for p in pages for i in range(ratio)]
                torch.testing.assert_close(
                    tables,
                    torch.tensor([expected[: ratio + 1]], dtype=torch.int32),
                )

    def test_page_size_1_lists_every_token_slot(self):
        req_to_token = torch.tensor([[7, 3, 9, 0], [2, 8, 5, 1]], dtype=torch.int32)
        tables = atm.triton_mla_block_tables(
            req_to_token=req_to_token,
            req_pool_indices=torch.tensor([1, 0]),
            max_kv_len=3,
            page_size=1,
        )
        torch.testing.assert_close(tables, req_to_token[[1, 0], :3])

    def test_page_ids_follow_each_requests_pages(self):
        """A row lists the request's pages in order, one per page_size tokens,
        and is wide enough for the longest sequence."""
        pages = [[5, 2, 9], [1, 7, 3]]
        req_to_token = torch.cat(
            [
                torch.cat([p * _PAGE_SIZE + torch.arange(_PAGE_SIZE) for p in row]).to(
                    torch.int32
                )[None]
                for row in pages
            ]
        )
        for max_kv_len, width in ((64, 1), (65, 2), (128, 2), (129, 3)):
            with self.subTest(max_kv_len=max_kv_len):
                tables = atm.triton_mla_block_tables(
                    req_to_token=req_to_token,
                    req_pool_indices=torch.tensor([1, 0]),
                    max_kv_len=max_kv_len,
                    page_size=_PAGE_SIZE,
                )
                torch.testing.assert_close(
                    tables,
                    torch.tensor([pages[1][:width], pages[0][:width]]).to(torch.int32),
                )


class TestUsesAiterTritonMlaDecode(CustomTestCase):
    def test_only_single_token_decode_runs_aiter_triton_mla(self):
        """Verify and draft extend carry several query tokens per sequence; they
        must keep running the ASM mla_decode_fwd."""
        backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
        for name, enabled, mode, spec_info, expected in (
            ("decode", True, ForwardMode.DECODE, None, True),
            ("disabled", False, ForwardMode.DECODE, None, False),
            ("spec_decode", True, ForwardMode.DECODE, object(), False),
            ("idle", True, ForwardMode.IDLE, None, False),
            ("extend", True, ForwardMode.EXTEND, None, False),
            ("target_verify", True, ForwardMode.TARGET_VERIFY, object(), False),
            ("draft_extend", True, ForwardMode.DRAFT_EXTEND_V2, object(), False),
        ):
            with self.subTest(name):
                backend.use_aiter_triton_mla_decode = enabled
                self.assertIs(
                    backend._uses_aiter_triton_mla_decode(mode, spec_info), expected
                )

    def test_draft_decode_logs_the_skip_and_keeps_the_existing_dispatch(self):
        backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
        backend.use_aiter_triton_mla_decode = True
        with mock.patch.object(aiter_backend, "print_info_once") as log:
            self.assertFalse(
                backend._uses_aiter_triton_mla_decode(ForwardMode.DECODE, object())
            )
        log.assert_called_once_with(
            "aiter gfx1250: draft decode batches skip the aiter Triton MLA decode"
        )


_SPLIT_PAGES = [[3, 5, 7, 9], [2, 4, 6, 8]]
_SPLIT_SEQ_LENS = [100, 64]
_PERSIST_FIELDS = (
    "work_metadata",
    "work_info_set",
    "work_indptr",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


class _NoopKernel:
    def __getitem__(self, _grid):
        return lambda *_args, **_kwargs: None


def _metadata_backend(triton):
    """An MLA AiterAttnBackend holding only what decode metadata reads, PS on."""
    bs = len(_SPLIT_PAGES)
    max_context_len = len(_SPLIT_PAGES[0]) * _PAGE_SIZE
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.device = torch.device("cpu")
    backend.use_mla = True
    backend.use_aiter_triton_mla_decode = triton
    backend.decode_uses_page_table = False
    backend.use_sliding_window_kv_pool = False
    backend.dcp_world_size = 1
    backend.use_mla_dcp_asm = False
    backend.page_size = _PAGE_SIZE
    backend.max_context_len = max_context_len
    backend.max_split_per_batch = 16
    backend.max_num_partitions = 1
    backend._decode_partition_buckets = []
    backend._kv_indices_scratch = None
    backend.req_to_token = torch.stack(
        [
            torch.cat([p * _PAGE_SIZE + torch.arange(_PAGE_SIZE) for p in row])
            for row in _SPLIT_PAGES
        ]
    ).to(torch.int32)
    backend.kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    backend.qo_indptr_ = torch.zeros(bs + 1, dtype=torch.int32)
    backend.kv_last_page_len = torch.ones(bs, dtype=torch.int32)
    backend.cuda_graph_kv_last_page_len = torch.ones(bs, dtype=torch.int32)
    backend.cuda_graph_kv_indices = torch.zeros(bs * max_context_len, dtype=torch.int32)
    backend.cuda_graph_triton_mla_block_tables = torch.zeros(
        (bs, len(_SPLIT_PAGES[0])), dtype=torch.int32
    )
    backend.cuda_graph_triton_mla_seqused_k = torch.zeros(bs, dtype=torch.int32)
    for name in _PERSIST_FIELDS:
        setattr(backend, name, torch.zeros(1, dtype=torch.int32))
    backend.make_mla_decode_meta_data_buffer = mock.Mock(
        side_effect=lambda *args, **kwargs: tuple(
            torch.zeros(1, dtype=torch.int32) for _ in range(6)
        )
    )
    backend.make_mla_meta_data = mock.Mock()
    return backend


def _decode_batch():
    seq_lens = torch.tensor(_SPLIT_SEQ_LENS, dtype=torch.int64)
    return types.SimpleNamespace(
        batch_size=len(_SPLIT_SEQ_LENS),
        req_pool_indices=torch.arange(len(_SPLIT_SEQ_LENS)),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=int(seq_lens.sum()),
        spec_info=None,
        forward_mode=ForwardMode.DECODE,
        out_cache_loc=None,
    )


class TestTritonMlaDecodeMetadataSplit(CustomTestCase):
    """A decode batch aiter Triton MLA decode takes gets a block table and no
    persist metadata; with the capability off the same batch still builds PS
    metadata."""

    def setUp(self):
        for target, replacement in (
            ("reset_verify_attn_plan_cache", lambda: None),
            ("create_flashinfer_kv_indices_triton", _NoopKernel()),
            ("_use_mla_ps_kernel", True),
        ):
            patcher = mock.patch.object(aiter_backend, target, replacement, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _assert_triton(self, backend):
        fm = backend.forward_metadata
        width = -(-max(_SPLIT_SEQ_LENS) // _PAGE_SIZE)
        torch.testing.assert_close(
            fm.triton_mla_block_tables[:, :width],
            torch.tensor([row[:width] for row in _SPLIT_PAGES], dtype=torch.int32),
        )
        torch.testing.assert_close(
            fm.triton_mla_seqused_k,
            torch.tensor(_SPLIT_SEQ_LENS, dtype=torch.int32),
        )
        self.assertIsNone(fm.work_metadata)
        self.assertIsNone(fm.reduce_indptr)
        backend.make_mla_decode_meta_data_buffer.assert_not_called()
        backend.make_mla_meta_data.assert_not_called()

    def _assert_persist(self, backend):
        fm = backend.forward_metadata
        self.assertIsNone(fm.triton_mla_block_tables)
        self.assertIsNone(fm.triton_mla_seqused_k)
        self.assertIsNotNone(fm.work_metadata)
        backend.make_mla_meta_data.assert_called_once()

    def test_eager_metadata_splits_triton_and_persist_batches(self):
        triton = _metadata_backend(triton=True)
        triton.init_forward_metadata(_decode_batch())
        self._assert_triton(triton)
        self.assertEqual(
            triton.forward_metadata.triton_mla_block_tables.shape[1],
            -(-max(_SPLIT_SEQ_LENS) // _PAGE_SIZE),
        )

        persist = _metadata_backend(triton=False)
        persist.init_forward_metadata(_decode_batch())
        self._assert_persist(persist)
        persist.make_mla_decode_meta_data_buffer.assert_called_once()

    def test_cuda_graph_metadata_splits_triton_and_persist_batches(self):
        for triton, check in (
            (True, self._assert_triton),
            (False, self._assert_persist),
        ):
            with self.subTest(triton=triton):
                backend = _metadata_backend(triton=triton)
                fb = _decode_batch()
                backend._apply_cuda_graph_metadata(
                    bs=fb.batch_size,
                    req_pool_indices=fb.req_pool_indices,
                    seq_lens=fb.seq_lens,
                    seq_lens_sum=fb.seq_lens_sum,
                    forward_mode=fb.forward_mode,
                    spec_info=None,
                    seq_lens_cpu=fb.seq_lens_cpu,
                    verify_tokens_per_req=None,
                )
                check(backend)
                fm = backend.forward_metadata
                if triton:
                    self.assertEqual(
                        fm.triton_mla_block_tables.data_ptr(),
                        backend.cuda_graph_triton_mla_block_tables.data_ptr(),
                    )
                    self.assertEqual(
                        fm.triton_mla_seqused_k.data_ptr(),
                        backend.cuda_graph_triton_mla_seqused_k.data_ptr(),
                    )
                else:
                    for name in _PERSIST_FIELDS:
                        self.assertIs(getattr(fm, name), getattr(backend, name))


class TestAiterGfx1250MlaDecodeDispatch(CustomTestCase):
    def setUp(self):
        self.kernel = _RecordingKernel()
        self.asm_out = torch.full((1, _HEADS, _KV_LORA_RANK), 7.0)
        for patcher in (
            mock.patch.dict(sys.modules, _fake_aiter(self.kernel)),
            mock.patch.object(
                aiter_backend, "prefer_mla_gluon_decode", lambda **_: False
            ),
            mock.patch.object(
                aiter_backend.AiterAttnBackend,
                "_mla_decode_fwd_with_head_pad",
                lambda *args, **kwargs: self.asm_out,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _decode(self, backend, layer, bs):
        return backend._forward_mla_decode(_q(layer, bs), layer, None, None)

    def test_triton_metadata_returns_the_kernel_output(self):
        backend = _backend(bs=2)
        layer = _layer()
        layer.scaling = 0.125
        o = self._decode(backend, layer, 2)

        self.assertEqual(tuple(o.shape), (2, _HEADS, _KV_LORA_RANK))
        self.assertEqual(o.dtype, torch.bfloat16)
        self.assertTrue(torch.all(o == _KERNEL_OUT))
        (call,) = self.kernel.calls
        self.assertEqual(
            set(call),
            {
                "q",
                "kv_buffer",
                "out",
                "cu_seqlens_q",
                "seqused_k",
                "max_seqlen_kv",
                "block_tables",
                "softmax_scale",
                "kv_lora_rank",
                "qk_rope_head_dim",
                "causal",
                "q_descale",
                "kv_descale",
            },
        )
        self.assertEqual(tuple(call["q"].shape), (2, _HEADS, _KV_LORA_RANK + _ROPE_DIM))
        self.assertIs(call["out"], o)
        self.assertEqual(call["softmax_scale"], 0.125)
        self.assertEqual(call["max_seqlen_kv"], 0)
        self.assertIs(call["causal"], True)
        self.assertIsNone(call["q_descale"])
        self.assertIsNone(call["kv_descale"])
        fm = backend.forward_metadata
        self.assertIs(call["block_tables"], fm.triton_mla_block_tables)
        self.assertIs(call["seqused_k"], fm.triton_mla_seqused_k)
        self.assertIs(call["cu_seqlens_q"], fm.qo_indptr)
        self.assertEqual(
            tuple(call["kv_buffer"].shape),
            (1024 // _PAGE_SIZE, _PAGE_SIZE, 1, _KV_LORA_RANK + _ROPE_DIM),
        )
        self.assertEqual(call["kv_lora_rank"], _KV_LORA_RANK)
        self.assertEqual(call["qk_rope_head_dim"], _ROPE_DIM)

    def test_kv_buffer_is_viewed_in_kernel_blocks(self):
        for page_size, block in ((1, 1), (16, 16), (256, 128)):
            with self.subTest(page_size=page_size):
                backend = _backend()
                backend.page_size = page_size
                self._decode(backend, _layer(), 1)
                self.assertEqual(
                    tuple(self.kernel.calls[-1]["kv_buffer"].shape),
                    (1024 // block, block, 1, _KV_LORA_RANK + _ROPE_DIM),
                )

    def test_no_triton_metadata_runs_the_asm_decode(self):
        """Batches without a block table must reach the ASM decode."""
        backend = _backend(triton=False)
        self.assertIs(self._decode(backend, _layer(), 1), self.asm_out)
        self.assertEqual(self.kernel.calls, [])

    def test_metadata_without_triton_fields_runs_the_asm_decode(self):
        """A backend built without __init__ and metadata without the gfx1250
        fields, as test_aiter_fp8_q_unified_attention.py builds them, keep the
        ASM decode."""
        backend = _backend(triton=False)
        fm = backend.forward_metadata
        backend.forward_metadata = types.SimpleNamespace(
            **{
                name: getattr(fm, name)
                for name in vars(fm)
                if not name.startswith("triton_mla_")
            }
        )
        self.assertIs(self._decode(backend, _layer(), 1), self.asm_out)
        self.assertEqual(self.kernel.calls, [])

    def test_soft_capped_layer_runs_the_asm_decode(self):
        """aiter's Triton mla_decode_fwd has no soft cap and would silently drop
        it; the ASM decode takes logit_cap and rejects what it cannot apply."""
        layer = _layer()
        layer.logit_cap = 30.0
        with mock.patch.object(aiter_backend, "print_info_once") as log:
            self.assertIs(self._decode(_backend(), layer, 1), self.asm_out)
        self.assertEqual(self.kernel.calls, [])
        log.assert_called_once_with(
            "aiter gfx1250: MLA layers with logit_cap=30.0 skip the aiter Triton "
            "MLA decode"
        )

    def test_soft_capped_layer_keeps_the_asm_ps_preference_under_triton(self):
        """A "triton" backend's batches outside the Triton kernel prefer ASM PS
        over mla_gluon as "asm" does; only "gluon" holds them on mla_gluon."""
        layer = _layer()
        layer.logit_cap = 30.0
        gluon_out = torch.full((1, _HEADS, _KV_LORA_RANK), 5.0)
        for value, expected in (
            ("triton", self.asm_out),
            ("asm", self.asm_out),
            ("gluon", gluon_out),
        ):
            with (
                self.subTest(value),
                mock.patch.multiple(
                    aiter_backend,
                    prefer_mla_gluon_decode=lambda **_: True,
                    mla_gluon_decode=lambda **_: gluon_out,
                    _use_mla_ps_kernel=True,
                ),
                mock.patch.multiple(
                    aiter_backend.AiterAttnBackend,
                    _asm_ps_supports_qlen=lambda self, q_len: True,
                    _resolve_mla_gluon_min_kv_seq_len=lambda self, fb: 1,
                ),
            ):
                backend = _backend()
                backend.mla_decode_backend = value
                self.assertIs(self._decode(backend, layer, 1), expected)
        self.assertEqual(self.kernel.calls, [])

    def test_explicit_asm_and_gluon_on_gfx1250_bf16_take_their_kernels(self):
        """Where auto picks triton, an explicit asm or gluon leaves the Triton
        decode off and reaches the ASM decode or mla_gluon as on other archs."""
        gluon_out = torch.full((1, _HEADS, _KV_LORA_RANK), 5.0)
        for value, expected in (("asm", self.asm_out), ("gluon", gluon_out)):
            with (
                self.subTest(value),
                mock.patch.object(atm, "is_gfx1250_supported", return_value=True),
                mock.patch.multiple(
                    aiter_backend,
                    prefer_mla_gluon_decode=lambda **_: True,
                    mla_gluon_decode=lambda **_: gluon_out,
                    _use_mla_ps_kernel=True,
                ),
                mock.patch.multiple(
                    aiter_backend.AiterAttnBackend,
                    _asm_ps_supports_qlen=lambda self, q_len: True,
                    _resolve_mla_gluon_min_kv_seq_len=lambda self, fb: 1,
                ),
            ):
                resolved = atm.resolve_aiter_mla_decode_backend(
                    value,
                    use_mla=True,
                    input_dtype=torch.bfloat16,
                    dcp_world_size=1,
                    qk_rope_head_dim=_ROPE_DIM,
                )
                self.assertEqual(resolved, value)
                backend = _backend(triton=False)
                backend.mla_decode_backend = resolved
                self.assertFalse(backend.use_aiter_triton_mla_decode)
                self.assertIs(self._decode(backend, _layer(), 1), expected)
        self.assertEqual(self.kernel.calls, [])

    def test_kv_descale_matches_the_asm_decode(self):
        """The MLA store writes K unscaled; only an fp8 cache takes the k_scale."""
        for name, kv_dtype, layer_k_scale, expected in (
            ("bf16", torch.bfloat16, 2.0, None),
            ("fp8_layer_scale", aiter_backend.fp8_dtype, 2.0, 2.0),
            ("fp8_backend_scale", aiter_backend.fp8_dtype, None, 1.0),
        ):
            with self.subTest(name):
                backend = _backend(kv_dtype=kv_dtype)
                layer = _layer(k_scale=layer_k_scale)
                backend.forward_decode(
                    _q(layer, 1),
                    None,
                    None,
                    layer,
                    types.SimpleNamespace(forward_mode=ForwardMode.DECODE),
                    save_kv_cache=False,
                )
                kv_descale = self.kernel.calls[-1]["kv_descale"]
                if expected is None:
                    self.assertIsNone(kv_descale)
                else:
                    self.assertEqual(kv_descale.item(), expected)


if __name__ == "__main__":
    unittest.main()
