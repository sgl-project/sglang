"""When the aiter backend sends MLA decode to aiter's gfx1250 paged Gluon kernel."""

import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention import aiter_mla_gluon as amg
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


def _backend(
    page_size=_PAGE_SIZE,
    dcp_world_size=1,
    kv_dtype=torch.bfloat16,
    input_dtype=torch.bfloat16,
):
    """An AiterAttnBackend holding only what MLA decode reads."""
    kv_buffer = torch.zeros((1024, 1, _KV_LORA_RANK + _ROPE_DIM), dtype=kv_dtype)
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.use_mla = True
    backend.input_dtype = input_dtype
    backend.page_size = page_size
    backend.dcp_world_size = dcp_world_size
    backend.use_mla_dcp_asm = False
    pool = types.SimpleNamespace(start_layer=0, layer_transfer_counter=None)
    view = types.SimpleNamespace(get_key_buffer=lambda view_layer_id: kv_buffer)
    pool.layer_router = KVLayerRouter.from_layer_maps(
        pool, [("full", view, {lid: lid for lid in range(8)})]
    )
    backend.token_to_kv_pool = pool
    # Request r starts at page r, so every request has its own page table.
    backend.req_to_token = (
        torch.arange(4096, dtype=torch.int32)[None, :]
        + page_size * torch.arange(4, dtype=torch.int32)[:, None]
    ) % kv_buffer.shape[0]
    backend.device = torch.device("cpu")
    backend.kv_cache_dtype = kv_dtype
    backend.k_scale = backend.v_scale = torch.tensor([1.0])
    backend.use_mla_auto_kv_splits = False
    backend.forward_metadata = mock.MagicMock(max_q_len=1)
    backend.paged_gluon_mla_decode = None
    return backend


def _layer(layer_id=0, k_scale=None):
    return types.SimpleNamespace(
        layer_id=layer_id,
        qk_head_dim=_KV_LORA_RANK + _ROPE_DIM,
        v_head_dim=_KV_LORA_RANK,
        tp_q_head_num=_HEADS,
        scaling=1.0,
        k_scale=None if k_scale is None else torch.tensor(k_scale),
        v_scale=None,
        logit_cap=0.0,
    )


def _batch(req_pool_indices, forward_mode=ForwardMode.DECODE, spec_info=None):
    bs = len(req_pool_indices)
    return types.SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.tensor(req_pool_indices),
        seq_lens=torch.full((bs,), 100),
        forward_mode=forward_mode,
        spec_info=spec_info,
    )


def _q(layer, n_tokens):
    return torch.zeros((n_tokens, _HEADS * layer.qk_head_dim), dtype=torch.bfloat16)


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["out"].fill_(_KERNEL_OUT)


class _AiterGfx1250TestCase(CustomTestCase):
    def setUp(self):
        self.kernel = _RecordingKernel()
        amg._paged_mla_decode_fwd.cache_clear()
        self.addCleanup(amg._paged_mla_decode_fwd.cache_clear)
        patcher = mock.patch.dict(sys.modules, _fake_aiter(self.kernel))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _create(self, backend, mla_gluon=True, qk_rope_head_dim=_ROPE_DIM):
        amg._paged_mla_decode_fwd.cache_clear()
        with envs.SGLANG_AITER_MLA_GLUON.override(mla_gluon):
            return amg.PagedGluonMlaDecode.create(
                backend,
                kv_lora_rank=_KV_LORA_RANK,
                qk_rope_head_dim=qk_rope_head_dim,
            )


class TestPagedGluonMlaDecodeGate(_AiterGfx1250TestCase):
    def _forward(self, decode, layer, forward_batch):
        bs = forward_batch.batch_size
        o = torch.zeros((bs, _HEADS * layer.v_head_dim), dtype=torch.bfloat16)
        decode.forward(
            q=_q(layer, bs),
            o=o,
            layer=layer,
            forward_batch=forward_batch,
            kv_descale=None,
        )
        return self.kernel.calls[-1]

    def test_paged_pool_uses_the_kernel(self):
        decode = self._create(_backend())
        self.assertIsNotNone(decode)
        self.assertIs(decode.mla_decode_fwd, self.kernel)

    def test_mla_gluon_kill_switch_also_turns_off_gfx1250(self):
        """SGLANG_AITER_MLA_GLUON=0 must disable every aiter Gluon MLA decode."""
        self.assertIsNone(self._create(_backend(), mla_gluon=False))

    def test_backend_settings_the_kernel_cannot_serve_decline(self):
        """Only page size 64 is validated; fp16 activations would be run as bf16,
        DCP needs aiter's own CP path, and a 0 rope dim is a kernel shape error."""
        for kwargs in (
            dict(page_size=1),
            dict(page_size=16),
            dict(page_size=32),
            dict(page_size=128),
            dict(dcp_world_size=2),
            dict(input_dtype=torch.float16),
        ):
            with self.subTest(**kwargs):
                self.assertIsNone(self._create(_backend(**kwargs)))
        with self.subTest(qk_rope_head_dim=0):
            self.assertIsNone(self._create(_backend(), qk_rope_head_dim=0))

    def test_page_table_is_rebuilt_at_the_first_kernel_layer(self):
        """The pool's first layer need not run this kernel; the page table
        must still follow each new batch."""
        backend = _backend()
        decode = self._create(backend)
        first = self._forward(decode, _layer(3), _batch([1]))
        self._forward(decode, _layer(4), _batch([1]))
        second = self._forward(decode, _layer(3), _batch([2]))

        expected = backend.req_to_token[2, ::_PAGE_SIZE] // _PAGE_SIZE
        self.assertFalse(torch.equal(first["block_tables"], second["block_tables"]))
        torch.testing.assert_close(second["block_tables"][0], expected)


class TestAiterGfx1250MlaDecodeDispatch(_AiterGfx1250TestCase):
    def setUp(self):
        super().setUp()
        self.asm_out = torch.full((1, _HEADS, _KV_LORA_RANK), 7.0)
        for patcher in (
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

    def _decode(self, backend, layer, forward_batch):
        return backend._forward_mla_decode(
            _q(layer, forward_batch.batch_size), layer, forward_batch, None
        )

    def test_accepted_batch_returns_the_paged_kernel_output(self):
        backend = _backend()
        backend.paged_gluon_mla_decode = self._create(backend)
        o = self._decode(backend, _layer(), _batch([1, 2]))

        self.assertEqual(tuple(o.shape), (2, _HEADS, _KV_LORA_RANK))
        self.assertEqual(o.dtype, torch.bfloat16)
        self.assertTrue(torch.all(o == _KERNEL_OUT))
        call = self.kernel.calls[0]
        torch.testing.assert_close(
            call["seqused_k"], torch.tensor([100, 100], dtype=torch.int32)
        )
        torch.testing.assert_close(
            call["cu_seqlens_q"], torch.tensor([0, 1, 2], dtype=torch.int32)
        )

    def test_kv_descale_matches_the_asm_decode(self):
        """The MLA store writes K unscaled; only an fp8 cache takes the k_scale."""
        for name, kv_dtype, layer_k_scale, expected in (
            ("bf16", torch.bfloat16, 2.0, None),
            ("fp8_layer_scale", amg.fp8_dtype, 2.0, 2.0),
            ("fp8_backend_scale", amg.fp8_dtype, None, 1.0),
        ):
            with self.subTest(name):
                backend = _backend(kv_dtype=kv_dtype)
                backend.paged_gluon_mla_decode = self._create(backend)
                layer = _layer(k_scale=layer_k_scale)
                backend.forward_decode(
                    _q(layer, 1), None, None, layer, _batch([1]), save_kv_cache=False
                )
                kv_descale = self.kernel.calls[-1]["kv_descale"]
                if expected is None:
                    self.assertIsNone(kv_descale)
                else:
                    self.assertEqual(kv_descale.item(), expected)

    def test_declined_batch_runs_the_asm_decode(self):
        """What the paged kernel declines must reach the ASM decode, not skip it."""
        for name, use_paged, batch in (
            ("spec", True, _batch([1], spec_info=object())),
            ("idle", True, _batch([], forward_mode=ForwardMode.IDLE)),
            ("extend", True, _batch([1], forward_mode=ForwardMode.EXTEND)),
            (
                "target_verify",
                True,
                _batch([1], forward_mode=ForwardMode.TARGET_VERIFY),
            ),
            ("no_paged_kernel", False, _batch([1])),
        ):
            with self.subTest(name):
                backend = _backend()
                if use_paged:
                    backend.paged_gluon_mla_decode = self._create(backend)
                self.assertIs(self._decode(backend, _layer(), batch), self.asm_out)
        self.assertEqual(self.kernel.calls, [])


class TestPagedGluonMlaDecodeDisablesPersist(_AiterGfx1250TestCase):
    def setUp(self):
        super().setUp()
        for name, value in (
            ("_use_mla_ps_kernel", True),
            ("fast_mode", True),
            ("intra_batch_mode", True),
        ):
            patcher = mock.patch.object(aiter_backend, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _create_in_backend(self, backend):
        with envs.SGLANG_AITER_MLA_GLUON.override(True):
            return backend._create_paged_gluon_mla_decode(
                kv_lora_rank=_KV_LORA_RANK, qk_rope_head_dim=_ROPE_DIM
            )

    def test_paged_gluon_turns_off_asm_persist_metadata(self):
        """aiter ships no persistent-scheduling MLA ASM kernel for gfx1250: batches
        the paged kernel declines run the non-persist ASM decode, so no persist
        metadata may be built for them."""
        backend = _backend()
        self.assertIsNotNone(self._create_in_backend(backend))
        self.assertFalse(backend._use_mla_decode_persist_metadata())
        self.assertFalse(aiter_backend.intra_batch_mode)

    def test_pool_without_paged_gluon_keeps_asm_persist_metadata(self):
        backend = _backend(page_size=1)
        self.assertIsNone(self._create_in_backend(backend))
        self.assertTrue(backend._use_mla_decode_persist_metadata())


class TestGfx950MlaGluonOnGfx1250(CustomTestCase):
    def test_gfx950_mla_gluon_is_never_selected_on_gfx1250(self):
        """gfx1250 must not pick (or report as enabled) the gfx950 kernel;
        other arches keep picking it."""
        mla_gluon = object()
        modules = {
            name: types.ModuleType(name)
            for name in (
                "triton",
                "triton.experimental",
                "triton.experimental.gluon",
                "triton.experimental.gluon.language",
                "aiter",
                "aiter.ops",
                "aiter.ops.triton",
                "aiter.ops.triton.gluon",
                "aiter.ops.triton.gluon.mla_gluon",
            )
        }
        modules["triton.experimental.gluon.language"].PaddedSharedLayout = (
            lambda cga_layout=None: None
        )
        modules["aiter.ops.triton.gluon.mla_gluon"].mla_gluon = mla_gluon
        self.addCleanup(amg._gluon_fn.cache_clear)
        with mock.patch.dict(sys.modules, modules):
            with envs.SGLANG_AITER_MLA_GLUON.override(True):
                for gfx1250, expected in ((True, None), (False, mla_gluon)):
                    with self.subTest(gfx1250=gfx1250):
                        amg._gluon_fn.cache_clear()
                        with mock.patch.object(
                            amg, "is_gfx1250_supported", return_value=gfx1250
                        ):
                            self.assertIs(amg._gluon_fn(), expected)


if __name__ == "__main__":
    unittest.main()
