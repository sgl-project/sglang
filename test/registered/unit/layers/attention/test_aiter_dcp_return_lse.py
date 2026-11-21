"""ROCm DCP MLA asks for the LSE by keyword; the backends must accept it."""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HEADS = 4
_KV_LORA_RANK = 8
_ROPE_DIM = 4
_QK_HEAD_DIM = _KV_LORA_RANK + _ROPE_DIM
_TOKENS = 3


def _layer():
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=_HEADS,
        qk_head_dim=_QK_HEAD_DIM,
        v_head_dim=_KV_LORA_RANK,
        logit_cap=0.0,
        is_cross_attention=False,
        k_scale=None,
        v_scale=None,
    )


def _dcp_mla_backend():
    """An AiterAttnBackend holding only what the DCP MLA dispatch reads."""
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.use_mla = True
    backend.dcp_world_size = 2
    backend.kv_cache_dtype = torch.bfloat16
    backend.muse_extend_fly_gfx1250 = False
    backend.forward_metadata = SimpleNamespace(
        max_q_len=_TOKENS,
        max_kv_len=_TOKENS,
        kv_indptr=None,
        kv_indices=None,
        qo_indptr=None,
    )
    k_buffer = torch.zeros(16, 1, _QK_HEAD_DIM)
    v_buffer = torch.zeros(16, 1, _KV_LORA_RANK)
    layer_kv = SimpleNamespace(key=lambda: k_buffer, value=lambda: v_buffer)
    backend.token_to_kv_pool = SimpleNamespace(
        layer_router=SimpleNamespace(read=lambda _layer_id: layer_kv)
    )
    return backend


def _batch(mode):
    return SimpleNamespace(
        forward_mode=mode,
        attn_attend_prefix_cache=False,
        out_cache_loc=None,
    )


class TestAiterDcpReturnLse(CustomTestCase):
    def setUp(self):
        self.out = torch.ones(_TOKENS, _HEADS, _KV_LORA_RANK)
        self.lse = torch.full((_TOKENS, _HEADS), 2.0)

    def _forward(self, mode, dcp_method, k=None, v=None):
        backend = _dcp_mla_backend()
        layer = _layer()
        q = torch.zeros(_TOKENS, _HEADS, _QK_HEAD_DIM)
        with mock.patch.object(
            backend, dcp_method, return_value=(self.out, self.lse)
        ) as dcp:
            result = backend.forward(
                q,
                k,
                v,
                layer,
                _batch(mode),
                save_kv_cache=False,
                return_lse=True,
            )
        dcp.assert_called_once()
        return result

    def test_dcp_decode_accepts_return_lse_and_returns_pair(self):
        out, lse = self._forward(ForwardMode.DECODE, "_forward_decode_dcp")
        self.assertIs(out, self.out)
        self.assertIs(lse, self.lse)

    def test_dcp_target_verify_accepts_return_lse_and_returns_pair(self):
        k_window = torch.zeros(_TOKENS, 1, _QK_HEAD_DIM)
        v_window = torch.zeros(_TOKENS, 1, _KV_LORA_RANK)
        out, lse = self._forward(
            ForwardMode.TARGET_VERIFY, "_forward_verify_dcp", k_window, v_window
        )
        self.assertIs(out, self.out)
        self.assertIs(lse, self.lse)


class TestDcpBackendsAcceptReturnLse(CustomTestCase):
    """forward_mla_rocm passes return_lse=True to whichever backend serves DCP."""

    def test_extend_and_decode_take_return_lse(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        for cls in (aiter_backend.AiterAttnBackend, DeepseekSparseAttnBackend):
            for name in ("forward_extend", "forward_decode"):
                with self.subTest(backend=cls.__name__, method=name):
                    params = inspect.signature(getattr(cls, name)).parameters
                    self.assertIn("return_lse", params)


if __name__ == "__main__":
    unittest.main()
