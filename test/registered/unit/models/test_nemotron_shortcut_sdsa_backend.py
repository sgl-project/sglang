"""Which serving setups SDSA sparse attention accepts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention import sdsa_backend  # noqa: E402
from sglang.srt.layers.attention.trtllm_mha_backend import (  # noqa: E402
    TRTLLMHAAttnBackend,
)
from sglang.srt.mem_cache.sdsa_kv_pool import SdsaTokenToKVPool  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MODULE = "sglang.srt.layers.attention.sdsa_backend"


def _runner(*, kv_dtype=torch.bfloat16, num_kv_heads=1):
    model_config = SimpleNamespace(
        hf_config="hf_config",
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(index_topk=2048),
        get_num_kv_heads=lambda tp: num_kv_heads,
    )
    return SimpleNamespace(
        model_config=model_config,
        kv_cache_dtype=kv_dtype,
        token_to_kv_pool=SdsaTokenToKVPool.__new__(SdsaTokenToKVPool),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1, 4096)),
    )


def _create(*, disagg="null", hicache=False, full_backend=None, index_head_dim=128):
    full_backend = full_backend or TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    with (
        patch(
            "sglang.srt.configs.nemotron_shortcut.sdsa_index_head_dim",
            return_value=index_head_dim,
        ),
        patch(
            f"{_MODULE}.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=disagg),
        ),
        patch(
            f"{_MODULE}.get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=hicache),
        ),
        patch(f"{_MODULE}.SdsaAttnBackend", return_value="sdsa"),
    ):
        return sdsa_backend.create_sdsa_backend(_runner(), full_backend)


class TestCreateSdsaBackend(CustomTestCase):
    def test_dense_models_keep_the_configured_backend(self):
        full = object()
        self.assertIs(_create(full_backend=full, index_head_dim=None), full)

    def test_supported_setup_gets_sdsa(self):
        self.assertEqual(_create(), "sdsa")

    def test_unsupported_setups_are_rejected(self):
        with self.assertRaises(ValueError):
            _create(full_backend=object())
        # The index-key cache is not part of PD transfer or HiCache offload.
        with self.assertRaises(NotImplementedError):
            _create(disagg="decode")
        with self.assertRaises(NotImplementedError):
            _create(hicache=True)


class TestSdsaAttnBackendInit(CustomTestCase):
    def _init(self, runner):
        def base_init(backend, model_runner):
            backend.data_type = model_runner.kv_cache_dtype

        with (
            patch.object(TRTLLMHAAttnBackend, "__init__", base_init),
            patch(
                f"{_MODULE}.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
        ):
            sdsa_backend.SdsaAttnBackend(runner)

    def test_kv_cache_dtypes(self):
        self._init(_runner(kv_dtype=torch.float8_e4m3fn))
        # The sparse path casts the query to E4M3 only.
        for kv_dtype in (torch.float8_e5m2, torch.uint8):
            with self.assertRaises(NotImplementedError):
                self._init(_runner(kv_dtype=kv_dtype))

    def test_more_than_one_kv_head_per_rank_is_rejected(self):
        # Top-k slots index one-token pages of a single KV head.
        with self.assertRaises(NotImplementedError):
            self._init(_runner(num_kv_heads=2))


if __name__ == "__main__":
    unittest.main()
