import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import mla_paged_decode_gluon_hip as adapter


def make_inputs(rows=1):
    device = torch.device("meta")
    cache = torch.empty((655361, 1, 576), dtype=torch.float8_e4m3fn, device=device)
    return dict(
        q=torch.empty((rows, 12, 576), dtype=torch.bfloat16, device=device),
        k_buffer=cache,
        v_buffer=cache[..., :512],
        o=torch.empty((rows, 12, 512), dtype=torch.bfloat16, device=device),
        kv_indptr=torch.empty((rows + 1,), dtype=torch.int32, device=device),
        kv_indices=torch.empty((268435456,), dtype=torch.int64, device=device),
        attn_logits=torch.empty((rows, 12, 1, 512), device=device),
        attn_lse=torch.empty((rows, 12, 1), device=device),
        num_kv_splits=torch.empty((rows,), dtype=torch.int32, device=device),
    )


def call_decode(decode, inputs, *, has_mla=True, logit_cap=0.0):
    return decode(
        inputs["q"],
        inputs["k_buffer"],
        inputs["v_buffer"],
        inputs["o"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["attn_logits"],
        inputs["attn_lse"],
        inputs["num_kv_splits"],
        256,
        192**-0.5,
        1.0,
        1.0,
        logit_cap=logit_cap,
        has_mla=has_mla,
    )


class TestMlaPagedDecode(unittest.TestCase):
    def test_entrypoint_selection_is_exact(self):
        expected = {
            1: "paged_attention_decode_m1",
            2: "paged_attention_decode_m2",
            4: "paged_attention_decode_m4",
            8: "paged_attention_decode_m8",
            12: "paged_attention_decode_m12_16",
            16: "paged_attention_decode_m12_16",
            24: "paged_attention_decode_m24_32",
            32: "paged_attention_decode_m24_32",
            64: "paged_attention_decode_m64",
            128: "paged_attention_decode_m128",
            256: "paged_attention_decode_m256",
        }
        for rows, name in expected.items():
            self.assertEqual(adapter.entrypoint_name(rows), name)
        for rows in (0, 3, 10, 20, 48, 257):
            self.assertIsNone(adapter.entrypoint_name(rows))

    def test_backend_qualification_uses_mla_cache_contract(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=("KimiK3ForConditionalGeneration",)
            ),
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            v_head_dim=128,
        )
        model_runner = SimpleNamespace(
            model_config=model_config,
            gpu_id=0,
            kv_index_translator=SimpleNamespace(is_translating=False),
            is_draft_worker=False,
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        backend = SimpleNamespace(
            use_mla=True,
            dcp_size=1,
            page_size=1,
            max_context_len=1048576,
            enable_deterministic=False,
            num_head=12,
            # Generic config heads are not MLA latent-cache heads. Kimi-K3
            # reports 96 checkpoint KV heads, or 12 per TP8 rank, while the
            # absorbed MLA cache contract checked by covered() has one head.
            num_kv_head=12,
        )
        with (
            mock.patch.object(adapter, "is_hip", return_value=True),
            mock.patch.object(adapter, "_has_required_gluon_api", return_value=True),
            mock.patch.object(adapter, "_rocm_arch", return_value="gfx950"),
            mock.patch.object(
                adapter,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=8),
            ),
            mock.patch.object(
                adapter,
                "get_server_args",
                return_value=SimpleNamespace(
                    enable_lora=False, speculative_algorithm=None
                ),
            ),
        ):
            self.assertTrue(adapter.qualified_k3_mla_backend(backend, model_runner))
            model_runner.kv_index_translator.is_translating = True
            self.assertFalse(adapter.qualified_k3_mla_backend(backend, model_runner))

    def test_runtime_contract_rejects_semantic_changes(self):
        inputs = make_inputs(2)
        backend = SimpleNamespace(max_context_len=1048576)
        kwargs = dict(
            logit_cap=0.0,
            sinks=None,
            xai_temperature_len=-1,
            has_mla=True,
            use_pdl=False,
            page_size=1,
            score_mod=None,
            aux_tensors=None,
        )
        args = (
            backend,
            inputs["q"],
            inputs["k_buffer"],
            inputs["v_buffer"],
            inputs["o"],
            inputs["kv_indptr"],
            inputs["kv_indices"],
            192**-0.5,
            1.0,
            1.0,
        )
        self.assertTrue(adapter.covered(*args, **kwargs))
        self.assertFalse(adapter.covered(*args, **{**kwargs, "has_mla": False}))
        self.assertFalse(adapter.covered(*args, **{**kwargs, "logit_cap": 1.0}))
        bad_indices = {**inputs, "kv_indices": inputs["kv_indices"][:1024]}
        bad_args = (
            backend,
            bad_indices["q"],
            bad_indices["k_buffer"],
            bad_indices["v_buffer"],
            bad_indices["o"],
            bad_indices["kv_indptr"],
            bad_indices["kv_indices"],
            192**-0.5,
            1.0,
            1.0,
        )
        self.assertFalse(adapter.covered(*bad_args, **kwargs))

        m1_inputs = make_inputs()
        m1_args = (
            backend,
            m1_inputs["q"],
            m1_inputs["k_buffer"],
            m1_inputs["v_buffer"],
            m1_inputs["o"],
            m1_inputs["kv_indptr"],
            m1_inputs["kv_indices"],
            192**-0.5,
            1.0,
            1.0,
        )
        self.assertTrue(adapter.covered(*m1_args, **kwargs))

    def test_install_preserves_fallback_and_propagates_launch_failure(self):
        native = mock.Mock(return_value="native")
        backend = SimpleNamespace(decode_attention_fwd=native, max_context_len=1048576)
        with (
            mock.patch.object(adapter, "can_install", return_value=True),
            mock.patch.object(adapter, "rank0_log"),
        ):
            self.assertTrue(adapter.install(backend, object()))

        unsupported = make_inputs(3)
        self.assertEqual(
            call_decode(backend.decode_attention_fwd, unsupported), "native"
        )
        native.assert_called_once()

        supported = make_inputs(2)
        with mock.patch.object(adapter, "run", side_effect=RuntimeError("launch")):
            with self.assertRaisesRegex(RuntimeError, "launch"):
                call_decode(backend.decode_attention_fwd, supported)
        self.assertEqual(native.call_count, 1)

    def test_installed_kernel_must_fill_native_output(self):
        native = mock.Mock()
        backend = SimpleNamespace(decode_attention_fwd=native, max_context_len=1048576)
        with (
            mock.patch.object(adapter, "can_install", return_value=True),
            mock.patch.object(adapter, "rank0_log"),
        ):
            adapter.install(backend, object())
        inputs = make_inputs(2)
        with mock.patch.object(adapter, "run", return_value=inputs["o"]):
            self.assertIsNone(call_decode(backend.decode_attention_fwd, inputs))
        native.assert_not_called()

        wrong = torch.empty((2, 12, 512), dtype=torch.float32, device="meta")
        with mock.patch.object(adapter, "run", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "output ABI"):
                call_decode(backend.decode_attention_fwd, inputs)


if __name__ == "__main__":
    unittest.main()
