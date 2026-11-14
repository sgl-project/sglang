"""CPU-only tests of the actual upstream binding and configuration contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.models.dflash import _make_dflash_hidden_norm
from sglang.srt.runtime_context import get_context, get_schedule, get_spec
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.dmtp_runtime import (
    adapt_legacy_fa3_kernel,
    scoped_config,
    validate_recipe,
)
from sglang.srt.speculative.dmtp_worker import DMTPWorkerV2
from sglang.srt.speculative.mtp_first_controller import MTPFirstController
from sglang.srt.speculative.native_sglang_mtp import NativeSGLangMTP


class PortTest(unittest.TestCase):
    def test_legacy_context_norm_gamma_matches_oracle(self):
        for cfg in (
            SimpleNamespace(hidden_norm_zero_centered_gamma=True),
            SimpleNamespace(dflash_config=dict(hidden_norm_zero_centered_gamma=True)),
        ):
            module = _make_dflash_hidden_norm(cfg, 16, 1e-6)
            self.assertIsInstance(module, GemmaRMSNorm)
            module._weight_loader(module.weight, torch.linspace(-0.2, 0.2, 16))
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(32, 16).to(dtype)
                expected = (
                    x.float()
                    * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
                    * (1 + module.weight.float())
                ).to(dtype)
                torch.testing.assert_close(
                    module.forward_native(x), expected, rtol=0, atol=0
                )
        plain = _make_dflash_hidden_norm(SimpleNamespace(), 16, 1e-6)
        self.assertIsInstance(plain, RMSNorm)
        global_gemma = _make_dflash_hidden_norm(
            SimpleNamespace(), 16, 1e-6, GemmaRMSNorm
        )
        self.assertIsInstance(global_gemma, GemmaRMSNorm)

    def test_legacy_fa3_only_omits_inactive_new_option(self):
        calls = []

        def old(q, *, causal=False, softmax_scale=None):
            calls.append((q, causal, softmax_scale))
            return q

        adapted = adapt_legacy_fa3_kernel(old)
        self.assertIs(adapt_legacy_fa3_kernel(adapted), adapted)
        self.assertEqual(adapted(7, causal=True, softmax_scale=0.125, only_qv=False), 7)
        self.assertEqual(calls, [(7, True, 0.125)])
        with self.assertRaises(NotImplementedError):
            adapted(7, only_qv=True)
        with self.assertRaises(TypeError):
            adapted(7, unsupported=1)

        def new(q, *, only_qv=False):
            return q

        self.assertIs(adapt_legacy_fa3_kernel(new), new)

    def test_private_mtp_uses_new_signature_and_restores_config(self):
        with get_context().override_server_args(
            speculative_algorithm="DFLASH", speculative_draft_model_path="/draft"
        ) as args:
            module = SimpleNamespace(
                logits_processor=torch.nn.Identity(),
                set_embed_and_head=lambda embed, head: None,
            )

            def constructor(**kwargs):
                self.assertEqual(get_spec().speculative_algorithm, "EAGLE")
                self.assertEqual(
                    resolving_view(kwargs["server_args"]).speculative_algorithm, "EAGLE"
                )
                self.assertIsNot(kwargs["server_args"], args)
                self.assertNotIn("tp_rank", kwargs)
                self.assertEqual(kwargs["draft_attention_backend"], "fa3")
                return SimpleNamespace(model_runner=SimpleNamespace(model=module))

            owner = SimpleNamespace(
                server_args=args,
                gpu_id=0,
                nccl_port=12345,
                model_runner=SimpleNamespace(
                    model_config=SimpleNamespace(context_len=8192)
                ),
                target_worker=SimpleNamespace(
                    random_seed=42,
                    model_runner=SimpleNamespace(
                        model=SimpleNamespace(get_embed_and_head=lambda: (None, None))
                    ),
                ),
            )
            with patch(
                "sglang.srt.speculative.native_sglang_mtp.TpModelWorker", constructor
            ):
                NativeSGLangMTP(owner)
            self.assertEqual(get_spec().speculative_algorithm, "DFLASH")
            self.assertEqual(get_spec().speculative_draft_model_path, "/draft")
            self.assertEqual(resolving_view(args).speculative_algorithm, "DFLASH")

    def test_unvalidated_modes_fail_before_model_loading(self):
        fields = dict(
            page_size=1,
            quantization=None,
            speculative_num_draft_tokens=8,
            speculative_draft_attention_backend="fa3",
            max_running_requests=32,
            enable_lora=False,
            enable_two_batch_overlap=False,
            enable_unified_memory=False,
            speculative_dflash_draft_kv_ring=False,
        )
        with get_context().override_server_args(**fields):
            validate_recipe()
            for name, value in (
                ("page_size", 64),
                ("max_running_requests", 33),
                ("speculative_num_draft_tokens", 7),
                ("enable_lora", True),
                ("enable_two_batch_overlap", True),
                ("enable_unified_memory", True),
                ("speculative_dflash_draft_kv_ring", True),
            ):
                with scoped_config(**{name: value}):
                    with self.assertRaises(ValueError):
                        validate_recipe()

    def test_native_verifier_and_forward_are_inherited(self):
        for name in (
            "forward_batch_generation",
            "_accept_block",
            "_append_target_hidden_to_draft_kv_by_loc",
            "_update_target_mamba_state_after_verify",
        ):
            self.assertIs(getattr(DMTPWorkerV2, name), getattr(DFlashWorkerV2, name))

    def test_scope_restores_after_exception_and_nesting(self):
        with get_context().override_server_args(
            speculative_num_draft_tokens=8, max_running_requests=32
        ):
            with scoped_config(speculative_num_draft_tokens=7, max_running_requests=4):
                self.assertEqual(get_spec().speculative_num_draft_tokens, 7)
                self.assertEqual(get_schedule().max_running_requests, 4)
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with scoped_config(speculative_num_draft_tokens=2):
                        raise RuntimeError("intentional")
                self.assertEqual(get_spec().speculative_num_draft_tokens, 7)
            self.assertEqual(get_spec().speculative_num_draft_tokens, 8)
            self.assertEqual(get_schedule().max_running_requests, 32)

    def test_sampling_guard_rejects_unported_features(self):
        c = MTPFirstController.__new__(MTPFirstController)
        c.opt_sync, c.target_only, c.trace_dir = True, True, None
        info = SimpleNamespace(
            need_min_p_sampling=False,
            has_custom_logit_processor=False,
            logit_bias=None,
            grammars=None,
            grammar_mask=None,
            penalizer_orchestrator=None,
        )
        batch = SimpleNamespace(sampling_info=info)
        self.assertIsNone(c.validate_sampling(batch))
        info.grammar_mask = object()
        with self.assertRaises(ValueError):
            c.validate_sampling(batch)


if __name__ == "__main__":
    unittest.main()
