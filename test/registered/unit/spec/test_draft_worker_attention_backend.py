"""Which attention backend a draft worker gets from the published spec."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.kimi_k3_hook import apply_kimi_k3_spec_backend_defaults
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.layers.attention import attention_registry
from sglang.srt.layers.attention.attention_registry import NonCausalDraftAttention
from sglang.srt.runtime_context import get_context, override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import draft_worker_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_GET_CONFIG = "sglang.srt.hf_transformers.get_config"


def _dspark_draft_config(*, mla=False, sliding_window=None):
    config = {
        "architectures": ["DSparkDraftModel"],
        "num_hidden_layers": 2,
        "block_size": 5,
        "dflash_config": {
            "target_layer_ids": [1],
            "mask_token_id": 3,
            "attention_mode": "mla" if mla else "gqa",
        },
    }
    if sliding_window is not None:
        config |= dict(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=sliding_window,
        )
    return config


def _inkling2_draft_config(*, local_layer_ids):
    mtp_config = dict(
        max_position_embeddings=4096,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        q_bias=False,
        o_bias=False,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=1e4,
        local_layer_ids=local_layer_ids,
        sliding_window_size=128,
        block_size=8,
        markov_rank=4,
        markov_head_type="rnn_hidden",
        target_layer_ids=[1, 4],
    )
    return {
        "architectures": ["Inkling2ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "vocab_size": 100,
            "n_layers_per_attn_res_block": 2,
            "local_layer_ids": [0],
        },
        "mtp_config": mtp_config,
    }


class TestDraftWorkerAttentionBackend(CustomTestCase):
    def _resolve(
        self,
        *,
        algorithm="DSPARK",
        named_backend="aiter",
        aiter_non_causal=True,
        draft_is_mla=False,
        draft_has_sliding_window=False,
        layout="nhd",
        unified_verify=True,
        unified_memory=False,
        deterministic=False,
    ):
        with (
            get_context().override_server_args(
                speculative_algorithm=algorithm,
                speculative_draft_attention_backend=named_backend,
                attention_backend="aiter",
                enable_unified_memory=unified_memory,
                enable_deterministic_inference=deterministic,
            ),
            patch(
                "sglang.srt.utils.aiter.aiter_unified_attention_supports_non_causal",
                return_value=aiter_non_causal,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_config.draft_is_mla",
                return_value=draft_is_mla,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_config."
                "draft_has_sliding_window",
                return_value=draft_has_sliding_window,
            ),
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override(layout),
            envs.SGLANG_AITER_UNIFIED_VERIFY.override(unified_verify),
        ):
            return draft_worker_common.resolve_draft_worker_attention_backend()

    def test_named_aiter_serves_a_dspark_mha_draft_or_raises(self):
        """A named aiter that cannot run the non-causal DSpark draft raises, not falls back."""
        self.assertEqual(self._resolve(), "aiter")
        for case in (
            dict(aiter_non_causal=False),
            dict(draft_is_mla=True),
            dict(draft_has_sliding_window=True),
            dict(layout="vectorized_5d"),
            dict(unified_verify=False),
            dict(unified_memory=True),
            dict(deterministic=True),
        ):
            with self.subTest(**case):
                with self.assertRaisesRegex(ValueError, "cannot run this DSpark draft"):
                    self._resolve(**case)

    def test_compressed_kv_draft_keeps_its_override_for_a_named_aiter(self):
        """A compressed-KV draft skips the aiter checks and keeps main's fallback, not a raise."""
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_config."
            "draft_uses_compressed_kv_attention",
            return_value=True,
        ):
            self.assertEqual(self._resolve(draft_is_mla=True), "triton")

    def test_aiter_is_taken_only_when_named_for_a_dspark_draft(self):
        """An inherited aiter, or aiter named for another algorithm's draft,
        keeps the platform fallback: triton on HIP, flashinfer elsewhere."""
        for hip_version, fallback in (("7.2", "triton"), (None, "flashinfer")):
            with (
                self.subTest(hip=hip_version),
                patch.object(torch.version, "hip", hip_version),
            ):
                self.assertEqual(self._resolve(named_backend=None), fallback)
                self.assertEqual(self._resolve(algorithm="DFLASH"), fallback)


class TestKimiK3DSparkDraftBackendDefault(CustomTestCase):
    def _declared_draft_backend(
        self,
        *,
        is_gfx950=True,
        aiter_non_causal=True,
        declared=None,
        draft_config=None,
        **server_args,
    ):
        server_args = (
            dict(
                speculative_algorithm="DSPARK",
                speculative_draft_model_path="draft",
                prefill_attention_backend="aiter",
                decode_attention_backend="aiter",
                linear_attn_verify_backend="triton",
            )
            | server_args
        )
        args = ServerArgs(model_path="dummy", **server_args)
        with (
            override_platform(is_sm100=False),
            patch(
                "sglang.kernels.arch.current_arch",
                return_value=SimpleNamespace(
                    hip=SimpleNamespace(gcn_arch="gfx950" if is_gfx950 else "gfx942")
                ),
            ),
            patch(
                "sglang.srt.utils.aiter.aiter_unified_attention_supports_non_causal",
                return_value=aiter_non_causal,
            ),
            patch.dict(
                attention_registry.NONCAUSAL_DRAFT_ATTENTION_BACKENDS, declared or {}
            ),
            patch(_GET_CONFIG, return_value=draft_config or _dspark_draft_config()),
        ):
            apply_kimi_k3_spec_backend_defaults(args)
        return resolution_result(args, "speculative_draft_attention_backend")

    def test_kimi_k3_names_aiter_for_its_dspark_draft_on_gfx950(self):
        """The hook names aiter for the DSpark draft on gfx950 only,
        and not for a rejected draft or an operator-named backend."""
        self.assertEqual(self._declared_draft_backend(), "aiter")
        for case in (
            dict(is_gfx950=False),
            dict(aiter_non_causal=False),
            dict(enable_deterministic_inference=True),
            dict(draft_config=_dspark_draft_config(mla=True)),
            dict(draft_config=_dspark_draft_config(sliding_window=128)),
            dict(speculative_draft_model_path=None),
            dict(speculative_draft_attention_backend="triton"),
            dict(prefill_attention_backend="triton"),
            dict(speculative_algorithm="EAGLE"),
        ):
            with self.subTest(**case):
                self.assertNotEqual(self._declared_draft_backend(**case), "aiter")

    def test_hook_asks_the_declared_capability_not_the_backend_name(self):
        """The hook defaults the draft to a declared backend that accepts it;
        any other target backend leaves it unset."""
        reasons = []

        def declared(*, default_here=True, reject_reason=None):
            def reject(**needs):
                reasons.append(needs)
                return reject_reason

            return {
                "fake_noncausal": NonCausalDraftAttention(
                    reject_reason=reject,
                    default_on_this_device=lambda: default_here,
                )
            }

        fake_target = dict(
            prefill_attention_backend="fake_noncausal",
            decode_attention_backend="fake_noncausal",
        )
        self.assertEqual(
            self._declared_draft_backend(declared=declared(), **fake_target),
            "fake_noncausal",
        )
        self.assertFalse(reasons[-1]["draft_is_mla"])
        self._declared_draft_backend(
            declared=declared(),
            draft_config=_dspark_draft_config(mla=True, sliding_window=128),
            **fake_target,
        )
        self.assertTrue(reasons[-1]["draft_is_mla"])
        self.assertTrue(reasons[-1]["draft_has_sliding_window"])
        for case in (
            dict(default_here=False),
            dict(reject_reason="cannot"),
        ):
            with self.subTest(**case):
                self.assertIsNone(
                    self._declared_draft_backend(
                        declared=declared(**case), **fake_target
                    )
                )
        self.assertIsNone(
            self._declared_draft_backend(
                prefill_attention_backend="triton", decode_attention_backend="triton"
            )
        )


class TestAiterDSparkDraftDefault(CustomTestCase):
    def _default_here(self, hip):
        from sglang.srt.utils.aiter import aiter_dspark_draft_default_on_this_device

        with patch(
            "sglang.kernels.arch.current_arch",
            return_value=SimpleNamespace(hip=hip),
        ):
            return aiter_dspark_draft_default_on_this_device()

    def test_exact_gfx950_only(self):
        """aiter's DSpark draft default is gfx950-only;
        other gfx95x, gfx942, or non-HIP devices must fail the gate."""
        self.assertTrue(self._default_here(SimpleNamespace(gcn_arch="gfx950")))
        for hip in (
            SimpleNamespace(gcn_arch="gfx951"),
            SimpleNamespace(gcn_arch="gfx942"),
            None,
        ):
            with self.subTest(hip=hip):
                self.assertFalse(self._default_here(hip))

    def test_aiter_declares_the_noncausal_draft_capability(self):
        """The resolver and the Kimi-K3 hook reach aiter through its registration,
        not by naming aiter."""
        from sglang.srt.utils import aiter as aiter_utils

        declared = attention_registry.NONCAUSAL_DRAFT_ATTENTION_BACKENDS["aiter"]
        self.assertIs(
            declared.reject_reason, aiter_utils.aiter_dspark_draft_reject_reason
        )
        self.assertIs(
            declared.default_on_this_device,
            aiter_utils.aiter_dspark_draft_default_on_this_device,
        )


class TestDraftAttentionTraits(CustomTestCase):
    def _traits(self, config):
        from sglang.srt.speculative.dspark_components.dspark_config import (
            draft_attention_traits,
        )

        with patch(_GET_CONFIG, return_value=config):
            return draft_attention_traits(
                draft_model_path="draft",
                trust_remote_code=False,
                revision=None,
                json_model_override_args="{}",
                model_config_parser=None,
            )

    def test_reads_mla_and_sliding_window_from_the_checkpoint(self):
        self.assertEqual(self._traits(_dspark_draft_config()), (False, False))
        self.assertEqual(self._traits(_dspark_draft_config(mla=True)), (True, False))
        self.assertEqual(
            self._traits(_dspark_draft_config(sliding_window=128)), (False, True)
        )

    def test_inkling2_draft_is_read_as_the_draft_model_is_built(self):
        """The bundled Inkling2 draft is translated before its traits are read."""
        self.assertEqual(
            self._traits(_inkling2_draft_config(local_layer_ids=[0])), (False, True)
        )
        self.assertEqual(
            self._traits(_inkling2_draft_config(local_layer_ids=[])), (False, False)
        )

    def test_no_named_checkpoint_has_no_traits(self):
        from sglang.srt.speculative.dspark_components.dspark_config import (
            draft_attention_traits,
        )

        self.assertIsNone(
            draft_attention_traits(
                draft_model_path=None,
                trust_remote_code=False,
                revision=None,
                json_model_override_args="{}",
                model_config_parser=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
