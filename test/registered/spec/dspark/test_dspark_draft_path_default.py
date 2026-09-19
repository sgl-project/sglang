import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args._model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            _BUNDLED_MODEL_PATH,
        )
        self.assertEqual(
            resolution_result(server_args, "speculative_num_draft_tokens"), 6
        )

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            "deepseek-ai/some-other-dspark-draft",
        )


class TestDsparkDpAttentionMoeA2aGate(CustomTestCase):
    """Gate contract for DSpark + dp attention + MoE a2a backends."""

    def _dp_server_args(self, *, moe_a2a_backend: str) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.enable_dp_attention = True
        server_args.enable_dp_lm_head = True
        server_args.dp_size = 2
        server_args.tp_size = 2
        server_args.moe_a2a_backend = moe_a2a_backend
        return server_args

    def test_only_megamoe_is_admitted(self):
        """Both sides of the allowlist: megamoe passes, others raise by name."""
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(self._dp_server_args(moe_a2a_backend="megamoe"))
            for backend in ("deepep", "pplx"):
                with self.assertRaisesRegex(ValueError, backend):
                    _handle_dspark(self._dp_server_args(moe_a2a_backend=backend))

    def test_a2a_backend_with_compact_verify_mode_raises(self):
        server_args = self._dp_server_args(moe_a2a_backend="megamoe")
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaisesRegex(ValueError, "static"):
                _handle_dspark(server_args)


class TestDsparkFoldedSamplingDefault(CustomTestCase):
    def test_sharded_greedy_default_and_sampling_override(self):
        from sglang.srt.environ import DsparkFoldedSampling, envs
        from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
            _resolve_folded_sampling,
        )

        model = SimpleNamespace(
            lm_head=SimpleNamespace(org_vocab_size=128, weight=torch.empty(1)),
            markov_head=SimpleNamespace(supports_sharded_greedy=True),
        )
        args = dict(
            model=model,
            gamma=5,
            max_bs=64,
            device="cpu",
            tp_rank=0,
            available_memory_gb=16,
        )
        with envs.SGLANG_DSPARK_FOLDED_SAMPLING.override(
            DsparkFoldedSampling.AUTO.value
        ):
            self.assertFalse(_resolve_folded_sampling(**args))
            model.markov_head.supports_sharded_greedy = False
            self.assertTrue(_resolve_folded_sampling(**args))
        model.markov_head.supports_sharded_greedy = True
        with envs.SGLANG_DSPARK_FOLDED_SAMPLING.override(
            DsparkFoldedSampling.FORCE.value
        ):
            self.assertTrue(_resolve_folded_sampling(**args))


class TestDsparkCandidateRoute(CustomTestCase):
    """Opt-in must reject unsupported routes before deriving candidate tables."""

    def _construct(self, head, *, folded_sampling=False):
        from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
            DsparkDraftSampler,
        )

        return DsparkDraftSampler(
            model=SimpleNamespace(
                markov_head=head,
                sample_from_anchor=True,
                lm_head=SimpleNamespace(org_vocab_size=16, weight=torch.empty(1)),
            ),
            gamma=4,
            max_bs=2,
            device="cpu",
            tp_sync=SimpleNamespace(),
            folded_sampling=folded_sampling,
        )

    def test_opt_in_rejects_unsupported_routes_before_table_preparation(self):
        from sglang.srt.models.dspark import VanillaMarkov
        from sglang.srt.speculative.dspark_components import dspark_draft_sampler

        class OtherMarkov(VanillaMarkov):
            pass

        for folded, fused, tp, head_type in (
            (True, True, 1, VanillaMarkov),
            (False, False, 1, VanillaMarkov),
            (False, True, 2, VanillaMarkov),
            (False, True, 1, OtherMarkov),
        ):
            with self.subTest(folded=folded, fused=fused, tp=tp, head=head_type):
                head = head_type(vocab_size=16, markov_rank=4)
                with (
                    envs.SGLANG_DSPARK_MARKOV_CANDIDATE_K.override(8),
                    envs.SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV.override(fused),
                    patch.object(
                        dspark_draft_sampler,
                        "get_tensor_model_parallel_world_size",
                        return_value=tp,
                    ),
                    patch.object(head, "prepare_candidates") as prepare,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "DSpark candidates require"
                    ):
                        self._construct(head, folded_sampling=folded)
                    prepare.assert_not_called()

    def test_disabled_candidates_preserve_folded_sampling_without_table(self):
        from sglang.srt.models.dspark import VanillaMarkov

        head = VanillaMarkov(vocab_size=16, markov_rank=4)
        with (
            envs.SGLANG_DSPARK_MARKOV_CANDIDATE_K.override(0),
            patch.object(head, "prepare_candidates") as prepare,
        ):
            sampler = self._construct(head, folded_sampling=True)
        prepare.assert_not_called()
        self.assertTrue(sampler.folded_sampling)
        self.assertEqual(tuple(sampler.out.shape), (8,))
        self.assertEqual(sampler.out.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
