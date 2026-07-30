import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _route_kimi_k3_fp8_dspark_verify_to_decode,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
    server_args.model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


def _make_verify_routing_args(
    *,
    architecture: str = "KimiK3ForConditionalGeneration",
    kv_cache_dtype: str = "fp8_e4m3",
    speculative_attention_mode: str = "prefill",
    prefill_attention_backend: str = "fa3",
    decode_attention_backend: str = "flashmla",
) -> SimpleNamespace:
    hf_config = SimpleNamespace(architectures=[architecture])
    server_args = SimpleNamespace(
        _resolved_overrides=[],
        attention_backend=prefill_attention_backend,
        prefill_attention_backend=None,
        decode_attention_backend=decode_attention_backend,
        kv_cache_dtype=kv_cache_dtype,
        speculative_attention_mode=speculative_attention_mode,
    )
    server_args.get_model_config = lambda: SimpleNamespace(hf_config=hf_config)
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
        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

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
            server_args.speculative_draft_model_path,
            "deepseek-ai/some-other-dspark-draft",
        )


class TestKimiK3Fp8VerifyRouting(CustomTestCase):
    def test_routes_fp8_fa3_verify_to_flashmla_decode(self):
        server_args = _make_verify_routing_args()

        _route_kimi_k3_fp8_dspark_verify_to_decode(server_args)

        self.assertEqual(server_args.speculative_attention_mode, "decode")

    def test_keeps_bf16_verify_on_prefill(self):
        server_args = _make_verify_routing_args(kv_cache_dtype="auto")

        _route_kimi_k3_fp8_dspark_verify_to_decode(server_args)

        self.assertEqual(server_args.speculative_attention_mode, "prefill")

    def test_keeps_other_models_on_prefill(self):
        server_args = _make_verify_routing_args(architecture="DeepseekV4ForCausalLM")

        _route_kimi_k3_fp8_dspark_verify_to_decode(server_args)

        self.assertEqual(server_args.speculative_attention_mode, "prefill")

    def test_keeps_non_flashmla_decode_backend_on_prefill(self):
        server_args = _make_verify_routing_args(decode_attention_backend="triton")

        _route_kimi_k3_fp8_dspark_verify_to_decode(server_args)

        self.assertEqual(server_args.speculative_attention_mode, "prefill")

    def test_keeps_explicit_decode_mode(self):
        server_args = _make_verify_routing_args(speculative_attention_mode="decode")

        _route_kimi_k3_fp8_dspark_verify_to_decode(server_args)

        self.assertEqual(server_args.speculative_attention_mode, "decode")


if __name__ == "__main__":
    unittest.main()
