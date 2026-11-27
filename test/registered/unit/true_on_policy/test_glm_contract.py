import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.true_on_policy.glm import validate_glm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**overrides):
    values = dict(
        model_type="glm_moe_dsa",
        index_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        hidden_act="silu",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        hidden_size=6144,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        num_hidden_layers=8,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        # GLM-5's checkpoint lists the same sharing per layer.
        indexer_types=[
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
            "shared",
        ],
        index_kpool=None,
    )
    return SimpleNamespace(**(values | overrides))


def _server_args(**overrides):
    values = dict(
        kv_cache_dtype="bf16",
        dsa_prefill_backend="flashmla_sparse",
        dsa_decode_backend="flashmla_sparse",
        dsa_topk_backend="flashinfer",
        page_size=64,
        disable_cuda_graph=True,
        enable_waterfill=False,
        enforce_shared_experts_fusion=False,
        enable_single_batch_overlap=False,
        enable_hisparse=False,
        speculative_algorithm=None,
    )
    return SimpleNamespace(**(values | overrides))


def _validate(config, server_args=None):
    with ExitStack() as stack:
        for name, kwargs in (
            ("get_server_args", dict(return_value=server_args or _server_args())),
            ("validate_ep_runtime", {}),
            ("validate_envs", {}),
            ("get_parallel", dict(return_value=SimpleNamespace(moe_ep_size=4))),
            ("get_deepep_mode", dict(return_value=SimpleNamespace(value="normal"))),
            ("is_tbo_enabled", dict(return_value=False)),
        ):
            stack.enter_context(
                patch(f"sglang.srt.true_on_policy.glm.{name}", **kwargs)
            )
        validate_glm(config, None)


class TestValidateGLM(unittest.TestCase):
    def test_recipe_config_passes(self):
        _validate(_config())

    def test_index_layouts_the_trainer_does_not_run_are_refused(self):
        """Serving reads indexer_types, cli_factor and index_topk_pattern before the
        freq/offset rule the trainer runs; a different sharing picks different keys."""
        for overrides in (
            dict(indexer_types=["full"] * 8),
            dict(indexer_types=None, cli_factor=2),
            dict(indexer_types=None, index_topk_pattern="FFFSSSSF"),
            dict(indexer_types=None, index_skip_topk_offset=None),
        ):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(
                    NotImplementedError, "rule the trainer runs"
                ):
                    _validate(_config(**overrides))
        _validate(_config(indexer_types=None, index_topk_pattern="FFFSSSFS"))
        with self.assertRaisesRegex(NotImplementedError, "index_kpool"):
            _validate(_config(index_kpool=4))

    def test_overlap_and_hisparse_are_refused(self):
        """SBO reorders the MoE program and hisparse changes the indexed KV."""
        for overrides in (
            dict(enable_single_batch_overlap=True),
            dict(enable_hisparse=True),
        ):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(
                    NotImplementedError, "single-batch overlap"
                ):
                    _validate(_config(), _server_args(**overrides))
        with patch.dict(
            "os.environ", {"SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO": "1"}
        ):
            with self.assertRaisesRegex(NotImplementedError, "single-batch overlap"):
                _validate(_config())


if __name__ == "__main__":
    unittest.main()
