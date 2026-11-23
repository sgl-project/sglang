import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.true_on_policy import glm5_next_dsa
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KDA_LAYERS = (0, 2)


def _config(**overrides):
    values = dict(
        architectures=["Glm5NextForConditionalGeneration"],
        qk_rope_head_dim=0,
        qk_nope_head_dim=256,
        kv_lora_rank=512,
        num_attention_heads=64,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        # The checkpoint's own (unscaled) rope entry must be admitted.
        rope_scaling={
            "rope_theta": 800000.0,
            "partial_rotary_factor": 1.0,
            "rope_type": "default",
        },
        num_hidden_layers=4,
        indexer_types=["full", "full", "full", "full"],
        is_kda_layer=lambda layer: layer in _KDA_LAYERS,
    )
    return SimpleNamespace(**(values | overrides))


def _server_args(**overrides):
    values = dict(
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
        kv_cache_dtype="bfloat16",
        page_size=256,
        chunked_prefill_size=8192,
        disable_radix_cache=False,
        enable_hisparse=False,
        dcp_size=1,
    )
    return SimpleNamespace(**(values | overrides))


def _validate(config=None, server_args=None):
    with patch("sglang.srt.true_on_policy.glm.validate_envs"):
        glm5_next_dsa.validate_glm5_next_dsa(
            config or _config(), server_args or _server_args()
        )


class TestValidateGlm5NextDSA(unittest.TestCase):
    def test_recipe_passes(self):
        _validate()

    def test_refuses_what_changes_the_program(self):
        """Each checkpoint shape, backend or feature the qualified program does not run is refused by name."""
        cases = (
            (dict(qk_rope_head_dim=64), {}, "qk_rope_head_dim=0"),
            (dict(index_kpool=1), {}, "index_kpool=4"),
            (dict(index_kpool_always_select_tail=False), {}, "always_select_tail"),
            (dict(index_topk=1024), {}, "index_topk=2048"),
            (dict(index_n_heads=64), {}, "index_n_heads=32"),
            (dict(rope_scaling={"type": "yarn"}), {}, "rope_scaling"),
            (
                dict(indexer_types=["full", "shared", "full", "full"]),
                {},
                "reuses another layer's selection",
            ),
            ({}, dict(dsa_decode_backend="flashmla_sparse"), "TileLang"),
            ({}, dict(kv_cache_dtype="fp8_e4m3"), "BF16 latent KV"),
            ({}, dict(page_size=64), "256-token pages"),
            ({}, dict(chunked_prefill_size=8190), "64-token grid"),
            ({}, dict(enable_hisparse=True), "hisparse"),
            ({}, dict(dcp_size=2), "context parallelism"),
        )
        for config, server_args, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(NotImplementedError, message):
                    _validate(_config(**config), _server_args(**server_args))

    def test_a_shared_kda_layer_entry_is_not_a_dsa_refusal(self):
        """Only DSA layers own selections; a non-full entry on a KDA layer does not refuse the model."""
        _validate(_config(indexer_types=["shared", "full", "full", "full"]))

    def test_fused_topk_env_is_required_off(self):
        with patch("sglang.srt.true_on_policy.glm.validate_envs") as validate_envs:
            glm5_next_dsa.validate_glm5_next_dsa(_config(), _server_args())
        validate_envs.assert_called_once_with({"SGLANG_DSA_FUSE_TOPK": False})


class TestServingRefusals(unittest.TestCase):
    def test_extends_off_the_group_grid_are_refused(self):
        """An extend starting mid-group would compress that group from the tail ring, a program decode never runs."""
        glm5_next_dsa.require_group_aligned_extends([64, 4], [128, 8])
        with self.assertRaisesRegex(NotImplementedError, "group boundary"):
            glm5_next_dsa.require_group_aligned_extends([64, 3], [128, 9])

    def test_fused_topk_mapping_is_refused(self):
        logits = torch.zeros((2, 8))
        lens = torch.ones(2, dtype=torch.int32)
        for name in ("page_table", "page_table_row_index", "topk_offsets"):
            kwargs = dict(page_table=None, page_table_row_index=None, topk_offsets=None)
            kwargs[name] = torch.zeros((2, 8), dtype=torch.int32)
            with self.subTest(name=name):
                with self.assertRaisesRegex(NotImplementedError, "fused top-k"):
                    glm5_next_dsa.select_serving_tokens(
                        logits, lens, lens, row_starts=None, out_rows=None, **kwargs
                    )

    def test_selection_refuses_non_fp32_logits(self):
        lens = torch.ones(2, dtype=torch.int32)
        with self.assertRaisesRegex(NotImplementedError, "FP32 logits"):
            glm5_next_dsa.select_tokens(
                torch.zeros((2, 8), dtype=torch.bfloat16), lens, lens
            )

    def test_head_gate_refuses_gradients(self):
        x = torch.zeros((2, 4096), dtype=torch.bfloat16)
        weight = torch.zeros((32, 4096), requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "forward-only"):
            glm5_next_dsa.head_gate(
                x, weight, torch.ones((2, 32, 1)), n_heads=32, softmax_scale=1.0
            )

    def test_sparse_attention_refuses_other_shapes(self):
        """The TileLang kernel is compiled per head count and width; only the qualified one runs."""
        q = torch.zeros((2, 64, 512), dtype=torch.bfloat16)
        kv = torch.zeros((8, 512), dtype=torch.bfloat16)
        ids = torch.zeros((2, 2051), dtype=torch.int32)
        for args in (
            (q[:, :16], kv, ids),
            (q, kv.float(), ids),
            (q, kv, torch.zeros((2, 2176), dtype=torch.int32)),
        ):
            with self.assertRaisesRegex(NotImplementedError, "sparse MLA"):
                glm5_next_dsa.sparse_attention(*args, scale=1.0)


if __name__ == "__main__":
    unittest.main()
