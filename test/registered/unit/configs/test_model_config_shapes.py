import math
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    _quant_config_to_dict,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_text_config(**overrides):
    defaults = dict(
        architectures=["MixtralForCausalLM"],
        model_type="mixtral",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=2,
        vocab_size=32000,
        num_key_value_heads=8,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_external_mla_config(**overrides):
    defaults = dict(
        architectures=["ExternalMLAForCausalLM"],
        model_type="external_mla",
        use_mla=True,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=96,
    )
    defaults.update(overrides)
    return _make_text_config(**defaults)


class TestModelConfigShapes(CustomTestCase):
    def _derive_shapes(self, text_config):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = text_config
        model_config.hf_text_config = text_config
        model_config._derive_model_shapes()
        return model_config

    def test_external_model_opts_into_mla(self):
        model_config = self._derive_shapes(_make_external_mla_config())

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.qk_nope_head_dim, 128)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.v_head_dim, 96)
        self.assertEqual(model_config.head_dim, 192)
        self.assertEqual(model_config.swa_kv_lora_rank, 512)
        self.assertEqual(model_config.swa_qk_rope_head_dim, 64)
        self.assertIsNone(model_config.index_head_dim)
        self.assertAlmostEqual(model_config.scaling, 192**-0.5)

    def test_external_model_requires_explicit_mla_opt_in(self):
        for use_mla in (False, None):
            with self.subTest(use_mla=use_mla):
                config = _make_external_mla_config(use_mla=use_mla)
                self.assertEqual(
                    self._derive_shapes(config).attention_arch, AttentionArch.MHA
                )
        config = _make_external_mla_config()
        del config.use_mla
        self.assertEqual(self._derive_shapes(config).attention_arch, AttentionArch.MHA)

    def test_external_model_allocates_compressed_mla_cache(self):
        import torch

        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()

        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.runtime_context import get_context

        model_config = self._derive_shapes(_make_external_mla_config())
        configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
        configurator.model_config = model_config
        configurator.use_mla_backend = model_config.attention_arch == AttentionArch.MLA
        configurator.is_draft_worker = False
        configurator.is_hybrid_swa = False
        configurator.mambaish_config = None
        configurator.kv_cache_dtype = torch.float32
        configurator.kv_cache_dtype_str = "auto"
        configurator.kv_shard_rank = None
        configurator.device = "cpu"
        configurator.layer_info = SimpleNamespace(
            num_effective_layers=2, start_layer=0, end_layer=2
        )

        with get_context().override_server_args(page_size=1, enable_memory_saver=False):
            pool = configurator._build_token_to_kv_pool(
                sizes=SimpleNamespace(max_total_num_tokens=8),
                is_dsa_model=False,
                is_dsv4_model=False,
                req_to_token_pool=None,
            )

        self.assertIsInstance(pool, MLATokenToKVPool)
        self.assertEqual(len(pool.kv_buffer), 2)
        for buffer in pool.kv_buffer:
            # One shared latent + RoPE row per token, including a padding slot.
            self.assertEqual(tuple(buffer.shape), (9, 1, 576))

    def test_external_mla_uses_nested_text_config(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(
            architectures=["ExternalMLAForConditionalGeneration"],
            model_type="external_vlm",
        )
        model_config.hf_text_config = _make_external_mla_config()

        model_config._derive_model_shapes()

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.kv_lora_rank, 512)
        self.assertEqual(model_config.v_head_dim, 96)

    def test_external_mla_rejects_missing_or_invalid_dimensions(self):
        for name in (
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        ):
            config = _make_external_mla_config()
            delattr(config, name)
            with self.subTest(name=name, missing=True):
                with self.assertRaisesRegex(ValueError, name):
                    self._derive_shapes(config)
            for value in (None, -1, True, 1.5, "64"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self._derive_shapes(_make_external_mla_config(**{name: value}))
        for name in ("kv_lora_rank", "v_head_dim"):
            with self.subTest(name=name, value=0):
                with self.assertRaisesRegex(ValueError, name):
                    self._derive_shapes(_make_external_mla_config(**{name: 0}))
        with self.assertRaisesRegex(ValueError, "qk_nope_head_dim.*qk_rope_head_dim"):
            self._derive_shapes(
                _make_external_mla_config(qk_nope_head_dim=0, qk_rope_head_dim=0)
            )

    def test_external_mla_allows_zero_rope_dimension(self):
        model_config = self._derive_shapes(
            _make_external_mla_config(qk_rope_head_dim=0)
        )

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.qk_rope_head_dim, 0)
        self.assertEqual(model_config.head_dim, 128)
        self.assertAlmostEqual(model_config.scaling, 128**-0.5)

    def test_external_mla_applies_rope_scaling(self):
        for field in ("rope_scaling", "rope_parameters"):
            with self.subTest(field=field):
                model_config = self._derive_shapes(
                    _make_external_mla_config(
                        **{field: {"factor": 4.0, "mscale_all_dim": 1}}
                    )
                )
                mscale = 0.1 * math.log(4.0) + 1.0
                self.assertAlmostEqual(model_config.scaling, mscale**2 / math.sqrt(192))

    def test_existing_mla_architecture_keeps_its_specialized_shapes(self):
        model_config = self._derive_shapes(
            _make_external_mla_config(architectures=["DeepseekV3ForCausalLM"])
        )

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 256)

    def test_bailing_nextn_draft_keeps_the_target_attention(self):
        """A Bailing draft is renamed BailingMoeForCausalLMNextN: the V2.5 one is
        MLA, the V2 one keeps the GQA shapes of its target."""
        gqa = _make_text_config(
            architectures=["BailingMoeForCausalLMNextN"], head_dim=128
        )
        self.assertEqual(self._derive_shapes(gqa).attention_arch, AttentionArch.MHA)
        mla = _make_text_config(
            architectures=["BailingMoeForCausalLMNextN"],
            head_dim=192,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rope_scaling=None,
        )
        self.assertEqual(self._derive_shapes(mla).attention_arch, AttentionArch.MLA)

    def test_optional_head_dims_default_when_none(self):
        text_config = _make_text_config(
            head_dim=None,
            v_head_dim=None,
            swa_head_dim=None,
            swa_v_head_dim=None,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.v_head_dim, 128)
        self.assertEqual(model_config.swa_head_dim, 128)
        self.assertEqual(model_config.swa_v_head_dim, 128)
        self.assertEqual(text_config.head_dim, 128)
        self.assertEqual(text_config.v_head_dim, 128)
        self.assertEqual(text_config.swa_head_dim, 128)
        self.assertEqual(text_config.swa_v_head_dim, 128)

    def test_explicit_head_dims_are_preserved(self):
        text_config = _make_text_config(
            head_dim=128,
            v_head_dim=96,
            swa_head_dim=64,
            swa_v_head_dim=48,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.v_head_dim, 96)
        self.assertEqual(model_config.swa_head_dim, 64)
        self.assertEqual(model_config.swa_v_head_dim, 48)

    def test_v_head_dim_zero_falls_back_to_head_dim(self):
        # deepseek-vl2-tiny's language_config sets use_mla=False + v_head_dim=0
        # as an MLA-disabled sentinel. The non-MLA KV pool sizes its V buffer
        # from v_head_dim directly, so a literal 0 collapses V-cache to a
        # zero-width tensor and `v_cache[indices] = v` fails with a shape
        # mismatch inside `_set_kv_buffer_impl`. v_head_dim=0 must be treated
        # identically to `None` and fall back to head_dim.
        text_config = _make_text_config(head_dim=128, v_head_dim=0)

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.v_head_dim, 128)
        self.assertEqual(text_config.v_head_dim, 128)

    def test_ling_mla_nope_shapes(self):
        text_config = _make_text_config(
            architectures=["BailingMoeV3ForCausalLM"],
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            use_mla_nope=True,
            v_head_dim=128,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.qk_rope_head_dim, 0)
        self.assertEqual(model_config.scaling, 1 / math.sqrt(128))

    def test_ling_mla_rope_shapes(self):
        text_config = _make_text_config(
            architectures=["BailingMoeV3ForCausalLM"],
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            use_mla_nope=False,
            v_head_dim=128,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.scaling, 1 / math.sqrt(192))

    def test_sarvam_mla_shapes(self):
        text_config = _make_text_config(
            architectures=["SarvamMLAForCausalLM"],
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            rope_scaling=None,
            v_head_dim=128,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 192)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.scaling, 1 / math.sqrt(192))

    def test_quant_config_objects_are_normalized(self):
        quant_config = SimpleNamespace(to_dict=lambda: {"quant_method": "test"})

        self.assertEqual(_quant_config_to_dict(quant_config), {"quant_method": "test"})


if __name__ == "__main__":
    unittest.main()
