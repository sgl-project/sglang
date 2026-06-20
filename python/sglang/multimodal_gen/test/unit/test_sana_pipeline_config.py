from sglang.multimodal_gen.configs.pipeline_configs.sana import SanaPipelineConfig


def test_sana_vae_precision_defaults_to_fp32():
    assert SanaPipelineConfig().vae_precision == "fp32"
