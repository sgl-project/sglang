import os
import tempfile
import unittest

from sglang.multimodal_gen.configs.pipeline_configs.stablediffusion3 import (
    StableDiffusion3PipelineConfig,
)


class TestStableDiffusion3PipelineConfig(unittest.TestCase):
    def test_selects_fp16_clip_text_encoder_variant(self):
        config = StableDiffusion3PipelineConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            full = os.path.join(tmp_dir, "model.safetensors")
            fp16 = os.path.join(tmp_dir, "model.fp16.safetensors")
            open(full, "a").close()
            open(fp16, "a").close()

            selected = config.select_text_encoder_weight_files(
                safetensors_list=[full, fp16],
                component_model_path=tmp_dir,
                component_name="text_encoder_2",
                text_encoder_precision="fp16",
            )

        self.assertEqual(selected, [fp16])

    def test_leaves_t5_text_encoder_shards_unchanged(self):
        config = StableDiffusion3PipelineConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            shard = os.path.join(tmp_dir, "model-00001-of-00002.safetensors")
            open(shard, "a").close()

            selected = config.select_text_encoder_weight_files(
                safetensors_list=[shard],
                component_model_path=tmp_dir,
                component_name="text_encoder_3",
                text_encoder_precision="fp32",
            )

        self.assertEqual(selected, [shard])


if __name__ == "__main__":
    unittest.main()
