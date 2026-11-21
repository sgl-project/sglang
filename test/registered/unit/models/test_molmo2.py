import unittest

import torch

from sglang.srt.models.molmo2 import Molmo2VisionTower, _text_block_weights
from sglang.srt.multimodal.molmo2 import _image_string, _process_images
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.molmo2 import _split_per_image
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PATCH, COL, START, END, LOW_START = 9, 10, 6, 8, 7


def _image_tokens(grid):
    low_h, low_w, high_h, high_w = grid
    return (
        [LOW_START]
        + [PATCH] * (low_h * low_w)
        + [END, START]
        + ([PATCH] * high_w + [COL]) * high_h
        + [END]
    )


class TestMolmo2ImageString(CustomTestCase):
    def test_layout_matches_transformers_molmo2(self):
        self.assertEqual(
            _image_string(
                [2, 2, 2, 3],
                image_use_col_tokens=True,
                use_single_crop_col_tokens=False,
                use_single_crop_start_token=True,
            ),
            "<low_res_im_start>"
            + "<im_patch>" * 4
            + "<im_end><im_start>"
            + ("<im_patch>" * 3 + "<im_col>") * 2
            + "<im_end>",
        )


class TestMolmo2PoolingIndices(CustomTestCase):
    def test_per_image_local_indices_become_global(self):

        def image_processor(images, return_tensors, **kwargs):
            num_crops = images[0]
            return {
                "pixel_values": torch.zeros(num_crops, 4, 6),
                "image_token_pooling": torch.tensor([[0, 1, -1, 3]]),
                "image_grids": torch.tensor([[1, 1, 0, 0]]),
                "image_num_crops": torch.tensor([num_crops]),
            }

        out = _process_images(
            image_processor=image_processor, images=[2, 1], image_kwargs={}
        )
        self.assertEqual(
            out["image_token_pooling"].tolist(), [[0, 1, -1, 3], [8, 9, -1, 11]]
        )
        self.assertEqual(out["image_num_crops"].tolist(), [2, 1])
        self.assertEqual(tuple(out["pixel_values"].shape), (3, 4, 6))


class TestMolmo2SplitPerImage(CustomTestCase):
    grids = [[2, 2, 2, 3], [1, 1, 1, 2]]

    def _processor_output(self):
        return {
            "pixel_values": torch.arange(3 * 4 * 6, dtype=torch.float32).view(3, 4, 6),
            "image_token_pooling": torch.cat(
                [
                    torch.arange(10 * 4).view(10, 4) % 8,
                    8 + torch.tensor([[0, 1, 2, -1], [3, -1, -1, -1], [1, 2, 3, 0]]),
                ]
            ),
            "image_grids": torch.tensor(self.grids),
            "image_num_crops": torch.tensor([2, 1]),
        }

    def _input_ids(self):
        return torch.tensor(
            [1, 50]
            + _image_tokens(self.grids[0])
            + [51]
            + _image_tokens(self.grids[1])
            + [52]
        )

    def test_each_image_gets_its_runs_crops_and_local_pooling(self):
        input_ids = self._input_ids()
        output = self._processor_output()
        items = _split_per_image(
            processor_output=output,
            runs=BaseMultimodalProcessor.get_mm_items_offset(input_ids, PATCH),
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(
            [sum(end - start + 1 for start, end in item.offsets) for item in items],
            [10, 3],
        )
        self.assertEqual([len(item.offsets) for item in items], [3, 2])
        for item in items:
            for start, end in item.offsets:
                self.assertTrue(bool((input_ids[start : end + 1] == PATCH).all()))
        self.assertTrue(torch.equal(items[0].feature, output["pixel_values"][:2]))
        self.assertTrue(torch.equal(items[1].feature, output["pixel_values"][2:]))
        self.assertEqual(
            items[1].model_specific_data["image_token_pooling"].tolist(),
            [[0, 1, 2, -1], [3, -1, -1, -1], [1, 2, 3, 0]],
        )

    def test_patch_count_mismatch_raises(self):
        input_ids = self._input_ids()
        input_ids = input_ids[input_ids != COL]
        input_ids = torch.cat([input_ids[:3], input_ids[4:]])
        with self.assertRaises(ValueError):
            _split_per_image(
                processor_output=self._processor_output(),
                runs=BaseMultimodalProcessor.get_mm_items_offset(input_ids, PATCH),
            )


class TestMolmo2TextWeights(CustomTestCase):
    def test_ff_proj_stores_up_before_gate(self):
        weight = torch.cat([torch.full((3, 2), 1.0), torch.full((3, 2), 2.0)])
        loaded = dict(
            _text_block_weights(
                "model.transformer.blocks.5.mlp.ff_proj.weight",
                weight,
                qkv_sizes=(4, 2, 2),
            )
        )
        self.assertTrue(
            torch.equal(loaded["model.layers.5.mlp.gate_proj.weight"], weight[3:])
        )
        self.assertTrue(
            torch.equal(loaded["model.layers.5.mlp.up_proj.weight"], weight[:3])
        )

    def test_att_proj_splits_query_key_value(self):
        weight = torch.arange(8.0).view(8, 1)
        loaded = dict(
            _text_block_weights(
                "model.transformer.blocks.0.self_attn.att_proj.weight",
                weight,
                qkv_sizes=(4, 2, 2),
            )
        )
        self.assertEqual(
            loaded["model.layers.0.self_attn.q_proj.weight"].flatten().tolist(),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            loaded["model.layers.0.self_attn.k_proj.weight"].flatten().tolist(), [4, 5]
        )
        self.assertEqual(
            loaded["model.layers.0.self_attn.v_proj.weight"].flatten().tolist(), [6, 7]
        )

    def test_unknown_weight_raises(self):
        with self.assertRaises(ValueError):
            list(
                _text_block_weights(
                    "model.transformer.blocks.0.unknown.weight",
                    torch.zeros(1),
                    qkv_sizes=(1, 1, 1),
                )
            )


class TestMolmo2VisionTowerSdpa(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                device="cpu",
                tp_size=1,
                mm_attention_backend="sdpa",
            ),
            role="test",
        )

    def test_each_crop_attends_only_within_itself(self):
        vit_config = {
            "image_patch_size": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh",
            "image_default_input_size": [4, 4],
            "image_num_pos": 4,
        }
        torch.manual_seed(0)
        tower = Molmo2VisionTower(vit_config, feature_layers=(2, 0))
        for param in tower.parameters():
            param.data.normal_(0, 0.5)
        crops = torch.randn(2, 4, 12)

        both = tower(crops)

        self.assertEqual(tuple(both.shape), (2, 4, 32))
        torch.testing.assert_close(both[:1], tower(crops[:1]))
        torch.testing.assert_close(both[1:], tower(crops[1:]))


if __name__ == "__main__":
    unittest.main()
