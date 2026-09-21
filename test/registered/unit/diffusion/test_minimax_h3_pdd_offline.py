# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3FinalLayer
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.tools import build_minimax_h3_pdd_weights as build
from sglang.multimodal_gen.tools import fuse_minimax_h3_pdd_heads as fusion
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxH3PDDOffline(CustomTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.heads = {
            f"{name}.{suffix}": torch.arange(8 * width * size, dtype=torch.float32)
            .reshape((8, width, size) if suffix == "weight" else (8, width))
            .div(100)
            for name, width in (("video_out", 6), ("audio_out", 4))
            for suffix, size in (("weight", 5), ("bias", 1))
        }
        self.path = self.root / "pdd_fused_heads.safetensors"
        save_file(self.heads, str(self.path))

    def layer(self, rank=0, tp_size=1):
        layer = object.__new__(MiniMaxH3FinalLayer)
        torch.nn.Module.__init__(layer)
        for name, width in (("video_out", 6), ("audio_out", 4)):
            setattr(
                layer,
                name,
                SimpleNamespace(
                    output_size=width,
                    input_size=5,
                    output_size_per_partition=width // tp_size,
                    tp_rank=rank,
                ),
            )
        layer.load_pdd_fused_heads(str(self.path))
        return layer

    def test_tp_projection_matches_full_head(self):
        h = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        layers = [self.layer(rank, 2) for rank in range(2)]
        for step in (0, 7):
            with set_forward_context(step, None):
                outputs = [layer._project(h) for layer in layers]
            for i, name in enumerate(("video_out", "audio_out")):
                expected = torch.nn.functional.linear(
                    h,
                    self.heads[f"{name}.weight"][step],
                    self.heads[f"{name}.bias"][step],
                )
                torch.testing.assert_close(
                    torch.cat([output[i] for output in outputs], dim=-1), expected
                )

    def test_schedule_validation_and_warmup(self):
        stage = object.__new__(MiniMaxH3TimestepPreparationStage)
        stage._pdd_config = {
            "num_inference_steps": 9,
            "video_shift": 12.0,
            "audio_shift": 3.0,
        }

        def batch(points, video_shift=12.0, audio_shift=3.0, warmup=False):
            return SimpleNamespace(
                num_inference_steps=points,
                is_warmup=warmup,
                extra={
                    MINIMAX_H3_SIGMAS_EXTRA_KEY: {
                        name: minimax_h3_time_shift_sigmas(
                            num_steps=points, shift_scale=shift
                        )
                        for name, shift in (
                            ("video", video_shift),
                            ("audio", audio_shift),
                        )
                    }
                },
            )

        valid = batch(9)
        stage._apply_pdd_schedule(valid)
        for invalid in (
            batch(5),
            batch(8),
            batch(10),
            batch(9, video_shift=6),
            batch(9, audio_shift=6),
        ):
            with self.assertRaisesRegex(ValueError, "match the fused heads"):
                stage._apply_pdd_schedule(invalid)
        for points in (1, 3, 50):
            warmup = batch(points, warmup=True)
            stage._apply_pdd_schedule(warmup)
            for name in ("video", "audio"):
                self.assertEqual(
                    warmup.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY][name],
                    valid.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY][name][: max(2, points)],
                )

    def test_conversion_keeps_heads_outside_transformer(self):
        base = self.root / "base"
        base.mkdir()
        save_file(
            {"blocks.0.attn.out_proj.weight": torch.zeros(2, 3)},
            str(base / "model.safetensors"),
        )
        (base / "config.json").write_text("{}")
        index = {"weight_map": {"blocks.0.attn.out_proj.weight": "model.safetensors"}}
        (base / "model.safetensors.index.json").write_text(json.dumps(index))
        lora = {
            "transformer_blocks.0.attn.to_out.0.lora_down": torch.ones(1, 3),
            "transformer_blocks.0.attn.to_out.0.lora_up": torch.ones(2, 1),
        }
        for key, value in self.heads.items():
            lora[
                key.replace("video_out", "proj_out").replace(
                    "audio_out", "audio_proj_out"
                )
            ] = value
        lora_path = self.root / "lora.safetensors"
        save_file(
            lora,
            str(lora_path),
            metadata={"lora_rank": "1", "pdd_num_steps": "8", "pdd_block_size": "2"},
        )
        out = self.root / "out"
        with patch("sys.argv", ["build", str(base), str(lora_path), str(out)]):
            self.assertEqual(build.main(), 0)
        with patch("sys.argv", ["fuse", str(out)]):
            self.assertEqual(fusion.main(), 0)
        self.assertEqual(
            _list_safetensors_files(str(out / "transformer")),
            [str(out / "transformer/model.safetensors")],
        )
        torch.testing.assert_close(
            load_file(str(out / "transformer/model.safetensors"))[
                "blocks.0.attn.out_proj.weight"
            ],
            torch.ones(2, 3),
        )
        self.assertTrue((out / "pdd_fused_heads.safetensors").is_file())
        self.assertEqual(
            json.loads((out / "transformer/model.safetensors.index.json").read_text()),
            index,
        )
        self.assertEqual(
            json.loads((out / "pdd_config.json").read_text())["num_inference_steps"], 5
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
