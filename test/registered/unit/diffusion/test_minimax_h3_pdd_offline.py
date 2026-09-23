# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
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
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


def _make_heads() -> dict[str, torch.Tensor]:
    heads = {}
    for name, width in (("video_out", 6), ("audio_out", 4)):
        heads[f"{name}.weight"] = (
            torch.arange(8 * width * 5, dtype=torch.float32).reshape(8, width, 5) / 100
        )
        heads[f"{name}.bias"] = (
            torch.arange(8 * width, dtype=torch.float32).reshape(8, width) / 100
        )
    return heads


class TestMiniMaxH3PDDOffline(CustomTestCase):
    def test_tp_projection_matches_full_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heads = _make_heads()
            path = root / "heads.safetensors"
            save_file(heads, str(path))
            arch = MiniMaxH3DiTArchConfig(
                hidden_size=5, latents_dim=6, audio_latents_dim=4, patch_size=(1, 1, 1)
            )
            layers = []
            for rank in range(2):
                with patch(
                    "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                    return_value=SimpleNamespace(world_size=2, rank_in_group=rank),
                ):
                    layer = MiniMaxH3FinalLayer(
                        arch, None, prefix="final_layer", use_adaln_cache=True
                    )
                layer.load_pdd_fused_heads(str(path))
                layers.append(layer)

            h = torch.arange(15, dtype=torch.float32).reshape(3, 5)
            for step in (0, 7):
                with self.subTest(step=step), set_forward_context(step, None):
                    outputs = [layer._project(h) for layer in layers]
                    for i, name in enumerate(("video_out", "audio_out")):
                        expected = torch.nn.functional.linear(
                            h,
                            heads[f"{name}.weight"][step],
                            heads[f"{name}.bias"][step],
                        )
                        torch.testing.assert_close(
                            torch.cat([output[i] for output in outputs], dim=-1),
                            expected,
                        )

    def test_schedule_validation_and_warmup(self):
        apply_schedule = partial(
            MiniMaxH3TimestepPreparationStage._apply_pdd_schedule,
            SimpleNamespace(
                _pdd_config={
                    "num_inference_steps": 9,
                    "video_shift": 12.0,
                    "audio_shift": 3.0,
                }
            ),
        )

        sigmas = {
            name: minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=shift)
            for name, shift in (("video", 12.0), ("audio", 3.0))
        }
        batch = SimpleNamespace(
            is_warmup=False, extra={MINIMAX_H3_SIGMAS_EXTRA_KEY: sigmas}
        )
        apply_schedule(batch)
        wrong_shift = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=6.0)
        for name, invalid in (
            ("video", sigmas["video"][:-1]),
            ("video", sigmas["video"] + [0.0]),
            ("video", wrong_shift),
            ("audio", wrong_shift),
        ):
            batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = {**sigmas, name: invalid}
            with (
                self.subTest(modality=name, sigmas=invalid),
                self.assertRaisesRegex(ValueError, "match the fused heads"),
            ):
                apply_schedule(batch)
        batch.is_warmup = True
        batch.num_inference_steps = 3
        apply_schedule(batch)
        self.assertEqual(
            batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY],
            {name: values[:3] for name, values in sigmas.items()},
        )

    def test_conversion_keeps_heads_outside_transformer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            heads = _make_heads()
            base = root / "base"
            base.mkdir()
            weight_name = "blocks.0.attn.out_proj.weight"
            save_file({weight_name: torch.zeros(2, 3)}, str(base / "model.safetensors"))
            (base / "config.json").write_text("{}")
            index = {"weight_map": {weight_name: "model.safetensors"}}
            (base / "model.safetensors.index.json").write_text(json.dumps(index))
            lora = {
                "transformer_blocks.0.attn.to_out.0.lora_down": torch.ones(1, 3),
                "transformer_blocks.0.attn.to_out.0.lora_up": torch.ones(2, 1),
            }
            for key, value in heads.items():
                lora[
                    key.replace("video_out", "proj_out").replace(
                        "audio_out", "audio_proj_out"
                    )
                ] = value
            lora_path = root / "lora.safetensors"
            save_file(
                lora,
                str(lora_path),
                metadata={
                    "lora_rank": "1",
                    "pdd_num_steps": "8",
                    "pdd_block_size": "2",
                },
            )
            out = root / "out"
            build.main([str(base), str(lora_path), str(out)])
            transformer = out / "transformer"
            self.assertEqual(
                list(transformer.glob("*.safetensors")),
                [transformer / "model.safetensors"],
            )
            torch.testing.assert_close(
                load_file(str(transformer / "model.safetensors"))[weight_name],
                torch.ones(2, 3),
            )
            self.assertEqual(
                json.loads((transformer / "model.safetensors.index.json").read_text()),
                index,
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
