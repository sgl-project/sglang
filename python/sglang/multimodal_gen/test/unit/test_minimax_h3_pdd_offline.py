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

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_PDD_FUSED_HEADS_WEIGHT_NAME_PREFIX,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    MiniMaxH3FinalLayer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import WeightsUpdater
from sglang.multimodal_gen.test.unit.host_sync_guard import CountedUploads
from sglang.multimodal_gen.tools import build_minimax_h3_pdd_weights as build
from sglang.test.test_utils import CustomTestCase


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

    def test_heads_are_uploaded_once_not_per_step(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "heads.safetensors"
            heads = _make_heads()
            save_file(heads, str(path))
            arch = MiniMaxH3DiTArchConfig(
                hidden_size=5, latents_dim=6, audio_latents_dim=4, patch_size=(1, 1, 1)
            )
            with patch(
                "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                return_value=SimpleNamespace(world_size=1, rank_in_group=0),
            ):
                layer = MiniMaxH3FinalLayer(
                    arch, None, prefix="final_layer", use_adaln_cache=True
                )
            layer.load_pdd_fused_heads(str(path))
            # the loaded heads stay on the host; count their copies to the device
            layer._pdd_heads = {
                key: value.as_subclass(CountedUploads)
                for key, value in layer._pdd_heads.items()
            }
            CountedUploads.uploads = 0
            h = torch.arange(15, dtype=torch.float32).reshape(3, 5)
            for step in range(8):
                with set_forward_context(step, None):
                    outputs = layer._project(h)
                expected = [
                    torch.nn.functional.linear(
                        h, heads[f"{name}.weight"][step], heads[f"{name}.bias"][step]
                    )
                    for name in ("video_out", "audio_out")
                ]
                torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
            # one weight and one bias stack per head for the whole loop
            self.assertEqual(CountedUploads.uploads, 4)

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
            # Three warmup updates need four sigma boundaries on the serving grid.
            {name: values[:4] for name, values in sigmas.items()},
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


def _make_final_layers(
    heads_path: Path | None, *, tp_size: int
) -> list[MiniMaxH3FinalLayer]:
    """One final layer per TP rank, with the heads file loaded when given."""
    arch = MiniMaxH3DiTArchConfig(
        hidden_size=5, latents_dim=6, audio_latents_dim=4, patch_size=(1, 1, 1)
    )
    layers = []
    for rank in range(tp_size):
        with patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
            return_value=SimpleNamespace(world_size=tp_size, rank_in_group=rank),
        ):
            layer = MiniMaxH3FinalLayer(
                arch, None, prefix="final_layer", use_adaln_cache=True
            )
        if heads_path is not None:
            layer.load_pdd_fused_heads(str(heads_path))
        layers.append(layer)
    return layers


def _project_on_each_rank(
    layers: list[MiniMaxH3FinalLayer], *, step: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # float64 rows: the stacks are kept in float32, so the copies a forward holds are copies, as on a GPU
    h = torch.arange(15, dtype=torch.float64).reshape(3, 5)
    with set_forward_context(step, None):
        return [layer._project(h) for layer in layers]


def _make_transformer_updater(
    final_layer: MiniMaxH3FinalLayer,
) -> tuple[WeightsUpdater, MiniMaxH3DiTModel]:
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model.adaln_cache = None
    model.final_layer = final_layer
    model.register_parameter("probe", torch.nn.Parameter(torch.zeros(2)))
    pipeline = SimpleNamespace(modules={"transformer": model}, model_path="")
    return WeightsUpdater(pipeline), model


class TestMiniMaxH3PDDHeadWeightUpdate(CustomTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.loaded_path = root / "loaded.safetensors"
        save_file(_make_heads(), str(self.loaded_path))
        self.updated_heads = {
            key: 1 - 2 * stack for key, stack in _make_heads().items()
        }
        self.updated_path = root / "updated.safetensors"
        save_file(self.updated_heads, str(self.updated_path))

    def test_head_stack_update_serves_what_loading_it_serves(self):
        """On each TP rank, and after a forward already holds the old stacks."""
        layers = _make_final_layers(self.loaded_path, tp_size=2)
        _project_on_each_rank(layers, step=0)

        for layer in layers:
            for key, full_stack in self.updated_heads.items():
                layer.load_pdd_fused_head_stack(key=key, full_stack=full_stack)

        layers_loaded_updated = _make_final_layers(self.updated_path, tp_size=2)
        for step in (0, 7):
            with self.subTest(step=step):
                torch.testing.assert_close(
                    _project_on_each_rank(layers, step=step),
                    _project_on_each_rank(layers_loaded_updated, step=step),
                    rtol=0,
                    atol=0,
                )

    def test_head_stack_update_keeps_its_own_copy(self):
        """The update tensor is the sender's buffer, which the sender reuses."""
        layers = _make_final_layers(self.loaded_path, tp_size=1)
        for key, full_stack in self.updated_heads.items():
            sender_buffer = full_stack.clone()
            layers[0].load_pdd_fused_head_stack(key=key, full_stack=sender_buffer)
            sender_buffer.zero_()

        torch.testing.assert_close(
            _project_on_each_rank(layers, step=3),
            _project_on_each_rank(
                _make_final_layers(self.updated_path, tp_size=1), step=3
            ),
            rtol=0,
            atol=0,
        )

    def test_tensor_update_loads_head_stacks_beside_parameters(self):
        layers = _make_final_layers(self.loaded_path, tp_size=1)
        updater, model = _make_transformer_updater(layers[0])
        named_tensors = [
            (f"{MINIMAX_H3_PDD_FUSED_HEADS_WEIGHT_NAME_PREFIX}{key}", full_stack)
            for key, full_stack in self.updated_heads.items()
        ]
        named_tensors.insert(1, ("probe", torch.ones(2)))

        ok, message = updater.update_weights_from_tensor(named_tensors)

        self.assertTrue(ok, message)
        torch.testing.assert_close(
            _project_on_each_rank(layers, step=3),
            _project_on_each_rank(
                _make_final_layers(self.updated_path, tp_size=1), step=3
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(model.probe, torch.ones(2))

    def test_tensor_update_fails_on_a_head_stack_it_cannot_load(self):
        """Skipping the stack would leave the engine serving the heads it had."""
        prefix = MINIMAX_H3_PDD_FUSED_HEADS_WEIGHT_NAME_PREFIX
        full_stack = self.updated_heads["video_out.weight"]
        cases = (
            (None, "video_out.weight", full_stack, "serves no PDD heads"),
            (self.loaded_path, "video_out.weight", full_stack[:6], "must have shape"),
            (self.loaded_path, "video_out.scale", full_stack, "must be one of"),
        )
        for heads_path, key, update_tensor, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                layers = _make_final_layers(heads_path, tp_size=1)
                updater, _ = _make_transformer_updater(layers[0])

                ok, message = updater.update_weights_from_tensor(
                    [(f"{prefix}{key}", update_tensor)]
                )

                self.assertFalse(ok)
                self.assertIn(expected_message, message)


if __name__ == "__main__":
    unittest.main(verbosity=3)
