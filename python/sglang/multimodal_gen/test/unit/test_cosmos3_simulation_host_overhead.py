import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.multimodal_gen.runtime.models.dits import cosmos3_sim_auto, cosmos_dreams
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_layout import (
    SensorGrid,
    StreamItem,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.sim_auto import (
    _FlatLayout,
)
from sglang.test.test_utils import CustomTestCase


class _UploadRecorder(TorchDispatchMode):
    """Records tensors built from host data (torch.tensor, or a list used as an
    index): a blocking host-to-device copy on CUDA. 0-dim ones are Python
    scalars, which kernels read on the host."""

    def __init__(self):
        super().__init__()
        self.uploads = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.lift_fresh.default and args[0].dim() > 0:
            self.uploads.append(args[0])
        return func(*args, **(kwargs or {}))


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        return tokens, None


def _items(condition_frames=()):
    grid = SensorGrid(2, 2, 1, 1)
    return [
        StreamItem("rgb_control", (0, 1), (0, 1, 2), grid, 0.1),
        StreamItem("rgb_target", (0, 1), (0, 1, 2), grid, 0.1, condition_frames),
        StreamItem("lidar_control", (2,), (0, 1), grid, 0.1),
        StreamItem("lidar_target", (2,), (0, 1), grid, 0.1),
    ]


class TestCosmos3SimulationHostOverhead(CustomTestCase):
    def test_dreams_null_action_slots_zeroed_without_uploads(self):
        attention = SimpleNamespace(
            local_num_attention_heads=1,
            local_num_key_value_heads=1,
            head_dim=2,
            to_qkv=lambda hidden: (torch.cat([hidden, hidden, hidden + 1], -1), None),
            norm_q=None,
            norm_k=None,
            attn=lambda q, k, v: q,
            to_out=lambda out: (out, None),
        )
        hidden = torch.randn(1, 20, 2)
        slices = cosmos_dreams.null_action_token_slices(
            (0, 1), num_frames=2, tokens_per_frame=10, action_tokens_per_frame=4
        )
        recorder = _UploadRecorder()
        with (
            patch.object(
                cosmos_dreams, "_apply_qwen3_qk_norm_rope", lambda q, k, *_: (q, k)
            ),
            recorder,
        ):
            _, _, v = cosmos_dreams._joint_attention(
                attention,
                hidden,
                prefix_k=torch.zeros(1, 0, 1, 2),
                prefix_v=torch.zeros(1, 0, 1, 2),
                cos_sin_cache=None,
                rope_positions=None,
                null_token_slices=slices,
            )
        self.assertEqual(recorder.uploads, [])
        expected = (hidden + 1).view(1, 20, 1, 2).clone()
        expected[:, [0, 1, 2, 3, 10, 11, 12, 13]] = 0.0
        self.assertTrue(torch.equal(v, expected))

    def test_sim_auto_steps_scatter_and_gather_without_uploads(self):
        items = _items(condition_frames=(0,))
        latents = [
            torch.randn(1, 3, 6, 2, 2),
            torch.randn(1, 3, 6, 2, 2),
            torch.randn(1, 4, 2, 2, 2),
            torch.randn(1, 4, 2, 2, 2),
        ]
        layout = _FlatLayout.of(
            items, latents, num_views=2, noisy_rgb=(1, 2), noisy_sweeps=(1,)
        )
        x = torch.randn(1, layout.size)
        recorder = _UploadRecorder()
        with recorder:
            current = layout.scatter(latents, x)
            flat = layout.gather([current[1], current[3]])
        self.assertEqual(recorder.uploads, [])
        # the generated frames hold x and read back as x; the others are untouched
        self.assertTrue(torch.equal(flat, x))
        rgb = current[1].view(1, 3, 2, 3, 2, 2)
        self.assertTrue(
            torch.equal(rgb[:, :, :, 0], latents[1].view(1, 3, 2, 3, 2, 2)[:, :, :, 0])
        )
        self.assertTrue(torch.equal(current[3][:, :, 0], latents[3][:, :, 0]))

    def _sim_auto_model(self):
        model = cosmos3_sim_auto.Cosmos3SimAutoTransformer.__new__(
            cosmos3_sim_auto.Cosmos3SimAutoTransformer
        )
        nn.Module.__init__(model)
        model._rope_inputs_key = model._rope_inputs = None
        model._generated_masks_key = model._generated_masks = None
        return model

    def test_sim_auto_time_embed_masks_built_once_per_chunk(self):
        model = self._sim_auto_model()
        model.proj_in = _Projection()
        model.patchify = lambda latent, frames, h, w: latent.flatten(2).transpose(1, 2)
        model.time_embedder = lambda timestep: timestep.view(1, 1).expand(1, 3)
        model.rig_view_embed = nn.Embedding(4, 3)
        model.rig_lidar_id = 3
        items = _items(condition_frames=(0,))[:2]
        latents = [torch.randn(1, 3, 6, 2, 2), torch.randn(1, 3, 6, 2, 2)]
        rig_view_ids = torch.tensor([0, 1])

        def embed(timestep):
            with patch.object(
                cosmos3_sim_auto, "add_rig_view_rows", lambda hidden, rows, n: hidden
            ):
                return model.embed_items(
                    items, latents, timestep=timestep, rig_view_ids=rig_view_ids
                )

        timesteps = [torch.tensor(500.0), torch.tensor(250.0)]
        with patch.object(torch, "tensor", wraps=torch.tensor) as upload:
            steps = [embed(timestep) for timestep in timesteps]
        # one generated-frame mask (the target's) for the whole chunk
        self.assertEqual(upload.call_count, 1)
        model._generated_masks_key = None
        self.assertTrue(torch.equal(steps[1], embed(timesteps[1])))

    def test_sim_auto_rope_inputs_built_once_per_chunk(self):
        model = self._sim_auto_model()
        items = _items()[::2]  # controls only: no output projection
        num_tokens = sum(item.num_tokens for item in items)
        model.num_hidden_layers = 0
        model.gen_layers = []
        model.ulysses = None
        model.norm_moe_gen = lambda hidden: hidden
        model.embed_items = lambda items, latents, **_: torch.zeros(1, num_tokens, 3)
        model.chunk_position_ids = Mock(return_value=torch.zeros(3, num_tokens))
        model.language_model = SimpleNamespace(
            rotary_emb=SimpleNamespace(
                build_rope_cache_inputs=lambda ids, cache_dtype: (ids, ids)
            )
        )
        for timestep in (torch.ones(1), torch.zeros(1), None):
            model(
                items=items,
                latents=[None, None],
                timestep=timestep,
                text_extent=7,
                und_kv=[],
                history_kv=[],
                mask=None,
                rig_view_ids=None,
            )
        # the chunk's denoise steps and its clean refresh share the RoPE inputs
        self.assertEqual(model.chunk_position_ids.call_count, 1)


if __name__ == "__main__":
    unittest.main()
