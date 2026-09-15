"""EAGLE3 aux hidden capture for Kimi-K3 (kimi-k3-eagle3.1-mla contract).

The draft checkpoint is trained against one-based completed-layer ids
([2, 46, 90] on the 93-layer target) tapping the plain prefix stream -- NOT
the AttnRes aggregate that DSPARK captures. These tests pin that contract on
KimiK3LinearModel.forward with stub decoder layers, plus the setter
validation on KimiK3LinearForCausalLM.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.kimi_k3 as kimi_k3_mod
from sglang.srt.layers.attn_residual import aggregate_stream_torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM, KimiK3LinearModel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

_H = 64
_V = 128
_EPS = 1e-5


class _FakePPGroup:
    def __init__(self, world_size=1, is_last_rank=True):
        self.world_size = world_size
        self.is_first_rank = True
        self.is_last_rank = is_last_rank


class _NoopRecorder:
    def with_current_layer(self, _layer_idx):
        return contextlib.nullcontext()


def _make_linear_model(layers, *, attn_res_block_size, eagle3_ids):
    """Bare KimiK3LinearModel with stub layers; skips __init__ and the
    distributed stack (get_pp_group / recorder are patched at call sites)."""
    model = object.__new__(KimiK3LinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=_H,
        num_hidden_layers=len(layers),
        attn_res_block_size=attn_res_block_size,
        rms_norm_eps=_EPS,
    )
    model.pp_group = _FakePPGroup()
    model._dp_attention = False
    model._trim_padded_attn = False
    model.dspark_layers_to_capture = None
    model.eagle3_layers_to_capture = eagle3_ids
    model.alt_streams = None
    model.embed_tokens = nn.Embedding(_V, _H).to("cuda", torch.bfloat16)
    model.layers = nn.ModuleList(layers)
    model.start_layer = 0
    model.end_layer = len(layers)
    model.norm = RMSNorm(_H, eps=_EPS).to("cuda", torch.bfloat16)
    # Only referenced (never called) when the stub layers leave the bank empty.
    model.output_attn_res_proj = None
    model.output_attn_res_norm = None
    return model


def _run_model(model, input_ids):
    positions = torch.arange(input_ids.shape[0], device=input_ids.device)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False)
    )
    with (
        patch.object(kimi_k3_mod, "get_pp_group", lambda: model.pp_group),
        patch.object(
            kimi_k3_mod,
            "get_global_expert_distribution_recorder",
            lambda: _NoopRecorder(),
        ),
    ):
        return model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
        )


class _AttnResStubLayer(nn.Module):
    """Attn-res boundary protocol: returns (full_head, None, False).

    head evolves as head <- head * mul + add; mutate_in_place clobbers the
    incoming head tensor storage instead of producing a fresh one (to prove
    the capture clones).
    """

    def __init__(self, mul, add, mutate_in_place=False):
        super().__init__()
        self.mul = mul
        self.add = add
        self.mutate_in_place = mutate_in_place
        self.keep_sharded_seen = []

    def forward(
        self,
        *,
        positions,
        hidden_states,
        forward_batch,
        residual,
        attn_res,
        zero_allocator,
        input_sharded=False,
        keep_sharded=False,
    ):
        assert residual is None, "attn-res steady state carries (head, None)"
        self.keep_sharded_seen.append(keep_sharded)
        if self.mutate_in_place:
            hidden_states.mul_(self.mul)
            hidden_states.add_(self.add)
            return hidden_states, None, False
        return hidden_states * self.mul + self.add, None, False


class _StandardStubLayer(nn.Module):
    """Standard delayed-add protocol: returns (hidden, residual, False) with
    the stream split as hidden + residual."""

    def __init__(self, mul, add):
        super().__init__()
        self.mul = mul
        self.add = add

    def forward(
        self,
        *,
        positions,
        hidden_states,
        forward_batch,
        residual,
        attn_res,
        zero_allocator,
        input_sharded=False,
        keep_sharded=False,
    ):
        assert attn_res is None
        stream = hidden_states if residual is None else hidden_states + residual
        nxt = stream * self.mul + self.add
        return nxt * 0.75, nxt * 0.25, False


def _expected_heads(embeds, layers):
    heads = [embeds]
    for layer in layers:
        heads.append(heads[-1] * layer.mul + layer.add)
    return heads


class TestKimiK3Eagle3Capture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _inputs(self, num_tokens=5):
        gen = torch.Generator(device="cuda").manual_seed(0)
        return torch.randint(0, _V, (num_tokens,), generator=gen, device="cuda")

    def test_capture_prefix_stream_attn_res_path(self):
        torch.manual_seed(0)
        layers = [
            _AttnResStubLayer(mul=1.1, add=0.01),
            _AttnResStubLayer(mul=0.9, add=-0.02),
            _AttnResStubLayer(mul=1.05, add=0.03),
            _AttnResStubLayer(mul=0.95, add=0.0),
        ]
        model = _make_linear_model(layers, attn_res_block_size=2, eagle3_ids=(2, 4))
        input_ids = self._inputs()
        hidden, aux = _run_model(model, input_ids)

        embeds = model.embed_tokens(input_ids)
        heads = _expected_heads(embeds, layers)
        # one-based [2, 4] -> stream after zero-based layers 1 and 3.
        self.assertEqual(len(aux), 2)
        torch.testing.assert_close(aux[0], heads[2])
        torch.testing.assert_close(aux[1], heads[4])
        for captured in aux:
            self.assertEqual(captured.shape, (input_ids.shape[0], _H))
            self.assertEqual(captured.dtype, torch.bfloat16)
        self.assertEqual(hidden.shape, (input_ids.shape[0], _H))

    def test_capture_prefix_stream_standard_path(self):
        torch.manual_seed(0)
        layers = [
            _StandardStubLayer(mul=1.2, add=0.01),
            _StandardStubLayer(mul=0.8, add=0.02),
            _StandardStubLayer(mul=1.0, add=-0.01),
        ]
        model = _make_linear_model(layers, attn_res_block_size=None, eagle3_ids=(1, 3))
        input_ids = self._inputs()
        _, aux = _run_model(model, input_ids)

        embeds = model.embed_tokens(input_ids)
        heads = _expected_heads(embeds, layers)
        # one-based [1, 3] -> stream after zero-based layers 0 and 2; the
        # standard path still carries the delayed add (hidden + residual).
        self.assertEqual(len(aux), 2)
        torch.testing.assert_close(aux[0], heads[1])
        torch.testing.assert_close(aux[1], heads[3])

    def test_capture_is_cloned_from_stream(self):
        torch.manual_seed(0)
        # Layer after the capture point clobbers the incoming head in place;
        # a non-cloned capture would be corrupted by it.
        layers = [
            _AttnResStubLayer(mul=1.1, add=0.01),
            _AttnResStubLayer(mul=2.0, add=0.5, mutate_in_place=True),
            _AttnResStubLayer(mul=1.0, add=0.0),
        ]
        model = _make_linear_model(layers, attn_res_block_size=2, eagle3_ids=(1,))
        input_ids = self._inputs()
        _, aux = _run_model(model, input_ids)

        embeds = model.embed_tokens(input_ids)
        expected = embeds * 1.1 + 0.01  # head after zero-based layer 0
        torch.testing.assert_close(aux[0], expected)

    def test_sp_attn_res_carry_disabled_by_capture(self):
        torch.manual_seed(0)
        sp_patches = (
            patch.object(kimi_k3_mod.envs.SGLANG_K3_SP_ATTN_RES, "get", lambda: True),
            patch.object(kimi_k3_mod.k3_sp_collective, "enabled", lambda: True),
        )

        # Control: no capture configured -> the row-sharded carry engages.
        control_layers = [_AttnResStubLayer(mul=1.0, add=0.0) for _ in range(3)]
        control = _make_linear_model(
            control_layers, attn_res_block_size=2, eagle3_ids=None
        )
        input_ids = self._inputs()
        with sp_patches[0], sp_patches[1]:
            _run_model(control, input_ids)
        for layer in control_layers:
            self.assertEqual(layer.keep_sharded_seen, [True])

        layers = [_AttnResStubLayer(mul=1.0, add=0.0) for _ in range(3)]
        model = _make_linear_model(layers, attn_res_block_size=2, eagle3_ids=(2,))
        with sp_patches[0], sp_patches[1]:
            _run_model(model, input_ids)
        for layer in layers:
            # eagle3 capture forces the row-sharded carry off, like dspark's.
            self.assertEqual(layer.keep_sharded_seen, [False])

    def test_aggregate_stream_differs_from_prefix(self):
        # Pins WHY eagle3 must not reuse _dspark_capture_stream: with a
        # non-empty AttnRes bank the aggregate is a different feature stream.
        gen = torch.Generator(device="cuda").manual_seed(0)
        T, NB = 8, 2
        head = torch.randn(T, _H, generator=gen, device="cuda").bfloat16()
        bank = torch.randn(T, NB, _H, generator=gen, device="cuda").bfloat16()
        score_norm = RMSNorm(_H, eps=_EPS).to("cuda", torch.bfloat16)
        score_proj = nn.Linear(_H, 1, bias=False).to("cuda", torch.bfloat16)
        mixed = aggregate_stream_torch(
            head, bank, NB, lambda x: (score_proj(x), None), score_norm
        )
        self.assertFalse(torch.allclose(mixed, head))
        # nvb == 0 degenerates to the prefix itself (bank empty).
        self.assertIs(
            aggregate_stream_torch(
                head, bank, 0, lambda x: (score_proj(x), None), score_norm
            ),
            head,
        )


class TestSetEagle3LayersToCapture(CustomTestCase):
    def _bare_lm(self, *, num_layers=93, pp_world_size=1, is_last_rank=True):
        lm = object.__new__(KimiK3LinearForCausalLM)
        nn.Module.__init__(lm)
        lm.config = SimpleNamespace(num_hidden_layers=num_layers)
        lm.pp_group = _FakePPGroup(world_size=pp_world_size, is_last_rank=is_last_rank)
        lm.model = SimpleNamespace(eagle3_layers_to_capture=None)
        lm.capture_aux_hidden_states = False
        return lm

    def test_default_ids_for_93_layers_match_checkpoint(self):
        lm = self._bare_lm(num_layers=93)
        lm.set_eagle3_layers_to_capture(None)
        # [2, 46, 90]: the one-based ids kimi-k3-eagle3.1-mla declares.
        self.assertEqual(lm.model.eagle3_layers_to_capture, (2, 46, 90))
        self.assertTrue(lm.capture_aux_hidden_states)

    def test_checkpoint_ids_stored_unchanged(self):
        lm = self._bare_lm()
        lm.set_eagle3_layers_to_capture([2, 46, 90])
        self.assertEqual(lm.model.eagle3_layers_to_capture, (2, 46, 90))

    def test_unsorted_or_duplicated_ids_rejected(self):
        with self.assertRaises(ValueError):
            self._bare_lm().set_eagle3_layers_to_capture([46, 2, 90])
        with self.assertRaises(ValueError):
            self._bare_lm().set_eagle3_layers_to_capture([2, 2, 46])

    def test_out_of_range_ids_rejected(self):
        with self.assertRaises(ValueError):
            self._bare_lm().set_eagle3_layers_to_capture([0, 46, 90])
        with self.assertRaises(ValueError):
            self._bare_lm().set_eagle3_layers_to_capture([2, 46, 94])

    def test_pp_greater_than_one_rejected(self):
        with self.assertRaises(NotImplementedError):
            self._bare_lm(pp_world_size=2).set_eagle3_layers_to_capture([2, 46, 90])

    def test_non_last_rank_is_noop(self):
        lm = self._bare_lm(is_last_rank=False)
        lm.set_eagle3_layers_to_capture([2, 46, 90])
        self.assertIsNone(lm.model.eagle3_layers_to_capture)
        self.assertFalse(lm.capture_aux_hidden_states)


if __name__ == "__main__":
    unittest.main()
