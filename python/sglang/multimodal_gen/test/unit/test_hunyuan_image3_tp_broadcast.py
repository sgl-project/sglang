"""HunyuanImage-3 denoising: tensor-parallel input broadcasts, per-step host
work, the prefix K/V cache, the MoE block and the timestep schedule."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits import (
    hunyuan_image3,
    hunyuan_image3_utils,
)
from sglang.multimodal_gen.runtime.pipelines.hunyuan_image3 import (
    HunyuanImage3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    ar_stage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import forbid_host_syncs
from sglang.test.test_utils import CustomTestCase


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def contiguous(self):
        return self

    def reshape(self, *_shape):
        return self


class _FakeTPGroup:
    world_size = 2

    def __init__(self):
        self.broadcast_calls = []

    def broadcast(self, tensor, src):
        self.broadcast_calls.append((tensor, src))
        return tensor


class TestHunyuanImage3TPBroadcast(CustomTestCase):
    def test_static_inputs_are_broadcast_once(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        attention_mask = _FakeTensor((2, 1, 8, 8))
        cos = _FakeTensor((2, 8, 4))
        sin = _FakeTensor((2, 8, 4))

        with (
            patch.object(ar_stage, "model_parallel_is_initialized", return_value=True),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._broadcast_static_inputs(attention_mask, (cos, sin))

        self.assertEqual(result, (attention_mask, (cos, sin)))
        self.assertEqual(
            tp_group.broadcast_calls,
            [(attention_mask, 0), (cos, 0), (sin, 0)],
        )

    def test_backbone_forward_only_broadcasts_dynamic_hidden_states(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        hidden_states = _FakeTensor((2, 3, 4))
        attention_mask = _FakeTensor((2, 1, 3, 3))
        cos = _FakeTensor((2, 3, 2))
        sin = _FakeTensor((2, 3, 2))
        output = MagicMock()
        output.shape = (6, 4)
        output.view.return_value = "reshaped-output"
        stage._cache_dit_adapter = None
        stage.ar_model = SimpleNamespace(forward_block=MagicMock(return_value=output))

        with (
            patch.object(ar_stage, "model_parallel_is_initialized", return_value=True),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._backbone_forward(
                num_image_tokens=2,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                custom_pos_emb=(cos, sin),
                first_step=True,
            )

        self.assertEqual(result, "reshaped-output")
        self.assertEqual(tp_group.broadcast_calls, [(hidden_states, 0)])


class TestHunyuanImage3DenoiseStep(CustomTestCase):
    def test_timestep_frequencies_are_built_once(self):
        # the host-built frequencies are uploaded once, not on each of the five
        # timestep embeddings per step (each pageable upload blocks the host)
        hunyuan_image3_utils._timestep_freqs.cache_clear()
        t = torch.tensor([999.0, 500.0])
        half = 128
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        expected = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        with patch.object(hunyuan_image3_utils.torch, "exp", wraps=torch.exp) as exp:
            for _ in range(3):
                actual = hunyuan_image3_utils.timestep_embedding(t, 2 * half)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(exp.call_count, 1)

    def test_image_tokens_move_through_precomputed_positions(self):
        # scatter/gather through positions taken once per request: the per-step
        # masked_select sized its output on the host every step
        generator = torch.Generator().manual_seed(0)
        bsz, seqlen, n_embd, token_h, token_w = 2, 10, 6, 2, 2
        image_mask = torch.zeros(bsz, seqlen, dtype=torch.bool)
        image_mask[0, 3:7] = True
        image_mask[1, 5:9] = True
        hidden = torch.randn(bsz, seqlen, n_embd, generator=generator)
        image_seq = torch.randn(bsz, token_h * token_w, n_embd, generator=generator)
        stage = object.__new__(ar_stage.HunyuanImage3AR)
        stage.ar_model = SimpleNamespace(
            time_embed=lambda t: t,
            time_embed_2=lambda t: t,
            patch_embed=lambda images, t_emb: (image_seq, token_h, token_w),
            final_layer=lambda x, t_emb, h, w: x,
        )
        reference_index = (
            torch.arange(seqlen).unsqueeze(0).expand(bsz, -1).masked_select(image_mask)
        ).reshape(bsz, -1)
        scattered = hidden.clone().scatter_(
            1, reference_index.unsqueeze(-1).expand(-1, -1, n_embd), image_seq
        )
        extracted = hidden.masked_select(image_mask.unsqueeze(-1)).reshape(
            -1, token_h * token_w, n_embd
        )

        index = stage._mask_positions(image_mask, bsz, seqlen, hidden.device)
        t = torch.tensor([999.0, 999.0])
        with forbid_host_syncs():
            actual = (
                stage._instantiate_vae_tokens_first_step(
                    hidden.clone(), None, t, index
                ),
                stage._extract_diffusion_pred(hidden, t, index, token_h, token_w),
            )
        torch.testing.assert_close(actual, (scattered, extracted), atol=0, rtol=0)


class TestHunyuanImage3PrefixKVCache(CustomTestCase):
    BS, SEQ, HEADS, KV_HEADS, DIM, PREFIX = 2, 16, 4, 2, 8, 8

    def _mask(self):
        idx = torch.arange(self.SEQ)
        mask = (idx.unsqueeze(0) <= idx.unsqueeze(1)).repeat(self.BS, 1, 1)
        mask[:, 2:5, 2:5] = True  # clean image block inside the prefix
        mask[:, 8:14, 8:14] = True  # noisy image block after it
        return mask.unsqueeze(1)

    def _qkv(self, seed):
        g = torch.Generator().manual_seed(seed)
        n = self.BS * self.SEQ
        return (
            torch.randn(n, self.HEADS, self.DIM, generator=g),
            torch.randn(n, self.KV_HEADS, self.DIM, generator=g),
            torch.randn(n, self.KV_HEADS, self.DIM, generator=g),
        )

    def _rows(self, t):
        return t.view(self.BS, self.SEQ, *t.shape[1:])

    def _suffix(self, t):
        return self._rows(t)[:, self.PREFIX :].reshape(-1, *t.shape[1:])

    def _meta(self, mask, first_step, prefix_len=None, cache_slot=0):
        meta = hunyuan_image3_utils.create_hunyuan_image_attention_meta(
            mask, None, first_step
        )
        meta.prefix_len = prefix_len
        meta.cache_slot = cache_slot
        return meta

    def _later_step(self, step0):
        # A later denoising step: new suffix rows, prefix rows unchanged.
        later = self._qkv(seed=2)
        for src, dst in zip(step0, later):
            self._rows(dst)[:, : self.PREFIX] = self._rows(src)[:, : self.PREFIX]
        return later

    def _suffix_step(self, manager, qkv, mask, cache_slot=0):
        suffix_mask = mask[:, :, self.PREFIX :, :]
        return manager(
            *(self._suffix(t) for t in qkv),
            self._meta(suffix_mask, False, self.PREFIX, cache_slot),
            attention_mask=suffix_mask,
        )

    def test_suffix_step_matches_full_recompute(self):
        mask = self._mask()
        step0 = self._qkv(seed=1)
        later = self._later_step(step0)
        manager = hunyuan_image3_utils.ImageKVCacheManager()
        manager(*step0, self._meta(mask, True, self.PREFIX), attention_mask=mask)

        out = self._suffix_step(manager, later, mask)

        full = hunyuan_image3_utils.ImageKVCacheManager()(
            *later, self._meta(mask, False), attention_mask=mask
        )
        torch.testing.assert_close(out, self._suffix(full))

    def test_cfg_slots_keep_separate_prefixes(self):
        """The cond and uncond halves share each attention module; caching one
        half's prefix must not replace the other's."""
        mask = self._mask()
        step0 = self._qkv(seed=1)
        later = self._later_step(step0)
        manager = hunyuan_image3_utils.ImageKVCacheManager()
        manager(*step0, self._meta(mask, True, self.PREFIX, 0), attention_mask=mask)
        manager(
            *self._qkv(seed=3),
            self._meta(mask, True, self.PREFIX, 1),
            attention_mask=mask,
        )

        out = self._suffix_step(manager, later, mask, cache_slot=0)

        full = hunyuan_image3_utils.ImageKVCacheManager()(
            *later, self._meta(mask, False), attention_mask=mask
        )
        torch.testing.assert_close(out, self._suffix(full))

    def test_prefix_ends_at_first_varying_token_unless_a_block_straddles(self):
        image_mask = torch.zeros(1, self.SEQ, dtype=torch.bool)
        image_mask[0, 8:14] = True
        timestep_index = torch.tensor([[7]])
        clean_before = SimpleNamespace(
            gen_image_slices=[[slice(8, 14)]], joint_image_slices=[[slice(2, 5)]]
        )
        self.assertEqual(
            ar_stage._step_invariant_prefix_len(
                image_mask, timestep_index, clean_before
            ),
            7,
        )

        straddling = SimpleNamespace(
            gen_image_slices=[[slice(8, 14)]], joint_image_slices=[[slice(5, 9)]]
        )
        self.assertIsNone(
            ar_stage._step_invariant_prefix_len(image_mask, None, straddling)
        )

    def test_segmented_attention_matches_masked_attention(self):
        """FA3 segments (causal text runs, bidirectional image blocks) must equal
        SDPA with the dense mask, for full steps and suffix-only steps."""
        mask = self._mask()
        blocks = [slice(2, 5), slice(8, 14)]
        segments = tuple(
            hunyuan_image3_utils.build_attention_segments(self.SEQ, blocks)
            for _ in range(self.BS)
        )
        self.assertIsNone(
            hunyuan_image3_utils.build_attention_segments(
                self.SEQ, [slice(2, 6), slice(5, 9)]
            )
        )

        def attend(q, k, v, causal):
            # Reference for FA3 varlen: GQA, causal mask aligned bottom-right.
            g = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
            m = None
            if causal:
                m = torch.ones(q.shape[0], k.shape[0], dtype=torch.bool).tril(
                    k.shape[0] - q.shape[0]
                )
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=m
            )
            return out.transpose(0, 1)

        q, k, v = self._qkv(seed=4)
        full = hunyuan_image3_utils.ImageKVCacheManager()(
            q, k, v, self._meta(mask, False), attention_mask=mask
        )
        seg = hunyuan_image3_utils.segmented_attention(
            self._rows(q), self._rows(k), self._rows(v), segments, attend
        )
        torch.testing.assert_close(seg.reshape(full.shape), full)

        suffix = hunyuan_image3_utils.segmented_attention(
            self._rows(q)[:, self.PREFIX :],
            self._rows(k),
            self._rows(v),
            segments,
            attend,
        )
        torch.testing.assert_close(suffix, self._rows(full)[:, self.PREFIX :])


class TestHunyuanImage3MoEAndSchedule(CustomTestCase):
    def test_shared_mlp_sees_layer_input_when_experts_run_in_place(self):
        """The CUDA FusedMoE kernel overwrites its input with the expert
        output; the shared MLP must still see the layer input."""
        block = hunyuan_image3.HunYuanSparseMoeBlock.__new__(
            hunyuan_image3.HunYuanSparseMoeBlock
        )
        torch.nn.Module.__init__(block)

        def experts_in_place(x, topk_output):
            out = x * 10
            x.copy_(out)
            return out

        block.gate = lambda x: (torch.zeros(x.shape[0], 2), None)
        block.topk = lambda x, logits: None
        block.experts = experts_in_place
        block.shared_mlp = lambda x: x.clone()
        x = torch.randn(4, 8)

        out = block(x.clone())

        torch.testing.assert_close(out, 11 * x)

    def test_timesteps_follow_official_schedule(self):
        """Official FlowMatchDiscreteScheduler: t_i = 1000 * shift(1 - i / N)
        for i < N, with shift(s) = k * s / (1 + (k - 1) * s)."""
        shift, steps = 3.0, 10
        scheduler = HunyuanImage3Pipeline._build_scheduler(shift)
        scheduler.set_timesteps(steps)

        sigmas = torch.linspace(1, 0, steps + 1)[:-1]
        expected = 1000 * shift * sigmas / (1 + (shift - 1) * sigmas)
        torch.testing.assert_close(
            scheduler.timesteps.float().cpu(), expected, rtol=0, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
