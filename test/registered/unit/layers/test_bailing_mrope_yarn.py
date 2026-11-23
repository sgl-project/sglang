"""Bailing VideoRoPE with YaRN must keep the language model's MLA RoPE on text."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
# The DsV3MLA selection and cos/sin cache checks run on CPU (no kernel launch).
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import copy
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.srt.configs.model_config import ModelConfig, compute_mla_mscale_scaling
from sglang.srt.layers.rotary_embedding import (
    bailing_mrope,
    base,
    factory,
    rope_variant,
)
from sglang.srt.layers.rotary_embedding.bailing_mrope import (
    BailingMRotaryEmbedding,
    YaRNScalingBailingMRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.yarn import deepseek_yarn_inv_freq
from sglang.srt.models.bailing_moe_v3 import (
    DsV3MLA,
    isolate_bailing_video_rope,
    resolve_bailing_mla_rope,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import published_topology


@pytest.fixture(autouse=True)
def rotary_runtime(monkeypatch):
    monkeypatch.setattr(factory, "_ROPE_DICT", {})
    if not torch.cuda.is_available():
        monkeypatch.setattr(base, "_is_cpu", True)
        monkeypatch.setattr(rope_variant, "get_device", lambda: "cpu")
        monkeypatch.setattr(bailing_mrope, "get_device", lambda: "cpu")
    with published_topology(device="cuda" if torch.cuda.is_available() else "cpu"):
        yield


THETA = 6000000
SECTION = [8, 12, 12]
YARN = dict(
    factor=4,
    original_max_position_embeddings=262144,
    beta_fast=32,
    beta_slow=1,
    mscale=1,
    mscale_all_dim=0,
)
# float32 keeps the CPU cache unrounded; on CUDA the cache is float32 regardless.
DTYPE = torch.float32


class DsV3MLAVideoRopeTests(unittest.TestCase):
    """Replays DsV3MLA's VideoRoPE handling around the MLA parent's in-place retag."""

    isolate = staticmethod(isolate_bailing_video_rope)
    resolve = staticmethod(resolve_bailing_mla_rope)

    def run_layer(self, params, skip_rope=False):
        seen = {}
        parent_init = DeepseekV2AttentionMLA.__init__

        def parent(module, config, *args, **kwargs):
            seen["parent_scaling"] = args[8]
            parent_init(module, config, *args, **kwargs)

        def capture(*args, rope_scaling, **kwargs):
            seen["rope_type"] = rope_scaling["rope_type"]
            return torch.nn.Identity()

        group = SimpleNamespace(world_size=1, rank_in_group=0, cpu_group=None)
        config = BailingHybridConfig(hidden_size=128, num_attention_heads=2)
        with (
            get_parallel().override(tp_group=group, attn_tp_group=group),
            torch.device("meta"),
            patch.object(DeepseekV2AttentionMLA, "__init__", parent),
            patch(
                "sglang.srt.models.deepseek_v2.get_rope_wrapper",
                return_value=torch.nn.Identity(),
            ),
            patch("sglang.srt.models.bailing_moe_v3.get_rope", side_effect=capture),
        ):
            DsV3MLA(
                config,
                128,
                2,
                128,
                64,
                128,
                None,
                512,
                rope_theta=THETA,
                rope_scaling=params,
                max_position_embeddings=1048576,
                skip_rope=skip_rope,
            )
        return seen["parent_scaling"], seen.get("rope_type")

    def video(self, **params):
        return dict(params, rope_theta=THETA, mrope_section=SECTION, video_rope=True)

    def test_plain_video_rope_stays_unscaled(self):
        params = self.video(rope_type="default")
        _, rope_type = self.run_layer(params)
        self.assertEqual(rope_type, "default")
        self.assertEqual(params["rope_type"], "default")

    def test_yarn_request_keeps_deepseek_yarn_without_touching_config(self):
        for request in (dict(rope_type="yarn"), dict(type="yarn"), {}):
            with self.subTest(**request):
                params = self.video(**request, **YARN)
                original = dict(params)
                _, rope_type = self.run_layer(params)
                self.assertEqual(rope_type, "deepseek_yarn")
                self.assertEqual(params, original)

    def test_leftover_factor_under_default_type_scales_nothing(self):
        params = self.video(rope_type="default", **dict(YARN, mscale_all_dim=1))
        parent_view, rope_type = self.run_layer(params)
        self.assertEqual(rope_type, "default")
        # Language-only mode resolves the same dict to no scaling at all.
        self.assertEqual(compute_mla_mscale_scaling(parent_view, 0.125), 0.125)

    def test_yarn_attention_scaling_matches_language_only(self):
        params = self.video(rope_type="yarn", **dict(YARN, mscale_all_dim=1))
        parent_view, _ = self.run_layer(params)
        text = {
            k: v for k, v in params.items() if k not in ("mrope_section", "video_rope")
        }
        text["rope_type"] = "deepseek_yarn"
        self.assertEqual(
            compute_mla_mscale_scaling(parent_view, 0.125),
            compute_mla_mscale_scaling(text, 0.125),
        )

    def test_yarn_type_without_factor_stays_unscaled(self):
        _, rope_type = self.run_layer(self.video(rope_type="yarn"))
        self.assertEqual(rope_type, "default")

    def test_text_dict_is_passed_through(self):
        params = dict(rope_type="yarn", rope_theta=THETA, **YARN)
        self.assertIs(self.isolate(params), params)

    def test_default_text_resolves_without_scaling(self):
        config = SimpleNamespace(
            rope_parameters=dict(rope_type="default", rope_theta=THETA, **YARN)
        )
        self.assertEqual(self.resolve(config), (THETA, None))
        self.assertEqual(self.run_layer(None), (None, None))

    def test_skip_rope_does_not_construct_video_rope(self):
        params = self.video(rope_type="yarn", **YARN)
        _, rope_type = self.run_layer(params, skip_rope=True)
        self.assertIsNone(rope_type)
        self.assertEqual(params["rope_type"], "yarn")


class YaRNBailingRotaryTests(unittest.TestCase):
    def pair(self, orig, factor, **kw):
        yarn = {**dict(beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0), **kw}
        text = DeepseekScalingRotaryEmbedding(
            64, 64, orig, THETA, False, factor, DTYPE, **yarn
        )
        video = YaRNScalingBailingMRotaryEmbedding(
            64,
            64,
            orig,
            THETA,
            False,
            factor,
            DTYPE,
            mrope_section=SECTION,
            video_rope=True,
            **yarn,
        )
        return text, video

    def rows(self, rope, positions):
        index = torch.as_tensor(positions) - rope.position_start
        return rope.cos_sin_cache[index.to(rope.cos_sin_cache.device)]

    def test_factory_selects_variants(self):
        video = dict(
            YARN,
            rope_type="deepseek_yarn",
            original_max_position_embeddings=4096,
            mrope_section=SECTION,
            video_rope=True,
        )
        rope = get_rope(64, 64, 16384, THETA, False, video, DTYPE)
        self.assertIsInstance(rope, YaRNScalingBailingMRotaryEmbedding)
        self.assertIs(get_rope(64, 64, 16384, THETA, False, dict(video), DTYPE), rope)
        self.assertEqual(rope.position_start, -16384)
        self.assertEqual(rope.mrope_section, [12, 12, 8])
        text = {
            k: v for k, v in video.items() if k not in ("mrope_section", "video_rope")
        }
        self.assertIs(
            type(get_rope(64, 64, 16384, THETA, False, text, DTYPE)),
            DeepseekScalingRotaryEmbedding,
        )
        # The MLA parent retags plain VideoRoPE dicts deepseek_yarn; they must not scale.
        plain = dict(rope_type="deepseek_yarn", mrope_section=SECTION, video_rope=True)
        self.assertIs(
            type(get_rope(64, 64, 4096, THETA, False, plain, DTYPE)),
            DeepseekScalingRotaryEmbedding,
        )
        native = dict(plain, rope_type="default")
        self.assertIs(
            type(get_rope(64, 64, 4096, THETA, False, native, DTYPE)),
            BailingMRotaryEmbedding,
        )

    def test_text_rows_match_deepseek_yarn(self):
        cases = [
            dict(orig=4096, factor=4),
            dict(orig=2048, factor=2, mscale_all_dim=1),
            dict(orig=1024, factor=8, attn_factor=1.25, mscale=0.707),
            dict(orig=1024, factor=1, attn_factor=1.25),
        ]
        for case in cases:
            with self.subTest(**case):
                text, video = self.pair(**case)
                n = text.cos_sin_cache.shape[0]
                start = -video.position_start
                # The initial cache covers the whole YaRN window, factor 8 included.
                rows = video.cos_sin_cache[start : start + n]
                if rows.is_cuda:
                    self.assertTrue(torch.equal(rows, text.cos_sin_cache))
                else:
                    # CPU vectorized sin/cos may differ by an ULP with different
                    # batch sizes, even when frequencies and angles are equal.
                    torch.testing.assert_close(
                        rows, text.cos_sin_cache, rtol=1e-6, atol=1e-7
                    )

    def test_cache_covers_centred_media_offsets_for_any_factor(self):
        # Original 128 with factor 16: a 2046-row image centres H at -1022.
        _, video = self.pair(orig=128, factor=16)
        self.assertLessEqual(video.position_start, -1022)
        self.assertGreaterEqual(
            video.cos_sin_cache.shape[0] + video.position_start, 2048
        )

    def test_negative_rows_mirror_positive_rows(self):
        _, video = self.pair(orig=1024, factor=4)
        positions = torch.tensor([1, 7, 1000, 4000, 4095])
        pos, neg = self.rows(video, positions), self.rows(video, -positions)
        torch.testing.assert_close(neg[:, :32], pos[:, :32], rtol=0, atol=1e-6)
        torch.testing.assert_close(neg[:, 32:], -pos[:, 32:], rtol=0, atol=1e-6)

    def test_cache_growth_keeps_offset_and_scaling(self):
        grown = list(range(4096, 20001, 37))
        _, video = self.pair(orig=256, factor=4)
        video._ensure_cos_sin_cache_length(10000)
        device = video.cos_sin_cache.device
        inv = deepseek_yarn_inv_freq(
            base=THETA,
            rotary_dim=64,
            max_position_embeddings=256,
            scaling_factor=4,
            beta_fast=32,
            beta_slow=1,
            extrapolation_factor=1,
            device=device,
        )
        t = torch.tensor(grown, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv)
        expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * video.mscale
        torch.testing.assert_close(self.rows(video, grown), expected, rtol=0, atol=1e-5)

    def test_factory_cache_covers_context_larger_than_yarn_window(self):
        # The serving context may exceed original * factor. A centered image
        # must still fit on the negative side, even after positive cache growth.
        scaling = dict(
            rope_type="deepseek_yarn",
            factor=4,
            original_max_position_embeddings=128,
            mrope_section=SECTION,
            video_rope=True,
        )
        video = get_rope(64, 64, 2048, THETA, False, scaling, DTYPE)
        config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=1),
            text_config=SimpleNamespace(
                max_position_embeddings=2048,
                image_patch_token=10,
                video_patch_token=11,
                image_start_token=20,
                video_start_token=21,
                rope_parameters=scaling,
            ),
        )
        tokens = torch.tensor([20] + [10] * 2046 + [30])
        positions, _ = video.bailing_3drope_get_input_positions_tensor(
            tokens, config, torch.tensor([[1, 2046, 1]]), None
        )
        self.assertEqual(positions.min().item(), -1021)
        old_start = video.position_start
        old = video.cos_sin_cache.clone()
        video._ensure_cos_sin_cache_length(2303)
        self.assertEqual(video.position_start, old_start)
        self.assertTrue(torch.equal(video.cos_sin_cache[: len(old)], old))
        indices = positions - video.position_start
        self.assertGreaterEqual(indices.min().item(), 0)
        self.assertLess(indices.max().item(), video.cos_sin_cache.shape[0])
        self.assertGreater(video.cos_sin_cache.shape[0] + video.position_start, 4606)

    def test_unit_mscale_growth_preserves_yarn_frequencies(self):
        _, video = self.pair(orig=128, factor=4, mscale_all_dim=1)
        self.assertEqual(video.mscale, 1.0)
        video._ensure_cos_sin_cache_length(2048)
        inv = video._compute_inv_freq(video.base)
        freqs = torch.einsum(
            "i,j -> ij", torch.tensor([1024.0, 4096.0], device=inv.device), inv
        )
        expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        torch.testing.assert_close(self.rows(video, [1024, 4096]), expected)

    def test_default_factory_ignores_leftover_yarn_keys(self):
        scaling = dict(
            rope_type="default", **YARN, mrope_section=SECTION, video_rope=True
        )
        video = get_rope(64, 64, 1024, THETA, False, scaling, DTYPE)
        self.assertIs(type(video), BailingMRotaryEmbedding)
        self.assertEqual(video.scaling_factor, 1.0)
        self.assertEqual(video.mscale, 1.0)

    def qk(self, n, heads=3):
        gen = torch.Generator(device="cuda").manual_seed(0)
        query = torch.randn(
            n, heads, 192, dtype=torch.bfloat16, device="cuda", generator=gen
        )
        key = torch.randn(n, 1, 576, dtype=torch.bfloat16, device="cuda", generator=gen)
        return query[..., 128:], key[..., 512:]

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_text_forward_matches_1d_mla_rope_bitwise(self):
        for is_neox in (False, True):
            with self.subTest(is_neox=is_neox):
                yarn = dict(beta_fast=32, beta_slow=1, mscale=1, mscale_all_dim=0)
                text = DeepseekScalingRotaryEmbedding(
                    64, 64, 262144, THETA, is_neox, 4, torch.bfloat16, **yarn
                )
                video = YaRNScalingBailingMRotaryEmbedding(
                    64,
                    64,
                    262144,
                    THETA,
                    is_neox,
                    4,
                    torch.bfloat16,
                    mrope_section=SECTION,
                    video_rope=True,
                    **yarn,
                )
                gen = torch.Generator(device="cuda").manual_seed(3)
                positions = torch.randint(
                    0, 1048576, (4096,), device="cuda", generator=gen
                )
                positions[:4] = torch.tensor(
                    [0, 262144, 1040000, 1048575], device="cuda"
                )
                query, key = self.qk(4096, heads=16)
                a, b = text(positions, query.clone(), key.clone())
                c, d = video(positions.expand(3, -1), query.clone(), key.clone())
                self.assertTrue(torch.equal(a, c))
                self.assertTrue(torch.equal(b, d))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_media_positions_use_scaled_frequencies_per_axis(self):
        text, video = self.pair(orig=262144, factor=4)
        gen = torch.Generator().manual_seed(1)
        t = torch.randint(0, 1040000, (64,), generator=gen)
        h = t + torch.randint(-40, 40, (64,), generator=gen)
        w = t + torch.randint(-40, 40, (64,), generator=gen)
        h[0], w[0] = -17, -3
        positions = torch.stack([t, h, w]).cuda()
        query, key = self.qk(64)
        c, d = video(positions, query.clone(), key.clone())
        # Bailing layout: pairs [0, 24) alternate h/w, pairs [24, 32) carry t. Take each
        # pair's cos/sin from the 1D YaRN cache row of that axis's position.
        axis = torch.tensor([j % 2 + 1 if j < 24 else 0 for j in range(32)])
        per_pair = positions.cpu()[axis].T
        cache = text.cos_sin_cache
        rows = cache[per_pair.abs().to(cache.device)].cpu().double()
        pair = torch.arange(32)
        cos = rows[:, pair, pair][:, None, :]
        sin = (rows[:, pair, pair + 32] * per_pair.sign())[:, None, :]

        def rotate(x):
            x = x.cpu().double()
            even, odd = x[..., 0::2], x[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = even * cos - odd * sin
            out[..., 1::2] = odd * cos + even * sin
            return out

        torch.testing.assert_close(
            c.cpu().double(), rotate(query), rtol=0.008, atol=0.015625
        )
        torch.testing.assert_close(
            d.cpu().double(), rotate(key), rtol=0.008, atol=0.015625
        )


@pytest.mark.parametrize("storage", ["checkpoint", "override"])
@pytest.mark.parametrize("placement", ["top", "text"])
@pytest.mark.parametrize("key", ["rope_scaling", "rope_parameters"])
def test_yarn_checkpoint_and_override_reach_target_and_nextn(
    tmp_path, storage, placement, key
):
    config = {
        "model_type": "bailing_moe_v3_vl",
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "mrope_section": SECTION,
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 42,
            "num_nextn_predict_layers": 1,
            "num_attention_heads": 32,
            "max_position_embeddings": 262080,
            "rope_theta": THETA,
            "short_conv_kernel_size": 4,
            "expert_swiglu_limit_list": [4] * 42,
            "share_expert_swiglu_limit_list": [4] * 42,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    rope = dict(
        YARN,
        rope_type="yarn",
        rope_theta=THETA,
        partial_rotary_factor=0.5,
        mrope_section=SECTION,
        video_rope=True,
    )
    request = {"text_config": {"max_position_embeddings": 1048576}}
    (request if placement == "top" else request["text_config"])[key] = rope
    if storage == "checkpoint":
        config["text_config"].update(request.pop("text_config"))
        config.update(request)
        request = {}
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = dict(
        model_path=str(tmp_path),
        trust_remote_code=False,
        model_override_args=json.dumps(request),
    )
    target = ModelConfig(**args)
    draft = ModelConfig(**args, is_draft_model=True, speculative_algorithm="NEXTN")
    before = copy.deepcopy(target.hf_text_config.rope_parameters)
    assert draft.hf_text_config.rope_parameters == before
    for model in (target, draft):
        assert model.context_len == 1048576
        assert model.hf_text_config.max_position_embeddings == 1048576
        assert all(
            model.hf_text_config.rope_parameters[k] == v for k, v in rope.items()
        )
        assert model.model_is_mrope
    assert (
        draft.hf_text_config.rope_parameters
        is not target.hf_text_config.rope_parameters
    )
    projected = target.hf_config.get_nextn_config()
    projected.rope_parameters["rope_type"] = "deepseek_yarn"
    assert target.hf_text_config.rope_parameters == before


def test_default_video_cache_keeps_existing_reservation():
    rope = get_rope(
        64,
        64,
        1024,
        THETA,
        False,
        dict(rope_type="default", mrope_section=SECTION, video_rope=True),
        DTYPE,
    )
    before = rope.cos_sin_cache
    rope._ensure_cos_sin_cache_length(1500)
    assert rope.cos_sin_cache is before
    assert rope.position_start == -1024
    assert rope.cos_sin_cache.shape == (2048, 64)
    assert not rope._match_cuda_rope


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
