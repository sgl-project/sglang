# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Wan-Animate-2 block mask, shape math and per-clip reference K/V ownership.

Single-process and CPU only, in the style of ``test_sp_shard.py`` / ``test_usp_ring_tail_pad.py``:
``_get_block_mask_from_layout`` is called unbound with a stub ``self`` and patched module globals; the
shape helpers are called directly. The parallelism rejections live in ``test_wan_animate_2_config.py``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.wan_animate_2 import WanAnimate2Config
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2 import (
    WanAnimate2Transformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_block import (
    WanAnimate2TransformerBlock,
    _apply_rope_interleaved,
    _InContextLayout,
    _make_score_mod,
    _sp_gather_qkv,
    _sp_scatter_out,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_clip_conditioning import (
    WanAnimate2ClipConditioning,
    WanAnimate2ReferenceKV,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformerBlock
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.denoising import (
    WanAnimate2DenoisingStage,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

_DIT = "sglang.multimodal_gen.runtime.models.dits.wan_animate_2"
_BLOCK = "sglang.multimodal_gen.runtime.models.dits.wan_animate_2_block"
_DENOISING = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".wan_animate_2.denoising"
)


# A short last clip against the full-length clip the mask is built for: this clip is
# (2, 1, 2) = 2 frames x 2 tokens, the reference video (1, 1, 2), the full clip (4, 2, 2) =
# 4 frames x 4 tokens. Every mask-side field (max_*) differs from the clip's own, so a mask
# built from the clip's grid, or attention padding that disagrees with the mask, fails below.
_SHORT_CLIP_LAYOUT = _InContextLayout((2, 1, 2), (1, 1, 2), (4, 2, 2))


def _mask_probe(mask_fn):
    def valid(q_idx, kv_idx):
        out = mask_fn(
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(q_idx),
            torch.tensor(kv_idx),
        )
        return bool(out.item())

    return valid


def test_block_mask_head_agnostic_and_masks_correctly():
    captured = {}

    def fake_create_block_mask(mask_fn, *, B, H, Q_LEN, KV_LEN, device, _compile):
        captured.update(mask_fn=mask_fn, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN)
        return "SENTINEL_BLOCK_MASK"

    stub = SimpleNamespace(_block_masks={}, _device=torch.device("cpu"))
    layout = _SHORT_CLIP_LAYOUT
    with patch(f"{_DIT}.create_block_mask", side_effect=fake_create_block_mask) as cbm:
        first = WanAnimate2Transformer3DModel._get_block_mask_from_layout(stub, layout)
        second = WanAnimate2Transformer3DModel._get_block_mask_from_layout(stub, layout)

    # Cached: the second call never rebuilds.
    assert first == "SENTINEL_BLOCK_MASK"
    assert second is first
    assert cbm.call_count == 1

    # Head/batch-agnostic so SP head-sharding cannot change the mask.
    assert captured["B"] is None
    assert captured["H"] is None
    # Sized from the full clip (16 gen + 12 reference-video tokens), each side ceil'd to 128.
    assert captured["Q_LEN"] == 128
    assert captured["KV_LEN"] == 256

    valid = _mask_probe(captured["mask_fn"])
    # Generation queries attend every generation token of the full clip, not the gen pad.
    assert valid(0, 0) and valid(5, 15)
    assert not valid(5, 16)
    # A pad query row attends nothing.
    assert not valid(16, 0)
    # Frame stride is the full clip's 4 tokens per frame, not this clip's 2: query 5 is in
    # generation frame 1 and attends reference-video frame 1 (keys 128..131) only. A mask
    # built from the clip's own grid would put query 5 in frame 2 and query 2 in frame 1.
    assert valid(5, 128) and valid(5, 131)
    assert not valid(5, 132)
    assert not valid(2, 128)
    # Frame 0 is the reference image; it has no reference-video counterpart.
    assert not valid(0, 128)
    # Frame 3 attends keys 136..139; 140 is past the 12 reference-video tokens (pad).
    assert valid(13, 136) and valid(13, 139)
    assert not valid(13, 140)


def test_attention_pads_tokens_into_the_slots_the_block_mask_admits():
    # The mask is built once per request from the full clip while the attention pads every
    # clip's tokens; a frame stride or reference-video offset that drifts between the two
    # is a silent attention-pattern bug no shape check catches.
    layout = _SHORT_CLIP_LAYOUT
    captured = {}

    def fake_flex_attention(q, k, v, *, block_mask, score_mod):
        captured.update(q=q, k=k, v=v, block_mask=block_mask)
        return torch.zeros_like(q)

    batch, heads, dim = 1, 2, 8

    def tokens(values, scale):
        base = torch.tensor(values, dtype=torch.float32).view(1, len(values), 1, 1)
        return (base * scale).expand(batch, len(values), heads, dim).clone()

    # Token ids 1..4 are generation frame 0 (1, 2) and frame 1 (3, 4); 1, 2 x 1000 are the
    # reference-video frame.
    q, k, v = (
        tokens([1, 2, 3, 4], 1.0),
        tokens([1, 2, 3, 4], 10.0),
        tokens([1, 2, 3, 4], 100.0),
    )
    reference_k, reference_v = tokens([1, 2], 1000.0), tokens([1, 2], 10000.0)
    stub = SimpleNamespace(log_scale=0.0)
    with patch(f"{_BLOCK}._flex_attention_compiled", side_effect=fake_flex_attention):
        out = WanAnimate2TransformerBlock._in_context_attention(
            stub, q, k, v, reference_k, reference_v, "SENTINEL_BLOCK_MASK", layout
        )

    assert captured["block_mask"] == "SENTINEL_BLOCK_MASK"
    assert tuple(out.shape) == (batch, 4, heads, dim)
    q_padded = captured["q"].transpose(1, 2)  # back to [B, L, N, D]
    k_padded = captured["k"].transpose(1, 2)
    v_padded = captured["v"].transpose(1, 2)
    assert tuple(q_padded.shape) == (
        batch,
        layout.padded_generation_video_len,
        heads,
        dim,
    )
    assert tuple(k_padded.shape) == (batch, layout.padded_kv_len, heads, dim)

    def slot(t, i):
        return t[0, i, 0, 0].item()

    # Generation tokens sit frame by frame at the full clip's 4-token stride.
    assert [slot(q_padded, i) for i in range(8)] == [1, 2, 0, 0, 3, 4, 0, 0]
    assert [slot(k_padded, i) for i in range(8)] == [10, 20, 0, 0, 30, 40, 0, 0]
    assert q_padded[:, 8:].abs().sum() == 0
    # Reference-video tokens start at padded_generation_video_len with the same stride.
    reference_start = layout.padded_generation_video_len
    assert [slot(k_padded, reference_start + i) for i in range(4)] == [1000, 2000, 0, 0]
    assert [slot(v_padded, reference_start + i) for i in range(4)] == [
        10000,
        20000,
        0,
        0,
    ]
    assert k_padded[:, 8:reference_start].abs().sum() == 0
    assert k_padded[:, reference_start + 4 :].abs().sum() == 0

    # Every slot that holds a real token is one the mask built from the same layout routes:
    # all gen pairs, and frame-1 queries (slots 4, 5) to the reference-video keys.
    valid = _mask_probe(layout.block_mask_fn())
    gen_slots = [0, 1, 4, 5]
    assert all(valid(qi, ki) for qi in gen_slots for ki in gen_slots)
    assert all(valid(qi, reference_start + ki) for qi in (4, 5) for ki in (0, 1))
    assert not any(valid(qi, reference_start + ki) for qi in (0, 1) for ki in (0, 1))


def test_in_context_attention_hands_flex_a_score_mod_only_when_log_scale_is_nonzero():
    # At log_scale 0 the bias is an exact identity, so flex must run without a score_mod;
    # otherwise it must get the cached partial (compiled flex guards on its identity).
    layout = _SHORT_CLIP_LAYOUT
    captured = []

    def fake_flex_attention(q, k, v, *, block_mask, score_mod):
        captured.append(score_mod)
        return torch.zeros_like(q)

    gen, reference = torch.zeros(1, 4, 2, 8), torch.zeros(1, 2, 2, 8)
    with patch(f"{_BLOCK}._flex_attention_compiled", side_effect=fake_flex_attention):
        for log_scale in (0.0, -1.3):
            WanAnimate2TransformerBlock._in_context_attention(
                SimpleNamespace(log_scale=log_scale),
                gen,
                gen,
                gen,
                reference,
                reference,
                "SENTINEL_BLOCK_MASK",
                layout,
            )
    assert captured[0] is None
    assert captured[1] is _make_score_mod(layout.max_tokens_per_frame, -1.3)


def test_make_score_mod_is_cached_by_key():
    # Compiled flex_attention guards on score_mod identity; a fresh closure per call
    # would recompile every clip.
    assert _make_score_mod(4, 0.0) is _make_score_mod(4, 0.0)
    assert _make_score_mod(4, 0.0) is not _make_score_mod(4, 1.3)
    assert _make_score_mod(4, 0.0) is not _make_score_mod(8, 0.0)


def test_score_mod_bias_region_matches_the_official_formula():
    # The official wan_animate_2_model.py adds log_scale to keys with index in [hw, 2*hw),
    # hw = full-clip tokens per frame (external literal: the distilled checkpoint depends on
    # it). In the padded [gen | reference-video] layout those are the keys of generation
    # frame 1; the reference-video slice at padded_generation_video_len is not biased.
    layout = _SHORT_CLIP_LAYOUT
    hw = layout.max_tokens_per_frame  # 4
    score_mod = _make_score_mod(hw, 5.0)

    def biased(kv_idx):
        out = score_mod(torch.tensor([1.0]), 0, 0, 0, torch.tensor(kv_idx))
        return out.item() == 6.0

    probes = list(range(0, 3 * hw)) + [layout.padded_generation_video_len]
    assert [kv for kv in probes if biased(kv)] == [4, 5, 6, 7]

    # Base weights ship log_scale 0.0: the score_mod must then be an exact identity.
    identity = _make_score_mod(hw, 0.0)
    score = torch.randn(5)
    for kv_idx in (0, hw, 2 * hw, layout.padded_generation_video_len):
        assert torch.equal(
            identity(score.clone(), 0, 0, 0, torch.tensor(kv_idx)), score
        )


class _FakeAllToAll4D:
    """Single-process all_to_all_4D for SP=2: split scatter_dim into (SP, .) and fold
    that factor onto gather_dim (and the inverse)."""

    SP = 2

    def __init__(self):
        self.calls = []

    def __call__(self, x, scatter_dim, gather_dim):
        self.calls.append((scatter_dim, gather_dim))
        b, s, h, c = x.shape
        sp = self.SP
        if (scatter_dim, gather_dim) == (2, 1):
            assert h % sp == 0
            return (
                x.reshape(b, s, sp, h // sp, c)
                .permute(0, 2, 1, 3, 4)
                .reshape(b, sp * s, h // sp, c)
            )
        if (scatter_dim, gather_dim) == (1, 2):
            assert s % sp == 0
            return (
                x.reshape(b, sp, s // sp, h, c)
                .permute(0, 2, 1, 3, 4)
                .reshape(b, s // sp, sp * h, c)
            )
        raise AssertionError((scatter_dim, gather_dim))


def _reference_pass_stub(captured: list) -> SimpleNamespace:
    # Minimal block for forward_ref: identity norm/modulation, q/k/v straight from the stream,
    # attn1 records its keyword arguments.
    def attn1(q, k, v, **kwargs):
        captured.append({"q": q, "k": k, **kwargs})
        return q

    return SimpleNamespace(
        local_num_heads=2,
        _modulate=lambda t: tuple(torch.zeros(1, 1, 8) for _ in range(6)),
        norm1=lambda x, shift, scale: x,
        _qkv=lambda x, n: (
            x.reshape(1, -1, n, 4),
            x.reshape(1, -1, n, 4) + 1,
            x.reshape(1, -1, n, 4) + 2,
        ),
        attn1=attn1,
        _cross_and_ffn=lambda hidden, attn, *rest: hidden,
    )


def _run_reference_pass(
    stub,
    *,
    seq_len: int,
    num_valid: int,
    sp_size: int,
    key_mask=None,
    k_cache=None,
    v_cache=None,
    cos=None,
    sin=None,
):
    cos = torch.ones(num_valid, 2) if cos is None else cos
    sin = torch.zeros(num_valid, 2) if sin is None else sin
    with (
        patch(f"{_BLOCK}.get_sp_world_size", return_value=sp_size),
        patch(f"{_BLOCK}._sp_gather_qkv", side_effect=lambda q, k, v: (q, k, v)),
        patch(f"{_BLOCK}._sp_scatter_out", side_effect=lambda x: x),
    ):
        return WanAnimate2TransformerBlock.forward_ref(
            stub,
            hidden_states=torch.arange(seq_len * 8, dtype=torch.float32).reshape(
                1, seq_len, 8
            ),
            encoder_hidden_states=torch.zeros(1, 4, 8),
            timestep_modulation=torch.zeros(1, 6, 8),
            reference_video_rope_cos=cos,
            reference_video_rope_sin=sin,
            reference_video_num_tokens=num_valid,
            index=0,
            k_cache={} if k_cache is None else k_cache,
            v_cache={} if v_cache is None else v_cache,
            reference_video_key_mask=key_mask,
        )


@pytest.mark.parametrize("seq_len,sp_size", [(6, 1), (8, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_cache_reuses_rotated_key_and_preserves_padding(
    seq_len, sp_size, dtype
):
    captured, keys, values = [], {}, {}
    angles = torch.arange(12, dtype=torch.float32).view(6, 2) / 7
    stub = _reference_pass_stub(captured)
    qkv = stub._qkv
    raw_keys = []

    def project(x, n):
        projected = tuple(t.to(dtype) for t in qkv(x, n))
        raw_keys.append(projected[1])
        return projected

    stub._qkv = project
    _run_reference_pass(
        stub,
        seq_len=seq_len,
        num_valid=6,
        sp_size=sp_size,
        k_cache=keys,
        v_cache=values,
        cos=angles.cos(),
        sin=angles.sin(),
    )
    raw = torch.arange(seq_len * 8, dtype=torch.float32).view(1, seq_len, 2, 4)
    expected = torch.cat(
        [
            _apply_rope_interleaved(
                (raw[:, :6] + 1).to(dtype), angles.cos(), angles.sin()
            ),
            (raw[:, 6:] + 1).to(dtype),
        ],
        dim=1,
    )
    torch.testing.assert_close(keys[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(values[0], (raw + 2).to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(keys[0], captured[0]["k"], rtol=0, atol=0)
    assert keys[0].data_ptr() == raw_keys[0].data_ptr()


def test_reference_pass_passes_no_key_mask_when_the_sequence_has_no_padding():
    # A dense all-true key mask would route the reference stream through SDPA instead of the
    # configured attention backend; with no padding the pass must hand attn1 attn_mask=None and
    # RoPE the whole sequence.
    captured: list = []
    _run_reference_pass(
        _reference_pass_stub(captured), seq_len=6, num_valid=6, sp_size=1
    )
    assert len(captured) == 1
    assert captured[0]["attn_mask"] is None
    assert captured[0]["q"].shape == (1, 6, 2, 4)


@pytest.mark.parametrize("seq_len,sp_size", [(4, 1), (6, 2)])
def test_generation_reuses_reference_cache_without_rotating_it(seq_len, sp_size):
    stub = _reference_pass_stub([])
    captured = []

    def attend(q, k, v, ref_k, ref_v, *rest):
        captured.append((q, k, ref_k, ref_v))
        return q

    stub._in_context_attention = attend
    keys = {0: torch.randn(1, 2, 2, 4)}
    values = {0: torch.randn_like(keys[0])}
    saved_key = keys[0].clone()
    angles = torch.arange(8, dtype=torch.float32).view(4, 2) / 7
    hidden = torch.arange(seq_len * 8, dtype=torch.float32).view(1, seq_len, 8)
    with (
        patch(f"{_BLOCK}.get_sp_world_size", return_value=sp_size),
        patch(f"{_BLOCK}._sp_gather_qkv", side_effect=lambda q, k, v: (q, k, v)),
        patch(f"{_BLOCK}._sp_scatter_out", side_effect=lambda x: x),
    ):
        for _ in range(2):
            WanAnimate2TransformerBlock.forward_gen(
                stub,
                hidden_states=hidden,
                encoder_hidden_states=torch.zeros(1, 4, 8),
                timestep_modulation=torch.zeros(1, 6, 8),
                gen_cos=angles.cos(),
                gen_sin=angles.sin(),
                block_mask=None,
                layout=_SHORT_CLIP_LAYOUT,
                index=0,
                k_cache=keys,
                v_cache=values,
            )
    raw = hidden.view(1, seq_len, 2, 4)
    for q, k, ref_k, ref_v in captured:
        for actual, source in ((q, raw), (k, raw + 1)):
            expected = torch.cat(
                [
                    _apply_rope_interleaved(source[:, :4], angles.cos(), angles.sin()),
                    source[:, 4:],
                ],
                dim=1,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert ref_k is keys[0] and ref_v is values[0]
    torch.testing.assert_close(keys[0], saved_key, rtol=0, atol=0)


def test_reference_pass_masks_the_sp_tail_padding_with_the_per_clip_mask():
    # Gathered under SP the sequence carries tail padding: the pad keys are masked, and a mask
    # built once per clip by the caller is used as is.
    captured: list = []
    _run_reference_pass(
        _reference_pass_stub(captured), seq_len=8, num_valid=6, sp_size=2
    )
    mask = captured[0]["attn_mask"]
    assert mask is not None and mask.dtype == torch.bool and tuple(mask.shape) == (1, 8)
    assert mask[0, :6].all() and not mask[0, 6:].any()
    assert captured[0]["skip_sequence_parallel_override"] is True

    per_clip = torch.tensor([[True] * 6 + [False] * 2])
    captured.clear()
    _run_reference_pass(
        _reference_pass_stub(captured),
        seq_len=8,
        num_valid=6,
        sp_size=2,
        key_mask=per_clip,
    )
    assert captured[0]["attn_mask"] is per_clip


def test_gather_then_scatter_round_trips():
    # _sp_gather_qkv: [B,S_local,N_tp,C] -> [B,S_full,N_local,C]; _sp_scatter_out inverts it.
    fake = _FakeAllToAll4D()
    with patch(f"{_BLOCK}.sequence_model_parallel_all_to_all_4D", side_effect=fake):
        q = torch.arange(1 * 4 * 8 * 16, dtype=torch.float32).reshape(1, 4, 8, 16)
        k = q + 1000.0
        v = q + 2000.0

        gq, gk, gv = _sp_gather_qkv(q, k, v)
        assert tuple(gq.shape) == (1, 8, 4, 16)
        assert tuple(gk.shape) == (1, 8, 4, 16)
        assert tuple(gv.shape) == (1, 8, 4, 16)

        back = _sp_scatter_out(gq)
        assert tuple(back.shape) == tuple(q.shape)
        assert torch.equal(back, q)

    assert fake.calls == [(2, 1), (1, 2)]


@pytest.mark.parametrize(
    "full_clip_grid, expected",
    [
        # 4 frames x 4 tokens: 16 gen + 12 reference-video tokens, one 128 block each.
        ((4, 2, 2), (4, 16, 12, 128, 128, 256)),
        # 4 frames x 100 tokens: 400 -> 512, 300 -> 384.
        ((4, 10, 10), (100, 400, 300, 512, 384, 896)),
    ],
)
def test_layout_pads_the_full_clip_geometry_to_128_token_blocks(
    full_clip_grid, expected
):
    # The reference video has no counterpart for the generation stream's reference-image
    # frame, so its token count is one frame short of the generation stream's.
    layout = _InContextLayout((3, 2, 2), (2, 2, 2), full_clip_grid)
    assert (
        layout.max_tokens_per_frame,
        layout.max_generation_video_num_tokens,
        layout.max_reference_video_num_tokens,
        layout.padded_generation_video_len,
        layout.padded_reference_video_len,
        layout.padded_kv_len,
    ) == expected


class _KvTaggingTransformer:
    """Fake DiT: ``build_reference_kv`` tags the K/V with the clip's reference-video latents,
    ``__call__`` records the (clip_cond, reference_kv) pair each forward was handed."""

    def __init__(self):
        self.context_calls = 0
        self.forward_calls: list[
            tuple[WanAnimate2ClipConditioning, WanAnimate2ReferenceKV]
        ] = []

    def build_reference_kv(self, *, clip_cond):
        latents = clip_cond.reference_video_latents
        return WanAnimate2ReferenceKV(k={0: latents}, v={0: latents})

    def prepare_context(self, prompt_embeddings, image_embeddings):
        self.context_calls += 1
        return prompt_embeddings

    def __call__(self, *, hidden_states, clip_cond, reference_kv, **_unused):
        self.forward_calls.append((clip_cond, reference_kv))
        return hidden_states.unsqueeze(0)


class _TwoStepScheduler:
    timesteps = (torch.tensor(1000.0), torch.tensor(500.0))

    def set_timesteps(self, *, sigmas, device):
        pass

    def step(self, noise_pred, t, sample, return_dict):
        return (sample,)


def _clip_conditioning(reference_video_latents):
    return WanAnimate2ClipConditioning(
        reference_video_latents=reference_video_latents,
        generation_video_grid_sizes=(2, 1, 1),
        reference_video_frame_0_image_embeddings=torch.zeros(1, 1, 4),
        reference_video_condition=torch.zeros(20, 1, 2, 2),
        prompt_ref_embeddings=torch.zeros(1, 4),
        generation_condition=torch.zeros(20, 2, 2, 2),
        reference_video_grid_sizes=(1, 1, 1),
        full_clip_grid_sizes=(2, 1, 1),
        num_frames=4,
        init_noise=torch.zeros(16, 2, 2, 2),
    )


def _batch_without_perf_recording() -> SimpleNamespace:
    """The clip loop reads only the perf-recording fields off the batch (and hands it to
    set_forward_context); no metrics object means no step records."""
    return SimpleNamespace(metrics=None, perf_dump_path=None)


def _fake_denoising_stage(transformer, *, events: list[str]) -> SimpleNamespace:
    """Stand-in for WanAnimate2DenoisingStage in _denoise_single_clip; the residency hooks
    record into ``events`` so a test can read the use interval back."""
    return SimpleNamespace(
        transformer=transformer,
        scheduler=_TwoStepScheduler(),
        step_profile=lambda: None,
        begin_declared_component_use=lambda *, component_name, module: events.append(
            f"begin:{component_name}"
        ),
        _finish_active_component_use=lambda: events.append("finish"),
    )


def _single_clip_request_state(guidance_scale: float) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_embeddings=torch.zeros(1, 4),
        negative_prompt_embeddings=torch.zeros(1, 4),
        reference_image_embeddings=torch.zeros(1, 1, 4),
        inputs=SimpleNamespace(num_inference_steps=2, guidance_scale=guidance_scale),
    )


def test_each_clip_is_denoised_with_kv_from_its_own_reference_video():
    """Consecutive clips that share seed and clip index but not the reference video (two
    requests on one DiT) must each be denoised with K/V built from their own reference video,
    never with K/V left over from the previous one."""
    transformer = _KvTaggingTransformer()
    stage = _fake_denoising_stage(transformer, events=[])
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(flow_shift=5.0), enable_cfg_parallel=False
    )
    # guidance_scale 1.0: CFG off, one forward per step.
    request_state = _single_clip_request_state(guidance_scale=1.0)
    first = _clip_conditioning(torch.full((16, 1, 2, 2), 1.0))
    second = _clip_conditioning(torch.full((16, 1, 2, 2), 2.0))

    with patch(
        f"{_DENOISING}.get_local_torch_device", return_value=torch.device("cpu")
    ):
        for clip_cond in (first, second):
            WanAnimate2DenoisingStage._denoise_single_clip(
                stage,
                batch=_batch_without_perf_recording(),
                server_args=server_args,
                request_state=request_state,
                clip_condition=clip_cond,
                clip_index=0,
            )

    # 2 clips x 2 steps; every forward of a clip gets the K/V built from that clip's latents.
    assert len(transformer.forward_calls) == 4
    assert transformer.context_calls == 2
    for clip_cond, reference_kv in transformer.forward_calls:
        assert reference_kv.k[0] is clip_cond.reference_video_latents
    assert transformer.forward_calls[0][1] is not transformer.forward_calls[2][1]


def test_dit_work_of_a_clip_runs_inside_one_declared_transformer_use():
    """The stage declares three component uses, so the residency manager does not activate
    the DiT at stage entry; the reference pass and every CFG forward of a clip must run after
    the transformer use begins and before it is finished, or an offloaded DiT is never
    brought back to the device."""
    events: list[str] = []

    class _EventTransformer(_KvTaggingTransformer):
        def build_reference_kv(self, *, clip_cond):
            events.append("reference_kv")
            return super().build_reference_kv(clip_cond=clip_cond)

        def __call__(self, **kwargs):
            events.append("forward")
            return super().__call__(**kwargs)

        def prepare_context(self, *args):
            events.append("context")
            return super().prepare_context(*args)

    stage = _fake_denoising_stage(_EventTransformer(), events=events)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(flow_shift=5.0), enable_cfg_parallel=False
    )
    with patch(
        f"{_DENOISING}.get_local_torch_device", return_value=torch.device("cpu")
    ):
        WanAnimate2DenoisingStage._denoise_single_clip(
            stage,
            batch=_batch_without_perf_recording(),
            server_args=server_args,
            # guidance_scale 3.0: two forwards (cond, uncond) per step.
            request_state=_single_clip_request_state(guidance_scale=3.0),
            clip_condition=_clip_conditioning(torch.zeros(16, 1, 2, 2)),
            clip_index=0,
        )

    assert events == [
        "begin:transformer",
        "reference_kv",
        "context",
        "forward",
        "context",
        "forward",
        "forward",
        "forward",
        "finish",
    ]


def _ensure_single_process_parallel_runtime() -> None:
    # The DiT's parallel linear layers need the TP group even on a meta device.
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_dit_constructs_only_its_own_blocks():
    """Every block the DiT constructs is the in-context block; the parent's block set is not
    built and then replaced (40 block objects per construction, allocated on whatever device
    the loader initializes on)."""
    _ensure_single_process_parallel_runtime()
    config = WanAnimate2Config()
    config.arch_config.num_layers = 2
    constructed: list[type] = []
    original_init = WanTransformerBlock.__init__

    def recording_init(self, *args, **kwargs):
        constructed.append(type(self))
        original_init(self, *args, **kwargs)

    with (
        patch.object(WanTransformerBlock, "__init__", recording_init),
        torch.device("meta"),
    ):
        model = WanAnimate2Transformer3DModel(config, hf_config={})

    assert [type(block) for block in model.blocks] == [WanAnimate2TransformerBlock] * 2
    assert constructed == [WanAnimate2TransformerBlock] * 2
