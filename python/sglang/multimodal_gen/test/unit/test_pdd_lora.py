# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from sglang.multimodal_gen.runtime.pipelines_core.lora.format_adapter import (
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
    LoRAFormat,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pdd_lora import (
    apply_pdd_head_bank,
    arm_pdd_step,
    extract_pdd_payload,
    fuse_pdd_head,
    pdd_linear,
    pdd_sampling_plan,
    pdd_time_grid,
    project_pdd_or_base,
    shifted_sigma,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)


def _official_state(
    *,
    n: int = 4,
    hidden: int = 8,
    video_out: int = 6,
    audio_out: int = 3,
    rank: int = 2,
) -> dict[str, torch.Tensor]:
    state = {
        "proj_out.weight": torch.randn(n, video_out, hidden),
        "proj_out.bias": torch.randn(n, video_out),
        "audio_proj_out.weight": torch.randn(n, audio_out, hidden),
        "audio_proj_out.bias": torch.randn(n, audio_out),
        "transformer_blocks.0.attn.to_q.lora_down": torch.randn(rank, hidden),
        "transformer_blocks.0.attn.to_q.lora_up": torch.randn(hidden, rank),
        "transformer_blocks.0.ff.net.0.proj.lora_down": torch.randn(rank, hidden),
        "transformer_blocks.0.ff.net.0.proj.lora_up": torch.randn(hidden * 2, rank),
    }
    return state


def test_extract_official_3d_bank_and_leave_backbone():
    state = _official_state()
    extracted = extract_pdd_payload(
        state,
        metadata={
            "pdd_num_steps": "4",
            "pdd_block_size": "2",
            "lora_rank": "2",
            "lora_alpha": "2.0",
        },
        video_out_features=6,
        audio_out_features=3,
    )
    assert extracted is not None
    lora_state, bank = extracted
    assert bank.num_steps == 4
    assert bank.block_size == 2
    assert bank.nfe == 2
    assert tuple(bank.video_weight.shape) == (4, 6, 8)
    assert "proj_out.weight" not in lora_state
    assert "audio_proj_out.bias" not in lora_state
    assert "transformer_blocks.0.attn.to_q.lora_down" in lora_state
    assert detect_lora_format_from_state_dict(lora_state) == LoRAFormat.NON_DIFFUSERS_SD
    normalized = normalize_lora_state_dict(lora_state)
    assert "transformer_blocks.0.attn.to_q.lora_A" in normalized
    assert "transformer_blocks.0.attn.to_q.lora_B" in normalized


def test_extract_comfy_stacked_set_weight():
    n, hidden, video_out, audio_out = 4, 8, 6, 3
    state = {
        "proj_out.set_weight": torch.randn(n * video_out, hidden),
        "proj_out.set_bias": torch.randn(n * video_out),
        "audio_proj_out.set_weight": torch.randn(n * audio_out, hidden),
        "audio_proj_out.set_bias": torch.randn(n * audio_out),
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, hidden),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(hidden, 2),
    }
    extracted = extract_pdd_payload(
        state, video_out_features=video_out, audio_out_features=audio_out
    )
    assert extracted is not None
    lora_state, bank = extracted
    assert tuple(bank.video_weight.shape) == (n, video_out, hidden)
    assert tuple(bank.audio_bias.shape) == (n, audio_out)
    assert "proj_out.set_weight" not in lora_state
    assert "transformer_blocks.0.attn.to_q.lora_A.weight" in lora_state


def test_plain_lora_is_not_pdd():
    state = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(4, 8),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(8, 4),
        "proj_out.lora_A.weight": torch.randn(4, 8),
        "proj_out.lora_B.weight": torch.randn(6, 4),
    }
    assert extract_pdd_payload(state, video_out_features=6, audio_out_features=3) is None


def test_sampling_plan_matches_official_mean_velocity():
    dts = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    plan = pdd_sampling_plan(dts, start=0, block_size=2)
    torch.testing.assert_close(
        plan, torch.tensor([1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0], dtype=torch.float64)
    )
    plan = pdd_sampling_plan(dts, start=2, block_size=2)
    torch.testing.assert_close(
        plan, torch.tensor([0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0], dtype=torch.float64)
    )


def test_fuse_head_equals_weighted_per_head_linear():
    n, out, inn = 4, 5, 7
    bank_w = torch.randn(n, out, inn)
    bank_b = torch.randn(n, out)
    hidden = torch.randn(3, inn)
    plan = torch.tensor([0.1, 0.2, 0.3, 0.4])
    fused_w, fused_b = fuse_pdd_head(bank_w, bank_b, plan)
    actual = torch.nn.functional.linear(hidden, fused_w, fused_b)
    expected = torch.zeros(3, out)
    for i, weight in enumerate(plan.tolist()):
        expected = expected + weight * torch.nn.functional.linear(
            hidden, bank_w[i], bank_b[i]
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_pdd_linear_tp_shard_matches_full_then_slice():
    n, out, inn = 2, 8, 4
    bank_w = torch.randn(n, out, inn)
    bank_b = torch.randn(n, out)
    hidden = torch.randn(5, inn)
    plan = torch.tensor([0.25, 0.75])
    full = pdd_linear(hidden, bank_w, bank_b, plan, tp_size=1, tp_rank=0)
    shard0 = pdd_linear(hidden, bank_w, bank_b, plan, tp_size=2, tp_rank=0)
    shard1 = pdd_linear(hidden, bank_w, bank_b, plan, tp_size=2, tp_rank=1)
    torch.testing.assert_close(torch.cat([shard0, shard1], dim=-1), full)


def test_h3_nine_sigma_points_land_on_32_grid_blocks():
    """Official 8 NFE / 9 sigma points share every 4th 32-grid boundary."""
    n = 32
    video_sigmas = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=12.0)
    assert len(video_sigmas) == 9
    assert len(video_sigmas) - 1 == 8
    base = torch.linspace(1.0, 0.0, 9)
    grid = torch.linspace(1.0, 0.0, n + 1)
    for step, sigma in enumerate(video_sigmas[:-1]):
        start = step * 4
        torch.testing.assert_close(
            shifted_sigma(12.0, base[step]),
            shifted_sigma(12.0, grid[start]),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            torch.tensor(sigma),
            shifted_sigma(12.0, grid[start]).float(),
            rtol=1e-5,
            atol=1e-5,
        )
        plan = pdd_sampling_plan(pdd_time_grid(12.0, n).diff(), start, 4)
        assert int((plan > 0).sum().item()) == 4
        torch.testing.assert_close(plan[start : start + 4].sum(), torch.tensor(1.0, dtype=plan.dtype))


class _Head(nn.Module):
    def __init__(self, inn: int, out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out, inn))
        self.bias = nn.Parameter(torch.zeros(out))
        self.tp_size = 1
        self.tp_rank = 0

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight, self.bias), None


class _Final(nn.Module):
    def __init__(self, hidden: int, video_out: int, audio_out: int):
        super().__init__()
        self.video_out = _Head(hidden, video_out)
        self.audio_out = _Head(hidden, audio_out)
        self._pdd_nfe = None
        self._pdd_block_size = None
        self._pdd_video_plan = None
        self._pdd_audio_plan = None


def test_apply_and_arm_selects_the_trained_block():
    hidden, video_out, audio_out, n = 8, 6, 3, 4
    layer = _Final(hidden, video_out, audio_out)
    state = _official_state(n=n, hidden=hidden, video_out=video_out, audio_out=audio_out)
    _, bank = extract_pdd_payload(
        state,
        metadata={"pdd_block_size": "2"},
        video_out_features=video_out,
        audio_out_features=audio_out,
    )
    apply_pdd_head_bank(layer, bank, video_shift=12.0, audio_shift=3.0)
    assert layer._pdd_nfe == 2
    hidden_x = torch.randn(2, hidden)
    arm_pdd_step(layer, 0)
    out0 = project_pdd_or_base(layer, hidden_x)
    plan0 = layer._pdd_video_plan.clone()
    arm_pdd_step(layer, 1)
    out1 = project_pdd_or_base(layer, hidden_x)
    assert out0 is not None and out1 is not None
    assert not torch.allclose(out0[0], out1[0])
    torch.testing.assert_close(plan0[:2].sum(), torch.tensor(1.0))
    torch.testing.assert_close(plan0[2:].abs().sum(), torch.tensor(0.0))


def test_denoise_loop_calls_arm_pdd_step():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MINIMAX_H3_AUDIO_ROW_WIDTH,
        MINIMAX_H3_VIDEO_ROW_WIDTH,
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=False,
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    armed: list[int] = []

    class _Model:
        arm_pdd_step = None

        def prepare_adaln_plans(self, _timesteps):
            return None

        def _arm(self, step: int) -> None:
            armed.append(step)

    model = _Model()
    model.arm_pdd_step = model._arm

    def _forward(_model, _fk, _step):
        return (
            torch.zeros(int(branch.img_pos.shape[0]), MINIMAX_H3_VIDEO_ROW_WIDTH),
            torch.zeros(int(branch.audio_pos.shape[0]), MINIMAX_H3_AUDIO_ROW_WIDTH),
        )

    minimax_h3_denoise_loop(
        model=model,
        model_forward=_forward,
        positive=branch,
        initial_video_rows=torch.zeros(
            int(branch.img_pos.shape[0]), MINIMAX_H3_VIDEO_ROW_WIDTH
        ),
        initial_audio_rows=torch.zeros(
            int(branch.audio_pos.shape[0]), MINIMAX_H3_AUDIO_ROW_WIDTH
        ),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
    )
    assert armed == [0, 1]


def test_accepts_mxfp8_input_unwraps_lora_wrapper():
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _accepts_mxfp8_input,
    )

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = None

    class _Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.base_layer = inner

    assert _accepts_mxfp8_input(_Wrapper(_Inner())) is False
