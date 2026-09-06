"""Unit tests for Qwen-Image joint text-image Q/K/V destination buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import join_seqs
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    apply_unquantized_linear,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageCrossAttention,
    _joint_qkv_head_views,
    _project_qkv_into_joint_buffers,
    _use_joint_qkv_buffers,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

DIM = 32
NUM_HEADS = 2
HEAD_DIM = 8
INNER_DIM = NUM_HEADS * HEAD_DIM
SEQ_TXT = 5
SEQ_IMG = 13
QWEN_IMAGE = "sglang.multimodal_gen.runtime.models.dits.qwen_image"
SP_WORLD_SIZE = f"{QWEN_IMAGE}.get_sp_world_size"
PROJECT_INTO_JOINT_BUFFERS = f"{QWEN_IMAGE}._project_qkv_into_joint_buffers"
FUSED_INPLACE_QKNORM = (
    "sglang.multimodal_gen.runtime.layers.layernorm.can_use_fused_inplace_qknorm"
)
# ColumnParallelLinear reads only the group's size and rank.
_TP_GROUP = SimpleNamespace(world_size=1, rank_in_group=0)


def _linear(*, bias: bool = True) -> ColumnParallelLinear:
    linear = ColumnParallelLinear(
        DIM,
        INNER_DIM,
        bias=bias,
        gather_output=False,
        params_dtype=torch.bfloat16,
        tp_group=_TP_GROUP,
    )
    with torch.no_grad():
        linear.weight.normal_()
        if bias:
            linear.bias.normal_()
    return linear.cuda()


def _passthrough(x: torch.Tensor) -> tuple[torch.Tensor, None]:
    return x, None


class _RecordingAttention:
    # Stands in for USPAttention: keeps the Q/K/V it was handed and returns a
    # deterministic function of them in the [B, S, H, D] layout.
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]] = []

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        self.calls.append((q.clone(), k.clone(), v.clone(), kwargs))
        if kwargs["q_prefix"] is not None:
            q = torch.cat([kwargs["q_prefix"], q], dim=1)
            k = torch.cat([kwargs["k_prefix"], k], dim=1)
            v = torch.cat([kwargs["v_prefix"], v], dim=1)
        return q + k + v


def _attention(*, bias: bool = True) -> QwenImageCrossAttention:
    attn = object.__new__(QwenImageCrossAttention)
    nn.Module.__init__(attn)
    attn.head_dim = HEAD_DIM
    attn.local_num_heads = NUM_HEADS
    attn.added_kv_proj_dim = DIM
    attn.qk_norm = True
    attn.use_fused_qkv = False
    attn.use_fused_qkv_epilogue = False
    attn.use_fused_added_qkv = False
    attn._unquantized_added_qkv_is_packed = False
    attn.separate_unquantized_qkv_proj = True
    attn.separate_convrot_qkv_proj = False
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        setattr(attn, name, _linear(bias=bias))
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = RMSNorm(HEAD_DIM, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            norm.weight.normal_()
        setattr(attn, name, norm)
    attn.to_out = [_passthrough]
    attn.to_add_out = _passthrough
    attn.attn = _RecordingAttention()
    return attn


def _streams() -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.randn(1, SEQ_IMG, DIM, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(1, SEQ_TXT, DIM, device="cuda", dtype=torch.bfloat16)
    return hidden, encoder


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQwenImageJointQkvBuffers(CustomTestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260902)
        # The runtime forwards under no_grad; the out= GEMMs of the joint path
        # reject grad-tracking operands.
        grad_was_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self.addCleanup(torch.set_grad_enabled, grad_was_enabled)

    def _forward(self, attn: QwenImageCrossAttention, **kwargs):
        hidden, encoder = _streams()
        with patch(SP_WORLD_SIZE, return_value=1):
            return attn.forward(
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                image_rotary_emb=None,
                **kwargs,
            )

    def _assert_joint_path_matches_join_seqs_path(
        self, attn: QwenImageCrossAttention
    ) -> None:
        outputs = []
        for joint in (True, False):
            torch.manual_seed(20260902)
            attn.separate_unquantized_qkv_proj = joint
            attn.attn.calls.clear()
            outputs.append((self._forward(attn), attn.attn.calls[0]))
        (joint_out, joint_call), (ref_out, ref_call) = outputs

        self.assertIsNone(joint_call[3]["q_prefix"])
        for joint_qkv, ref_qkv in zip(joint_call[:3], ref_call[:3]):
            self.assertEqual(
                joint_qkv.shape, (1, SEQ_TXT + SEQ_IMG, NUM_HEADS, HEAD_DIM)
            )
            self.assertTrue(torch.equal(joint_qkv, ref_qkv))
        self.assertTrue(torch.equal(joint_out[0], ref_out[0]))
        self.assertTrue(torch.equal(joint_out[1], ref_out[1]))

    def test_forward_is_bitwise_identical_to_join_seqs_path(self):
        self._assert_joint_path_matches_join_seqs_path(_attention())

    def test_forward_copies_back_when_qk_norm_returns_fresh_tensors(self):
        """The non-fused QK-norm returns new tensors instead of writing the
        buffer views in place; their values must still reach the joint buffers."""
        with patch(FUSED_INPLACE_QKNORM, return_value=False):
            self._assert_joint_path_matches_join_seqs_path(_attention())

    def test_masked_forward_bypasses_joint_buffers(self):
        """A masked sequence hands text Q/K/V to attention as a prefix, so the
        joint buffers must not be built for it."""
        attn = _attention()
        mask = torch.ones(1, SEQ_TXT, dtype=torch.bool, device="cuda")
        with patch(PROJECT_INTO_JOINT_BUFFERS) as project:
            self._forward(attn, encoder_hidden_states_mask=mask)
        project.assert_not_called()
        self.assertIsNotNone(attn.attn.calls[0][3]["q_prefix"])

    def test_stale_convrot_flag_falls_through_to_per_layer_projections(self):
        """LoRA mounting swaps the projections after construction, so the
        ConvRot flag alone must not select the shared-input helper."""
        attn = _attention()
        attn.separate_unquantized_qkv_proj = False
        mask = torch.ones(1, SEQ_TXT, dtype=torch.bool, device="cuda")
        outputs = []
        for convrot in (False, True):
            torch.manual_seed(20260902)
            attn.separate_convrot_qkv_proj = convrot
            outputs.append(self._forward(attn, encoder_hidden_states_mask=mask))
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))

    def test_joint_projection_without_bias_matches_reference(self):
        attn = _attention(bias=False)
        hidden, encoder = _streams()
        bufs = _project_qkv_into_joint_buffers(
            attn=attn,
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            seq_len_txt=SEQ_TXT,
        )
        img_layers = (attn.to_q, attn.to_k, attn.to_v)
        txt_layers = (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
        for buf, img_layer, txt_layer in zip(bufs, img_layers, txt_layers):
            expected = join_seqs(
                apply_unquantized_linear(encoder, txt_layer.weight, None),
                apply_unquantized_linear(hidden, img_layer.weight, None),
                0,
            )
            self.assertTrue(torch.equal(buf, expected))

    def test_head_views_alias_the_joint_buffers(self):
        hidden, encoder = _streams()
        bufs = _project_qkv_into_joint_buffers(
            attn=_attention(),
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            seq_len_txt=SEQ_TXT,
        )
        views = _joint_qkv_head_views(
            bufs, seq_len_txt=SEQ_TXT, num_heads=NUM_HEADS, head_dim=HEAD_DIM
        )
        img_query, txt_query = views[0], views[3]

        self.assertEqual(img_query.shape, (1, SEQ_IMG, NUM_HEADS, HEAD_DIM))
        self.assertEqual(txt_query.shape, (1, SEQ_TXT, NUM_HEADS, HEAD_DIM))
        self.assertTrue(img_query.is_contiguous())
        self.assertTrue(txt_query.is_contiguous())
        # In-place QK-norm/RoPE on the views must land in the joint buffer.
        img_query.zero_()
        txt_query.fill_(1.0)
        self.assertTrue(torch.all(bufs[0][:, SEQ_TXT:] == 0))
        self.assertTrue(torch.all(bufs[0][:, :SEQ_TXT] == 1))

    def test_eligibility_rejects_every_unsupported_layout(self):
        attn = _attention()
        hidden, encoder = _streams()

        def eligible(h=hidden, e=encoder, *, masked=False, sp=1, attention=attn):
            with patch(SP_WORLD_SIZE, return_value=sp):
                return _use_joint_qkv_buffers(
                    attn=attention,
                    hidden_states=h,
                    encoder_hidden_states=e,
                    masked=masked,
                )

        self.assertTrue(eligible())
        self.assertFalse(eligible(masked=True))
        self.assertFalse(eligible(sp=2))
        self.assertFalse(eligible(h=hidden.expand(2, -1, -1).contiguous()))
        strided_hidden = hidden.transpose(1, 2).contiguous().transpose(1, 2)
        strided_encoder = encoder.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(eligible(h=strided_hidden))
        self.assertFalse(eligible(e=strided_encoder))
        packed = _attention()
        packed.separate_unquantized_qkv_proj = False
        self.assertFalse(eligible(attention=packed))
        # An unmerged LoRA adds its delta in forward; the bare addmm would drop it.
        lora_wrapped = _attention()
        lora_wrapped.to_q = BaseLayerWithLoRA(lora_wrapped.to_q)
        self.assertFalse(eligible(attention=lora_wrapped))


if __name__ == "__main__":
    unittest.main()
