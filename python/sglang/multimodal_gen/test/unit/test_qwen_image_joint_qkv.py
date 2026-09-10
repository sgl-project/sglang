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
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
    ConvRotInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_sgl_kernel import (
    sgl_kernel_convrot_available,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageCrossAttention,
    _joint_qkv_head_views,
    _project_qkv_into_joint_buffers,
    _use_joint_qkv_buffers,
)
from sglang.test.test_utils import CustomTestCase

DIM = 32
# ConvRot rotates 256-wide input groups.
CONVROT_DIM = 256
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


requires_convrot_kernel = unittest.skipUnless(
    sgl_kernel_convrot_available(),
    "needs a GPU in sgl-kernel's convrot table and a build with the convrot ops",
)


def _linear(
    *,
    bias: bool = True,
    dim: int = DIM,
    dtype: torch.dtype = torch.bfloat16,
    quant_config: ConvRotInt8Config | None = None,
    prefix: str = "",
) -> ColumnParallelLinear:
    linear = ColumnParallelLinear(
        dim,
        INNER_DIM,
        bias=bias,
        gather_output=False,
        params_dtype=dtype,
        quant_config=quant_config,
        prefix=prefix,
        tp_group=_TP_GROUP,
    )
    with torch.no_grad():
        linear.weight.normal_()
        if bias:
            linear.bias.normal_()
    linear = linear.cuda()
    if quant_config is not None:
        linear.quant_method.process_weights_after_loading(linear)
    return linear


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


def _attention(
    *,
    bias: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    quant_config: ConvRotInt8Config | None = None,
) -> QwenImageCrossAttention:
    dim = DIM if quant_config is None else CONVROT_DIM
    attn = object.__new__(QwenImageCrossAttention)
    nn.Module.__init__(attn)
    attn.head_dim = HEAD_DIM
    attn.local_num_heads = NUM_HEADS
    attn.added_kv_proj_dim = dim
    attn.qk_norm = True
    attn.use_fused_qkv = False
    attn.use_fused_qkv_epilogue = False
    attn.use_fused_added_qkv = False
    attn._unquantized_added_qkv_is_packed = False
    attn.separate_unquantized_qkv_proj = quant_config is None
    attn.separate_convrot_qkv_proj = quant_config is not None
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        setattr(
            attn,
            name,
            _linear(
                bias=bias,
                dim=dim,
                dtype=dtype,
                quant_config=quant_config,
                prefix=f"transformer_blocks.0.attn.{name}",
            ),
        )
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = RMSNorm(HEAD_DIM, eps=1e-6).to(device="cuda", dtype=dtype)
        with torch.no_grad():
            norm.weight.normal_()
        setattr(attn, name, norm)
    attn.to_out = [_passthrough]
    attn.to_add_out = _passthrough
    attn.attn = _RecordingAttention()
    return attn


def _streams(
    *, dim: int = DIM, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.randn(1, SEQ_IMG, dim, device="cuda", dtype=dtype)
    encoder = torch.randn(1, SEQ_TXT, dim, device="cuda", dtype=dtype)
    return hidden, encoder


def _joint_flag(attn: QwenImageCrossAttention) -> str:
    return (
        "separate_convrot_qkv_proj"
        if attn.separate_convrot_qkv_proj
        else "separate_unquantized_qkv_proj"
    )


class _JointQkvCase(CustomTestCase):
    # Forward helpers shared by the unquantized and the ConvRot test classes.

    def setUp(self) -> None:
        torch.manual_seed(20260902)
        # The runtime forwards under no_grad; the out= GEMMs of the joint path
        # reject grad-tracking operands.
        grad_was_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self.addCleanup(torch.set_grad_enabled, grad_was_enabled)

    def _forward(
        self,
        attn: QwenImageCrossAttention,
        *,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        hidden, encoder = _streams(dim=attn.added_kv_proj_dim, dtype=dtype)
        with patch(SP_WORLD_SIZE, return_value=1):
            return attn.forward(
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                image_rotary_emb=None,
                **kwargs,
            )

    def _assert_joint_path_matches_join_seqs_path(
        self, attn: QwenImageCrossAttention, *, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        flag = _joint_flag(attn)
        outputs = []
        for joint in (True, False):
            torch.manual_seed(20260902)
            setattr(attn, flag, joint)
            attn.attn.calls.clear()
            with patch(
                PROJECT_INTO_JOINT_BUFFERS, wraps=_project_qkv_into_joint_buffers
            ) as project:
                out = self._forward(attn, dtype=dtype)
            self.assertEqual(project.call_count, 1 if joint else 0)
            outputs.append((out, attn.attn.calls[0]))
        (joint_out, joint_call), (ref_out, ref_call) = outputs

        self.assertIsNone(joint_call[3]["q_prefix"])
        for joint_qkv, ref_qkv in zip(joint_call[:3], ref_call[:3]):
            self.assertEqual(
                joint_qkv.shape, (1, SEQ_TXT + SEQ_IMG, NUM_HEADS, HEAD_DIM)
            )
            self.assertTrue(torch.equal(joint_qkv, ref_qkv))
        self.assertTrue(torch.equal(joint_out[0], ref_out[0]))
        self.assertTrue(torch.equal(joint_out[1], ref_out[1]))

    def _assert_forward_bypasses_joint_buffers(
        self, attn: QwenImageCrossAttention, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The out= kernels reject these operand dtypes, so the joint path must
        # not be entered; the flag-off forward is the same code the fallback runs.
        with patch(
            PROJECT_INTO_JOINT_BUFFERS, wraps=_project_qkv_into_joint_buffers
        ) as project:
            out = self._forward(attn, dtype=dtype)
        project.assert_not_called()
        return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQwenImageJointQkvBuffers(_JointQkvCase):
    def test_forward_is_bitwise_identical_to_join_seqs_path(self):
        self._assert_joint_path_matches_join_seqs_path(_attention())

    def test_forward_copies_back_when_qk_norm_returns_fresh_tensors(self):
        """The non-fused QK-norm returns new tensors instead of writing the
        buffer views in place; their values must still reach the joint buffers."""
        with patch(FUSED_INPLACE_QKNORM, return_value=False):
            self._assert_joint_path_matches_join_seqs_path(_attention())

    def test_fp16_parameters_and_streams_take_the_joint_path(self):
        """--dit-precision fp16 keeps every operand FP16, which the out= GEMMs
        accept like BF16."""
        self._assert_joint_path_matches_join_seqs_path(
            _attention(dtype=torch.float16), dtype=torch.float16
        )

    def test_fp32_streams_under_bf16_autocast_bypass_joint_buffers(self):
        """FP32 streams under BF16 autocast must take the join_seqs path; the
        out= GEMMs of the joint path do not autocast."""
        attn = _attention()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._assert_forward_bypasses_joint_buffers(attn, dtype=torch.float32)

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

    def test_eligibility_requires_one_dtype_across_streams_weights_and_autocast(
        self,
    ):
        """The out= GEMMs neither promote nor autocast, so they stand in for
        F.linear only when every operand already has the dtype it would use."""
        attn = _attention()
        hidden, encoder = _streams()

        def eligible(h=hidden, e=encoder, *, attention=attn):
            with patch(SP_WORLD_SIZE, return_value=1):
                return _use_joint_qkv_buffers(
                    attn=attention,
                    hidden_states=h,
                    encoder_hidden_states=e,
                    masked=False,
                )

        self.assertFalse(eligible(h=hidden.float(), e=encoder.float()))
        self.assertFalse(eligible(e=encoder.float()))
        fp32_bias = _attention()
        fp32_bias.to_k.bias.data = fp32_bias.to_k.bias.data.float()
        self.assertFalse(eligible(attention=fp32_bias))
        with torch.autocast("cuda", dtype=torch.float16):
            self.assertFalse(eligible())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.assertTrue(eligible())
            self.assertFalse(eligible(h=hidden.float(), e=encoder.float()))
        self.assertTrue(eligible())
        fp16 = _attention(dtype=torch.float16)
        h16, e16 = _streams(dtype=torch.float16)
        self.assertTrue(eligible(h=h16, e=e16, attention=fp16))


@requires_convrot_kernel
class TestQwenImageJointQkvBuffersConvRot(_JointQkvCase):
    """The same forward with the six projections on convrot_int8's sgl-kernel
    backend, which writes the joint buffers through its out= op."""

    def setUp(self) -> None:
        super().setUp()
        self.config = ConvRotInt8Config(backend="sgl_kernel")

    def test_forward_is_bitwise_identical_to_join_seqs_path(self):
        attn = _attention(quant_config=self.config)
        self.assertEqual(attn.to_q.weight.dtype, torch.int8)
        self._assert_joint_path_matches_join_seqs_path(attn)

    def test_fp16_streams_bypass_joint_buffers_and_return_fp16(self):
        """FP16 streams must skip the joint buffers (the out= kernel stores BF16
        only) and the forward must still return FP16."""
        attn = _attention(dtype=torch.float16, quant_config=self.config)
        self.assertEqual(attn.to_q.bias.dtype, torch.bfloat16)
        out = self._assert_forward_bypasses_joint_buffers(attn, dtype=torch.float16)
        self.assertEqual(out[0].dtype, torch.float16)
        self.assertEqual(out[1].dtype, torch.float16)

    def test_eligibility_requires_bf16_streams(self):
        attn = _attention(quant_config=self.config)
        hidden, encoder = _streams(dim=CONVROT_DIM)

        def eligible(h, e):
            with patch(SP_WORLD_SIZE, return_value=1):
                return _use_joint_qkv_buffers(
                    attn=attn, hidden_states=h, encoder_hidden_states=e, masked=False
                )

        self.assertTrue(eligible(hidden, encoder))
        self.assertFalse(eligible(hidden.half(), encoder.half()))
        self.assertFalse(eligible(hidden, encoder.half()))


if __name__ == "__main__":
    unittest.main()
