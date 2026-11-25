"""The srt-backed Qwen3-VL text model against Hugging Face's.

``test_two_ranks`` torchruns this file for the tensor-parallel cases.
"""

import os
import subprocess
import sys

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel as HFQwen3VLTextModel,
)

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tp_group,
    maybe_init_distributed_environment_and_model_parallel,
    use_tensor_parallel_group,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_text import (
    Qwen3VLTextModel,
)

_TWO_RANKS = int(os.environ.get("WORLD_SIZE", "1")) == 2
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
single_rank = pytest.mark.skipif(_TWO_RANKS, reason="single-rank case")
two_ranks = pytest.mark.skipif(not _TWO_RANKS, reason="launched by test_two_ranks")


class _PerProjection(QuantizationConfig):
    """Every projection on its own unquantized diffusion layer."""

    def __init__(self, row_partition=True, embedding=False):
        super().__init__()
        self.row_partition = row_partition
        self.embedding = embedding

    def get_name(self):
        return "per_projection_test"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None

    def supports_input_partition(self, prefix, input_size_per_partition):
        return self.row_partition

    def quantizes_embedding(self, prefix):
        return self.embedding


def _text_config(num_layers=3):
    config = Qwen3VLTextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [4, 6, 6],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def _relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm()).item()


def _row_fp8(weight):
    """Per-output-row FP8 E4M3, as the weight-only FP8 checkpoints store it."""
    scale = weight.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0
    return (weight.float() / scale[:, None]).to(torch.float8_e4m3fn), scale


def _models(config, deepstack_layers, dtype=torch.bfloat16, **kwargs):
    """Hugging Face's text model, and the srt-backed model built in the current
    TP group and loaded from the same checkpoint names. With weight-only FP8,
    the reference holds the dequantized weights."""
    torch.manual_seed(0)
    ref = HFQwen3VLTextModel(config).cuda().to(dtype).eval()
    for param in ref.parameters():
        param.data.normal_(0, 0.05)
    with set_default_torch_dtype(dtype):
        model = Qwen3VLTextModel(
            config, deepstack_visual_indexes=range(deepstack_layers), **kwargs
        )
    model.cuda()
    fp8 = kwargs.get("weight_only_fp8", False)
    for name, tensor in list(ref.state_dict().items()):
        if fp8 and name.endswith("proj.weight"):
            quantized, scale = _row_fp8(tensor)
            dequantized = (quantized.float() * scale[:, None]).to(dtype)
            ref.state_dict()[name].copy_(dequantized)
            model.load_weight(name, quantized)
            model.load_weight(name.replace(".weight", ".weight_scale"), scale)
        else:
            model.load_weight(name, tensor)
    return ref, model


def _inputs(config, batch=2, seq_len=37, deepstack_layers=2, dtype=torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(1)
    embeds = torch.randn(
        batch, seq_len, config.hidden_size, generator=gen, device="cuda"
    ).to(dtype)
    # text, a 2x3 image grid, text: M-RoPE positions as Qwen3-VL lays them out
    text = torch.arange(10).expand(3, -1)
    t = torch.zeros(6)
    h, w = torch.arange(2).repeat_interleave(3), torch.arange(3).repeat(2)
    image = torch.stack([t, h, w]) + 10
    tail = (torch.arange(seq_len - 16) + 13).expand(3, -1)
    row = torch.cat([text, image, tail], dim=1).long()
    position_ids = row[:, None].expand(3, batch, seq_len).cuda()
    mask = torch.zeros(batch, seq_len, dtype=torch.bool, device="cuda")
    mask[:, 10:16] = True
    deepstack = [
        torch.randn(
            int(mask.sum()), config.hidden_size, generator=gen, device="cuda"
        ).to(dtype)
        for _ in range(deepstack_layers)
    ]
    return embeds, position_ids, mask, deepstack


def _reference(ref, embeds, position_ids, mask, deepstack):
    """Hugging Face's last hidden state, and each layer's output as the next
    layer reads it (deepstack features included); the last one before the
    final norm."""
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(
            lambda _m, _a, out: layer_outputs.append(
                (out[0] if isinstance(out, tuple) else out).clone()
            )
        )
        for layer in ref.layers
    ]
    try:
        last = ref(
            inputs_embeds=embeds.clone(),
            position_ids=position_ids,
            visual_pos_masks=mask if deepstack else None,
            deepstack_visual_embeds=deepstack or None,
            use_cache=False,
        ).last_hidden_state
    finally:
        for hook in hooks:
            hook.remove()
    for i, embeds_i in enumerate(deepstack):
        layer_outputs[i][mask] += embeds_i
    return last, layer_outputs


def _check(output_hidden_states=True, tolerance=1e-2, dtype=torch.bfloat16, **kwargs):
    config = _text_config()
    ref, model = _models(config, deepstack_layers=2, dtype=dtype, **kwargs)
    embeds, position_ids, mask, deepstack = _inputs(config, dtype=dtype)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        expected, expected_layers = _reference(
            ref, embeds, position_ids, mask, deepstack
        )
        actual = model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            visual_pos_masks=mask,
            deepstack_visual_embeds=deepstack,
            output_hidden_states=output_hidden_states,
        )
    assert _relative_error(actual.last_hidden_state, expected) < tolerance
    if output_hidden_states:
        assert len(actual.hidden_states) == len(expected_layers)
        for got, want in zip(actual.hidden_states, expected_layers):
            assert _relative_error(got, want) < tolerance
    else:
        assert actual.hidden_states is None
    return model


# Each case: kwargs for the srt-backed model.
_PROJECTION_CASES = {
    "fused": {},
    "per_projection": {"quant_config": _PerProjection()},
    "row_gathered": {"quant_config": _PerProjection(row_partition=False)},
    "replicated": {"quant_config": _PerProjection(), "replicate_projections": True},
    "weight_only_fp8": {"weight_only_fp8": True},
}


@single_rank
@pytest.mark.usefixtures("srt_runtime")
@pytest.mark.parametrize("output_hidden_states", [False, True])
def test_srt_text_model_matches_reference(output_hidden_states):
    _check(output_hidden_states)


@single_rank
@pytest.mark.usefixtures("srt_runtime")
@pytest.mark.parametrize("case", sorted(_PROJECTION_CASES))
def test_projection_layouts_match_reference(case):
    _check(**_PROJECTION_CASES[case])


@single_rank
@pytest.mark.usefixtures("srt_runtime")
def test_fp32_matches_reference():
    _check(tolerance=1e-4, dtype=torch.float32)


@single_rank
@pytest.mark.usefixtures("srt_runtime")
def test_srt_text_model_without_final_norm_returns_the_residual_stream():
    config = _text_config()
    ref, model = _models(config, deepstack_layers=0)
    ref.norm = torch.nn.Identity()
    model.drop_final_norm()
    embeds, position_ids, _, _ = _inputs(config, deepstack_layers=0)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        expected = ref(
            inputs_embeds=embeds.clone(), position_ids=position_ids, use_cache=False
        ).last_hidden_state
        actual = model(inputs_embeds=embeds, position_ids=position_ids)
    assert _relative_error(actual.last_hidden_state, expected) < 1e-2


@single_rank
@pytest.mark.usefixtures("srt_runtime")
def test_quantized_embedding_keeps_the_checkpoint_format():
    with set_default_torch_dtype(torch.bfloat16):
        model = Qwen3VLTextModel(
            _text_config(1),
            quant_config=_PerProjection(embedding=True),
            prefix="model.language_model",
        )
    assert isinstance(model.embed_tokens, VocabParallelEmbedding)
    assert tuple(model.embed_tokens.weight.shape) == (512, 128)


@single_rank
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_ranks():
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    cmd += ["--nproc-per-node=2", "-m", "pytest", "-q", __file__]
    subprocess.run(cmd, check=True, timeout=1200)


@pytest.fixture(scope="module")
def tensor_parallel():
    from sglang.srt.runtime_context import get_context, publish
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    maybe_init_distributed_environment_and_model_parallel(tp_size=2, sp_size=1)
    if get_context()._server_args is None:
        publish(
            SrtServerArgs(model_path="dummy", tp_size=2),
            role="diffusion_gpu_worker",
        )
    with use_tensor_parallel_group(get_tp_group()):
        yield


@two_ranks
@pytest.mark.usefixtures("tensor_parallel")
@pytest.mark.parametrize("output_hidden_states", [False, True])
def test_tensor_parallel_srt_text_model_matches_reference(output_hidden_states):
    _check(output_hidden_states)


@two_ranks
@pytest.mark.usefixtures("tensor_parallel")
@pytest.mark.parametrize("case", sorted(_PROJECTION_CASES))
def test_tensor_parallel_projection_layouts_match_reference(case):
    _check(**_PROJECTION_CASES[case])
