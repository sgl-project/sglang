import pytest
import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.lilicorr import LiLiCorrHead, check_head_weight_coverage
from sglang.srt.speculative.lilicorr_utils import LiLiCorrConfig
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 linears require CUDA"
)

HIDDEN, BLOCK, TOPK = 256, 4, 4
CONFIG = LiLiCorrConfig(
    candidate_topk=TOPK,
    hidden_size=128,
    num_layers=1,
    num_heads=4,
    mlp_ratio=2.0,
    factor_dim=64,
    vector_eps=1e-4,
    logit_scale=8.0,
)


def _head(quant_config, state=None):
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        head = LiLiCorrHead(
            model_hidden_size=HIDDEN,
            block_size=BLOCK,
            rms_norm_eps=1e-6,
            config=CONFIG,
            quant_config=quant_config,
            prefix="lilicorr",
        ).cuda()
    finally:
        torch.set_default_dtype(default)
    if state is None:
        torch.manual_seed(0)
        for param in head.parameters():
            param.data.normal_(0.0, 0.05)
    else:
        head.load_state_dict(state)
    for module in head.modules():
        if isinstance(module, LinearBase):
            module.quant_method.process_weights_after_loading(module)
    head.materialize_inference_buffers(torch.device("cuda"), torch.bfloat16)
    return head


def _score(head, bs):
    torch.manual_seed(1)
    slots = BLOCK - 1
    # The sampler slices the block's first row off, so bs > 1 is non-contiguous.
    hidden = torch.randn(bs, BLOCK, HIDDEN, device="cuda", dtype=torch.bfloat16)
    log_probs = torch.randn(bs, 1, slots, TOPK, device="cuda").log_softmax(-1)
    return head.score(
        token_embeddings=torch.randn(
            bs, 1, slots, TOPK, HIDDEN, device="cuda", dtype=torch.bfloat16
        ),
        candidate_log_probs=log_probs.sort(dim=-1, descending=True).values,
        pass_hidden=hidden[:, 1:, :].unsqueeze(1),
        anchor_hidden=torch.randn(bs, 1, HIDDEN, device="cuda", dtype=torch.bfloat16),
        anchor_valid=torch.ones(bs, 1, dtype=torch.bool, device="cuda"),
    )


def test_online_fp8_head_scores_a_batch():
    reference = _head(None)
    quantized = _head(Fp8Config(), state=reference.state_dict())
    for got, want in zip(_score(quantized, bs=2), _score(reference, bs=2)):
        torch.testing.assert_close(got.float(), want.float(), atol=0.1, rtol=0)


def test_static_fp8_head_requires_input_scale():
    head = LiLiCorrHead(
        model_hidden_size=HIDDEN,
        block_size=BLOCK,
        rms_norm_eps=1e-6,
        config=CONFIG,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="static"
        ),
        prefix="lilicorr",
    )
    names = {f"lilicorr.{name}" for name, _ in head.named_parameters()}
    assert any(name.endswith(".input_scale") for name in names)
    seen = {name for name in names if not name.endswith(".input_scale")}
    with pytest.raises(ValueError, match="missing"):
        check_head_weight_coverage(head, seen)
