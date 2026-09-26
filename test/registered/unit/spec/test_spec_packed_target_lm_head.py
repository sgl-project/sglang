"""A target lm_head that stores its weight packed (compressed-tensors
``weight_packed``, GGUF ``qweight``) has no ``.weight``. The paths that borrow
the target head for speculative decoding used to read that tensor, or to refuse
the head outright, while the target's own logits already came out of
``lm_head.quant_method.apply``. These cases pin the shared contract: such a
head is admitted through its quant method, and a NEXTN draft gets it as a
module. The DFlash selector's side lives in test_dflash_logits.py."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    should_apply_lm_head_quant_method,
)
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_text import Qwen3_5ForCausalLM
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _PackedHeadMethod:
    """Stands in for CompressedTensorsLinearMethod on a WNA16 head: logits come
    out of apply() over a captured dense weight, and the layer it is handed must
    be the packed one (no ``.weight``, no bias)."""

    def __init__(self, dense_weight):
        self.dense_weight = dense_weight
        self.seen_dtypes = []

    def apply(self, layer, x, bias):
        assert not hasattr(layer, "weight")
        assert bias is None
        self.seen_dtypes.append(x.dtype)
        return torch.matmul(x.to(self.dense_weight.dtype), self.dense_weight.T)


def _packed_head(dense_weight):
    """A ParallelLMHead-shaped module whose only weight tensor is packed."""
    head = nn.Module()
    head.weight_packed = nn.Parameter(
        torch.empty(dense_weight.shape[0], 1, dtype=torch.int32),
        requires_grad=False,
    )
    head.quant_method = _PackedHeadMethod(dense_weight)
    head.org_vocab_size = dense_weight.shape[0]
    return head


def _dense_head(weight):
    head = nn.Module()
    head.weight = nn.Parameter(weight, requires_grad=False)
    return head


@pytest.mark.parametrize(
    "method_name",
    [
        "ModelOptFp4LinearMethod",
        "ModelOptNvFp4A16LinearMethod",
        "ModelOptFp8LinearMethod",
    ],
)
def test_gate_refuses_a_modelopt_method_on_a_head_without_dense_weight(method_name):
    """Each ModelOpt branch of the gate reads ``lm_head.weight.dtype``; a stale
    ModelOpt method on a head without that tensor must be refused, not crash.
    Dropping any one name from the gate's ModelOpt set turns its case red."""
    method_cls = type(
        method_name,
        (),
        {"apply": lambda self, layer, x, bias: None, "quant_mode": "w4a4"},
    )
    head = nn.Module()
    head.weight_scale = nn.Parameter(torch.empty(1))
    assert not should_apply_lm_head_quant_method(head, method_cls())


def test_logits_path_keeps_the_fp32_cast_gguf_heads_had():
    """Heads without a dense weight used to get their logits from the trailing
    fallback branch, which casts activations to fp32 under
    --enable-fp32-lm-head (the GGUF kernels take fp32 activations). Admitting
    them through the gate must not drop that cast, nor add one without the
    flag. This is also the case that forces the gate to admit such a head: a
    refusal now ends in the error branch instead."""
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    head = _packed_head(torch.randn(6, 4, dtype=torch.bfloat16))

    for use_fp32, expected in ((False, torch.bfloat16), (True, torch.float32)):
        processor = SimpleNamespace(use_fp32_lm_head=use_fp32, rl_on_policy_target=None)
        logits = LogitsProcessor._compute_lm_head(processor, hidden, head)
        assert logits.shape == (2, 6)
        assert head.quant_method.seen_dtypes[-1] == expected


def test_logits_path_refuses_a_head_it_cannot_project():
    head = nn.Module()  # neither a dense weight nor a quant method
    head.quant_method = None
    processor = SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
    with pytest.raises(ValueError, match="neither a dense weight"):
        LogitsProcessor._compute_lm_head(processor, torch.randn(1, 4), head)


_TARGET_CLASSES = {
    "qwen3_5.Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGeneration,
    "qwen3_5.Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGeneration,
    "qwen3_5_text.Qwen3_5ForCausalLM": Qwen3_5ForCausalLM,
}


def _target(model_cls, lm_head, embed=None):
    """A target model stub bound to the real accessor of ``model_cls``."""
    embed = torch.randn(4, 2) if embed is None else embed
    target = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=embed)),
        lm_head=lm_head,
    )
    target.get_embed_and_head = lambda: model_cls.get_embed_and_head(target)
    return target


@pytest.mark.parametrize(
    "model_cls", list(_TARGET_CLASSES.values()), ids=list(_TARGET_CLASSES)
)
def test_qwen3_5_hands_out_no_head_tensor_for_a_packed_lm_head(model_cls):
    """get_embed_and_head read ``self.lm_head.weight`` unconditionally, which
    is where NEXTN on a packed-head checkpoint died. A packed head now yields
    None (the module is shared instead); a dense head is handed out as before."""
    target = _target(model_cls, _packed_head(torch.randn(4, 2)))
    embed, head = target.get_embed_and_head()
    assert embed is target.model.embed_tokens.weight
    assert head is None

    target.lm_head = _dense_head(torch.randn(4, 2))
    assert target.get_embed_and_head()[1] is target.lm_head.weight


def _draft_worker(target, draft_model):
    worker = object.__new__(EagleDraftWorker)
    worker.hot_token_id = None
    worker.speculative_algorithm = SimpleNamespace(is_eagle3=lambda: False)
    worker.target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=target))
    worker.draft_runner = SimpleNamespace(model=draft_model)
    return worker


def test_init_lm_head_shares_a_packed_target_head_as_a_module():
    """The NEXTN path end to end: the real accessor yields no tensor for the
    packed head, and the worker hands the draft the whole module."""
    shared = {}
    draft = SimpleNamespace(
        set_embed_and_head=lambda embed, head: shared.update(head=head),
        set_lm_head_from_target=lambda module: shared.update(module=module),
    )
    target = _target(Qwen3_5ForConditionalGeneration, _packed_head(torch.randn(4, 2)))

    with envs.SGLANG_ENABLE_PP_SPEC.override(False):
        _draft_worker(target, draft).init_lm_head()

    assert shared["head"] is None
    assert shared["module"] is target.lm_head


def test_init_lm_head_refuses_a_packed_target_head_the_draft_cannot_take():
    """Without module sharing the draft would silently decode with whatever
    head it loaded on its own."""
    draft = SimpleNamespace(set_embed_and_head=lambda embed, head: None)
    target = _target(Qwen3_5ForConditionalGeneration, _packed_head(torch.randn(4, 2)))

    with envs.SGLANG_ENABLE_PP_SPEC.override(False):
        with pytest.raises(ValueError, match="packed.*set_lm_head_from_target"):
            _draft_worker(target, draft).init_lm_head()


def test_init_lm_head_keeps_accepting_a_pipeline_stage_without_the_head():
    """A non-last pipeline stage carries a PPMissingLayer: no weight, but not
    packed. It yields head=None as before and must not be mistaken for a
    packed head."""
    draft = SimpleNamespace(set_embed_and_head=lambda embed, head: None)
    target = _target(Qwen3_5ForCausalLM, PPMissingLayer())

    with envs.SGLANG_ENABLE_PP_SPEC.override(False):
        _draft_worker(target, draft).init_lm_head()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
