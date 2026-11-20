"""CPU execution of DSpark's A3 draft-model dispatch and tensor contracts.

Extract the production methods without importing the accelerator runtime. The
device RoPE operator is mocked; these checks do not establish NPU graph support.
"""

import ast
import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
MODEL = ROOT / "python/sglang/srt/models/deepseek_v4_dspark.py"
MHC = ROOT / "python/sglang/srt/models/deepseek_v4_mhc.py"
MODELSLIM = ROOT / "python/sglang/srt/layers/quantization/modelslim/modelslim.py"

register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def _extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = []
    for name in names:
        body = tree.body
        for part in name.split("."):
            node = next(
                n
                for n in body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part
            )
            body = node.body
        nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class _AttentionBase(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.compress_ratio = 0
        self.n_local_heads = 2
        self.head_dim = 4
        self.softmax_scale = 0.5
        self.is_dsv41 = True


def _namespace(npu=True, a3=True):
    ns = {
        "torch": torch,
        "_is_npu": npu,
        "_is_npu_a3": a3,
        "_is_cuda": False,
        "_PAD_NUM_HEADS": 64,
        "MqaAttentionBase": _AttentionBase,
        "RadixAttention": Mock(),
        "add_prefix": lambda name, prefix: name,
        "envs": SimpleNamespace(),
        "get_platform": lambda: SimpleNamespace(is_blackwell=False),
        "get_is_capture_mode": lambda: False,
        "fused_q_norm_rope": Mock(side_effect=AssertionError("CUDA query kernel")),
        "fused_rope_inplace": Mock(),
        "apply_rotary_emb": Mock(side_effect=lambda x, *a, **kw: x),
    }
    return _extract(MODEL, ["DSparkAttention"], ns)


@pytest.mark.parametrize("npu,a3", [(True, True), (True, False), (False, False)])
def test_attention_initialization_does_not_require_removed_kernel_env(npu, a3):
    ns = _namespace(npu, a3)
    attn = ns["DSparkAttention"](SimpleNamespace(sliding_window=128), layer_id=0)
    assert not hasattr(attn, "_use_fast_kernel")


@pytest.mark.parametrize("q_head_norm", [False, True])
@pytest.mark.parametrize("with_out", [False, True])
@pytest.mark.parametrize("npu,a3", [(True, True), (True, False), (False, False)])
def test_query_dispatch_preserves_a3_a5_and_gpu_head_norm_contract(
    q_head_norm, with_out, npu, a3
):
    ns = _namespace(npu, a3)
    ns["fused_q_norm_rope"] = Mock(
        side_effect=lambda q, q_out, *args: q_out.copy_(q * 2)
    )
    attn = ns["DSparkAttention"](SimpleNamespace(sliding_window=128), layer_id=0)
    attn.wq_a = lambda x: (x, None)
    attn.q_norm = lambda x: x
    attn.wq_b = lambda x: (x, None)
    attn.q_head_norm = q_head_norm
    attn.rope_head_dim = 2
    attn.eps = 1e-6
    attn.freqs_cis = torch.ones(16, 1, dtype=torch.complex64)
    x = torch.arange(24, dtype=torch.bfloat16).view(3, 8)
    positions = torch.tensor([1, 3, 5])
    rope = Mock()
    rope.for_freqs.return_value.get_cos_sin.return_value = (
        torch.ones(3, 1, 1, 2),
        torch.zeros(3, 1, 1, 2),
    )
    rms_norm = Mock(side_effect=lambda q, *a: (q * 2,))
    out = torch.empty(3, 2, 4, dtype=x.dtype) if with_out else None
    with patch.dict(
        "sys.modules",
        {
            "torch_npu": SimpleNamespace(npu_rms_norm=rms_norm),
            "sglang.srt.hardware_backend.npu.dsv4.dsv4_rope": SimpleNamespace(
                Dsv4NpuRoPE=rope
            ),
        },
    ):
        result = attn._compute_q(x, positions, q_out=out)
    expected = x.view(3, 2, 4) * (2 if q_head_norm else 1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    native_npu_rope = npu and (a3 or q_head_norm)
    assert rms_norm.call_count == int(npu and q_head_norm)
    assert rope.apply_rotary_mul_inplace.call_count == int(native_npu_rope)
    if native_npu_rope:
        assert rope.apply_rotary_mul_inplace.call_args.kwargs["qk_nope_dim"] == 2
        assert rope.for_freqs.return_value.get_cos_sin.call_args.args[0] is positions
    assert ns["fused_q_norm_rope"].call_count == int(not npu and q_head_norm)
    assert ns["fused_rope_inplace"].call_count == int(not npu and not q_head_norm)
    assert ns["apply_rotary_emb"].call_count == int(npu and not a3 and not q_head_norm)
    if with_out:
        assert result is out


@pytest.mark.parametrize("npu,a3", [(True, True), (True, False), (False, False)])
def test_a3_draft_attention_keeps_native_heads_and_matching_sink(npu, a3):
    ns = _namespace(npu, a3)
    pool = SimpleNamespace(uniform_fp8=False)
    ns["_resolve_dspark_pool"] = lambda: pool
    backend = SimpleNamespace(
        forward=Mock(side_effect=lambda **kw: torch.ones_like(kw["q"]))
    )
    ns["_apply_wo_a_bf16_matmul"] = lambda o, w, **kw: torch.einsum(
        "bgd,grd->bgr", o, w
    )
    attn = ns["DSparkAttention"](SimpleNamespace(sliding_window=128), layer_id=0)
    attn.rope_head_dim = 2
    attn.q_head_norm = False
    attn.n_local_groups = 1
    attn.o_lora_rank = 2
    attn.freqs_cis = torch.ones(16, 1, dtype=torch.complex64)
    attn.wo_a = SimpleNamespace(weight=torch.ones(2, 8))
    attn.wo_b = lambda x, **kw: (x, None)
    attn.kv_proj_only = lambda x: x[:, :4]
    attn._store_block_kv = Mock()
    attn._local_attn_sink = Mock(return_value=torch.zeros(2 if a3 else 64))
    attn._compute_q = lambda x, p, q_out=None: (
        torch.ones(x.shape[0], 2, 4) if q_out is None else q_out.fill_(1)
    )
    rope = Mock()
    rope.for_freqs.return_value.get_cos_sin.return_value = (None, None)
    fb = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_idle=lambda: False,
            is_decode=lambda: True,
            is_target_verify=lambda: False,
        )
    )
    with patch.dict(
        "sys.modules",
        {
            "sglang.srt.model_executor.forward_context": SimpleNamespace(
                get_attn_backend=lambda: backend
            ),
            "sglang.srt.hardware_backend.npu.dsv4.dsv4_rope": SimpleNamespace(
                Dsv4NpuRoPE=rope
            ),
        },
    ):
        result = attn(torch.tensor([1, 2]), torch.ones(2, 8), fb)
    q = backend.forward.call_args.kwargs["q"]
    assert q.shape == (2, 2 if a3 else 64, 4)
    attn._local_attn_sink.assert_called_once_with(2 if a3 else None)
    assert result.shape == (2, 2)


@pytest.mark.parametrize("a3", [True, False])
@pytest.mark.parametrize("tokens", [0, 3])
def test_a3_final_premix_collapse_reuses_target_bf16_path(a3, tokens):
    ns = _namespace(npu=True, a3=a3)
    combine = _extract(MHC, ["combine_premix"], {"torch": torch})["combine_premix"]
    ns.update(
        mhc=SimpleNamespace(combine_premix=Mock(side_effect=combine)),
        get_global_expert_distribution_recorder=lambda: SimpleNamespace(
            disable_this_region=nullcontext
        ),
        LogitsProcessorOutput=lambda **kw: SimpleNamespace(**kw),
    )
    _extract(MODEL, ["DeepseekV4ForCausalLMDSpark._forward_impl"], ns)
    x = torch.arange(tokens * 32, dtype=torch.bfloat16).view(tokens, 4, 8) / 7
    pre = torch.arange(tokens * 4, dtype=torch.float32).view(tokens, 4) / 13
    model = SimpleNamespace(
        stages=[lambda p, x, fb, old_pre: (x, pre)],
        hc_pre_from_prev_sublayer=True,
        hc_mult=4,
    )
    legacy = Mock(side_effect=lambda x, p, m, dtype: combine(x.view(-1, m, 8), p))
    with patch.dict(
        "sys.modules",
        {"sglang.kernels.ops.layernorm.mhc": SimpleNamespace(mhc_combine=legacy)},
    ):
        result = ns["_forward_impl"](model, None, None, None, x).hidden_states
    expected = (x.float() * pre.unsqueeze(-1)).sum(1).to(x.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert ns["mhc"].combine_premix.call_count == int(a3)
    assert legacy.call_count == int(not a3)


@pytest.mark.parametrize("for_quant_config", [False, True])
@pytest.mark.parametrize(
    "source,target",
    [
        ("mtp.2.markov_head.embed.weight", "markov_head.markov_w1.weight"),
        ("mtp.2.markov_head.head.weight", "markov_head.markov_w2.weight"),
        ("mtp.2.markov_head.markov_w1.weight", "markov_head.markov_w1.weight"),
        ("mtp.2.markov_head.markov_w2.weight", "markov_head.markov_w2.weight"),
        ("mtp.2.confidence_head.proj.weight", "confidence_head.proj.weight"),
        ("mtp.0.attn.wq_b.weight_scale", "stages.0.self_attn.wq_b.weight_scale"),
        (
            "mtp.1.ffn.experts.0.w1.weight",
            "stages.1.mlp.experts.0.gate_proj.weight",
        ),
    ],
)
def test_modelslim_mtp_names_resolve_for_loading_and_quant_config(
    source, target, for_quant_config
):
    remap = _extract(MODELSLIM, ["remap_dspark_weight_name_npu"], {})[
        "remap_dspark_weight_name_npu"
    ]
    assert remap(source, for_quant_config=for_quant_config) == target


def test_modelslim_draft_keeps_stage_specific_vocab_weights():
    remap = _extract(MODELSLIM, ["remap_dspark_weight_name_npu"], {})[
        "remap_dspark_weight_name_npu"
    ]
    assert remap("mtp.0.embed.weight") == "embed_tokens.weight"
    assert remap("mtp.2.head.weight") == "lm_head.weight"
    assert remap("mtp.2.norm.weight") == "stages.2.norm.weight"
