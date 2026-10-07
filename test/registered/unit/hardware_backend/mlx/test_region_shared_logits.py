"""Shared processor capture preserves pruning, scaling and sampler outputs."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.hardware_backend.mlx.export_validation import (
    ServingForwardArg,
    ServingForwardExportWrapper,
    serving_forward_args,
)
from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxLoweringRegistry,
    build_mlx_fx_plan,
    make_mlx_fx_executor,
)
from sglang.srt.hardware_backend.mlx.region_runner import (
    MlxRegionRunner,
    _nontrivial_logits_reason,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci, register_mps_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


class Body(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("hidden", hidden)
        block = torch.nn.Module()
        block.attn = RadixAttention(1, 8, 8**-0.5, num_kv_heads=1, layer_id=0)
        self.layers = torch.nn.ModuleList([block])

    def forward(self, input_ids, positions, forward_batch):
        return self.hidden


def _runner(device, scale, fp32):
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(
        8, 64, bias=False, device=device, dtype=torch.bfloat16
    )
    model.logits_processor = LogitsProcessor(
        SimpleNamespace(vocab_size=61, final_logit_softcapping=None),
        skip_all_gather=True,
        logit_scale=scale,
    )
    model.logits_processor.use_fp32_lm_head = fp32
    kv = torch.zeros(1, 1, 8, device=device)
    runner = MlxRegionRunner.__new__(MlxRegionRunner)
    runner.model_runner = SimpleNamespace(
        device=device,
        model=model,
        attention_layers=[object()],
        attn_backend=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros(4, 8, dtype=torch.int32, device=device)
            ),
            token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda layer: (kv, kv)),
        ),
    )
    runner._decode_batch_sizes = (1, 2, 4)
    runner._prefill_token_buckets = (8, 32)
    runner._prefill_batch_sizes = (1, 2, 4)
    runner._max_prefill_padding_ratio = 3.0
    runner._executors = {}
    return runner


def _graph(wrapper, args):
    graph = torch.export.export(wrapper, args, strict=False).module()
    for node in tuple(graph.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            assert not node.users
            graph.graph.erase_node(node)
    graph.recompile()
    return graph


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("mode", ["decode", "extend", "extend_batch"])
@pytest.mark.parametrize("scale,fp32", [(None, False), (0.25, False), (None, True)])
def test_captured_processor_matches_contiguous_serving(device, mode, scale, fp32):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    with get_context().override_server_args():
        torch.manual_seed(42)
        runner = _runner(device, scale, fp32)
        indices = None
        if mode == "decode":
            batch = runner._synthetic_batch(("decode", 3))
            padded = runner._pad_decode_batch(batch, 4)
            hidden = torch.randn(4, 8).to(device, torch.bfloat16)
            real_hidden = hidden[:3]
        elif mode == "extend":
            batch = runner._synthetic_batch(("extend", 5))
            padded = runner._pad_extend_batch(batch, 8)
            hidden = torch.randn(8, 8).to(device, torch.bfloat16)
            real_hidden = hidden[:5]
        else:
            batch = runner._synthetic_batch(("extend_batch", 3, 5))
            padded, indices = runner._pad_packed_extend_batch(batch, 4, 8)
            hidden = torch.randn(32, 8).to(device, torch.bfloat16)
            real_hidden = hidden.reshape(4, 8, 8)[:3, :5].reshape(15, 8)
        model = runner.model_runner.model
        model.body = Body(hidden)
        wrapper = ServingForwardExportWrapper(model, runner.model_runner, padded)
        args = serving_forward_args(padded, indices)
        expected = model.logits_processor(
            batch.input_ids, real_hidden, model.lm_head, batch
        ).next_token_logits
        graph = _graph(wrapper, args)
        plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_export_decoder())
        plan.require_fully_supported()
        torch.testing.assert_close(
            graph(*args)[: batch.batch_size], expected, atol=0, rtol=0
        )
        compiled = make_mlx_fx_executor(plan, list(args)) if device == "mps" else None
        execute = (
            (lambda *inputs: compiled(*inputs)[0]) if compiled is not None else graph
        )
        runner._executors[runner._executor_key(batch)] = SimpleNamespace(
            execute=execute
        )
        with mock.patch.object(
            model.logits_processor,
            "forward",
            side_effect=AssertionError("runner must not execute a second processor"),
        ):
            actual = runner.execute(batch).next_token_logits
        assert actual.shape == (batch.batch_size, 61)
        assert actual.dtype == torch.float32
        # BF16 matmul uses different reduction kernels across runtimes.
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.008, rtol=0.03)
        assert _nontrivial_logits_reason(model) is None
        # Reuse the captured graph with changed live last-token positions.
        if mode == "extend_batch":
            changed = list(args)
            changed[ServingForwardArg.SAMPLING_INDICES] = indices - 1
            ref = graph(*changed)
            torch.testing.assert_close(
                execute(*changed).cpu(), ref.cpu(), atol=0.008, rtol=0.03
            )


def test_unsafe_inplace_scaling_is_not_admitted():
    class Mutate(torch.nn.Module):
        def forward(self, value):
            return value.mul_(0.25)

    graph = torch.export.export(
        Mutate(), (torch.ones(3, 8),), strict=False
    ).graph_module
    plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_export_decoder())
    assert any(node.target == torch.ops.aten.mul_.Tensor for node in plan.unsupported)


def test_inplace_scaling_requires_a_registered_functional_lowering():
    class Scale(torch.nn.Module):
        def forward(self, value, weight):
            return (value @ weight).mul_(0.25)

    graph = torch.export.export(
        Scale(), (torch.ones(3, 8), torch.ones(8, 4)), strict=False
    ).graph_module
    plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry())
    assert any(node.target == torch.ops.aten.mul_.Tensor for node in plan.unsupported)


def test_alias_of_matmul_is_not_admitted_for_inplace_scaling():
    class Alias(torch.nn.Module):
        def forward(self, value, weight):
            logits = value @ weight
            view = logits.view(-1)
            logits.mul_(0.25)
            return view, logits

    graph = torch.export.export(
        Alias(), (torch.ones(3, 8), torch.ones(8, 4)), strict=False
    ).graph_module
    plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_export_decoder())
    assert any(node.target == torch.ops.aten.mul_.Tensor for node in plan.unsupported)


def test_softcap_and_missing_head_remain_rejected():
    with get_context().override_server_args():
        runner = _runner("cpu", None, False)
        runner.model_runner.model.logits_processor.final_logit_softcapping = 30.0
        assert "softcapping" in _nontrivial_logits_reason(runner.model_runner.model)
        del runner.model_runner.model.lm_head
        assert "lm_head" in _nontrivial_logits_reason(runner.model_runner.model)


def test_quantized_head_is_rejected_before_region_execution():
    with get_context().override_server_args():
        runner = _runner("cpu", None, False)
        runner.model_runner.model.lm_head.quant_method = SimpleNamespace(
            apply=lambda *args: pytest.fail("unsupported projection must not run")
        )
        assert "quantized" in _nontrivial_logits_reason(runner.model_runner.model)
