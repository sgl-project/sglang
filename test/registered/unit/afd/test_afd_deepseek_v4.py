"""V4 operand identity and mHC ownership across the real AFD stage boundary."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch

from sglang.srt.afd.config import AFDExecutionMode
from sglang.srt.afd.contracts import AFDError, AFDRole
from sglang.srt.afd.model_adapters.deepseek_v4 import (
    DeepseekV4AFDAdapter,
    owns_checkpoint_weight,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(ids):
    lengths = torch.full((len(ids),), 3, dtype=torch.int32)
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=len(ids),
        input_ids=torch.tensor(ids),
        req_pool_indices=torch.tensor(ids),
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        seq_lens_sum=int(lengths.sum()),
        out_cache_loc=torch.tensor(ids),
        positions=lengths.long() - 1,
    )


@pytest.mark.parametrize(
    "name,ffn",
    [
        ("layers.0.ffn.gate.weight", True),
        ("model.layers.0.mlp.experts.0.down_proj.weight", True),
        ("model.layers.0.mlp.shared_experts.up_proj.weight", True),
        ("model.layers.0.mlp.gate.tid2eid", True),
        ("model.layers.0.hc_ffn_fn", False),
        ("model.layers.0.post_attention_layernorm.weight", False),
        ("model.layers.0.self_attn.compressor.wkv.weight", False),
        ("model.embed_tokens.weight", False),
        ("lm_head.weight", False),
    ],
)
def test_role_weight_ownership_includes_router_but_keeps_mhc_on_a(name, ffn):
    assert owns_checkpoint_weight(name, AFDExecutionMode.FFN) == ffn
    assert owns_checkpoint_weight(name, AFDExecutionMode.ATTENTION) != ffn
    assert owns_checkpoint_weight(name, AFDExecutionMode.OFF)


@pytest.mark.parametrize(
    "empty,capturing", [(False, False), (False, True), (True, True)]
)
def test_hash_pipeline_preserves_int64_ids_and_refreshes_once_per_stage(
    monkeypatch, empty, capturing
):
    from sglang.srt import runtime_context
    from sglang.srt.afd.pipeline import AFDFFNPipeline

    adapter = object.__new__(DeepseekV4AFDAdapter)
    stages = [
        SimpleNamespace(
            index=i,
            input_ids=() if empty else (torch.zeros(2, 1, dtype=torch.int64),),
            collective_input_ids=torch.empty(0, 1, dtype=torch.int64),
            merge_sizes=(0, 2) if empty else (3, 2),
            merged_input_ids=None,
            graph_output=None,
            forward_batch=None,
        )
        for i in range(2)
    ]
    step = [0]
    gathers, receives, seen = [], [], []
    remote = torch.tensor([80003, 0], dtype=torch.int64)

    def gather(local, *, sizes):
        assert local.dtype == torch.int64 and local.shape == (sizes[0], 1)
        gathers.append(local.clone())
        return [torch.cat((local.flatten(), remote)).reshape(-1, 1)]

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_group=SimpleNamespace(all_gatherv=gather)),
    )

    def receive(buffers):
        if buffers and buffers[0].dtype == torch.int64:
            index = next(
                i
                for i, s in enumerate(stages)
                if buffers[0].data_ptr() == s.input_ids[0].data_ptr()
            )
            buffers[0].copy_(
                torch.tensor([[70001 + step[0] + index], [2**33 + step[0] + index]])
            )
            receives.append((step[0], index))

    layer = SimpleNamespace(mlp=SimpleNamespace(is_hash=True))

    def output(hidden, batch, *, input_ids):
        seen.append(input_ids.clone())
        return hidden

    layer.compute_ffn_output = output

    def compute(*, layer, stage, **kwargs):
        hidden = torch.ones(sum(stage.merge_sizes), 4)
        adapter._compute_ffn_output(
            decoder,
            hidden,
            stage,
            rows=() if empty else (2,),
            strides=() if empty else (3,),
        )
        return (), None

    decoder = layer
    pipeline = AFDFFNPipeline.__new__(AFDFFNPipeline)
    pipeline._adapter = SimpleNamespace(
        num_layers=3, requires_token_ids=True, local_compute=compute
    )
    pipeline._connector = SimpleNamespace(
        transport=SimpleNamespace(
            capturing=capturing,
            receive_dispatch=receive,
            wait=lambda _: None,
            return_result=lambda _: None,
        )
    )
    for iteration in range(2):
        step[0] = iteration
        pipeline._run_layers(
            stages=stages,
            recv_buffers=[(torch.zeros(2, 4),)] * 2,
            participates=(True, True),
        )
        for stage in stages:
            expected = (
                remote
                if empty
                else torch.cat(
                    (stage.input_ids[0].flatten(), torch.tensor([0]), remote)
                )
            )
            torch.testing.assert_close(stage.merged_input_ids, expected)
        for index in range(2):
            expected = stages[index].merged_input_ids
            for actual in seen[-6 + index :: 2]:
                torch.testing.assert_close(actual, expected)
    assert len(gathers) == 4
    assert receives == ([] if empty else [(i, s) for i in range(2) for s in range(2)])


def test_two_s2_stages_materialize_native_mhc_once_with_no_ffn_sum(monkeypatch):
    from sglang.srt.layers.layer_boundary import append_stages, layer_stack
    from sglang.srt.models import deepseek_v4
    from sglang.srt.models import deepseek_v4_boundary as boundary
    from sglang.test import boundary_fixtures as fixture

    monkeypatch.setattr(
        deepseek_v4, "afd_execution_mode", lambda: AFDExecutionMode.ATTENTION
    )
    parallel = fixture.parallel_of(attn_dp=4, attn_tp=1)
    monkeypatch.setattr(boundary, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        boundary, "get_moe_a2a_backend", lambda: SimpleNamespace(is_none=lambda: True)
    )
    monkeypatch.setattr(
        boundary,
        "get_cp_decode_attn_tp_ctx",
        lambda: SimpleNamespace(use_decode_attn_tp=False),
    )
    config = SimpleNamespace(hc_pre_from_prev_sublayer=False, model_type="deepseek_v4")
    post = torch.tensor([0.25, 0.75])
    comb = torch.tensor([[0.9, 0.1], [0.2, 0.8]])

    def update(value, streams, p, c):
        return (
            torch.einsum("bsh,st->bth", streams, c) + value[:, None] * p[None, :, None]
        )

    def pre(streams, *args, **kwargs):
        return streams.mean(1), post, comb, True

    layer = SimpleNamespace(
        config=config,
        boundary_pre_from_prev=False,
        boundary_mhc_fusions=False,
        use_fused_mhc_post_pre=False,
        afd_local_ffn=True,
        hc_pre=pre,
        hc_post=update,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        self_attn=SimpleNamespace(maybe_use_decode_attn_tp=lambda _: nullcontext()),
        _run_boundary_attention=lambda hidden, positions, batch, quantized: hidden + 3,
    )
    # Attention's native ATTN_TP output contract is exercised by the upstream
    # suite. Here TP=1 must complete its declared group exactly once.
    layer._bind_attention_transport = lambda hidden, batch, call, rows, other: (
        boundary.AttentionTransport(layer, hidden, batch, call, rows)
    )
    layer._bind_moe_transport = lambda *args: (
        deepseek_v4.DeepseekV4CudaDecoderLayer._bind_moe_transport(layer, *args)
    )
    layer.forward_attention_for_afd = MethodType(
        deepseek_v4.DeepseekV4CudaDecoderLayer.forward_attention_for_afd, layer
    )
    with fixture.planning(parallel), layer_stack(final_read=boundary.FinalRead()):
        layer.attn_boundary, layer.ffn_boundary = append_stages(
            *(
                (d, torch.nn.Identity())
                for d in boundary.stages(config, 0, layer=layer, afd_mode="attention")
            )
        )
    adapter = object.__new__(DeepseekV4AFDAdapter)
    adapter.role, adapter.num_layers = AFDRole.ATTENTION, 1
    adapter.inner = SimpleNamespace(layers=[layer])
    batch = _batch([0, 1, 2])
    source = torch.arange(24.0).reshape(3, 2, 4)
    stages = adapter.split_step(
        hidden_states=source,
        residual=None,
        positions=batch.positions,
        forward_batch=batch,
        stages=2,
    )
    for shift in (0, 11):
        for stage in stages:
            streams = source[stage.token_start : stage.token_stop] + shift
            operand, _ = adapter.local_compute(
                layer=0, stage=stage, hidden_states=streams, residual=None
            )
            returned = operand * 1.5
            actual, residual = adapter.finish_layer(
                layer=0, stage=stage, ffn_output=returned, residual=None
            )
            after_attn = update(streams.mean(1) + 3, streams, post, comb)
            expected = update(after_attn.mean(1) * 1.5, after_attn, post, comb)
            torch.testing.assert_close(actual, expected)
            assert residual is None and stage.forward_batch.residual_stream is None
    assert batch.residual_stream is None


@pytest.mark.parametrize("trtllm", [False, True])
def test_private_decode_operands_refresh_without_aliasing_s2_or_warmup(
    monkeypatch, trtllm
):
    from sglang.srt import runtime_context
    from sglang.srt.afd.metadata import DSV4MetadataGuard, metadata_guard_for_backend
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    backend_type = DeepseekV4AttnBackend
    if trtllm:
        from sglang.srt.layers.attention.deepseek_v4_trtllm_backend import (
            DeepseekV4TrtllmAttnBackend,
        )

        backend_type = DeepseekV4TrtllmAttnBackend
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(
            kernel=SimpleNamespace(dsv4_attn_backend="trtllm" if trtllm else "flashmla")
        ),
    )
    guard_type = metadata_guard_for_backend("dsv4", default_guard=DSV4MetadataGuard)
    backend = object.__new__(backend_type)
    backend.trtllm_attn, backend.mega_attn = trtllm, False
    backend._interleaved_stage_scratch = {
        i: (torch.zeros(8), torch.zeros(4)) for i in range(2)
    }
    backend.topk = 0
    backend.is_dspark_draft = False
    backend._early_verify_metadata_ids = set()
    backend.forward_metadata = original = object()
    backend.tail_forward_metadata = tail = object()
    backend.encoder_replay, backend.encoder_row_floor = "original", 17
    backend.token_to_kv_pool = SimpleNamespace(
        request_window=None,
        swa_req_ring_size=None,
        translate_loc_from_full_to_swa=lambda loc: loc + 100,
    )

    def raw(batch):
        backend.encoder_replay = batch.encoder_swa_replay
        backend.encoder_row_floor = batch.encoder_swa_row_floor
        return backend.init_forward_metadata_decode(
            128, batch.req_pool_indices, batch.seq_lens.int(), batch.out_cache_loc
        )

    def materialize(raw_metadata):
        core = SimpleNamespace(
            low_ratios=(),
            seq_lens=raw_metadata.seq_lens + 0,
            ids=raw_metadata.req_pool_indices + 0,
        )
        # Native init_forward_metadata_in_graph dispatches by this real type.
        from sglang.srt.layers.attention.deepseek_v4_backend import DSV4Metadata

        full = DSV4Metadata(core_attn_metadata=core, indexer_metadata=None)
        return full

    monkeypatch.setattr(backend, "_build_forward_metadata", raw)
    monkeypatch.setattr(backend, "make_forward_metadata_from_raw_decode", materialize)

    batches = [_batch([1, 2]), _batch([7])]
    stages = [
        SimpleNamespace(index=i, forward_batch=b, attention_metadata=object())
        for i, b in enumerate(batches)
    ]
    guards = [guard_type(backend=backend, stage=s, bucket_rows=4) for s in stages]
    for guard, batch in zip(guards, batches):
        guard.capture(batch)
        guard.activate_in_graph()
    assert stages[0].attention_metadata is not stages[1].attention_metadata
    if trtllm:
        for i, stage in enumerate(stages):
            assert (
                stage.attention_metadata.trtllm_stage_scratch
                is backend._interleaved_stage_scratch[i]
            )
        with pytest.raises(AFDError, match="BACKEND_UNSUPPORTED"):
            DSV4MetadataGuard.validate_backend(backend)
    assert (
        guards[0]._raw_metadata.seq_lens.data_ptr()
        != guards[1]._raw_metadata.seq_lens.data_ptr()
    )
    first = stages[0].attention_metadata
    guards[0].graph_forward_batch.freqs_cis_c4 = torch.tensor([-1])
    guards[0].graph_forward_batch.freqs_cis_c128 = torch.tensor([-1])
    guards[0].activate_in_graph()  # capture after warmup must record raw->full again
    assert stages[0].attention_metadata is not first
    assert not hasattr(guards[0].graph_forward_batch, "freqs_cis_c4")
    assert not hasattr(guards[0].graph_forward_batch, "freqs_cis_c128")
    for guard in reversed(guards):
        guard.restore()
    assert (
        backend.forward_metadata is original and backend.tail_forward_metadata is tail
    )
    assert backend.encoder_replay == "original" and backend.encoder_row_floor == 17
    next_batch = _batch([9])
    before = guards[0]._addresses()
    guards[0].prepare_replay(next_batch)
    guards[0].assert_stable()
    assert before == guards[0]._addresses()
    torch.testing.assert_close(
        guards[0]._raw_metadata.req_pool_indices, torch.tensor([9, 0, 0, 0])
    )
    torch.testing.assert_close(
        guards[0]._raw_metadata.seq_lens, torch.tensor([3, 1, 1, 1], dtype=torch.int32)
    )
    assert stages[0].forward_batch is batches[0]
    guards[0].restore()
    guards[0]._raw_metadata.seq_lens = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(AFDError, match="ADDRESS_DRIFT"):
        guards[0].assert_stable()
    for guard in guards:
        guard.close()


def test_ffn_native_finalize_counts_replicated_shared_expert_once(monkeypatch):
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.models import deepseek_v4
    from sglang.srt.runtime_context import get_forward

    monkeypatch.setattr(deepseek_v4, "afd_execution_mode", lambda: AFDExecutionMode.FFN)
    monkeypatch.setattr(
        moe_utils, "post_experts_output_is_complete", lambda **kw: False
    )
    outputs = []
    for rank in range(4):
        monkeypatch.setattr(
            moe_utils, "get_parallel", lambda: SimpleNamespace(tp_size=4, tp_rank=rank)
        )
        pieces = SimpleNamespace(
            get_merged=lambda: torch.full((2, 4), float(rank + 1)),
            shared=torch.full((2, 4), 7.0),
            shared_is_replicated=True,
        )
        layer = SimpleNamespace(
            mlp=SimpleNamespace(
                is_hash=False, forward_normal=lambda hidden, **kw: pieces
            )
        )
        with get_forward().scoped(mlp_reduce_scatter=True):
            outputs.append(
                deepseek_v4.DeepseekV4CudaDecoderLayer.compute_ffn_output(
                    layer, torch.ones(2, 4)
                )
            )
    torch.testing.assert_close(torch.stack(outputs).sum(0), torch.full((2, 4), 17.0))
