"""CPU contract tests for DSpark device lengths and NPU graph routing."""

import __future__

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from test_npu_flash_mla import mod

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SOURCE = Path(__file__).resolve().parents[5] / "python/sglang/srt"


def source_method(path, cls, method, namespace):
    """Run the real orchestration method without importing device-only workers."""
    tree = ast.parse((SOURCE / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    function = next(
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    exec(
        compile(
            module,
            str(SOURCE / path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[method]


@pytest.fixture
def backend():
    obj = mod.AscendAttnBackend.__new__(mod.AscendAttnBackend)
    obj.use_paged_mtp = True
    obj.use_flash_mla = obj.use_mla = False
    obj.is_draft_worker = True
    obj.needs_cpu_seq_lens = False
    obj.device = "cpu"
    obj.page_size, obj.speculative_num_draft_tokens = 128, 7
    obj.max_context_len = 512
    obj.graph_mode = obj.enable_torch_compile = False
    obj.enable_sparsity_driven_kv_offload = False
    obj.is_hybrid_swa = obj.use_sliding_window_kv_pool = False
    obj.q_head_num_padding = None
    obj.req_to_token = torch.arange(1024, dtype=torch.int32).reshape(2, 512)
    obj.req_to_token_pool = SimpleNamespace(req_to_token=obj.req_to_token)
    obj.graph_metadata = {"block_tables": torch.zeros(2, 4, dtype=torch.int32)}
    obj.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=Mock(return_value=torch.zeros(8, 128, 32)),
        get_value_buffer=Mock(return_value=torch.zeros(8, 128, 32)),
    )
    with patch.object(
        mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
    ):
        yield obj


def batch(lengths=(127, 0)):
    return SimpleNamespace(
        batch_size=len(lengths),
        forward_mode=ForwardMode.TARGET_VERIFY,
        # Match the int64 input lengths created by ScheduleBatch and graph buffers.
        seq_lens=torch.tensor(lengths, dtype=torch.int64),
        seq_lens_cpu=None,
        req_pool_indices=torch.arange(len(lengths)),
        out_cache_loc=None,
        extend_seq_lens=None,
        extend_seq_lens_cpu=[],
        spec_algorithm=SimpleNamespace(
            is_dflash=lambda: False, is_dflash_family=lambda: True
        ),
        spec_info=SimpleNamespace(
            draft_token_num=7, spec_input_type=mod.SpecInputType.DFLASH_VERIFY
        ),
        global_num_token_non_padded_cpu=len(lengths) * 7,
    )


def test_eager_metadata_uses_valid_device_end_and_preserves_prefix(backend):
    fb = batch()
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
        patch.object(
            torch.Tensor, "item", side_effect=AssertionError("unexpected scalar sync")
        ),
    ):
        backend.init_forward_metadata(fb)
    assert backend.forward_metadata.seq_lens.tolist() == [134, 0]
    assert backend.forward_metadata.cu_seqlens_q.tolist() == [0, 7, 14]
    assert fb.seq_lens.tolist() == [127, 0]
    assert fb.seq_lens_cpu is None
    assert backend.forward_metadata.seq_lens.data_ptr() != fb.seq_lens.data_ptr()


def test_graph_refresh_keeps_storage_and_masks_reused_padding_slots(backend):
    fb = batch()
    with (
        patch.object(
            mod, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
        patch.object(
            torch.Tensor, "item", side_effect=AssertionError("unexpected scalar sync")
        ),
    ):
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        metadata = backend.forward_metadata
        pointers = (metadata.seq_lens.data_ptr(), metadata.cu_seqlens_q.data_ptr())
        fb.seq_lens.copy_(torch.tensor([254, 127], dtype=torch.int64))
        backend.init_forward_metadata_out_graph(fb)
        updated = metadata.seq_lens.clone()
        fb.seq_lens.copy_(torch.tensor([255, 0], dtype=torch.int64))
        backend.init_forward_metadata_out_graph(fb)
    assert updated.tolist() == [261, 134]
    assert metadata.seq_lens.tolist() == [262, 0]
    assert fb.seq_lens.tolist() == [255, 0]
    assert pointers == (
        metadata.seq_lens.data_ptr(),
        metadata.cu_seqlens_q.data_ptr(),
    )
    assert metadata.seq_lens_cpu_list is metadata.seq_lens_cpu_int is None


@pytest.mark.parametrize("needs_cpu", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_shared_metadata_respects_cpu_mirror_flag(backend, needs_cpu, graph):
    backend.needs_cpu_seq_lens = needs_cpu
    fb = batch((121, 249))
    fb.seq_lens_cpu = torch.tensor([128, 256], dtype=torch.int32) if needs_cpu else None
    if graph:
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        metadata = backend.forward_metadata
        pointers = (metadata.seq_lens.data_ptr(), metadata.cu_seqlens_q.data_ptr())
    with patch.object(
        torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")
    ):
        if graph:
            backend.init_forward_metadata_out_graph(fb)
        else:
            backend.init_forward_metadata(fb)
    metadata = backend.forward_metadata
    assert metadata.seq_lens.tolist() == [128, 256]
    assert metadata.cu_seqlens_q.tolist() == [0, 7, 14]
    assert metadata.block_tables.dtype == torch.int32
    assert fb.seq_lens.tolist() == [121, 249]
    if needs_cpu:
        assert fb.seq_lens_cpu.tolist() == [128, 256]
        if graph:
            assert metadata.seq_lens_cpu_list == [121, 249]
        else:
            torch.testing.assert_close(metadata.seq_lens_cpu_int, fb.seq_lens_cpu)
    else:
        assert metadata.seq_lens_cpu_list is metadata.seq_lens_cpu_int is None
    if graph:
        assert pointers == (
            metadata.seq_lens.data_ptr(),
            metadata.cu_seqlens_q.data_ptr(),
        )


@pytest.mark.parametrize("causal", [True, False])
def test_forward_uses_device_lengths_without_host_attention_and_restores_dp_padding(
    backend, causal
):
    fb = batch()
    backend.init_forward_metadata(fb)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        qk_head_dim=16,
        v_head_dim=16,
        sliding_window_size=-1,
        scaling=0.25,
        attn_type=mod.AttentionType.DECODER
        if causal
        else mod.AttentionType.ENCODER_ONLY,
    )
    kernel = Mock(side_effect=lambda **kw: torch.ones_like(kw["query"]))
    module_name = "sgl_kernel_npu.attention.paged_mtp_attention"
    with (
        patch.dict(
            sys.modules,
            {module_name: SimpleNamespace(paged_mtp_attention_decode=kernel)},
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")),
        patch.object(
            torch.Tensor, "item", side_effect=AssertionError("unexpected scalar sync")
        ),
    ):
        output = backend.forward_mtp(
            torch.zeros(17, 4, 16), None, None, layer, fb, save_kv_cache=False
        )
    assert output.shape == (17, 64)
    assert torch.count_nonzero(output[14:]) == 0
    assert kernel.call_args.kwargs["seq_lens"].tolist() == [134, 0]
    assert kernel.call_args.kwargs["max_kv_len"] == 519
    assert kernel.call_args.kwargs["causal"] is causal


@pytest.mark.parametrize(
    "needs_cpu,mirror,allocation",
    [
        (False, None, None),
        (False, None, 256),
        (False, 127, 256),
        (True, 127, 256),
        (True, None, 256),
        (True, None, None),
    ],
)
def test_draft_preserves_cpu_lengths_unless_backend_opts_out(
    needs_cpu, mirror, allocation
):
    method = source_method(
        "speculative/dspark_components/dspark_draft.py",
        "DraftBlockProposer",
        "_run_forward",
        {
            "torch": torch,
            "ForwardBatch": SimpleNamespace,
            "ForwardMode": ForwardMode,
            "CaptureHiddenMode": SimpleNamespace(NULL=0),
            "SpeculativeAlgorithm": SimpleNamespace(DSPARK=object()),
            "envs": SimpleNamespace(
                SGLANG_DSPARK_EMBED_IN_GRAPH=SimpleNamespace(get=lambda: False)
            ),
            "_make_num_token_non_padded": lambda *args: None,
            "DraftForwardResult": SimpleNamespace,
        },
    )
    draft_block_ids_view = source_method(
        "speculative/dspark_components/dspark_draft.py",
        "DraftBlockProposer",
        "draft_block_ids_view",
        {"torch": torch},
    )
    runner = SimpleNamespace(
        attn_backend=SimpleNamespace(needs_cpu_seq_lens=needs_cpu),
        forward=Mock(
            return_value=SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=torch.zeros(7, 16)),
                can_run_graph=False,
            )
        ),
        decode_cuda_graph_runner=None,
        kv_index_translator=object(),
    )
    proposer = SimpleNamespace(
        gamma=7,
        query_token_num=7,
        sample_from_anchor=True,
        _draft_block_ids_buf=None,
        _mask_token_id=0,
        draft_model=object(),
        draft_model_runner=runner,
        _draft_block_spec_info=object(),
        _fill_dp_moe_sync_metadata=Mock(),
        deferred_commit=None,
    )
    proposer.draft_block_ids_view = lambda **kwargs: draft_block_ids_view(
        proposer, **kwargs
    )
    fb = batch((127,))
    fb.seq_lens_cpu = torch.tensor([mirror]) if mirror is not None else None
    draft = SimpleNamespace(
        bonus_tokens=torch.tensor([1]),
        nxt_kv_lens_cpu=torch.tensor([allocation]) if allocation else None,
        nxt_kv_lens_sum=allocation,
    )
    window = SimpleNamespace(
        positions_2d=torch.arange(7).reshape(1, 7),
        verify_cache_loc_2d=torch.arange(7).reshape(1, 7),
        kv_loc_plan=SimpleNamespace(bind=Mock()),
    )
    kwargs = dict(
        batch=fb,
        draft_input=draft,
        verify_window=window,
        bs=1,
        device="cpu",
        embed_module=lambda x: torch.zeros(1, 7, 16),
    )
    if needs_cpu and mirror is None and allocation is None:
        with pytest.raises(RuntimeError, match="batch.seq_lens_cpu"):
            method(proposer, **kwargs)
        runner.forward.assert_not_called()
        return
    method(proposer, **kwargs)
    forwarded = runner.forward.call_args.args[0]
    assert forwarded.seq_lens.tolist() == [127]
    if needs_cpu:
        expected = mirror + 7 if mirror is not None else allocation
        assert forwarded.seq_lens_cpu.tolist() == [expected]
        assert forwarded.seq_lens_sum == expected
    else:
        assert forwarded.seq_lens_cpu is forwarded.seq_lens_sum is None


@pytest.mark.parametrize(
    "needs_cpu,mirror", [(False, None), (False, 127), (True, None), (True, 127)]
)
def test_target_verify_preserves_cpu_path_unless_backend_opts_out(needs_cpu, mirror):
    method = source_method(
        "speculative/dspark_components/dspark_verify.py",
        "TargetVerifyExecutor",
        "run_non_compact",
        {
            "torch": torch,
            "DFlashVerifyInput": SimpleNamespace,
            "CaptureHiddenMode": SimpleNamespace(FULL=0),
        },
    )
    fb = batch((127,))
    fb.seq_lens_cpu = torch.tensor([mirror]) if mirror is not None else None
    fb.seq_lens_sum = mirror
    executor = SimpleNamespace(
        verify_num_draft_tokens=8,
        _verify_backend_self_adds_seq_lens=lambda: False,
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(needs_cpu_seq_lens=needs_cpu)
            )
        ),
        _forward_prepared_verify=Mock(return_value=object()),
    )
    kwargs = dict(
        batch=fb,
        draft_input=SimpleNamespace(
            nxt_kv_lens_cpu=torch.tensor([256]), nxt_kv_lens_sum=256
        ),
        verify_ids_2d=torch.zeros(1, 8),
        verify_window=SimpleNamespace(
            positions_2d=torch.arange(8),
            verify_cache_loc=torch.arange(8),
            kv_loc_plan=object(),
        ),
        sampling_info=None,
    )
    method(executor, **kwargs)
    assert fb.seq_lens.tolist() == [127]
    if not needs_cpu:
        if mirror is None:
            assert fb.seq_lens_cpu is fb.seq_lens_sum is None
        else:
            assert fb.seq_lens_cpu.tolist() == [mirror]
    else:
        expected = mirror + 8 if mirror is not None else 256
        assert fb.seq_lens_cpu.tolist() == [expected]
        assert fb.seq_lens_sum == expected


def test_device_length_graph_uses_plain_replay_after_loading_inputs(backend):
    method = source_method(
        "hardware_backend/npu/graph_runner/npu_graph_runner.py",
        "NPUGraphRunner",
        "execute",
        {
            "LogitsProcessorOutput": SimpleNamespace,
        },
    )
    fb = batch((127,))
    fb.needs_forward_metadata_init = lambda: True
    output = SimpleNamespace(next_token_logits=None, hidden_states=torch.ones(7, 16))
    runner = SimpleNamespace(
        _replay_attn_backend=lambda: backend,
        load_batch=Mock(),
        _make_graph_key=lambda bs: bs,
        bs=1,
        raw_bs=1,
        raw_num_token=7,
        is_dllm=False,
        backend=SimpleNamespace(
            replay=Mock(return_value=output),
            replay_with_input_update=Mock(side_effect=AssertionError("host update")),
        ),
    )
    with patch.object(
        torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")
    ):
        method(runner, fb)
    runner.load_batch.assert_called_once_with(fb, None)
    runner.backend.replay.assert_called_once_with(1, fb)
    runner.backend.replay_with_input_update.assert_not_called()
