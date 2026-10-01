from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture
def backend():
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.mtp_enabled = False
    backend.enable_decoder_swa_bounded_replay = True
    backend.low_ratios = ()
    backend.token_to_kv_pool = SimpleNamespace(request_window=None)
    backend._build_forward_metadata = Mock(
        side_effect=lambda *args, **kwargs: DSV4Metadata(
            core_attn_metadata=Mock(), indexer_metadata=None
        )
    )
    backend.init_forward_metadata_in_graph = Mock()
    backend._build_late_layer_tail_metadata = Mock(
        side_effect=lambda batch: DSV4Metadata(
            core_attn_metadata=None,
            indexer_metadata=None,
            late_layer_tail=SimpleNamespace(
                cp_metadata=None, extend_seq_lens_cpu=[128]
            ),
        )
    )
    return backend


@pytest.mark.parametrize("phase", ["capture", "replay"])
@pytest.mark.parametrize("rows_per_request", [[44], [24, 24], [80]])
def test_eager_tail_does_not_reach_prefill_graph(backend, phase, rows_per_request):
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        encoder_swa_replay=None,
        max_seq_len_override=256,
    )
    captured = backend.init_forward_metadata_for_breakable_cuda_graph_capture(batch)
    for _ in range(3):
        backend.init_forward_metadata(batch)
        eager_tail = backend.tail_forward_metadata
        eager = BlockIds(torch.arange(256).view(256, 1), [256])
        backend._publish_candidate_metadata(eager)
        torch.testing.assert_close(
            eager_tail.candidate_metadata.blocks, eager.blocks[-128:]
        )

        with enable_breakable_cuda_graph():
            if phase == "capture":
                captured = (
                    backend.init_forward_metadata_for_breakable_cuda_graph_capture(
                        batch
                    )
                )
            else:
                backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
                    captured, batch, static_forward_batch=batch
                )
            published = BlockIds(
                torch.arange(sum(rows_per_request)).view(-1, 1), rows_per_request
            )
            backend._publish_candidate_metadata(published)

        assert backend.forward_metadata is captured
        assert captured.candidate_metadata is published
        assert backend.tail_forward_metadata is None
        torch.testing.assert_close(
            eager_tail.candidate_metadata.blocks, eager.blocks[-128:]
        )


@pytest.mark.parametrize("cp", [False, True])
def test_eager_candidate_tail_uses_current_request_lengths(backend, cp):
    tail = SimpleNamespace(
        cp_metadata=object() if cp else None,
        local_lens_cpu=[1, 0, 2],
        extend_seq_lens_cpu=[2, 0, 3],
    )
    backend.forward_metadata = DSV4Metadata(None, None)
    backend.tail_forward_metadata = DSV4Metadata(None, None, late_layer_tail=tail)
    published = BlockIds(torch.arange(8).view(8, 1), [5, 0, 3])
    backend._publish_candidate_metadata(published)
    expected = [4, 6, 7] if cp else [3, 4, 5, 6, 7]
    assert (
        backend.tail_forward_metadata.candidate_metadata.blocks.flatten().tolist()
        == expected
    )
    backend.forward_metadata = backend.tail_forward_metadata
    backend._publish_candidate_metadata(published)
    assert backend.forward_metadata.candidate_metadata is published


def test_breakable_model_forward_skips_eager_tail(monkeypatch):
    from sglang.srt.models import deepseek_v4 as model

    layer_stack = SimpleNamespace(
        pp_group=SimpleNamespace(world_size=1),
        engram_hasher=None,
        late_layer_start=20,
        start_layer=0,
        end_layer=0,
        _check_late_layer_tail_readers=Mock(),
    )
    monkeypatch.setattr(model, "is_cp_active", lambda batch: False)
    monkeypatch.setattr(
        model, "get_attn_backend", lambda: SimpleNamespace(tail_forward_metadata=None)
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    hidden = torch.zeros(44, 4, 8)
    ids = torch.arange(44)
    with enable_breakable_cuda_graph():
        output, _, tail = model.DeepseekV4Model._forward_layers_hc_pre_from_prev(
            layer_stack, ids, hidden, batch, ids, None, False, []
        )
    assert output is hidden
    assert tail is None
    layer_stack._check_late_layer_tail_readers.assert_not_called()


def test_breakable_aux_output_keeps_full_rows(monkeypatch):
    from sglang.srt.models import deepseek_v4 as model

    hidden = torch.zeros(44, 8)
    aux = [torch.ones_like(hidden)]
    network = SimpleNamespace(
        vision=None,
        pp_group=SimpleNamespace(is_last_rank=True),
        capture_aux_hidden_states=True,
        model=SimpleNamespace(
            forward=Mock(return_value=((hidden, None), aux)), late_layer_start=20
        ),
        lm_head=object(),
        logits_processor=Mock(),
    )
    monkeypatch.setattr(
        model,
        "get_attn_tp_context",
        lambda: SimpleNamespace(maybe_input_scattered=lambda batch: nullcontext()),
    )
    monkeypatch.setattr(
        model, "get_attn_backend", lambda: SimpleNamespace(tail_forward_metadata=None)
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    ids = torch.arange(44)
    with enable_breakable_cuda_graph():
        result = model.DeepseekV4ForCausalLM.forward(network, ids, ids, batch)
    network.logits_processor.assert_called_once_with(
        ids, hidden, network.lm_head, batch, aux, hidden_states_before_norm=None
    )
    assert result is network.logits_processor.return_value


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
