import copy
from types import SimpleNamespace

import pytest
import torch
from test_transformers_backend_runtime import (
    ReferenceAttentionBackend,
    config_for,
    model_settings,
    packed_batch,
)
from test_transformers_backend_runtime import runtime as runtime_fixture
from torch import nn
from transformers import AutoModel

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _build_layer_model_forward_kwargs,
    _resolve_transformer_layer_model,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.models.transformers import TransformersEmbeddingModel
from sglang.srt.models.transformers.graph import TransformersGraphBody
from sglang.test.ci.ci_register import register_cpu_ci

runtime = runtime_fixture

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.mark.parametrize("kind", ["bert", "roberta", "qwen3"])
def test_native_graph_boundary_packs_hf_inputs_and_discovers_registered_attention(
    runtime, kind
):
    config = config_for(kind)
    settings = model_settings(config)
    reference = AutoModel.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper = TransformersEmbeddingModel(copy.deepcopy(config), model_config=settings)
    wrapper.load_weights(reference.state_dict().items())
    batch = packed_batch()
    batch.input_embeds = None
    if kind != "bert":
        batch.token_type_ids = None
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    state_names = set(wrapper.state_dict())
    with torch.no_grad(), forward_context(ForwardContext(ReferenceAttentionBackend())):
        expected = wrapper(batch.input_ids, positions, batch, get_embedding=True)
        body = _resolve_transformer_layer_model(wrapper)
        assert isinstance(body, TransformersGraphBody)
        assert _build_layer_model_forward_kwargs(body, batch, None) == {
            "input_embeds": None
        }
        hidden = body(batch.input_ids, positions, batch)
        actual = wrapper(batch.input_ids, positions, batch, get_embedding=True)
    assert hidden.shape == (7, 16)
    torch.testing.assert_close(actual.embeddings, expected.embeddings)
    assert set(wrapper.state_dict()) == state_names
    layers = compute_attention_and_moe_layers(body)
    assert layers.attention_layers == [
        wrapper.attention_instances[str(i)] for i in range(2)
    ]
    assert all(layer is not None for layer in layers.attention_layers)


def test_graph_registry_refreshes_token_types_and_preserves_addresses(runtime):
    config = config_for("bert")
    wrapper = TransformersEmbeddingModel(config, model_config=model_settings(config))
    registry = CudaGraphBufferRegistry(
        device=torch.device("cpu"), max_bs=2, max_num_tokens=8
    )
    wrapper.register_prefill_graph_inputs(registry)
    slot = registry.get_slot("token_type_ids")
    address = slot.buffer.data_ptr()
    batch = packed_batch()
    registry.fill_from(
        batch, raw_bs=2, padded_bs=2, raw_num_tokens=7, padded_num_tokens=8
    )
    torch.testing.assert_close(
        slot.buffer, torch.cat([batch.token_type_ids, torch.zeros(1, dtype=torch.long)])
    )
    batch.token_type_ids = None
    registry.fill_from(
        batch, raw_bs=2, padded_bs=2, raw_num_tokens=7, padded_num_tokens=8
    )
    assert slot.buffer.data_ptr() == address
    assert not slot.buffer.any()


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_prefill_runner_replay_protocol_keeps_hf_boundary_and_pooling_tail(
    runtime, full, use_embeds
):
    config = config_for("bert")
    settings = model_settings(config)
    reference = AutoModel.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper = TransformersEmbeddingModel(copy.deepcopy(config), model_config=settings)
    wrapper.load_weights(reference.state_dict().items())
    body = wrapper.get_prefill_graph_model()
    registry = CudaGraphBufferRegistry(
        device=torch.device("cpu"), max_bs=2, max_num_tokens=7
    )
    wrapper.register_prefill_graph_inputs(registry)
    if use_embeds:
        registry.register_slot(
            GraphSlot(
                "input_embeds",
                lambda _bs, tokens: (tokens, 16),
                torch.float32,
                copy_from_fb=False,
            )
        )
    raw = packed_batch()
    raw.positions = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    raw.mm_input_embeds = None
    raw.attn_tp_sequence_sharded = False
    with torch.no_grad(), forward_context(ForwardContext(ReferenceAttentionBackend())):
        expected = wrapper(raw.input_ids, raw.positions, raw, get_embedding=True)
    registry.fill_from(
        raw, raw_bs=2, padded_bs=2, raw_num_tokens=7, padded_num_tokens=7
    )
    static = copy.copy(raw)
    static.input_embeds = (
        registry.get_slot("input_embeds").buffer if use_embeds else None
    )
    static.token_type_ids = registry.get_slot("token_type_ids").buffer
    original = body.forward
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner.layer_model = body
    runner._is_full_backend = full
    runner._input_embeds_arg_idx = 3
    runner.buffer_registry = registry
    runner.model_runner = SimpleNamespace(
        model=wrapper, pp_group=SimpleNamespace(is_first_rank=True)
    )
    runner._prefill_forward_context = lambda *_args, **_kwargs: forward_context(
        ForwardContext(ReferenceAttentionBackend())
    )
    runner.backend = SimpleNamespace(
        replay=lambda _key, fb, **_kwargs: original(
            fb.input_ids, fb.positions, fb, input_embeds=fb.input_embeds
        )
    )
    with torch.no_grad():
        actual = runner._execute_body_capture(
            raw, static, 7, 7, ShapeKey(size=7), get_embedding=True
        )
    torch.testing.assert_close(actual.embeddings, expected.embeddings)
    assert body.forward == original
    if use_embeds:
        torch.testing.assert_close(
            static.input_embeds, wrapper.get_input_embeddings()(raw.input_ids)
        )


def test_vision_layers_are_excluded_and_uncached_mm_is_rejected():
    from sglang.srt.models.transformers.graph import TransformersGraphMixin

    class Owner(TransformersGraphMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.text_config = SimpleNamespace(num_hidden_layers=2)
            self.config = SimpleNamespace(text_config=self.text_config)
            self.model = nn.Module()
            self.model.visual = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.model.language_model = nn.Module()
            layers = []
            for index in range(2):
                layer = nn.Module()
                layer.attn = nn.Identity()
                layer.attn._sglang_attention_key = str(index)
                layers.append(layer)
            self.model.language_model.layers = nn.ModuleList(layers)
            self.start_layer, self.end_layer = 0, 2

    owner = Owner()
    with pytest.raises(ValueError, match="cached feature"):
        owner.get_prefill_graph_model()
    owner._mm_cache_enabled = True
    body = owner.get_prefill_graph_model()
    assert body.layers == list(owner.model.language_model.layers)


def test_auxiliary_hidden_states_survive_graph_replay_boundary(runtime):
    from sglang.srt.models.transformers.speculative import AuxiliaryHiddenStateCapture

    config = config_for("qwen3")
    settings = model_settings(config)
    reference = AutoModel.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper = TransformersEmbeddingModel(copy.deepcopy(config), model_config=settings)
    wrapper.load_weights(reference.state_dict().items())
    wrapper._aux_capture = AuxiliaryHiddenStateCapture(wrapper.model, [0, 1], 2)
    body = wrapper.get_prefill_graph_model()
    batch = packed_batch()
    batch.token_type_ids = None
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    with torch.no_grad(), forward_context(ForwardContext(ReferenceAttentionBackend())):
        hidden, auxiliary = body(batch.input_ids, positions, batch)
    assert len(auxiliary) == 2
    original = body.forward
    body.forward = lambda *_args, **_kwargs: (hidden, auxiliary)
    wrapper._aux_capture.reset()
    try:
        actual = wrapper._run_hf_backbone(batch.input_ids, None, positions, batch)
        torch.testing.assert_close(actual, hidden)
        for actual, expected in zip(wrapper._aux_capture.collect(), auxiliary):
            torch.testing.assert_close(actual, expected)
    finally:
        body.forward = original
        wrapper._aux_capture.close()


def test_pipeline_graph_boundary_uses_proxy_hidden_states():
    class Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_config = SimpleNamespace(num_hidden_layers=2)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.model.layers[1]._sglang_attention_key = "1"
            self.start_layer, self.end_layer = 1, 2
            self.pp_group = SimpleNamespace(is_first_rank=False)

        def _run_hf_backbone_eager(self, input_ids, input_embeds, **_kwargs):
            assert input_ids is None
            return input_embeds + 1

    owner = Owner()
    body = TransformersGraphBody(owner)
    hidden = torch.randn(5, 4)
    result = body(
        None,
        torch.arange(5),
        SimpleNamespace(),
        pp_proxy_tensors={"hidden_states": hidden},
    )
    torch.testing.assert_close(result, hidden + 1)


@pytest.fixture
def cpu_attention_dispatch():
    from sglang.srt.layers.radix_attention import _unified_attention_with_output_impl

    def dispatch(
        query,
        key,
        value,
        output,
        save_kv_cache,
        layer_id,
        use_mha_companion=False,
        **kwargs,
    ):
        return _unified_attention_with_output_impl(
            query,
            key,
            value,
            output,
            save_kv_cache,
            layer_id,
            use_mha_companion,
            False,
            **kwargs,
        )

    library = torch.library.Library("sglang", "IMPL", "CPU")
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "sglang::unified_attention_with_output", "CPU"
    ):
        library.impl("unified_attention_with_output", dispatch)
    try:
        yield
    finally:
        library._destroy()


@pytest.mark.parametrize("kind", ["bert", "qwen3"])
@pytest.mark.parametrize("capture_aux", [False, True])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_piecewise_adapter_fullgraph_cpu_compilation(
    runtime, cpu_attention_dispatch, kind, capture_aux, use_embeds
):
    from test_transformers_window_attention import native_batch

    from sglang.srt.compilation.compile import install_torch_compiled
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        enable_tc_piecewise_cuda_graph,
        set_tc_piecewise_forward_context,
    )
    from sglang.srt.models.transformers.speculative import AuxiliaryHiddenStateCapture

    config = config_for(kind)
    settings = model_settings(config)
    reference = AutoModel.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper = TransformersEmbeddingModel(copy.deepcopy(config), model_config=settings)
    wrapper.load_weights(reference.state_dict().items())
    if capture_aux:
        wrapper._aux_capture = AuxiliaryHiddenStateCapture(wrapper.model, [0, 1], 2)
    body = wrapper.get_prefill_graph_model()
    graphs = []

    def eager_backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    install_torch_compiled(body, backend_factory=eager_backend, fullgraph=True)
    for lengths in ([4, 3], [5, 4]):
        batch, backend = native_batch(lengths)
        batch.positions = torch.cat([torch.arange(length) for length in lengths])
        batch.global_num_token_non_padded_cpu = sum(lengths)
        batch.mha_return_lse = False
        if kind == "bert":
            batch.token_type_ids = torch.arange(sum(lengths)) % 2
        with torch.no_grad(), forward_context(ForwardContext(backend)):
            inputs = {
                "input_ids": None if use_embeds else batch.input_ids,
                "positions": batch.positions,
                "forward_batch": batch,
                "input_embeds": wrapper.get_input_embeddings()(batch.input_ids)
                if use_embeds
                else None,
            }
            expected = body(**inputs)
            reference_hidden, offset = [], 0
            for length in lengths:
                reference_inputs = {
                    "input_ids": batch.input_ids[offset : offset + length][None]
                }
                if batch.token_type_ids is not None:
                    reference_inputs["token_type_ids"] = batch.token_type_ids[
                        offset : offset + length
                    ][None]
                reference_hidden.append(
                    reference(**reference_inputs).last_hidden_state[0]
                )
                offset += length
            torch.testing.assert_close(
                expected[0] if capture_aux else expected, torch.cat(reference_hidden)
            )
            with (
                enable_tc_piecewise_cuda_graph(),
                set_tc_piecewise_forward_context(
                    batch,
                    list(wrapper.attention_instances.values()),
                    None,
                    [],
                    [],
                    num_tokens=sum(lengths),
                    raw_num_tokens=sum(lengths),
                ),
            ):
                actual = body(**inputs)
        if capture_aux:
            torch.testing.assert_close(actual[0], expected[0])
            for actual_aux, expected_aux in zip(actual[1], expected[1]):
                torch.testing.assert_close(actual_aux, expected_aux)
        else:
            torch.testing.assert_close(actual, expected)
    assert len(graphs) == 1
    assert (
        sum(
            "unified_attention_with_output" in str(node.target)
            for node in graphs[0].graph.nodes
        )
        == 2
    )
    if capture_aux:
        wrapper._aux_capture.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
