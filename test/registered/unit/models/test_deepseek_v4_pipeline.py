"""Hyper-connection stage construction and completion across pipeline ranks."""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common import moe_transport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestVPPConstruction(CustomTestCase):
    """Ownership, loader names and endpoint modules must agree across stages."""

    @contextmanager
    def construction(self, rank, size):
        from sglang.srt.layers.layer_boundary import factories
        from sglang.srt.layers.utils import layer_list
        from sglang.srt.models import deepseek_v4

        parallel = SimpleNamespace(
            pp_rank=rank,
            pp_size=size,
            tp_size=1,
            enable_dp_lm_head=False,
            pp_group=SimpleNamespace(
                is_first_rank=rank == 0,
                is_last_rank=rank == size - 1,
                world_size=size,
            ),
        )
        config = SimpleNamespace(
            num_hidden_layers=12,
            hidden_size=4,
            vocab_size=8,
            rms_norm_eps=1e-6,
            hc_eps=1e-6,
            hc_mult=2,
            hc_pre_from_prev_sublayer=True,
            model_type="deepseek_v4",
            tie_word_embeddings=False,
            q_lora_rank=2,
        )

        class Decoder(torch.nn.Module):
            def __init__(self, *, layer_id, prefix, **options):
                super().__init__()
                self.layer_id = layer_id
                self.prefix = prefix
                self.construction_stack = factories._require_stack()
                self.weight = torch.nn.Parameter(torch.full((1,), float(layer_id)))

        with ExitStack() as stack:
            for name, value in (
                ("_is_cuda", False),
                ("_is_hip", False),
                ("_is_npu", False),
                ("_is_xpu", False),
                ("DeepseekV4DecoderLayer", Decoder),
            ):
                stack.enter_context(patch.object(deepseek_v4, name, value))
            for name, value in (
                ("get_parallel", parallel),
                (
                    "get_exec",
                    SimpleNamespace(
                        features=SimpleNamespace(
                            enable_decoder_swa_bounded_replay=False
                        )
                    ),
                ),
                ("is_dp_attention_enabled", False),
                ("is_cross_layer_mhc_fusion_enabled", False),
                ("_is_fused_mhc_post_pre_enabled_xpu", False),
                ("wo_a_fp8_gemm_enabled", False),
                ("use_npu_arch35_mxfp8_wo_a", False),
                ("get_num_fused_shared_experts", 0),
                ("get_attn_tp_context", SimpleNamespace(init_context=Mock())),
            ):
                stack.enter_context(patch.object(deepseek_v4, name, return_value=value))
            stack.enter_context(
                patch.object(layer_list, "get_parallel", return_value=parallel)
            )
            stack.enter_context(
                patch("torch.get_device_module", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch.object(deepseek_v4.EngramLayout, "from_config", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    deepseek_v4.DecoderReplayTail, "maybe_create", return_value=None
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.offloader.get_offloader",
                    return_value=SimpleNamespace(
                        wrap_modules=lambda modules, **kw: list(modules)
                    ),
                )
            )
            for name in (
                "VocabParallelEmbedding",
                "ParallelLMHead",
                "RMSNorm",
                "LogitsProcessor",
            ):
                stack.enter_context(
                    patch.object(
                        deepseek_v4,
                        name,
                        side_effect=lambda *a, **kw: torch.nn.Identity(),
                    )
                )
            yield deepseek_v4, config

    def test_loader_constructs_only_resolved_vpp_layers(self):
        from sglang.srt.model_loader import loader

        for rank, expected in ((0, (0, 3, 4, 5, 7, 8)), (1, (1, 2, 6, 9, 10, 11))):
            with (
                self.subTest(rank=rank),
                self.construction(rank, 2) as (module, config),
            ):
                config.architectures = ["DeepseekV4ForCausalLM"]
                model_config = SimpleNamespace(
                    hf_config=config,
                    num_hidden_layers=12,
                    is_generation=True,
                    is_multimodal=False,
                )
                parallel = module.get_parallel()
                parallel.pp_virtual_stages = 3
                parallel.pp_vpp_layer_partition = [1, 2, 3, 1, 2, 3]
                with (
                    patch.object(loader, "get_parallel", return_value=parallel),
                    patch.object(
                        loader,
                        "get_model_architecture",
                        return_value=(
                            module.DeepseekV4ForCausalLM,
                            "DeepseekV4ForCausalLM",
                        ),
                    ),
                    patch.object(
                        loader, "resolve_shared_experts_config", return_value=None
                    ),
                    patch.object(
                        loader,
                        "get_model",
                        return_value=SimpleNamespace(
                            override=lambda **kw: nullcontext()
                        ),
                    ),
                    patch.object(loader, "check_stage_producers", return_value=None),
                ):
                    model = loader._initialize_model(
                        model_config, SimpleNamespace(draft_model_idx=None)
                    )
                self.assertEqual(model.layer_ids, expected)
                self.assertEqual(
                    tuple(
                        i
                        for i, layer in enumerate(model.model.layers)
                        if hasattr(layer, "weight")
                    ),
                    expected,
                )
                self.assertEqual(model.vpp_layout.virtual_stages, 3)

    def test_interleaved_ownership_preserves_global_weight_names_and_endpoints(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout
        from sglang.srt.layers.utils import PPMissingLayer

        # Unequal stages expose both gaps and accidental local renumbering.
        layout = VPPLayout(12, 2, partition=(1, 2, 4, 5))
        expected = ((0, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10, 11))
        all_names = set()
        for rank in range(2):
            with (
                self.subTest(rank=rank),
                self.construction(rank, 2) as (module, config),
            ):
                model = module.DeepseekV4ForCausalLM(
                    config, prefix="root", vpp_layout=layout
                )
                self.assertEqual(model.layer_ids, expected[rank])
                self.assertIs(model.vpp_layout, layout)
                self.assertEqual(
                    tuple(s.stage_id for s in model.vpp_stages), (rank, rank + 2)
                )
                stacks = []
                for stage in model.vpp_stages:
                    layers = [model.model.layers[i] for i in stage.layer_ids]
                    stacks.append(layers[0].construction_stack)
                    for layer in layers:
                        self.assertIs(layer.construction_stack, stacks[-1])
                self.assertIsNot(stacks[0], stacks[1])
                owned = tuple(
                    i
                    for i, layer in enumerate(model.model.layers)
                    if not isinstance(layer, PPMissingLayer)
                )
                self.assertEqual(owned, expected[rank])
                names = set(model.model.layers.state_dict())
                self.assertEqual(names, {f"{i}.weight" for i in expected[rank]})
                self.assertFalse(all_names & names)
                all_names.update(names)
                for i in owned:
                    self.assertEqual(
                        model.model.layers[i].prefix, f"root.model.layers.{i}"
                    )
                self.assertEqual(
                    isinstance(model.model.embed_tokens, PPMissingLayer), rank != 0
                )
                self.assertEqual(
                    isinstance(model.model.norm, PPMissingLayer), rank != 1
                )
                self.assertEqual(isinstance(model.lm_head, PPMissingLayer), rank != 1)
                with self.assertRaisesRegex(NotImplementedError, "logical-stage"):
                    model.model._forward_impl(None, None, None, None)
        self.assertEqual(all_names, {f"{i}.weight" for i in range(12)})

    def test_absent_layout_keeps_contiguous_pp_ownership(self):
        from sglang.srt.layers.utils import PPMissingLayer

        for size in (1, 2, 3):
            for rank in range(size):
                with (
                    self.subTest(size=size, rank=rank),
                    self.construction(rank, size) as (module, config),
                ):
                    model = module.DeepseekV4ForCausalLM(config)
                    expected = tuple(
                        range(rank * (12 // size), (rank + 1) * (12 // size))
                    )
                    self.assertEqual(model.layer_ids, expected)
                    self.assertIsNone(model.vpp_layout)
                    self.assertEqual(model.vpp_stages, ())
                    self.assertEqual(
                        tuple(
                            i
                            for i, layer in enumerate(model.model.layers)
                            if not isinstance(layer, PPMissingLayer)
                        ),
                        expected,
                    )

    def test_contiguous_pp_does_not_create_whole_tail_graphs(self):
        for rank in (0, 1):
            with (
                self.subTest(rank=rank),
                self.construction(rank, 2) as (module, config),
            ):
                with patch.object(module.DecoderReplayTail, "maybe_create") as create:
                    model = module.DeepseekV4Model(config)
                self.assertIsNone(model.decoder_replay_tail)
                create.assert_not_called()

    def test_incompatible_layout_is_rejected_before_layer_allocation(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout

        with (
            self.construction(0, 2) as (module, config),
            patch.object(module, "DeepseekV4DecoderLayer") as decoder,
        ):
            for layout, message in (
                (VPPLayout(8, 2), "layer count"),
                (VPPLayout(12, 3), "physical size"),
            ):
                with (
                    self.subTest(layout=layout),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    module.DeepseekV4ForCausalLM(config, vpp_layout=layout)
            decoder.assert_not_called()


class TestVPPStageForward(CustomTestCase):
    """Stage cuts must preserve mHC state and run each global layer once."""

    @contextmanager
    def execution(self):
        from sglang.srt.models import deepseek_v4

        context = SimpleNamespace(
            input_scattered=False,
            maybe_input_scattered=lambda batch: nullcontext(),
        )
        parallel = SimpleNamespace(attn_dp_size=1, attn_cp_size=1)
        with ExitStack() as stack:
            for name, value in (
                ("_is_cuda", True),
                ("_is_npu", False),
                ("_is_hip", False),
            ):
                stack.enter_context(patch.object(deepseek_v4, name, value))
            for name, value in (
                ("get_parallel", parallel),
                ("get_forward", SimpleNamespace(sp_active=False)),
                ("get_attn_tp_context", context),
                ("is_cp_active", False),
                ("check_cuda_graph_backend", False),
                ("get_attn_backend", SimpleNamespace()),
                (
                    "get_global_expert_distribution_recorder",
                    SimpleNamespace(with_current_layer=lambda i: nullcontext()),
                ),
            ):
                stack.enter_context(patch.object(deepseek_v4, name, return_value=value))
            stack.enter_context(
                patch(
                    "sglang.srt.layers.moe.mega_gate.mega_gate_forward_scope",
                    side_effect=lambda batch: nullcontext(),
                )
            )
            # The ordinary loop asks for an optional fused post/read shortcut
            # before Engram. Keep the CPU stand-in on the unfused path.
            stack.enter_context(
                patch.object(
                    boundary,
                    "HcRead",
                    return_value=SimpleNamespace(
                        bind=lambda *args: SimpleNamespace(next_boundary=None)
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.kernels.ops.layernorm.mhc.hc_broadcast",
                    side_effect=lambda rows, count: (
                        rows[:, None].expand(-1, count, -1).clone()
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.kernels.ops.layernorm.mhc.mhc_combine",
                    side_effect=lambda rows, pre, count, dtype: (
                        rows.view(rows.shape[0], count, rows.shape[1] // count)
                        * pre[:, :, None]
                    ).sum(1),
                )
            )
            yield deepseek_v4, parallel, context

    def make_model(self, module, layout, rank, lagged, calls):
        from sglang.srt.layers.layer_boundary.layout import SumGroup
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum
        from sglang.srt.layers.utils import PPMissingLayer

        class Layer(torch.nn.Module):
            engram = None
            hc_cfg = SimpleNamespace()

            def __init__(self, index):
                super().__init__()
                self.index = index
                self.reduction = Mock(side_effect=lambda value: value * 2)
                self.fail = False
                if lagged and index == 3:
                    self.engram = Mock(
                        side_effect=lambda streams, hashes, batch, **kw: (
                            streams + hashes[:, None, None]
                        )
                    )
                    self.engram.layer_hash_index = 0

            def forward(self, hidden_states, forward_batch, **kw):
                calls.append(self.index)
                state = residual_batch.read_output(
                    hidden_states, forward_batch, boundary.FinalRead(), None
                )
                residual = state.residual if lagged else state
                # The next layer consumes the preceding layer's pre, so dropping
                # it at a cut changes both this contribution and the final head.
                mixed = (
                    residual[:, 0]
                    if not lagged or state.pre is None
                    else (residual * state.pre[:, :, None]).sum(1)
                )
                value = (mixed + self.index + 1)[:, None].expand_as(residual).clone()
                update = boundary.HcUpdate(self, False)
                if lagged:
                    pre = (
                        torch.tensor([0.25, 0.5]).expand(residual.shape[0], -1).clone()
                    )
                    update.post = lambda output, old, **kw: mhc.HcState(
                        old.residual + output, pre
                    )
                else:
                    update.post = lambda output, old, **kw: old + output
                residual_batch.set_written(state, forward_batch)
                output = residual_batch.stream_of(forward_batch).record(
                    value,
                    update,
                    declared_sum=DeclaredSum(SumGroup.ATTN_TP, self.reduction),
                )
                if self.fail:
                    raise RuntimeError("decoder failed")
                return output

        body = module.DeepseekV4Model.__new__(module.DeepseekV4Model)
        torch.nn.Module.__init__(body)
        body.config = SimpleNamespace(model_type="deepseek_v4")
        body.vpp_layout = layout
        body.vpp_stages = layout.stages_for_rank(rank) if layout else ()
        owned = layout.layer_ids_for_rank(rank) if layout else tuple(range(6))
        body.start_layer, body.end_layer = min(owned), max(owned) + 1
        body.layers = torch.nn.ModuleList(
            Layer(i) if i in owned else PPMissingLayer() for i in range(6)
        )
        body.pp_group = SimpleNamespace(
            is_first_rank=rank == 0, is_last_rank=rank == 1, world_size=2
        )
        if layout is None:
            body.pp_group = SimpleNamespace(
                is_first_rank=True, is_last_rank=True, world_size=1
            )
        body.hc_mult, body.hidden_size = 2, 4
        body.hc_pre_from_prev_sublayer = lagged
        body.dspark_layers_to_capture = body.late_layer_start = None
        body.engram_hasher = (
            Mock(side_effect=lambda ids, batch: ids[:, None]) if lagged else None
        )
        body.use_fused_mhc_post_pre = False
        body._can_run_tbo = Mock(return_value=False)
        body.embed_tokens = Mock(
            side_effect=lambda ids: ids.float()[:, None].expand(-1, 4).clone()
        )
        body.norm = Mock(side_effect=lambda rows: rows * 3)
        body.hc_head = Mock(side_effect=lambda rows, *args: rows.mean(1))
        body.hc_head_fn = body.hc_head_scale = body.hc_head_base = None
        model = module.DeepseekV4ForCausalLM.__new__(module.DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.model, model.pp_group = body, body.pp_group
        model.vision = None
        model.capture_aux_hidden_states = False
        model.prepare_language_model_inputs = Mock(
            side_effect=lambda ids, batch, embeds: (ids, embeds)
        )
        model.lm_head = object()
        model.logits_processor = Mock(
            side_effect=lambda ids, rows, head, metadata, aux, **kw: SimpleNamespace(
                logits=rows.sum(-1), before_norm=kw["hidden_states_before_norm"]
            )
        )
        return model

    def test_interleaved_stage_calls_match_unpartitioned_forward(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        layout = VPPLayout(6, 2, partition=(1, 2, 1, 2))
        for lagged in (False, True):
            for tokens in (0, 3):
                with (
                    self.subTest(lagged=lagged, tokens=tokens),
                    self.execution() as (module, _, _),
                ):
                    calls = []
                    models = [
                        self.make_model(module, layout, r, lagged, calls)
                        for r in range(2)
                    ]
                    ranges = [(m.model.start_layer, m.model.end_layer) for m in models]
                    ids = [torch.arange(tokens), torch.arange(tokens) + 10]
                    embeds = [None, ids[1].float()[:, None].expand(-1, 4).clone() * 2]
                    batches = [
                        SimpleNamespace(
                            residual_stream=None,
                            forward_mode=SimpleNamespace(is_extend=lambda: False),
                        )
                        for _ in ids
                    ]
                    wires = [None, None]
                    for stage in layout.stages:
                        model = models[stage.physical_rank]
                        for j, batch in enumerate(batches):
                            batch.freqs_cis_c4 = batch.freqs_cis_c128 = object()
                            wires[j] = model.forward_vpp_stage(
                                stage.stage_id,
                                ids[j],
                                ids[j],
                                batch,
                                input_embeds=embeds[j] if stage.stage_id == 0 else None,
                                pp_proxy_tensors=wires[j],
                            )
                            self.assertIsNone(batch.residual_stream)
                            self.assertFalse(hasattr(batch, "freqs_cis_c4"))
                            if stage.stage_id < layout.logical_size - 1:
                                self.assertIsInstance(wires[j], PPProxyTensors)
                                self.assertEqual(
                                    set(wires[j].tensors),
                                    {
                                        "hidden_states",
                                        "hc_pre",
                                        "vpp_engram_hash_ids",
                                    }
                                    if lagged
                                    else {"hidden_states"},
                                )
                                if lagged:
                                    torch.testing.assert_close(
                                        wires[j].tensors["vpp_engram_hash_ids"],
                                        ids[j][:, None],
                                    )
                                self.assertTrue(
                                    all(
                                        isinstance(t, torch.Tensor)
                                        for t in wires[j].tensors.values()
                                    )
                                )
                    self.assertEqual(
                        calls,
                        [i for s in layout.stages for _ in ids for i in s.layer_ids],
                    )
                    self.assertEqual(
                        ranges,
                        [(m.model.start_layer, m.model.end_layer) for m in models],
                    )
                    self.assertEqual(models[0].model.embed_tokens.call_count, 1)
                    self.assertEqual(models[1].model.embed_tokens.call_count, 0)
                    self.assertEqual(
                        models[0].prepare_language_model_inputs.call_count, 2
                    )
                    self.assertEqual(
                        models[1].prepare_language_model_inputs.call_count, 0
                    )
                    self.assertEqual(models[0].model.norm.call_count, 0)
                    self.assertEqual(models[1].model.norm.call_count, 2)
                    self.assertEqual(models[0].logits_processor.call_count, 0)
                    self.assertEqual(models[1].logits_processor.call_count, 2)
                    for rank, model in enumerate(models):
                        for i in layout.layer_ids_for_rank(rank):
                            self.assertEqual(
                                model.model.layers[i].reduction.call_count, 2
                            )
                    for j, batch in enumerate(batches):
                        serial = self.make_model(module, None, 0, lagged, [])
                        expected = serial.forward(
                            ids[j], ids[j], batch, input_embeds=embeds[j]
                        )
                        torch.testing.assert_close(wires[j].logits, expected.logits)
                        torch.testing.assert_close(
                            wires[j].before_norm, expected.before_norm
                        )

    def test_contiguous_pp_preserves_lagged_mhc_and_shared_state(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
        from sglang.srt.models import deepseek_v4_vpp_state

        layout = VPPLayout(6, 2, virtual_stages=1)
        for tokens in (0, 3):
            with self.subTest(tokens=tokens), self.execution() as (module, parallel, _):
                calls = []
                models = [
                    self.make_model(module, layout, rank, True, calls)
                    for rank in range(2)
                ]
                for model in models:
                    model.model.config.model_type = "deepseek_v41"
                    model.model.vpp_layout = None
                    model.model.vpp_stages = ()
                ids = torch.arange(tokens)
                batch = SimpleNamespace(
                    residual_stream=None,
                    forward_mode=SimpleNamespace(is_extend=lambda: False),
                )
                shared_kv = torch.arange(tokens * 2).reshape(tokens, 2)

                def export(tensors, **kwargs):
                    tensors["vpp_source_kv"] = shared_kv

                def install(tensors, **kwargs):
                    torch.testing.assert_close(tensors["vpp_source_kv"], shared_kv)

                with (
                    patch.object(
                        module,
                        "get_attn_backend",
                        return_value=SimpleNamespace(forward_metadata=object()),
                    ),
                    patch.object(module, "get_token_to_kv_pool", return_value=object()),
                    patch.object(
                        module,
                        "get_req_to_token_pool",
                        return_value=SimpleNamespace(req_to_token=object()),
                    ),
                    patch.object(
                        deepseek_v4_vpp_state, "export_state", side_effect=export
                    ),
                    patch.object(
                        deepseek_v4_vpp_state, "install_state", side_effect=install
                    ),
                ):
                    parallel.pp_rank = 0
                    wire = models[0].forward(ids, ids, batch)
                    self.assertIsInstance(wire, PPProxyTensors)
                    self.assertEqual(
                        set(wire.tensors),
                        {
                            "hidden_states",
                            "hc_pre",
                            "vpp_engram_hash_ids",
                            "vpp_source_kv",
                        },
                    )
                    parallel.pp_rank = 1
                    result = models[1].forward(ids, ids, batch, pp_proxy_tensors=wire)
                serial = self.make_model(module, None, 0, True, [])
                expected = serial.forward(ids, ids, batch)
                torch.testing.assert_close(result.logits, expected.logits)
                torch.testing.assert_close(result.before_norm, expected.before_norm)
                self.assertEqual(calls, list(range(6)))
                self.assertIsNone(batch.residual_stream)
                models[0].logits_processor.assert_not_called()

    def test_stage_errors_do_not_publish_or_retain_residual_state(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        layout = VPPLayout(6, 2, partition=(1, 2, 1, 2))
        with self.execution() as (module, _, _):
            calls = []
            model = self.make_model(module, layout, 0, True, calls)
            ids = torch.arange(3)
            batch = SimpleNamespace(residual_stream=None)
            with self.assertRaisesRegex(ValueError, "not owned"):
                model.forward_vpp_stage(1, ids, ids, batch)
            with self.assertRaisesRegex(ValueError, "requires a pipeline handoff"):
                model.forward_vpp_stage(2, ids, ids, batch)
            wire = PPProxyTensors({"hidden_states": torch.zeros(3, 8)})
            with self.assertRaisesRegex(ValueError, "requires hc_pre"):
                model.forward_vpp_stage(2, ids, ids, batch, pp_proxy_tensors=wire)
            self.assertEqual(calls, [])
            self.assertIsNone(batch.residual_stream)
            model.model.layers[0].fail = True
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                model.forward_vpp_stage(0, ids, ids, batch)
            self.assertIsNone(batch.residual_stream)
            model.logits_processor.assert_not_called()

    def test_unsupported_execution_paths_fail_before_running_layers(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout

        layout = VPPLayout(6, 2, partition=(1, 2, 1, 2))
        with self.execution() as (module, parallel, context):
            calls = []
            model = self.make_model(module, layout, 0, False, calls)
            ids = torch.arange(3)
            batch = SimpleNamespace(residual_stream=None)
            for target, name, value in (
                (module, "_is_cuda", False),
                (parallel, "attn_dp_size", 2),
                (parallel, "attn_cp_size", 2),
                (context, "input_scattered", True),
                (
                    module,
                    "get_forward",
                    Mock(return_value=SimpleNamespace(sp_active=True)),
                ),
                (model, "capture_aux_hidden_states", True),
                (model, "vision", object()),
                (model.model, "dspark_layers_to_capture", [0]),
                (model.model, "decoder_replay_tail", object()),
                (model.model, "_can_run_tbo", Mock(return_value=True)),
            ):
                with self.subTest(field=name), patch.object(target, name, value):
                    with self.assertRaises(NotImplementedError):
                        model.forward_vpp_stage(0, ids, ids, batch)
                    self.assertEqual(calls, [])
                    self.assertIsNone(batch.residual_stream)
            model.logits_processor.assert_not_called()


class TestContiguousPPValidation(CustomTestCase):
    def test_eager_text_prefill_only(self):
        from sglang.srt.arg_groups import deepseek_v4_hook
        from sglang.srt.model_executor.cuda_graph_config import Backend

        cfg = SimpleNamespace(
            pp_size=2,
            pp_virtual_stages=1,
            disaggregation_mode="prefill",
            language_model_only=True,
            attn_dp_size=1,
            attn_cp_size=1,
            enable_prefill_cp=False,
            speculative_algorithm=None,
            enable_layernorm_sp=False,
            enable_attn_tp_input_scattered=False,
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=True,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED),
                decode=SimpleNamespace(backend=Backend.DISABLED),
            ),
            enable_return_hidden_states=False,
            enable_return_aux_hidden_states=False,
            enable_return_routed_experts=False,
            enable_torch_compile=False,
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            dsv4_attn_backend="auto",
        )
        platform = SimpleNamespace(is_cuda=True)
        with (
            patch.object(deepseek_v4_hook, "resolving_view", return_value=cfg),
            patch.object(
                deepseek_v4_hook,
                "model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(model_type="deepseek_v41")
                ),
            ),
            patch.object(deepseek_v4_hook, "get_platform", return_value=platform),
            patch.object(deepseek_v4_hook, "attn_dp_enabled_of", return_value=False),
            patch(
                "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            deepseek_v4_hook.validate_deepseek_v41_features(object())
            for target, name, value, feature in (
                (cfg, "disaggregation_mode", "decode", "outside PD prefill"),
                (cfg, "language_model_only", False, "vision"),
                (platform, "is_cuda", False, "non-CUDA"),
                (cfg, "attn_dp_size", 2, "attention DP"),
                (cfg, "attn_cp_size", 2, "attention CP"),
                (cfg, "speculative_algorithm", "DSPARK", "speculative"),
                (cfg.cuda_graph_config.prefill, "backend", Backend.FULL, "CUDA graphs"),
                (cfg.cuda_graph_config.decode, "backend", Backend.FULL, "CUDA graphs"),
            ):
                with (
                    self.subTest(feature=feature, field=name),
                    patch.object(target, name, value),
                    self.assertRaisesRegex(ValueError, feature),
                ):
                    deepseek_v4_hook.validate_deepseek_v41_features(object())


class TestPipelineCompletion(unittest.TestCase):
    def test_real_stages_write_mhc_once_before_pipeline_handoff(self):
        from sglang.srt.layers.layer_boundary import append_stages, layer_stack
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.models import deepseek_v4
        from sglang.srt.models.deepseek_v2 import MoEOutput
        from sglang.test import boundary_fixtures as fixture

        for lagged in (False, True):
            for fused in (False, True):
                with self.subTest(lagged=lagged, fused=fused):
                    config = SimpleNamespace(
                        hc_pre_from_prev_sublayer=lagged,
                        model_type="deepseek_v41" if lagged else "deepseek_v4",
                    )
                    layer = SimpleNamespace(
                        config=config,
                        boundary_pre_from_prev=lagged,
                        boundary_mhc_fusions=fused,
                        use_fused_mhc_post_pre=False,
                        _can_fuse_ffn_mhc=fused,
                        _can_fuse_attn_mhc=False,
                        hc_cfg=SimpleNamespace(hidden=4),
                        mlp=SimpleNamespace(tp_size=2),
                        ffn_hc=object(),
                        attn_hc=object(),
                        self_attn=SimpleNamespace(
                            attn_tp_size=2, accepts_mxfp8_swizzled_input=lambda: False
                        ),
                        engram=None,
                        input_layernorm=fixture.Norm(),
                        post_attention_layernorm=fixture.Norm(),
                        hc_post=Mock(
                            side_effect=lambda value, streams, post, comb: (
                                streams + value[:, None]
                            )
                        ),
                    )
                    layer._bind_attention_transport = lambda *args: (
                        deepseek_v4.DeepseekV4CudaDecoderLayer._bind_attention_transport(
                            layer, *args
                        )
                    )
                    layer._bind_moe_transport = lambda *args: (
                        deepseek_v4.DeepseekV4CudaDecoderLayer._bind_moe_transport(
                            layer, *args
                        )
                    )
                    parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=2)
                    source = torch.arange(24.0).reshape(3, 2, 4)
                    batch = SimpleNamespace(
                        input_ids=torch.arange(3),
                        forward_mode=ForwardMode.EXTEND,
                        attn_cp_metadata=None,
                        residual_stream=ResidualStream(source),
                    )
                    call = boundary.LayerCall()
                    sums = Mock(side_effect=lambda value: value * 2)
                    attn_sums = Mock(side_effect=lambda value: value * 2)
                    attn_group = SimpleNamespace(all_reduce=attn_sums)
                    post = Mock(
                        side_effect=lambda hc, value, streams, **kw: (
                            streams + value[:, None]
                        )
                    )
                    with (
                        fixture.planning(parallel),
                        patch(
                            "sglang.srt.models.deepseek_v4_boundary.get_moe_a2a_backend",
                            return_value=SimpleNamespace(is_none=lambda: True),
                        ),
                        patch.object(
                            moe_transport,
                            "get_moe_a2a_backend",
                            return_value=SimpleNamespace(is_none=lambda: True),
                        ),
                        patch.object(boundary, "get_parallel", return_value=parallel),
                        patch.object(
                            moe_transport, "get_parallel", return_value=parallel
                        ),
                        patch("sglang.srt.layers.moe.post_experts_all_reduce", sums),
                        patch.object(mhc, "run_moe_post", post),
                        patch.object(mhc, "run_attn_post", post),
                        patch.object(mhc, "make_boundary", return_value=None),
                        patch.object(
                            boundary,
                            "_sum_group",
                            return_value=attn_group,
                        ),
                        patch(
                            "sglang.srt.layers.layer_boundary.row_transport._sum_group",
                            return_value=attn_group,
                        ),
                        patch(
                            "sglang.srt.models.deepseek_v4_boundary.get_cp_decode_attn_tp_ctx",
                            return_value=SimpleNamespace(use_decode_attn_tp=False),
                        ),
                        patch.object(
                            deepseek_v4,
                            "is_cross_layer_mhc_fusion_enabled",
                            return_value=fused,
                        ),
                        patch.object(
                            deepseek_v4,
                            "_is_fused_mhc_post_pre_enabled_xpu",
                            return_value=False,
                        ),
                        patch(
                            "sglang.srt.models.deepseek_v4_boundary.can_fuse_all_reduce",
                            return_value=True,
                        ),
                        patch.object(
                            boundary.HcReadCall,
                            "read",
                            side_effect=lambda value, norm, quant_format="", **kw: (
                                value.mean(1),
                                boundary.ReadState(value, call=call),
                            ),
                        ),
                    ):
                        with layer_stack(
                            next_layers=[lambda: boundary.stages(config, 1)]
                        ):
                            declarations = boundary.stages(config, 0, layer=layer)
                            attn, ffn = append_stages(
                                *((d, fixture.Norm()) for d in declarations)
                            )
                        operand = attn.prepare(source, batch, read_context=call)
                        attended = attn.finish(
                            mhc.AttnOutput(torch.ones_like(operand)), batch
                        )
                        operand = ffn.prepare(attended, batch, read_context=call)
                        with ffn.exit(batch) as exit:
                            pieces = MoEOutput(
                                torch.ones_like(operand),
                                None,
                                SimpleNamespace(quant_method=None),
                                1.0,
                                False,
                            )
                            output = exit.finish(pieces)
                        self.assertIsNone(batch.residual_stream.pending)
                        self.assertIsNone(batch.residual_stream.transport)
                        wire = residual_batch.to_pp(output, batch)
                        with layer_stack(
                            previous_layers=[lambda: boundary.stages(config, 0)],
                            final_read=boundary.FinalRead(),
                        ):
                            receiver, _ = append_stages(
                                *(
                                    (d, fixture.Norm())
                                    for d in boundary.stages(config, 1, layer=layer)
                                )
                            )
                        received_batch = SimpleNamespace(**vars(batch))
                        received = receiver.from_pp(wire, received_batch)
                        read = receiver.prepare(
                            received, received_batch, read_context=call
                        )
                        torch.testing.assert_close(read, (source + 4).mean(1))
                        # Receiving an already-completed output must not reduce it again.
                        self.assertEqual(attn_sums.call_count, 1)
                        residual_batch.clear(received_batch)
                    torch.testing.assert_close(output, source + 4)
                    torch.testing.assert_close(wire["hidden_states"], source + 4)
                    sums.assert_called_once()
                    self.assertEqual((post if lagged else layer.hc_post).call_count, 2)
                    self.assertIsNone(batch.residual_stream)


if __name__ == "__main__":
    unittest.main()
