"""Native Direct construction and control, with symbolic device/model hooks."""

import functools
import types
from unittest.mock import patch

from ...contracts import ExecutionTrace
from .event_backend import make_event_backend
from .native_hooks import native_session


def decode_batch(tensor, kv):
    """Shape-only ForwardBatch inputs; native code owns slicing and stage ranges."""
    fields = (
        "out_cache_loc",
        "out_cache_loc_virtual",
        "origin_out_cache_loc",
        "kv_loc_plan",
        "input_embeds",
        "token_type_ids",
        "req_pool_indices",
        "req_pool_indices_cpu",
        "seq_lens",
        "seq_lens_cpu",
        "extend_seq_lens",
        "extend_prefix_lens",
        "extend_seq_lens_cpu",
        "extend_prefix_lens_cpu",
        "extend_logprob_start_lens_cpu",
        "lora_ids",
        "global_num_token_non_padded",
        "num_token_non_padded",
    )
    batch = types.SimpleNamespace(**dict.fromkeys(fields))
    batch.input_ids = tensor
    batch.positions = None
    batch.batch_size = len(tensor)
    batch.global_num_token_non_padded_cpu = len(tensor)
    batch.spec_info = None
    batch.rids = list(range(len(tensor)))
    batch.forward_mode = types.SimpleNamespace(is_decode_or_idle=lambda: True)
    batch.effective_kv = kv
    return batch


def execute_native(root, provider, sources):
    runtime = make_event_backend(provider)
    mapping = provider.mapping
    layout, geometry = mapping.layout, mapping.geometry
    actors_a = []
    placement = [None] * layout.batch_size
    with native_session(root, runtime) as native:
        config_fields = dict(provider.config.get("native_config", {}))
        config_fields.setdefault("attention_lanes", layout.attention_ranks)
        config_fields.setdefault("lanes", layout.ffn_ranks)
        config_fields.setdefault("stages", layout.stages)
        config_fields.setdefault("experimental_direct_stage_rows", layout.stage_rows)
        config_fields.setdefault("ffn_backend", "direct")
        config = native.config.AFDConfig(**config_fields)
        exchange_cls = native.expert_exchange.AFDDirectExchange
        pipeline_cls = native.pipeline.AFDAttentionPipeline
        model_adapter_cls = native.model_adapters_qwen3_moe.Qwen3AFDAdapter
        role_enum = native.contracts.AFDRole

        def binding(exchange, *, stage, direction, config):
            stream = runtime.Stream(
                runtime.current[0], stage, 0 if direction == "dispatch" else 1
            )
            return runtime.Binding(stream)

        original_dispatch = exchange_cls.dispatch
        original_run_layer = exchange_cls.run_layer

        def dispatch(exchange, **kwargs):
            # Hook the numerical pack boundary; native dispatch retains all
            # buffer selection, zero/copy, peer loops and exchange ordering.
            exchange.buffers[kwargs["stage"]][1].label = "packed_hidden"
            return original_dispatch(exchange, **kwargs)

        def run_layer(exchange, *, layer, stage):
            runtime.current[0].position = layer, stage
            return original_run_layer(exchange, layer=layer, stage=stage)

        def make_model(actor):
            def attention(index, **kwargs):
                actor.position = index, kwargs["forward_batch"].sim_stage
                actor.compute("attention_module", rows=len(kwargs["hidden_states"]))
                return kwargs["hidden_states"], kwargs["residual"]

            def route(hidden):
                actor.compute("router", rows=len(hidden))
                actor.compute("topk", rows=len(hidden))
                return types.SimpleNamespace(
                    topk_ids=runtime.Tensor((len(hidden), geometry.topk), "I64"),
                    topk_weights=runtime.Tensor((len(hidden), geometry.topk), "F32"),
                )

            class Experts:
                moe_tp_size = 1
                moe_ep_size = layout.ffn_ranks
                num_local_experts = geometry.experts // layout.ffn_ranks
                reduce_results = False

                def __call__(self, hidden, routing):
                    actor.compute(
                        "routed_experts_local_combine",
                        rows=len(hidden),
                        ep=self.moe_ep_size,
                        local_experts=self.num_local_experts,
                    )
                    return hidden

            model_config = types.SimpleNamespace(
                hidden_size=geometry.hidden,
                num_hidden_layers=geometry.layers,
                num_experts=geometry.experts,
                num_experts_per_tok=geometry.topk,
            )
            layers = [
                types.SimpleNamespace(
                    forward_attention_for_afd=functools.partial(attention, index),
                    mlp=types.SimpleNamespace(
                        route=route, experts=Experts(), shared_experts=None
                    ),
                    ffn_boundary=types.SimpleNamespace(
                        complete_now=lambda value, batch: value
                    ),
                )
                for index in range(geometry.layers)
            ]
            return types.SimpleNamespace(
                model=types.SimpleNamespace(config=model_config, layers=layers)
            )

        # Only the hardware communicator binding and numerical pack boundary
        # are hooked. The actual exchange constructor and methods execute.
        with (
            patch.object(exchange_cls, "_make_binding", binding),
            patch.object(exchange_cls, "dispatch", dispatch),
            patch.object(exchange_cls, "run_layer", run_layer),
        ):
            for role, count in (
                (role_enum.ATTENTION, config.attention_lane_count),
                (role_enum.FFN, config.lanes),
            ):
                for rank in range(count):
                    actor = runtime.Actor(
                        "A" if role == role_enum.ATTENTION else "F", rank
                    )
                    runtime.current[0] = actor
                    actor.compute("graph_replay_overhead")
                    if role == role_enum.ATTENTION:
                        actor.compute(
                            "embedding", rows=layout.stage_rows * config.stages
                        )
                    transport = runtime.Transport(actor)
                    transport.role, transport._device = role, "cpu-symbolic"
                    transport.lane = rank
                    model = make_model(actor)
                    attention_backend = types.SimpleNamespace(
                        init_forward_metadata=lambda batch: None, forward_metadata={}
                    )
                    adapter = model_adapter_cls(
                        role=role, model=model, attention_backend=attention_backend
                    )
                    exchange = exchange_cls(
                        transport=transport,
                        adapter=adapter,
                        config=config,
                        device="cpu-symbolic",
                    )
                    transport.ffn_backend = exchange
                    if role == role_enum.FFN:
                        exchange.run_ffn_step()
                        continue
                    hidden = runtime.Tensor(
                        (layout.stage_rows * config.stages, geometry.hidden)
                    )
                    batch = decode_batch(
                        runtime.Tensor((len(hidden),), "I64"),
                        provider.config["effective_kv"],
                    )
                    stages = adapter.split_step(
                        hidden_states=hidden,
                        residual=None,
                        positions=None,
                        forward_batch=batch,
                        stages=config.stages,
                    )
                    for stage in stages:
                        stage.forward_batch.sim_stage = stage.index
                        for slot in range(stage.request_stop - stage.request_start):
                            index = rank * len(hidden) + stage.request_start + slot
                            placement[index] = dict(
                                attention_rank=rank, stage=stage.index, slot=slot
                            )
                    shape = types.SimpleNamespace(
                        lane_bucket_rows=tuple(
                            tuple(stage.rows for stage in stages)
                            for _ in range(config.attention_lane_count)
                        ),
                        lane_stage_rows=tuple(
                            tuple(stage.rows for stage in stages)
                            for _ in range(config.attention_lane_count)
                        ),
                    )
                    exchange.prepare_step(shape=shape)
                    pipeline = pipeline_cls(
                        adapter=adapter,
                        connector=types.SimpleNamespace(transport=transport),
                        config=config,
                        shape_factory=None,
                    )
                    pipeline._run_layers(
                        stages=stages,
                        recv_buffers=[
                            runtime.Tensor((stage.rows, geometry.hidden))
                            for stage in stages
                        ],
                        participates=exchange.active_stages,
                        uses_backend=True,
                    )
                    # The native adapter owns stage joining. Pricing this model
                    # boundary remains explicit, as do output-head costs below.
                    actor.position = geometry.layers, -1
                    actor.compute("stage_join", rows=len(hidden))
                    adapter.join_step(stages=stages)
                    actors_a.append(actor)

        for actor in actors_a:
            actor.compute("final_norm", rows=layout.stage_rows * config.stages)

        def group(kind):
            return runtime.clock.submit(
                "A-output",
                kind,
                provider.quote(
                    kind,
                    world=config.attention_lane_count,
                    global_rows=layout.batch_size,
                ),
                deps=tuple(actor.last for actor in actors_a),
                resources=tuple(actor.name + ".compute" for actor in actors_a),
                metadata=dict(
                    world=config.attention_lane_count, global_rows=layout.batch_size
                ),
            )

        gather = group("lm_hidden_gather")
        for actor in actors_a:
            actor.last = gather
            actor.compute("lm_head", rows=layout.batch_size)
        exchange = group("logits_exchange")
        for actor in actors_a:
            actor.last = exchange
            actor.compute("logits_postprocess", rows=layout.stage_rows * config.stages)
            actor.compute("sampling", rows=layout.stage_rows * config.stages)
        if any(len(wire["posts"]) != 2 for wire in runtime.wires.values()):
            raise ValueError("UNPAIRED_DIRECT_MESSAGE")
        fence = runtime.clock.submit(
            "step",
            "decode_complete",
            0,
            deps=tuple(actor.last for actor in runtime.actors),
        )
        runtime.clock.finish(fence)
    return ExecutionTrace(
        runtime.clock.export(),
        sources,
        layout.attention_ranks + layout.ffn_ranks,
        placement,
    )
