import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.configs.model_config import AttentionArch, ModelConfig, is_dspark_draft
from sglang.srt.configs.spark2_5 import Spark2_5Config, configure_spark25_dspark
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.routed_cache import RoutedKVCache
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_WINDOW,
    PlacementContext,
    draft_kv_profile,
    place_fused_draft,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_draft_pool import bind_fused_draft
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.pool_configurator import HybridSWAPoolConfigurator
from sglang.srt.models.spark2_5 import Spark2_5ForCausalLM
from sglang.srt.models.spark2_5_dspark import DSparkSpark25DraftModel
from sglang.srt.runtime_context import (
    get_context,
    get_memory,
    get_parallel,
    get_schedule,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import arena_of, plan_stand_in

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _config():
    return Spark2_5Config(
        architectures=["Spark2_5ForCausalLM"],
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=4,
        num_draft_layers=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=32,
        sliding_window=3,
        dspark_block_size=3,
        markov_rank=4,
        markov_head_type="vanilla",
        mask_token_id=2,
        target_layer_ids=[0, 3],
        rope_parameters={
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.5},
        },
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        tie_word_embeddings=True,
    )


def _initialize(model):
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            values = torch.arange(param.numel()).reshape(param.shape)
            param.copy_(
                torch.sin(values + index) * 0.2
                if param.ndim > 1
                else torch.ones_like(param)
            )


def _runtime_config(config, *, is_draft):
    mc = object.__new__(ModelConfig)
    mc.hf_config = mc.hf_text_config = config
    mc.is_draft_model = is_draft
    mc.context_len = config.max_position_embeddings
    mc.disable_hybrid_swa_memory = False
    mc.attention_arch = AttentionArch.MHA
    mc.head_dim = mc.v_head_dim = mc.swa_head_dim = mc.swa_v_head_dim = config.head_dim
    mc.sliding_window_size = config.sliding_window - 1
    mc.num_nextn_predict_layers = None
    mc._derive_hybrid_model()
    return mc


def _fused_pool(draft, *, page_size, target_window):
    profile = draft_kv_profile(
        _runtime_config(draft.config, is_draft=True),
        num_layers=2,
        attn_tp_size=1,
    )
    assert set(profile.kinds) == {LAYER_WINDOW}
    placement = place_fused_draft(
        profile=profile,
        num_runners=1,
        ctx=PlacementContext(host_names=("full", "swa"), target_window=target_window),
        store_dtype=torch.float32,
    ).placement
    assert placement.hosts() == (("swa",) if target_window >= 2 else ("full",))
    specs = [
        MHASubPoolSpec(
            name=name,
            layer_num=1,
            head_num=1,
            head_dim=4,
            store_dtype=torch.float32,
            grow_direction=direction,
            draft_region=placement.region(name),
        )
        for name, direction in (("full", "up"), ("swa", "down"))
    ]
    arena = arena_of(
        total_bytes=64 * page_size * sum(s.entry_bytes() for s in specs),
        sub_pool_specs=specs,
        page_size=page_size,
        device="cpu",
        fused_draft=placement,
    )
    views = {
        s.name: MHAView(arena=arena, spec=s, page_size=page_size, model_layer_ids=(i,))
        for i, s in enumerate(specs)
    }
    target_pool = RoutedKVCache(
        arena=arena,
        owner="full",
        layer_maps=[("full", views["full"], {0: 0}), ("swa", views["swa"], {1: 0})],
        page_size=page_size,
        start_layer=0,
        end_layer=1,
        size=32 * page_size,
        head_num=1,
        head_dim=4,
    )
    allocator = ChainAllocator(
        roles=ChainRoles(owner="full", window="swa"),
        arena=arena,
        kvcache=target_pool,
        views=views,
        device="cpu",
        page_size=page_size,
        full_max_total_num_tokens=32 * page_size,
        swa_max_total_num_tokens=16 * page_size,
    )
    pool = bind_fused_draft(
        placement=placement,
        runner=0,
        arena=arena,
        plan=plan_stand_in(arena, placement),
        owner="full",
        target_state=None,
        model=draft,
        model_config=_runtime_config(draft.config, is_draft=True),
        req_to_token_pool=None,
        page_size=page_size,
    ).token_to_kv_pool
    return pool, allocator


def _swa_pool(draft, *, unified, page_size, target_window=2):
    virtual = torch.tensor([2, 3, 4, 5]) * page_size
    full_loc = torch.tensor([8, 6, 4, 2]) * page_size
    swa_loc = torch.tensor([1, 3, 5, 7]) * page_size
    offsets = torch.tensor([page_size - 1, 0, page_size - 1, 0])
    allocator = None
    if unified:
        pool, allocator = _fused_pool(
            draft, page_size=page_size, target_window=target_window
        )
        allocator.full_attn_allocator.ids.bind(
            virtual // page_size, full_loc // page_size
        )
        allocator.swa_attn_allocator.ids.bind(
            virtual // page_size, swa_loc // page_size
        )
        if target_window < 2:
            swa_loc = full_loc
    else:
        pool = SWAKVPool(
            size=32 * page_size,
            size_swa=16 * page_size,
            page_size=page_size,
            dtype=torch.float32,
            head_num=1,
            head_dim=4,
            swa_attention_layer_ids=[0, 1],
            full_attention_layer_ids=[],
            device="cpu",
        )
        mapping = torch.zeros(33 * page_size, dtype=torch.int64)
        for offset in range(page_size):
            mapping[full_loc + offset] = swa_loc + offset
        pool.register_mapping(mapping)
        virtual = full_loc
    reader = RunnerIds(
        req_to_token=(virtual + offsets).view(2, 2),
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool,
        page_size=page_size,
        device="cpu",
        is_draft_worker=True,
    )
    return pool, reader, full_loc + offsets, swa_loc + offsets


def _rope(x, positions, theta, rotary_dim):
    freqs = theta ** (-torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    angles = positions[:, None, None] * freqs
    left, right = x[..., :rotary_dim].chunk(2, dim=-1)
    return torch.cat(
        [
            left * angles.cos() - right * angles.sin(),
            right * angles.cos() + left * angles.sin(),
            x[..., rotary_dim:],
        ],
        dim=-1,
    )


class _SDPAttention:
    def forward(self, q, k, v, layer, batch, save_kv_cache, **kwargs):
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim).transpose(0, 1)
        k = k.repeat_interleave(layer.tp_q_head_num // layer.tp_k_head_num, dim=1)
        v = v.repeat_interleave(layer.tp_q_head_num // layer.tp_v_head_num, dim=1)
        offsets = torch.arange(q.shape[1])
        mask = offsets[None, :] >= offsets[:, None] - layer.sliding_window_size
        if layer.sliding_window_size < 0:
            mask = torch.ones_like(mask)
        if layer.attn_type == AttentionType.DECODER:
            mask &= offsets[None, :] <= offsets[:, None]
        out = F.scaled_dot_product_attention(
            q, k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=layer.scaling
        )
        return out.transpose(0, 1).reshape(q.shape[1], -1)


def _reference_layer(hidden, layer, rope_params, positions, batch):
    normalized = F.rms_norm(hidden, (8,), layer.input_layernorm.weight, eps=1e-6)
    attn = layer.self_attn
    qkv = F.linear(normalized, attn.q_k_v_proj.weight)
    q, k, v = qkv[:, :8], qkv[:, 8:12], qkv[:, 12:]
    rotary_dim = int(4 * rope_params["partial_rotary_factor"])
    q = _rope(q.view(-1, 2, 4), positions, rope_params["rope_theta"], rotary_dim)
    k = _rope(k.view(-1, 1, 4), positions, rope_params["rope_theta"], rotary_dim)
    output = _SDPAttention().forward(q, k, v.view(-1, 1, 4), attn.attn, batch, False)
    gate = F.linear(normalized, attn.g_proj.weight).sigmoid()
    output = (output.view(-1, 2, 4) * gate[..., None]).reshape(-1, 8)
    hidden = hidden + F.linear(output, attn.out_proj.weight)
    normalized = F.rms_norm(
        hidden, (8,), layer.post_attention_layernorm.weight, eps=1e-6
    )
    gate, up = F.linear(normalized, layer.mlp.gate_up_proj.weight).chunk(2, dim=-1)
    return hidden + F.linear(
        F.gelu(gate, approximate="none") * up, layer.mlp.down_proj.weight
    )


class TestSpark25DSpark(CustomTestCase):
    def setUp(self):
        previous_backend = get_fused_op_backend()
        self.addCleanup(set_fused_op_backend, previous_backend)
        set_fused_op_backend(KernelBackend.TORCH)
        self.enterContext(envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"))
        self.enterContext(get_context().override_server_args())
        self.enterContext(
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                tp_group=None,
                attn_dcp_size=1,
                attn_dcp_rank=0,
                pp_size=1,
                pp_rank=0,
                pp_group=SimpleNamespace(
                    world_size=1, rank=0, is_first_rank=True, is_last_rank=True
                ),
            )
        )
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )

    def test_bundled_config_keeps_target_and_draft_attention_independent(self):
        target = _config()
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = model_config.hf_text_config = target
        model_config.is_draft_model = True
        model_config.speculative_algorithm = "DSPARK"
        model_config._config_draft_model()
        draft = model_config.hf_config
        parsed = parse_dspark_draft_config(draft_hf_config=draft)
        self.assertTrue(is_dspark_draft(draft))
        self.assertEqual((parsed.num_hidden_layers, parsed.num_target_layers), (2, 4))
        self.assertEqual((parsed.gamma, parsed.target_layer_ids), (3, [0, 3]))
        self.assertEqual(draft.layer_types, ["sliding_attention"] * 2)
        self.assertTrue(draft.is_hybrid_swa)
        self.assertEqual(target.layer_types[-1], "full_attention")
        self.assertEqual(target.num_hidden_layers, 4)
        self.assertEqual(target.architectures, ["Spark2_5ForCausalLM"])

    def test_swa_context_commits_use_the_same_ids_as_window_reads(self):
        draft = DSparkSpark25DraftModel(config=configure_spark25_dspark(_config()))
        _initialize(draft)
        hidden = torch.linspace(-2, 3, 32).reshape(4, 8)
        positions = torch.tensor([1, 3, 4, 9])
        for unified, target_window in ((False, 2), (True, 2), (True, 1)):
            for page_size in (1, 2):
                pool, reader, full_loc, swa_loc = _swa_pool(
                    draft,
                    unified=unified,
                    page_size=page_size,
                    target_window=target_window,
                )
                injector = TargetHiddenKvInjector(
                    draft_model=draft,
                    draft_model_runner=SimpleNamespace(
                        token_to_kv_pool=pool, runner_ids=reader
                    ),
                    model_runner=SimpleNamespace(device="cpu"),
                    device="cpu",
                    verify_num_draft_tokens=2,
                    block_pos_offsets=torch.arange(2),
                )
                torch.testing.assert_close(
                    reader.ids_in(reader.window_kv_kind, reader.req_to_token.flatten()),
                    swa_loc,
                )
                for commits in (None, torch.tensor([1, 1]), torch.tensor([0, 0])):
                    with self.subTest(
                        unified=unified,
                        target_window=target_window,
                        page_size=page_size,
                        commits=commits,
                    ):
                        for layer_id in range(2):
                            pool.get_key_buffer(layer_id).zero_()
                            pool.get_value_buffer(layer_id).zero_()
                        locations = full_loc.view(2, 2).clone()
                        indices = torch.arange(4)
                        if commits is not None:
                            valid = (
                                torch.arange(2)[None, :] < commits[:, None]
                            ).flatten()
                            locations.flatten()[~valid] = 999999
                            indices = valid.nonzero().flatten()
                        injector.inject_target_hidden(
                            target_hidden=hidden,
                            target_hidden_is_projected=True,
                            cache_loc=full_loc,
                            positions=positions,
                            cache_loc_2d=locations if commits is not None else None,
                            commit_lens=commits,
                        )
                        for layer_id, layer in enumerate(draft.layers):
                            qkv = F.linear(hidden, layer.self_attn.q_k_v_proj.weight)
                            k = _rope(qkv[:, 8:12].view(-1, 1, 4), positions, 10000, 4)
                            v = qkv[:, 12:].view(-1, 1, 4)
                            expected_k = torch.zeros_like(pool.get_key_buffer(layer_id))
                            expected_v = torch.zeros_like(
                                pool.get_value_buffer(layer_id)
                            )
                            expected_k[swa_loc[indices]] = k[indices]
                            expected_v[swa_loc[indices]] = v[indices]
                            torch.testing.assert_close(
                                pool.get_key_buffer(layer_id), expected_k
                            )
                            torch.testing.assert_close(
                                pool.get_value_buffer(layer_id), expected_v
                            )

    def test_prefill_and_ragged_verify_inject_into_swa(self):
        draft = DSparkSpark25DraftModel(config=configure_spark25_dspark(_config()))
        _initialize(draft)
        hidden = torch.linspace(-2, 3, 64).reshape(4, 16)
        for unified in (False, True):
            pool, reader, _, swa_loc = _swa_pool(draft, unified=unified, page_size=2)
            injector = TargetHiddenKvInjector(
                draft_model=draft,
                draft_model_runner=SimpleNamespace(
                    token_to_kv_pool=pool, runner_ids=reader
                ),
                model_runner=SimpleNamespace(
                    device="cpu",
                    runner_ids=reader,
                    prefill_attention_backend_str="triton",
                ),
                device="cpu",
                verify_num_draft_tokens=2,
                block_pos_offsets=torch.arange(2),
            )
            batch = SimpleNamespace(
                extend_lens=[2, 2],
                prefix_lens=[0, 0],
                out_cache_loc=reader.req_to_token.flatten(),
                req_pool_indices=torch.arange(2),
                seq_lens=torch.tensor([2, 2]),
                seq_lens_cpu=torch.tensor([2, 2]),
                forward_mode=ForwardMode.EXTEND,
                spec_info=None,
                batch_size=2,
            )
            plan = reader.own_plan(batch)
            for ragged in (False, True):
                for layer_id in range(2):
                    pool.get_key_buffer(layer_id).zero_()
                    pool.get_value_buffer(layer_id).zero_()
                if ragged:
                    batch.seq_lens = torch.tensor([10, 20])
                    positions = torch.tensor([10, 11, 20, 21])
                    indices = torch.tensor([0])
                    injector.inject_ragged(
                        batch=batch,
                        layout=None,
                        hidden_strided=hidden,
                        commit_lens=torch.tensor([1, 0]),
                        bs=2,
                        iteration_ids=plan,
                    )
                else:
                    placement = injector.place_prefill(batch)
                    positions = torch.tensor([0, 1, 0, 1])
                    indices = torch.tensor([0, 2])
                    mapping = (
                        reader.space(reader.window_kv_kind).table
                        if unified
                        else pool.full_to_swa_index_mapping
                    )
                    saved_mapping = mapping.clone()
                    mapping.zero_()
                    try:
                        injector.inject_prefill(
                            placement,
                            target_hidden=hidden[indices],
                            token_indices=indices,
                            target_hidden_is_projected=False,
                        )
                    finally:
                        mapping.copy_(saved_mapping)
                projected = draft.project_target_hidden(hidden)
                for layer_id, layer in enumerate(draft.layers):
                    qkv = F.linear(projected, layer.self_attn.q_k_v_proj.weight)
                    expected_k = torch.zeros_like(pool.get_key_buffer(layer_id))
                    expected_v = torch.zeros_like(pool.get_value_buffer(layer_id))
                    expected_k[swa_loc[indices]] = _rope(
                        qkv[:, 8:12].view(-1, 1, 4), positions, 10000, 4
                    )[indices]
                    expected_v[swa_loc[indices]] = qkv[:, 12:].view(-1, 1, 4)[indices]
                    torch.testing.assert_close(
                        pool.get_key_buffer(layer_id), expected_k
                    )
                    torch.testing.assert_close(
                        pool.get_value_buffer(layer_id), expected_v
                    )

    def test_private_swa_draft_budget_scales_with_swa_capacity(self):
        target = _runtime_config(_config(), is_draft=False)
        draft = _runtime_config(configure_spark25_dspark(_config()), is_draft=True)
        self.enterContext(get_memory().override(enable_unified_memory=False))
        self.enterContext(get_schedule().override(swa_full_tokens_ratio=0.25))
        configurator = HybridSWAPoolConfigurator(
            SimpleNamespace(
                model_config=target,
                kv_cache_dtype=torch.float32,
                kv_cache_dtype_str="auto",
                layer_info=SimpleNamespace(
                    full_attention_layer_ids=[3], swa_attention_layer_ids=[0, 1, 2]
                ),
                spec_algorithm=SpeculativeAlgorithm.DSPARK,
                is_draft_worker=False,
                spec_aux_config=SimpleNamespace(
                    draft_model_config=draft, dflash_draft_num_layers=2
                ),
                dflash_draft_cell_size_per_token=64,
                draft_attn_tp_size=1,
                sliding_window_size=2,
                page_size=1,
                fused_entry_bytes=lambda _: None,
            )
        )
        # One target FULL layer and three target + two draft SWA layers, 32 B/row.
        sizes = configurator.calculate_pool_sizes(128 * 32 + 32 * 5 * 32, page_size=1)
        self.assertEqual(
            (sizes.full_max_total_num_tokens, sizes.swa_max_total_num_tokens), (128, 32)
        )

    def test_bundled_loader_and_shared_embedding(self):
        target = Spark2_5ForCausalLM(config=_config())
        draft = DSparkSpark25DraftModel(config=configure_spark25_dspark(_config()))
        _initialize(draft)
        source = {name: p.detach().clone() for name, p in draft.named_parameters()}
        embedding = torch.arange(32 * 8).float().view(32, 8) / 32
        weights = [
            ("model.embedding.weight", embedding),
            ("draft_model.embed_tokens.weight", torch.full_like(embedding, -9)),
        ]
        for name, value in source.items():
            if "gate_up_proj" in name:
                gate, up = value.chunk(2, dim=0)
                weights.extend(
                    [
                        (
                            "draft_model." + name.replace("gate_up_proj", "gate_proj"),
                            gate,
                        ),
                        ("draft_model." + name.replace("gate_up_proj", "up_proj"), up),
                    ]
                )
            else:
                weights.append(("draft_model." + name, value))
        target.load_weights(iter(weights))
        draft.load_weights(iter(weights))
        draft.attach_shared_modules(
            embed_tokens=target.model.embed_tokens, lm_head=target.lm_head
        )
        self.assertIs(draft.embed_tokens, target.model.embed_tokens)
        self.assertIs(draft.lm_head, target.lm_head)
        ids = torch.tensor([1, 4, 7])
        torch.testing.assert_close(draft.forward_embed(ids), embedding[ids])
        for name, expected in source.items():
            torch.testing.assert_close(dict(draft.named_parameters())[name], expected)

    def test_target_feature_projection_normalizes_each_capture(self):
        draft = DSparkSpark25DraftModel(config=configure_spark25_dspark(_config()))
        _initialize(draft)
        hidden = torch.linspace(-4, 3, 48).reshape(3, 16)
        normalized = torch.cat(
            [
                F.rms_norm(chunk, (8,), norm.weight, eps=1e-6)
                for chunk, norm in zip(hidden.chunk(2, dim=-1), draft.fc_norm)
            ],
            dim=-1,
        )
        expected = F.rms_norm(
            F.linear(normalized, draft.fc.weight),
            (8,),
            draft.hidden_norm.weight,
            eps=1e-6,
        )
        torch.testing.assert_close(draft.project_target_hidden(hidden), expected)
        with self.assertRaisesRegex(ValueError, "target hidden shape"):
            draft.project_target_hidden(hidden[:, :8])

    def test_context_kv_uses_sliding_rope_and_committed_physical_rows(self):
        draft = DSparkSpark25DraftModel(config=configure_spark25_dspark(_config()))
        _initialize(draft)
        pool = MHATokenToKVPool(
            size=16,
            page_size=1,
            head_num=1,
            head_dim=4,
            dtype=torch.float32,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        positions = torch.tensor([1, 3, 4, 9])
        hidden = torch.linspace(-2, 3, 32).reshape(4, 8)
        locations = torch.tensor([[7, 8], [11, 999999]])
        for commit_lens in (None, torch.tensor([1, 1]), torch.tensor([0, 0])):
            with self.subTest(commit_lens=commit_lens):
                for buf in [*pool.k_buffer, *pool.v_buffer]:
                    buf.zero_()
                draft.write_target_hidden_kv(
                    target_hidden=hidden,
                    target_hidden_is_projected=True,
                    pool=pool,
                    positions=positions,
                    cache_loc=torch.tensor([7, 8, 11, 12]),
                    cache_loc_2d=locations if commit_lens is not None else None,
                    commit_lens=commit_lens,
                )
                indices = (
                    torch.arange(4)
                    if commit_lens is None
                    else (torch.arange(2)[None, :] < commit_lens[:, None])
                    .flatten()
                    .nonzero()
                    .flatten()
                )
                loc = (
                    torch.tensor([7, 8, 11, 12])
                    if commit_lens is None
                    else locations.flatten()[indices]
                )
                for index, layer in enumerate(draft.layers):
                    attn = layer.self_attn
                    qkv = F.linear(hidden, attn.q_k_v_proj.weight)
                    k, v = qkv[:, 8:12].view(-1, 1, 4), qkv[:, 12:].view(-1, 1, 4)
                    expected_k = torch.zeros_like(pool.k_buffer[index])
                    expected_v = torch.zeros_like(pool.v_buffer[index])
                    expected_k[loc] = _rope(k, positions, theta=10000, rotary_dim=4)[
                        indices
                    ]
                    expected_v[loc] = v[indices]
                    torch.testing.assert_close(pool.k_buffer[index], expected_k)
                    torch.testing.assert_close(pool.v_buffer[index], expected_v)

    def test_capture_includes_residual_and_last_layer_before_final_norm(self):
        target = Spark2_5ForCausalLM(config=_config())
        _initialize(target)
        target.set_dspark_layers_to_capture([0, 3])
        hidden = torch.linspace(-2, 3, 32).reshape(4, 8)
        positions = torch.arange(4)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=positions,
            attn_cp_metadata=None,
            can_run_tbo=False,
            global_num_token_non_padded_cpu=None,
            residual_stream=None,
        )
        with forward_context(ForwardContext(attn_backend=_SDPAttention())):
            result, captures = target.model(
                input_ids=positions,
                positions=positions,
                forward_batch=batch,
                input_embeds=hidden.clone(),
            )
            expected = hidden.clone()
            for index, layer in enumerate(target.model.layers):
                params = target.config.rope_parameters[target.config.layer_types[index]]
                expected = _reference_layer(expected, layer, params, positions, batch)
                if index in (0, 3):
                    torch.testing.assert_close(
                        captures[0 if index == 0 else 1], expected
                    )
            torch.testing.assert_close(
                result, F.rms_norm(expected, (8,), target.model.norm.weight, eps=1e-6)
            )
        self.assertIsNone(batch.residual_stream)

    def test_draft_forward_matches_dense_reference_with_shared_embedding(self):
        target = Spark2_5ForCausalLM(config=_config())
        draft = DSparkSpark25DraftModel(config=configure_spark25_dspark(_config()))
        _initialize(target)
        _initialize(draft)
        draft.attach_shared_modules(
            embed_tokens=target.model.embed_tokens, lm_head=target.lm_head
        )
        positions = torch.arange(4)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=positions,
            attn_cp_metadata=None,
            can_run_tbo=False,
            global_num_token_non_padded_cpu=None,
            residual_stream=None,
        )
        with forward_context(ForwardContext(attn_backend=_SDPAttention())):
            result = draft(
                input_ids=positions, positions=positions, forward_batch=batch
            )
        expected = F.embedding(positions, target.model.embed_tokens.weight)
        for layer in draft.layers:
            self.assertEqual(layer.self_attn.attn.attn_type, AttentionType.ENCODER_ONLY)
            self.assertEqual(layer.self_attn.attn.sliding_window_size, 2)
            expected = _reference_layer(
                expected,
                layer,
                draft.config.rope_parameters["sliding_attention"],
                positions,
                batch,
            )
        torch.testing.assert_close(
            result.hidden_states,
            F.rms_norm(expected, (8,), draft.norm.weight, eps=1e-6),
        )
        self.assertIsNone(batch.residual_stream)


if __name__ == "__main__":
    unittest.main()
