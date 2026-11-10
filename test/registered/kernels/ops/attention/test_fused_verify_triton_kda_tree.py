"""Correctness tests for factorized KDA tree verification and commit."""

import pytest
import torch

from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("width", [2, 17, 32, 49, 65])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_factorized_tree_matches_serial_model_shape(width, lower_bound):
    torch.manual_seed(4096)
    device = "cuda"
    batch_size = 1
    num_query_heads, num_value_heads = 16, 32
    key_dim = value_dim = 128
    wave_width = (width - 1 + 3) // 4
    parent_values = [-1] + [0] * wave_width
    depth_values = [0] + [1] * wave_width
    for depth in range(2, 5):
        previous_start = 1 + (depth - 2) * wave_width
        parent_values.extend(range(previous_start, previous_start + wave_width))
        depth_values.extend([depth] * wave_width)
    parents = torch.tensor([parent_values[:width]], dtype=torch.int32, device=device)
    depths = torch.tensor([depth_values[:width]], dtype=torch.int32, device=device)
    total = batch_size * width
    q = torch.randn(
        1, total, num_query_heads, key_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total, num_value_heads, value_dim, dtype=torch.bfloat16, device=device
    )
    a = torch.randn(
        1, total, num_value_heads, key_dim, dtype=torch.bfloat16, device=device
    )
    b = torch.randn(1, total, num_value_heads, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(num_value_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_value_heads, key_dim, dtype=torch.bfloat16, device=device)
    state = torch.randn(
        batch_size,
        num_value_heads,
        value_dim,
        key_dim,
        dtype=torch.float32,
        device=device,
    )
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, width], dtype=torch.int32, device=device)
    serial_states = torch.empty(
        batch_size,
        width,
        num_value_heads,
        value_dim,
        key_dim,
        dtype=torch.float32,
        device=device,
    )
    tree_query_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        key_dim,
        dtype=torch.float32,
        device=device,
    )
    tree_value_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        width,
        dtype=torch.float32,
        device=device,
    )
    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        is_kda=True,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        intermediate_state_indices=indices,
        cache_steps=width,
        retrieve_parent_token=parents,
        lower_bound=lower_bound,
    )
    serial_out = fused_sigmoid_gating_delta_rule_update(
        **common, intermediate_states_buffer=serial_states
    )
    factor_kwargs = dict(
        intermediate_states_buffer=None,
        tree_depths=depths,
        tree_max_depth=5,
        tree_query_factors=tree_query_factors,
        tree_value_factors=tree_value_factors,
        force_factorized_tree=True,
    )
    keys = k[0].float().repeat_interleave(num_value_heads // num_query_heads, dim=1)
    keys = keys / torch.sqrt((keys * keys).sum(-1, keepdim=True) + 1e-6)
    gate_input = a[0].float() + dt_bias.float()
    if lower_bound is None:
        gate = -A_log.exp()[:, None] * torch.nn.functional.softplus(gate_input)
    else:
        gate = lower_bound * torch.sigmoid(A_log.exp()[:, None] * gate_input)
    transforms = {
        "tree_processed_k": keys,
        "tree_processed_decay": gate.exp(),
        "tree_processed_beta": torch.sigmoid(b[0].float()),
    }
    precomputed = {
        name: value.transpose(0, 1).unsqueeze(0).contiguous()
        for name, value in transforms.items()
    }
    for extra in ({}, {**precomputed, "force_precomputed_tree_transforms": True}):
        factorized_out = fused_sigmoid_gating_delta_rule_update(
            **common, **factor_kwargs, **extra
        )
        torch.testing.assert_close(factorized_out, serial_out, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("width", [2, 16, 17, 32, 33, 49, 65])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_factorized_tree_raw_factor_commit_matches_serial_selected_state(
    width, lower_bound
):
    from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
        commit_kda_replayssm_tree_all_layers,
    )

    torch.manual_seed(8192)
    device = "cuda"
    batch_size = 1
    num_slots = 5
    num_query_heads = num_value_heads = 4
    key_dim = value_dim = 32

    # Accept up to 16 nodes on a path that skips node 1, branching off its ancestors.
    path = [0, *range(2, min(width, 17))]
    parent_values = [-1, 0] + [
        path[path.index(node) - 1] if node in path else path[(node - 17) % 15]
        for node in range(2, width)
    ]
    parents = torch.tensor(
        [parent_values[:width]] * batch_size, dtype=torch.int32, device=device
    )
    total = batch_size * width
    q = torch.randn(
        1, total, num_query_heads, key_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total, num_value_heads, value_dim, dtype=torch.bfloat16, device=device
    )
    a = torch.randn(
        1, total, num_value_heads, key_dim, dtype=torch.bfloat16, device=device
    )
    b = torch.randn(1, total, num_value_heads, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(num_value_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_value_heads, key_dim, dtype=torch.bfloat16, device=device)
    state = torch.randn(
        num_slots,
        num_value_heads,
        value_dim,
        key_dim,
        dtype=torch.float32,
        device=device,
    )
    initial_state = state.clone()
    state_indices = torch.tensor([3], dtype=torch.int32, device=device)
    factor_indices = torch.tensor([2], dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, total + 1, width, dtype=torch.int32, device=device)
    serial_states = torch.full(
        (num_slots, width, num_value_heads, value_dim, key_dim),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    common = dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        is_kda=True,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        intermediate_state_indices=state_indices,
        cache_steps=width,
        retrieve_parent_token=parents,
        lower_bound=lower_bound,
    )
    serial_out = fused_sigmoid_gating_delta_rule_update(
        **common, intermediate_states_buffer=serial_states
    )

    tree_query_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        key_dim,
        dtype=torch.float32,
        device=device,
    )
    tree_value_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        width,
        dtype=torch.float32,
        device=device,
    )
    tree_rawv = torch.full(
        (num_slots, num_value_heads, width, value_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    tree_rawk = torch.full(
        (num_slots, num_query_heads, width, key_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    tree_g = torch.full(
        (num_slots, num_value_heads, width, key_dim),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    tree_beta = torch.full(
        (num_slots, num_value_heads, width),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    factor_kwargs = dict(
        intermediate_states_buffer=None,
        tree_max_depth=16,
        tree_query_factors=tree_query_factors,
        tree_value_factors=tree_value_factors,
        force_factorized_tree=True,
        cache_ring=True,
        replayssm_rawv=tree_rawv,
        replayssm_rawk=tree_rawk,
        replayssm_g=tree_g,
        replayssm_beta=tree_beta,
        factor_cache_indices=factor_indices,
    )
    factorized_out = fused_sigmoid_gating_delta_rule_update(**common, **factor_kwargs)

    torch.testing.assert_close(state, initial_state, rtol=0, atol=0)
    torch.testing.assert_close(factorized_out, serial_out, rtol=2e-2, atol=2e-2)
    expected_k = k[0].reshape(batch_size, width, num_query_heads, key_dim)
    expected_v = v[0].reshape(batch_size, width, num_value_heads, value_dim)
    for request_idx, factor_slot in enumerate(factor_indices.tolist()):
        torch.testing.assert_close(
            tree_rawk[factor_slot, :, :width],
            expected_k[request_idx].transpose(0, 1),
        )
        torch.testing.assert_close(
            tree_rawv[factor_slot, :, :width],
            expected_v[request_idx].transpose(0, 1),
        )
        assert not torch.isnan(tree_g[factor_slot, :, :width]).any()
        assert not torch.isnan(tree_beta[factor_slot, :, :width]).any()

    # Unrelated candidates must not affect this path, even if their raw commit
    # entries are subsequently poisoned. Keep every ancestor on the path intact.
    reject_nodes = torch.tensor(
        [node for node in range(width) if node not in path],
        dtype=torch.long,
        device=device,
    )
    for values in (q, k, v, a, b):
        values[:, reject_nodes] = 4 * torch.randn_like(values[:, reject_nodes])
    perturbed_out = fused_sigmoid_gating_delta_rule_update(**common, **factor_kwargs)
    torch.testing.assert_close(
        perturbed_out[:, path], factorized_out[:, path], rtol=0, atol=0
    )
    torch.testing.assert_close(state, initial_state, rtol=0, atol=0)
    for cache in (tree_rawv, tree_rawk, tree_g, tree_beta):
        cache[factor_indices[0], :, reject_nodes] = float("nan")
    last_node = path[-1]
    track_node = path[len(path) // 2]
    accept_index = torch.tensor(
        [path],
        dtype=torch.int32,
        device=device,
    )
    accept_lens = torch.tensor([len(path)], dtype=torch.int32, device=device)
    track_indices = torch.tensor([4], dtype=torch.int32, device=device)
    track_nodes = torch.tensor([track_node], dtype=torch.int32, device=device)
    committed_state = state.clone().unsqueeze(0)
    commit_kwargs = dict(
        checkpoint_state=committed_state,
        rawv_cache=tree_rawv.unsqueeze(0),
        rawk_cache=tree_rawk.unsqueeze(0),
        gk_cache=tree_g.unsqueeze(0),
        beta_cache=tree_beta.unsqueeze(0),
        ssm_state_indices=state_indices,
        factor_cache_indices=factor_indices,
        accept_index=accept_index,
        accept_lens=accept_lens,
        tree_width=width,
        mamba_track_indices=track_indices,
        mamba_steps_to_track=track_nodes,
        null_block_id=-1,
    )
    gated_lens = torch.zeros_like(accept_lens)
    gated_kwargs = commit_kwargs | {"accept_lens": gated_lens}
    commit_kda_replayssm_tree_all_layers(**gated_kwargs)
    torch.testing.assert_close(committed_state[0], state, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit_kda_replayssm_tree_all_layers(**gated_kwargs)
    torch.testing.assert_close(committed_state[0], state, rtol=0, atol=0)
    gated_lens.copy_(accept_lens)
    graph.replay()
    torch.testing.assert_close(committed_state[0, :3], state[:3], rtol=0, atol=0)

    torch.testing.assert_close(
        committed_state[0, state_indices[0]],
        serial_states[state_indices[0], last_node],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        committed_state[0, track_indices[0]],
        serial_states[state_indices[0], track_node],
        rtol=1e-5,
        atol=1e-5,
    )

    # First continue from the committed prefix, then reuse the same state and
    # scratch slots for another request with a different prefix. Stale sibling
    # data and the previous request's factors must not participate in either.
    for reset_prefix in (False, True):
        state.copy_(committed_state[0])
        if reset_prefix:
            state[state_indices[0]].copy_(initial_state[1])
        before_verify = state.clone()
        serial_out = fused_sigmoid_gating_delta_rule_update(
            **common, intermediate_states_buffer=serial_states
        )
        factorized_out = fused_sigmoid_gating_delta_rule_update(
            **common, **factor_kwargs
        )
        torch.testing.assert_close(state, before_verify, rtol=0, atol=0)
        torch.testing.assert_close(factorized_out, serial_out, rtol=2e-2, atol=2e-2)
        for cache in (tree_rawv, tree_rawk, tree_g, tree_beta):
            cache[factor_indices[0], :, reject_nodes] = float("nan")
        committed_state[0].copy_(state)
        commit_kda_replayssm_tree_all_layers(**commit_kwargs)
        torch.testing.assert_close(
            committed_state[0, state_indices[0]],
            serial_states[state_indices[0], last_node],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            committed_state[0, track_indices[0]],
            serial_states[state_indices[0], track_node],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(committed_state[0, :3], state[:3], rtol=0, atol=0)


@pytest.mark.parametrize("production_layout", [False, True])
@pytest.mark.parametrize(
    "paths,track_nodes",
    [
        ([[0]], [0]),
        ([[0, 1]], [0]),
        ([[0, 1, 4, 13]], [4]),
    ],
)
def test_tree_conv_raw_commit_matches_full_window(
    paths, track_nodes, production_layout
):
    """Raw-node tree verify must preserve output and restore exact conv windows."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_tree_raw_commit_with_mask,
    )

    torch.manual_seed(8128 + sum(map(len, paths)))
    device = "cuda"
    batch_size, tree_width, dim, kernel_width = 1, 49, 384, 4
    num_slots = 7
    x = torch.randn(batch_size, dim, tree_width, dtype=torch.bfloat16, device=device)
    weight = torch.randn(dim, kernel_width, dtype=torch.float32, device=device)
    bias = torch.randn(dim, dtype=torch.float32, device=device)
    initial_state = torch.randn(
        num_slots, dim, kernel_width - 1, dtype=torch.bfloat16, device=device
    )
    state_indices = torch.tensor([1], dtype=torch.int32, device=device)
    factor_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    parents = torch.zeros(batch_size, tree_width, dtype=torch.int32, device=device)
    for node in range(1, tree_width):
        parents[:, node] = (node - 1) // 3
    unused_edges = torch.full_like(parents, -1)

    dense_scratch = torch.full(
        (1, batch_size, tree_width, dim, kernel_width - 1),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    raw_scratch = torch.full_like(dense_scratch, float("nan"))
    dense_state = initial_state.clone()
    raw_state = initial_state.clone()
    common = dict(
        x=x,
        weight=weight,
        bias=bias,
        activation="silu",
        conv_state_indices=state_indices,
        intermediate_state_indices=factor_indices,
        retrieve_next_token=unused_edges,
        retrieve_next_sibling=unused_edges,
        retrieve_parent_token=parents,
        direct_parent=True,
    )
    dense_out = causal_conv1d_update(
        conv_state=dense_state,
        intermediate_conv_window=dense_scratch[0],
        **common,
    )
    raw_out = causal_conv1d_update(
        conv_state=raw_state,
        intermediate_conv_window=raw_scratch[0],
        save_tree_raw_only=True,
        **common,
    )

    assert torch.equal(raw_out, dense_out)
    assert torch.equal(raw_state, initial_state)
    assert torch.isnan(raw_scratch[..., :-1]).all()
    assert torch.isfinite(raw_scratch[..., -1]).all()

    max_accept = max(map(len, paths))
    accept_index = torch.full(
        (batch_size, max_accept), -1, dtype=torch.int32, device=device
    )
    accept_lens = torch.tensor(
        [len(path) for path in paths], dtype=torch.int32, device=device
    )
    for request, path in enumerate(paths):
        accept_index[request, : len(path)] = torch.tensor(
            path, dtype=torch.int32, device=device
        )
    track_indices = torch.tensor([4], dtype=torch.int32, device=device)
    track_nodes_tensor = torch.tensor(track_nodes, dtype=torch.int32, device=device)

    expected = initial_state.clone()
    for request, path in enumerate(paths):
        local_last = path[-1] - request * tree_width
        expected[state_indices[request]] = dense_scratch[0, request, local_last]
        expected[track_indices[request]] = dense_scratch[
            0, request, track_nodes[request]
        ]

    commit_state = raw_state.unsqueeze(0)
    commit_scratch = raw_scratch
    if production_layout:
        commit_state = commit_state.transpose(-1, -2).contiguous()
        commit_scratch = commit_scratch.transpose(-1, -2).contiguous()

    fused_conv_tree_raw_commit_with_mask(
        commit_state,
        commit_scratch,
        state_indices,
        factor_indices,
        accept_index,
        accept_lens,
        tree_width,
        track_indices,
        track_nodes_tensor,
    )
    committed = (
        commit_state.transpose(-1, -2).squeeze(0)
        if production_layout
        else commit_state.squeeze(0)
    )
    assert torch.equal(committed, expected)


@pytest.mark.parametrize("width", [2, 17, 32, 33, 49, 65])
def test_factorized_tree_cuda_graph_replay_reads_new_inputs(width):
    """Factorized tree paths must remain address-stable and replay-safe."""
    from types import SimpleNamespace

    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )
    from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.runtime_context import get_context
    from sglang.srt.speculative.ubtree_info import create_ubtree_verify_input

    torch.manual_seed(width * 1000 + 49)
    device = "cuda"
    batch_size = 1
    num_heads = 4
    dim = 32
    wave_width = (width - 1 + 3) // 4
    parent_values = [-1] + [0] * wave_width
    depth_values = [0] + [1] * wave_width
    for depth in range(2, 5):
        previous_start = 1 + (depth - 2) * wave_width
        parent_values.extend(range(previous_start, previous_start + wave_width))
        depth_values.extend([depth] * wave_width)

    parents = torch.tensor([parent_values[:width]], dtype=torch.int32, device=device)
    depths = torch.tensor([depth_values[:width]], dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, width], dtype=torch.int32, device=device)
    indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
    A_log = torch.randn(num_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_heads, dim, dtype=torch.bfloat16, device=device)
    state = torch.randn(
        batch_size, num_heads, dim, dim, dtype=torch.float32, device=device
    )
    q = torch.randn(1, width, num_heads, dim, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn_like(q)
    b = torch.randn(1, width, num_heads, dtype=torch.bfloat16, device=device)

    state_pool = SimpleNamespace(
        enable_linear_replayssm=False,
        replayssm_request_ring=False,
        size=batch_size,
    )
    runner = SimpleNamespace(
        device=device,
        server_args=SimpleNamespace(enable_mis=False),
        is_draft_worker=False,
        req_to_token_pool=SimpleNamespace(
            get_mamba_indices=lambda requests: requests.to(torch.int32).clone(),
            translate_mamba_indices=lambda slots: slots,
            state_sub_pool="state",
            holds_state_slots=True,
            req_index_to_mamba_index_mapping=indices.clone(),
        ),
        token_to_kv_pool=SimpleNamespace(
            layer_router=SimpleNamespace(
                part_view=lambda part: state_pool,
                part_indices=lambda part: {},
            )
        ),
        runner_ids=SimpleNamespace(state_ids=lambda sub_pool: IdSpace(key="state")),
    )
    with get_context().override_server_args(
        speculative_algorithm="UBTREE",
        speculative_eagle_topk=2,
        speculative_dflash_block_size=5,
    ):
        backend = MambaAttnBackendBase(runner)
        backend.init_cuda_graph_state(1, width)
        spec = create_ubtree_verify_input(None, width)
    spec.retrieve_parent_token = parents
    spec.tree_depths = depths
    spec.retrieve_next_token = torch.zeros_like(parents)
    spec.retrieve_next_sibling = torch.full_like(parents, -1)
    batch = SimpleNamespace(
        batch_size=1,
        req_pool_indices=indices,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=spec,
        seq_lens_cpu=None,
        ragged_verify_layout=None,
    )
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    metadata = backend.forward_metadata
    depth_pointer = metadata.tree_depths.data_ptr()

    def run(query_factors, value_factors, *, eager=False):
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            is_kda=True,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_states_buffer=None,
            intermediate_state_indices=indices,
            cache_steps=width,
            retrieve_parent_token=(
                spec.retrieve_parent_token if eager else metadata.retrieve_parent_token
            ),
            tree_depths=spec.tree_depths if eager else metadata.tree_depths,
            tree_max_depth=5,
            tree_query_factors=query_factors,
            tree_value_factors=value_factors,
            force_factorized_tree=True,
        )

    graph_query_factors = torch.empty(
        batch_size, num_heads, width, dim, dtype=torch.float32, device=device
    )
    graph_value_factors = torch.empty(
        batch_size, num_heads, width, width, dtype=torch.float32, device=device
    )
    # Compile before capture.
    run(graph_query_factors, graph_value_factors)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run(graph_query_factors, graph_value_factors)

    ref_query_factors = torch.empty_like(graph_query_factors)
    ref_value_factors = torch.empty_like(graph_value_factors)
    # Replace proposal tensors as well as changing recurrent inputs. Refresh via
    # both public paths: replay preparation and the post-draft plan-stream fixup.
    # Eager uses the live proposal, so stale captured topology cannot pass.
    for fanout in (3, 4):
        new_parents = [-1] + [(i - 1) // fanout for i in range(1, width)]
        new_depths = [0] * width
        for i in range(1, width):
            new_depths[i] = new_depths[new_parents[i]] + 1
        spec.retrieve_parent_token = torch.tensor(
            [new_parents], dtype=torch.int32, device=device
        )
        spec.tree_depths = torch.tensor([new_depths], dtype=torch.int32, device=device)
        for tensor in (q, k, v, a, b):
            tensor.copy_(torch.randn_like(tensor))
        if fanout == 3:
            backend.init_forward_metadata_out_graph(batch)
        else:
            backend.update_verify_buffers_to_fill_after_draft(spec, 1)
        assert backend.forward_metadata.tree_depths.data_ptr() == depth_pointer
        graph.replay()
        replay_out = graph_out.clone()
        eager_out = run(ref_query_factors, ref_value_factors, eager=True)
        torch.testing.assert_close(replay_out, eager_out, rtol=0, atol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
