import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.lilicorr import (
    lilicorr_greedy_path,
    lilicorr_topk_lse,
)
from sglang.srt.models.lilicorr import (
    LiLiCorrHead,
    check_conv_weight_coverage,
    check_head_weight_coverage,
)
from sglang.srt.speculative.lilicorr_components.lilicorr_candidates import (
    lilicorr_candidates,
    per_request_last_row,
    publish_anchor,
    resolve_vocab_shard,
    target_input_embeddings,
)
from sglang.srt.speculative.lilicorr_components.lilicorr_config import (
    parse_lilicorr_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_GEOMETRY = {
    "lilicorr_enabled": True,
    "lilicorr_candidate_topk": 4,
    "lilicorr_hidden_size": 8,
    "lilicorr_num_layers": 2,
    "lilicorr_num_heads": 2,
    "lilicorr_mlp_ratio": 2.0,
    "lilicorr_factor_dim": 4,
    "lilicorr_vector_eps": 1e-6,
    "lilicorr_logit_scale": 10.0,
}


def _lilicorr_config(**overrides):
    dflash_config = {**_GEOMETRY, **overrides}
    for key, value in list(dflash_config.items()):
        if value is None:
            del dflash_config[key]
    return parse_lilicorr_draft_config(
        draft_hf_config={"num_hidden_layers": 2, "dflash_config": dflash_config}
    )


def _head(*, model_hidden_size=16, block_size=5, **overrides):
    config = _lilicorr_config(**overrides)
    torch.manual_seed(0)
    head = LiLiCorrHead(
        model_hidden_size=model_hidden_size,
        block_size=block_size,
        rms_norm_eps=1e-6,
        config=config,
    )
    # Random weights, because a head at its zero-initialized construction values
    # scores every candidate identically and would hide a real scoring bug.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.2)
    head.materialize_inference_buffers(torch.device("cpu"), torch.float32)
    return head


def _lattice(head, *, bs=3, model_hidden_size=16):
    slots = head.num_candidate_slots
    topk = head.candidate_topk
    torch.manual_seed(1)
    return {
        "token_embeddings": torch.randn(bs, slots, topk, model_hidden_size),
        "candidate_token_ids": torch.randint(0, 64, (bs, slots, topk)),
        "candidate_log_probs": torch.randn(bs, slots, topk).log_softmax(dim=-1),
        "pass_hidden": torch.randn(bs, slots, model_hidden_size),
        "anchor_hidden": torch.randn(bs, model_hidden_size),
        "anchor_valid": torch.ones(bs, dtype=torch.bool),
    }


# --- config ----------------------------------------------------------------


@pytest.mark.parametrize(
    "dflash_config", [{}, {"lilicorr_enabled": False}], ids=["absent", "disabled"]
)
def test_no_head_geometry_is_reported_against_the_architecture_string(dflash_config):
    """Absence and an explicit disable are the same case, and neither is a
    per-field error: the checkpoint asked for this head by declaring the
    architecture and then did not say which head, so that is what the message
    has to name."""
    with pytest.raises(ValueError, match="LiLiCorrDraftModel"):
        parse_lilicorr_draft_config(
            draft_hf_config={
                "num_hidden_layers": 2,
                "dflash_config": dflash_config,
            }
        )


@pytest.mark.parametrize("dropped", sorted(set(_GEOMETRY) - {"lilicorr_enabled"}))
def test_every_geometry_field_is_required(dropped):
    """None of these may be defaulted. Most change a tensor shape and would be
    caught at weight load, but logit_scale and vector_eps do not: a guessed value
    builds a head that loads cleanly and scores a different function."""
    with pytest.raises(ValueError, match=dropped):
        _lilicorr_config(**{dropped: None})


@pytest.mark.parametrize("topk", [3, 5, 6, 12])
def test_a_candidate_topk_that_is_not_a_power_of_two_is_refused(topk):
    """The tiled candidate top-k holds its selected tiles in one Triton lane
    group, and tl.arange needs a power-of-two extent. Refusing at load beats
    failing inside a kernel on the first decode, and beats silently falling back
    to the reference path, which is far slower than the head is allowed to be."""
    with pytest.raises(ValueError, match="power of two"):
        _lilicorr_config(lilicorr_candidate_topk=topk)


@pytest.mark.parametrize("topk", [1, 2, 4, 8, 16])
def test_power_of_two_candidate_topk_is_accepted(topk):
    assert _lilicorr_config(lilicorr_candidate_topk=topk).candidate_topk == topk


def test_zero_head_width_means_as_wide_as_the_draft():
    """The exporter records 0 for a head with no token_proj, which is the one
    field allowed to be non-positive."""
    config = _lilicorr_config(lilicorr_hidden_size=0)
    assert config.resolve_hidden_size(model_hidden_size=16) == 16
    assert isinstance(_head(lilicorr_hidden_size=0).token_proj, torch.nn.Identity)


# --- the head --------------------------------------------------------------


def test_head_parameter_names_match_the_exported_checkpoint_subtree():
    """Pins weight compatibility with the training export. A renamed submodule
    here loads nothing under that name and the base loader ignores what it
    cannot resolve, so the head would serve its construction values."""
    head = _head()
    names = set(dict(head.named_parameters()))
    expected_leaves = {
        "pass_hidden_proj.weight",
        "pass_hidden_proj.bias",
        "token_proj.weight",
        "token_proj.bias",
        "context_proj.weight",
        "context_proj.bias",
        "slot_embedding",
        "rank_embedding",
        "relative_slot_bias",
        "same_slot_bias",
        "factor_input_proj.weight",
        "factor_input_proj.bias",
        "out_head.weight",
        "in_head.weight",
        "anchor_out_head.weight",
        "output_norm.weight",
        "anchor_norm.weight",
        "feature_mlp.0.weight",
        "feature_mlp.1.weight",
        "feature_mlp.3.weight",
        # nn.MultiheadAttention's exported layout, which the trained head uses.
        "layers.0.attn.in_proj_weight",
        "layers.0.attn.in_proj_bias",
        "layers.0.attn.out_proj.weight",
        "layers.0.attn.out_proj.bias",
        "layers.0.attn_norm.weight",
        "layers.0.mlp_norm.weight",
        "layers.0.mlp.0.weight",
        "layers.0.mlp.2.weight",
    }
    assert expected_leaves <= names


def test_score_shapes_and_scoring_before_materialize_raises():
    head = _head()
    slots, topk = head.num_candidate_slots, head.candidate_topk
    lattice = _lattice(head)
    start, pair = head.score(
        token_embeddings=lattice["token_embeddings"].unsqueeze(1),
        candidate_log_probs=lattice["candidate_log_probs"].unsqueeze(1),
        pass_hidden=lattice["pass_hidden"].unsqueeze(1),
        anchor_hidden=lattice["anchor_hidden"].unsqueeze(1),
        anchor_valid=lattice["anchor_valid"].unsqueeze(1),
    )
    assert start.shape == (3, 1, topk)
    assert pair.shape == (3, 1, slots - 1, topk, topk)

    unmaterialized = LiLiCorrHead(
        model_hidden_size=16,
        block_size=5,
        rms_norm_eps=1e-6,
        config=_lilicorr_config(),
    )
    with pytest.raises(RuntimeError, match="materialize_inference_buffers"):
        unmaterialized.score(
            token_embeddings=lattice["token_embeddings"].unsqueeze(1),
            candidate_log_probs=lattice["candidate_log_probs"].unsqueeze(1),
            pass_hidden=lattice["pass_hidden"].unsqueeze(1),
            anchor_hidden=lattice["anchor_hidden"].unsqueeze(1),
            anchor_valid=lattice["anchor_valid"].unsqueeze(1),
        )


def test_select_commits_candidates_from_the_lattice():
    head = _head()
    lattice = _lattice(head)
    selected = head.select(**lattice)
    assert selected.shape == (3, head.num_candidate_slots)
    # Every committed token must be one of that slot's candidates.
    assert (selected.unsqueeze(-1) == lattice["candidate_token_ids"]).any(-1).all()


def test_an_invalid_anchor_ignores_whatever_is_in_the_buffer():
    """An invalid anchor is zeroed by multiplication rather than by a branch, so
    the captured graph needs no host sync. The graph replays at the padded bucket
    batch size, so rows past the live batch read stale anchor memory -- their
    scores must not depend on it. Note this zeroes the *projected* state, which
    is not the same as feeding a zero anchor through a biased context_proj."""
    head = _head()
    lattice = _lattice(head)
    invalid = {**lattice, "anchor_valid": torch.zeros(3, dtype=torch.bool)}
    stale = {
        **invalid,
        "anchor_hidden": lattice["anchor_hidden"].roll(1, dims=0) * 7.0,
    }
    torch.testing.assert_close(head.select(**invalid), head.select(**stale))


def test_a_precomputed_projected_table_scores_like_raw_embeddings():
    """The folded path gathers rows of embed_tokens.weight @ token_proj.weight.T
    + bias instead of embedding then projecting. token_proj is affine, so this
    must be the same function of the token id."""
    head = _head()
    lattice = _lattice(head)
    embed_tokens = torch.nn.Embedding(64, 16)
    with torch.no_grad():
        embed_tokens.weight.normal_(0.0, 0.2)
    table = head.build_token_table(embed_tokens)
    assert table is not None and table.shape == (64, head.hidden_size)

    ids = lattice["candidate_token_ids"]
    raw = head.select(**{**lattice, "token_embeddings": embed_tokens(ids).detach()})
    folded = head.select(
        **{**lattice, "token_embeddings": table[ids]}, already_projected=True
    )
    torch.testing.assert_close(raw, folded)


def test_head_weight_coverage_is_required_in_both_directions():
    """The base loader ignores what it cannot resolve, so both a missing tensor and
    a surplus one are silent. Either produces a low but believable acceptance
    length, so both must raise."""
    head = _head()
    names = {f"lilicorr.{name}" for name, _ in head.named_parameters()}
    check_head_weight_coverage(head, set(names))

    with pytest.raises(ValueError, match="missing 1 head parameter"):
        check_head_weight_coverage(head, names - {"lilicorr.out_head.weight"})
    with pytest.raises(ValueError, match="no parameter for"):
        check_head_weight_coverage(head, names | {"lilicorr.layers.9.attn_norm.weight"})


def test_an_identity_token_proj_would_drop_the_checkpoints_projection():
    """The live case for the surplus direction: at head width == draft width the
    head builds token_proj as an Identity, so a checkpoint trained with a real
    projection has those tensors dropped and scores without them."""
    wide = _head(lilicorr_hidden_size=0)
    assert isinstance(wide.token_proj, torch.nn.Identity)
    names = {f"lilicorr.{name}" for name, _ in wide.named_parameters()}
    assert not any(name.startswith("lilicorr.token_proj") for name in names)
    with pytest.raises(ValueError, match="lilicorr_hidden_size"):
        check_head_weight_coverage(
            wide, names | {"lilicorr.token_proj.weight", "lilicorr.token_proj.bias"}
        )


# --- kernels ---------------------------------------------------------------


def test_topk_lse_returns_full_vocab_normalized_log_probs():
    """The head consumes val - lse, which must equal log_softmax over the whole
    vocabulary: raw top-k logits would score a different function."""
    torch.manual_seed(0)
    logits = torch.randn(7, 300)
    vals, ids, lse = lilicorr_topk_lse(logits, 5)
    expected_vals, expected_ids = torch.log_softmax(logits, dim=-1).topk(5, dim=-1)
    torch.testing.assert_close(vals - lse.unsqueeze(-1), expected_vals)
    torch.testing.assert_close(ids, expected_ids.to(torch.int64))


def test_greedy_path_follows_the_conditioned_argmax_recurrence():
    """c_0 = argmax(start), then c_s = argmax_c pair[s-1, c_{s-1}, c]. Ties break
    toward the lower index, which is what makes the fused kernel and the torch
    path commit the same path."""
    torch.manual_seed(0)
    bs, slots, k = 3, 4, 4
    log_start = torch.randn(bs, k)
    log_pair = torch.randn(bs, slots - 1, k, k)
    ids = torch.randint(0, 100, (bs, slots, k))

    actual = lilicorr_greedy_path(log_start, log_pair, ids)

    expected = torch.empty(bs, slots, dtype=ids.dtype)
    for row in range(bs):
        cur = int(log_start[row].argmax())
        expected[row, 0] = ids[row, 0, cur]
        for slot in range(1, slots):
            cur = int(log_pair[row, slot - 1, cur].argmax())
            expected[row, slot] = ids[row, slot, cur]
    torch.testing.assert_close(actual, expected)


def test_greedy_path_breaks_ties_toward_the_lower_candidate():
    log_start = torch.zeros(1, 4)
    log_pair = torch.zeros(1, 2, 4, 4)
    ids = torch.arange(12).view(1, 3, 4)
    torch.testing.assert_close(
        lilicorr_greedy_path(log_start, log_pair, ids),
        torch.tensor([[0, 4, 8]]),
    )


# --- candidates ------------------------------------------------------------


def test_candidates_are_normalized_log_probs_with_global_ids():
    torch.manual_seed(0)
    hidden = torch.randn(6, 8)
    weight = torch.randn(50, 8)
    log_probs, ids = lilicorr_candidates(
        hidden_states=hidden,
        weight=weight,
        num_org=40,
        org_vocab_start=100,
        topk=4,
    )
    reference = torch.log_softmax(torch.matmul(hidden, weight[:40].T), dim=-1)
    expected_vals, expected_ids = reference.topk(4, dim=-1)
    torch.testing.assert_close(log_probs, expected_vals)
    torch.testing.assert_close(ids, expected_ids.to(torch.int64) + 100)


def test_chunking_cannot_change_a_candidate():
    """Rows are chunked only to cap the [chunk, vocab] logits buffer, and every
    operation is per-row, so the chunk width must not be observable."""
    torch.manual_seed(0)
    hidden = torch.randn(9, 8)
    weight = torch.randn(60, 8)
    kwargs = dict(
        hidden_states=hidden, weight=weight, num_org=60, org_vocab_start=0, topk=3
    )
    wide = lilicorr_candidates(**kwargs, chunk_size=256)
    narrow = lilicorr_candidates(**kwargs, chunk_size=2)
    torch.testing.assert_close(wide[0], narrow[0])
    torch.testing.assert_close(wide[1], narrow[1])


def test_a_topk_wider_than_this_ranks_vocabulary_slice_is_refused():
    with pytest.raises(ValueError, match="exceeds this rank"):
        lilicorr_candidates(
            hidden_states=torch.randn(2, 8),
            weight=torch.randn(8, 8),
            num_org=3,
            org_vocab_start=0,
            topk=4,
        )


def test_candidates_combine_across_vocab_shards():
    """Pins the TP contract: the global top-k, the global log-partition and the
    id offset. Getting any of them wrong returns plausible candidates normalized
    by one shard's partition, which no single-rank test observes."""
    torch.manual_seed(0)
    hidden = torch.randn(4, 8)
    full_weight = torch.randn(24, 8)
    topk = 3

    # This process plays rank 1 of tp=2: vocabulary rows 12..24.
    rank0_logits = torch.matmul(hidden, full_weight[:12].T)
    rank0_vals, rank0_ids = rank0_logits.topk(topk, dim=-1)
    rank0_lse = torch.logsumexp(rank0_logits, dim=-1)

    class _FakeTpGroup:
        world_size = 2

        def all_gather_into_tensor(self, output, packed):
            rows = packed.numel() // (2 * topk + 1)
            mine = packed.view(rows, 2 * topk + 1)
            theirs = torch.empty_like(mine)
            theirs[:, :topk] = rank0_vals
            theirs[:, topk : 2 * topk] = rank0_ids.to(torch.float32)
            theirs[:, 2 * topk] = rank0_lse
            output.copy_(torch.cat([theirs, mine], dim=0).view(-1))

    log_probs, ids = lilicorr_candidates(
        hidden_states=hidden,
        weight=full_weight[12:],
        num_org=12,
        org_vocab_start=12,
        topk=topk,
        tp_group=_FakeTpGroup(),
    )
    reference = torch.log_softmax(torch.matmul(hidden, full_weight.T), dim=-1)
    expected_vals, expected_ids = reference.topk(topk, dim=-1)
    torch.testing.assert_close(log_probs, expected_vals)
    torch.testing.assert_close(ids, expected_ids.to(torch.int64))


def test_vocab_shard_resolution_and_added_vocab_refusal():
    assert resolve_vocab_shard(SimpleNamespace(weight=torch.empty(32, 4))) == (32, 0)
    assert resolve_vocab_shard(
        SimpleNamespace(
            weight=torch.empty(32, 4),
            shard_indices=SimpleNamespace(
                num_org_elements=16, org_vocab_start_index=16, num_added_elements=0
            ),
        )
    ) == (16, 16)
    with pytest.raises(NotImplementedError, match="added vocabulary"):
        resolve_vocab_shard(
            SimpleNamespace(
                weight=torch.empty(32, 4),
                shard_indices=SimpleNamespace(
                    num_org_elements=16, org_vocab_start_index=0, num_added_elements=2
                ),
            )
        )


# --- the anchor ------------------------------------------------------------


def test_anchor_rows_from_commit_lens():
    ends = per_request_last_row(
        num_rows=6, positions=None, commit_lens=torch.tensor([2, 1, 3])
    )
    torch.testing.assert_close(ends, torch.tensor([1, 2, 5]))


def test_anchor_rows_recovered_from_prefill_positions():
    """Prefill does not forward commit_lens, but rows are request-major and each
    request's positions increase strictly, so a request ends wherever the next
    position fails to increase."""
    positions = torch.tensor([0, 1, 2, 0, 1, 0])
    ends = per_request_last_row(num_rows=6, positions=positions, commit_lens=None)
    torch.testing.assert_close(ends, torch.tensor([2, 4, 5]))

    single = per_request_last_row(
        num_rows=1, positions=torch.tensor([7]), commit_lens=None
    )
    torch.testing.assert_close(single, torch.tensor([0]))


def test_unrecoverable_anchor_rows_return_none_rather_than_a_guess():
    """A wrong anchor is a silent acceptance regression, so an input the
    boundaries cannot be read from must leave the anchor unset."""
    assert per_request_last_row(num_rows=4, positions=None, commit_lens=None) is None
    assert (
        per_request_last_row(
            num_rows=4, positions=torch.tensor([0, 1]), commit_lens=None
        )
        is None
    )


# --- the worker seam ------------------------------------------------------


def test_draft_graph_batch_sizes_reads_the_capture_buckets(monkeypatch):
    """The folded head is captured once per bucket and the compile prewarm has to
    cover every one, so this must be the list the engine actually captures."""
    from sglang.srt.speculative.lilicorr_components import (
        lilicorr_draft_sampler as sampler_mod,
    )

    monkeypatch.setattr(
        sampler_mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(bs=[8, 1, 4, 0])
                )
            )
        ),
    )
    assert sampler_mod.draft_graph_batch_sizes() == [1, 4, 8]


def test_an_engine_that_captures_no_buckets_keeps_the_head_eager(monkeypatch):
    """The static buffers are sized from the largest bucket, so with no buckets
    there is nothing to size them from. Refusing is correct; building against a
    guessed size would serve a head whose buffers do not match the replay."""
    from sglang.srt.speculative.lilicorr_components import (
        lilicorr_draft_sampler as sampler_mod,
    )

    monkeypatch.setattr(
        sampler_mod, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(sampler_mod, "draft_graph_batch_sizes", lambda: [])
    assert (
        sampler_mod.build_lilicorr_draft_sampler(
            worker=SimpleNamespace(), lm_head=SimpleNamespace()
        )
        is None
    )


def test_a_lilicorr_head_is_dispatched_to_the_folded_sampler():
    """The head must reach the graph fold rather than the eager fallback: eager
    costs a large fraction of throughput, so a silent demotion would read as a
    believable but wrong throughput number."""
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    lm_head = SimpleNamespace(weight=torch.empty(16, 4))
    built = {}
    worker = SimpleNamespace(
        block_size=5,
        selector=None,
        lilicorr=object(),
        ps=SimpleNamespace(tp_rank=0),
        draft_model=SimpleNamespace(lm_head=None),
        device="cpu",
        # The name the surrounding DFLASH code reads, not ours.
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=lm_head))
        ),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            worker_mod,
            "build_lilicorr_draft_sampler",
            lambda **kwargs: built.setdefault("kwargs", kwargs),
        )
        worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker)
    assert built["kwargs"]["lm_head"] is lm_head
    assert built["kwargs"]["worker"] is worker


def test_the_target_embedding_table_is_the_targets_and_not_the_drafts():
    """The head embeds candidate ids with the table it was trained against. The
    draft's own table exists on Nemotron-3.5 drafts and would load and run."""
    target_embed = object()
    worker = SimpleNamespace(
        draft_model=SimpleNamespace(
            get_input_embeddings=lambda: pytest.fail("used the draft's table")
        ),
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(get_input_embeddings=lambda: target_embed)
            )
        ),
    )
    assert target_input_embeddings(worker) is target_embed


def test_publishing_the_anchor_picks_each_requests_last_committed_row():
    ctx_hidden = torch.arange(12, dtype=torch.float32).view(6, 2)
    published = {}
    draft_sampler = SimpleNamespace(
        set_anchor=lambda rows, bs: published.update(rows=rows, bs=bs)
    )

    anchor = publish_anchor(
        draft_sampler=draft_sampler,
        ctx_hidden=ctx_hidden,
        positions=None,
        commit_lens=torch.tensor([2, 1, 3]),
    )
    torch.testing.assert_close(anchor, ctx_hidden[[1, 2, 5]])
    assert published["bs"] == 3

    # Unrecoverable boundaries must clear the graph's buffer rather than leave
    # the previous step's anchor at that address for a padded replay to read.
    assert (
        publish_anchor(
            draft_sampler=draft_sampler,
            ctx_hidden=ctx_hidden,
            positions=None,
            commit_lens=None,
        )
        is None
    )
    assert published["bs"] == 0


# --- grouped convolution coverage ------------------------------------------


class _FakeConv:
    def __init__(self, taps=2, group_size=16):
        self.taps = taps
        self.group_size = group_size


class _FakeDraft:
    """Minimal stand-in: the coverage check only reads parameter names and layers."""

    def __init__(self, n_layers=5, conv=True):
        self._names = []
        self.layers = []
        for i in range(n_layers):
            self._names.append(f"layers.{i}.self_attn.q_proj.weight")
            if conv:
                for w in ("attention_conv", "mlp_conv"):
                    self._names.append(f"layers.{i}.{w}.base_kernel")
                    self._names.append(f"layers.{i}.{w}.kernel_projection.weight")
            self.layers.append(
                SimpleNamespace(attention_conv=_FakeConv())
                if conv
                else SimpleNamespace()
            )

    def named_parameters(self):
        return [(n, None) for n in self._names]

    def conv_names(self):
        return {n for n in self._names if ".attention_conv." in n or ".mlp_conv." in n}


def test_a_conv_draft_declares_four_tensors_per_layer():
    """Two wrapped sublayers, each a base kernel and a projection: 4 per layer."""
    assert len(_FakeDraft(n_layers=5).conv_names()) == 20


def test_matched_conv_checkpoint_and_conv_free_parent_both_pass():
    """The check must be inert on the two configurations that are actually correct."""
    conv = _FakeDraft()
    check_conv_weight_coverage(conv, conv.conv_names())
    check_conv_weight_coverage(_FakeDraft(conv=False), set())


def test_conv_tensors_with_no_conv_built_raises():
    """The silent one: dflash_config defaults both geometry keys to 0, so a
    checkpoint whose config lost them builds no convolution at all and every
    tensor is dropped without a word. The draft then serves as its conv-free
    parent at a lower but entirely believable acceptance length."""
    with pytest.raises(ValueError, match="built no convolution modules"):
        check_conv_weight_coverage(_FakeDraft(conv=False), _FakeDraft().conv_names())


def test_conv_built_with_no_conv_tensors_raises():
    """The other direction serves kernel_projection at its random init."""
    with pytest.raises(ValueError, match="checkpoint carries none"):
        check_conv_weight_coverage(_FakeDraft(), set())


def test_a_partial_conv_checkpoint_raises():
    """A layer count mismatch leaves some tensors resolved and some dropped."""
    draft = _FakeDraft(n_layers=5)
    with pytest.raises(ValueError, match="do not correspond"):
        check_conv_weight_coverage(draft, _FakeDraft(n_layers=4).conv_names())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
