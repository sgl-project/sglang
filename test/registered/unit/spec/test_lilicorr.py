import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.kernels.ops.speculative.lilicorr import (
    lilicorr_greedy_path,
    lilicorr_sample_path,
    lilicorr_topk_lse,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
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
    # Random weights: a head at its zero-initialized construction values scores
    # every candidate identically and would hide a real scoring bug.
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
        "candidate_tokens": torch.randint(0, 64, (bs, slots, topk)),
        "candidate_log_probs": torch.randn(bs, slots, topk).log_softmax(dim=-1),
        "pass_hidden": torch.randn(bs, slots, model_hidden_size),
        "anchor_hidden": torch.randn(bs, model_hidden_size),
        "anchor_valid": torch.ones(bs, dtype=torch.bool),
    }


class _FakeShardedHead(VocabParallelEmbedding):
    """A vocab-parallel head carrying only the fields the shard resolver reads.

    ``VocabParallelEmbedding.__init__`` needs an initialized distributed group.
    """

    def __init__(self, *, num_org, start, added=0):
        nn.Module.__init__(self)
        self.shard_indices = SimpleNamespace(
            num_org_elements=num_org,
            org_vocab_start_index=start,
            num_added_elements=added,
        )


# --- config ----------------------------------------------------------------


@pytest.mark.parametrize(
    "dflash_config", [{}, {"lilicorr_enabled": False}], ids=["absent", "disabled"]
)
def test_no_head_geometry_is_reported_against_the_architecture_string(dflash_config):
    """Absence and an explicit disable are the same case, and neither is a
    per-field error: the checkpoint asked for this head by declaring the
    architecture and then did not say which head."""
    with pytest.raises(ValueError, match="LiLiCorrDraftModel"):
        parse_lilicorr_draft_config(
            draft_hf_config={
                "num_hidden_layers": 2,
                "dflash_config": dflash_config,
            }
        )


@pytest.mark.parametrize("dropped", sorted(set(_GEOMETRY) - {"lilicorr_enabled"}))
def test_every_geometry_field_is_required(dropped):
    """No field may acquire a default. Most change a tensor shape and would be
    caught at weight load, but logit_scale and vector_eps would not: a guessed
    value builds a head that loads cleanly and scores a different function."""
    with pytest.raises(ValueError, match=dropped):
        _lilicorr_config(**{dropped: None})


@pytest.mark.parametrize("topk", [3, 6])
def test_a_candidate_topk_that_is_not_a_power_of_two_is_refused(topk):
    """The tiled candidate top-k holds its selected tiles in one Triton lane
    group, and tl.arange needs a power-of-two extent. Refusing at load beats
    silently falling back to the far slower reference path."""
    with pytest.raises(ValueError, match="power of two"):
        _lilicorr_config(lilicorr_candidate_topk=topk)


@pytest.mark.parametrize("topk", [32, 64])
def test_a_candidate_topk_wider_than_the_fused_kernel_is_refused(topk):
    """A wider pool is a power of two and loads fine, but falls off the fused
    greedy commit onto the torch path at roughly three launches per slot. Config
    parse is where that gets refused, for the same reason as the power-of-two
    case: silently deoptimized is worse than not served."""
    with pytest.raises(ValueError, match="fused greedy commit"):
        _lilicorr_config(lilicorr_candidate_topk=topk)


def test_zero_head_width_means_as_wide_as_the_draft():
    """The exporter records 0 for a head with no token_proj, which is the one
    field allowed to be non-positive."""
    config = _lilicorr_config(lilicorr_hidden_size=0)
    assert config.resolve_hidden_size(model_hidden_size=16) == 16
    assert isinstance(_head(lilicorr_hidden_size=0).token_proj, nn.Identity)


# --- the head --------------------------------------------------------------


def test_head_parameter_names_match_the_exported_checkpoint_subtree():
    """Pins weight compatibility with the training export. A renamed submodule
    loads nothing under that name, and the base loader ignores what it cannot
    resolve, so the head would serve its construction values."""
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
    assert (selected.unsqueeze(-1) == lattice["candidate_tokens"]).any(-1).all()


def test_an_invalid_anchor_ignores_whatever_is_in_the_buffer():
    """An invalid anchor is zeroed by multiplication rather than by a branch, so
    the captured graph needs no host sync. The graph replays at the padded bucket
    batch size, so rows past the live batch read stale anchor memory and their
    scores must not depend on it."""
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
    must be the same function of the token."""
    head = _head()
    lattice = _lattice(head)
    embed_tokens = nn.Embedding(64, 16)
    with torch.no_grad():
        embed_tokens.weight.normal_(0.0, 0.2)
    table = head.build_token_table(embed_tokens)
    assert table is not None and table.shape == (64, head.hidden_size)

    tokens = lattice["candidate_tokens"]
    raw = head.select(**{**lattice, "token_embeddings": embed_tokens(tokens).detach()})
    folded = head.select(
        **{**lattice, "token_embeddings": table[tokens]}, already_projected=True
    )
    torch.testing.assert_close(raw, folded)


def test_head_weight_coverage_is_required_in_both_directions():
    """The base loader ignores what it cannot resolve, so both a missing tensor
    and a surplus one are silent, and either produces a low but believable
    acceptance length."""
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
    vals, tokens, lse = lilicorr_topk_lse(logits, 5)
    expected_vals, expected_tokens = torch.log_softmax(logits, dim=-1).topk(5, dim=-1)
    torch.testing.assert_close(vals - lse.unsqueeze(-1), expected_vals)
    torch.testing.assert_close(tokens, expected_tokens.to(torch.int64))


def test_greedy_path_follows_the_conditioned_argmax_recurrence():
    """c_0 = argmax(start), then c_s = argmax_c pair[s-1, c_{s-1}, c]."""
    torch.manual_seed(0)
    bs, slots, k = 3, 4, 4
    log_start = torch.randn(bs, k)
    log_pair = torch.randn(bs, slots - 1, k, k)
    tokens = torch.randint(0, 100, (bs, slots, k))

    actual = lilicorr_greedy_path(log_start, log_pair, tokens)

    expected = torch.empty(bs, slots, dtype=tokens.dtype)
    for row in range(bs):
        cur = int(log_start[row].argmax())
        expected[row, 0] = tokens[row, 0, cur]
        for slot in range(1, slots):
            cur = int(log_pair[row, slot - 1, cur].argmax())
            expected[row, slot] = tokens[row, slot, cur]
    torch.testing.assert_close(actual, expected)


def test_greedy_path_breaks_ties_toward_the_lower_candidate():
    """Ties must break toward the lower index, which is what makes the fused
    kernel and the torch path commit the same tokens."""
    log_start = torch.zeros(1, 4)
    log_pair = torch.zeros(1, 2, 4, 4)
    tokens = torch.arange(12).view(1, 3, 4)
    torch.testing.assert_close(
        lilicorr_greedy_path(log_start, log_pair, tokens),
        torch.tensor([[0, 4, 8]]),
    )


# --- candidates ------------------------------------------------------------


def test_candidates_are_normalized_log_probs_with_global_tokens():
    torch.manual_seed(0)
    hidden = torch.randn(6, 8)
    weight = torch.randn(50, 8)
    log_probs, tokens = lilicorr_candidates(
        hidden_states=hidden,
        weight=weight,
        num_org=40,
        org_vocab_start=100,
        topk=4,
    )
    reference = torch.log_softmax(torch.matmul(hidden, weight[:40].T), dim=-1)
    expected_vals, expected_tokens = reference.topk(4, dim=-1)
    torch.testing.assert_close(log_probs, expected_vals)
    torch.testing.assert_close(tokens, expected_tokens.to(torch.int64) + 100)


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
    rank0_vals, rank0_tokens = rank0_logits.topk(topk, dim=-1)
    rank0_lse = torch.logsumexp(rank0_logits, dim=-1)

    class _FakeTpGroup:
        world_size = 2

        def all_gather_into_tensor(self, output, packed):
            rows = packed.numel() // (2 * topk + 1)
            mine = packed.view(rows, 2 * topk + 1)
            theirs = torch.empty_like(mine)
            theirs[:, :topk] = rank0_vals
            theirs[:, topk : 2 * topk] = rank0_tokens.to(torch.float32)
            theirs[:, 2 * topk] = rank0_lse
            output.copy_(torch.cat([theirs, mine], dim=0).view(-1))

    log_probs, tokens = lilicorr_candidates(
        hidden_states=hidden,
        weight=full_weight[12:],
        num_org=12,
        org_vocab_start=12,
        topk=topk,
        tp_group=_FakeTpGroup(),
    )
    reference = torch.log_softmax(torch.matmul(hidden, full_weight.T), dim=-1)
    expected_vals, expected_tokens = reference.topk(topk, dim=-1)
    torch.testing.assert_close(log_probs, expected_vals)
    torch.testing.assert_close(tokens, expected_tokens.to(torch.int64))


def test_vocab_shard_resolution_and_added_vocab_refusal():
    assert resolve_vocab_shard(SimpleNamespace(weight=torch.empty(32, 4))) == (32, 0)
    assert resolve_vocab_shard(_FakeShardedHead(num_org=16, start=16)) == (16, 16)
    with pytest.raises(NotImplementedError, match="added vocabulary"):
        resolve_vocab_shard(_FakeShardedHead(num_org=16, start=0, added=2))


# --- the anchor ------------------------------------------------------------


def test_verify_anchor_rows_follow_the_padded_block_stride():
    """The verify buffer is [bs, block_size] flattened, so requests sit at a
    constant stride and only the first commit_lens[i] rows of each are live.
    Reading it as packed picks another request's row for every request after the
    first, which costs acceptance and raises nothing."""
    # bs=2, block_size=16: request 1 starts at row 16 however much request 0 committed.
    torch.testing.assert_close(
        per_request_last_row(
            num_rows=32, extend_lens=None, commit_lens=torch.tensor([3, 5])
        ),
        torch.tensor([2, 20]),
    )
    # Single request: padded and packed readings coincide, which is why a
    # concurrency-1 benchmark cannot see the difference.
    torch.testing.assert_close(
        per_request_last_row(
            num_rows=16, extend_lens=None, commit_lens=torch.tensor([7])
        ),
        torch.tensor([6]),
    )


def test_prefill_anchor_rows_come_from_extend_lens_not_from_positions():
    """Prefill rows are packed request-major. The lengths are passed in rather
    than inferred from positions, because with a cached prefix a request's
    positions start mid-sequence and never reset -- [0, 1, 10, 11] is two
    requests of two tokens, which no reset detector can see."""
    torch.testing.assert_close(
        per_request_last_row(
            num_rows=6, extend_lens=torch.tensor([3, 2, 1]), commit_lens=None
        ),
        torch.tensor([2, 4, 5]),
    )
    torch.testing.assert_close(
        per_request_last_row(
            num_rows=4, extend_lens=torch.tensor([2, 2]), commit_lens=None
        ),
        torch.tensor([1, 3]),
    )


def test_unrecoverable_anchor_rows_return_none_rather_than_a_guess():
    """A wrong anchor is a silent acceptance regression, so an input the
    boundaries cannot be read from must leave the anchor unset."""
    assert per_request_last_row(num_rows=4, extend_lens=None, commit_lens=None) is None
    # Lengths that do not account for every row cannot locate them.
    assert (
        per_request_last_row(
            num_rows=4, extend_lens=torch.tensor([1, 1]), commit_lens=None
        )
        is None
    )
    # A row count that is not a whole number of blocks is not the padded layout.
    assert (
        per_request_last_row(
            num_rows=7, extend_lens=None, commit_lens=torch.tensor([2, 2])
        )
        is None
    )


def test_publishing_the_anchor_selects_the_padded_rows():
    # bs=2 at block_size=4, so the live rows are 1 and 4+3-1=6.
    ctx_hidden = torch.arange(16, dtype=torch.float32).view(8, 2)
    published = {}
    draft_sampler = SimpleNamespace(
        set_anchor=lambda rows, bs: published.update(rows=rows, bs=bs)
    )

    anchor = publish_anchor(
        draft_sampler=draft_sampler,
        ctx_hidden=ctx_hidden,
        commit_lens=torch.tensor([2, 3]),
    )
    torch.testing.assert_close(anchor, ctx_hidden[[1, 6]])
    assert published["bs"] == 2

    # Unrecoverable boundaries must clear the graph's buffer rather than leave the
    # previous step's anchor at that address for a padded replay to read.
    assert publish_anchor(draft_sampler=draft_sampler, ctx_hidden=ctx_hidden) is None
    assert published["bs"] == 0


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
    there is nothing to size them from. Building against a guessed size would
    serve a head whose buffers do not match the replay."""
    from sglang.srt.speculative.lilicorr_components import (
        lilicorr_draft_sampler as sampler_mod,
    )

    monkeypatch.setattr(
        sampler_mod, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(sampler_mod, "draft_graph_batch_sizes", lambda: [])
    assert (
        sampler_mod.build_lilicorr_draft_sampler(
            head=object(),
            draft_model=SimpleNamespace(),
            embed_tokens=None,
            lm_head=SimpleNamespace(),
            block_size=5,
        )
        is None
    )


def test_a_lilicorr_head_is_dispatched_to_the_folded_sampler():
    """The head must reach the graph fold rather than the eager fallback: eager
    costs a large fraction of throughput, so a silent demotion would read as a
    believable but wrong throughput number."""
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    lm_head = SimpleNamespace(weight=torch.empty(16, 4))
    embed_tokens = object()
    head = object()
    built = {}
    worker = SimpleNamespace(
        block_size=5,
        selector=None,
        lilicorr=head,
        ps=SimpleNamespace(tp_rank=0),
        draft_model=SimpleNamespace(lm_head=None),
        device="cpu",
        # The name the surrounding DFLASH code reads, not ours.
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    lm_head=lm_head, get_input_embeddings=lambda: embed_tokens
                )
            )
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
    assert built["kwargs"]["head"] is head
    # The head embeds candidate tokens with the target's table, which is the one
    # it was trained against; the draft's own table exists on Nemotron-3.5 drafts
    # and would load, run, and score the wrong function.
    assert built["kwargs"]["embed_tokens"] is embed_tokens


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


# --- sampled commit --------------------------------------------------------


def _sampled_lattice(bs=2, slots=3, k=4, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(bs, k),
        torch.randn(bs, slots - 1, k, k),
        torch.arange(bs * slots * k).view(bs, slots, k),
    )


def test_a_greedy_masked_row_walks_the_greedy_path_bit_identically():
    """The claim every published acceptance number rests on: with the mask set the
    sampled kernel must commit the same tokens as ``lilicorr_greedy_path``, not
    merely similar ones."""
    log_start, log_pair, tokens = _sampled_lattice()
    picked, _ = lilicorr_sample_path(
        log_start,
        log_pair,
        tokens,
        uniforms=torch.rand(2, 3),
        temperatures=torch.full((2,), 0.7),
        greedy_mask=torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(picked, lilicorr_greedy_path(log_start, log_pair, tokens))


def test_the_sampled_commit_converges_on_the_greedy_path_as_temperature_vanishes():
    """Same claim by the other route, with the mask off: the proposal is
    ``softmax(psi / T)``, so a vanishing temperature must reproduce the argmax. This
    is what makes the sampled path a superset rather than a different drafter."""
    log_start, log_pair, tokens = _sampled_lattice(seed=1)
    picked, _ = lilicorr_sample_path(
        log_start,
        log_pair,
        tokens,
        uniforms=torch.full((2, 3), 0.5),
        temperatures=torch.full((2,), 1e-4),
        greedy_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(picked, lilicorr_greedy_path(log_start, log_pair, tokens))


def test_a_greedy_row_reports_a_point_mass_on_the_token_it_committed():
    """Verify computes ``min(1, p/q)``. Handing a greedy row its temperature softmax
    instead of a point mass would make that the wrong test for that row and would
    perturb the output distribution."""
    log_start, log_pair, tokens = _sampled_lattice(bs=1, seed=2)
    picked, q = lilicorr_sample_path(
        log_start,
        log_pair,
        tokens,
        uniforms=torch.rand(1, 3),
        temperatures=torch.full((1,), 1.0),
        greedy_mask=torch.ones(1, dtype=torch.bool),
    )
    torch.testing.assert_close(q.sum(-1), torch.ones(1, 3))
    assert torch.equal(q.max(-1).values, torch.ones(1, 3))
    committed = q.argmax(-1)
    assert torch.equal(torch.gather(tokens, 2, committed.unsqueeze(-1)).squeeze(-1), picked)


def test_the_proposal_is_a_distribution_over_that_slots_candidates():
    """Rejection sampling is only lossless if ``q`` is the distribution the token was
    actually drawn from. A ``q`` correct only up to a renormalization would accept at
    the wrong rate, silently and in the flattering direction."""
    log_start, log_pair, tokens = _sampled_lattice(bs=3, slots=4, seed=3)
    _, q = lilicorr_sample_path(
        log_start,
        log_pair,
        tokens,
        uniforms=torch.rand(3, 4),
        temperatures=torch.tensor([0.5, 1.0, 2.0]),
        greedy_mask=torch.zeros(3, dtype=torch.bool),
    )
    torch.testing.assert_close(q.sum(-1), torch.ones(3, 4))
    assert (q >= 0).all()


def test_a_sampling_row_leaves_a_greedy_row_in_the_same_batch_unchanged():
    """Rows are independent, and a kernel leaking the committed predecessor or the
    temperature across programs would still look correct on a uniform batch."""
    log_start, log_pair, tokens = _sampled_lattice(bs=2, seed=4)
    mixed_tokens, mixed_q = lilicorr_sample_path(
        log_start,
        log_pair,
        tokens,
        uniforms=torch.full((2, 3), 0.9),
        temperatures=torch.full((2,), 1.5),
        greedy_mask=torch.tensor([True, False]),
    )
    alone_tokens, alone_q = lilicorr_sample_path(
        log_start[:1],
        log_pair[:1],
        tokens[:1],
        uniforms=torch.full((1, 3), 0.9),
        temperatures=torch.full((1,), 1.5),
        greedy_mask=torch.ones(1, dtype=torch.bool),
    )
    assert torch.equal(mixed_tokens[:1], alone_tokens)
    torch.testing.assert_close(mixed_q[:1], alone_q)
