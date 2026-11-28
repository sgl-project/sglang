"""ROCm: the Inkling2 MTP draft's Markov RNN step in two launches from a persistent
zero initial state, and the f_rnn store after verify in one launch."""

from sglang.kernels.ops.speculative.inkling2_mtp_hip import (
    mix_rmsnorm_last,
    rnn_step,
    rnn_step_fits,
)


def run_rnn_step(head, state, prev_tokens, hidden, f_rnn):
    """Inkling2MTPHead._rnn_step in two launches (rnn_step) instead of rnn_inputs,
    hipBLASLt's split-K GEMM and its reduction, and rnn_gates; None when
    rnn_step_fits rules it out (more than a few rows)."""
    proj = head.rnn_proj
    if not rnn_step_fits(state, hidden, proj.weight, proj.bias):
        return None
    return rnn_step(
        state, prev_tokens, head.markov_w1.weight, hidden, f_rnn, proj.weight, proj.bias
    )


# Rows up to which a step computes its folded bias in the sampling kernel (each
# row's programs read the fold weight block again).
_FOLD_MAX_ROWS = 4


def folds_step_bias(head, prev_tokens):
    """Whether the sampling kernel computes this step's folded bias (its GEMM and
    the candidate pass in one launch): a few rows over the rank's vocab shard."""
    return (
        head.vocab_shard is not None
        and head._fold_weight is not None
        and len(prev_tokens) <= _FOLD_MAX_ROWS
    )


# Rows of the persistent zero initial RNN state (larger draft batches fill their own).
_ZERO_STATE_ROWS = 1024


def attach_zero_state(head, weight):
    """Makes Inkling2MTPHead's initial RNN state once, when the fold is attached,
    rather than filling it in every draft (no step writes into its input state)."""
    head._hip_zero_state = weight.new_zeros((_ZERO_STATE_ROWS, head.markov_rank))


def zero_state(head, hidden_states):
    """The initial RNN state for hidden_states' rows from attach_zero_state's
    buffer (attach_shared_modules attaches it before init_state can run), or None
    when it does not cover them."""
    zeros = head._hip_zero_state
    rows = hidden_states.shape[0]
    if (
        rows > zeros.shape[0]
        or zeros.dtype != hidden_states.dtype
        or zeros.device != hidden_states.device
    ):
        return None
    return zeros[:rows]


def store_last_f_rnn(model, req_pool_indices, target_hidden, commit_lens):
    """store_f_rnn of row commit_lens[r] - 1 of each request's block of
    target_hidden (requests, block, features): the gather, the norm and the copy."""
    assert model._f_rnn_buf is not None and model._fuse_alpha is not None
    reqs, width = target_hidden.shape[:2]
    targets = target_hidden.view(
        reqs * width, model.num_context_features, model.config.hidden_size
    )
    norm = model.markov_head.dflare_norm
    mix_rmsnorm_last(
        targets,
        width,
        commit_lens,
        req_pool_indices.view(-1),
        model._fuse_alpha[-1].to(targets.dtype),
        norm.weight,
        norm.variance_epsilon,
        model._f_rnn_buf,
    )
