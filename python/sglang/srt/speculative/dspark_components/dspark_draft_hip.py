"""ROCm: a DSpark draft step that keeps its vocab shard of the step logits (sharded
accept) computes its folded step bias in the sampling kernel."""

# Rows up to which a step computes its folded bias in the sampling kernel (each
# row's programs read the fold weight block again).
_FOLD_MAX_ROWS = 4


def step_fold(markov_head, prev_tokens):
    """markov_head.step_fold() for a step that keeps its shard of the logits and
    has a few rows: the sampling kernel then computes the folded step bias (its
    GEMM and the candidate pass in one launch). None otherwise."""
    if len(prev_tokens) > _FOLD_MAX_ROWS:
        return None
    return markov_head.step_fold()
