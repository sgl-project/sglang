# Single source of truth for the sgl-eval commit every CI variant installs.
# Meant to be sourced, not executed -- each variant then installs
# "$SGL_EVAL_SPEC" with its own pip invocation, since those differ (uv pip on
# CUDA/CPU, `docker exec ... pip` on AMD, `python3 -m pip` on NPU).
#
# sgl-eval is git-only and cannot be declared in python/pyproject.toml (see the
# note there). Every eval that shells out to the `sgl-eval` CLI fails without
# it, and a bump moves scoring for all of them at once -- so re-baseline
# MODEL_SCORE_THRESHOLDS in
# test/registered/eval/test_text_models_gsm8k_eval.py, and the mmlu thresholds
# of run_eval's other callers, before changing this.
SGL_EVAL_REF="6690895609dcbc5df1e7b00dd57c9502b868ec4d"
# No spaces around the "@": PEP 508 allows omitting them, and a spec with
# spaces breaks any caller that expands $SGL_EVAL_SPEC unquoted (the AMD CI
# installer flattens its argv through `eval`, which re-split this into
# "sgl-eval" / "@" / "git+https://..." and made pip reject a bare "@").
SGL_EVAL_SPEC="sgl-eval@git+https://github.com/sgl-project/sgl-eval.git@${SGL_EVAL_REF}"
