"""Replay TestExtendAutotuneDummy's scenario against any SGLang tree on PYTHONPATH.

Prints the EXTEND dummy maybe_flashinfer_autotune_extend requests for a non-speculative
and a speculative target (4096-token ceiling, 12-slot request pool, pool-capped
dummy), or "no EXTEND dummy". get_disagg is stubbed when the module reads it (the base
commit does), so the result reflects the sizing and speculative logic only.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.runner import flashinfer_autotune as autotune

for speculative in (False, True):
    mr = SimpleNamespace(
        model=SimpleNamespace(),
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_speculative=lambda s=speculative: s),
        attn_backend=SimpleNamespace(extend_dummy_seqs_capped_by_req_pool=True),
        req_to_token_pool=SimpleNamespace(size=12),
    )
    runner = SimpleNamespace(
        model_runner=mr,
        _alloc_dummy_decode_buffers=Mock(return_value=SimpleNamespace()),
        _dummy_run=Mock(),
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(autotune, "max_prefill_buffer_tokens", return_value=4096)
        )
        stack.enter_context(
            patch.object(
                autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND,
                "get",
                return_value=True,
            )
        )
        stack.enter_context(
            patch.object(
                autotune,
                "run_flashinfer_autotune_forward",
                side_effect=lambda mr, forward_fn, run_lm_head: forward_fn(),
            )
        )
        stack.enter_context(patch.object(autotune, "autotune_dummy_run_ctx"))
        stack.enter_context(patch.object(autotune.torch.cuda, "empty_cache"))
        if hasattr(autotune, "get_disagg"):
            stack.enter_context(
                patch.object(
                    autotune,
                    "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode="null"),
                )
            )
        autotune.maybe_flashinfer_autotune_extend(runner, decode_num_tokens=48)
    call = runner._dummy_run.call_args
    if call is None:
        print(f"speculative={speculative}: no EXTEND dummy")
    else:
        kw = call.kwargs
        seqs, per_req = kw["batch_size"], kw["extend_num_tokens_per_req"]
        print(
            f"speculative={speculative}: {seqs} seqs x {per_req} tokens = "
            f"{seqs * per_req} tokens, mode={kw['forward_mode_override'].name}"
        )
