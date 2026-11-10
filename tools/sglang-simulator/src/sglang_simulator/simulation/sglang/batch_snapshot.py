"""Snapshot model work before SGLang's run_batch can mutate prepared inputs."""

from statistics import pvariance

from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest


def capture_batch_entry(batch) -> ScheduleBatch:
    mode = getattr(getattr(batch, "forward_mode", None), "name", "").lower()
    reqs = list(batch.reqs)
    if not reqs:
        return ScheduleBatch(forward_mode=mode)
    if mode not in {"extend", "decode", "mixed"}:
        raise ValueError(f"AISimulate cannot estimate forward mode {mode!r}")
    decoding = set()
    if mode == "mixed":
        decoding = {id(r) for r in (getattr(batch, "decoding_reqs", None) or [])}
        if not decoding or not decoding.issubset({id(r) for r in reqs}):
            raise ValueError("Mixed batch has missing or inconsistent decoding_reqs")
    seq_lens = getattr(batch, "seq_lens_cpu", None)
    if seq_lens is None:
        seq_lens = getattr(batch, "seq_lens", None)
    if mode in {"decode", "mixed"}:
        if seq_lens is None or len(seq_lens) < len(reqs):
            raise ValueError("Decode prediction needs prepared sequence lengths")
        # Beam member rows (batch_tail.append_beam_tail) ride a pure-decode
        # batch after the reqs-aligned rows; they are computed rows, so only
        # that layout may extend seq_lens past len(reqs).
        if len(seq_lens) > len(reqs) and (
            mode != "decode" or getattr(batch, "beam_tail", None) is None
        ):
            raise ValueError("Decode prediction needs prepared sequence lengths")
    rows = []
    # Batch-owned vectors are constant for the whole batch; request fields
    # support older SGLang when the batch does not carry them.
    lengths = getattr(batch, "extend_lens", None)
    prefixes = getattr(batch, "prefix_lens", None)
    for i, req in enumerate(reqs):
        is_decode = mode == "decode" or id(req) in decoding
        if is_decode:
            # Prepared decode lengths include the current query token.
            extend, prior = 1, int(seq_lens[i]) - 1
        else:
            extend = (
                lengths[i]
                if lengths is not None
                else getattr(req, "extend_input_len", None)
            )
            if extend is None:
                extend = getattr(req, "extend_len", None)
            prior = (
                prefixes[i]
                if prefixes is not None
                else getattr(req, "prefix_len", None)
            )
            if extend is None or prior is None:
                raise ValueError("Prefill prediction needs extend and prefix lengths")
            extend, prior = int(extend), int(prior)
        if extend < 1 or prior < 0:
            raise ValueError("Invalid scheduled token counts")
        rows.append(ScheduleRequest(extend, prior, "decode" if is_decode else "extend"))
    # Beam member rows share their leader's prepared decode length; each is
    # one computed decode slot, so the estimator input must count them.
    if mode == "decode":
        for i in range(len(reqs), len(seq_lens)):
            extend, prior = 1, int(seq_lens[i]) - 1
            if extend < 1 or prior < 0:
                raise ValueError("Invalid scheduled token counts")
            rows.append(ScheduleRequest(extend, prior, "decode"))
    return ScheduleBatch(rows, mode)


def batch_to_fpm_metrics(batch: ScheduleBatch) -> dict:
    """Translate scheduled work to native FPM features (prior KV, not total length)."""
    prefill, decode = [], []
    if batch.forward_mode not in {"extend", "decode", "mixed"}:
        raise ValueError("Prediction requires an extend/decode/mixed snapshot")
    for req in batch.reqs:
        phase = (
            req.forward_mode if batch.forward_mode == "mixed" else batch.forward_mode
        )
        if phase not in {"extend", "decode"}:
            raise ValueError("Mixed prediction requires per-request scheduler phases")
        (prefill if phase == "extend" else decode).append(req)
    lengths = [r.extend_length for r in prefill]
    decode_kv = [r.past_kv_length for r in decode]
    return {
        "scheduled_requests": {
            "num_prefill_requests": len(prefill),
            "sum_prefill_tokens": sum(lengths),
            "sum_prefill_kv_tokens": sum(r.past_kv_length for r in prefill),
            "var_prefill_length": pvariance(lengths) if lengths else 0.0,
            "num_decode_requests": len(decode),
            "sum_decode_kv_tokens": sum(decode_kv),
            "var_decode_kv_tokens": pvariance(decode_kv) if decode_kv else 0.0,
            # Request-level SDK features need aligned arrays in scheduler order,
            # including mixed batches; do not concatenate the phase groups.
            "extend_lengths": [r.extend_length for r in batch.reqs],
            "past_kv_lengths": [r.past_kv_length for r in batch.reqs],
        }
    }
