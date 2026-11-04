"""Snapshot model work before SGLang's run_batch can mutate prepared inputs."""

from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest


def capture_batch_entry(batch) -> ScheduleBatch:
    mode = getattr(getattr(batch, "forward_mode", None), "name", "").lower()
    reqs = list(batch.reqs)
    if not reqs:
        return ScheduleBatch(forward_mode=mode)
    if mode not in {"extend", "decode", "mixed"}:
        raise ValueError(f"Native AISimulate cannot estimate forward mode {mode!r}")
    decoding = set()
    if mode == "mixed":
        decoding = {id(r) for r in (getattr(batch, "decoding_reqs", None) or [])}
        if not decoding or not decoding.issubset({id(r) for r in reqs}):
            raise ValueError("Mixed batch has missing or inconsistent decoding_reqs")
    seq_lens = getattr(batch, "seq_lens_cpu", None)
    if seq_lens is None:
        seq_lens = getattr(batch, "seq_lens", None)
    if mode in {"decode", "mixed"}:
        if seq_lens is None or len(seq_lens) != len(reqs):
            raise ValueError("Decode prediction needs prepared sequence lengths")
    rows = []
    for i, req in enumerate(reqs):
        is_decode = mode == "decode" or id(req) in decoding
        if is_decode:
            # Prepared decode lengths include the current query token.
            extend, prior = 1, int(seq_lens[i]) - 1
        else:
            # Prefer batch-owned fields; request fields support older SGLang.
            lengths = getattr(batch, "extend_lens", None)
            prefixes = getattr(batch, "prefix_lens", None)
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
    return ScheduleBatch(rows, mode)
