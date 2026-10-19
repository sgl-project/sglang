"""CORAL_SPEC_FUSE: fuse the eager per-step spec-decode bookkeeping (topk=1 MTP) into a few Triton launches.

(a) draft-extend post-processing (EagleDraftWorker._draft_extend_for_decode): stock code builds select_index
    (arange + add + sub), gathers the selected logits rows and hidden rows, argmaxes the draft-head logits and makes
    ones_like probs: ~11 tiny launches. Here: one split-V partial-argmax launch over ONLY the selected rows + one
    finalize launch that also gathers the hidden row. Same result: argmax with first-index tie-break (torch.argmax).
(b) spec_utils._verify_commit_step_indices: ~17 one-element-per-request integer ops -> one launch, same dtypes.
    v0.5.22 already ships an equivalent fused kernel, so that hook is skipped there.
"""
import inspect
import textwrap

import torch
import triton
import triton.language as tl

ARGMAX_BLOCK = 2048


@triton.jit
def _de_partial_argmax(logits_ptr, stride_l, al_ptr, part_v_ptr, part_i_ptr, V, dtn, NK, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    k = tl.program_id(1)
    row = b * dtn + tl.load(al_ptr + b).to(tl.int64) - 1
    offs = k * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(logits_ptr + row * stride_l + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    idx = tl.min(tl.where(x == m, offs, 2147483647), axis=0)            # first index of the max
    tl.store(part_v_ptr + b * NK + k, m)
    tl.store(part_i_ptr + b * NK + k, idx)


@triton.jit
def _de_finalize(part_v_ptr, part_i_ptr, al_ptr, hid_ptr, stride_h, out_idx_ptr, out_p_ptr, out_hid_ptr, D, dtn, NK,
                 NKP: tl.constexpr, DB: tl.constexpr, HAS_HID: tl.constexpr):
    b = tl.program_id(0)
    ks = tl.arange(0, NKP)
    v = tl.load(part_v_ptr + b * NK + ks, mask=ks < NK, other=float("-inf"))
    i = tl.load(part_i_ptr + b * NK + ks, mask=ks < NK, other=2147483647)
    m = tl.max(v, axis=0)
    best = tl.min(tl.where(v == m, i, 2147483647), axis=0)              # chunks are in index order: first max wins
    tl.store(out_idx_ptr + b, best.to(tl.int64))
    tl.store(out_p_ptr + b, 1.0)
    if HAS_HID:
        row = b * dtn + tl.load(al_ptr + b).to(tl.int64) - 1
        for d0 in range(0, D, DB):
            dd = d0 + tl.arange(0, DB)
            tl.store(out_hid_ptr + b * D + dd, tl.load(hid_ptr + row * stride_h + dd, mask=dd < D), mask=dd < D)


def draft_extend_post(logits, hidden, accept_lens, dtn):
    """-> (topk_index [bs,1] int64, topk_p [bs,1] fp32 ones, hidden_sel [bs, D] or None)."""
    bs = accept_lens.shape[0]
    V = logits.shape[-1]
    NK = triton.cdiv(V, ARGMAX_BLOCK)
    dev = logits.device
    pv = torch.empty(bs, NK, device=dev, dtype=torch.float32)
    pi = torch.empty(bs, NK, device=dev, dtype=torch.int32)
    _de_partial_argmax[(bs, NK)](logits, logits.stride(0), accept_lens, pv, pi, V, dtn, NK, BLOCK=ARGMAX_BLOCK)
    out_idx = torch.empty(bs, 1, device=dev, dtype=torch.int64)
    out_p = torch.empty(bs, 1, device=dev, dtype=torch.float32)
    if hidden is not None:
        D = hidden.shape[-1]
        out_h = torch.empty(bs, D, device=dev, dtype=hidden.dtype)
        _de_finalize[(bs,)](pv, pi, accept_lens, hidden, hidden.stride(0), out_idx, out_p, out_h, D, dtn, NK,
                            NKP=triton.next_power_of_2(NK), DB=1024, HAS_HID=True)
    else:
        out_h = None
        _de_finalize[(bs,)](pv, pi, accept_lens, pv, 0, out_idx, out_p, pv, 0, dtn, NK,
                            NKP=triton.next_power_of_2(NK), DB=1024, HAS_HID=False)
    return out_idx, out_p, out_h


@triton.jit
def _commit_idx_kernel(ai_ptr, s0, s1, al_ptr, sl_ptr, last_ptr, trk_ptr, dtn, interval, HAS_TRACK: tl.constexpr):
    i = tl.program_id(0)
    al = tl.load(al_ptr + i).to(tl.int64)
    off = i.to(tl.int64) * dtn
    last = tl.load(ai_ptr + i * s0 + (al - 1) * s1).to(tl.int64) - off
    tl.store(last_ptr + i, last)
    if HAS_TRACK:
        pre = tl.load(sl_ptr + i).to(tl.int64)
        post = pre + al
        cross = (pre // interval) != (post // interval)
        tp = post // interval * interval
        ith = tl.maximum(tp - pre - 1, 0)
        cand = tl.load(ai_ptr + i * s0 + ith * s1).to(tl.int64) - off
        tl.store(trk_ptr + i, tl.where(cross, cand, -1))


def install_commit(mod, log):
    try:
        src = inspect.getsource(mod._verify_commit_step_indices)
    except (OSError, TypeError):
        src = ""
    if "fused_commit_track_indices" in src:
        log("spec fuse: upstream commit path is already fused; not overriding")
        return

    o = mod._verify_commit_step_indices

    def _verify_commit_step_indices(*, batch, accept_index, accept_lens, draft_token_num):
        if accept_index.dim() != 2 or accept_lens.numel() == 0:
            return o(batch=batch, accept_index=accept_index, accept_lens=accept_lens, draft_token_num=draft_token_num)
        bs = accept_lens.shape[0]
        dt = torch.result_type(accept_index, accept_lens)
        last = torch.empty(bs, device=accept_lens.device, dtype=dt)
        track = batch.mamba_track_indices is not None
        trk = torch.empty(bs, device=accept_lens.device, dtype=dt) if track else last
        interval = mod.get_exec().mamba.mamba_track_interval if track else 1
        sl = batch.seq_lens if track else accept_lens
        _commit_idx_kernel[(bs,)](accept_index, accept_index.stride(0), accept_index.stride(1), accept_lens, sl, last,
                                  trk, draft_token_num, interval, HAS_TRACK=track)
        return (last, trk) if track else (last, None)

    mod._verify_commit_step_indices = _verify_commit_step_indices
    log("spec fuse: _verify_commit_step_indices -> 1 launch")


_SEL_OLD = """        select_index = (
            torch.arange("""
_SEL_NEW = """        _coral_ok = _coral_de_ok(self)
        select_index = None if _coral_ok else (
            torch.arange("""
_GATHER_OLD = """        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]"""
_GATHER_NEW = """        if _coral_ok:
            _coral_res = _coral_de_post(draft_logits_output.next_token_logits, draft_logits_output.hidden_states,
                                        batch_result.accept_lens, self.speculative_num_draft_tokens)
            draft_logits_output.hidden_states = _coral_res[2]
        else:
            draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
                select_index
            ]
            if draft_logits_output.hidden_states is not None:
                draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                    select_index
                ]"""
_ARGMAX_OLD = """        elif self.topk == 1 and not _is_hip:"""
_ARGMAX_NEW = """        elif _coral_ok:
            ret_topk_index, ret_topk_p, ret_draft_probs = _coral_res[0], _coral_res[1], None
        elif self.topk == 1 and not _is_hip:"""

# v0.5.22: the GPU path no longer builds select_index in Python and the graph
# runner selects rows before lm_head. Only the eager path still gathers rows.
_V22_REORG_OLD = """        # Reorganize the spec info for the next batch
        if not can_run_decode_cuda_graph:
            draft_logits_output.next_token_logits = (
                draft_logits_output.next_token_logits[select_index]
            )
            if draft_logits_output.hidden_states is not None:
                draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                    select_index
                ]"""
_V22_REORG_NEW = """        # Reorganize the spec info for the next batch
        _coral_ok = _coral_de_ok(self, can_run_decode_cuda_graph)
        _coral_res = None
        if not can_run_decode_cuda_graph:
            if _coral_ok:
                _coral_res = _coral_de_post(draft_logits_output.next_token_logits,
                                            draft_logits_output.hidden_states,
                                            batch_result.accept_lens,
                                            self.speculative_num_draft_tokens)
                draft_logits_output.hidden_states = _coral_res[2]
            else:
                draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
                    select_index
                ]
                if draft_logits_output.hidden_states is not None:
                    draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                        select_index
                    ]"""


def install_draft_extend(mod, log):
    cls = mod.EagleDraftWorker
    fn = cls._draft_extend_for_decode
    fn = getattr(fn, "__wrapped__", fn)
    src = textwrap.dedent(inspect.getsource(fn))
    src = textwrap.indent(src, "    ")                                   # keep the 8-space body indentation of the snippets
    if src.count(_V22_REORG_OLD) == 1 and src.count(_ARGMAX_OLD) == 1:
        src = src.replace(_V22_REORG_OLD, _V22_REORG_NEW).replace(_ARGMAX_OLD, _ARGMAX_NEW)
    elif all(src.count(old) == 1 for old in (_SEL_OLD, _GATHER_OLD, _ARGMAX_OLD)):
        src = src.replace(_SEL_OLD, _SEL_NEW).replace(_GATHER_OLD, _GATHER_NEW).replace(_ARGMAX_OLD, _ARGMAX_NEW)
    else:
        log("spec fuse: draft-extend rewrite DECLINED (source mismatch)")
        return
    src = textwrap.dedent(src)

    def ok(self, graph=False):
        return (not graph and self.topk == 1 and mod._is_cuda
                and not mod.get_spec().speculative_use_rejection_sampling
                and not self.seed_dsa_topk_from_draft_extend)

    g = mod.__dict__                                                     # live module globals (later patches stay visible)
    g["_coral_de_ok"] = ok
    g["_coral_de_post"] = draft_extend_post
    loc = {}
    exec(compile(src, f"<coral_spec_fuse:{mod.__name__}>", "exec"), g, loc)
    cls._draft_extend_for_decode = loc["_draft_extend_for_decode"]
    log("spec fuse: draft-extend post-processing -> 2 launches")
