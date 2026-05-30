"""Dump helper to put on 217 at hardware_backend/npu/attention/_dsv4_dump.py"""
import os, sys

def _dsv4_dump_once(tag, kwargs):
    flag_path = "/tmp/_dsv4_dump_" + tag
    if os.path.exists(flag_path):
        return None
    try:
        open(flag_path, "w").close()
    except Exception:
        pass
    rank = os.environ.get("RANK", "?")
    lines = [f"[DSV4-DUMP {tag} rank={rank}]"]
    try:
        import torch as _t
    except Exception:
        _t = None
    for k, v in kwargs.items():
        try:
            if _t is not None and isinstance(v, _t.Tensor):
                lines.append(
                    f"  {k} = Tensor(shape={tuple(v.shape)} dtype={v.dtype} device={v.device})"
                )
                vals = v.detach().cpu()
                vstr = (
                    str(vals.tolist())
                    if vals.numel() <= 64
                    else str(vals.tolist())[:300] + "..."
                )
                lines.append(f"    vals={vstr}")
            else:
                lines.append(f"  {k} = {v!r}")
        except Exception as ex:
            lines.append(f"  {k} = <dump fail: {ex}>")
    msg = "\n".join(lines)
    print(msg, flush=True)
    try:
        sys.stderr.write(msg + "\n"); sys.stderr.flush()
    except Exception:
        pass
    return None


# =====================================================================
# Per graph-replay ForwardMetadata dump (mainbase vs reference compare)
# ---------------------------------------------------------------------
# Call site: deepseek_v4_ascend_backend.py, just before
#   `self.forward_metadata = fm` at the end of
#   init_forward_metadata_replay_cuda_graph (~line 493).
#
# Gated by env DSV4_REPLAY_DUMP=1 (no-op otherwise). DEBUG ONLY:
# the .cpu()/.tolist() calls force a device sync -> turn OFF for any
# performance / accuracy run.
#
# Optional filters (same env names must be used in the reference repo
# so the two logs line up):
#   DSV4_DUMP_DP_RANK=<int>    only print this attention-dp rank
#   DSV4_DUMP_SEQ_MIN=<int>    only when max(seq_lens) >= MIN
#   DSV4_DUMP_SEQ_MAX=<int>    only when max(seq_lens) <= MAX
#   DSV4_DUMP_MAX_STEPS=<int>  stop printing after N non-idle replays / dp
#
# Naming is canonical (identical field names in both repos) so blocks
# can be diffed directly. Alignment key across the two runs:
# (dp, seq_lens) — req_pool_indices / pool slot ids / page ids DIFFER
# between runs and are NOT value-comparable across repos. Only the
# "semantic" group (seqused / start_pos / positions_cmp_padding_* /
# actual_seq_lengths_*) is value-comparable; loc/page_table fields are
# only structurally comparable (nonzero count / n_valid_per_req).
# =====================================================================

_DSV4_REPLAY_STEP = {}


def _dsv4_rank_dp():
    rank = os.environ.get("RANK")
    if rank is None:
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                rank = _dist.get_rank()
        except Exception:
            pass
    if rank is None:
        rank = "?"
    dp = "?"
    try:
        from sglang.srt.layers.dp_attention import get_attention_dp_rank
        dp = get_attention_dp_rank()
    except Exception:
        pass
    return rank, dp


def _dsv4_mode_label(forward_mode):
    """Return a short mode label, or None if idle/empty (-> skip dump)."""
    if forward_mode is None:
        return None
    is_idle = getattr(forward_mode, "is_idle", None)
    try:
        if callable(is_idle) and is_idle():
            return None
    except Exception:
        pass
    for name, lbl in (
        ("is_decode", "decode"),
        ("is_target_verify", "target_verify"),
        ("is_draft_extend", "draft_extend"),
        ("is_extend", "extend"),
    ):
        fn = getattr(forward_mode, name, None)
        try:
            if callable(fn) and fn():
                return lbl
        except Exception:
            pass
    return str(forward_mode)


def dump_replay_metadata(fm, forward_mode, bs, req_pool_indices, seq_lens):
    if os.environ.get("DSV4_REPLAY_DUMP", "0") != "1":
        return
    mode = _dsv4_mode_label(forward_mode)
    if mode is None:
        return  # forward_mode empty/idle -> skip
    try:
        import torch as _t  # noqa: F401
    except Exception:
        return

    rank, dp = _dsv4_rank_dp()

    def _list(v):
        return v.detach().cpu().tolist()

    try:
        seq_list = _list(seq_lens)[:bs]
    except Exception:
        seq_list = None

    # DP-attention idle/padding ranks run mode=decode with seq_lens ALL ZERO
    # (they are NOT forward_mode.is_idle()). Skip them so only the rank(s)
    # actually holding a request print — otherwise every DP rank logs each step.
    if not seq_list or max(seq_list) <= 0:
        return

    # per-dp real-work replay index (only counts steps past the all-zero guard),
    # so DSV4_DUMP_MAX_STEPS caps real decode steps on the active rank.
    key = str(dp)
    step = _DSV4_REPLAY_STEP.get(key, 0)
    _DSV4_REPLAY_STEP[key] = step + 1

    want_dp = os.environ.get("DSV4_DUMP_DP_RANK")
    if want_dp is not None and str(dp) != str(want_dp):
        return

    smax = max(seq_list)
    lo = os.environ.get("DSV4_DUMP_SEQ_MIN")
    hi = os.environ.get("DSV4_DUMP_SEQ_MAX")
    try:
        if lo is not None and smax < int(lo):
            return
        if hi is not None and smax > int(hi):
            return
    except Exception:
        pass

    # boundary-only: print only steps where some req hits a c4/c128 compression
    # boundary (seq_len % ratio == 0, i.e. a compressed token is produced this
    # step). DSV4_DUMP_BOUNDARY_ONLY=128 -> c128 events (rare, long-range),
    # =4 -> c4 events, =1/other -> c4-or-c128 (c4 is a superset of c128).
    bnd = os.environ.get("DSV4_DUMP_BOUNDARY_ONLY")
    if bnd:
        try:
            _r = int(bnd)
        except Exception:
            _r = 4
        if _r not in (4, 128):
            _r = 4
        if not any((s > 0 and s % _r == 0) for s in seq_list):
            return

    cap = os.environ.get("DSV4_DUMP_MAX_STEPS")
    try:
        if cap is not None and step >= int(cap):
            return
    except Exception:
        pass

    try:
        rpi = _list(req_pool_indices)[:bs]
    except Exception:
        rpi = None

    lines = [
        f"[DSV4-REPLAY rank={rank} dp={dp} step={step} bs={bs} mode={mode} "
        f"seq_lens={seq_list} req_pool={rpi}]"
    ]

    def emit_1d(name):
        v = getattr(fm, name, None)
        if v is None:
            return
        try:
            lines.append(f"  {name}={_list(v)}")
        except Exception as e:
            lines.append(f"  {name}=<err {e}>")

    def emit_loc(name):
        v = getattr(fm, name, None)
        if v is None:
            return
        try:
            t = v.detach().cpu()
            nz = int((t != 0).sum().item())
            head = t.flatten()[:8].tolist()
            lines.append(f"  {name}: shape={tuple(v.shape)} nonzero={nz} head={head}")
        except Exception as e:
            lines.append(f"  {name}: <err {e}>")

    def emit_pt(name, sentinel):
        v = getattr(fm, name, None)
        if v is None:
            return
        try:
            t = v.detach().cpu()[:bs]
            valid = (t != sentinel).sum(dim=1).tolist()
            # actual non-sentinel (normally-allocated) page ids per req, in order
            # (capped so a stray full row can't blow up the log).
            vals = []
            for r in range(t.shape[0]):
                row = t[r]
                pv = row[row != sentinel].tolist()
                if len(pv) > 32:
                    pv = pv[:32] + ["+%d" % (len(pv) - 32)]
                vals.append(pv)
            lines.append(
                f"  {name}: shape={tuple(v.shape)} sentinel={sentinel} "
                f"n_valid_per_req={valid} vals={vals}"
            )
        except Exception as e:
            lines.append(f"  {name}: <err {e}>")

    def emit_kmeta():
        km = getattr(fm, "kernel_metadata", None)
        if not km or not hasattr(km, "get"):
            return
        for k in ("c1a_metadata", "c4a_metadata", "c128a_metadata", "li_quant_metadata"):
            v = km.get(k)
            if v is None:
                continue
            try:
                lines.append(
                    f"  kmeta.{k}: shape={tuple(v.shape)} "
                    f"sum={int(v.detach().cpu().sum().item())}"
                )
            except Exception as e:
                lines.append(f"  kmeta.{k}: <err {e}>")

    # semantic group (value-comparable across runs/repos)
    for n in (
        "seqused",
        "start_pos",
        "positions_cmp_padding_c4",
        "positions_cmp_padding_c128",
        "actual_seq_lengths_kv",
        "actual_seq_lengths_q_pa",
    ):
        emit_1d(n)
    # loc group (structural only: pool slot ids differ across runs)
    for n in ("c4_loc", "c128_loc", "c4_state_loc", "c128_state_loc", "swa_loc"):
        emit_loc(n)
    # page-table group (structural: page ids differ; compare n_valid pattern)
    emit_pt("c4_page_table", -1)
    emit_pt("c128_page_table", -1)
    emit_pt("c4_state_page_table", 0)  # mainbase fills state tables with 0
    emit_pt("c128_state_page_table", 0)
    emit_pt("swa_page_table", -1)
    emit_kmeta()

    # single stream only (avoid 2x duplication when caller merges stdout+stderr)
    print("\n".join(lines), flush=True)
    return None


def dump_hidden_state(tag, hs, forward_batch, layer_id=None):
    """Per-layer / per-token hidden_state digest for mainbase-vs-khalil
    numerical-divergence bisection (EAGER mode). Gated by env HS_DUMP=1.
    Skips DP-idle batches (only the rank holding a real request prints).

    Per call prints, for hidden_states hs of shape [T, ...]:
      - gnorm  : global L2 norm over all T tokens
      - gmean  : mean(|.|)
      - probe  : last token's first 8 raw fp values (exact, for fp-level compare)
      - toknorm: per-token L2 norm (capped)

    Compare two repos by (tag, L, token index): the first layer/token whose
    gnorm/probe differ beyond fp noise is the op that diverges.
    Optional: HS_DUMP_DP=<dp> to restrict to one attention-dp rank.
    """
    if os.environ.get("HS_DUMP", "0") != "1":
        return
    try:
        import torch as _t  # noqa: F401
    except Exception:
        return
    fm = getattr(forward_batch, "forward_mode", None)
    try:
        if fm is not None and hasattr(fm, "is_idle") and fm.is_idle():
            return
    except Exception:
        pass
    # DP-attention idle/padding ranks run mode=decode (or a dummy extend) but are
    # NOT forward_mode.is_idle(): their seq_lens are all-zero. WITHOUT this guard
    # every one of the 16 DP ranks dumps each step (the 15 idle ranks chase the
    # all-to-all with a dummy forward) -> 16x host objects -> OOM-kills the server
    # at ~18k lines. Mirror of the dump_replay_metadata guard: only the rank that
    # actually holds the request prints, which is also the clean single-rank data
    # we want for the cross-repo comparison.
    try:
        _sl = getattr(forward_batch, "seq_lens", None)
        if _sl is not None:
            _slist = _sl.detach().cpu().tolist()
            if not _slist or max(_slist) <= 0:
                return
    except Exception:
        pass
    # prefill-only (default): skip decode dumps. Decode is the bulk and the
    # cumulative .cpu()/.tolist() host objects OOM-kill the server; prefill
    # (same prompt) is the clean cross-repo comparison anyway.
    # Set HS_DUMP_PREFILL_ONLY=0 to also dump decode.
    if os.environ.get("HS_DUMP_PREFILL_ONLY", "0") == "1":
        try:
            if _dsv4_mode_label(fm) == "decode":
                return
        except Exception:
            pass
    if hs is None:
        return
    try:
        h = hs.detach()
        if h.numel() == 0 or h.shape[0] == 0:
            return
        T = h.shape[0]
        # Skip huge-T forwards (startup memory-profiling / chunked prefill):
        # the .float() copy would OOM and they are noise vs the real request.
        if T > int(os.environ.get("HS_DUMP_MAX_T", "1024") or 1024):
            return
        hf = h.float().reshape(T, -1)
        rank, dp = _dsv4_rank_dp()
        want_dp = os.environ.get("HS_DUMP_DP")
        if want_dp is not None and str(dp) != str(want_dp):
            return
        mode = _dsv4_mode_label(fm) or "?"
        # lightweight digest: scalars (sum/norm/mean) + small slices only.
        # NO full per-token list -> keeps each line tiny so decode dumps don't
        # pile up host objects and OOM-kill the server. sum is the sensitive
        # cross-repo divergence signal; head/tail are exact fp values to eyeball.
        ssum = hf.sum().item()
        gnorm = hf.norm().item()
        gmean = hf.abs().mean().item()
        head = ["%.6e" % v for v in hf.reshape(-1)[:8].tolist()]
        tail = ["%.6e" % v for v in hf[-1, :8].tolist()]
        print(
            f"[HS dp={dp} mode={mode} {tag} L={layer_id} T={T} "
            f"sum={ssum:.6e} gnorm={gnorm:.6e} gmean={gmean:.6e} head={head} tail={tail}",
            flush=True,
        )
    except Exception as e:
        print(f"[HS-ERR {tag} L={layer_id}: {e}", flush=True)
    return None


def dump_moe_gate(layer_id, forward_batch, router_logits, topk_ids, topk_weights):
    """MoE routing bisection: per-layer gate output + selected experts, to find
    whether the L4 divergence is a top-k ROUTING FLIP (different experts) vs an
    expert-FFN numeric diff. Gated by HS_DUMP=1, layers <=5, active rank only.

    Prints:
      - rlsum    : router_logits (gate output) sum — sensitive cross-repo signal
                   for whether the GATE computation itself diverges.
      - idsum    : sum of all selected expert ids (cheap global routing digest).
      - argmaxsum: sum of per-token argmax(router_logits) — pure top-1 routing.
      - tok0/tokN: first/last token's sorted top-k expert ids (order-independent).
    [MOEIDS] (L<=4): full sorted top-k ids flattened, for exact offline diff to
    pinpoint the FIRST layer whose expert SELECTION differs mb-vs-khalil.
    """
    if os.environ.get("HS_DUMP", "0") != "1":
        return
    try:
        import torch as _t  # noqa: F401
    except Exception:
        return
    fm = getattr(forward_batch, "forward_mode", None)
    mode = _dsv4_mode_label(fm) or "?"
    rank, dp = _dsv4_rank_dp()
    want_dp = os.environ.get("HS_DUMP_DP")
    if want_dp is not None and str(dp) != str(want_dp):
        return
    try:
        if topk_ids is None:
            return
        ids = topk_ids.detach()
        if ids.numel() == 0 or ids.shape[0] == 0:
            return
        T = ids.shape[0]
        if T > int(os.environ.get("HS_DUMP_MAX_T", "1024") or 1024):
            return
        ids_sorted = ids.sort(dim=-1).values  # order-independent within top-k
        idsum = int(ids.sum().item())
        rlsum = float(router_logits.detach().float().sum().item())
        argmaxsum = int(router_logits.detach().argmax(dim=-1).sum().item())
        wsum = (
            float(topk_weights.detach().float().sum().item())
            if topk_weights is not None
            else 0.0
        )
        tok0 = ids_sorted[0].tolist()
        tokN = ids_sorted[-1].tolist()
        print(
            f"[MOEGATE dp={dp} mode={mode} L={layer_id} T={T} "
            f"rlsum={rlsum:.6e} idsum={idsum} argmaxsum={argmaxsum} "
            f"wsum={wsum:.6e} tok0={tok0} tokN={tokN}",
            flush=True,
        )
        if layer_id <= 4:
            print(
                f"[MOEIDS dp={dp} mode={mode} L={layer_id} "
                f"ids={ids_sorted.reshape(-1).tolist()}]",
                flush=True,
            )
    except Exception as e:
        print(f"[MOEGATE-ERR L={layer_id}: {e}", flush=True)
    return None
