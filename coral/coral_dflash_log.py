"""Per-step DFlash2 selector confidence vs verify outcome, for fitting an adaptive verify length offline
(CORAL_DFLASH_LOG=<path prefix>). MEASUREMENT ONLY: it syncs the GPU once per step, so never time a run with it on.

Wraps DFlashWorkerV2._accept_block (SGLang v0.5.21). After the stock accept, for each request in the batch it
records, for draft positions p = 1..gamma (gamma = block_size - 1):
  Probabilities are the selector's T=1 transition softmax along the path it walked (q_soft, captured into the
  draft CUDA graph by the _SelectorDraftSampler wrapper below). Its own q_out is a point mass under greedy.
  q_draft[p]   selector probability of the token it drafted at p (-1 if that token is not among its candidates,
               e.g. the eager fallback ran and the captured buffers are stale -- such steps are flagged)
  q1[p], q2[p] the selector's top-1 / top-2 candidate probabilities at p
  trank[p]     rank (0 = top) of the TARGET's token at p among the selector's top-k candidates, -1 if absent.
               Only meaningful while the prefix up to p-1 was accepted (the target's prediction is conditioned
               on the drafted prefix); the offline analysis uses p <= accept_len + 1.
  accept       accepted drafts (0..gamma); prefix = context length before the block
Greedy path only (target_predict comes from the accept); sampled requests are skipped.

One JSON line per step and request: <path>.<pid>.jsonl, flushed every 32 rows and at exit."""
import atexit, json, os

import torch

_ROWS = []
_PATH = None


def _flush():
    if _ROWS and _PATH:
        with open(f"{_PATH}.{os.getpid()}.jsonl", "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in _ROWS))
        _ROWS.clear()


def _wrap_sampler(mod, log):
    """Add a static q_soft buffer to _SelectorDraftSampler and fill it in __call__ (so inside the captured draft
    graph): softmax(scores) at slot 0 from the anchor, and at slot e from the candidate the path took at e-1.
    Reads the lattice the stock call builds (via a wrapped _selector_lattice); the drafted tokens are unchanged."""
    S = getattr(mod, "_SelectorDraftSampler", None)
    if S is None or not hasattr(mod, "_selector_lattice"):
        log("dflash log: no _SelectorDraftSampler; logging the sampler's own q")
        return
    o_init, o_call, o_lat = S.__init__, S.__call__, mod._selector_lattice
    last = {}

    def _selector_lattice(draft_model, pred_hidden, anchor_token_ids):
        ids, scores = o_lat(draft_model, pred_hidden, anchor_token_ids)
        last["v"] = (ids, scores)
        return ids, scores

    def __init__(self, *a, **k):
        o_init(self, *a, **k)
        self.q_soft = torch.zeros_like(self.q_out)

    def __call__(self, hidden_states, input_ids):
        r = o_call(self, hidden_states, input_ids)
        ids, scores = last.pop("v")
        bs, gamma, K = ids.shape
        tokens = self.out[: bs * gamma].view(bs, gamma)
        idx = (ids == tokens[..., None]).float().argmax(-1)                      # [bs, gamma]
        row0 = scores[:, 0, 0].float()                                          # [bs, K]
        rows = scores[:, 1:].float().gather(2, idx[:, :-1, None, None].expand(-1, -1, 1, K))[:, :, 0]
        self.q_soft[:bs].copy_(torch.softmax(torch.cat([row0[:, None], rows], 1), -1))
        return r

    mod._selector_lattice = _selector_lattice
    S.__init__, S.__call__ = __init__, __call__
    log("dflash log: selector T=1 path softmax captured into the draft graph (q_soft)")


def install(mod, log):
    global _PATH
    _PATH = os.environ["CORAL_DFLASH_LOG"]
    os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
    atexit.register(_flush)
    _wrap_sampler(mod, log)
    cls = mod.DFlashWorkerV2
    orig = cls._accept_block

    def _accept_block(self, *, candidates, next_token_logits, sampling_info, draft_input, prefix_lens, bs):
        out = orig(self, candidates=candidates, next_token_logits=next_token_logits, sampling_info=sampling_info,
                   draft_input=draft_input, prefix_lens=prefix_lens, bs=bs)
        try:
            accept_len, target_predict = out[0], out[5]
            ds = getattr(self, "_draft_sampler", None)
            if target_predict is None or ds is None or getattr(ds, "candidate_out", None) is None:
                return out
            with torch.no_grad():
                cid = ds.candidate_out[:bs]                     # [bs, gamma, k]
                q = getattr(ds, "q_soft", ds.q_out)[:bs].float()  # [bs, gamma, k]
                gamma = cid.shape[1]
                drafts = candidates[:, 1:1 + gamma].to(cid.dtype)
                tgt = target_predict.view(bs, -1)[:, :gamma].to(cid.dtype)
                hit_d = cid == drafts[..., None]
                q_draft = torch.where(hit_d.any(-1), (q * hit_d).sum(-1), torch.full_like(q[..., 0], -1.0))
                qs = q.sort(-1, descending=True).values
                hit_t = cid == tgt[..., None]
                q_t = (q * hit_t).sum(-1, keepdim=True)
                trank = torch.where(hit_t.any(-1), (q > q_t).sum(-1), torch.full_like(q_t[..., 0], -1).long())
                pack = torch.stack([q_draft, qs[..., 0], qs[..., 1], trank.float()], -1).cpu()
                acc = accept_len.view(-1)[:bs].cpu().tolist()
                pre = prefix_lens.view(-1)[:bs].cpu().tolist() if prefix_lens is not None else [None] * bs
            for i in range(bs):
                p = pack[i]
                _ROWS.append({"accept": int(acc[i]), "prefix": None if pre[i] is None else int(pre[i]),
                              "q_draft": [round(v, 4) for v in p[:, 0].tolist()],
                              "q1": [round(v, 4) for v in p[:, 1].tolist()],
                              "q2": [round(v, 4) for v in p[:, 2].tolist()],
                              "trank": [int(v) for v in p[:, 3].tolist()],
                              "stale": bool((p[:, 0] < 0).any())})
            if len(_ROWS) >= 32:  # the server is killed, not exited: atexit may never run
                _flush()
        except Exception as exc:  # never break serving for a measurement
            if not getattr(_accept_block, "_warned", False):
                _accept_block._warned = True
                log(f"dflash log failed: {type(exc).__name__}: {exc}")
        return out

    cls._accept_block = _accept_block
    log(f"dflash selector-confidence log -> {_PATH}.<pid>.jsonl (MEASUREMENT ONLY: syncs every step)")
