"""Cheaper DFlash2 selector head (SGLang v0.5.21, Qwen3.6-35B-A3B-FP8, TP=1).

DFlash2DraftModel.compute_candidates runs every drafted slot's hidden state through the TARGET's full lm_head
(248,320 x 2048 bf16 = 1.02 GB per pass) only to pick the selector's top-K (16) candidate ids and their unary logits.
In the b8 64-128K nsys profile that is half of the 676 us "LM head GEMM (512x8 nvjet)" bucket (the other half is the
target's verify head, which decides the output and is untouched here), plus a 248K-wide radix top-k.

Nothing here can change output text: the selector only decides which tokens are DRAFTED, and the target verify
(full bf16 head, stock code) accepts or rejects them. The only thing that can move is acceptance (tau).

Two independent switches, read once when the hook installs:

CORAL_SELECTOR_VOCAB=<path>  score only a subset of vocab rows. The rows are gathered ONCE into a contiguous
                     [S, 2048] copy when the worker attaches the target head to the draft (before graph capture), and
                     the top-k ids over the subset are mapped back to global ids. A target token outside the subset
                     can never be drafted, so the subset's coverage of target tokens bounds the tau loss
                     (tools/build_selector_vocab.py measures it). Accepted formats (load_vocab_ids):
                       .npy / .pt / .json / .txt        a list of global ids (ranked lists: most frequent first)
                       .safetensors or a checkpoint dir  a speculators/EAGLE-3 draft vocab map: d2t [S] (target id of
                                                         draft row i = i + d2t[i]), t2d [V] bool (cross-checked) --
                                                         the 32K maps shipped with the EAGLE-3 and RedHat DSpark drafts
                                                         -- or an "ids" tensor
CORAL_SELECTOR_VOCAB_N=n     keep only the first n ids of the file (ranked files only: a prefix of a frequency rank).
CORAL_SELECTOR_VOCAB_EXTRA=a,b,...  ids always added (e.g. 248046 = <|im_end|>, which the 32K d2t maps do NOT contain:
                     without it the drafter can never propose the end of a turn, one forced rejection per response).
CORAL_SELECTOR_FP8=1|scaled_mm|w8a16  a selector-only FP8 copy of the head (or of the subset). The verify head stays
                     bf16. Built once next to the bf16 rows (+254 MB for the full vocab, +32 MB for a 64K subset).
                       scaled_mm (=1): per-row weight scale, per-token dynamic activation scale, cuBLASLt FP8
                                   (torch._scaled_mm rowwise, W8A8). Halves the bytes the GEMM reads.
                       w8a16:      128x128 block scales, bf16 activations, coral_kernels.fp8w_gemv (Triton, M <= 16 per
                                   launch; larger M is chunked). Untuned for N = 248K / 32K -- the bench decides.
                     FP8 perturbs the logits by ~1-3 %: top-K membership near the K-th boundary and the unary logits
                     the lattice adds can shift slightly. Tau risk is small but not zero; measure it.
CORAL_SELECTOR_CHECK=N       for the first N EAGER calls (never inside capture) also run the stock full head and log
                     top-1 agreement / top-K overlap / max |unary logit diff| on the shared ids. Measurement only.

Contract kept: compute_candidates(hidden [N, H]) -> (candidate_ids [N, K] int64 GLOBAL ids, unary_logits [N, K] fp32
after the draft's own output_multiplier / softcap transform), the same as upstream. The module-level candidate_topk
(and so lilicorr's with_partition lse over the FULL vocabulary) is not touched: only DFlash2's compute_candidates is
replaced, and DFlash2 discards the lse.

Where the copy is built: DFlashWorkerV2._maybe_build_draft_sampler assigns `draft_model.lm_head = target_lm_head`
right before the draft CUDA graph is captured (and the eager fallback assigns it lazily). The hook overrides
DFlash2DraftModel.__setattr__ to build on that assignment, so the gather never runs inside capture. The copy is keyed
on the weight's data_ptr: a later weight update (RL / update_weights_from_*) rebuilds it on the next EAGER call; a
stale copy inside a replayed graph would still only change drafts, never outputs.

Register in sitecustomize.INSTALLERS on "sglang.srt.models.dflash" (install).
"""
import json, os, struct, sys

import numpy as np

FP8_MAX = 448.0  # float8_e4m3fn


def _log_default(msg):
    print(f"[coral] {msg} (pid {os.getpid()})", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# vocab subset files
# ---------------------------------------------------------------------------------------------------------------
_ST_DTYPES = {"I64": np.int64, "I32": np.int32, "I16": np.int16, "U8": np.uint8, "BOOL": np.bool_, "I8": np.int8}


def _read_safetensors(path):
    """Tiny reader (header + numpy views) so this works without the safetensors package and on CPU tests."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        base = 8 + n
        out = {}
        for name, h in header.items():
            if name not in ("d2t", "t2d", "ids") or h["dtype"] not in _ST_DTYPES:
                continue  # skip the (possibly multi-GB) weights: only the vocab map is needed
            a, b = h["data_offsets"]
            f.seek(base + a)
            out[name] = np.frombuffer(f.read(b - a), dtype=_ST_DTYPES[h["dtype"]]).reshape(h["shape"])
    return out


def _ids_from_d2t(t):
    """speculators / SpecForge convention: target id of draft row i = i + d2t[i] (SGLang's own EAGLE-3 loader reads it
    the same way: hot_token_id = d2t + arange, llama_eagle3.py:339-345). t2d, when present, must mark the same ids."""
    d2t = t["d2t"].astype(np.int64)
    ids = np.arange(d2t.shape[0], dtype=np.int64) + d2t
    if ids.size > 1 and not np.all(np.diff(ids) > 0):
        raise ValueError("i + d2t[i] is not strictly increasing: not the speculators offset convention")
    if "t2d" in t:
        nz = np.flatnonzero(t["t2d"].astype(bool))
        if not np.array_equal(nz, ids):
            raise ValueError("t2d and d2t disagree (t2d's True positions != i + d2t[i])")
    return ids


def load_vocab_ids(path, vocab_size=None, n=None, extra=()):
    """-> int64 [S] global token ids, duplicates dropped, file order kept (so a ranked file's prefix is its top-n)."""
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        cands = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
        hits = [c for c in cands if {"d2t", "ids"} & set(_safetensors_keys(c))]
        if not hits:
            raise FileNotFoundError(f"{path}: no *.safetensors with a d2t / ids tensor")
        path = hits[0]
    if path.endswith(".safetensors"):
        t = _read_safetensors(path)
        if "d2t" in t:
            ids = _ids_from_d2t(t)
        elif "ids" in t:
            ids = t["ids"].astype(np.int64).reshape(-1)
        elif "t2d" in t:
            ids = np.flatnonzero(t["t2d"].astype(bool)).astype(np.int64)
        else:
            raise ValueError(f"{path}: needs a d2t, t2d or ids tensor")
    elif path.endswith(".npy"):
        ids = np.load(path).astype(np.int64).reshape(-1)
    elif path.endswith(".pt"):
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=True)  # Hitesh's tokmap/top*.pt: a list of ids
        ids = (obj.cpu().numpy() if hasattr(obj, "cpu") else np.asarray(obj)).astype(np.int64).reshape(-1)
    elif path.endswith(".json"):
        obj = json.load(open(path))
        ids = np.asarray(obj["ids"] if isinstance(obj, dict) else obj, dtype=np.int64).reshape(-1)
    else:
        ids = np.asarray(open(path).read().replace(",", " ").split(), dtype=np.int64)
    if n:
        ids = ids[: int(n)]
    ids = np.concatenate([ids, np.asarray(list(extra), dtype=np.int64)])
    _, first = np.unique(ids, return_index=True)
    ids = ids[np.sort(first)]
    if ids.size == 0:
        raise ValueError(f"{path}: empty vocab subset")
    if ids.min() < 0 or (vocab_size is not None and ids.max() >= vocab_size):
        raise ValueError(f"{path}: ids outside [0, {vocab_size}): min {ids.min()}, max {ids.max()}")
    return ids


def _safetensors_keys(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return [k for k in json.loads(f.read(n)) if k != "__metadata__"]


# ---------------------------------------------------------------------------------------------------------------
# the reduced / FP8 head (pure torch: CPU-testable; the CUDA kernels are picked by device)
# ---------------------------------------------------------------------------------------------------------------
class SelectorHead:
    """Selector-only copy of (rows of) the target head. ids: global id of each row (None = rows 0..num_org-1)."""

    def __init__(self, weight, num_org, ids=None, fp8=None):
        import torch
        self.src_ptr = weight.data_ptr()
        self.num_org = int(num_org)
        self.fp8 = fp8 or None
        dev = weight.device
        with torch.no_grad():
            if ids is not None:
                ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=dev)
                if int(ids_t.max()) >= self.num_org:
                    raise ValueError(f"subset id {int(ids_t.max())} >= org vocab {self.num_org} (padding rows)")
                w = weight.index_select(0, ids_t).contiguous()  # the one gather: S rows, ~134 MB at 32K
                self.ids = ids_t
            else:
                w = weight[: self.num_org]  # full head: a view, no copy (the FP8 path below makes its own)
                self.ids = None
            self.rows = int(w.shape[0])
            if self.fp8 is None:
                self.w = w
            elif self.fp8 == "scaled_mm":
                # Per-row scale; rows padded to a multiple of 16 (cuBLASLt FP8 needs N % 16 == 0). Padded rows are
                # zero, so their logits are 0 -- masked to -inf before the top-k (a real logit can be negative).
                pad = (-self.rows) % 16
                wf = w.float()
                s = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
                q = (wf / s).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                if pad:
                    q = torch.cat([q, q.new_zeros(pad, q.shape[1])])
                    s = torch.cat([s, s.new_ones(pad, 1)])
                self.w, self.w_scale = q.contiguous(), s.reshape(1, -1).contiguous()  # [1, Npad] for rowwise
                del wf
            elif self.fp8 == "w8a16":
                # 128x128 block scales: the layout coral_kernels.fp8w_gemv (and the checkpoint's FP8 linears) use.
                N, K = w.shape
                assert K % 128 == 0, K
                nb = -(-N // 128)
                wp = torch.zeros(nb * 128, K, dtype=torch.float32, device=dev)
                wp[:N] = w.float()
                blk = wp.view(nb, 128, K // 128, 128)
                s = blk.abs().amax(dim=(1, 3)).clamp_min(1e-12) / FP8_MAX  # [nb, K/128]
                q = (blk / s[:, None, :, None]).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                self.w, self.w_scale = q.view(nb * 128, K)[:N].contiguous(), s.float().contiguous()
                del wp, blk
                if dev.type == "cuda":  # import (not compile) the Triton module now, outside any capture
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    import coral_kernels  # noqa: F401
            else:
                raise ValueError(f"CORAL_SELECTOR_FP8={self.fp8!r}: use 1, scaled_mm or w8a16")

    def nbytes(self):
        return self.w.numel() * self.w.element_size() if (self.ids is not None or self.fp8) else 0

    def logits(self, hidden):
        """[N, H] -> [N, R] bf16 (R = rows, or rows padded to 16 for scaled_mm with padding set to -inf)."""
        import torch
        if self.fp8 is None:
            return torch.matmul(hidden.to(self.w.dtype), self.w.T)
        x = hidden.to(torch.bfloat16)
        if self.fp8 == "scaled_mm":
            sx = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
            xq = (x.float() / sx).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            if x.is_cuda:
                y = torch._scaled_mm(xq, self.w.T, scale_a=sx, scale_b=self.w_scale, out_dtype=torch.bfloat16)
            else:  # CPU reference with the same quantized operands (tests only)
                y = ((xq.float() * sx) @ (self.w.float() * self.w_scale.T).T).to(torch.bfloat16)
            if y.shape[1] > self.rows:
                y[:, self.rows:] = float("-inf")
            return y
        # w8a16
        if x.is_cuda:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from coral_kernels import fp8w_gemv
            x = x.contiguous()
            if x.shape[0] <= 16:
                return fp8w_gemv(x, self.w, self.w_scale)
            return torch.cat([fp8w_gemv(x[i:i + 16], self.w, self.w_scale) for i in range(0, x.shape[0], 16)])
        N, K = self.w.shape
        nb = self.w_scale.shape[0]
        wd = torch.zeros(nb * 128, K, dtype=torch.float32)
        wd[:N] = self.w.float()
        wd = (wd.view(nb, 128, K // 128, 128) * self.w_scale[:, None, :, None]).view(nb * 128, K)[:N]
        return (x.float() @ wd.T).to(torch.bfloat16)

    def topk(self, hidden, k, topk_fn):
        """-> (global ids [N, k] int64, vals [N, k] logits dtype). topk_fn is dflash._radix_topk (flashinfer radix
        top-k when available; it needs a contiguous 2-D input, which the logits here always are)."""
        vals, idx = topk_fn(self.logits(hidden), k)
        idx = idx.long()
        if self.ids is None:
            return idx, vals
        return self.ids[idx], vals  # local row -> global token id; one tiny gather, capture-safe


def config_from_env(env=None):
    env = os.environ if env is None else env
    vocab = env.get("CORAL_SELECTOR_VOCAB") or None
    fp8 = env.get("CORAL_SELECTOR_FP8") or None
    if fp8 in ("0", "off", "false"):
        fp8 = None
    if fp8 == "1":
        fp8 = "scaled_mm"
    extra = [int(t) for t in (env.get("CORAL_SELECTOR_VOCAB_EXTRA") or "").replace(",", " ").split()]
    return dict(vocab=vocab, vocab_n=int(env.get("CORAL_SELECTOR_VOCAB_N") or 0) or None, extra=extra, fp8=fp8,
                check=int(env.get("CORAL_SELECTOR_CHECK") or 0))


# ---------------------------------------------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------------------------------------------
def install(mod, log=None):
    """Post-import hook for sglang.srt.models.dflash: replace DFlash2DraftModel.compute_candidates."""
    import torch
    log = log or _log_default
    cfg = config_from_env()
    if not cfg["vocab"] and not cfg["fp8"]:
        log("selector head: neither CORAL_SELECTOR_VOCAB nor CORAL_SELECTOR_FP8 set; not patching")
        return
    cls = mod.DFlash2DraftModel
    orig_cc, orig_setattr = cls.compute_candidates, cls.__setattr__
    ids_cache = {}
    check_left = [cfg["check"]]

    def _ids(vocab_size):
        if cfg["vocab"] and "ids" not in ids_cache:
            ids_cache["ids"] = load_vocab_ids(cfg["vocab"], vocab_size=vocab_size, n=cfg["vocab_n"],
                                              extra=cfg["extra"])
        return ids_cache.get("ids")

    def _build(self, lm_head):
        tp = mod.get_parallel().tp_size
        # TP > 1 shards the head by vocab: a subset would need per-shard id lists and the all-gather merge
        # candidate_topk does. Not needed for the TP=1 Qwen3.6 runs, so refuse rather than half-support it.
        assert tp == 1, f"CORAL_SELECTOR_* supports TP=1 only (tp_size={tp})"
        weight = getattr(lm_head, "weight", None)
        if not mod.is_dense_head_weight(weight):
            raise RuntimeError("CORAL_SELECTOR_*: needs the dense bf16 target lm_head (Qwen3.6-FP8 keeps it bf16)")
        num_org = int(lm_head.org_vocab_size)
        head = SelectorHead(weight, num_org, ids=_ids(num_org), fp8=cfg["fp8"])
        if head.rows < self.candidate_selector.top_k:
            raise ValueError(f"selector subset has {head.rows} rows < top_k {self.candidate_selector.top_k}")
        object.__setattr__(self, "_coral_sel_head", head)  # plain attribute: not a buffer, not in state_dict
        log(f"selector head: rows={head.rows}/{num_org} fp8={cfg['fp8']} copy={head.nbytes() / 2**20:.0f} MiB "
            f"(full bf16 head {weight.numel() * weight.element_size() / 2**20:.0f} MiB)")
        return head

    def __setattr__(self, name, value):
        orig_setattr(self, name, value)
        # The worker attaches the target head here right before draft-graph capture: build the copy now, eagerly.
        if name == "lm_head" and value is not None and getattr(value, "weight", None) is not None:
            _build(self, value)

    def compute_candidates(self, hidden):
        assert self.lm_head is not None, "draft_model.lm_head unset before capture"
        head = self.__dict__.get("_coral_sel_head")
        if head is None or head.src_ptr != self.lm_head.weight.data_ptr():
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("CORAL_SELECTOR_*: selector head not built before CUDA-graph capture")
            head = _build(self, self.lm_head)
        ids, vals = head.topk(hidden, self.candidate_selector.top_k, mod._radix_topk)
        out = ids, self._transform_unary_logits(vals)
        if check_left[0] > 0 and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            check_left[0] -= 1
            ref_ids, ref_u = orig_cc(self, hidden)
            k = ref_ids.shape[1]
            top1 = (ref_ids[:, 0] == ids[:, 0]).float().mean().item()
            overlap = (ref_ids[:, :, None] == ids[:, None, :]).any(-1).float().sum(-1).mean().item() / k
            # unary logit drift on ids both heads chose (FP8 numerics; exactly 0 for a bf16 subset up to GEMM order)
            match = ref_ids[:, :, None] == ids[:, None, :]
            diff = (ref_u[:, :, None] - out[1][:, None, :]).abs().masked_fill(~match, 0).amax().item()
            log(f"selector check N={hidden.shape[0]} top1_agree={top1:.3f} topk_overlap={overlap:.3f} "
                f"max_unary_diff={diff:.4f}")
        return out

    cls.__setattr__ = __setattr__
    cls.compute_candidates = compute_candidates
    log(f"selector head patched {cls.__module__}.{cls.__name__} vocab={cfg['vocab']} n={cfg['vocab_n']} "
        f"extra={cfg['extra']} fp8={cfg['fp8']}")
