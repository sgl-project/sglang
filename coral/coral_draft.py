"""Draft-side cost cuts for DFlash2 on SGLang v0.5.21 (incoai Qwen3.6-35B-A3B DFlash2, block 8). Env-switched,
installed by sitecustomize on two modules: sglang.srt.models.dflash and sglang.srt.speculative.dflash_worker_v2.

None of these touch the target verify, so output text can't change; the FP8 ones can only move acceptance (tau).

CORAL_DRAFT_FP8_HEAD=1   The selector's candidate projection (draft hidden [7*bs, 2048] @ target lm_head[:248320]^T,
                         dflash.py:89-102) runs on an FP8 copy of the head: per-row e4m3 weights, per-token dynamic
                         activations, SGLang's apply_fp8_linear (CUTLASS). The target's own logits keep the BF16 head.
                         Stock: one BF16 nvjet 512x8 GEMM per step in the draft graph, 356 us of the 6.5-7.5 ms step
                         on H100 (w1 profile). Costs +508 MB of GPU memory (the copy is built in the worker's __init__,
                         before the KV pool is sized).
CORAL_DRAFT_FP8_FC=1     Same for the draft's `fc` (8 target-layer features -> hidden, [2048, 16384], nn.Linear, so
                         --speculative-draft-model-quantization never covers it). Eager, once per step on the verified
                         block (dflash_worker_v2.py:1865). Stock 32 us.
CORAL_DRAFT_FP8_KV_FUSED=1  Needed with --speculative-draft-model-quantization fp8: an FP8 qkv_proj fails
                         can_dflash_use_fused_qkv_proj (dflash_utils.py:845-866), so context-KV materialization drops
                         from one stacked GEMM + one Triton kernel to 6 per-layer full-QKV GEMMs + norm + rope launches
                         (dflash_worker_v2.py:1901-1920). This rebuilds the fused helper from the dequantized FP8 K/V
                         rows (the same weights the FP8 forward uses).
CORAL_DRAFT_GEMV=1       The draft's tiny BF16 nn.Linear projections at decode M: DFlashGroupedConv.kernel_projection
                         [512, 2048] (x12 per step) and CandidateSelector.hidden_projection [256, 2048] (x1) ->
                         coral_kernels.rows_gemv. cuBLAS gives kernel_projection 8 CTAs (grid 4x2): 5.4-5.8 us each.
                         CORAL_DRAFT_GEMV_MAX_M (default 16) caps the token count.
CORAL_DRAFT_TOPK2=<C>    Selector top-16 over the 248320-wide logits as two torch.topk passes (C chunks, then the C*16
                         survivors) instead of flashinfer's radix top-k, which runs 21 CTAs for 7 rows (47.8 us).
                         Exact as a set; tie order may differ. C must divide the vocab (248320 = 2^9*5*97: 128/256/512).
CORAL_DFLASH_MAMBA_FUSE=1  The post-verify mamba bookkeeping (dflash_worker_v2.py:2039-2059, ~15 eager elementwise
                         launches, ~50 us of GPU time incl. launch gaps) as one Triton kernel. Integer-exact.
CORAL_DRAFT_COMPILE=1     Compiles DFlashDraftModel.forward as a whole. Inductor can fuse elementwise and small
                         projection work across the six-layer loop while preserving the attention and Triton calls.
                         CORAL_DRAFT_COMPILE_DYNAMIC (default 0) and CORAL_DRAFT_COMPILE_BACKEND (default inductor)
                         control compilation. The SGLang draft CUDA graph still captures the compiled callable.
"""
import math
import os
import sys
from types import SimpleNamespace as NS

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU / macOS: only the torch reference paths run
    triton = tl = None

FP8_MAX = 448.0
DFLASH_MOD = "sglang.srt.models.dflash"
WORKER_MOD = "sglang.srt.speculative.dflash_worker_v2"


def _log(msg):
    print(f"[coral] {msg} (pid {os.getpid()})", file=sys.stderr, flush=True)


def _on(name):
    return os.environ.get(name, "") not in ("", "0")


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# ---------------------------------------------------------------------------------------------------------------
# FP8 linear: per-row (output channel) weights, per-token dynamic activations
# ---------------------------------------------------------------------------------------------------------------
class Fp8Pack:
    """wq [N, K] e4m3 (row-major), scale [N] fp32. `wt` / `scale_row` are the layout Fp8LinearMethod hands
    apply_fp8_linear after online quantization (fp8.py:1022-1040): weight [K, N] (a transposed view), scale [1, N]."""

    def __init__(self, wq, scale):
        self.wq, self.scale = wq, scale
        self.wt = wq.t()
        self.scale_row = scale.view(1, -1).contiguous()
        self.n, self.k = wq.shape


@torch.no_grad()
def quantize_rows_fp8(w, chunk=32768):
    """w [N, K] -> Fp8Pack. Chunked so the fp32 temporaries stay at chunk*K*4 bytes (268 MB for the head)."""
    n = w.shape[0]
    wq = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(n, dtype=torch.float32, device=w.device)
    for s in range(0, n, chunk):
        blk = w[s:s + chunk].float()
        sc = blk.abs().amax(dim=1).clamp_min(1e-12) / FP8_MAX
        wq[s:s + chunk] = (blk / sc[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        scale[s:s + chunk] = sc
    return Fp8Pack(wq, scale)


def fp8_linear_ref(x, pack):
    """Torch reference of what the GPU path computes: per-token e4m3 activations x per-row e4m3 weights."""
    x2 = x.reshape(-1, x.shape[-1]).float()
    xs = x2.abs().amax(dim=1).clamp_min(1e-12) / FP8_MAX
    xq = (x2 / xs[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    y = (xq @ pack.wq.float().t()) * xs[:, None] * pack.scale[None, :]
    return y.to(x.dtype).view(*x.shape[:-1], pack.n)


def _fp8_linear_sglang(x, pack):
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear, cutlass_fp8_supported
    return apply_fp8_linear(input=x, weight=pack.wt, weight_scale=pack.scale_row, input_scale=None,
                            cutlass_fp8_supported=cutlass_fp8_supported())


def fp8_linear(x, pack):
    return _fp8_linear_sglang(x, pack) if x.is_cuda else fp8_linear_ref(x, pack)


# ---------------------------------------------------------------------------------------------------------------
# CORAL_DRAFT_FP8_HEAD: candidate projection through an FP8 copy of the target head
# ---------------------------------------------------------------------------------------------------------------
def head_num_org(lm_head):
    """Mirrors candidate_topk (dflash.py:128-133)."""
    try:
        from sglang.srt.runtime_context import get_parallel
        tp1 = get_parallel().tp_size == 1
    except Exception:
        tp1 = True
    return int(lm_head.org_vocab_size) if tp1 else int(lm_head.shard_indices.num_org_elements)


def head_pack(lm_head, num_org, log=_log):
    """The cached FP8 copy of lm_head.weight[:num_org]; built here unless a CUDA graph is being captured (then None:
    the caller keeps the BF16 path rather than capture a 1 GB quantization pass into the graph)."""
    cache = lm_head.__dict__.setdefault("_coral_fp8_head", {})
    pack = cache.get(num_org)
    if pack is None:
        w = getattr(lm_head, "weight", None)
        if w is None or w.dtype not in (torch.bfloat16, torch.float16) or w.dim() != 2 or _capturing():
            return None
        pack = cache[num_org] = quantize_rows_fp8(w.data[:num_org])
        log(f"draft fp8 head: {num_org}x{w.shape[1]} -> e4m3 per-row ({pack.wq.numel() / 2**20:.0f} MiB extra)")
    return pack


def _install_fp8_head(mod, log):
    orig = mod._project_candidate_logits

    def _project_candidate_logits(hidden, lm_head, *, num_org, use_quant_head):
        if not use_quant_head:
            pack = head_pack(lm_head, num_org, log)
            if pack is not None:
                return fp8_linear(hidden.to(lm_head.weight.dtype).contiguous(), pack)
        return orig(hidden, lm_head, num_org=num_org, use_quant_head=use_quant_head)

    mod._project_candidate_logits = _project_candidate_logits
    log("draft fp8 head: candidate_topk projects through the FP8 head copy")


# ---------------------------------------------------------------------------------------------------------------
# CORAL_DRAFT_FP8_FC: the 8-feature context projection
# ---------------------------------------------------------------------------------------------------------------
def fc_pack(model, log=_log):
    fc = getattr(model, "fc", None)
    if fc is None or getattr(model, "is_nemotron_35_draft", False) or not isinstance(fc, torch.nn.Linear):
        return None
    pack = fc.__dict__.get("_coral_fp8")
    if pack is None and fc.bias is None and fc.weight.dtype in (torch.bfloat16, torch.float16) and not _capturing():
        pack = fc.__dict__["_coral_fp8"] = quantize_rows_fp8(fc.weight.data)
        fc.forward = lambda x, _p=pack: fp8_linear(x.contiguous(), _p)
        log(f"draft fp8 fc: {tuple(fc.weight.shape)} -> e4m3 per-row")
    return pack


def _install_fp8_fc(mod, log):
    cls = mod.DFlashDraftModel
    orig = cls.project_target_hidden

    def project_target_hidden(self, target_hidden):
        fc_pack(self, log)  # swaps self.fc.forward once; the stock body then calls self.fc(...)
        return orig(self, target_hidden)

    cls.project_target_hidden = project_target_hidden
    log("draft fp8 fc: patched DFlashDraftModel.project_target_hidden")


# ---------------------------------------------------------------------------------------------------------------
# CORAL_DRAFT_GEMV: tiny BF16 nn.Linear projections -> rows_gemv
# ---------------------------------------------------------------------------------------------------------------
_GEMV = None  # set lazily to coral_kernels.rows_gemv; tests replace it


def _rows_gemv(x, w):
    global _GEMV
    if _GEMV is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from coral_kernels import rows_gemv
        _GEMV = rows_gemv
    return _GEMV(x, w)


def gemv_ok(x2, w, max_m):
    k = w.shape[1]
    return (x2.is_cuda and x2.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and w.is_contiguous()
            and 0 < x2.shape[0] <= max_m and w.shape[0] <= 1024 and k <= 4096 and k & (k - 1) == 0)


def gemv_forward(lin, max_m):
    def forward(x):
        w = lin.weight
        x2 = x.reshape(-1, x.shape[-1])
        if lin.bias is None and gemv_ok(x2, w, max_m):
            return _rows_gemv(x2.contiguous(), w).view(*x.shape[:-1], w.shape[0])
        return F.linear(x, w, lin.bias)
    return forward


def _install_gemv(mod, log):
    max_m = int(os.environ.get("CORAL_DRAFT_GEMV_MAX_M", "16"))
    for cls_name, attr in (("DFlashGroupedConv", "kernel_projection"), ("CandidateSelector", "hidden_projection")):
        cls = getattr(mod, cls_name, None)
        if cls is None:
            log(f"draft gemv: {cls_name} not found, skipped")
            continue
        orig_init = cls.__init__

        def __init__(self, *a, _orig=orig_init, _attr=attr, **k):
            _orig(self, *a, **k)
            lin = getattr(self, _attr)
            lin.forward = gemv_forward(lin, max_m)

        cls.__init__ = __init__
    log(f"draft gemv: kernel_projection / hidden_projection -> rows_gemv at M <= {max_m}")


# ---------------------------------------------------------------------------------------------------------------
# CORAL_DRAFT_TOPK2: two-pass top-k
# ---------------------------------------------------------------------------------------------------------------
def topk2(scores, k, chunks):
    """Sorted top-k of scores [n, V] via a per-chunk top-k (n*chunks rows -> many CTAs) and a top-k of the survivors.
    Returns (vals, ids) like _radix_topk. The global top-k is contained in the union of per-chunk top-ks."""
    n, v = scores.shape
    width = v // chunks
    vals1, idx1 = torch.topk(scores.view(n, chunks, width), k, dim=-1)
    base = torch.arange(0, v, width, device=scores.device, dtype=idx1.dtype).view(1, chunks, 1)
    ids1 = (idx1 + base).view(n, chunks * k)
    vals, j = torch.topk(vals1.reshape(n, chunks * k), k, dim=-1)
    return vals, ids1.gather(1, j)


def _install_topk2(mod, log):
    chunks = int(os.environ["CORAL_DRAFT_TOPK2"])
    orig = mod._radix_topk

    def _radix_topk(scores, k):
        v = scores.shape[-1]
        if scores.dim() == 2 and v % chunks == 0 and v // chunks >= k and scores.is_contiguous():
            return topk2(scores, k, chunks)
        return orig(scores, k)

    mod._radix_topk = _radix_topk
    log(f"draft topk2: selector top-k in two passes over {chunks} chunks")


# ---------------------------------------------------------------------------------------------------------------
# CORAL_DRAFT_FP8_KV_FUSED: keep the fused context-KV helper with an FP8 draft
# ---------------------------------------------------------------------------------------------------------------
def dequant_kv_rows(qkv_proj, q_size, kv_size, hidden):
    """BF16 K|V rows [2*kv_size, hidden] of an online-FP8 QKVParallelLinear (weight stored transposed [K, N] with a
    per-channel [1, N] or per-tensor scale, fp8.py:1020-1040), or None if the layout isn't recognised."""
    w, s = getattr(qkv_proj, "weight", None), getattr(qkv_proj, "weight_scale", None)
    n = q_size + 2 * kv_size
    if w is None or s is None or w.dtype != torch.float8_e4m3fn:
        return None
    if tuple(w.shape) == (hidden, n):
        w = w.t()
    elif tuple(w.shape) != (n, hidden):
        return None
    rows = w[q_size:n].float()
    s = s.float().reshape(-1)
    if s.numel() == 1:
        rows = rows * s
    elif s.numel() == n:
        rows = rows * s[q_size:n, None]
    else:
        return None
    return rows.to(torch.bfloat16).contiguous()


def _fp8_kv_proxies(draft_model):
    """Stand-ins exposing exactly what FusedKVMaterializeHelper reads (fused_kv_materialize.py:296-326), with q_size 0
    so its qkv_w[q_size : q_size + 2*kv_size] slice is the dequantized K|V block. None if any layer is ineligible
    (the same gates as DFlashWorkerV2._init_fused_kv_helper, dflash_worker_v2.py:944-975, minus the quant one)."""
    layers = getattr(draft_model, "layers", None)
    if not layers or not getattr(draft_model, "supports_fused_context_kv", True):
        return None
    hidden = int(draft_model.config.hidden_size)
    out = []
    for layer in layers:
        attn = layer.self_attn
        qkv = attn.qkv_proj
        if type(getattr(qkv, "quant_method", None)).__name__ != "Fp8LinearMethod" or getattr(qkv, "bias", None) is not None:
            return None
        for sc in (getattr(attn.attn, "k_scale", None), getattr(attn.attn, "v_scale", None)):
            if sc is not None and not math.isclose(float(sc), 1.0):
                return None
        if not bool(getattr(attn.rotary_emb, "is_neox_style", True)):
            return None
        kv = dequant_kv_rows(qkv, attn.q_size, attn.kv_size, hidden)
        if kv is None:
            return None
        out.append(NS(self_attn=NS(num_kv_heads=attn.num_kv_heads, head_dim=attn.head_dim, rotary_emb=attn.rotary_emb,
                                   q_size=0, kv_size=attn.kv_size, qkv_proj=NS(weight=kv), k_norm=attn.k_norm)))
    return out


def _install_fp8_kv_fused(mod, log):
    cls = mod.DFlashWorkerV2
    orig = cls._init_fused_kv_helper

    def _init_fused_kv_helper(self):
        orig(self)
        if self._fused_kv_helper is not None:
            return
        try:
            proxies = _fp8_kv_proxies(self.draft_model)
            if proxies is None:
                return
            first = self.draft_model.layers[0].self_attn
            self._fused_kv_helper = mod._get_fused_kv_materialize_helper()(
                layers=proxies, rotary_emb=first.rotary_emb, num_kv_heads=first.num_kv_heads, head_dim=first.head_dim,
                device=self.device,
                max_position_hint=self.target_worker.model_runner.model_config.context_len + int(self.block_size))
            self._use_fused_kv_materialize = True
            log(f"draft fp8 kv fused: fused KV materialization re-enabled on dequantized FP8 K/V ({len(proxies)} layers)")
        except Exception as e:  # leave the stock per-layer fallback in place
            self._fused_kv_helper, self._use_fused_kv_materialize = None, False
            log(f"draft fp8 kv fused: failed ({e}); per-layer path stays")

    cls._init_fused_kv_helper = _init_fused_kv_helper


# ---------------------------------------------------------------------------------------------------------------
# Worker __init__: build the FP8 copies before the KV pool is sized and before any graph capture
# ---------------------------------------------------------------------------------------------------------------
def _install_prebuild(mod, log):
    cls = mod.DFlashWorkerV2
    orig = cls.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        try:
            if _on("CORAL_DRAFT_FP8_HEAD") and getattr(self, "selector", None) is not None:
                lm_head = getattr(self._target_worker.model_runner.model, "lm_head", None)
                if lm_head is not None and getattr(lm_head, "weight", None) is not None:
                    head_pack(lm_head, head_num_org(lm_head), log)
            if _on("CORAL_DRAFT_FP8_FC"):
                fc_pack(self.draft_model, log)
        except Exception as e:
            log(f"draft fp8 prebuild failed ({e}); copies will be built lazily outside capture")

    cls.__init__ = __init__


# ---------------------------------------------------------------------------------------------------------------
# CORAL_DFLASH_MAMBA_FUSE: post-verify mamba bookkeeping in one launch
# ---------------------------------------------------------------------------------------------------------------
def mamba_track_ref(commit_lens, pre, post, interval):
    """dflash_worker_v2.py:2039-2059 verbatim (interval None = no radix mamba tracking)."""
    last = commit_lens.to(torch.int64) - 1
    if interval is None:
        return last, None
    to_track_mask = pre // interval != post // interval
    tracking_point = post // interval * interval
    to_track_ith = torch.clamp(tracking_point - pre - 1, min=0)
    can_track_mask = to_track_mask & (to_track_ith < commit_lens.to(to_track_ith.dtype))
    steps = torch.where(can_track_mask, to_track_ith.to(torch.int64),
                        torch.full_like(to_track_ith, -1, dtype=torch.int64))
    return last, steps


if triton is not None:
    @triton.jit
    def _mamba_track_kernel(commit_ptr, pre_ptr, post_ptr, last_ptr, steps_ptr, n, interval,
                            HAS_TRACK: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        c = tl.load(commit_ptr + offs, mask=m, other=1).to(tl.int64)
        tl.store(last_ptr + offs, c - 1, mask=m)
        if HAS_TRACK:
            pre = tl.load(pre_ptr + offs, mask=m, other=0).to(tl.int64)
            post = tl.load(post_ptr + offs, mask=m, other=0).to(tl.int64)
            # Lengths are >= 0, so Triton's truncating // equals torch's floor //.
            q_post = post // interval
            ith = tl.maximum(q_post * interval - pre - 1, 0)
            can = ((pre // interval) != q_post) & (ith < c)
            tl.store(steps_ptr + offs, tl.where(can, ith, -1), mask=m)


def mamba_track_triton(commit_lens, pre, post, interval):
    n = commit_lens.shape[0]
    last = torch.empty(n, dtype=torch.int64, device=commit_lens.device)
    steps = torch.empty(n, dtype=torch.int64, device=commit_lens.device) if interval is not None else None
    block = max(16, min(1024, triton.next_power_of_2(max(n, 1))))
    has = interval is not None
    _mamba_track_kernel[(triton.cdiv(max(n, 1), block),)](
        commit_lens, pre if has else commit_lens, post if has else commit_lens, last, steps if has else last, n,
        int(interval) if has else 1, HAS_TRACK=has, BLOCK=block)
    return last, steps


_MAMBA_TRACK = None  # tests replace it; None = Triton on CUDA tensors, the stock path otherwise


def _install_mamba_fuse(mod, log):
    cls = mod.DFlashWorkerV2
    orig = cls._update_target_mamba_state_after_verify

    def _update_target_mamba_state_after_verify(self, *, batch, seq_lens_pre_verify, seq_lens_post_verify,
                                                commit_lens):
        if not self._need_mamba_verify_commit:
            return
        attn_backend = self.target_worker.model_runner.attn_backend
        fn = _MAMBA_TRACK or (mamba_track_triton if triton is not None and commit_lens.is_cuda else None)
        if fn is None or not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return orig(self, batch=batch, seq_lens_pre_verify=seq_lens_pre_verify,
                        seq_lens_post_verify=seq_lens_post_verify, commit_lens=commit_lens)
        interval = (mod.mamba_track_grid(batch.tree_cache.page_size)
                    if batch.mamba_track_indices is not None else None)
        last, steps = fn(commit_lens, seq_lens_pre_verify, seq_lens_post_verify, interval)
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=steps,
            model=self.target_worker.model_runner.model,
            req_pool_indices=batch.req_pool_indices[: commit_lens.shape[0]],
        )

    cls._update_target_mamba_state_after_verify = _update_target_mamba_state_after_verify
    log("dflash mamba fuse: post-verify mamba bookkeeping in one Triton launch")


# ---------------------------------------------------------------------------------------------------------------
# CORAL_DRAFT_COMPILE: whole-drafter operation fusion
# ---------------------------------------------------------------------------------------------------------------
def _compile(fn, dynamic, backend):
    return torch.compile(fn, dynamic=dynamic, backend=backend)


def _install_compile(mod, log):
    cls = mod.DFlashDraftModel
    orig = cls.forward
    dynamic = os.environ.get("CORAL_DRAFT_COMPILE_DYNAMIC", "0") not in ("", "0")
    backend = os.environ.get("CORAL_DRAFT_COMPILE_BACKEND", "inductor")
    compiled = _compile(orig, dynamic, backend)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None,
                get_embedding=False, pp_proxy_tensors=None):
        return compiled(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    cls.forward = forward
    log(f"draft compile: whole DFlashDraftModel.forward ({backend}, dynamic={dynamic})")


# ---------------------------------------------------------------------------------------------------------------
def install(mod, log=_log):
    """Entry point for sitecustomize's INSTALLERS; dispatches on the module being imported."""
    name = mod.__name__
    if name == DFLASH_MOD:
        if _on("CORAL_DRAFT_FP8_HEAD"):
            _install_fp8_head(mod, log)
        if _on("CORAL_DRAFT_FP8_FC"):
            _install_fp8_fc(mod, log)
        if _on("CORAL_DRAFT_GEMV"):
            _install_gemv(mod, log)
        if _on("CORAL_DRAFT_TOPK2"):
            _install_topk2(mod, log)
        if _on("CORAL_DRAFT_COMPILE"):
            _install_compile(mod, log)
    elif name == WORKER_MOD:
        if _on("CORAL_DRAFT_FP8_HEAD") or _on("CORAL_DRAFT_FP8_FC"):
            _install_prebuild(mod, log)
        if _on("CORAL_DRAFT_FP8_KV_FUSED"):
            _install_fp8_kv_fused(mod, log)
        if _on("CORAL_DFLASH_MAMBA_FUSE"):
            _install_mamba_fuse(mod, log)
