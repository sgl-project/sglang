"""Scalar stats of DSV4.1 intermediates. Off unless DEBUG env is set.

Does not dump full tensors. Catches NaN/Inf/RMS collapse/explosion/dead routes.
Not a known-good oracle.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_events: list[dict[str, Any]] = []
_step = 0
_dumped_weights = False


def enabled() -> bool:
    return bool(envs.SGLANG_DEBUG_DSV41_PROBE_STATS.get())


def _rank() -> int:
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().tp_group.rank_in_group)
    except Exception:
        return int(os.environ.get("RANK", "0"))


def tensor_stats(t: Optional[torch.Tensor], name: str) -> dict[str, Any]:
    if t is None:
        return {"name": name, "none": True}
    x = t.detach()
    n = int(x.numel())
    out: dict[str, Any] = {
        "name": name,
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "n": n,
    }
    if n == 0:
        out.update(nan=0, inf=0, zero_frac=0.0, rms=0.0, mean=0.0, absmax=0.0)
        return out
    # Chunk so a vocab embedding or packed KV cannot fp32-clone itself into OOM.
    chunk = 1 << 20
    flat = x.reshape(-1)
    nan = inf = n_zero = 0
    ss = 0.0
    sm = 0.0
    n_fin = 0
    absmax = 0.0
    vmin = float("inf")
    vmax = float("-inf")
    for i in range(0, n, chunk):
        z = flat[i : i + chunk]
        nan += int(torch.isnan(z).sum().item())
        inf += int(torch.isinf(z).sum().item())
        n_zero += int((z == 0).sum().item())
        zf = z.float()
        fin = torch.isfinite(zf)
        n_f = int(fin.sum().item())
        if n_f == 0:
            continue
        w = zf[fin]
        ss += float(w.square().sum().item())
        sm += float(w.sum().item())
        n_fin += n_f
        absmax = max(absmax, float(w.abs().max().item()))
        vmin = min(vmin, float(w.min().item()))
        vmax = max(vmax, float(w.max().item()))
    out["nan"] = nan
    out["inf"] = inf
    out["zero_frac"] = n_zero / n
    if n_fin == 0:
        out.update(rms=float("nan"), mean=float("nan"), absmax=float("nan"), min=None, max=None)
        return out
    out["rms"] = (ss / n_fin) ** 0.5
    out["mean"] = sm / n_fin
    out["absmax"] = absmax
    out["min"] = vmin
    out["max"] = vmax
    return out


def record(name: str, t: Optional[torch.Tensor], extra: Optional[dict[str, Any]] = None) -> None:
    if not enabled():
        return
    ev = tensor_stats(t, name)
    if extra:
        ev.update(extra)
    _events.append(ev)


def record_meta(name: str, extra: dict[str, Any]) -> None:
    """Scalar / config probe with no tensor payload."""
    if not enabled():
        return
    ev: dict[str, Any] = {"name": name}
    ev.update(extra)
    _events.append(ev)


def record_q_split(
    name: str,
    q: torch.Tensor,
    n_local: int,
    rope_dim: int = 64,
) -> None:
    """CSA2 Q is often FlashMLA-padded to 64 heads; only [:n_local] is written."""
    if not enabled() or q is None or q.ndim < 3:
        return
    h = int(q.shape[1])
    extra = {"n_local_heads": int(n_local), "kernel_heads": h}
    record(name, q, extra)
    local = q[:, :n_local] if n_local > 0 and h >= n_local else q
    record(f"{name}_local", local, extra)
    if n_local > 0 and h > n_local:
        record(f"{name}_pad", q[:, n_local:], extra)
    if local.shape[-1] > rope_dim:
        record(f"{name}_local_nope", local[..., :-rope_dim], extra)
        record(f"{name}_local_rope", local[..., -rope_dim:], extra)


def _official_rope_tail(
    x_tail: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Official inference/model.py apply_rotary_emb for SGLang [T, ..., 64] layout."""
    xc = torch.view_as_complex(x_tail.float().unflatten(-1, (-1, 2)).contiguous())
    fc = freqs_cis[positions] if freqs_cis.shape[0] != x_tail.shape[0] else freqs_cis
    while fc.ndim < xc.ndim:
        fc = fc.unsqueeze(-2)
    return torch.view_as_real(xc * fc).flatten(-2)


def record_rope_vs_official(
    name: str,
    x_pre_tail: torch.Tensor,
    x_post_tail: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    if not enabled() or x_pre_tail is None or x_post_tail is None:
        return
    ref = _official_rope_tail(x_pre_tail, freqs_cis, positions)
    post = x_post_tail.detach().float()
    diff = (ref - post).abs()
    a = ref.reshape(-1)
    b = post.reshape(-1)
    denom = float(a.norm().item() * b.norm().item()) + 1e-12
    record_meta(
        name,
        {
            "absmax_err": float(diff.max().item()),
            "rms_err": float(diff.square().mean().sqrt().item()),
            "cosine": float(a.dot(b).item() / denom),
            "ref_rms": float(ref.square().mean().sqrt().item()),
            "post_rms": float(post.square().mean().sqrt().item()),
        },
    )


def record_qb_path(
    layer_id: int,
    q: torch.Tensor,
    q_out: torch.Tensor,
    *,
    q_head_norm: bool,
    rope_dim: int,
    freqs_cis: Optional[torch.Tensor] = None,
    rope_base: Optional[float] = None,
    positions: Optional[torch.Tensor] = None,
) -> None:
    """8-head Q after wq_b, before/after RoPE. Do not pass the 64-head pad buffer."""
    if not enabled():
        return
    extra = {"q_head_norm": bool(q_head_norm)}
    record(f"L{layer_id}.qb.pre_rope", q, extra)
    if q.ndim >= 3 and q.shape[-1] > rope_dim:
        record(f"L{layer_id}.qb.pre_rope_nope", q[..., :-rope_dim], extra)
        record(f"L{layer_id}.qb.pre_rope_rope", q[..., -rope_dim:], extra)
    record(f"L{layer_id}.qb.post_rope", q_out, extra)
    if q_out.ndim >= 3 and q_out.shape[-1] > rope_dim:
        record(f"L{layer_id}.qb.post_rope_nope", q_out[..., :-rope_dim], extra)
        record(f"L{layer_id}.qb.post_rope_rope", q_out[..., -rope_dim:], extra)
    if layer_id in (0, 2) and freqs_cis is not None and freqs_cis.numel():
        # Full table is max_position_embeddings long; sample only.
        sample = freqs_cis[:8].detach()
        mag = sample.abs()
        record_meta(
            f"L{layer_id}.freqs",
            {
                "abs_max": float(mag.max().item()),
                "abs_min": float(mag.min().item()),
                "dtype": str(freqs_cis.dtype).replace("torch.", ""),
                "shape": list(freqs_cis.shape),
                "rope_base": rope_base,
                "q_head_norm": bool(q_head_norm),
            },
        )
        if positions is not None and q.ndim >= 3 and q_out.ndim >= 3:
            record_rope_vs_official(
                f"L{layer_id}.qb.rope_vs_official",
                q[..., -rope_dim:],
                q_out[..., -rope_dim:],
                freqs_cis,
                positions,
            )


def record_ids(name: str, ids: Optional[torch.Tensor], extra: Optional[dict[str, Any]] = None) -> None:
    if not enabled() or ids is None:
        return
    x = ids.detach()
    n = int(x.numel())
    ev: dict[str, Any] = {
        "name": name,
        "shape": list(x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "n": n,
    }
    if n == 0:
        _events.append(ev)
        return
    xi = x.long()
    ev["min"] = int(xi.min().item())
    ev["max"] = int(xi.max().item())
    ev["n_neg"] = int((xi < 0).sum().item())
    uniq = torch.unique(xi)
    ev["n_unique"] = int(uniq.numel())
    if extra:
        ev.update(extra)
    _events.append(ev)


def record_moe(layer_id: int, hidden: torch.Tensor, topk_output: Any) -> None:
    if not enabled():
        return
    record(f"L{layer_id}.moe_in", hidden)
    ids = getattr(topk_output, "topk_ids", None)
    w = getattr(topk_output, "topk_weights", None)
    record_ids(f"L{layer_id}.moe_topk_ids", ids)
    if w is not None:
        record(f"L{layer_id}.moe_topk_w", w)
        wf = w.detach().float()
        if wf.numel():
            _events[-1]["row_sum_mean"] = float(wf.sum(dim=-1).mean().item())


def record_logits(logits: Optional[torch.Tensor]) -> None:
    if not enabled() or logits is None:
        return
    record("logits", logits)
    x = logits.detach()
    if x.numel() == 0:
        return
    last = x[-1] if x.ndim >= 1 else x
    if last.ndim > 1:
        last = last.reshape(-1)
    last_f = last.float()
    if not torch.isfinite(last_f).any():
        return
    p = torch.softmax(last_f, dim=-1)
    entropy = float((-(p * (p + 1e-12).log()).sum()).item())
    topv, topi = torch.topk(last_f, k=min(8, last_f.numel()))
    _events[-1]["entropy"] = entropy
    _events[-1]["argmax"] = int(last_f.argmax().item())
    _events[-1]["top8_ids"] = [int(i) for i in topi.tolist()]
    _events[-1]["top8_vals"] = [float(v) for v in topv.tolist()]


def maybe_record_weights(model: Any) -> None:
    global _dumped_weights
    if not enabled() or _dumped_weights:
        return
    _dumped_weights = True
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    w = getattr(embed, "weight", None)
    record("weight.embed", w)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return
    attn = getattr(layers[0], "self_attn", None)
    if attn is None:
        return
    for attr in ("wqkv_a", "wkv", "wq_b", "wo_a", "wo_b"):
        mod = getattr(attn, attr, None)
        wt = getattr(mod, "weight", None) if mod is not None else None
        record(f"weight.L0.{attr}", wt)
        scale = getattr(mod, "weight_scale_inv", None) if mod is not None else None
        if scale is None:
            scale = getattr(wt, "scale", None) if wt is not None else None
        record(f"weight.L0.{attr}.scale", scale)
    qn = getattr(attn, "q_norm", None)
    record("weight.L0.q_norm", getattr(qn, "weight", None) if qn is not None else None)
    cfg = getattr(model, "config", None)
    if cfg is None:
        cfg = getattr(getattr(model, "model", None), "config", None)
    if cfg is not None:
        record_meta(
            "cfg",
            {
                "config_type": type(cfg).__name__,
                "model_type": getattr(cfg, "model_type", None),
                "q_head_norm": getattr(cfg, "q_head_norm", "MISSING"),
                "num_attention_heads": getattr(cfg, "num_attention_heads", None),
                "qk_rope_head_dim": getattr(cfg, "qk_rope_head_dim", None),
                "rope_theta": getattr(cfg, "rope_theta", None),
                "compress_rope_theta": getattr(cfg, "compress_rope_theta", None),
            },
        )
    knh = None
    try:
        knh = int(attn._kernel_num_heads(6))
    except Exception:
        knh = None
    record_meta(
        "cfg.L0.attn",
        {
            "q_head_norm": bool(getattr(attn, "q_head_norm", None)),
            "n_local_heads": int(getattr(attn, "n_local_heads", -1)),
            "n_heads": int(getattr(attn, "n_heads", -1)),
            "attn_tp_size": int(getattr(attn, "attn_tp_size", -1)),
            "kernel_num_heads_tok6": knh,
            "rope_base": float(getattr(attn, "rope_base", float("nan"))),
            "compress_ratio": int(getattr(attn, "compress_ratio", -1)),
            "qk_rope_head_dim": int(getattr(attn, "qk_rope_head_dim", -1)),
        },
    )


def flush(tag: str = "") -> None:
    if not enabled() or not _events:
        return
    global _step
    out_dir = envs.SGLANG_DEBUG_DSV41_PROBE_STATS_DIR.get() or "/tmp/dsv41-probe"
    os.makedirs(out_dir, exist_ok=True)
    rank = _rank()
    path = os.path.join(out_dir, f"rank{rank}_step{_step}.json")
    payload = {"rank": rank, "step": _step, "tag": tag, "events": list(_events)}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.write("\n")
    os.replace(tmp, path)
    nans = [e["name"] for e in _events if e.get("nan") or e.get("inf")]
    rms = {e["name"]: e.get("rms") for e in _events if "rms" in e}
    logger.warning(
        "dsv41 probe stats wrote %s events=%d nan/inf=%s embed_rms=%s logits=%s",
        path,
        len(_events),
        nans[:12],
        rms.get("embed"),
        next((e for e in _events if e.get("name") == "logits"), None),
    )
    _events.clear()
    _step += 1
