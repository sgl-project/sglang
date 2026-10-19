"""One-shot probe bundle for the A5/CANN9.2 all-zero-output incident.

Delete once the incident is closed; nothing here is load-bearing.

SGLANG_ZERO_PROBE=1 enables, in a single server run:
  0. fingerprint (first forward, once): which sglang tree is actually
     running, torch/torch_npu/CANN versions, arch35 detection, custom-op
     availability, extra-ops .so mtime, SGLANG_*/ASCEND_*/DEEPEP_* env the
     process saw, and two synthetic kernel checks (argmax + bf16 GEMM,
     NPU vs CPU);
  1. model entry: raw input_ids/positions/seq_lens/out_cache_loc of prefill
     batches (catches a tokenizer/IPC layer feeding zeros into the model);
  2. per-layer residual-stream stats on the first prefill batches
     (localizes the first layer whose output goes NaN/zero);
  3. sampler entry, every step: next_token_logits stats + NPU-vs-CPU argmax
     cross-check. Sampling runs outside NPU graphs, so decode steps are
     covered too even while decode replays graphs.

Layer probes fire on prefill (eager) only: decode replay never executes
python. Installation swaps each layer's bound ``forward`` for a wrapper
function but never replaces the module object -- code that walks
``model.layers[i].self_attn`` (prime_rope_cos_sin) must keep seeing the real
modules; the module-replacement variant of this probe crashed exactly there.
Graph capture (decode buckets) never hits a device sync inside the probe
because wrappers only print inside an EXTEND window. Every probe is
try/except-guarded.
"""

from __future__ import annotations

import importlib.metadata as _md
import os
import time

import torch
import torch.nn as nn

_ON = os.environ.get("SGLANG_ZERO_PROBE") == "1"
_MAX_SAMPLER = int(os.environ.get("SGLANG_ZERO_PROBE_MAX_STEPS", "40"))
_MAX_EXTEND = 2  # extend batches whose per-layer stats get printed
_DUMP_DIR = os.environ.get("SGLANG_ZERO_PROBE_DUMP_DIR", "")
_attn_dumped = [False]

_sampler_ct = 0
_extend_ct = 0
_layer_print_budget = 600
_installed = False
_fingerprint_done = False
# True only between the entry of an EXTEND forward and the entry of the next
# forward; layer wrappers probe solely inside this window.
_extend_window = [False]


def _rank() -> int:
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


def _p(msg: str) -> None:
    print(f"[ZPROBE r{_rank()}] {msg}", flush=True)


def _stats(t: torch.Tensor) -> str:
    f = t.detach().float()
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={f.min().item():.5g} max={f.max().item():.5g} "
        f"mean={f.mean().item():.6g} nan={int(torch.isnan(f).sum())} "
        f"inf={int(torch.isinf(f).sum())}"
    )


def _fingerprint() -> None:
    global _fingerprint_done
    if _fingerprint_done:
        return
    _fingerprint_done = True
    _p(f"fingerprint ts={time.strftime('%F %T')}")
    try:
        import sglang

        _p(f"sglang file: {sglang.__file__}")
    except Exception as e:  # noqa: BLE001
        _p(f"sglang import failed: {e!r}")
    try:
        import torch_npu

        _p(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
    except Exception as e:  # noqa: BLE001
        _p(f"torch_npu import failed: {e!r}")
    try:
        with open("/usr/local/Ascend/ascend-toolkit/latest/version.cfg") as f:
            _p("CANN version.cfg: " + " ".join(f.read().split()))
    except Exception as e:  # noqa: BLE001
        _p(f"version.cfg read failed: {e!r}")
    try:
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        _p(f"is_npu_arch35={is_npu_arch35()}")
    except Exception as e:  # noqa: BLE001
        _p(f"is_npu_arch35 probe failed: {e!r}")
    for pkg in (
        "torch",
        "torch-npu",
        "triton",
        "triton-ascend",
        "sgl-kernel",
        "sgl-kernel-npu",
        "deep_ep",
        "vllm-ascend",
        "numpy",
        "pydantic",
        "transformers",
    ):
        try:
            _p(f"pkg {pkg}=={_md.version(pkg)}")
        except Exception:  # noqa: BLE001 - absence is the information
            pass
    try:
        so = os.environ.get("SGLANG_DSPARK_A5_EXTRA_OPS_SO")
        if so and os.path.isfile(so):
            _p(
                f"extra-ops so: {so} mtime="
                f"{time.strftime('%F %T', time.localtime(os.stat(so).st_mtime))}"
            )
        else:
            _p(f"extra-ops so: not set/found ({so})")
    except Exception as e:  # noqa: BLE001
        _p(f"extra-ops stat failed: {e!r}")
    env_dump = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("SGLANG_", "ASCEND_", "DEEPEP_", "PYTORCH_NPU", "TASK_QUEUE"))
    }
    _p("env: " + " ".join(f"{k}={v}" for k, v in sorted(env_dump.items())))
    _ops_report()
    _kernel_selfcheck()


def _ops_report() -> None:
    def has(ns: str, op: str) -> str:
        try:
            return f"{ns}.{op}={'yes' if hasattr(getattr(torch.ops, ns), op) else 'NO'}"
        except Exception as e:  # noqa: BLE001
            return f"{ns}.{op}=err({e!r})"

    _p(
        "ops: "
        + " ".join(
            (
                has("custom", "npu_kv_quant_sparse_attn_sharedkv"),
                has("custom", "npu_sparse_attn_sharedkv"),
                has("_C_ascend", "npu_kv_quant_sparse_attn_sharedkv_v2"),
                has("npu", "sparse_attn_sharedkv"),
            )
        )
    )


def _kernel_selfcheck() -> None:
    try:
        capturing = False
        try:
            capturing = torch.npu.is_current_stream_capturing()
        except Exception:  # noqa: BLE001 - API missing on old trees
            pass
        if capturing:
            _p("kernel selfcheck skipped (capturing)")
            return
        dev = "npu" if torch.npu.is_available() else "cpu"
        vocab = 129280  # DSV4-ish vocab; shape sensitivity is the point
        g = torch.Generator(device="cpu").manual_seed(2026)
        x = (
            torch.randn(8, vocab, generator=g, dtype=torch.float32)
            .to(dev)
            .to(torch.bfloat16)
        )
        with torch.no_grad():
            ids_npu = x.argmax(-1)
            ids_cpu = x.float().cpu().argmax(-1)
        ok = bool((ids_npu.cpu() == ids_cpu).all())
        _p(f"selfcheck argmax bf16[{x.shape}]: match={ok}")
        a = torch.randn(512, 512, generator=g).to(dev).to(torch.bfloat16)
        b = torch.randn(512, 512, generator=g).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            c_npu = (a @ b).float().cpu()
        c_cpu = a.float().cpu() @ b.float().cpu()
        rel = (c_npu - c_cpu).abs().max().item() / c_cpu.abs().max().item()
        _p(f"selfcheck bf16 matmul[512,512]: rel_err={rel:.3g}")
    except Exception as e:  # noqa: BLE001
        _p(f"kernel selfcheck error: {e!r}")


def probe_sampler(logits_output) -> None:
    """First line of Sampler.forward; takes its first parameter."""
    global _sampler_ct
    if not _ON or _sampler_ct >= _MAX_SAMPLER:
        return
    _sampler_ct += 1
    try:
        lg = getattr(logits_output, "next_token_logits", None)
        if lg is None and isinstance(logits_output, torch.Tensor):
            lg = logits_output
        if lg is None or lg.numel() == 0:
            _p(f"sampler {_sampler_ct}: no logits")
            return
        with torch.no_grad():
            ids_npu = lg.detach().argmax(-1)
            ids_cpu = lg.detach().float().cpu().argmax(-1)
            match = bool((ids_npu.detach().cpu() == ids_cpu).all())
        _p(f"sampler {_sampler_ct} logits: {_stats(lg)}")
        _p(
            f"sampler {_sampler_ct} argmax: "
            f"npu={ids_npu[:8].detach().cpu().tolist()} "
            f"cpu={ids_cpu[:8].tolist()} match={match}"
        )
    except Exception as e:  # noqa: BLE001 - a probe must never kill serving
        _p(f"sampler probe error: {e!r}")


def _probe_named(tag: str, args, kwargs, out) -> None:
    if not (_ON and _extend_window[0] and _extend_ct <= _MAX_EXTEND):
        return
    global _layer_print_budget
    if _layer_print_budget <= 0:
        return
    _layer_print_budget -= 1
    try:
        # DSV4 passes the attention input as kw "x" (deepseek_v4.py
        # self.self_attn(x=hidden_states, ...)); other modules use
        # "hidden_states" or a positional tensor.
        inp = kwargs.get("x")
        if not isinstance(inp, torch.Tensor):
            inp = kwargs.get("hidden_states")
        if not isinstance(inp, torch.Tensor):
            inp = next((a for a in args if isinstance(a, torch.Tensor)), None)
        outs = out if isinstance(out, tuple) else (out,)
        o0 = next((t for t in outs if isinstance(t, torch.Tensor)), None)
        _p(
            f"extend {_extend_ct} {tag}: "
            f"in[{_stats(inp) if inp is not None else '?'}] "
            f"out[{_stats(o0) if o0 is not None else '?'}]"
        )
    except Exception as e:  # noqa: BLE001
        _p(f"{tag} probe error: {e!r}")


def _patch_forward(mod: nn.Module, tag: str) -> int:
    """Swap one module's bound forward for a stats-printing wrapper."""
    orig = mod.forward
    if getattr(orig, "_zprobe", False):
        return 0

    def wrapped(*a, _orig=orig, _tag=tag, **kw):
        out = _orig(*a, **kw)
        _probe_named(_tag, a, kw, out)
        return out

    wrapped._zprobe = True
    mod.forward = wrapped
    return 1


_CUSTOM_OP_NAMES = (
    "npu_kv_quant_sparse_attn_sharedkv",
    "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "inplace_partial_rotary_mul",
    "npu_quant_lightning_indexer",
    "npu_quant_lightning_indexer_metadata",
    "compressor",
    "indexer_compress_epilog",
)
# Writer of the arch35 packed-KV pool; a torch_npu builtin whose version
# follows the image's torch_npu, while the reader comes from the vendors
# package -- the pair can skew across images.
_NPU_OP_NAMES = ("kv_compress_epilog",)


def _op_probe(name: str, args, kwargs, res) -> None:
    if not (_ON and _extend_window[0] and _extend_ct <= _MAX_EXTEND):
        return
    global _layer_print_budget
    if _layer_print_budget <= 0:
        return
    _layer_print_budget -= 1
    try:
        inp = kwargs.get("x")
        if not isinstance(inp, torch.Tensor):
            inp = kwargs.get("q") if not isinstance(kwargs.get("q"), str) else None
        if not isinstance(inp, torch.Tensor):
            inp = next((a for a in args if isinstance(a, torch.Tensor)), None)
        outs = res if isinstance(res, (tuple, list)) else (res,)
        o0 = next((t for t in outs if isinstance(t, torch.Tensor)), None)
        # inplace_* ops mutate args and return None: the "in" stats read here
        # are already the post-call (output) values for them.
        _p(
            f"extend {_extend_ct} op {name}: "
            f"in[{_stats(inp) if inp is not None else '?'}] "
            f"out[{_stats(o0) if o0 is not None else 'None(inplace)'}]"
        )
    except Exception as e:  # noqa: BLE001
        _p(f"op {name} probe error: {e!r}")


def _install_custom_ops_stats() -> None:
    """Wrap every torch.ops.custom.* call site via namespace-level patching.

    torch's _OpNamespace resolves ops lazily through __getattr__ and caches;
    assigning a plain attribute shadows the resolved overload packet, so all
    sglang call sites (ascend_dsv4_backend, dsv4_rope, ...) route through the
    wrapper without touching those files.
    """
    if not _ON:
        return
    n = 0
    for ns_obj, names in (
        (torch.ops.custom, _CUSTOM_OP_NAMES),
        (torch.ops.npu, _NPU_OP_NAMES),
    ):
        for name in names:
            try:
                real = getattr(ns_obj, name)
            except Exception:  # noqa: BLE001 - absence is the information
                continue
            if callable(real) and not getattr(real, "_zprobe", False):

                def wrapped(*a, _real=real, _name=name, **kw):
                    res = _real(*a, **kw)
                    _op_probe(_name, a, kw, res)
                    return res

                wrapped._zprobe = True
                try:
                    setattr(ns_obj, name, wrapped)
                    n += 1
                except Exception as e:  # noqa: BLE001
                    _p(f"wrap {name} failed: {e!r}")
    _p(f"custom op stats armed on {n} ops")


_packed_kv_done = False
_ARCH35_GROUP = 64  # _NPU_ARCH35_KV_QUANT_GROUP_SIZE in dsv4_memory_pool.py


def probe_packed_kv(*, pool, buf, loc, cache) -> None:
    """Inspect scale bytes of the arch35 packed-KV row right after the write.

    Row layout (writer convention): [rope rope_dim x bf16 = 2*rope_dim B]
    [nope fp8 payload nope_dim B] [E8M0 scale, 1 B per quant group]. A healthy
    scale byte sits near 0x7F (2^0); 0xF0..0xFF bytes mean the reader would
    dequant with ~2^115..2^128 -- the 1e37 garbage signature.
    """
    global _packed_kv_done
    if not (_ON and _extend_window[0] and not _packed_kv_done):
        return
    _packed_kv_done = True
    try:
        nope, rope, group = (
            pool.qk_nope_head_dim,
            pool.qk_rope_head_dim,
            _ARCH35_GROUP,
        )
        off = nope + 2 * rope
        n = (nope + group - 1) // group
        rows = buf.reshape(-1, buf.shape[-1])
        raw = rows.contiguous().view(torch.uint8)
        _p(
            f"kv layout: nope={nope} rope={rope} group={group} off={off} "
            f"nscale={n} buf={tuple(buf.shape)} dtype={buf.dtype} "
            f"row_bytes={rows.shape[-1]}"
        )
        _p(f"kv quant input (pre-write bf16 cache): {_stats(cache)}")
        slots = loc.reshape(-1)[:4].to(torch.int64).cpu().tolist()
        for s in slots:
            row = raw[s]
            _p(
                f"kv slot {s}: scale@{off}={row[off : off + n].tolist()} "
                f"rope_head={row[:8].tolist()} "
                f"nope_head={row[2 * rope : 2 * rope + 8].tolist()}"
            )
    except Exception as e:  # noqa: BLE001
        _p(f"kv probe error: {e!r}")


def _install_packed_kv_probe() -> None:
    """Wrap DSV4NPUTokenToKVPool._write_a5_packed_kv without editing its file."""
    if not _ON:
        return
    try:
        import sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool as mp

        orig = mp.DSV4NPUTokenToKVPool._write_a5_packed_kv
        if getattr(orig, "_zprobe", False):
            return

        def wrapped(self, *, buf, loc, cache, _orig=orig):
            _orig(self, buf=buf, loc=loc, cache=cache)
            probe_packed_kv(pool=self, buf=buf, loc=loc, cache=cache)

        wrapped._zprobe = True
        mp.DSV4NPUTokenToKVPool._write_a5_packed_kv = wrapped
        _p("packed kv probe armed on _write_a5_packed_kv")
    except Exception as e:  # noqa: BLE001
        _p(f"packed kv probe install error: {e!r}")


def _install_attn_capture() -> None:
    """Capture the first sparse-attention op call (q, kv pages, metadata, out).

    SGLANG_ZERO_PROBE_DUMP_DIR=<dir> enables it; the dump feeds
    a5_attn_op_replay.py for cross-image differential testing of the
    npu_kv_quant_sparse_attn_sharedkv op: identical inputs must give
    identical outputs on every image.
    """
    if not (_ON and _DUMP_DIR and not _attn_dumped[0]):
        return
    try:
        import sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend as be_mod

        real_ops = be_mod._sparse_attn_ops

        def capturing_ops(is_dspark=False):
            meta_op, attn_op = real_ops(is_dspark)

            def wrapped_attn(q, **kwargs):
                res = attn_op(q, **kwargs)
                if not _attn_dumped[0]:
                    _attn_dumped[0] = True
                    try:
                        _dump_attn_call(q, kwargs, res)
                    except Exception as e:  # noqa: BLE001
                        _p(f"attn dump error: {e!r}")
                return res

            return meta_op, wrapped_attn

        be_mod._sparse_attn_ops = capturing_ops
        _p(f"attn op capture armed (dump dir {_DUMP_DIR})")
    except Exception as e:  # noqa: BLE001
        _p(f"attn capture install error: {e!r}")


def _dump_attn_call(q, kwargs, res) -> None:
    os.makedirs(_DUMP_DIR, exist_ok=True)
    out = res[0] if isinstance(res, tuple) else res
    payload = {"q": q.detach().cpu(), "out": out.detach().cpu()}
    scalars, tensors = {}, {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.detach()
        else:
            scalars[k] = v
    # The kv pool view can span the whole pool; keep only the pages the
    # block table references so the dump stays small and portable.
    kv = tensors.get("ori_kv")
    bt = tensors.get("ori_block_table")
    if kv is not None and bt is not None and kv.dim() >= 1:
        idx = torch.unique(bt.detach().cpu().long())
        payload["page_indices"] = idx
        payload["ori_kv_pages"] = kv.detach().cpu()[idx.to(kv.device)]
        tensors.pop("ori_kv")
        tensors.pop("ori_block_table", None)
        payload["ori_block_table"] = bt.detach().cpu()
    for k, v in tensors.items():
        payload[f"t_{k}"] = v.cpu()
    payload["scalars"] = scalars
    path = os.path.join(_DUMP_DIR, "attn_call_r0.pt")
    torch.save(payload, path)
    _p(f"attn op call dumped: {path} (out {_stats(out)})")


def _install(model) -> None:
    """Patch forwards of decoder layers and their attn/moe submodules.

    The module objects themselves stay in place: code that walks
    layers[i].self_attn & co. keeps seeing the real modules. Submodule-level
    tags (attn / moe / final_norm / lm_head) pin the failing op family in one
    run instead of a layer index alone.
    """
    global _installed
    if _installed:
        return
    _installed = True
    try:
        core = getattr(model, "model", model)
        layers = getattr(core, "layers", None)
        if layers is None or not isinstance(layers, nn.ModuleList):
            _p(f"no .model.layers found on {type(model)}")
            return
        n = 0
        for i, layer in enumerate(layers):
            n += _patch_forward(layer, f"layer {i}")
            for attr, part in (("self_attn", "attn"), ("mlp", "moe")):
                sub = getattr(layer, attr, None)
                if isinstance(sub, nn.Module):
                    n += _patch_forward(sub, f"layer {i} {part}")
        n += _patch_forward(core.norm, "final_norm") if isinstance(
            getattr(core, "norm", None), nn.Module
        ) else 0
        lm_head = getattr(model, "lm_head", None)
        n += _patch_forward(lm_head, "lm_head") if isinstance(lm_head, nn.Module) else 0
        _p(f"patched forward on {n} modules")
        # Zero logits can mean dead weights as easily as dead kernels: print
        # the head weights once so "checkpoint never loaded" is ruled out
        # from the same log.
        for path, mod in (
            ("embed_tokens", getattr(core, "embed_tokens", None)),
            ("lm_head", lm_head),
            ("final_norm", getattr(core, "norm", None)),
        ):
            w = getattr(mod, "weight", None)
            if isinstance(w, torch.Tensor):
                _p(f"weights {path}: {_stats(w)}")
    except Exception as e:  # noqa: BLE001
        _p(f"install error: {e!r}")


def probe_forward(model, input_ids, positions, forward_batch) -> None:
    """First line of DeepseekV4ForCausalLM.forward."""
    global _extend_ct
    if not _ON:
        return
    try:
        _fingerprint()
        _install_attn_capture()
        _install_custom_ops_stats()
        _install_packed_kv_probe()
        _install(model)
        # str() of an IntEnum member is its int on py>=3.11 ("1"), so match on
        # the name / the mode's own predicate instead of the stringified value.
        fm = getattr(forward_batch, "forward_mode", None)
        mname = str(getattr(fm, "name", None) or fm)
        try:
            is_extend = bool(fm.is_extend()) if hasattr(fm, "is_extend") else False
        except Exception:  # noqa: BLE001
            is_extend = False
        if not is_extend:
            is_extend = "EXTEND" in mname or "MIXED" in mname
        # Sticky until the next forward entry: layer wrappers of THIS forward
        # read it. Non-extend (decode/capture warmup) closes the window.
        _extend_window[0] = is_extend
        if not is_extend or _extend_ct > _MAX_EXTEND:
            return
        _extend_ct += 1
        fb_head = {
            k: getattr(forward_batch, k, None)[:8].detach().cpu().tolist()
            for k in ("seq_lens", "out_cache_loc")
            if getattr(forward_batch, k, None) is not None
        }
        _p(
            f"extend {_extend_ct}: mode={mname} "
            f"bs={getattr(forward_batch, 'batch_size', '?')} "
            f"input_ids[:16]={input_ids[:16].detach().cpu().tolist()} "
            f"positions[:8]={positions[:8].detach().cpu().tolist()} {fb_head}"
        )
    except Exception as e:  # noqa: BLE001
        _p(f"forward probe error: {e!r}")
