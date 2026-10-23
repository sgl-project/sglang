"""Experiment hooks for SGLang v0.5.22 + Qwen3.6-35B-A3B-FP8, loaded via PYTHONPATH=/opt/coral.

Each experiment is an independent env switch, installed as a post-import hook on the module it patches so it
also lands in the scheduler/TP subprocesses (they are spawned, not forked):

CORAL_MONOMOE=1      routed-expert MoE for M <= 8 tokens -> flashinfer mono_moe (FlashInfer 0.6.17).
CORAL_MONOMOE_CHECK=N  for the first N eager (non-capture) calls, also run the stock MoE and log rel. error.
CORAL_NORMQUANT=1    residual GemmaRMSNorm (hidden 2048) -> one Triton kernel that also emits the FP8
                     per-token-group(128) quant; the following block-FP8 linear consumes it pre-quantized
                     (bit-identical to stock; removes one quant launch per norm->FP8-linear edge).
CORAL_GEMV=1         unquantized BF16 linears with N <= 256, K = 2048 (in_proj_ba, MoE router) at M <= 4 tokens ->
                     rows-in-registers Triton GEMV (cuBLAS picks too few CTAs for these cold, tiny weights).
CORAL_SINGLE_STREAM_MOE=1  run the MoE block's shared expert and routed experts sequentially on one stream
                     (no alt-stream overlap, no hidden_states.clone()) so a persistent MoE kernel owns the GPU.
CORAL_FP8GEMV=1      block-FP8 linears at M <= 4 tokens -> weight-only FP8 split-K GEMV (bf16 activations, no
                     activation quant; per-shape tuned) except GDN in_proj_qkvz, where DeepGEMM is faster.
CORAL_MOE=1          whole MoE block (routed top-8 + sigmoid-gated shared expert) at M <= 4 tokens -> router GEMM +
                     2 fused W8A16 Triton launches (coral_moe_decode); bypasses SGLang's dual-stream MoE path.
CORAL_MOE_FP4=1      NVFP4 W4A16 target (nvidia/Qwen3.6-35B-A3B-NVFP4, --moe-runner-backend marlin): whole MoE block at
                     M <= CORAL_MOE_FP4_MAX_M (default 8) -> coral_moe_fp4.coral_moe_decode_fp4_mma (route + expert-grouped
                     e2m1 mma up + expert-loop down). Keeps a checkpoint-layout copy of the expert and shared-expert
                     weights before Marlin/FlashInfer repack them (~18 GB; prefill stays on stock Marlin).
                     CORAL_MOE_FP4_CHECK=N logs rel. error vs the stock path for the first N eager calls.
CORAL_GDN_SINGLE_STREAM=1  run GatedDeltaNet's in_proj_ba on the main stream: on the side stream it waits for the
                     SM-filling qkvz DeepGEMM to drain (measured 9.95 us in situ vs 2.08 us for the same GEMV).
"""
import importlib.abc, importlib.util, json, os, sys, weakref

FP8_MOD = "sglang.srt.layers.quantization.fp8"
NORM_MOD = "sglang.srt.layers.layernorm"
UNQ_MOD = "sglang.srt.layers.quantization.unquant"
QMOE_MOD = "sglang.srt.models.qwen2_moe"
Q35_MOD = "sglang.srt.models.qwen3_5"
GDNK_MOD = "sglang.srt.layers.attention.linear.kernels.gdn_triton"
GDNV_MOD = "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent"


def _log(msg):
    print(f"[coral] {msg} (pid {os.getpid()})", file=sys.stderr, flush=True)


def _install_monomoe(mod):
    import torch
    # flashinfer / token_dispatcher are imported lazily: importing them while fp8.py is still being
    # initialised re-enters sglang.srt.configs.model_config (circular import).
    E, N, K, MAX_M = 256, 512, 2048, 8
    check_left = [int(os.environ.get("CORAL_MONOMOE_CHECK", "0"))]
    # Per-device scratchpad + two 8-row output buffers. Layers alternate buffers by load order, so a layer's
    # output is never overwritten by the very next MoE layer while it may still alias that layer's input.
    bufs, order = {}, {}
    cls = mod.Fp8MoEMethod
    orig_pwal, orig_apply = cls.process_weights_after_loading, cls.apply

    def eligible(layer):
        w13 = getattr(layer, "w13_weight", None)
        return (w13 is not None and w13.dtype == torch.float8_e4m3fn and tuple(w13.shape) == (E, 2 * N, K)
                and tuple(layer.w2_weight.shape) == (E, K, N) and hasattr(layer, "w13_weight_scale_inv"))

    def process_weights_after_loading(self, layer):
        import flashinfer.fused_moe as fim
        orig_pwal(self, layer)
        if eligible(layer):
            # Duplicate gate/up in MonoMoE's TMA/WGMMA stripe order (+512 MB/layer); stock tensors stay for M > 8.
            layer._mono_w13 = fim.monomoe.interleave_for_tma_wgmma_up(layer.w13_weight.data).contiguous()
            layer._mono_s13 = layer.w13_weight_scale_inv.data.float().contiguous()
            layer._mono_s2 = layer.w2_weight_scale_inv.data.float().contiguous()
            dev = layer.w13_weight.device
            if dev not in bufs:
                bufs[dev] = (fim.monomoe.alloc_scratchpad(dev),
                             [torch.empty(MAX_M, K, dtype=torch.bfloat16, device=dev) for _ in range(2)])
            layer._mono_parity = order.setdefault(dev, [0])[0] % 2
            order[dev][0] += 1

    def apply(self, layer, dispatch_output):
        x = dispatch_output.hidden_states
        M = x.shape[0]
        if 0 < M <= MAX_M and hasattr(layer, "_mono_w13") and x.dtype == torch.bfloat16:
            import flashinfer.fused_moe as fim
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
            topk = dispatch_output.topk_output
            logits = topk.router_logits.to(torch.bfloat16)
            scratch, outs = bufs[x.device]
            # mono_moe always stores 8 rows: give it an 8-row buffer and hand back the M-row view.
            out = fim.mono_moe(x, logits, layer._mono_w13, layer._mono_s13, layer.w2_weight.data, layer._mono_s2,
                               topk.topk_ids.shape[1], scoring_func="softmax", renormalize=True,
                               out=outs[layer._mono_parity][:M], scratchpad=scratch, interleave_up=False)
            if check_left[0] > 0 and not torch.cuda.is_current_stream_capturing():
                check_left[0] -= 1
                ref = orig_apply(self, layer, dispatch_output).hidden_states.float()
                err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                _log(f"monomoe M={M} rel_err_vs_stock={err:.4f}")
            return StandardCombineInput(hidden_states=out)
        return orig_apply(self, layer, dispatch_output)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    _log(f"monomoe patched {cls.__module__}.{cls.__name__}")


# Pre-quantized norm outputs, keyed by data_ptr. A consumer may see the tagged tensor only through a view (views
# of views point _base at the root buffer, so attributes can't be followed). An entry is valid only while the
# tagged tensor is alive (weakref -> its storage can't have been reused) and x matches it exactly.
_PREQ = {}


def _install_normquant_linear(mod):
    import torch
    cls = mod.Fp8LinearMethod
    orig_apply = cls.apply

    def lookup(x):
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            return None
        e = _PREQ.get(x.data_ptr())
        if e is None:
            return None
        wr, shape, stride, q, s, M, ver = e
        if wr() is None or x.shape != shape or x.stride() != stride:
            return None
        return (q, s, M, ver)

    def apply(self, layer, x, bias=None, *args, **kwargs):
        pre = lookup(x)
        # Use the norm's pre-quantized copy only if x is untouched since (torch in-place ops bump _version,
        # which a view shares with its base).
        if pre is not None and self.block_quant and pre[3] == x._version and layer.weight.shape[1] == x.shape[1]:
            q, s, M, _ = pre
            return orig_apply(self, layer, (q, s), bias, *args, **kwargs)[:M]
        return orig_apply(self, layer, x, bias, *args, **kwargs)

    cls.apply = apply
    _log(f"normquant patched {cls.__module__}.{cls.__name__}.apply")


def _install_normquant_norm(mod):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from coral_kernels import add_gemma_rmsnorm_quant, add_gemma_rmsnorm
    cls = mod.GemmaRMSNorm
    orig = cls._forward_impl

    def _forward_impl(self, x, residual=None, post_residual_addition=None):
        if (residual is not None and post_residual_addition is None and x.dim() == 2 and x.shape[1] == 2048
                and x.is_contiguous() and residual.is_contiguous()):
            if _NORM_NOQUANT and x.shape[0] <= _MAX_M:  # consumers at decode sizes are W8A16: skip the FP8 copy
                add_gemma_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
                return x, residual
            q, s = add_gemma_rmsnorm_quant(x, residual, self.weight.data, self.variance_epsilon)
            _PREQ[x.data_ptr()] = (weakref.ref(x), x.shape, x.stride(), q, s, x.shape[0], x._version)
            if len(_PREQ) > 4096:  # dead entries are harmless (weakref check); just bound the dict
                for k in [k for k, v in _PREQ.items() if v[0]() is None]: _PREQ.pop(k, None)
            return x, residual
        return orig(self, x, residual, post_residual_addition)

    cls._forward_impl = _forward_impl
    _log(f"normquant patched {cls.__module__}.{cls.__name__}._forward_impl")


def _install_gemv(mod):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from coral_kernels import rows_gemv
    cls = mod.UnquantizedLinearMethod
    orig_apply = cls.apply

    def apply(self, layer, x, bias=None, *args, **kwargs):
        w = getattr(layer, "weight", None)
        if (bias is None and not args and not kwargs and w is not None and w.dim() == 2 and w.dtype == torch.bfloat16
                and w.shape[0] <= 256 and w.shape[1] == 2048 and isinstance(x, torch.Tensor) and x.dim() == 2
                and x.dtype == torch.bfloat16 and 0 < x.shape[0] <= _MAX_M and x.is_contiguous() and w.is_contiguous()):
            return rows_gemv(x, w)
        return orig_apply(self, layer, x, bias, *args, **kwargs)

    cls.apply = apply
    _log(f"gemv patched {cls.__module__}.{cls.__name__}.apply")


def _install_single_stream(mod):
    cls = mod.Qwen2MoeSparseMoeBlock
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.alt_stream = None

    cls.__init__ = __init__
    _log(f"single-stream MoE patched {cls.__module__}.{cls.__name__}")


def _install_gdn_single_stream(mod):
    cls = mod.Qwen3_5GatedDeltaNet
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.alt_stream = None

    cls.__init__ = __init__
    _log(f"gdn single-stream patched {cls.__module__}.{cls.__name__}")


def _install_moe(mod):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from coral_kernels import coral_moe_decode, coral_moe_decode_tab, coral_moe_decode_v9, coral_moe_decode_v10c
    cls = mod.Qwen2MoeSparseMoeBlock
    orig_forward = cls.forward
    cfg = [int(v) for v in os.environ.get("CORAL_MOE_CFG", "32,4,64,2").split(",")]
    max_m = int(os.environ.get("CORAL_MOE_MAX_M", "4"))
    assert 1 <= max_m <= 16, "CORAL_MOE_MAX_M: v9 / v10c go to 16 tokens"
    route = os.environ.get("CORAL_MOE_ROUTE", "")  # tab = v6 (route once, 3 launches); v9 = route once + no-split up
    use_tab = route == "tab"
    fakeq = os.environ.get("CORAL_MOE_FAKEQ") == "1"  # ablation: stock FP8 activation numerics inside our MoE  # larger M falls through (e.g. to MonoMoE for MTP verify)

    def weights(self):
        c = getattr(self, "_coral_moe", None)
        if c is None:
            ex, se = self.experts, self.shared_expert
            f32 = lambda t: t.data.float().contiguous()
            # linear scales may be stored TMA-aligned/padded for DeepGEMM: take the logical [N/128, K/128] block
            lin = lambda m: m.weight_scale_inv.data[:(m.weight.shape[0] + 127) // 128, :m.weight.shape[1] // 128].float().contiguous()
            c = self._coral_moe = dict(
                w13=ex.w13_weight.data, s13=f32(ex.w13_weight_scale_inv), w2=ex.w2_weight.data,
                s2=f32(ex.w2_weight_scale_inv), sh13=se.gate_up_proj.weight.data,
                shs13=lin(se.gate_up_proj), sh2=se.down_proj.weight.data, shs2=lin(se.down_proj), sg=self.shared_expert_gate.weight.data.reshape(-1).contiguous(),
                renorm=bool(self.topk.topk_config.renormalize), top_k=int(self.topk.topk_config.top_k))
        return c

    def eligible(self, x):
        ex = getattr(self, "experts", None)
        return (x.dim() == 2 and 0 < x.shape[0] <= max_m and x.dtype == torch.bfloat16 and self.shared_expert is not None
                and self.shared_expert_gate is not None and getattr(self, "tp_size", 1) == 1 and ex is not None
                and getattr(ex, "w13_weight", None) is not None and ex.w13_weight.dtype == torch.float8_e4m3fn
                and tuple(ex.w13_weight.shape) == (256, 1024, 2048) and self.topk.topk_config.top_k == 8
                and not self.topk.topk_config.use_grouped_topk and self.topk.topk_config.correction_bias is None)

    stats_path = os.environ.get("CORAL_MOE_STATS")  # measurement only (needs --disable-cuda-graph): distinct experts
    stats = {}

    def forward(self, hidden_states, forward_batch=None, *args, defer_finalize=False, **kwargs):
        # v0.5.21: decoder layers always pass defer_finalize= (True only with the multi-GPU all-reduce fusion; at TP1
        # the stock reduce_moe_output is a no-op, so returning y directly stays equivalent)
        num_tokens, hidden = hidden_states.shape
        x = hidden_states.view(-1, hidden)
        if not defer_finalize and not args and not kwargs and eligible(self, x):
            c = weights(self)
            nrm = getattr(self, "_coral_norm", None)
            st = getattr(nrm, "_coral_lg", None) if nrm is not None else None
            if st is not None and st[0] == x.data_ptr() and st[1] == tuple(x.shape) and st[2] == x._version:
                logits = st[3]  # computed by the fused add+norm+router launch (CORAL_NORM_ROUTER)
            else:
                logits, _ = self.gate(x)
            if stats_path and 1 < x.shape[0] <= 16:
                ids = logits.float().topk(c["top_k"], dim=-1).indices
                key = int(x.shape[0])
                u = int(torch.unique(ids).numel())
                h = stats.setdefault(key, {})
                h[u] = h.get(u, 0) + 1
                if sum(sum(v.values()) for v in stats.values()) % 2000 == 0:
                    with open(stats_path, "w") as f:
                        json.dump({str(k): v for k, v in stats.items()}, f)
            if _DBG_GPU_SLEEP and x.shape[0] > 1:  # DIAGNOSTIC ONLY: extra GPU time per verify MoE layer
                torch.cuda._sleep(_DBG_GPU_SLEEP)
            if route == "v10c" and not fakeq and x.shape[0] > 1:
                # expert-grouped (one owner CTA per distinct expert); for DFlash2 verify at M = 8 / 16. M = 1 -> v9.
                y = coral_moe_decode_v10c(x.contiguous(), logits.to(torch.bfloat16).contiguous(), c["sg"], c["w13"],
                                          c["s13"], c["sh13"], c["shs13"], c["w2"], c["s2"], c["sh2"], c["shs2"],
                                          top_k=c["top_k"], renorm=c["renorm"], max_m=max_m)
            elif route in ("v9", "v10c") and not fakeq:
                y = coral_moe_decode_v9(x.contiguous(), logits.to(torch.bfloat16).contiguous(), c["sg"], c["w13"],
                                        c["s13"], c["sh13"], c["shs13"], c["w2"], c["s2"], c["sh2"], c["shs2"],
                                        top_k=c["top_k"], renorm=c["renorm"], max_m=max(max_m, 4))
            elif use_tab and not fakeq:
                y = coral_moe_decode_tab(x.contiguous(), logits.to(torch.bfloat16).contiguous(), c["sg"], c["w13"],
                                         c["s13"], c["sh13"], c["shs13"], c["w2"], c["s2"], c["sh2"], c["shs2"],
                                         top_k=c["top_k"], renorm=c["renorm"], bn_up=cfg[0], sk_up=cfg[1],
                                         bn_dn=cfg[2], sk_dn=cfg[3])
            else:
                y = coral_moe_decode(x.contiguous(), logits.to(torch.bfloat16).contiguous(), c["sg"], c["w13"], c["s13"],
                                     c["sh13"], c["shs13"], c["w2"], c["s2"], c["sh2"], c["shs2"], top_k=c["top_k"],
                                     renorm=c["renorm"], bn_up=cfg[0], sk_up=cfg[1], bn_dn=cfg[2], sk_dn=cfg[3],
                                     fakeq=fakeq)
            return y.view(num_tokens, hidden)
        return orig_forward(self, hidden_states, forward_batch, *args, defer_finalize=defer_finalize, **kwargs)

    cls.forward = forward
    _log(f"moe patched {cls.__module__}.{cls.__name__}.forward cfg={cfg} route={route or 'per-cta'}")


MODELOPT_MOD = "sglang.srt.layers.quantization.modelopt_quant"


def _install_fp4_stash(mod):
    """Keep checkpoint-layout NVFP4 tensors before SGLang repacks them (Marlin for experts, FlashInfer for linears)."""
    import torch
    orig_prep = mod.prepare_moe_nvfp4_layer_for_marlin

    def prep(layer, *a, **k):
        w13, s13, w2, s2 = layer.w13_weight.data, layer.w13_weight_scale.data, layer.w2_weight.data, layer.w2_weight_scale.data
        E = w13.shape[0]
        if (w13.dtype == torch.uint8 and tuple(w13.shape[1:]) == (1024, 1024) and tuple(w2.shape[1:]) == (2048, 256)
                and s13.dtype == torch.float8_e4m3fn and tuple(s13.shape[1:]) == (1024, 128)
                and tuple(s2.shape[1:]) == (2048, 32)):
            g13 = layer.w13_weight_scale_2.data.float().reshape(E, -1)
            g13 = g13.expand(E, 2) if g13.shape[1] == 1 else g13[:, :2]
            layer._coral_fp4 = dict(w13=w13.clone(), s13=s13.clone(), g13=g13.contiguous().clone(), w2=w2.clone(),
                                    s2=s2.clone(), g2=layer.w2_weight_scale_2.data.float().reshape(E).clone())
        elif not getattr(mod, "_coral_fp4_warned", False):
            mod._coral_fp4_warned = True
            _log(f"fp4 stash: unexpected MoE layout w13 {tuple(w13.shape)} {w13.dtype} s13 {tuple(s13.shape)} {s13.dtype}")
        return orig_prep(layer, *a, **k)

    mod.prepare_moe_nvfp4_layer_for_marlin = prep
    cls = mod.ModelOptFp4LinearMethod
    orig_pwal = cls.process_weights_after_loading

    def pwal(self, layer):
        w, sc = layer.weight.data, layer.weight_scale.data
        if (w.dtype == torch.uint8 and sc.dtype == torch.float8_e4m3fn and tuple(w.shape) in ((1024, 1024), (2048, 256))
                and tuple(sc.shape) == (w.shape[0], w.shape[1] // 8)):
            layer._coral_fp4 = (w.clone(), sc.clone(), layer.weight_scale_2.data.float().reshape(-1).clone())
        return orig_pwal(self, layer)

    cls.process_weights_after_loading = pwal
    _log(f"fp4 stash patched {mod.__name__}.prepare_moe_nvfp4_layer_for_marlin + {cls.__name__}")


def _install_moe_fp4(mod):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import coral_moe_fp4 as cf
    cls = mod.Qwen2MoeSparseMoeBlock
    orig_forward = cls.forward
    max_m = int(os.environ.get("CORAL_MOE_FP4_MAX_M", "8"))
    assert 1 <= max_m <= 16
    check_left = [int(os.environ.get("CORAL_MOE_FP4_CHECK", "0"))]

    def weights(self):
        c = getattr(self, "_coral_moe_fp4", None)
        if c is None:
            ex, se = self.experts, self.shared_expert
            gu, dn = se.gate_up_proj._coral_fp4, se.down_proj._coral_fp4
            two = lambda g: (g.expand(2) if g.numel() == 1 else g[:2]).contiguous()
            c = self._coral_moe_fp4 = dict(
                ex._coral_fp4, sh13=gu[0], shs13=gu[1], shg13=two(gu[2]), sh2=dn[0], shs2=dn[1],
                shg2=dn[2][:1].contiguous(), sg=self.shared_expert_gate.weight.data.reshape(-1).contiguous(),
                renorm=bool(self.topk.topk_config.renormalize), top_k=int(self.topk.topk_config.top_k))
        return c

    def eligible(self, x):
        se = getattr(self, "shared_expert", None)
        return (x.dim() == 2 and 0 < x.shape[0] <= max_m and x.dtype == torch.bfloat16 and se is not None
                and self.shared_expert_gate is not None and getattr(self, "tp_size", 1) == 1
                and hasattr(getattr(self, "experts", None), "_coral_fp4")
                and hasattr(se.gate_up_proj, "_coral_fp4") and hasattr(se.down_proj, "_coral_fp4")
                and self.topk.topk_config.top_k == 8 and not self.topk.topk_config.use_grouped_topk
                and self.topk.topk_config.correction_bias is None)

    def forward(self, hidden_states, forward_batch=None, *args, defer_finalize=False, **kwargs):
        num_tokens, hidden = hidden_states.shape
        x = hidden_states.view(-1, hidden)
        if not defer_finalize and not args and not kwargs and eligible(self, x):
            c = weights(self)
            logits, _ = self.gate(x)
            y = cf.coral_moe_decode_fp4_mma(
                x.contiguous(), logits.to(torch.bfloat16).contiguous(), c["sg"], c["w13"], c["s13"], c["g13"],
                c["sh13"], c["shs13"], c["shg13"], c["w2"], c["s2"], c["g2"], c["sh2"], c["shs2"], c["shg2"],
                top_k=c["top_k"], renorm=c["renorm"])
            if check_left[0] > 0 and not torch.cuda.is_current_stream_capturing():
                check_left[0] -= 1
                ref = orig_forward(self, hidden_states.clone(), forward_batch, defer_finalize=False).view(-1, hidden).float()
                err = ((y.float() - ref).norm() / ref.norm().clamp_min(1e-6)).item()
                _log(f"moe_fp4 M={x.shape[0]} rel_err_vs_stock={err:.4f}")
            return y.view(num_tokens, hidden)
        return orig_forward(self, hidden_states, forward_batch, *args, defer_finalize=defer_finalize, **kwargs)

    cls.forward = forward
    _log(f"moe_fp4 patched {cls.__module__}.{cls.__name__}.forward max_m={max_m} cfg={cf._MMA_CFG[min(max_m, 8)]}")


# (N, K) -> (block_n, split_k), tuned on H100 cold-L2 at M = 1 (fp8_gemv_bench.py, 2026-10-10).
_FP8GEMV_CFG = {(2048, 4096): (64, 16),   # GDN out_proj / attn o_proj      1.32-1.37x vs quant+DeepGEMM
                (9216, 2048): (64, 8),    # attn q(+gate)/k/v merged        1.05x
                (1024, 2048): (64, 8),    # shared expert gate_up           1.31x
                (2048, 512): (64, 4)}     # shared expert down              1.03x
# M = 1 only: no-split W8A16 (block_n, kb, num_warps), at the pure-read ceiling (fp8gemv_nosplit.py, 2026-10-11).
# Includes GDN in_proj_qkvz (12288x2048), which otherwise goes to quant + DeepGEMM. CORAL_FP8GEMV_NS=0 disables.
_FP8NS_CFG = {(2048, 4096): (4, 2048, 8), (9216, 2048): (8, 1024, 4), (12288, 2048): (16, 512, 4),
              (1024, 2048): (4, 2048, 4), (2048, 512): (8, 512, 4)}
# M = 2..4 (MTP verify): hand-written CUDA mma.sync W8A16 (coral_mma.py), (N, K, M) -> (warps, row tiles), tuned H100
# cold-L2 2026-10-11 (fp8gemv_mma_test.py): M=4 5.56 / 9.89 / 11.63 us vs split-K 8.33 / 12.69 / 15.57 (ceil 4.54 /
# 7.8 / 9.77). CORAL_FP8GEMV_MMA=1 enables.
_MMA_CFG = {(2048, 4096, 2): (16, 1), (2048, 4096, 3): (16, 1), (2048, 4096, 4): (16, 1),
            (9216, 2048, 2): (16, 1), (9216, 2048, 3): (16, 1), (9216, 2048, 4): (16, 0),
            (12288, 2048, 2): (16, 1), (12288, 2048, 3): (16, 2), (12288, 2048, 4): (16, 0),
            # key M=5 stands for 5..8: direct-load variant (rt=0), flat in M (fp8gemv_mma8_test.py: 6.03 / 9.25 / 11.43 us at M=8)
            (2048, 4096, 5): (16, 0), (9216, 2048, 5): (16, 0), (12288, 2048, 5): (16, 0),
            # key M=9 stands for 9..16: direct16 (rt=-1, two token tiles), best warps at M=9/12/16 (bench/mma16_test.py,
            # H100 2026-10-15): M=16 7.50 / 13.31 / 17.09 us vs direct M=8 6.04 / 9.26 / 11.44 (weights read once)
            (2048, 4096, 9): (16, -1), (9216, 2048, 9): (4, -1), (12288, 2048, 9): (8, -1)}
# token cap for the dense GEMV patches (CORAL_GEMV, CORAL_FP8GEMV); > 4 only for MTP with more draft tokens.
# DFlash2 verify sends block_size tokens per request (8 or 16 at c=1), so the fork allows up to 16.
_MAX_M = int(os.environ.get("CORAL_MAX_M", "4"))
assert 1 <= _MAX_M <= 16, "CORAL_MAX_M: the kernels go to 16 tokens (coral_mma direct16)"
# fused add+norm+router (CORAL_NORM_ROUTER) and the GDN z-view forward: v0.5.18 stack had both at <= 8 tokens
_NORM_ROUTER_MAX_M = max(8, _MAX_M)
# CORAL_NORM_ROUTER_MT_MIN_M=m (needs CORAL_NORM_ROUTER=1): at m..cap tokens the fused add+norm+router runs
# add_norm_router_mt (tokens in parallel) instead of add_norm_router's serial token loop. 0 (default) = never.
# The serial loop was tuned for MTP's M <= 5; at DFlash2's 8-token verify it costs 8.4 us per call vs ~3.4 at M=1
# (nsys, b8, 64-128K SWE). bench/norm_router_m16_test.py (H100, 2026-10-15): y/h/logits bit-identical to serial, but mt
# LOSES up to M=12 (M=8: 7.98 vs 7.40 us) and wins only at M=16 (12.11 vs 13.24 us, ~45 us/step over 40 layers).
# So use CORAL_NORM_ROUTER_MT_MIN_M=13 at most, and only for block 16. (block_n, kc, num_warps) = best measured.
_NORM_ROUTER_MT_MIN_M = int(os.environ.get("CORAL_NORM_ROUTER_MT_MIN_M", "0"))
_NORM_ROUTER_MT_CFG = {8: (2, 512, 4), 16: (1, 512, 4)}
_ZVIEW_MAX_T = max(8, _MAX_M)
_DBG_GPU_SLEEP = int(float(os.environ.get("CORAL_DBG_GPU_SLEEP_US", "0")) * 1755)  # DIAGNOSTIC: ~cycles per us
# CORAL_NORM_NOQUANT=1 (needs CORAL_FP8GEMV=1): at M <= _MAX_M the norm skips its FP8 copy (no consumer reads it)
_NORM_NOQUANT = os.environ.get("CORAL_NORM_NOQUANT") == "1" and os.environ.get("CORAL_FP8GEMV") == "1"


def _install_fp8gemv(mod):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from coral_kernels import fp8w_gemv_splitk, fp8w_gemv_ns
    use_ns = os.environ.get("CORAL_FP8GEMV_NS", "1") == "1"
    use_mma = os.environ.get("CORAL_FP8GEMV_MMA") == "1"
    if use_mma:
        import coral_mma
        coral_mma._load()  # build/load the extension now, never inside CUDA-graph capture
    cls = mod.Fp8LinearMethod
    inner = cls.apply  # installed after normquant -> outermost wrapper

    def apply(self, layer, x, bias=None, *args, **kwargs):
        w = getattr(layer, "weight", None)
        if (bias is None and not args and not kwargs and self.block_quant and isinstance(x, torch.Tensor)
                and x.dim() == 2 and 0 < x.shape[0] <= _MAX_M and x.dtype == torch.bfloat16 and w is not None
                and w.dtype == torch.float8_e4m3fn):
            shp = tuple(w.shape)
            ns = use_ns and x.shape[0] == 1 and shp in _FP8NS_CFG
            Mx = x.shape[0]
            mma = _MMA_CFG.get(shp + (Mx if Mx <= 4 else 5 if Mx <= 8 else 9,)) if use_mma else None
            if ns or mma or (shp in _FP8GEMV_CFG and x.shape[0] <= 4):
                s = getattr(layer, "_coral_s", None)
                if s is None:  # contiguous fp32 [N/128, K/128] copy (the live tensor may be TMA-aligned/transposed)
                    N, K = w.shape
                    s = layer._coral_s = layer.weight_scale_inv.data[:(N + 127) // 128, :K // 128].float().contiguous()
                cctx = getattr(layer, "_coral_conv_ctx", None)
                if cctx is not None and x.shape[0] == 1 and shp == (12288, 2048):
                    from coral_kernels import fp8w_gemv_ns_conv
                    cs, ci, cw, cb = cctx
                    y = fp8w_gemv_ns_conv(x.contiguous(), w.data, s, cs, ci, cw, cb, 8192, True, 16, 512, 2)
                    _CONV_DONE[y.data_ptr()] = y._version
                    layer._coral_conv_ctx = None
                    if not _CONV_DONE.get("_said"):
                        _CONV_DONE["_said"] = 1
                        _log("gdn conv-fold active (qkvz GEMV + conv in one launch)")
                    return y
                if mma:
                    return coral_mma.w8a16_mma(x.contiguous(), w.data, s, mma[0], mma[1])
                if ns:
                    bn, kb, nw = _FP8NS_CFG[shp]
                    return fp8w_gemv_ns(x.contiguous(), w.data, s, bn, kb, nw)
                bn, sk = _FP8GEMV_CFG[shp]
                return fp8w_gemv_splitk(x.contiguous(), w.data, s, bn, sk)
        return inner(self, layer, x, bias, *args, **kwargs)

    cls.apply = apply
    _log(f"fp8gemv patched {cls.__module__}.{cls.__name__}.apply ns={use_ns} mma={use_mma}")


def _install_gdn_bv(mod):
    """Relaunch SGLang's own packed GDN decode kernel with BV=CORAL_GDN_BV (default 4), 1 warp: 1,024 CTAs instead of
    128 for HV=32, V=128. Same kernel, same per-element math -> bit-identical out + state (gdn_recur_tune.py: 4.71 ->
    2.52 us/layer at B=1). Only the launch grid changes; argument validation is SGLang's (called on first use)."""
    import triton
    from sglang.kernels.ops.attention.fla import fused_recurrent as fr
    kern = fr.fused_recurrent_gated_delta_rule_packed_decode_kernel
    orig = mod.fused_recurrent_gated_delta_rule_packed_decode
    bv = int(os.environ.get("CORAL_GDN_BV", "4"))
    checked = set()

    def packed_decode(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out, ssm_state_indices,
                      use_qk_l2norm_in_kernel=False):
        key = (tuple(mixed_qkv.shape), tuple(initial_state.shape[1:]))
        if key not in checked:  # first call per shape: let SGLang validate (and compute) once, eagerly
            checked.add(key)
            return orig(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out, ssm_state_indices,
                        use_qk_l2norm_in_kernel)
        B = mixed_qkv.shape[0]
        HV, V, K = initial_state.shape[-3:]
        H = (mixed_qkv.shape[1] - HV * V) // 2 // K
        kern[(triton.cdiv(V, bv), B * HV)](
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, o=out, h0=initial_state, ht=initial_state,
            ssm_state_indices=ssm_state_indices, scale=scale, stride_mixed_qkv_tok=mixed_qkv.stride(0),
            stride_a_tok=a.stride(0), stride_b_tok=b.stride(0), stride_init_state_token=initial_state.stride(0),
            stride_final_state_token=initial_state.stride(0), stride_indices_seq=ssm_state_indices.stride(0),
            H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=bv, SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, num_warps=1, num_stages=3)
        return out

    mod.fused_recurrent_gated_delta_rule_packed_decode = packed_decode
    _log(f"gdn packed decode relaunched BV={bv} in {mod.__name__}")


def _install_gdn_split_views(mod):
    """Qwen3.5/3.6 GDN decode: fused_qkvzba_split_reshape_cat_contiguous only copies [q|k|v|z] -> [q|k|v] + z and
    [b|a] -> b, a. With the contiguous projection layout those are slices of the inputs, so return views (zero
    kernels). Consumers (causal_conv1d_update, packed GDN decode) take row strides; z is made contiguous only when
    B > 1 (a [B,HV,V] slice with row stride 12288 cannot be flattened as a view)."""
    import torch
    orig = mod.fused_qkvzba_split_reshape_cat_contiguous

    def split_views(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
        if mixed_qkvz.dim() != 2 or mixed_qkvz.stride(-1) != 1 or mixed_ba.stride(-1) != 1:
            return orig(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)
        B = mixed_qkvz.shape[0]
        qkv_dim = 2 * num_heads_qk * head_qk + num_heads_v * head_v
        mixed_qkv = mixed_qkvz[:, :qkv_dim]
        z = mixed_qkvz[:, qkv_dim:qkv_dim + num_heads_v * head_v].reshape(B, num_heads_v, head_v)
        if B > 1:
            z = z.contiguous()
        b = mixed_ba[:, :num_heads_v]
        a = mixed_ba[:, num_heads_v:2 * num_heads_v]
        return mixed_qkv, z, b, a

    mod.fused_qkvzba_split_reshape_cat_contiguous = split_views
    _log(f"gdn split -> views in {mod.__name__}")


def _install_gdn_verify_bv(mod):
    """MTP target-verify GDN kernel (fused_sigmoid_gating_delta_rule_update): SGLang launches BV=32 / 1 warp
    (128 CTAs). Proxy the kernel object so only BV and the V-block grid dim change; same per-element math ->
    bit-identical o + intermediate states (gdn_verify_tune.py: 11.31 -> 6.61 us/layer at T=4, BV=4)."""
    import triton
    bv = int(os.environ.get("CORAL_GDN_VERIFY_BV", "4"))
    kern = mod.fused_sigmoid_gating_delta_rule_update_kernel

    class _BV:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                kw = dict(kw)
                kw["BV"] = bv
                nv = triton.cdiv(kw["V"], bv)
                # v0.5.21 on CUDA launches (NV, N, HV) with SPLIT_N_HV_GRID=True (fused_sigmoid_gating_recurrent.py:
                # 105, 454); v0.5.18 launched (NK, NV, N*HV). Replacing grid[1] on the new layout would run 32
                # sequence programs and compute only V columns < 16 (garbage o, out-of-bounds cu_seqlens reads).
                g = (nv, grid[1], grid[2]) if kw.get("SPLIT_N_HV_GRID") else (grid[0], nv, grid[2])
                return kern[g](*a, **kw)
            return launch

    mod.fused_sigmoid_gating_delta_rule_update_kernel = _BV()
    _log(f"gdn verify kernel relaunched BV={bv}")


def _install_fa_verify_splits(mod):
    """FA3 target verify (DFLASH/EAGLE TARGET_VERIFY goes through forward_extend): run with num_splits=CORAL_FA_VERIFY_SPLITS
    instead of 0 (= FA3's own heuristic). At batch 1 with 2 KV heads, pack-GQA puts 8 verify tokens x 8 q heads in one
    M tile per KV head, so only split-KV fills the 132 SMs; the b8 profile had the 10 attention layers at 0.92 ms vs a
    ~0.6 ms bandwidth floor at ~100K context (2 KB KV per token per layer), and under CUDA graphs the heuristic can be
    sized at capture time rather than for the live context. Split-KV changes the softmax combine order, so outputs
    can move in the last bits: bench/fa_verify_splits_test.py measures time and error first. Prefill and decode keep
    the backend's setting (decode has its own precomputed scheduler metadata keyed on num_splits)."""
    n = int(os.environ["CORAL_FA_VERIFY_SPLITS"])
    cls = mod.FlashAttentionBackend
    orig = cls.forward_extend

    def forward_extend(self, *a, **kw):
        fb = kw["forward_batch"] if "forward_batch" in kw else a[4]  # (q, k, v, layer, forward_batch, ...)
        if not fb.forward_mode.is_target_verify():
            return orig(self, *a, **kw)
        keep = self.num_splits
        self.num_splits = n
        try:
            return orig(self, *a, **kw)
        finally:
            self.num_splits = keep

    cls.forward_extend = forward_extend
    _log(f"fa3 target verify num_splits={n} in {mod.__name__}")


def _install_spec_mrope_fast(mod):
    """ForwardBatch.compute_spec_mrope_positions for text-only batches: positions repeated 3x in one op instead of
    zeros + add + repeat (mrope delta is 0 without multimodal input). Same values; py-spy (2026-10-11) had this
    method at ~6% of the MTP scheduler thread, which is CPU-bound with the draft token map."""
    cls = mod.ForwardBatch
    orig = cls.compute_spec_mrope_positions

    def compute_spec_mrope_positions(self, model_runner, batch, seq_positions=None):
        if all(m is None for m in batch.multimodal_inputs):
            if seq_positions is None:
                seq_positions = batch.spec_info.positions
            pos = seq_positions.reshape(1, -1)
            # text-only: the 3 M-RoPE rows are identical. CORAL_SPEC_MROPE_VIEW=1 -> broadcast view, no kernel (measured
            # neutral, 732.9 vs 733.2 tok/s); default keeps a real copy in case a consumer writes in place.
            self.mrope_positions = pos.expand(3, -1) if os.environ.get("CORAL_SPEC_MROPE_VIEW") == "1" else pos.repeat(3, 1)
            return
        return orig(self, model_runner, batch, seq_positions)

    cls.compute_spec_mrope_positions = compute_spec_mrope_positions
    _log(f"spec mrope fast path patched {cls.__module__}.{cls.__name__}")


def _install_norm_router_link(mod):
    """On a decoder layer's first forward, link its post-attention norm to its MoE router weight (and back), so the
    norm can compute the router logits in the same launch (CORAL_NORM_ROUTER)."""
    import torch
    for name in ("Qwen3_5LinearDecoderLayer", "Qwen3_5AttentionDecoderLayer"):
        cls = getattr(mod, name, None)
        if cls is None:
            continue

        def make(orig):
            def forward(self, *a, **k):
                if not getattr(self, "_coral_nr_linked", False):
                    self._coral_nr_linked = True
                    mlp = getattr(self, "mlp", None)
                    w = getattr(getattr(mlp, "gate", None), "weight", None)
                    norm = getattr(self, "post_attention_layernorm", None)
                    if (w is not None and norm is not None and hasattr(mlp, "experts") and w.dtype == torch.bfloat16
                            and tuple(w.shape) == (256, 2048)):
                        norm._coral_router = w.data.contiguous()
                        mlp._coral_norm = norm
                return orig(self, *a, **k)
            return forward

        cls.forward = make(cls.forward)
        _log(f"norm-router link patched {cls.__module__}.{name}.forward")


def _install_norm_router(mod):
    """GemmaRMSNorm with a linked router: residual add + norm + router GEMV in one launch (coral_kernels.add_norm_router,
    bit-identical outputs). Logits are stashed on the norm, keyed by the output tensor; the MoE forward picks them up
    instead of launching the router. Decode sizes only (M <= _NORM_ROUTER_MAX_M). From CORAL_NORM_ROUTER_MT_MIN_M
    tokens up, the multi-token kernel (add_norm_router_mt) replaces the serial one."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from coral_kernels import add_norm_router, add_norm_router_mt
    cls = mod.GemmaRMSNorm
    inner = cls._forward_impl

    def _forward_impl(self, x, residual=None, post_residual_addition=None):
        wr = getattr(self, "_coral_router", None)
        if (wr is not None and residual is not None and post_residual_addition is None and x.dim() == 2
                and 0 < x.shape[0] <= _NORM_ROUTER_MAX_M and x.shape[1] == 2048 and x.dtype == torch.bfloat16 and x.is_contiguous()
                and residual.is_contiguous()):
            M = x.shape[0]
            if 0 < _NORM_ROUTER_MT_MIN_M <= M:
                bn, kc, nw = _NORM_ROUTER_MT_CFG[8 if M <= 8 else 16]
                y, h, lg = add_norm_router_mt(x, residual, self.weight.data, self.variance_epsilon, wr, bn, kc, nw)
            else:
                bn, nw = (4, 8) if M == 1 else (2, 4)
                y, h, lg = add_norm_router(x, residual, self.weight.data, self.variance_epsilon, wr, bn, nw)
            self._coral_lg = (y.data_ptr(), tuple(y.shape), y._version, lg)
            return y, h
        return inner(self, x, residual, post_residual_addition)

    cls._forward_impl = _forward_impl
    mt = f"M>={_NORM_ROUTER_MT_MIN_M}" if _NORM_ROUTER_MT_MIN_M > 0 else "off"
    _log(f"norm-router patched {cls.__module__}.{cls.__name__}._forward_impl (cap {_NORM_ROUTER_MAX_M}, mt {mt})")


_CONV_DONE = {}  # data_ptr -> _version of qkvz outputs whose q|k|v already went through the conv update


def _install_gdn_conv_fold(mod):
    """GDN decode (M=1): hand in_proj_qkvz the layer's conv state / cache slot / conv weights for this call, so the
    FP8 GEMV dispatch runs fp8w_gemv_ns_conv (conv folded into the GEMV epilogue, one launch less per GDN layer).
    The backend's own causal_conv1d_update is then skipped for that tensor (_install_gdn_conv_skip)."""
    # v0.5.21 defaults SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV on: decode hands (qkvz, ba) to the backend's fused
    # unpack+conv (gdn_backend.fused_qkvzba_causal_conv1d_update_contiguous), which never calls causal_conv1d_update,
    # so the skip below could not stop a second conv + state update. The flag is read as a module global per call.
    mod._gdn_decode_fused_proj_conv = False
    cls = mod.Qwen3_5GatedDeltaNet
    orig_fwd = cls.forward

    said = set()

    def why(r):
        if r not in said:
            said.add(r)
            _log(f"gdn conv-fold declined: {r}")
        return None

    def ctx_for(self, hidden_states, forward_batch):
        if not (hidden_states.dim() == 2 and hidden_states.shape[0] == 1 and forward_batch.forward_mode.is_decode()):
            return None
        from sglang.srt.model_executor.forward_context import get_attn_backend
        be = get_attn_backend()
        lb = getattr(be, "linear_attn_backend", None)
        meta = getattr(lb, "forward_metadata", None)
        ci = getattr(meta, "mamba_cache_indices", None)
        if ci is None or getattr(meta, "replayssm_write_pos", None) is not None:
            return why(f"no cache indices (backend {type(be).__name__}, lb {type(lb).__name__})")
        attn = self.attn
        cw = getattr(attn, "conv_weights", None)
        if cw is None or cw.numel() != 8192 * 4 or getattr(attn, "activation", None) not in ("silu", "swish"):
            return why(f"conv weights {None if cw is None else tuple(cw.shape)} act {getattr(attn, 'activation', None)}")
        cs = lb.req_to_token_pool.mamba2_layer_cache(attn.layer_id).conv[0]
        if cs.dim() != 3 or cs.shape[-1] != 3 or cs.shape[-2] < 8192:
            return why(f"conv state shape {tuple(cs.shape)} stride {cs.stride()}")
        return (cs, ci, cw.reshape(8192, 4), getattr(attn, "bias", None))

    def forward(self, hidden_states, forward_batch, *a, **k):
        proj = self.in_proj_qkvz
        proj._coral_conv_ctx = ctx_for(self, hidden_states, forward_batch)
        try:
            return orig_fwd(self, hidden_states, forward_batch, *a, **k)
        finally:
            proj._coral_conv_ctx = None

    cls.forward = forward
    _log(f"gdn conv-fold patched {cls.__module__}.{cls.__name__}.forward")


def _install_gdn_conv_skip(mod):
    orig = mod.causal_conv1d_update

    def causal_conv1d_update(x, *a, **k):
        v = _CONV_DONE.pop(x.data_ptr(), None)
        if v is not None and v == x._version:
            if not _CONV_DONE.get("_said_skip"):
                _CONV_DONE["_said_skip"] = 1
                _log("gdn conv update skipped (already folded)")
            return x  # q|k|v already convolved (and the conv state updated) in the qkvz GEMV epilogue
        return orig(x, *a, **k)

    mod.causal_conv1d_update = causal_conv1d_update
    _log(f"gdn conv skip-if-folded patched {mod.__name__}.causal_conv1d_update")


def _install_gdn_verify_views(mod):
    """GDN extend/verify path: fused_qkv_split_gdn_prefill copies post-conv [T, q|k|v] into contiguous q, k, v. For
    T <= 8 (MTP target verify) return strided views instead: the verify kernel reads them bit-identically
    (gdn_verify_views_test.py: o + intermediate states equal). Prefill (T > 8) keeps the copies."""
    if hasattr(getattr(mod, "GDNAttnBackend", None), "_init_target_verify_qkv_routing"):
        # v0.5.21: upstream already hands the Triton verify kernel strided views (_use_strided_target_verify_qkv,
        # gdn_backend.py:582-608, 1017-1031) and skips this split; what the hook would still catch are the cases
        # upstream copies on purpose (small non-verify extends feeding FlashInfer prefill on SM90). Leave it off.
        _log("gdn verify views: not installed (v0.5.21 does it upstream)")
        return
    orig = mod.fused_qkv_split_gdn_prefill

    def split(mixed_qkv, num_q_heads, num_k_heads, num_v_heads, head_q, head_k, head_v):
        T = mixed_qkv.shape[0]
        if mixed_qkv.dim() != 2 or T > 8 or mixed_qkv.stride(-1) != 1:
            return orig(mixed_qkv, num_q_heads, num_k_heads, num_v_heads, head_q, head_k, head_v)
        qd, kd, vd = num_q_heads * head_q, num_k_heads * head_k, num_v_heads * head_v
        q = mixed_qkv[:, :qd].view(1, T, num_q_heads, head_q)
        k = mixed_qkv[:, qd:qd + kd].view(1, T, num_k_heads, head_k)
        v = mixed_qkv[:, qd + kd:qd + kd + vd].view(1, T, num_v_heads, head_v)
        return q, k, v

    mod.fused_qkv_split_gdn_prefill = split
    _log(f"gdn verify split -> views in {mod.__name__}")


def _install_gdn_zview(mod):
    """GDN forward for 2..8 tokens (MTP verify) without the z copy: SGLang reshapes z [B, HV, V] (a strided slice of
    qkvz) to [B*HV, V], which copies; here the gated RMSNorm runs on [B, HV*V] rows with group_size=V, a tiled weight
    and z as a strided view (gated_norm_group_test.py: bit-identical). Same steps as the original forward otherwise
    (input proj -> split views -> linear attention -> gated norm -> out_proj)."""
    import torch
    import triton
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm, rms_norm_gated
    cls = mod.Qwen3_5GatedDeltaNet
    orig = cls.forward
    ratios = getattr(mod, "_GDN_FUSED_QKVZBA_RATIOS", ())

    def forward(self, hidden_states, forward_batch, *a, **k):
        B = hidden_states.shape[0] if hidden_states.dim() == 2 else 0
        norm = self.norm
        if (a or k or not (1 < B <= _ZVIEW_MAX_T) or self.num_v_heads // self.num_k_heads not in ratios
                or (forward_batch.forward_mode.is_decode() and getattr(mod, "_gdn_decode_fused_proj_conv", False))
                or type(norm) is not RMSNorm or norm.group_size is not None or norm.bias is not None):
            return orig(self, hidden_states, forward_batch, *a, **k)
        nk = triton.cdiv(self.num_k_heads, self.attn_tp_size)
        nv = triton.cdiv(self.num_v_heads, self.attn_tp_size)
        qkv_dim = 2 * nk * self.head_k_dim + nv * self.head_v_dim
        vd = nv * self.head_v_dim
        qkvz, ba = self._forward_input_proj(hidden_states)
        mixed_qkv = qkvz[:, :qkv_dim]
        z = qkvz[:, qkv_dim:qkv_dim + vd]
        b = ba[:, :nv]
        a_ = ba[:, nv:2 * nv]
        core = self.attn(forward_batch, mixed_qkv=mixed_qkv, a=a_, b=b).reshape(B, vd)
        w = getattr(self, "_coral_norm_wtile", None)
        if w is None or w.shape[0] != vd:
            w = self._coral_norm_wtile = norm.weight.data.repeat(nv).contiguous()
        y = rms_norm_gated(x=core, weight=w, bias=None, z=z, eps=norm.eps, group_size=self.head_v_dim,
                           norm_before_gate=norm.norm_before_gate, is_rms_norm=True, activation=norm.activation)
        out, _ = self.out_proj(y)
        return out

    cls.forward = forward
    _log(f"gdn z-view forward (2..8 tokens) patched {cls.__module__}.{cls.__name__}.forward")


def _install_fa3_draft_extend_graph(mod):
    """Spec v2 only captures a draft-extend CUDA graph for an allow-list of attention backends that omits FA3 (the
    default here), so the MTP draft-extend forward runs eagerly every step (~13% of the CPU-bound scheduler thread).
    FA3's backend implements draft_extend_v2 in its graph capture/replay paths; let the allow-list's isinstance check
    accept it. eagle_worker_v2 references TritonAttnBackend only in that list."""
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    real = mod.TritonAttnBackend

    class _Meta(type):
        def __instancecheck__(cls, inst):
            return isinstance(inst, (real, FlashAttentionBackend))

    class TritonOrFA3(metaclass=_Meta):
        pass

    mod.TritonAttnBackend = TritonOrFA3
    _log(f"draft-extend CUDA graph allowed for FA3 in {mod.__name__}")


def _install_dbg_cpu_sleep(mod):
    """DIAGNOSTIC ONLY (CORAL_DBG_CPU_SLEEP_US): add a CPU sleep to every Scheduler.run_batch, to measure how much of
    an added per-step CPU cost reaches throughput (1.0 = CPU-bound, 0 = GPU-bound)."""
    import time
    us = float(os.environ["CORAL_DBG_CPU_SLEEP_US"]) / 1e6
    cls = mod.Scheduler
    orig = cls.run_batch

    def run_batch(self, *a, **k):
        time.sleep(us)
        return orig(self, *a, **k)

    cls.run_batch = run_batch
    _log(f"DIAGNOSTIC cpu sleep {us * 1e6:.0f} us per run_batch")


def _install_conv_bn(mod):
    """SGLang's causal_conv1d_update launches its Triton kernel with BLOCK_N=256: 8192 GDN channels -> 32 CTAs for 132
    SMs (latency-bound, ~4 us in the MTP verify). Relaunch with BLOCK_N=CORAL_CONV_BN: channels are independent, so the
    result is identical; only the grid changes."""
    bn = int(os.environ["CORAL_CONV_BN"])
    k = mod._causal_conv1d_update_kernel

    class _BN:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                kw["BLOCK_N"] = bn
                return k[grid](*a, **kw)
            return launch

    mod._causal_conv1d_update_kernel = _BN()
    _log(f"causal_conv1d_update relaunched BLOCK_N={bn} in {mod.__name__}")


def _install_fast_draft_chain(mod):
    """MTP with a draft token map: SGLang only uses its fused topk=1 draft chain (draft_topk1_postprocess writing the
    chain buffer) when hot_token_id is None, so with a map every draft step takes the generic path (select_top_k_tokens,
    gathers, cats: ~150 us/step of small kernels). Here the worker runs with hot_token_id = None (fused chain captured
    into the draft graph) and the two top-k helpers map draft-vocab indices to full-vocab ids themselves whenever they
    reduce over draft-head logits (width == map size). The draft-extend graph runner still sees the map at its own init
    (it sizes its logits buffer from it). Same tokens; fewer kernels."""
    import torch
    H = {"hot": None, "n": -1}
    o_fast, o_post = mod.fast_topk, mod.draft_topk1_postprocess

    def fast_topk(values, topk, dim=-1):
        p, i = o_fast(values, topk, dim)
        if H["hot"] is not None and values.shape[-1] == H["n"]:
            i = H["hot"][i]
        return p, i

    def draft_topk1_postprocess(logits, positions, draft_tokens=None, draft_token_column=0):
        if H["hot"] is not None and logits.shape[-1] == H["n"]:
            p, i = o_post(logits, positions, None, draft_token_column)
            i = H["hot"][i]
            if draft_tokens is not None:
                draft_tokens[:, draft_token_column:draft_token_column + 1].copy_(i)
            return p, i
        return o_post(logits, positions, draft_tokens, draft_token_column)

    mod.fast_topk = fast_topk
    mod.draft_topk1_postprocess = draft_topk1_postprocess

    ext = mod.EAGLEDraftExtendCudaGraphRunner

    class ExtRunner(ext):
        def __init__(self, eagle_worker, *a, **k):
            if H["hot"] is not None:
                eagle_worker.hot_token_id = H["hot"]
            try:
                super().__init__(eagle_worker, *a, **k)
            finally:
                if H["hot"] is not None:
                    eagle_worker.hot_token_id = None

    mod.EAGLEDraftExtendCudaGraphRunner = ExtRunner
    cls = mod.EagleDraftWorker
    orig = cls.init_cuda_graphs

    def init_cuda_graphs(self, *a, **k):
        if self.hot_token_id is not None:
            H["hot"], H["n"] = self.hot_token_id, int(self.hot_token_id.shape[0])
            self.hot_token_id = None
            _log(f"fast draft chain: draft vocab {H['n']} mapped inside the top-k helpers")
        return orig(self, *a, **k)

    cls.init_cuda_graphs = init_cuda_graphs

    # draft extend's topk=1 CUDA path takes torch.argmax over the draft-head logits (draft-vocab index); stock code
    # maps it at the start of draft_forward, which no longer happens with hot_token_id = None -> map it here.
    o_ext = cls._draft_extend_for_decode

    def _draft_extend_for_decode(self, batch, batch_result, *a, **k):
        r = o_ext(self, batch, batch_result, *a, **k)
        nd = getattr(batch_result, "next_draft_input", None)
        ti = getattr(nd, "topk_index", None)
        if H["hot"] is not None and ti is not None and getattr(nd, "_coral_mapped", None) is not ti:
            nd.topk_index = H["hot"][ti] if self.topk == 1 else ti
            nd._coral_mapped = nd.topk_index
        return r

    cls._draft_extend_for_decode = _draft_extend_for_decode


INSTALLERS = {}
if os.environ.get("CORAL_MOE") == "1":
    INSTALLERS.setdefault(QMOE_MOD, []).append(_install_moe)
if os.environ.get("CORAL_MOE_FP4") == "1":
    assert os.environ.get("CORAL_MOE") != "1", "CORAL_MOE (FP8) and CORAL_MOE_FP4 both patch the MoE block forward"
    INSTALLERS.setdefault(MODELOPT_MOD, []).append(_install_fp4_stash)
    INSTALLERS.setdefault(QMOE_MOD, []).append(_install_moe_fp4)
if os.environ.get("CORAL_GDN_SINGLE_STREAM") == "1":
    INSTALLERS.setdefault(Q35_MOD, []).append(_install_gdn_single_stream)
if os.environ.get("CORAL_SINGLE_STREAM_MOE") == "1":
    INSTALLERS.setdefault(QMOE_MOD, []).append(_install_single_stream)
if os.environ.get("CORAL_GEMV") == "1":
    INSTALLERS.setdefault(UNQ_MOD, []).append(_install_gemv)
if os.environ.get("CORAL_MONOMOE") == "1":
    INSTALLERS.setdefault(FP8_MOD, []).append(_install_monomoe)
if os.environ.get("CORAL_NORMQUANT") == "1":
    INSTALLERS.setdefault(FP8_MOD, []).append(_install_normquant_linear)
    INSTALLERS.setdefault(NORM_MOD, []).append(_install_normquant_norm)

if os.environ.get("CORAL_GDN_SPLIT_VIEWS") == "1":
    INSTALLERS.setdefault(Q35_MOD, []).append(_install_gdn_split_views)
if os.environ.get("CORAL_GDN_VERIFY_BV"):
    INSTALLERS.setdefault(GDNV_MOD, []).append(_install_gdn_verify_bv)
if os.environ.get("CORAL_GDN_BV"):
    INSTALLERS.setdefault(GDNK_MOD, []).append(_install_gdn_bv)
if os.environ.get("CORAL_FP8GEMV") == "1":
    INSTALLERS.setdefault(FP8_MOD, []).append(_install_fp8gemv)  # appended last -> wraps outermost
if os.environ.get("CORAL_NORM_ROUTER") == "1":
    INSTALLERS.setdefault(Q35_MOD, []).append(_install_norm_router_link)
    INSTALLERS.setdefault(NORM_MOD, []).append(_install_norm_router)  # after normquant -> wraps it
if os.environ.get("CORAL_GDN_CONV_FOLD") == "1":
    INSTALLERS.setdefault(Q35_MOD, []).append(_install_gdn_conv_fold)
    INSTALLERS.setdefault("sglang.srt.layers.attention.linear.gdn_backend", []).append(_install_gdn_conv_skip)
if os.environ.get("CORAL_GDN_VERIFY_VIEWS") == "1":
    INSTALLERS.setdefault("sglang.srt.layers.attention.linear.gdn_backend", []).append(_install_gdn_verify_views)
if os.environ.get("CORAL_GDN_ZVIEW") == "1":
    INSTALLERS.setdefault(Q35_MOD, []).append(_install_gdn_zview)
if os.environ.get("CORAL_FA3_DRAFT_EXTEND_GRAPH") == "1":
    INSTALLERS.setdefault("sglang.srt.speculative.eagle_worker_v2", []).append(_install_fa3_draft_extend_graph)
if os.environ.get("CORAL_DBG_CPU_SLEEP_US"):
    INSTALLERS.setdefault("sglang.srt.managers.scheduler", []).append(_install_dbg_cpu_sleep)
if os.environ.get("CORAL_CONV_BN"):
    INSTALLERS.setdefault("sglang.kernels.ops.mamba.causal_conv1d_triton", []).append(_install_conv_bn)
def _install_draft_head_slice(mod):
    """MTP draft steps 2..N (the draft_forward loop, captured into the draft CUDA graph) score only the first
    CORAL_DRAFT_HEAD_SLICE rows of the draft LM head; draft-extend (draft step 1) keeps the full token-map head.
    Requires a FREQUENCY-RANKED token map (e.g. /data/tokmap/rank49152.pt) so the prefix is the most frequent tokens.
    Outputs stay exact (verify uses the full target head); only acceptance can move. Prefix = same storage, so it is a
    view: the captured matmul just has fewer rows."""
    n = int(os.environ["CORAL_DRAFT_HEAD_SLICE"])
    cls = mod.EagleDraftWorker
    o_df = cls.draft_forward

    def draft_forward(self, forward_batch):
        w = self.draft_runner.model.lm_head.weight
        full = w.data
        if full.shape[0] <= n:
            return o_df(self, forward_batch)
        w.data = full[:n]
        try:
            return o_df(self, forward_batch)
        finally:
            w.data = full

    cls.draft_forward = draft_forward
    _log(f"draft head slice: draft steps 2+ score the top {n} token-map rows")


if os.environ.get("CORAL_SPEC_FUSE") == "1":
    def _install_spec_fuse_commit(mod):
        import coral_spec_fuse
        coral_spec_fuse.install_commit(mod, _log)

    def _install_spec_fuse_de(mod):
        import coral_spec_fuse
        coral_spec_fuse.install_draft_extend(mod, _log)

    INSTALLERS.setdefault("sglang.srt.speculative.spec_utils", []).append(_install_spec_fuse_commit)
    INSTALLERS.setdefault("sglang.srt.speculative.eagle_worker_v2", []).append(_install_spec_fuse_de)

if os.environ.get("CORAL_DFLASH_LOG"):
    def _install_dflash_log(mod):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import coral_dflash_log
        coral_dflash_log.install(mod, _log)

    INSTALLERS.setdefault("sglang.srt.speculative.dflash_worker_v2", []).append(_install_dflash_log)

# DFlash2 selector candidate head (coral_selector.py): reduced vocab and/or FP8 copy of the target lm_head for the
# selector's top-k only (the verify head is untouched, so output text cannot change; only tau can).
if os.environ.get("CORAL_SELECTOR_VOCAB") or os.environ.get("CORAL_SELECTOR_FP8"):
    def _install_selector(mod):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import coral_selector
        coral_selector.install(mod, _log)

    INSTALLERS.setdefault("sglang.srt.models.dflash", []).append(_install_selector)

# DFlash2 draft-side levers (coral_draft.py): fused post-verify mamba bookkeeping, rows_gemv for the draft's small
# nn.Linears, FP8 draft linears with the fused context-KV path kept, FP8 fc, two-pass top-k, FP8 candidate head.
_CORAL_DRAFT_ENVS = ("CORAL_DRAFT_FP8_HEAD", "CORAL_DRAFT_FP8_FC", "CORAL_DRAFT_FP8_KV_FUSED", "CORAL_DRAFT_GEMV",
                     "CORAL_DRAFT_TOPK2", "CORAL_DFLASH_MAMBA_FUSE", "CORAL_DRAFT_COMPILE")
# Both replace the selector's candidate projection; the 48K subset measured 102 us vs ~200 for an FP8 full head.
assert not (os.environ.get("CORAL_DRAFT_FP8_HEAD", "") not in ("", "0") and os.environ.get("CORAL_SELECTOR_VOCAB")), \
    "CORAL_DRAFT_FP8_HEAD and CORAL_SELECTOR_VOCAB both replace the selector head: set one (CORAL_SELECTOR_FP8 for both)"
if any(os.environ.get(k, "") not in ("", "0") for k in _CORAL_DRAFT_ENVS):
    def _install_coral_draft(mod):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import coral_draft
        coral_draft.install(mod, _log)

    INSTALLERS.setdefault("sglang.srt.models.dflash", []).append(_install_coral_draft)
    INSTALLERS.setdefault("sglang.srt.speculative.dflash_worker_v2", []).append(_install_coral_draft)

# DFlash2 adaptive target verify (coral_dflash_adaptive.py): selector path confidence picks one of
# CORAL_ADAPTIVE_VERIFY_TIERS captured target graphs while the draft stays at the full block.
if os.environ.get("CORAL_DFLASH_ADAPTIVE") == "1":
    def _install_dflash_adaptive(mod):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import coral_dflash_adaptive
        coral_dflash_adaptive.install(mod, _log)

    INSTALLERS.setdefault("sglang.srt.speculative.dflash_worker_v2", []).append(_install_dflash_adaptive)

# DFlash2 small selector tree: primary chain plus one rank-2 rescue node per position.
if os.environ.get("CORAL_DFLASH_TREE") == "1":
    def _install_dflash_tree(mod):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import coral_dflash_tree
        coral_dflash_tree.install(mod, _log)

    INSTALLERS.setdefault("sglang.srt.speculative.dflash_worker_v2", []).append(_install_dflash_tree)

if os.environ.get("CORAL_DRAFT_HEAD_SLICE"):
    INSTALLERS.setdefault("sglang.srt.speculative.eagle_worker_v2", []).append(_install_draft_head_slice)

if os.environ.get("CORAL_FAST_DRAFT_CHAIN") == "1":
    INSTALLERS.setdefault("sglang.srt.speculative.eagle_worker_v2", []).append(_install_fast_draft_chain)
if os.environ.get("CORAL_SPEC_MROPE_FAST") == "1":
    INSTALLERS.setdefault("sglang.srt.model_executor.forward_batch_info", []).append(_install_spec_mrope_fast)
if os.environ.get("CORAL_FA_VERIFY_SPLITS"):
    INSTALLERS.setdefault("sglang.srt.layers.attention.flashattention_backend", []).append(_install_fa_verify_splits)


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in INSTALLERS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            for fn in INSTALLERS[name]:
                fn(module)

        spec.loader.exec_module = exec_module
        return spec


if INSTALLERS:
    sys.meta_path.insert(0, _Hook())
