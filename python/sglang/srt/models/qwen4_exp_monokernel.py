"""Qwen4-Exp (Qwen3.8-Flash-Next) speculative decoding through the FlyDSL monokernel.

Target verify of R <= 8 requests x num_draft_tokens rows (topk 1) runs the whole model (embedding, PLE, 48 decoder
layers, final mixer) in 13 persistent launches; the MTP draft (draft extend / decode) runs in one launch. The kernel
reads sglang's own weights in place: the MXFP4 experts (FusedMoE w13 / w2, AITER layout, shared expert fused) and the
FP8 attention / GDN projections (Quark W8A8, AITER-shuffled). Param data_ptrs are stable after loading, so the
launches are CUDA-graph safe. Requires SGLANG_USE_AITER=1 (gfx950).
"""

import os
from typing import List, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import logger

MAX_REQS = 8


def monokernel_enabled() -> bool:
    return envs.SGLANG_ENABLE_QWEN4_MONOKERNEL.get()


def mtp_monokernel_enabled() -> bool:
    return envs.SGLANG_ENABLE_QWEN4_MTP_MONOKERNEL.get()


def _check_parallel() -> None:
    from sglang.srt.runtime_context import get_parallel

    par = get_parallel()
    if (par.tp_size, par.pp_size, par.attn_cp_size, par.moe_ep_size) != (1, 1, 1, 1):
        raise ValueError(
            "the Qwen4-Exp monokernel runs the whole model on one GPU: TP, PP, CP and EP must be 1 "
            f"(got tp={par.tp_size} pp={par.pp_size} cp={par.attn_cp_size} ep={par.moe_ep_size})"
        )


def _import_kernel():
    from sglang.kernels.ops.speculative.qwen4_exp_monokernel import (
        kernel,
        layout,
        pack,
        weights,
    )

    return kernel, layout, pack, weights


def _checkpoint_dir(draft: bool = False) -> str:
    """Local checkpoint directory of the served (or draft) model: the runners read the few tensors sglang does not keep
    in the kernel's layout (hyper-connection mixers, router, PLE) straight from its safetensors."""
    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    path = (args.speculative_draft_model_path if draft else None) or args.model_path
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        # Only the safetensors are read: do not require the rest of the snapshot (README, tokenizer files, ...).
        path = snapshot_download(
            path,
            revision=args.revision,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    return path


def _moe_params(experts, pack, weights, prefix, check_ids=(), dev="cuda"):
    """sglang FusedMoE params (AITER MXFP4 layout) -> the kernel's moe dict, with layout asserts; check_ids: experts
    whose bytes are compared with the checkpoint shuffled the same way (an AITER layout change fails loudly)."""
    nshared = experts.num_fused_shared_experts
    assert nshared == 1, (
        f"{prefix}: expected the shared expert fused into the FusedMoE (got {nshared})"
    )
    w13, w2 = experts.w13_weight, experts.w2_weight
    for w in (
        w13,
        w2,
    ):  # the AITER layout itself is checked byte-exact on the check_ids experts
        assert w.dtype == torch.uint8, (prefix, w.dtype)
    m = {
        "gu": w13.data,
        "gu_s": experts.w13_weight_scale.data.view(torch.uint8),
        "dn": w2.data,
        "dn_s": experts.w2_weight_scale.data.view(torch.uint8),
    }
    pack.check_moe_params(m)
    if check_ids:
        names = []
        for e in check_ids:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for w in ("weight", "weight_scale"):
                    src = (
                        "mlp.shared_expert"
                        if e == len(m["gu"]) - 1
                        else f"mlp.experts.{e}"
                    )
                    names.append(f"{prefix}{src}.{proj}.{w}")
        raw = weights.load(names, dev)
        for e in check_ids:
            src = "mlp.shared_expert" if e == len(m["gu"]) - 1 else f"mlp.experts.{e}"
            got = pack.unshuffle_moe_expert(
                m, e
            )  # real rows / K / groups (the AITER padding is not compared)
            ref = [
                raw[f"{prefix}{src}.{proj}.{w}"].view(torch.uint8)
                for w in ("weight", "weight_scale")
                for proj in ("gate_proj", "up_proj", "down_proj")
            ]
            for nm, g_, r_ in zip(
                ("gate", "up", "down", "gate_s", "up_s", "down_s"), got, ref
            ):
                assert torch.equal(g_, r_), (
                    f"{prefix} expert {e} {nm}: sglang FusedMoE bytes differ from the checkpoint in the AITER layout"
                )
    return m


# kernel name -> (sglang param of the decoder layer, rows, K, checkpoint tensors it merges)
# kernel name -> (sglang param of the decoder layer, kernel rows, K, checkpoint tensors it merges)
_GDN_PROJ = {
    "ip": (
        "linear_attn.in_proj_qkvz",
        16384,
        2560,
        ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    ),
    "out": ("linear_attn.out_proj", 2560, 6144, ("linear_attn.out_proj",)),
    "ba": (
        "linear_attn.in_proj_ba",
        128,
        2560,
        ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
    ),  # 96 real rows
}
# (the QSA decoder layer holds qkv_proj / o_proj itself and the QSA indexer as .indexer)
_QSA_PROJ = {
    "qp": (
        "qkv_proj",
        13312,
        2560,
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ),
    "qp2": ("indexer.index_qk_proj", 640, 2560, ("self_attn.indexer.index_qk_proj",)),
    "out": ("o_proj", 2560, 6144, ("self_attn.o_proj",)),
}


def _proj_params(layer, table, pack, weights, prefix, check=False, dev="cuda"):
    """sglang's Quark W8A8 FP8 per-channel params of a decoder layer -> {kernel name: (weight [rows, K] in the
    AITER shuffle_weight (16, 16) layout, F32 scale [rows])}. Layers sglang keeps AITER-shuffled are read in place;
    the narrow ones it keeps row-major (N not a multiple of 64, use_aiter_bpreshuffle_gemm) get a zero-padded,
    shuffled copy. check: byte-exact comparison of the un-shuffled weights / scales with the checkpoint."""
    from aiter.ops.shuffle import shuffle_weight

    from sglang.srt.layers.quantization.fp8_utils import use_aiter_bpreshuffle_gemm

    params = dict(layer.named_parameters())
    out = {}
    for key, (name, rows, k, srcs) in table.items():
        assert name + ".weight" in params, (
            prefix,
            name,
            sorted(n for n in params if "proj" in n),
        )
        w, sc = params[name + ".weight"].data, params[name + ".weight_scale"].data
        assert w.dtype == torch.float8_e4m3fn, (prefix, name, w.dtype)
        assert sc.dtype == torch.float32 and sc.is_contiguous(), (
            prefix,
            name,
            sc.dtype,
        )
        n = sc.numel()
        wt = (
            w.t() if tuple(w.shape) == (k, n) else w
        )  # sglang stores the [N, K] weight behind a transposed view
        assert tuple(wt.shape) == (n, k) and wt.is_contiguous(), (
            prefix,
            name,
            tuple(w.shape),
            w.stride(),
        )
        if n == rows and use_aiter_bpreshuffle_gemm(n):
            kw, ks = wt, sc.reshape(-1)
        else:
            assert n <= rows and not use_aiter_bpreshuffle_gemm(n), (
                prefix,
                name,
                n,
                rows,
            )
            kw = torch.zeros(rows, k, dtype=wt.dtype, device=wt.device)
            kw[:n] = wt
            kw = shuffle_weight(kw, (16, 16))
            ks = torch.ones(rows, dtype=torch.float32, device=sc.device)
            ks[:n] = sc.reshape(-1)
        out[key] = (kw, ks)
        if check:
            raw = weights.load(
                [f"{prefix}{x}.{y}" for x in srcs for y in ("weight", "weight_scale")],
                dev,
            )
            ref_w = torch.cat([raw[f"{prefix}{x}.weight"] for x in srcs]).view(
                torch.uint8
            )
            ref_s = torch.cat(
                [raw[f"{prefix}{x}.weight_scale"].float().reshape(-1) for x in srcs]
            )
            got_w = pack.unshuffle16(kw)
            assert ref_w.shape[0] == n
            assert torch.equal(got_w[:n], ref_w) and torch.equal(ks[:n], ref_s), (
                f"{prefix}{name}: sglang FP8 weight / scale differ from the checkpoint in the AITER layout"
            )
            assert n == rows or not got_w[n:].any(), (
                f"{prefix}{name}: padding rows not zero"
            )
    return out


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors(v)


class Qwen4ExpMonoRunner:
    """Target verify of R <= 8 requests x L = num_draft_tokens rows (topk 1)."""

    def __init__(self, model, config, ntok: int) -> None:
        _check_parallel()
        kernel, layout, pack, weights = _import_kernel()
        self._kernel, self._layout, self._pack, self._weights = (
            kernel,
            layout,
            pack,
            weights,
        )
        self.L = int(ntok)
        assert 1 <= self.L <= 4, (
            "the monokernel is built for 1..4 verify tokens per request"
        )
        self.max_reqs = min(MAX_REQS, layout.MAX_T // self.L)
        from sglang.srt.runtime_context import get_model

        text = config.get_text_config()
        ctx = get_model().context_length or text.max_position_embeddings
        assert ctx + 64 <= layout.MAX_CTX, (
            f"context length {ctx} exceeds the monokernel's MAX_CTX {layout.MAX_CTX}"
        )
        kinds = ["qsa" if t == "full_attention" else "gdn" for t in text.layer_types]
        n = len(kinds)
        if n % 4 or any(kinds[i] != ("qsa" if i % 4 == 3 else "gdn") for i in range(n)):
            raise ValueError(
                "monokernel expects a repeating 3x GDN + 1x QSA layer pattern"
            )
        if list(text.ple_layer_ids) != [2]:
            raise ValueError(
                "monokernel expects PLE only on layer 1 (ple_layer_ids == [2])"
            )
        self.kinds = kinds
        self.plans: List[Tuple[tuple, List[int]]] = [
            (("gdn",), [0]),
            (("gdn", "gdn", "qsa"), [1, 2, 3]),
        ]
        self.ple_launch = 1
        for b in range(1, n // 4):
            self.plans.append(
                (("gdn", "gdn", "gdn", "qsa"), list(range(4 * b, 4 * b + 4)))
            )
        weights.MODEL_DIR = _checkpoint_dir()
        self.packed = {}
        dev = torch.cuda.current_device()
        self.moe = {}
        for layer_id, kind in enumerate(kinds):
            # experts: sglang's FusedMoE params in place (spot-checked against the checkpoint on 2 layers)
            moe = _moe_params(
                model.layers[layer_id].mlp.experts,
                pack,
                weights,
                f"{weights.PFX}layers.{layer_id}.",
                check_ids=(0, 255, 511, 512) if layer_id in (0, 3) else (),
                dev=f"cuda:{dev}",
            )
            self.moe[layer_id] = moe
            proj = _proj_params(
                model.layers[layer_id],
                _GDN_PROJ if kind == "gdn" else _QSA_PROJ,
                pack,
                weights,
                f"{weights.PFX}layers.{layer_id}.",
                check=layer_id in (0, 3),
                dev=f"cuda:{dev}",
            )
            lw = weights.load_layer(
                layer_id, kind, device=f"cuda:{dev}", ple=layer_id == 1
            )
            self.packed[layer_id] = (
                pack.pack_gdn_layer(lw, moe=moe, proj=proj)
                if kind == "gdn"
                else pack.pack_qsa_layer(lw, moe=moe, proj=proj)
            )
            if layer_id == 1:
                self.packed_ple = pack.pack_ple(lw)
            del lw
            torch.cuda.empty_cache()
        S = self.max_reqs * self.L
        # one scratch for every R: a launch only accepts pairs tagged with its own (step, launch) epoch
        nbytes = max(
            layout.scratch_layout(4 + 2, r * self.L, r)[1]
            for r in range(1, self.max_reqs + 1)
        )
        self.scratch = torch.zeros(nbytes // 4, dtype=torch.int32, device="cuda")
        self.step = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.h_bufs = [
            torch.empty(S, 4 * 2560, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        self.ssq_bufs = [
            torch.zeros(S, 4, layout.NPART, dtype=torch.float32, device="cuda")
            for _ in range(2)
        ]
        self.emb_buf = torch.empty(S, 2560, dtype=torch.bfloat16, device="cuda")
        self.x_buf = torch.empty(S, 2560, dtype=torch.bfloat16, device="cuda")
        ple_emb = model.layers[1].ple.ple_embedding
        self.eos_token_id = int(text.eos_token_id)
        self.ngram_meta = (
            torch.cat(
                [
                    ple_emb.layer_multipliers,
                    ple_emb.ngram_heads_vocab_sizes,
                    ple_emb.ngram_heads_offsets,
                ]
            )
            .to(device="cuda", dtype=torch.int64)
            .contiguous()
        )
        assert self.ngram_meta.numel() == 3 + 2 * layout.NGRAM_HEADS
        self.ngram_table = ple_emb.ngram_embedding.weight
        self.ngram_scale = ple_emb.ngram_embedding.weight_scale
        assert (
            self.ngram_table.dtype == torch.float8_e4m3fn
            and self.ngram_table.shape[1] == layout.NGRAM_DIM
        )
        self.embed_w = model.embed_tokens.weight
        mixer = weights.load(
            [
                f"{weights.PFX}hyper_connection_mixer.{nm}.weight"
                for nm in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
            ],
            device=f"cuda:{dev}",
        )
        fin = {k.split("hyper_connection_mixer.")[1]: v for k, v in mixer.items()}
        self.packed_final = {
            "norm": fin["hc_norm.weight"].contiguous(),
            "down": pack.pack_hc_down(
                fin["input_mix_weight_down.weight"]
            ),  # no combine: gate tile zero
            "up": pack.pack_bf16(fin["input_mix_weight_up.weight"].contiguous()),
        }
        self._tables = {}
        self._launchers = {}
        self._pinned = []
        self._diag = set()
        self._eager = 0
        self._fallbacks = 0
        # own copies only (tensors that are not sglang's params, read in place), by category
        param_ptrs = {t.data_ptr() for t in model.parameters()}
        cats = {}
        for p in self.packed.values():
            for k_, v_ in p.items():
                own = sum(
                    t.numel() * t.element_size()
                    for t in _tensors(v_)
                    if t.data_ptr() not in param_ptrs
                )
                if own:
                    cats[k_] = cats.get(k_, 0) + own
        cats["ple"] = sum(
            t.numel() * t.element_size() for t in _tensors(self.packed_ple)
        )
        packed_gb = sum(cats.values()) / 2**30
        logger.info(
            "Qwen4-Exp monokernel own copies (MiB): "
            + ", ".join(
                f"{k_} {v_ / 2**20:.0f}"
                for k_, v_ in sorted(cats.items(), key=lambda x: -x[1])
            )
        )
        logger.info(
            f"Qwen4-Exp monokernel: L={self.L}, R <= {self.max_reqs}, packed {n} layers ({packed_gb:.1f} GiB), "
            f"{len(self.plans)} launches/verify"
        )

    # ------------------------------------------------------------------ helpers
    def applicable(self, forward_batch, input_ids: torch.Tensor) -> bool:
        """Pure target-verify batch of R <= max_reqs whole requests of L = draft_token_num rows each (topk 1)."""
        fb = forward_batch
        if not fb.forward_mode.is_target_verify():
            return False
        reason = self._refusal(fb, input_ids)
        # graph replays do not reach Python: these counts cover eager (uncaptured) verify forwards only
        if not torch.cuda.is_current_stream_capturing():
            self._eager += 1
            self._fallbacks += reason is not None
            if reason is not None or self._eager % 500 == 1:
                logger.info(
                    f"Qwen4-Exp monokernel: eager verify forwards {self._eager}, fallbacks "
                    f"{self._fallbacks}"
                    + (f" (last: {reason})" if reason is not None else "")
                )
        if reason is not None:
            return False
        R = fb.batch_size
        if R not in self._diag:
            self._diag.add(R)
            pos = fb.positions
            logger.info(
                f"Qwen4-Exp monokernel first verify at R={R}: L={self.L} ids={tuple(input_ids.shape)} "
                f"pos={(tuple(pos.shape), pos.dtype)} seq_lens={fb.seq_lens.dtype} "
                f"capturing={torch.cuda.is_current_stream_capturing()}"
            )
        return True

    def _refusal(self, fb, input_ids: torch.Tensor):
        R = fb.batch_size
        if not 1 <= R <= self.max_reqs or input_ids.numel() != R * self.L:
            return (
                f"bs {R} x {input_ids.numel()} rows (L={self.L}, R <= {self.max_reqs})"
            )
        spec = fb.spec_info
        if spec is None or int(spec.draft_token_num) != self.L or spec.topk != 1:
            return "draft tokens / topk"
        if (
            fb.positions is None
            or fb.positions.dim() not in (1, 2)
            or fb.req_pool_indices.numel() != R
        ):
            return "positions / req_pool_indices"
        return None

    def _launcher(
        self,
        R,
        plan,
        rtt_stride,
        pos_stride,
        map_i64,
        ple_first,
        final_mix,
        prelude,
        strides,
        first_src,
    ):
        key = (
            R,
            plan,
            rtt_stride,
            pos_stride,
            map_i64,
            ple_first,
            final_mix,
            prelude,
            strides,
            first_src,
        )
        if key not in self._launchers:
            iconv, issm_row, ple_ist_row, ictx_row, ssm_bf16 = strides
            self._launchers[key] = self._kernel.build_qwen4_exp_monokernel(
                plan,
                R,
                self.L,
                rtt_stride,
                pos_stride,
                map_i64,
                ple_first=ple_first,
                final_mix=final_mix,
                prelude=prelude,
                eos_token_id=self.eos_token_id,
                iconv_strides=iconv,
                issm_row=issm_row,
                ple_ist_row=ple_ist_row,
                ictx_row=ictx_row,
                first_src=first_src,
                ssm_bf16=bool(ssm_bf16),
            )
        return self._launchers[key]

    def _pointer_tables(self, forward_batch, model):
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        rtp = get_req_to_token_pool()
        kvp = get_token_to_kv_pool()
        fb = forward_batch
        R = fb.batch_size
        # the kernel reads a bf16 RoPE table; on HIP sglang keeps cos_sin_cache in f32 (cached bf16 cast)
        cos = model.layers[3].rotary_emb.cos_sin_cache_as(torch.bfloat16)
        mamba_map = rtp.req_index_to_mamba_index_mapping
        key = (
            R,
            fb.input_ids.data_ptr(),
            fb.positions.data_ptr(),
            fb.out_cache_loc.data_ptr(),
            fb.req_pool_indices.data_ptr(),
            fb.seq_lens.data_ptr(),
            cos.data_ptr(),
        )
        if key in self._tables:
            return self._tables[key]
        pack, LY = self._pack, self._layout
        assert (
            fb.input_ids.dtype == torch.int64 and fb.out_cache_loc.dtype == torch.int64
        )
        assert (
            fb.positions.dtype == torch.int64
            and fb.req_pool_indices.dtype == torch.int64
        )
        ngram_ctx = rtp.ngram_pool.context
        ngram_ictx = rtp.ngram_pool.intermediate_context
        assert (
            ngram_ctx.dtype == torch.int64
            and ngram_ctx.is_contiguous()
            and ngram_ctx.shape[1] == 2
        )
        assert (
            ngram_ictx is not None
            and ngram_ictx.dtype == torch.int64
            and ngram_ictx.shape[1] >= self.L
        )
        assert (
            ngram_ictx.stride(1) == 2
            and ngram_ictx.stride(2) == 1
            and ngram_ictx.shape[0] >= R
        )
        ictx_row = ngram_ictx.stride(0)

        def grow_for(launch_idx):
            g = pack.global_ptr_row(
                fb.positions,
                fb.out_cache_loc,
                fb.req_pool_indices,
                fb.seq_lens,
                rtp.req_to_token,
                cos,
                kvp.qsa_rope_position_buffer,
                mamba_map,
            )
            g[LY.G_PLE_EMB] = self.emb_buf.data_ptr()
            g[LY.G_FIN_OUT] = self.x_buf.data_ptr()
            g[LY.G_INPUT_IDS] = fb.input_ids.data_ptr()
            g[LY.G_EMBED_W] = self.embed_w.data_ptr()
            g[LY.G_NGRAM_W] = self.ngram_table.data_ptr()
            g[LY.G_NGRAM_SCALE] = self.ngram_scale.data_ptr()
            g[LY.G_NGRAM_CTX] = ngram_ctx.data_ptr()
            g[LY.G_NGRAM_META] = self.ngram_meta.data_ptr()
            g[LY.G_NGRAM_ICTX] = ngram_ictx[0].data_ptr()
            # residual stream of launch i: h_bufs[(i - 1) % 2] -> h_bufs[i % 2] (+ sum-of-squares partials)
            g[LY.G_HIN_SSQ] = self.ssq_bufs[(launch_idx - 1) % 2].data_ptr()
            g[LY.G_HOUT_SSQ] = self.ssq_bufs[launch_idx % 2].data_ptr()
            return g

        frow = [0] * LY.N_PTR
        frow[LY.P_HCA_NORM] = self.packed_final["norm"].data_ptr()
        frow[LY.P_HCA_DOWN] = self.packed_final["down"].data_ptr()
        frow[LY.P_HCA_UP] = self.packed_final["up"].data_ptr()
        frow[LY.P_HCA_INJ] = self.packed[len(self.kinds) - 1]["hca"][
            "inj"
        ].data_ptr()  # combine gate unused
        iconv = issm_row = ple_ist_row = ssm_bf16 = None
        tables = []
        for li, (plan, layers) in enumerate(self.plans):
            rows = []
            for kind, layer_id in zip(plan, layers):
                t = self.packed[layer_id]
                if kind == "gdn":
                    cache = rtp.mamba2_layer_cache(layer_id)
                    conv, ssm = cache.conv[0], cache.temporal
                    assert conv.is_contiguous() and ssm.is_contiguous()
                    # SSM states f32, or bf16 with SGLANG_MAMBA_SSM_DTYPE=bfloat16
                    assert conv.dtype == torch.bfloat16 and ssm.dtype in (
                        torch.float32,
                        torch.bfloat16,
                    )
                    assert (
                        conv[0].numel() * 2 == LY.CONV_SLOT_BYTES
                        and ssm[0].numel() * 4 == LY.SSM_SLOT_BYTES
                    )
                    assert ssm_bf16 is None or ssm_bf16 == (ssm.dtype == torch.bfloat16)
                    ssm_bf16 = ssm.dtype == torch.bfloat16
                    row = pack.gdn_ptr_row(t, conv, ssm)
                    issm = cache.intermediate_ssm
                    icw = cache.intermediate_conv_window[0]
                    assert (
                        issm is not None
                        and issm.dtype == ssm.dtype
                        and issm[0].is_contiguous()
                    )
                    assert (
                        issm.shape[1] >= self.L
                        and issm.shape[0] >= R
                        and icw.dtype == torch.bfloat16
                    )
                    assert icw.shape[0] >= R and icw.shape[1] >= self.L
                    row[LY.P_ISSM] = (
                        issm.data_ptr()
                    )  # request r -> row r (verify_intermediate_state_indices)
                    row[LY.P_ICONV] = icw.data_ptr()
                    st = (icw.stride(1), icw.stride(2), icw.stride(3), icw.stride(0))
                    assert iconv is None or iconv == st
                    iconv = st
                    assert issm_row is None or issm_row == issm.stride(0)
                    issm_row = issm.stride(0)
                    if li == self.ple_launch and layer_id == layers[0]:
                        pack.add_ple_ptrs(
                            row, self.packed_ple, rtp.short_conv_layer_cache(layer_id)
                        )
                        ple_ist = rtp.short_conv_layer_intermediate_cache(layer_id)
                        assert (
                            ple_ist is not None
                            and ple_ist.dtype == torch.bfloat16
                            and ple_ist[0].is_contiguous()
                        )
                        assert ple_ist.shape[1] >= self.L and ple_ist.shape[0] >= R
                        assert ple_ist.stride(1) == 10240 * LY.PLE_STATE_LEN
                        row[LY.P_PLE_ISTATE] = ple_ist.data_ptr()
                        ple_ist_row = ple_ist.stride(0)
                    rows.append(row)
                else:
                    rows.append(
                        pack.qsa_ptr_row(
                            t,
                            kvp.get_key_buffer(layer_id),
                            kvp.get_value_buffer(layer_id),
                            kvp.get_qsa_key_state_buffer(layer_id),
                            kvp.get_qsa_compressed_k_buffer(layer_id),
                        )
                    )
            rows.append(grow_for(li))
            if li == len(self.plans) - 1:
                rows.append(frow)
            host = torch.tensor(rows, dtype=torch.int64)
            if torch.cuda.is_current_stream_capturing():
                # the captured H2D copy re-reads the pinned source at every replay: keep it alive
                pinned = host.pin_memory()
                self._pinned.append(pinned)
                tables.append(pinned.to("cuda", non_blocking=True))
            else:
                tables.append(host.to("cuda"))
        pos = fb.positions
        meta = (
            rtp.req_to_token.stride(0),
            # 1-D text positions: all three MRoPE axes read the same element
            pos.stride(0) if pos.dim() == 2 else 0,
            mamba_map.dtype == torch.int64,
            (iconv, issm_row, ple_ist_row, ictx_row, ssm_bf16),
        )
        self._tables[key] = (tables, meta)
        return self._tables[key]

    # ------------------------------------------------------------------ forward
    def forward(
        self, input_ids: torch.Tensor, forward_batch, model
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (final-mixer output x [S, HIDDEN], residual stream after the last layer [S, HC_DIM])."""
        R = forward_batch.batch_size
        S = R * self.L
        tables, (rtt_stride, pos_stride, map_i64, strides) = self._pointer_tables(
            forward_batch, model
        )
        stream = torch.cuda.current_stream()
        h = self.h_bufs[
            1
        ]  # unused by the prelude launch (layer 0 reads the embedding rows)
        for i, (plan, _) in enumerate(self.plans):
            out = self.h_bufs[i % 2]
            self._launcher(
                R,
                plan,
                rtt_stride,
                pos_stride,
                map_i64,
                i == self.ple_launch,
                i == len(self.plans) - 1,
                i == 0,
                strides,
                "embed" if i == 0 else "global",
            )(
                h.data_ptr(),
                out.data_ptr(),
                tables[i].data_ptr(),
                self.scratch.data_ptr(),
                self.step.data_ptr(),
                i,
                stream=stream,
            )
            h = out
        self.step.add_(1)
        return self.x_buf[:S], h[:S]


class Qwen4ExpMTPMonoRunner:
    """MTP (NEXTN) draft forwards of R <= 8 requests through one FlyDSL launch each: input fusion + the QSA layer
    (FP8 projections, MXFP4 experts) + the draft mixer; draft extend runs R x num_draft_tokens rows, draft decode R x 1."""

    def __init__(self, mtp_model, ntok_extend: int) -> None:
        _check_parallel()
        kernel, layout, pack, weights = _import_kernel()
        self._kernel, self._layout, self._pack = kernel, layout, pack
        self.L_ext = int(ntok_extend)
        assert 1 <= self.L_ext <= 4
        self.max_reqs = min(MAX_REQS, layout.MAX_T // self.L_ext)
        from sglang.srt.runtime_context import get_spec

        assert get_spec().speculative_eagle_topk == 1, (
            "the draft monokernel runs chain drafts (topk 1)"
        )
        config = mtp_model.config
        weights.MODEL_DIR = _checkpoint_dir(draft=True)
        dev = torch.cuda.current_device()
        moe = _moe_params(
            mtp_model.model.layers[0].mlp.experts,
            pack,
            weights,
            "mtp.layers.0.",
            check_ids=(0, 511, 512),
            dev=f"cuda:{dev}",
        )
        proj = _proj_params(
            mtp_model.model.layers[0],
            _QSA_PROJ,
            pack,
            weights,
            "mtp.layers.0.",
            check=True,
            dev=f"cuda:{dev}",
        )
        lw = weights.load_mtp(device=f"cuda:{dev}")
        self.packed = pack.pack_mtp(lw, moe=moe, proj=proj)
        del lw
        torch.cuda.empty_cache()
        layer = mtp_model.model.layers[0]
        self.layer_id = layer.attn.layer_id
        self.rotary_emb = layer.rotary_emb
        self.embed_w = mtp_model.model.embed_tokens.weight
        self.eos_token_id = int(config.eos_token_id)
        S = self.max_reqs * self.L_ext
        nbytes = max(
            layout.scratch_layout(1 + 2, r * t, r)[1]
            for r in range(1, self.max_reqs + 1)
            for t in (1, self.L_ext)
        )
        self.scratch = torch.zeros(nbytes // 4, dtype=torch.int32, device="cuda")
        self.step = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.x_buf = torch.empty(S, 2560, dtype=torch.bfloat16, device="cuda")
        # output hidden alternates between two buffers: draft decode step i + 1 reads step i's output
        self.hc_bufs = [
            torch.empty(S, 4 * 2560, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        self._calls = 0
        self.ssq_buf = torch.zeros(
            S, 4, layout.NPART, dtype=torch.float32, device="cuda"
        )
        self.mamba_map = torch.zeros(
            4096, dtype=torch.int32, device="cuda"
        )  # QSA-only launch: slot unused
        self._tables = {}
        self._launchers = {}
        self._pinned = []
        logger.info(
            f"Qwen4-Exp MTP monokernel: MTP layer packed, R <= {self.max_reqs}, extend L={self.L_ext}, "
            f"decode L=1"
        )

    def applicable(self, forward_batch, input_ids: torch.Tensor) -> bool:
        fb = forward_batch
        R = fb.batch_size
        if (
            not 1 <= R <= self.max_reqs or fb.spec_info is None
        ):  # chain drafts only: topk == 1 checked at init
            return False
        hs = fb.spec_info.hidden_states
        if (
            hs is None
            or hs.dtype != torch.bfloat16
            or hs.shape[-1] != 4 * 2560
            or not hs.is_contiguous()
        ):
            return False
        n = input_ids.numel()
        if fb.forward_mode.is_draft_extend_v2():
            return n == R * self.L_ext and hs.shape[0] == n
        return fb.forward_mode.is_decode() and n == R and hs.shape[0] == R

    def _launcher(self, R, L, rtt_stride, pos_stride):
        key = (R, L, rtt_stride, pos_stride)
        if key not in self._launchers:
            self._launchers[key] = self._kernel.build_qwen4_exp_monokernel(
                ("qsa",),
                R,
                L,
                rtt_stride,
                pos_stride,
                False,
                final_mix=True,
                first_src="mtp",
                eos_token_id=self.eos_token_id,
            )
        return self._launchers[key]

    def _table(self, forward_batch, hidden_states, S):
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        fb = forward_batch
        cos = self.rotary_emb.cos_sin_cache_as(
            torch.bfloat16
        )  # bf16 RoPE table (f32 on HIP in sglang)
        key = (
            fb.batch_size,
            S,
            fb.input_ids.data_ptr(),
            fb.positions.data_ptr(),
            fb.out_cache_loc.data_ptr(),
            fb.req_pool_indices.data_ptr(),
            fb.seq_lens.data_ptr(),
            hidden_states.data_ptr(),
            cos.data_ptr(),
        )
        if key in self._tables:
            return self._tables[key]
        pack, LY = self._pack, self._layout
        rtp = get_req_to_token_pool()
        kvp = get_token_to_kv_pool()
        assert (
            fb.input_ids.dtype == torch.int64 and fb.out_cache_loc.dtype == torch.int64
        )
        assert (
            fb.positions.dtype == torch.int64
            and fb.req_pool_indices.dtype == torch.int64
        )
        g = pack.global_ptr_row(
            fb.positions,
            fb.out_cache_loc,
            fb.req_pool_indices,
            fb.seq_lens,
            rtp.req_to_token,
            cos,
            kvp.qsa_rope_position_buffer,
            self.mamba_map,
        )
        g[LY.G_INPUT_IDS] = fb.input_ids.data_ptr()
        g[LY.G_EMBED_W] = self.embed_w.data_ptr()
        g[LY.G_MTP_HIN] = hidden_states.data_ptr()
        g[LY.G_FIN_OUT] = self.x_buf.data_ptr()
        g[LY.G_HOUT_SSQ] = self.ssq_buf.data_ptr()
        rows = pack.mtp_rows(
            self.packed,
            kvp.get_key_buffer(self.layer_id),
            kvp.get_value_buffer(self.layer_id),
            kvp.get_qsa_key_state_buffer(self.layer_id),
            kvp.get_qsa_compressed_k_buffer(self.layer_id),
            g,
        )
        host = torch.tensor(rows, dtype=torch.int64)
        if torch.cuda.is_current_stream_capturing():
            # the captured H2D copy re-reads the pinned source at every replay: keep it alive
            pinned = host.pin_memory()
            self._pinned.append(pinned)
            table = pinned.to("cuda", non_blocking=True)
        else:
            table = host.to("cuda")
        pos = fb.positions
        meta = (rtp.req_to_token.stride(0), pos.stride(0) if pos.dim() == 2 else 0)
        self._tables[key] = (table, meta)
        return self._tables[key]

    def forward(
        self, input_ids: torch.Tensor, forward_batch, hidden_states: torch.Tensor
    ):
        """-> (draft mixer output x [S, HIDDEN], hyper-connection hidden [S, HC_DIM])."""
        R = forward_batch.batch_size
        S = input_ids.numel()
        table, (rtt_stride, pos_stride) = self._table(forward_batch, hidden_states, S)
        hc = self.hc_bufs[self._calls % 2]
        if hc.data_ptr() == hidden_states.data_ptr():
            self._calls += 1
            hc = self.hc_bufs[self._calls % 2]
        self._calls += 1
        self._launcher(R, S // R, rtt_stride, pos_stride)(
            0,
            hc.data_ptr(),
            table.data_ptr(),
            self.scratch.data_ptr(),
            self.step.data_ptr(),
            0,
            stream=torch.cuda.current_stream(),
        )
        self.step.add_(1)
        return self.x_buf[:S], hc[:S]
