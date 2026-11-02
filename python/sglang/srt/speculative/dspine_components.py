# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Adapted from Tencent/DSpine and vLLM PR #63704 (Apache-2.0).

import math

import torch
import torch.nn.functional as F
from torch import nn


def resolve_dspine_head_topk(config, num_layers: int | None = None) -> int:
    """Resolve the width K for serial last-write refinement.

    Read ``code_bridge_second_pass`` (mode ``serial_table``, ``first_exact`` is
    the last layer) and/or ``dflash_config.selected_prefix_last_write_refine``
    (kind ``exact_sender``), falling back to ``dflash_config.dspine_top_k`` /
    ``dflash_config.transition_top_k``. Every source found must agree.
    """
    if num_layers is None:
        num_layers = int(config.num_hidden_layers)
    num_layers = int(num_layers)
    dflash_cfg = getattr(config, "dflash_config", None) or {}
    found = {}
    second = getattr(config, "code_bridge_second_pass", None)
    if second is not None:
        second = dict(second)
        unknown = set(second) - {"mode", "first_exact", "head_topk"}
        if unknown:
            raise ValueError(f"Unknown code_bridge_second_pass keys: {sorted(unknown)}")
        if second.get("mode") != "serial_table":
            raise ValueError(
                "DSPINE serves the serial last-write refinement only; the "
                "checkpoint config says "
                f"code_bridge_second_pass.mode={second.get('mode')!r}"
            )
        if int(second.get("first_exact", -1)) != num_layers - 1:
            raise ValueError(
                "serial_table redoes the LAST write only: "
                "code_bridge_second_pass.first_exact must be "
                f"{num_layers - 1}, got {second.get('first_exact')!r}"
            )
        found["code_bridge_second_pass.head_topk"] = int(second["head_topk"])
    refine = dflash_cfg.get("selected_prefix_last_write_refine")
    if refine is not None:
        if refine.get("version", 1) != 1 or refine.get("weight", 1.0) != 1.0:
            raise ValueError(
                "DSPINE requires version-1 unscaled exact-sender refinement"
            )
        if refine.get("kind") != "exact_sender":
            raise ValueError(
                "DSPINE expects "
                "selected_prefix_last_write_refine.kind == 'exact_sender', "
                f"got {refine.get('kind')!r}"
            )
        found["dflash_config.selected_prefix_last_write_refine.head_topk"] = int(
            refine["head_topk"]
        )
    for key in ("dspine_top_k", "transition_top_k"):
        if dflash_cfg.get(key) is not None:
            found[f"dflash_config.{key}"] = int(dflash_cfg[key])
    if not found:
        raise ValueError(
            "DSPINE checkpoint config names no refinement width: add "
            "code_bridge_second_pass {mode: serial_table, first_exact, "
            "head_topk}, dflash_config.selected_prefix_last_write_refine."
            "head_topk, or a dflash_config top-k fallback"
        )
    values = set(found.values())
    if len(values) != 1:
        raise ValueError(f"DSPINE refinement widths disagree: {found}")
    k = values.pop()
    if k & (k - 1) or not 2 <= k <= 64:
        raise ValueError(f"DSPINE head_topk must be a power of two in [2, 64], got {k}")
    return k


class DSpineCodeBridge(nn.Module):
    """Post-layer writes with DFlash anchor-row semantics.

    ``dflash_config.selected_prefix_code_bridge`` specifies ``code_size``,
    ``rank``, and ``eps``. ``selected_prefix_feature_layers`` is 1-based and
    includes the last layer. Each write fuses the next layer's input RMSNorm
    (or the final norm). ``dflash_config.dspine_bridge_impl`` selects the
    ``fused`` Triton path (CUDA + BF16) or the ``eager`` torch path.
    """

    def __init__(self, config, params_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        cfg = getattr(config, "dflash_config", None) or {}
        if cfg.get("selected_prefix_terminal", True):
            raise ValueError("DSPINE supports terminal-disabled code bridges only")
        if cfg.get("selected_prefix_feature_write_mode") != "independent":
            raise ValueError("DSPINE requires independent learned writes")
        if cfg.get("selected_prefix_feature_writes") is not True:
            raise ValueError("DSPINE requires selected_prefix_feature_writes = true")
        if not cfg.get("draft_block_causal"):
            raise ValueError(
                "DSPINE requires causal draft attention (draft_block_causal)"
            )
        if cfg.get("selected_prefix_feature_reliability") is not None:
            raise ValueError("Reliability probes are not part of the DSPINE path")
        if cfg.get("selected_prefix_feature_next_token_rows", False):
            raise ValueError(
                "DSPINE's serial refinement is defined for anchor-row "
                "(DFlash) semantics; next-token rows are a DSpark layout"
            )
        bridge = cfg.get("selected_prefix_code_bridge")
        if not bridge:
            raise ValueError(
                "DSPINE draft config lacks dflash_config.selected_prefix_code_bridge"
            )
        self.hidden_size = int(config.hidden_size)
        self.code_size = int(bridge["code_size"])
        self.rank = int(bridge["rank"])
        self.num_layers = int(config.num_hidden_layers)
        self.block_size = int(config.block_size)
        self.eps = float(bridge["eps"])
        if self.block_size < 2:
            raise ValueError(
                "DSPINE block_size must include an anchor and at least one prediction"
            )
        if (
            min(self.hidden_size, self.code_size, self.rank, self.num_layers) < 1
            or not math.isfinite(self.eps)
            or self.eps <= 0
        ):
            raise ValueError(
                "DSPINE bridge dimensions and finite epsilon must be positive"
            )
        feature_layers = cfg.get("selected_prefix_feature_layers")
        if feature_layers is None:
            raise ValueError(
                "DSPINE requires dflash_config.selected_prefix_feature_layers"
            )
        self.active_layers = frozenset(int(i) - 1 for i in feature_layers)
        if not self.active_layers <= set(range(self.num_layers)):
            raise ValueError(
                "selected_prefix_feature_layers out of range: "
                f"{sorted(self.active_layers)}"
            )
        if self.num_layers - 1 not in self.active_layers:
            raise ValueError(
                "DSPINE refines the last layer's write, so the last layer "
                "must be active"
            )
        implementation = cfg.get("dspine_bridge_impl", "fused")
        if implementation not in ("fused", "eager"):
            raise ValueError(f"Unknown DSPINE bridge implementation {implementation!r}")
        self.implementation = implementation
        self.prev = nn.Linear(self.code_size, self.rank, bias=False, dtype=params_dtype)
        self.gate = nn.Linear(self.code_size, self.rank, bias=True, dtype=params_dtype)
        self.predecessor_gate = nn.Linear(
            self.code_size, self.rank, bias=False, dtype=params_dtype
        )
        self.out = nn.Linear(self.rank, self.code_size, bias=False, dtype=params_dtype)
        self.register_buffer(
            "codebook",
            torch.empty(config.vocab_size, self.code_size, dtype=params_dtype),
        )
        self.register_buffer(
            "read_weight",
            torch.empty(self.code_size, self.hidden_size, dtype=params_dtype),
        )
        self.register_buffer(
            "early_read_weights",
            torch.empty(
                self.num_layers - 1,
                self.code_size,
                self.hidden_size,
                dtype=params_dtype,
            ),
        )
        self.feature_write_weights = nn.Parameter(
            torch.empty(
                self.num_layers,
                self.hidden_size,
                self.rank,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        for name in ("independent_write_version", "terminal_disabled_version"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        self.register_buffer("packed_weight", None, persistent=False)
        self._loaded = set()
        self._ready = False

    def begin_load(self) -> None:
        self._loaded.clear()
        self._ready = False

    @torch.no_grad()
    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        destinations = dict(self.named_parameters()) | dict(self.named_buffers())
        if name not in destinations or name == "packed_weight":
            raise ValueError(f"Unexpected DSPINE code-bridge weight: {name}")
        dest = destinations[name]
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite DSPINE code-bridge weight: {name}")
        if tuple(dest.shape) != tuple(tensor.shape):
            raise ValueError(
                f"DSPINE code-bridge shape mismatch: {name}: "
                f"{tuple(tensor.shape)} vs {tuple(dest.shape)}"
            )
        dest.data.copy_(tensor)
        self._loaded.add(name)

    @torch.no_grad()
    def prepare(self) -> None:
        expected = (
            set(dict(self.named_parameters())) | set(dict(self.named_buffers()))
        ) - {"packed_weight"}
        missing = expected - self._loaded
        if missing:
            raise ValueError(f"Missing DSPINE code-bridge tensors: {sorted(missing)}")
        if self.out.weight.count_nonzero().item():
            raise ValueError(
                "Terminal-disabled checkpoint contains a nonzero terminal projection"
            )
        for name in ("independent_write_version", "terminal_disabled_version"):
            if getattr(self, name).item() != 1:
                raise ValueError(f"Unsupported DSPINE {name}")
        packed = torch.cat(
            (self.gate.weight, self.predecessor_gate.weight, self.prev.weight),
            dim=0,
        ).contiguous()
        # Preserve graph-captured storage during same-shape weight reloads.
        if self.packed_weight is None:
            self.packed_weight = packed
        else:
            self.packed_weight.copy_(packed)
        self._ready = True

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        return (
            self.read_weight
            if layer == self.num_layers - 1
            else self.early_read_weights[layer]
        )

    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One write after ``layer``.

        Returns (normed, out, raw, rms, packed): ``normed`` [m, H] is the
        next layer's pre-normed input, ``out`` [m, H] the updated residual
        stream, ``raw`` [m, H] the pre-write residual stream, ``rms`` [m] the
        fp32 per-row RMS, and ``packed`` [m, 3R] the code projections
        [Q | S | V].
        """
        if not self._ready:
            raise RuntimeError("DSPINE code-bridge weights were not prepared")
        if hidden.shape[0] % self.block_size:
            raise ValueError(
                "DSPINE code-bridge input must contain whole request blocks"
            )
        if (
            self.implementation == "fused"
            and hidden.is_cuda
            and hidden.dtype == torch.bfloat16
        ):
            return self._forward_fused(hidden, residual, input_ids, layer, next_norm)
        return self._forward_eager(hidden, residual, input_ids, layer, next_norm)

    def _read_codes(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused read: (raw [m,H], rms [m] fp32, codes [m,C])."""
        import triton

        from sglang.kernels.ops.speculative.dspine import (
            _normalize_codes_kernel,
            _prepare_read_kernel,
        )

        m, h = hidden.shape
        c, t = self.code_size, self.block_size
        raw = torch.empty_like(hidden)
        normalized = torch.empty_like(hidden)
        rms = torch.empty(m, device=hidden.device, dtype=torch.float32)
        _prepare_read_kernel[(m,)](
            hidden,
            hidden if residual is None else residual,
            raw,
            normalized,
            rms,
            h,
            self.eps,
            residual is not None,
            triton.next_power_of_2(h),
            enable_fp_fusion=False,
        )
        projected = F.linear(normalized, self.layer_read_weight(layer))
        codes = torch.empty_like(projected)
        anchors = input_ids.reshape(-1, t)[:, 0].contiguous()
        _normalize_codes_kernel[(m,)](
            projected,
            self.codebook,
            anchors,
            codes,
            c,
            t,
            True,
            self.eps,
            triton.next_power_of_2(c),
            enable_fp_fusion=False,
        )
        return raw, rms, codes

    def _forward_fused(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import triton

        from sglang.kernels.ops.speculative.dspine import (
            _feature_finish_kernel,
            _feature_message_kernel,
        )

        m = hidden.shape[0]
        r, t, h = self.rank, self.block_size, self.hidden_size
        raw, rms, codes = self._read_codes(hidden, residual, input_ids, layer)
        packed = F.linear(codes, self.packed_weight)
        message = torch.empty((m, r), device=hidden.device, dtype=hidden.dtype)
        _feature_message_kernel[(m,)](
            packed,
            self.gate.bias,
            message,
            r,
            t,
            triton.next_power_of_2(r),
            enable_fp_fusion=False,
        )
        delta = F.linear(message, self.feature_write_weights[layer])
        valid = torch.ones(m, device=raw.device, dtype=torch.bool)
        output = torch.empty_like(raw)
        normed = torch.empty_like(raw)
        _feature_finish_kernel[(m,)](
            raw,
            delta,
            rms,
            valid,
            next_norm.weight,
            output,
            normed,
            h,
            t,
            True,
            True,
            next_norm.variance_epsilon,
            triton.next_power_of_2(h),
            enable_fp_fusion=False,
        )
        return normed, output, raw, rms, packed

    def _forward_eager(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plain-torch path with the same rounding points as the fused kernels.

        Runs in hidden.dtype: BF16 on GPU, fp32 on CPU for reference checks.
        """
        t, r = self.block_size, self.rank
        raw = hidden if residual is None else hidden + residual
        f = raw.float()
        rms = f.square().mean(-1).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        normalized = (f / (rms[:, None] + self.eps)).to(raw.dtype)
        projected = F.linear(
            normalized,
            self.layer_read_weight(layer).to(normalized.dtype),
        ).float()
        codes = (
            math.sqrt(self.code_size)
            * projected
            / (projected.norm(dim=-1, keepdim=True) + self.eps)
        )
        anchors = input_ids.reshape(-1, t)[:, 0]
        anchor_codes = (
            math.sqrt(self.code_size) * F.embedding(anchors, self.codebook).float()
        )
        codes = codes.view(-1, t, self.code_size)
        codes = torch.cat((anchor_codes[:, None, :], codes[:, 1:]), dim=1).view(
            -1, self.code_size
        )
        packed = F.linear(codes.to(raw.dtype), self.packed_weight.to(raw.dtype))
        q, s, v = packed.float().view(-1, t, 3 * r).split(r, dim=-1)
        gate = torch.sigmoid(q[:, 1:] + s[:, :-1] + self.gate.bias.float())
        message = torch.cat((torch.zeros_like(q[:, :1]), gate * v[:, :-1]), dim=1).to(
            raw.dtype
        )
        delta = F.linear(
            message.view(-1, r),
            self.feature_write_weights[layer].to(raw.dtype),
        ).float()
        not_anchor = (torch.arange(raw.shape[0], device=raw.device) % t != 0).float()[
            :, None
        ]
        out = (f + (rms[:, None] * delta) * not_anchor).to(raw.dtype)
        x = out.float()
        y = (
            x
            * torch.rsqrt(
                x.square().mean(-1, keepdim=True) + next_norm.variance_epsilon
            )
        ).to(raw.dtype)
        normed = (y.float() * next_norm.weight.float()).to(raw.dtype)
        return normed, out, raw, rms, packed


@torch.no_grad()
def score_dspine_transitions(
    candidate_ids: torch.Tensor,
    raw: torch.Tensor,
    rms: torch.Tensor,
    query: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    *,
    codebook: torch.Tensor,
    pred_gate: torch.Tensor,
    prev: torch.Tensor,
    gate_bias: torch.Tensor,
    write: torch.Tensor,
    norm: torch.Tensor,
    lm_head_weight: torch.Tensor,
    norm_eps: float,
    write_gram: torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """fp32 [B, T-1, K, K] transition logits of the redone last write.

    For sender s (the anchor on the first proposal row, the previous
    row's candidate lane otherwise), e = sqrt(C) * codebook[s] and the
    refined state is
    x = raw + rms * write_w @ (sigmoid(query + pred_gate_w @ e + gate_b)
                               * (prev_w @ e)).
    Row [b, i, j, c] is the lm_head logit of candidate c under sender
    lane j. With ``normalize`` the final RMSNorm denominator
    1/sqrt(mean(x**2) + eps) is applied (matching ref_refine_table /
    serial_rs_table); without it the scores match the serial_table
    greedy table, whose positive scale cannot change the argmax.
    """
    b, s, k = candidate_ids.shape
    senders = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(b, 1, k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    sqrt_c = math.sqrt(codebook.shape[1])
    e = codebook[senders].float() * sqrt_c  # [B, S, K, C]
    gate = torch.sigmoid(query.float()[:, :, None, :] + e @ pred_gate.T + gate_bias)
    msg = gate * (e @ prev.T)  # [B, S, K, R]
    x = raw.float()
    scale = rms.float()
    # R-space algebra: keep the [B, S, K, H] refined states implicit.
    lm_c = lm_head_weight[candidate_ids].float()  # [B, S, K, H]
    nw = norm
    base = torch.einsum("bsh,bsch->bsc", x * nw, lm_c)
    tg_c = torch.matmul(lm_c * nw, write)  # [B, S, K, R]
    trans = torch.einsum("bsjr,bscr->bsjc", msg, tg_c)
    logits = base[:, :, None, :] + scale[:, :, None, None] * trans
    if normalize:
        wtx = torch.matmul(x, write)  # [B, S, R]
        sq = (
            x.square().sum(-1)[:, :, None]
            + 2 * scale[:, :, None] * torch.einsum("bsr,bskr->bsk", wtx, msg)
            + scale[:, :, None].square()
            * torch.einsum(
                "bskr,bskr->bsk",
                msg,
                torch.matmul(
                    msg, write.T @ write if write_gram is None else write_gram
                ),
            )
        )
        denom = torch.sqrt(sq.clamp_min(0) / x.shape[-1] + norm_eps)
        logits = logits / denom[:, :, :, None]
    return logits
