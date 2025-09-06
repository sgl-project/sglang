from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

from sglang.srt.layers.attention.triton_ops.infllmv2_stage1 import stage1_scores_bshd
from sglang.srt.layers.attention.triton_ops.infllmv2_stage2 import sparse_attn_stage2
from sglang.srt.mem_cache.infllmv2_memory import (
    InfLLM2KVViews, KVViewsSidecar, ContextMemoryManager
)

@dataclass
class InfLLM2Config:
    enable: bool = False
    topk: int = 8
    block_size: int = 64
    sw_span: int | None = None
    sink_len: int | None = None
    incremental: bool = True

class InfLLM2Runner:
    def __init__(self,
                 kv_views: Optional[InfLLM2KVViews] = None,
                 sidecar: Optional[KVViewsSidecar] = None,
                 cfg: Optional[InfLLM2Config] = None,
                 mem_mgr: Optional[ContextMemoryManager] = None):
        self.cfg = cfg or InfLLM2Config()
        self.kv_views = kv_views or InfLLM2KVViews()
        self.sidecar = sidecar or KVViewsSidecar(self.kv_views)
        self.mem_mgr = mem_mgr
    @staticmethod
    def _to(x: torch.Tensor, in_axes: str, out_axes: str) -> torch.Tensor:
        assert set(in_axes) == set(out_axes)
        return x.permute(*(in_axes.index(c) for c in out_axes)).contiguous()

    def _pack_incremental(self, k_BHSD: torch.Tensor, sc_done_bhk: torch.Tensor):
        B, HK, SC, D = k_BHSD.shape
        sc_old = sc_done_bhk
        sc_add = (torch.full_like(sc_old, SC) - sc_old).clamp_min(0)
        old_max = int(sc_old.max().item())
        add_max = int(sc_add.max().item())
        k_old = k_BHSD.new_zeros((B, HK, old_max, D))
        k_new = k_BHSD.new_zeros((B, HK, add_max, D))
        for b in range(B):
            for h in range(HK):
                so = int(sc_old[b,h].item()); sa = int(sc_add[b,h].item())
                if so > 0:
                    k_old[b,h,:so,:] = k_BHSD[b,h,:so,:]
                if sa > 0:
                    k_new[b,h,:sa,:] = k_BHSD[b,h,so:so+sa,:]
        return k_old, k_new, sc_old, sc_add

    def _get_cviews(self, state: dict, layer_id: int, k_BSHD: torch.Tensor):
        k_BHSD = self._to(k_BSHD, 'BSHD', 'BHSD')
        B, HK, SC, _ = k_BHSD.shape
        views = self.sidecar.ensure_handle(state)
        entry = views.get(layer_id)
        if not self.cfg.incremental or entry is None:
            return self.sidecar.get_or_build(state, layer_id, k_BHSD)
        sc_done_bhk = entry["sc_done_bhk"]
        if int(sc_done_bhk.max().item()) >= SC:
            return entry["c1"], entry["c2"]
        k_old, k_new, sc_old, sc_add = self._pack_incremental(k_BHSD, sc_done_bhk)
        return self.sidecar.update_append_batched(state, layer_id, k_old, k_new, sc_old, sc_add)

    @torch.no_grad()
    def forward(self, q_BSHD: torch.Tensor, k_BSHD: torch.Tensor, v_BSHD: torch.Tensor,
                state: Optional[dict], layer_id: int,
                kv_visible_sc: int | None = None, no_commit: bool = False,
                request_id: Optional[str] = None) -> torch.Tensor:
        assert self.cfg.enable
        if state is None and self.mem_mgr is not None and request_id is not None:
            state = self.mem_mgr.get_state(request_id)
        if state is None:
            mem = ContextMemoryManager(); state = mem.get_state("tmp")
        B, Sq, Hq, D = q_BSHD.shape
        _, Sk_full, Hk, _ = k_BSHD.shape
        if kv_visible_sc is not None:
            Sk_vis = min(int(kv_visible_sc), int(Sk_full))
            k_eff = k_BSHD[:, :Sk_vis, :, :]
            v_eff = v_BSHD[:, :Sk_vis, :, :]
        else:
            Sk_vis = int(Sk_full)
            k_eff, v_eff = k_BSHD, v_BSHD
        assert Hq % Hk == 0
        hg = Hq // Hk

        # c1/c2 views (with incremental or one-shot build when limiting visibility)
        if kv_visible_sc is not None:
            views = self.sidecar.ensure_handle(state)
            entry = views.get(layer_id)
            if entry is not None and int(entry["sc_done_bhk"].max().item()) == Sk_vis:
                c1_BHSD, c2_BHSD = entry["c1"], entry["c2"]
            else:
                c1_BHSD, c2_BHSD = self.kv_views.build_views_from_k(self._to(k_eff, 'BSHD', 'BHSD'))
        else:
            c1_BHSD, c2_BHSD = self._get_cviews(state, layer_id, k_BSHD)
        Sc1 = c1_BHSD.shape[2]

        # valid length per (B,HK) for c1 (used to mask tail in Stage‑1 probabilities)
        try:
            sc1_len, _ = self.sidecar.get_lengths(state, layer_id)
            valid_sc1_len = sc1_len
        except Exception:
            valid_sc1_len = None

        # 2) Stage-1：scores [B,Sq,Hk,Sc1_max]
        scores = stage1_scores_bshd(
            q_BSHD,
            self._to(c1_BHSD, 'BHSD', 'BSHD'),
            self._to(c2_BHSD, 'BHSD', 'BSHD'),
            hg,
            valid_sc1_len,
        )

        # 2.1) Varlen 掩码：对每个 (b,hk) 真实长度之外的 block 置 -inf
        if valid_sc1_len is not None:
            # valid_sc1_len: [B,HK]  → 广播到 [B,1,HK,1]
            lens = valid_sc1_len.clamp_min(0).clamp_max(scores.shape[-1])[:, None, :, None]
            arange = torch.arange(scores.shape[-1], device=scores.device)[None, None, None, :]
            invalid = arange >= lens
            scores = scores.masked_fill(invalid, float('-inf'))

        if self.cfg.sink_len is not None and self.cfg.sink_len > 0:
            sink_blocks = (self.cfg.sink_len + self.cfg.block_size - 1) // self.cfg.block_size
            if sink_blocks > 0:
                # 仅在真实长度内置 +inf（避免越界）
                arange = torch.arange(scores.shape[-1], device=scores.device)[None, None, None, :]
                if valid_sc1_len is not None:
                    lens = valid_sc1_len.clamp_max(scores.shape[-1])[:, None, :, None]
                    sink_mask = (arange < sink_blocks) & (arange < lens)
                else:
                    sink_mask = (arange < min(sink_blocks, scores.shape[-1]))
                scores = torch.where(sink_mask, torch.tensor(float('inf'), device=scores.device), scores)
        if self.cfg.sw_span is not None and self.cfg.sw_span > 0:
            n_blocks = (Sk_vis + self.cfg.block_size - 1)//self.cfg.block_size
            sw_begin_tok = max(0, Sk_vis - self.cfg.sw_span)
            sw_begin_block = min(n_blocks - 1, max(0, sw_begin_tok // self.cfg.block_size))
            arange = torch.arange(scores.shape[-1], device=scores.device)[None, None, None, :]
            if valid_sc1_len is not None:
                lens = valid_sc1_len.clamp_max(scores.shape[-1])[:, None, :, None]
                # 仅在 [sw_begin_block, lens) 内置 -inf，避免 SW 被 TopK 重复选
                sw_mask = (arange >= sw_begin_block) & (arange < lens)
            else:
                sw_mask = (arange >= sw_begin_block)
            scores = scores.masked_fill(sw_mask, float('-inf'))

        # 4) Varlen TopK：被 -inf 掩掉的尾部不会被选中；若极早期 lens < TopK，torch.topk 仍会返回，
        #    但全是 -inf 的位置不会影响 Stage-2（我们后续还有并集去重）
        k_eff = min(self.cfg.topk, scores.shape[-1])
        topk_idx = torch.topk(scores, k=k_eff, dim=-1).indices  # [B,Sq,Hk,K]

        # Stage‑2 sparse attention (union TopK/SW/Sink)
        out = sparse_attn_stage2(q_BSHD, k_eff, v_eff,
                                 topk_idx,
                                 block_size=self.cfg.block_size,
                                 sw_span=self.cfg.sw_span,
                                 sink_len=self.cfg.sink_len)
        return out