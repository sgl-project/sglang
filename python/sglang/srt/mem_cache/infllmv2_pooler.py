from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

class PoolerBase:
    def avg_pool1d(self, x: torch.Tensor, kernel: int, stride: int) -> torch.Tensor: raise NotImplementedError
    def max_pool1d(self, x: torch.Tensor, kernel: int, stride: int, padding: int) -> torch.Tensor: raise NotImplementedError
    def append_avg_pool1d_batched(self, prev_y, k_old, k_new, sc_old, sc_add, K, S) -> Tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError
    def append_max_pool1d_batched(self, prev_y, x_old, x_new, sc_old, sc_add, K, S, P) -> Tuple[torch.Tensor, torch.Tensor]: raise NotImplementedError

def _out_len(L: torch.Tensor, K: int, S: int, P: int = 0) -> torch.Tensor:
    num = L + 2*P - K
    return torch.where(num >= 0, num // S + 1, torch.zeros_like(num))

class TorchPooler(PoolerBase):
    @staticmethod
    def _to_ncl(x: torch.Tensor) -> torch.Tensor:
        B, HK, SC, D = x.shape
        return x.permute(0,1,3,2).reshape(B*HK, D, SC)
    @staticmethod
    def _from_ncl(y: torch.Tensor, B: int, HK: int) -> torch.Tensor:
        N, D, L = y.shape
        return y.reshape(B, HK, D, L).permute(0,1,3,2).contiguous()

    def avg_pool1d(self, x, kernel, stride):
        B, HK, SC, D = x.shape
        y = F.avg_pool1d(self._to_ncl(x), kernel_size=kernel, stride=stride, ceil_mode=False)
        return self._from_ncl(y, B, HK)

    def max_pool1d(self, x, kernel, stride, padding):
        B, HK, SC, D = x.shape
        y = F.max_pool1d(self._to_ncl(x), kernel_size=kernel, stride=stride, padding=padding, ceil_mode=False)
        return self._from_ncl(y, B, HK)

    @torch.no_grad()
    def append_avg_pool1d_batched(self, prev_y, k_old, k_new, sc_old, sc_add, K, S):
        device = k_old.device
        B, HK, _, D = k_old.shape
        N = B*HK
        tail_len = torch.minimum(sc_old, torch.tensor(K-1, device=device, dtype=sc_old.dtype))
        L_piece = tail_len + sc_add
        L_max = int(L_piece.max().item())
        xNCL = k_old.new_zeros((N, D, L_max))
        idx = 0
        for b in range(B):
            for hk in range(HK):
                tl = int(tail_len[b, hk].item()); add = int(sc_add[b, hk].item())
                if tl == 0 and add == 0:
                    idx += 1; continue
                s_old = int(sc_old[b, hk].item())
                if tl > 0:
                    tail = k_old[b, hk, s_old - tl:s_old, :]
                    xNCL[idx, :, :tl] = tail.transpose(0, 1)
                if add > 0:
                    newp = k_new[b, hk, :add, :]
                    xNCL[idx, :, tl:tl+add] = newp.transpose(0, 1)
                idx += 1
        yNCL = F.avg_pool1d(xNCL, kernel_size=K, stride=S, ceil_mode=False)
        def _ol(L, K, S, P=0):
            num = L + 2*P - K
            return torch.where(num >= 0, num // S + 1, torch.zeros_like(num))
        Y_old = _ol(sc_old, K, S, 0); Y_new = _ol(sc_old + sc_add, K, S, 0)
        delta = (Y_new - Y_old).clamp_min(0)
        y_len_piece = _ol(L_piece, K, S, 0)
        adds = []
        idx = 0
        for b in range(B):
            row = []
            for hk in range(HK):
                di = int(delta[b, hk].item()); yi = int(y_len_piece[b, hk].item())
                if di > 0:
                    y_tail = yNCL[idx, :, yi - di: yi].transpose(0, 1).contiguous()
                else:
                    y_tail = k_old.new_zeros((0, D))
                row.append(y_tail); idx += 1
            adds.append(row)
        y_concat_rows = []
        for b in range(B):
            rows = []
            for hk in range(HK):
                di = int(delta[b, hk].item()); add = adds[b][hk]
                rows.append(torch.cat([prev_y[b, hk], add[:di]], dim=0))
            y_concat_rows.append(torch.stack(rows, dim=0))
        y_concat = torch.stack(y_concat_rows, dim=0)
        return y_concat, delta

    @torch.no_grad()
    def append_max_pool1d_batched(self, prev_y, x_old, x_new, sc_old, sc_add, K, S, P):
        device = x_old.device
        B, HK, _, D = x_old.shape
        N = B*HK
        tail_len = torch.minimum(sc_old, torch.tensor(K-1, device=device, dtype=sc_old.dtype))
        L_piece = tail_len + sc_add
        L_max = int(L_piece.max().item())
        xNCL = x_old.new_zeros((N, D, L_max))
        idx = 0
        for b in range(B):
            for hk in range(HK):
                tl = int(tail_len[b, hk].item()); add = int(sc_add[b, hk].item())
                if tl == 0 and add == 0:
                    idx += 1; continue
                s_old = int(sc_old[b, hk].item())
                if tl > 0:
                    tail = x_old[b, hk, s_old - tl:s_old, :]
                    xNCL[idx, :, :tl] = tail.transpose(0, 1)
                if add > 0:
                    newp = x_new[b, hk, :add, :]
                    xNCL[idx, :, tl:tl+add] = newp.transpose(0, 1)
                idx += 1
        yNCL = F.max_pool1d(xNCL, kernel_size=K, stride=S, padding=P, ceil_mode=False)
        def _ol(L, K, S, P=0):
            num = L + 2*P - K
            return torch.where(num >= 0, num // S + 1, torch.zeros_like(num))
        Y_old = _ol(sc_old, K, S, P); Y_new = _ol(sc_old + sc_add, K, S, P)
        delta = (Y_new - Y_old).clamp_min(0)
        y_len_piece = _ol(L_piece, K, S, P)
        adds = []
        idx = 0
        for b in range(B):
            row = []
            for hk in range(HK):
                di = int(delta[b, hk].item()); yi = int(y_len_piece[b, hk].item())
                if di > 0:
                    y_tail = yNCL[idx, :, yi - di: yi].transpose(0, 1).contiguous()
                else:
                    y_tail = x_old.new_zeros((0, D))
                row.append(y_tail); idx += 1
            adds.append(row)
        y_concat_rows = []
        for b in range(B):
            rows = []
            for hk in range(HK):
                di = int(delta[b, hk].item()); add = adds[b][hk]
                rows.append(torch.cat([prev_y[b, hk], add[:di]], dim=0))
            y_concat_rows.append(torch.stack(rows, dim=0))
        y_concat = torch.stack(y_concat_rows, dim=0)
        return y_concat, delta