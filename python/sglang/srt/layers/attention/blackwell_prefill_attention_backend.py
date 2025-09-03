from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

# from sglang.srt.managers.schedule_batch import global_server_args_dict
# from sglang.srt.mem_cache.memory_pool import SWAKVPool
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


def _green(x: str) -> str:
    return f"\033[1;32m{x}\033[0m"


def _red(x: str) -> str:
    return f"\033[1;31m{x}\033[0m"


def _yellow(x: str) -> str:
    return f"\033[1;33m{x}\033[0m"


@dataclass
class ForwardMetaData:
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    page_table: Optional[torch.Tensor] = None
    seqlens_k: Optional[torch.Tensor] = None


class BlackwellPrefillAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        from sglang.srt.layers.attention.cute_ops.prefill_attention import (
            flash_attn_varlen_func,
        )

        super().__init__()
        self.flash_attn_func = flash_attn_varlen_func
        self.page_size = model_runner.page_size
        self.device = model_runner.device
        self.forward_metadata: Optional[ForwardMetaData] = None

        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        self._triton_backend = TritonAttnBackend(model_runner=model_runner, skip_prefill=False)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        assert (
            forward_batch.forward_mode.is_extend()
        ), "Only support extend (i.e., prefill) batches."

        max_seqlen_k = forward_batch.seq_lens_cpu.max().item()
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), pad=(1, 0)
        )
        seqlens_k = forward_batch.seq_lens.to(torch.int32)
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        if any(forward_batch.extend_prefix_lens_cpu):
            extend_seq_lens = forward_batch.extend_seq_lens
            cu_seqlens_q = F.pad(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), pad=(1, 0)
            )
        else:
            cu_seqlens_q = cu_seqlens_k

        if self.page_size > 1:
            strided_indices = torch.arange(0, page_table.shape[1], self.page_size, device=self.device)
            page_table = page_table[:, strided_indices] // self.page_size

        self.forward_metadata = ForwardMetaData(
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, page_table=page_table, seqlens_k=seqlens_k,
        )

        print("=" * 120)
        print(_red("forward_batch.encoder_lens:"), forward_batch.encoder_lens)
        print(_red("page_table:"), page_table)
        print(_red("max_seqlens_k:"), max_seqlen_k)
        print(_red("cu_seqlens_k:"), cu_seqlens_k)

        self._triton_backend.init_forward_metadata(forward_batch)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        raise RuntimeError("Prefill attention should not be captured in a CUDA graph.")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        raise RuntimeError("Prefill attention should not be replayed in a CUDA graph.")

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
    ):
        save_kv_cache = True

        torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)
        np.set_printoptions(suppress=True, precision=3, linewidth=120, formatter={"float": "{:>8.3f}".format})

        # if layer.layer_id == 0:
        #     print("=" * 120)
        #     print(_yellow("k_cache:"), k_cache.abs().sum().item())
        #     print(_yellow("v_cache:"), v_cache.abs().sum().item())

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer=layer,
                loc=forward_batch.out_cache_loc,
                cache_k=k,
                cache_v=v,
            )

        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim)

        metadata = self.forward_metadata
        if layer.layer_id == 0:
            print(_yellow(f"{layer.is_cross_attention=}"))
            print(_yellow(f"{forward_batch.encoder_out_cache_loc=}"))
            print(_yellow("k:"), "\n", k.shape, "\n", k.reshape(-1)[:-8], "\n", k.sum().item())
            print(_yellow("v:"), "\n", v.shape, "\n", v.reshape(-1)[:-8], "\n", v.sum().item())
            print(_yellow("cu_seqlens_q:"), metadata.cu_seqlens_q)
            print(_yellow("out_cache_loc:"), forward_batch.out_cache_loc.reshape(-1))
            print(_yellow("k_cache:"), "\n", k_cache.shape, "\n", k_cache.view(-1, layer.tp_k_head_num, layer.head_dim)[forward_batch.out_cache_loc].reshape(-1)[:8], "\n", torch.where(k_cache.reshape(-1) != 0.)[0], "\n", k_cache.sum().item())
            print(_yellow("v_cache:"), "\n", v_cache.shape, "\n", v_cache.view(-1, layer.tp_v_head_num, layer.head_dim)[forward_batch.out_cache_loc].reshape(-1)[:8], "\n", torch.where(v_cache.reshape(-1) != 0.)[0], "\n", v_cache.sum().item())
            print(_yellow("page_table:"), metadata.page_table)

        out = self.flash_attn_func(
            q=q.reshape(-1, layer.tp_q_head_num, layer.head_dim),
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=metadata.cu_seqlens_q,
            seqused_k=metadata.seqlens_k,
            page_table=metadata.page_table,
            softcap=layer.logit_cap,
            softmax_scale=layer.scaling,
            window_size=(layer.sliding_window_size, 0) if layer.sliding_window_size is not None and layer.sliding_window_size > 0 else (None, None),
            causal=True,
            learnable_sink=sinks.to(torch.bfloat16) if sinks is not None else None,
        )[0]

        ref = self._triton_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=False,
            sinks=sinks,
        )

        max_diff = (out.reshape(-1) - ref.reshape(-1)).float().abs().max().item()
        print(_yellow(f"out: {layer.layer_id=}"), "\n", out.reshape(-1)[:8], "\n", out.reshape(-1)[-8:])
        print(_green(f"ref: {layer.layer_id=}"), "\n", ref.reshape(-1)[:8], "\n", ref.reshape(-1)[-8:])
        if max_diff > 0.75:
            print(_yellow("sliding_window_size:"), "\n", f"{layer.sliding_window_size=}")
            print(_red("big difference!!!"), f"{layer.layer_id=} {max_diff=:<.5f} SAVING TO DISK!!!", flush=True)

            from pathlib import Path
            out_path = Path("/sgl-workspace/debug_output")

            torch.save(q, str((out_path / "q.pt").absolute()))
            torch.save(k, str((out_path / "k.pt").absolute()))
            torch.save(v, str((out_path / "v.pt").absolute()))
            torch.save(k_cache, str((out_path / "k_cache.pt").absolute()))
            torch.save(v_cache, str((out_path / "v_cache.pt").absolute()))
            torch.save(metadata.cu_seqlens_q, str((out_path / "cu_seqlens_q.pt").absolute()))
            torch.save(metadata.cu_seqlens_k, str((out_path / "cu_seqlens_k.pt").absolute()))
            torch.save(metadata.page_table, str((out_path / "page_table.pt").absolute()))
            torch.save(out, str((out_path / "out_backend.pt").absolute()))
            torch.save(ref, str((out_path / "ref_backend.pt").absolute()))

            torch.distributed.breakpoint()

            print("DONE", flush=True)
            assert False
        else:
            print(_green("okay"), f"{layer.layer_id=}", flush=True)

        return out.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        raise NotImplementedError(
            "BlackwellPrefillAttentionBackend does not support forward_decode"
        )

    forward = forward_extend

    def support_triton(self):
        return False
