import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ray
import torch
import torch.nn as nn
from ray.experimental.tqdm_ray import tqdm
from sgl_kernel import merge_state_v2
from transformers import AutoConfig, PretrainedConfig

import sglang.srt.distributed.parallel_state as parallel_state
import sglang.srt.layers.dp_attention as dp_attention
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import add_prefix, bind_or_assign, get_device_name

os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "0"


def get_config_file_name(attention_backend, num_local_heads) -> str:
    device_name = get_device_name().replace(" ", "_")
    return (
        f"H={num_local_heads},device_name={device_name}_attn={attention_backend}.json"
    )


def init_comm(tp_size):
    dp_attention._ATTN_TP_SIZE = tp_size
    parallel_state._TP = Fake()
    parallel_state._TP.rank_in_group = 0
    parallel_state._TP.world_size = tp_size


class DeepseekV2AttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
        quant_config: Optional = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        tp_size: int = 8,
        attention_backend: str = "flashinfer",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.quant_config = quant_config

        attn_tp_rank = 0
        attn_tp_size = tp_size
        self.attention_backend = attention_backend

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.attn_mha.kv_b_proj = None

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0
        self.init_w_kvc()

    def init_w_kvc(self):
        w = self.kv_b_proj.weight
        assert w.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        )
        assert (
            hasattr(self.quant_config, "weight_block_size")
            and self.quant_config.weight_block_size is not None
        )
        weight_block_size = self.quant_config.weight_block_size
        assert hasattr(self.kv_b_proj, "weight_scale_inv")
        weight = w
        weight_scale = self.kv_b_proj.weight_scale_inv
        w = block_quant_dequant(
            weight,
            weight_scale,
            weight_block_size,
            torch.bfloat16,
        )

        w_kc, w_vc = w.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)

        self.w_kc = bind_or_assign(
            self.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        )
        self.w_vc = bind_or_assign(self.w_vc, w_vc.contiguous().transpose(1, 2))

    def forward_absorb(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        forward_batch.attn_attend_prefix_cache = None
        forward_batch.mha_return_lse = None
        k_nope, k_pe = latent_cache.unsqueeze(1).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_nope_out = q_nope_out.transpose(0, 1)

        if (
            self.attention_backend == "fa3"
            or self.attention_backend == "flashinfer"
            or self.attention_backend == "cutlass_mla"
        ):
            attn_output = self.attn_mqa(
                q_nope_out, k_nope, k_nope, forward_batch, q_rope=q_pe, k_rope=k_pe
            )
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
            attn_output = self.attn_mqa(q, k, k_nope, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = torch.empty(
            (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=attn_bmm_output.view(
                -1, self.num_local_heads, self.v_head_dim
            ).transpose(0, 1),
        )
        return attn_bmm_output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[
                forward_batch.prefix_chunk_kv_indices[i]
            ].contiguous()

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]
            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        k_nope, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.unsqueeze(1)
        kv = self.kv_b_proj(k_nope.contiguous())[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if forward_batch.num_prefix_chunks is None:
            if has_extend_prefix:
                forward_batch.prepare_chunked_prefix_cache_info(q.device)
            else:
                forward_batch.num_prefix_chunks = 0
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )
        return attn_output


class Fake(object):
    pass


class DeepseekAttnWrapper(torch.nn.Module):
    def __init__(
        self,
        args,
        config: PretrainedConfig,
        device,
        prefix: str = "",
    ):
        super().__init__()
        init_comm(args.tp_size)
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.hidden_size = config.hidden_size
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=0,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            tp_size=args.tp_size,
            attention_backend=args.attention_backend,
        )
        self.num_local_heads = self.self_attn.num_local_heads
        self.page_size = 1
        self.kv_cache_dtype = torch.bfloat16
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
        self.device = device
        self.max_num_reqs = 128
        self.context_len = 65535
        self.max_total_num_tokens = 83994
        self.use_mla_backend = True
        self.init_kvcache()
        self.init_fake_model_runner(args)
        self.attn_backend = self._get_attention_backend(args.attention_backend)

    def init_fake_model_runner(self, args):
        model_config = ModelConfig(
            model_path=args.model,
            trust_remote_code=True,
            revision=None,
            context_length=self.context_len,
            model_override_args="{}",
            is_embedding=False,
            enable_multimodal=False,
            dtype="auto",
            quantization=None,
            hybrid_kvcache_ratio=None,
        )
        self.model_runner = Fake()
        self.model_runner.model_config = model_config
        self.model_runner.dtype = torch.bfloat16
        self.model_runner.device = self.device
        self.model_runner.kv_cache_dtype = self.kv_cache_dtype
        self.model_runner.req_to_token_pool = self.req_to_token_pool
        self.model_runner.token_to_kv_pool = self.token_to_kv_pool
        self.model_runner.sliding_window_size = None
        self.model_runner.page_size = self.page_size
        self.model_runner.is_hybrid = None

        self.model_runner.server_args = Fake()
        self.model_runner.server_args.page_size = self.page_size
        self.model_runner.server_args.kv_cache_dtype = "auto"
        self.model_runner.server_args.speculative_eagle_topk = None
        self.model_runner.server_args.speculative_num_draft_tokens = None
        self.model_runner.server_args.speculative_eagle_topk = None
        self.model_runner.server_args.enable_dp_attention = False

    def init_kvcache(self):
        self.req_to_token_pool = ReqToTokenPool(
            size=self.max_num_reqs,
            max_context_len=self.context_len + 4,
            device=self.device,
            enable_memory_saver=False,
        )

        self.token_to_kv_pool = MLATokenToKVPool(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=1,
        )

    def _get_attention_backend(self, attention_backend):
        if attention_backend == "flashinfer" or attention_backend == "flashmla":
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAAttnBackend,
            )

            return FlashInferMLAAttnBackend(self.model_runner)
        elif attention_backend == "fa3":
            assert (
                torch.cuda.get_device_capability()[0] == 8 and not self.use_mla_backend
            ) or torch.cuda.get_device_capability()[0] == 9, (
                "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
                "Please use `--attention-backend flashinfer`."
            )
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            return FlashAttentionBackend(self.model_runner)
        else:
            raise ValueError(
                f"Invalid attention backend: {self.server_args.attention_backend}"
            )

    def forward(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        num_iters: int = 10,
    ):
        for _ in range(3):
            self.self_attn.forward_normal_chunked_kv(q, latent_cache, forward_batch)
            self.self_attn.forward_absorb(q, latent_cache, forward_batch)

        latencies_mha: List[float] = []
        latencies_mla: List[float] = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            time0 = time.perf_counter()
            self.self_attn.forward_normal_chunked_kv(q, latent_cache, forward_batch)
            torch.cuda.synchronize()
            time1 = time.perf_counter()

            self.self_attn.forward_absorb(q, latent_cache, forward_batch)
            torch.cuda.synchronize()
            time2 = time.perf_counter()
            latencies_mha.append((time1 - time0))
            latencies_mla.append((time2 - time1))
        return (
            min(latencies_mla) * 1000000,
            min(latencies_mha) * 1000000,
        )  # us

    def create_requests(self, prefix_lens, extend_lens):
        out_cache_loc = torch.arange(
            sum(prefix_lens) + 1,
            sum(prefix_lens) + sum(extend_lens) + 1,
            dtype=torch.int32,
            device=self.device,
        )
        req_pool_indices = torch.arange(
            1, len(extend_lens) + 1, dtype=torch.int32, device=self.device
        )
        sum_len = 1
        for i in range(len(extend_lens)):
            token_len = prefix_lens[i] + extend_lens[i]
            self.req_to_token_pool.req_to_token[i + 1, :token_len] = torch.arange(
                sum_len, sum_len + token_len, dtype=torch.int32, device=self.device
            )
            sum_len += token_len

        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        ret = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(extend_lens),
            input_ids=torch.randint(
                0,
                100,
                (sum(extend_lens),),
                dtype=torch.int64,
                device=self.device,
            ),
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=self.device),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=sum(seq_lens),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64, device="cpu"),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
        )
        ret.extend_seq_lens = torch.tensor(
            extend_lens, dtype=torch.int32, device=self.device
        )
        ret.extend_prefix_lens = torch.tensor(
            prefix_lens, dtype=torch.int32, device=self.device
        )
        ret.extend_num_tokens = sum(extend_lens)
        ret.extend_prefix_lens_cpu = prefix_lens
        ret.extend_seq_lens_cpu = extend_lens

        self.attn_backend.init_forward_metadata(ret)

        q = torch.randn(
            (sum(extend_lens), self.self_attn.num_local_heads, self.qk_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        latent = torch.randn(
            (sum(extend_lens), self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        return q, latent, ret


class BenchmarkConfig(TypedDict):
    prefix_seq_tuple: List[Tuple[int, int]]


def run_perf(attn_wrapper: DeepseekAttnWrapper, prefix_lens, extend_lens):
    q, latent, forward_batch = attn_wrapper.create_requests(prefix_lens, extend_lens)
    mla, mha = attn_wrapper.forward(q, latent, forward_batch)
    return mla, mha


def generate_numbers_with_sum(n, s, allow_repeat):
    if s <= n:
        return [1] * s + [0] * (n - s)
    if allow_repeat:
        points = [0] + sorted(random.choices(range(1, s), k=n - 1)) + [s]
    else:
        ls = list(range(1, s))
        random.shuffle(ls)
        points = [0] + sorted(ls[: n - 1]) + [s]
    return [points[i + 1] - points[i] for i in range(n)]


def generate_avg_seqs(batch_size, prefix_len_sum, extend_len_sum):
    prefix_lens = [prefix_len_sum // batch_size] * batch_size
    for i in range(prefix_len_sum % batch_size):
        prefix_lens[i] += 1
    extend_lens = [extend_len_sum // batch_size] * batch_size
    for i in range(extend_len_sum % batch_size):
        extend_lens[i] += 1
    return prefix_lens, extend_lens


def generate_random_seqs(batch_size, prefix_len_sum, extend_len_sum):
    prefix_lens = generate_numbers_with_sum(batch_size, prefix_len_sum, True)
    extend_lens = generate_numbers_with_sum(batch_size, extend_len_sum, False)
    return prefix_lens, extend_lens


def generate_random_seqs_min_se_mul(
    batch_size, prefix_len_sum, extend_len_sum, seq_extend_mul
):
    for _ in range(1000):
        prefix_lens = generate_numbers_with_sum(batch_size, prefix_len_sum, True)
        extend_lens = generate_numbers_with_sum(batch_size, extend_len_sum, False)
        cur_mul = sum([e * (p + e) for p, e in zip(prefix_lens, extend_lens)])
        if cur_mul >= seq_extend_mul:
            return prefix_lens, extend_lens
    return None


class LogItem:
    def __init__(self, prefix_lens, extend_lens, mla_cost, mha_cost):
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.mla_cost = mla_cost
        self.mha_cost = mha_cost

    @property
    def pe(self):
        return sum([p * e for p, e in zip(self.prefix_lens, self.extend_lens)])

    @property
    def ee(self):
        return sum([e * e for p, e in zip(self.prefix_lens, self.extend_lens)])

    @property
    def se(self):
        return sum([(p + e) * e for p, e in zip(self.prefix_lens, self.extend_lens)])


class LogItems:
    def __init__(self):
        self.log_items: List[LogItem] = []

    def append(self, prefix_lens, extend_lens, mla_cost, mha_cost):
        self.log_items.append(LogItem(prefix_lens, extend_lens, mla_cost, mha_cost))

    def update(self, prefix_lens, extend_lens, mla_cost, mha_cost):
        for item in self.log_items:
            if item.prefix_lens == prefix_lens and item.extend_lens == extend_lens:
                item.mla_cost = min(mla_cost, item.mla_cost)
                item.mha_cost = min(mha_cost, item.mha_cost)
                mla_cost = item.mla_cost
                mha_cost = item.mha_cost


class LogUtils:

    @staticmethod
    def sort_by_mla_cost(log_items: List[LogItem]):
        log_items.sort(key=lambda x: x.mla_cost)

    @staticmethod
    def sort_by_pe_cost(log_items: List[LogItem]):
        log_items.sort(key=lambda x: x.pe)

    @staticmethod
    def sort_by_se_cost(log_items: List[LogItem]):
        log_items.sort(key=lambda x: x.se)

    @staticmethod
    def print(log_items: List[LogItem]):
        for item in log_items:
            print(
                f"{item.prefix_lens=}, {item.extend_lens=}, {item.mla_cost=}, {item.mha_cost=}, {item.pe=}, {item.ee=}, {item.se=}",
                flush=True,
            )


def run_perf_with_cnt(
    attn_wrapper: DeepseekAttnWrapper,
    batch_size: int,
    prefix_len_sum: int,
    extend_len_sum: int,
):
    log_items = LogItems()
    sample_cnt = (500 if extend_len_sum <= 4096 else 100) if batch_size > 1 else 1
    opt_cost_cnt = 0
    for idx in range(sample_cnt):
        if idx == 0:
            # Calculate the minimum seq_extend_mul value for very small prefix lengths
            prefix_lens, extend_lens = generate_avg_seqs(
                batch_size, prefix_len_sum, extend_len_sum
            )
        else:
            prefix_lens, extend_lens = generate_random_seqs(
                batch_size, prefix_len_sum, extend_len_sum
            )
        assert sum(extend_lens) == extend_len_sum, f"{extend_lens=}"
        assert all([e > 0 for e in extend_lens]), f"{extend_lens=}"
        mla_cost, mha_cost = run_perf(attn_wrapper, prefix_lens, extend_lens)

        log_items.append(prefix_lens, extend_lens, mla_cost, mha_cost)
        if mla_cost > mha_cost:
            opt_cost_cnt += 1

    if opt_cost_cnt < sample_cnt * 0.05:
        print(
            f"prefix {prefix_len_sum}, extend {extend_len_sum}, mla > mha {opt_cost_cnt}/{sample_cnt} < 0.05, skip"
        )
        return None

    # rerun samples
    samples = log_items.log_items
    for sample in samples:
        mla_cost, mha_cost = run_perf(
            attn_wrapper, sample.prefix_lens, sample.extend_lens
        )
        log_items.update(sample.prefix_lens, sample.extend_lens, mla_cost, mha_cost)

    sel_se_samples = [s for s in samples if s.mla_cost >= s.mha_cost]
    if not sel_se_samples:
        print(
            f"prefix {prefix_len_sum}, extend {extend_len_sum}, not find mla >= mha sample, skip"
        )
        return None
    LogUtils.sort_by_se_cost(sel_se_samples)
    opt_se = sel_se_samples[0].se
    filter_samples = [s for s in samples if s.se >= opt_se]
    cal_opt = lambda se: sum(
        [s.mla_cost - s.mha_cost for s in filter_samples if s.se >= se]
        + [s.mha_cost - s.mla_cost for s in filter_samples if s.se < se]
    )
    max_opt = cal_opt(opt_se)
    for se_idx in range(1, len(sel_se_samples)):
        cur_se = sel_se_samples[se_idx].se
        cur_opt = cal_opt(cur_se)
        if cur_opt > max_opt:
            max_opt = cur_opt
            opt_se = cur_se
        if all([s.mla_cost >= s.mha_cost for s in filter_samples if s.se >= cur_se]):
            break

    filter_samples = [s for s in samples if s.se >= opt_se]
    if not filter_samples:
        return None
    mla_costs = [s.mla_cost for s in filter_samples]
    mha_costs = [s.mha_cost for s in filter_samples]

    mla_cost_avg = sum(mla_costs) / len(mla_costs)
    mha_cost_avg = sum(mha_costs) / len(mha_costs)
    if sample_cnt == 1:
        opt_se = 0
        c0 = (1, 1)
        c1 = (0, 0)
        break_extend = True
    else:
        greater_cnt = sum([mla >= mha for mla, mha in zip(mla_costs, mha_costs)])
        break_extend = greater_cnt >= sample_cnt * 0.95
        c0 = (greater_cnt, len(mla_costs))
        filter_samples = [s for s in samples if s.se < opt_se]
        if filter_samples:
            mla_costs = [s.mla_cost for s in filter_samples]
            mha_costs = [s.mha_cost for s in filter_samples]
            greater_cnt = sum([mla >= mha for mla, mha in zip(mla_costs, mha_costs)])
            c1 = (greater_cnt, len(mla_costs))
        else:
            c1 = (0, 0)
    return mla_cost_avg, mha_cost_avg, opt_se, c0, c1, break_extend


class BenchmarkWorker:

    def __init__(self, args) -> None:
        torch.set_default_device("cuda")
        torch.cuda.manual_seed_all(0)
        self.seed = args.seed

        device = torch.device("cuda")

        config = AutoConfig.from_pretrained(
            args.model, trust_remote_code=True, revision=None
        )
        with set_default_torch_dtype(torch.bfloat16):
            self.attn_wrapper = DeepseekAttnWrapper(args, config, device)

    def num_local_heads(self):
        return self.attn_wrapper.num_local_heads

    def tune(
        self,
        batch_size: int,
    ) -> Dict:
        rst_configs = {}
        rst_confidence = {}
        for prefix_len_sum in tqdm(
            [0, batch_size] + list(range(256, 8 * 1024 + 1, 256))
        ):
            for extend_len_sum in range(256, 8193, 256):
                rst = run_perf_with_cnt(
                    self.attn_wrapper, batch_size, prefix_len_sum, extend_len_sum
                )
                if rst is None:
                    continue
                mla_cost, mha_cost, cur_se, c0, c1, break_extend = rst
                if prefix_len_sum not in rst_configs:
                    rst_configs[prefix_len_sum] = {}
                    rst_confidence[prefix_len_sum] = {}
                rst_configs[prefix_len_sum][extend_len_sum] = cur_se
                rst_confidence[prefix_len_sum][extend_len_sum] = cur_se, c0, c1
                if break_extend:
                    break
        return rst_configs, rst_confidence


def save_configs(
    configs: Dict,
    attention_backend: str,
    num_local_heads: int,
) -> None:
    filename = get_config_file_name(attention_backend, num_local_heads)

    print(f"Writing best config to {filename}...")
    try:
        with open(filename, "r") as f:
            configs_org = json.load(f)
    except FileNotFoundError:
        configs_org = {}

    for batch_size, config in configs.items():
        configs_org[str(batch_size)] = config
        print(f"update configs: {batch_size}: {configs_org[str(batch_size)]}")

    with open(filename, "w") as f:
        json.dump(configs_org, f, indent=4)
        f.write("\n")


def print_result(confidence):
    print(f"bs, prefix, extend, se threshold, >=se threshold, <se threshold")
    for bs, prefix_dict in confidence.items():
        for prefix, extend_dict in prefix_dict.items():
            for extend, (se, c0, c1) in extend_dict.items():
                print(
                    f"{bs}, {prefix}, {extend}, {se}, {c0[0]}/{c0[1]}, {c1[0]}/{c1[1]}"
                )


def main(args: argparse.Namespace):
    print(args)
    if args.tune:
        if args.batch_sizes is None:
            args.batch_sizes = list(range(1, 9))
        print(
            f"Tuning configs for batch size {args.batch_sizes} and attention backend {args.attention_backend}"
        )
        ray.init()
        num_gpus = int(ray.available_resources()["GPU"])
        workers = [
            ray.remote(num_gpus=1)(BenchmarkWorker).remote(args)
            for _ in range(num_gpus)
        ]

        def _distribute(method: str, inputs: List[Any]) -> List[Any]:
            outputs = []
            worker_idx = 0
            for input_args in inputs:
                worker = workers[worker_idx]
                worker_method = getattr(worker, method)
                output = worker_method.remote(*input_args)
                outputs.append(output)
                worker_idx = (worker_idx + 1) % num_gpus
            return ray.get(outputs)

        num_local_heads = ray.get(workers[0].num_local_heads.remote())
        print(f"{num_local_heads=}, {args.batch_sizes=}", flush=True)
        configs = _distribute(
            "tune",
            [(batch_size,) for batch_size in args.batch_sizes],
        )
        best_configs = {M: config for M, (config, _) in zip(args.batch_sizes, configs)}
        print_result(
            {M: confidence for M, (_, confidence) in zip(args.batch_sizes, configs)}
        )
        save_configs(
            best_configs,
            args.attention_backend,
            num_local_heads,
        )
        return

    worker = BenchmarkWorker(args)
    if args.chunked_prefill_size is not None:
        print(
            f"Run config for chunked prefill size {args.chunked_prefill_size} and attention backend {args.attention_backend}, num local heads {worker.num_local_heads()}"
        )
        mla = 0
        mha = 0
        assert len(args.prefix_lens) == 1 and len(args.extend_lens) == 1
        prefix_len = sum(args.prefix_lens)
        extend_len = sum(args.extend_lens)
        cur_e = 0
        while cur_e < extend_len:
            e = min(args.chunked_prefill_size, extend_len - cur_e)
            mla_cost, mha_cost = run_perf(worker.attn_wrapper, [prefix_len], [e])
            mla += mla_cost
            mha += mha_cost
            cur_e += e
            prefix_len += e
        print(mla, mha)
    else:
        print(
            f"Run config for prefix lens {args.prefix_lens} and extend lens {args.extend_lens}, num local heads {worker.num_local_heads()}"
        )
        rst = run_perf(worker.attn_wrapper, args.prefix_lens, args.extend_lens)
        print(rst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1")
    parser.add_argument("--tp-size", "--tp", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-backend", type=str, required=True)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--prefix-lens",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--extend-lens",
        type=int,
        nargs="+",
    )
    args = parser.parse_args()

    main(args)
