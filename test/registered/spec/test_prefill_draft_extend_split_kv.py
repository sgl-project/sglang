"""The split prefill draft extend writes the same draft KV bytes as the single pass.

A one-layer DSA NextN draft with dummy weights runs in a spawned process on the real
KV pools and attention backend. Each case runs one prefill batch through the single
draft extend and then through the split, from the same cache bytes, with
batch-invariant GEMMs so that a pass's row count does not change the rounding.
"""

import copy
import json
import multiprocessing as mp
import os
import tempfile
import traceback
import unittest
from array import array
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

# DeepSeek-V3.2's attention and indexer shapes, with a small hidden size and MoE. The
# target architecture lets the server arguments resolve as for a DSA server.
_DSV32_DRAFT = {
    "architectures": ["DeepseekV32ForCausalLM"],
    "model_type": "deepseek_v32",
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "moe_intermediate_size": 256,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 0,
    "moe_layer_freq": 1,
    "num_hidden_layers": 1,
    "num_nextn_predict_layers": 1,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "q_lora_rank": 256,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 2048,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    },
    "rope_theta": 10000,
    "max_position_embeddings": 16384,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "vocab_size": 4096,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "torch_dtype": "bfloat16",
}

# GLM-5.3's layout: no RoPE in the latent KV, and index keys pooled four to one.
_GLM53_DRAFT = {
    **_DSV32_DRAFT,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_kpool": 4,
    "index_kpool_compress": True,
    "index_kpool_always_select_tail": True,
    "indexer_rope_interleave": True,
}

# Past the split's bounds: 4,096 KV-only tokens and a request longer than index_topk.
# With index keys pooled four to one, one last token closes a group, the other doesn't.
_EXTEND_LENS = (3004, 1518)
_UNWRITTEN = 0x7F


def _build_runner(model_dir: str):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import bootstrap
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(
        model_path=model_dir,
        load_format="dummy",
        skip_tokenizer_init=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        disable_radix_cache=True,
        max_total_tokens=32768,
        context_length=8192,
        mem_fraction_static=0.3,
        random_seed=0,
    )
    server_args.resolve_once()
    port_args = PortArgs.init_new(server_args)
    publish(server_args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_bf16_gemm_config()

    model_config = ModelConfig.from_server_args(server_args, is_draft_model=True)
    bootstrap.init_parallel_runtime(
        server_args=server_args, device="cuda", dist_port=port_args.nccl_port
    )
    bootstrap.init_layer_runtime(model_config=model_config)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.3,
        gpu_id=0,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    runner.alloc_memory_pool()
    runner.init_attention_backends()
    runner.init_cuda_graphs()
    return runner


def _draft_worker(runner):
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.draft_runner = runner
    worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
    worker.topk = 1
    worker.seed_dsa_topk_from_draft_extend = False
    worker.dsa_extend_topk_buf = None
    worker.can_split_prefill_draft_extend = worker._can_split_prefill_draft_extend()
    return worker


def _prefill_batch(runner, extend_lens):
    from sglang.benchmark.one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.runtime_context import get_schedule
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    generator = torch.Generator().manual_seed(0)
    vocab_size = runner.model_config.vocab_size
    reqs = []
    for rid, extend_len in enumerate(extend_lens):
        ids = torch.randint(2, vocab_size, (extend_len,), generator=generator)
        req = Req(
            rid=str(rid),
            origin_input_text="",
            origin_input_ids=array("q", ids.tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.full_untruncated_fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.extend_end = extend_len
        reqs.append(req)
    tree_cache = TreeCacheNamespace(
        page_size=get_schedule().page_size,
        device=runner.device,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
    )
    batch.prepare_for_extend()
    if batch.input_ids is None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(runner.device)
        batch.prefill_input_ids_cpu = None
    return batch


class _DraftKV:
    """The draft KV bytes a prefill batch writes: its latent KV rows, the index-key
    pages that hold its tokens, and the tail slots of each request's open pooling
    group. Rows and pages start as a byte no write produces, so an unwritten one is
    found; tails start empty, as a new request's do."""

    def __init__(self, runner, batch):
        pool = runner.token_to_kv_pool
        self.pool = pool
        self.layer_id = runner.model.model.decoder.self_attn.attn_mqa.layer_id
        self.locs = batch.out_cache_loc
        self.pages = torch.unique(self.locs // pool.page_size)
        self.reqs = batch.req_pool_indices
        self.has_tails = pool.index_kpool > 1
        if self.has_tails:
            # Later reads use only the open group's tail slots; a closed group's slots
            # may keep stale entries on either path.
            width = pool.get_compress_tail_buffers(self.layer_id)[0].shape[1]
            kpool = pool.index_kpool
            open_slots = [
                (req, pos % width)
                for req, seq_len in zip(self.reqs.tolist(), batch.seq_lens.tolist())
                for pos in range(seq_len - seq_len % kpool, seq_len)
            ]
            self.tail_reqs, self.tail_slots = torch.tensor(
                open_slots, device=self.locs.device
            ).unbind(1)

    def _views(self):
        pool = self.pool
        views = [
            pool.get_key_buffer(self.layer_id).view(torch.uint8)[self.locs],
            pool.get_index_k_with_scale_buffer(layer_id=self.layer_id).view(
                torch.uint8
            )[self.pages],
        ]
        if self.has_tails:
            for tail in pool.get_compress_tail_buffers(self.layer_id):
                views.append(tail.view(torch.uint8)[self.tail_reqs, self.tail_slots])
        return views

    def reset(self):
        pool = self.pool
        pool.get_key_buffer(self.layer_id).view(torch.uint8)[self.locs] = _UNWRITTEN
        pool.get_index_k_with_scale_buffer(layer_id=self.layer_id).view(torch.uint8)[
            self.pages
        ] = _UNWRITTEN
        if self.has_tails:
            for tail in pool.get_compress_tail_buffers(self.layer_id):
                tail.view(torch.uint8)[self.reqs] = 0

    def snapshot(self):
        torch.cuda.synchronize()
        return [view.clone() for view in self._views()]


def _compare_split(hf_config: dict) -> dict:
    from sglang.srt.batch_invariant_ops import set_batch_invariant_mode
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    with tempfile.TemporaryDirectory() as model_dir:
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(hf_config, f)
        runner = _build_runner(model_dir)
        # The draft resolves its server arguments as a DSA target does.
        assert isinstance(runner.attn_backend, DeepseekSparseAttnBackend)
        worker = _draft_worker(runner)
        assert worker.can_split_prefill_draft_extend
        batch = _prefill_batch(runner, _EXTEND_LENS)
        plan = runner.runner_ids.own_plan(batch)
        hidden_size = runner.model_config.hidden_size
        generator = torch.Generator(device=runner.device).manual_seed(0)
        target_hidden = torch.randn(
            batch.extend_num_tokens,
            hidden_size,
            generator=generator,
            device=runner.device,
            dtype=torch.bfloat16,
        )
        next_token_ids = batch.input_ids[torch.tensor(_EXTEND_LENS).cumsum(0) - 1]

        kv = _DraftKV(runner, batch)
        model = runner.model.model
        snapshots = []
        kv_only_calls = []
        with set_batch_invariant_mode(True), torch.no_grad():
            for split in (False, True):
                kv.reset()
                worker.can_split_prefill_draft_extend = split
                with mock.patch.object(
                    model, "forward_kv_only", wraps=model.forward_kv_only
                ) as kv_only:
                    worker._draft_extend_for_prefill(
                        copy.copy(batch),
                        target_hidden,
                        next_token_ids,
                        None,
                        iteration_ids=plan,
                    )
                kv_only_calls.append(kv_only.call_count)
                snapshots.append(kv.snapshot())
        single, split = snapshots
        return {
            "kv_only_calls": kv_only_calls,
            "differ": [
                int((a != b).flatten(1).any(dim=1).sum()) for a, b in zip(single, split)
            ],
            # Latent KV rows and index-key pages the single pass left untouched.
            "unwritten": [
                int((view == _UNWRITTEN).flatten(1).all(dim=1).sum())
                for view in single[:2]
            ],
        }


def _run_in_child(hf_config: dict, queue):
    try:
        queue.put(_compare_split(hf_config))
    except BaseException:
        queue.put({"error": traceback.format_exc()})


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestSplitPrefillDraftExtendKV(CustomTestCase):
    def _check(self, hf_config: dict):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        child = ctx.Process(target=_run_in_child, args=(hf_config, queue))
        child.start()
        result = queue.get(timeout=900)
        child.join(timeout=60)
        self.assertNotIn("error", result, result.get("error"))
        self.assertEqual(result["kv_only_calls"], [0, 1])
        self.assertEqual(result["unwritten"], [0, 0])
        self.assertEqual(result["differ"], [0] * len(result["differ"]))

    def test_rope_latent_kv_and_index_keys(self):
        self._check(_DSV32_DRAFT)

    def test_no_rope_latent_kv_and_pooled_index_keys(self):
        self._check(_GLM53_DRAFT)


if __name__ == "__main__":
    unittest.main()
