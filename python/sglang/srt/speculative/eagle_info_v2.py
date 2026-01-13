from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
)
from sglang.srt.utils.common import is_cuda, is_hip, is_npu, next_power_of_2

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

# Try to import CUDA kernels, but provide torch fallbacks
_sgl_kernel_available = False
top_k_renorm_prob = None
top_p_renorm_prob = None
tree_speculative_sampling_target_only = None

if _is_cuda or _is_hip:
    try:
        from sgl_kernel import top_k_renorm_prob as _sgl_top_k_renorm_prob
        from sgl_kernel import top_p_renorm_prob as _sgl_top_p_renorm_prob
        from sgl_kernel import (
            tree_speculative_sampling_target_only as _sgl_tree_spec_sampling,
        )

        # Verify that the torch ops are actually available
        _ = torch.ops.sgl_kernel.top_k_renorm_probs
        _ = torch.ops.sgl_kernel.top_p_renorm_probs
        _ = torch.ops.sgl_kernel.tree_speculative_sampling_target_only

        _sgl_kernel_available = True
        top_k_renorm_prob = _sgl_top_k_renorm_prob
        top_p_renorm_prob = _sgl_top_p_renorm_prob
        tree_speculative_sampling_target_only = _sgl_tree_spec_sampling
    except (ImportError, AttributeError) as e:
        logger.info(f"sgl_kernel not available, using torch fallbacks: {e}")

# Try to import aiter top_k_renorm_probs as fallback for top_k_renorm_prob
if _is_hip:
    if top_k_renorm_prob is None:
        try:
            import aiter.ops.sampling  # noqa: F401

            _aiter_ops = torch.ops.aiter

            def _to_tensor_scalar_tuple(data):
                """Helper to convert data to (tensor, scalar) tuple for aiter ops"""
                if isinstance(data, torch.Tensor):
                    if data.numel() == 1:
                        # Single element tensor: pass both tensor and scalar
                        return data, data.item()
                    else:
                        # Multi-element tensor: pass tensor, use -1 as placeholder for scalar
                        return data, -1
                else:
                    # Scalar value: pass None for tensor, value for scalar
                    return None, data

            def _aiter_top_k_renorm_wrapper(probs, top_ks):
                """Wrapper to use aiter's top_k_renorm_probs"""
                probs_for_ops = probs.float().contiguous()
                return _aiter_ops.top_k_renorm_probs(
                    probs_for_ops,
                    *_to_tensor_scalar_tuple(top_ks),
                )

            top_k_renorm_prob = _aiter_top_k_renorm_wrapper
            logger.info("Using AITER top_k_renorm_probs for top_k filtering")
        except (ImportError, AttributeError, RuntimeError) as e:
            # Fallback to torch implementation if aiter also not available
            logger.info(
                f"AITER not available ({e}), using torch fallback for top_k_renorm_prob"
            )

            def _torch_top_k_renorm_prob(probs, top_ks):
                """Torch native top-k renormalization"""
                if isinstance(top_ks, (int, float)):
                    top_ks = torch.tensor(
                        [top_ks], device=probs.device, dtype=torch.int32
                    ).expand(probs.shape[0])
                probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
                probs_sort[
                    torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
                    >= top_ks.view(-1, 1)
                ] = 0.0
                probs_sort = probs_sort / (probs_sort.sum(dim=-1, keepdim=True) + 1e-8)
                return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)

            top_k_renorm_prob = _torch_top_k_renorm_prob
            logger.info("Using torch native top_k_renorm_prob fallback")

    # Torch fallback for top_p_renorm_prob
    if top_p_renorm_prob is None:
        # TODO (hubert): add HIP implementation
        def _torch_top_p_renorm_prob(probs: torch.Tensor, top_ps: torch.Tensor):
            """Torch native top-p renormalization"""
            if isinstance(top_ps, (int, float)):
                top_ps = torch.tensor(
                    [top_ps], device=probs.device, dtype=torch.float32
                ).expand(probs.shape[0])
            probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
            probs_sort = probs_sort / (probs_sort.sum(dim=-1, keepdim=True) + 1e-8)
            return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)

        top_p_renorm_prob = _torch_top_p_renorm_prob
        logger.info("Using torch native top_p_renorm_prob fallback")

    # Torch fallback for tree_speculative_sampling_target_only
    if tree_speculative_sampling_target_only is None:
        # TODO (hubert): add HIP implementation
        def _torch_tree_speculative_sampling_target_only(
            predicts: torch.Tensor,  # mutable
            accept_index: torch.Tensor,  # mutable
            accept_token_num: torch.Tensor,  # mutable
            candidates: torch.Tensor,
            retrive_index: torch.Tensor,
            retrive_next_token: torch.Tensor,
            retrive_next_sibling: torch.Tensor,
            uniform_samples: torch.Tensor,
            uniform_samples_for_final_sampling: torch.Tensor,
            target_probs: torch.Tensor,
            draft_probs: torch.Tensor,
            threshold_single: float = 1.0,
            threshold_acc: float = 1.0,
            deterministic: bool = True,
        ):
            """
            Torch native implementation of tree speculative sampling.
            Based on the algorithm from sgl-kernel CUDA implementation.

            Traverses the tree using retrive_next_token and retrive_next_sibling,
            performs rejection sampling at each node, and records accepted tokens.
            """
            batch_size = candidates.shape[0]
            num_draft_tokens = candidates.shape[1]
            vocab_size = target_probs.shape[2]

            # Process each batch item
            for bx in range(batch_size):
                prob_acc = 0.0
                cur_token_idx = 0  # Current token index within the batch
                coin = uniform_samples[bx, 0].item()
                last_accepted_retrive_idx = retrive_index[bx, 0].item()
                accept_index[bx, 0] = last_accepted_retrive_idx
                num_accepted_tokens = 0
                cur_index = 0

                # Traverse tree depth-first
                for j in range(1, accept_index.shape[1]):
                    cur_index = retrive_next_token[bx, cur_index].item()

                    while cur_index != -1:
                        draft_index = retrive_index[bx, cur_index].item()
                        draft_token_id = candidates[bx, cur_index].item()
                        # Access target_probs using batch index and token index
                        target_prob_single = target_probs[
                            bx, cur_token_idx, draft_token_id
                        ].item()
                        prob_acc += target_prob_single

                        # Acceptance criterion
                        if (
                            coin <= prob_acc / threshold_acc
                            or target_prob_single >= threshold_single
                        ):
                            # Accept token
                            prob_acc = 0.0
                            cur_token_idx = cur_index
                            coin = uniform_samples[bx, cur_index].item()
                            predicts[last_accepted_retrive_idx] = draft_token_id
                            num_accepted_tokens += 1
                            accept_index[bx, num_accepted_tokens] = draft_index
                            last_accepted_retrive_idx = draft_index
                            break
                        else:
                            # Reject: update draft_probs and try sibling
                            draft_probs[bx, cur_token_idx, draft_token_id] = (
                                target_probs[bx, cur_token_idx, draft_token_id]
                            )
                            cur_index = retrive_next_sibling[bx, cur_index].item()

                    if cur_index == -1:
                        break

                # Final sampling if needed
                accept_token_num[bx] = num_accepted_tokens
                coin = uniform_samples_for_final_sampling[bx].item()

                # Sample from the residual distribution
                cur_probs = target_probs[bx, cur_token_idx].clone()
                cur_draft_probs = draft_probs[bx, cur_token_idx]
                residual = torch.clamp(cur_probs - cur_draft_probs, min=0.0)
                residual_sum = residual.sum().item()

                if residual_sum > 1e-8:
                    residual = residual / residual_sum
                    # Sample using the coin
                    cumsum = torch.cumsum(residual, dim=0)
                    sampled_token = torch.searchsorted(cumsum, coin).item()
                    sampled_token = min(sampled_token, vocab_size - 1)
                else:
                    # Fallback: sample from original distribution
                    sampled_token = torch.multinomial(
                        cur_probs + 1e-8, num_samples=1
                    ).item()

                predicts[last_accepted_retrive_idx] = sampled_token

        tree_speculative_sampling_target_only = (
            _torch_tree_speculative_sampling_target_only
        )
        logger.info("Using torch native tree_speculative_sampling_target_only fallback")


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@dataclass
class EagleDraftInputV2Mixin:
    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Now seq_lens is correct
        batch.maybe_wait_verify_done()

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0
        alloc_len_per_decode = get_alloc_len_per_decode()
        for r in batch.reqs:
            # Over-allocation happens here
            x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + x)
            num_needed_tokens += x
            r.kv_allocated_len += x
            r.decode_batch_idx += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)

            # Assign cache locations
            batch.out_cache_loc = torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=batch.input_ids.device,
            )
            # FIXME(lsyin): align with the default code path
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )

        # Get a forward batch
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ModelWorkerBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        seq_lens_cpu_ = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        batch.extend_seq_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.extend_prefix_lens = seq_lens_cpu_.tolist()
        batch.extend_num_tokens = extend_num_tokens
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            accept_length = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            accept_index = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            return predict, accept_length, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            draft_probs = torch.zeros_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # Include the bonus token
        accept_length.add_(1)
        return predict, accept_length, accept_index


@triton.jit
def fill_new_verified_id(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    accept_length = tl.load(accept_lens + pid)

    verified_id_idx = num_draft_tokens * pid + accept_length - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    if _is_cuda or _is_hip:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc
