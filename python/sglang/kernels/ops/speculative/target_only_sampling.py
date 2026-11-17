# Copyright (c) 2025 by SGLang team.
# Copyright (c) 2024-2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-greedy tree verify on ROCm: Triton port of the CUDA
``tree_speculative_sampling_target_only`` (sgl_kernel speculative/sampling.cuh)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _target_only_residual(
    TargetProbs,
    DraftProbs,
    Candidates,
    RetriveNextSibling,
    tree_base,
    row_off,
    first_rejected,
    use_draft,
    offs,
    vocab_size,
):
    mask = offs < vocab_size
    q = tl.load(TargetProbs + row_off + offs, mask=mask, other=0.0)
    # No draft row past the last speculative slot: the bonus samples q itself.
    p = tl.load(DraftProbs + row_off + offs, mask=mask & use_draft, other=0.0)
    val = tl.maximum(q - p, 0.0)
    # Verification set p = q at every rejected child of the last accepted node.
    # Those writes come from other threads of this program, so zero the tokens
    # here instead of reading the stores back.
    child = first_rejected
    while child != -1:
        token = tl.load(Candidates + tree_base + child)
        val = tl.where(offs == token, 0.0, val)
        child = tl.load(RetriveNextSibling + tree_base + child)
    return val


@triton.jit
def tree_speculative_sampling_target_only_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    RetriveNextToken,
    RetriveNextSibling,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    threshold_single,
    threshold_acc,
    num_spec_tokens,
    num_draft_tokens,
    vocab_size,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tree_base = pid * num_draft_tokens
    accept_base = pid * num_spec_tokens

    coin = tl.load(UniformSamples + tree_base)
    last_accepted = tl.load(RetriveIndex + tree_base)
    tl.store(AcceptIndex + accept_base, last_accepted.to(tl.int32))

    prob_acc = tl.zeros((), dtype=tl.float32)
    num_accepted = tl.zeros((), dtype=tl.int32)
    # Local index of the last accepted node: its target row scores its children.
    cur_row = tl.zeros((), dtype=tl.int64)
    cur_index = tl.zeros((), dtype=tl.int64)
    step = 1
    exhausted = 0
    while (step < num_spec_tokens) and (exhausted == 0):
        cur_index = tl.load(RetriveNextToken + tree_base + cur_index)
        row_off = (tree_base + cur_row) * vocab_size
        accepted = 0
        while (cur_index != -1) and (accepted == 0):
            draft_index = tl.load(RetriveIndex + tree_base + cur_index)
            draft_token = tl.load(Candidates + tree_base + cur_index)
            target_prob = tl.load(TargetProbs + row_off + draft_token)
            prob_acc += target_prob
            if (target_prob > 0.0) and (
                (coin < tl.div_rn(prob_acc, threshold_acc))
                or (target_prob >= threshold_single)
            ):
                prob_acc = tl.zeros((), dtype=tl.float32)
                cur_row = cur_index
                coin = tl.load(UniformSamples + tree_base + cur_index)
                tl.store(Predicts + last_accepted, draft_token.to(tl.int32))
                num_accepted += 1
                tl.store(
                    AcceptIndex + accept_base + num_accepted,
                    draft_index.to(tl.int32),
                )
                last_accepted = draft_index
                accepted = 1
            else:
                tl.store(DraftProbs + row_off + draft_token, target_prob)
                cur_index = tl.load(RetriveNextSibling + tree_base + cur_index)
        exhausted = (cur_index == -1).to(tl.int32)
        step += 1
    tl.store(AcceptTokenNum + pid, num_accepted)

    # Sample the final token from relu(q - p) at the last accepted node, by the
    # CDF inversion of flashinfer's DeviceSamplingFromProb.
    use_draft = num_accepted != num_spec_tokens - 1
    row_off = (tree_base + cur_row) * vocab_size
    # Short of the last slot, the walk stopped because every child was rejected.
    first_rejected = tl.load(RetriveNextToken + tree_base + cur_row)
    first_rejected = tl.where(use_draft, first_rejected, -1)

    norm = tl.zeros((), dtype=tl.float32)
    for v_start in range(0, vocab_size, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        val = _target_only_residual(
            TargetProbs,
            DraftProbs,
            Candidates,
            RetriveNextSibling,
            tree_base,
            row_off,
            first_rejected,
            use_draft,
            offs,
            vocab_size,
        )
        norm += tl.sum(val, axis=0)
    u = tl.load(UniformSamplesFinal + pid) * norm

    sampled_id = vocab_size + tl.zeros((), dtype=tl.int32)
    last_valid_id = tl.full((), -1, dtype=tl.int32)
    aggregate = tl.zeros((), dtype=tl.float32)
    stop = 0
    for v_start in range(0, vocab_size, BLOCK_V):
        if stop == 0:
            offs = v_start + tl.arange(0, BLOCK_V)
            val = _target_only_residual(
                TargetProbs,
                DraftProbs,
                Candidates,
                RetriveNextSibling,
                tree_base,
                row_off,
                first_rejected,
                use_draft,
                offs,
                vocab_size,
            )
            valid = val > 0.0
            chunk_sum = tl.sum(val, axis=0)
            if aggregate + chunk_sum > u:
                above = ((aggregate + tl.cumsum(val, axis=0)) > u) & valid
                if tl.max(above.to(tl.int32), axis=0) > 0:
                    sampled_id = v_start + tl.argmax(above.to(tl.int32), axis=0)
            last_valid_id = tl.maximum(
                last_valid_id, tl.max(tl.where(valid, offs, -1), axis=0)
            )
            aggregate += chunk_sum
            if aggregate > u:
                stop = 1
    # u can sit at or past the summed mass when the coin is close to 1 or the
    # residual is empty: fall back to the last positive token, else the last id.
    if sampled_id == vocab_size:
        sampled_id = tl.where(last_valid_id == -1, vocab_size - 1, last_valid_id)
    tl.store(Predicts + last_accepted, sampled_id)


def _check_target_only_operand(name, tensor, shape, dtype, device):
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValueError(
            f"{name}: expected shape {shape} and dtype {dtype}, got "
            f"{tuple(tensor.shape)} and {tensor.dtype}"
        )
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def tree_speculative_sampling_target_only_triton(
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
    draft_probs: torch.Tensor,  # mutable
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,  # not used: Triton reductions are deterministic
) -> None:
    """Triton port of the CUDA ``tree_speculative_sampling_target_only``
    (``jit/include/sgl_kernel/speculative/sampling.cuh``) with the same operand
    contract and results; only fp32 summation order differs. It launches one
    program per request and neither allocates nor syncs, so CUDA graphs can
    capture it."""
    bs, num_draft_tokens = candidates.shape
    num_spec_tokens = accept_index.shape[-1]
    vocab_size = target_probs.shape[-1]
    tree = (bs, num_draft_tokens)
    rows = (bs, num_draft_tokens, vocab_size)
    for name, tensor, shape, dtype in (
        ("candidates", candidates, tree, torch.int64),
        ("retrive_index", retrive_index, tree, torch.int64),
        ("retrive_next_token", retrive_next_token, tree, torch.int64),
        ("retrive_next_sibling", retrive_next_sibling, tree, torch.int64),
        ("accept_index", accept_index, (bs, num_spec_tokens), torch.int32),
        ("accept_token_num", accept_token_num, (bs,), torch.int32),
        ("predicts", predicts, (bs * num_draft_tokens,), torch.int32),
        ("uniform_samples", uniform_samples, tree, torch.float32),
        (
            "uniform_samples_for_final_sampling",
            uniform_samples_for_final_sampling,
            (bs,),
            torch.float32,
        ),
        ("target_probs", target_probs, rows, torch.float32),
        ("draft_probs", draft_probs, rows, torch.float32),
    ):
        _check_target_only_operand(name, tensor, shape, dtype, candidates.device)
    if num_draft_tokens <= 0 or num_spec_tokens <= 0 or vocab_size <= 0:
        raise ValueError("draft, speculative and vocab sizes must be positive")
    if bs == 0:
        return

    tree_speculative_sampling_target_only_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        float(threshold_single),
        max(float(threshold_acc), 1e-9),
        num_spec_tokens,
        num_draft_tokens,
        vocab_size,
        BLOCK_V=4096,
        num_warps=4,
    )
