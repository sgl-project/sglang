from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class HierarchyBlock(DllmAlgorithm):
    """Fast-dLLM v2 hierarchical block decoding with token inheritance.

    Each step denoises only the first sub-block that still has masks, with
    logits shifted one position as in the AR checkpoint. A block that starts
    with a mask takes its first token from the last-position prediction of the
    request's previous block, carried per request in ``dllm_algo_state``.

    Fast-dVLM denoises with bidirectional attention inside the block, which is
    the model's default and what its CUDA graphs capture, and writes a resolved
    block's KV with one eager causal forward. That forward also predicts the
    inherited token. The first token after a prompt tail is predicted causally.
    """

    supported_architectures = ("Fast_dVLMForConditionalGeneration",)
    inherits_next_token = True
    # Backends that read each layer's attn_type on every forward, which the
    # switch between denoising and the final causal forward relies on.
    supported_attention_backends = ("flashinfer", "triton", "fa3")
    required_attention_backend = "flashinfer"

    @classmethod
    def configure_server_args(cls, server_args: ServerArgs) -> None:
        cfg = resolving_view(server_args)
        if cfg.device != "cuda":
            raise ValueError("HierarchyBlock supports CUDA GPU execution only")
        # The checkpoint's vision sub-config would make it multimodal, but the
        # dLLM path does not handle image inputs yet.
        if cfg.enable_multimodal:
            raise ValueError("HierarchyBlock does not support multimodal inputs yet")
        # Decode-phase graphs are captured with the model's bidirectional block
        # attention and replay the denoise forwards; the causal forwards run
        # eagerly. Prefill-phase graphs would otherwise pick those up.
        graph_config = with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        )
        # FDFO is on by default, but each block here is decoded in one run.
        declare_resolution(
            server_args,
            cls.__name__,
            cuda_graph_config=graph_config,
            enable_multimodal=False,
            dllm_fdfo=False,
        )

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        algorithm_config = config.algorithm_config
        self.threshold = algorithm_config.get("threshold", 0.9)
        self.sub_block_size = algorithm_config.get("sub_block_size", 8)
        if self.block_size % self.sub_block_size != 0:
            raise ValueError(
                f"sub_block_size {self.sub_block_size} must divide "
                f"block_size {self.block_size}"
            )
        # Seeding and inheritance read the logits at position i as the
        # prediction for token i + 1, the only layout Fast-dVLM is trained with.
        token_shift = algorithm_config.get("token_shift", 1)
        if token_shift != 1:
            raise ValueError(
                f"HierarchyBlock requires token_shift 1, got {token_shift}"
            )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Optional[int]]] = None,
    ) -> DllmRunOutput:
        batch_size = forward_batch.batch_size
        if batch_size == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], None, [], out.can_run_graph
        start_list = self._block_start_list(forward_batch)
        self._inherit_first_token(forward_batch, algo_states)
        self._seed_after_prompt_tail(model_runner, forward_batch, start_list)

        out = self._forward(model_runner, forward_batch)
        has_masks = any(start < self.block_size for start in start_list)
        if has_masks:
            for _ in range(self.max_steps(self.block_size)):
                done = self.step(forward_batch, out.logits_output.full_logits, None)
                if all(done):
                    break
                out = self._forward(model_runner, forward_batch)

        # The last forward saw every block fully resolved, so its final row
        # predicts the first token of each request's next block.
        full_logits = out.logits_output.full_logits
        next_states = (
            full_logits.view(batch_size, self.block_size, -1)[:, -1]
            .argmax(dim=-1)
            .tolist()
        )

        next_token_ids_list = []
        if has_masks:
            next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size)
            next_token_ids_list = [
                next_token_ids[i, start_list[i] :] for i in range(batch_size)
            ]
        return (
            out.logits_output,
            next_token_ids_list,
            None,
            next_states,
            out.can_run_graph,
        )

    def _forward(self, model_runner: ModelRunner, forward_batch: ForwardBatch):
        # A forward over fully resolved blocks writes the KV that stays cached.
        if self._denoise_mask(forward_batch).any():
            return model_runner.forward(forward_batch, pp_proxy_tensors=None)
        return _causal_forward(model_runner, forward_batch)

    def _denoise_mask(self, forward_batch: ForwardBatch) -> torch.Tensor:
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        masked = input_ids == self.mask_id
        if forward_batch.dllm_prompt_mask is not None:
            masked &= ~forward_batch.dllm_prompt_mask
        return masked

    def _inherit_first_token(
        self,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Optional[int]]],
    ) -> None:
        if not algo_states or all(state is None for state in algo_states):
            return
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        inherited = torch.tensor(
            [-1 if state is None else state for state in algo_states],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        first = input_ids[:, 0]
        use = self._denoise_mask(forward_batch)[:, 0] & (inherited >= 0)
        # In place to keep the input_ids tensor identity (CUDA graph safe).
        first.copy_(torch.where(use, inherited, first))

    def _seed_after_prompt_tail(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        start_list: List[int],
    ) -> None:
        # Blocks are aligned to absolute positions, so the first generated
        # block can start with committed prompt tokens.
        rows = [i for i, start in enumerate(start_list) if 0 < start < self.block_size]
        if not rows:
            return
        out = _causal_forward(model_runner, forward_batch)
        full_logits = out.logits_output.full_logits
        logits = full_logits.view(forward_batch.batch_size, self.block_size, -1)
        row_ids = torch.tensor(rows, device=logits.device)
        starts = torch.tensor([start_list[i] for i in rows], device=logits.device)
        preds = logits[row_ids, starts - 1].argmax(dim=-1)
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        input_ids[row_ids, starts] = preds.to(input_ids.dtype)

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        batch_size = forward_batch.batch_size
        vocab_size = full_logits.shape[-1]
        logits = full_logits.view(batch_size, self.block_size, vocab_size)
        # Position j is predicted by the logits at j - 1: [L0, L0, ..., LN-2].
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        masked = self._denoise_mask(forward_batch)
        done = ~masked.any(dim=1)

        # Only the first sub-block that still has masks is eligible.
        sub_has_mask = masked.view(batch_size, -1, self.sub_block_size).any(dim=2)
        active_sub = sub_has_mask.int().argmax(dim=1)
        sub_index = (
            torch.arange(self.block_size, device=input_ids.device)
            // self.sub_block_size
        )
        candidate = masked & (sub_index.unsqueeze(0) == active_sub.unsqueeze(1))

        # argmax keeps the first of exactly tied logits, as the reference does.
        # A predicted mask token would leave its position masked and stall the
        # row, so it falls back to the best other token.
        preds = torch.argmax(logits, dim=-1)
        top2 = logits.topk(2, dim=-1).indices
        runner_up = torch.where(
            top2[..., 0] == self.mask_id, top2[..., 1], top2[..., 0]
        )
        preds = torch.where(preds == self.mask_id, runner_up, preds)
        probs = torch.softmax(logits.float(), dim=-1)
        confidence = torch.gather(probs, dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(candidate, confidence, -float("inf"))

        # Accept tokens above the threshold, or fall back to the single most
        # confident candidate.
        transfer = confidence > self.threshold
        has_transfer = transfer.any(dim=1, keepdim=True)
        top1 = torch.zeros_like(transfer)
        top1.scatter_(1, confidence.argmax(dim=1, keepdim=True), True)
        transfer = torch.where(has_transfer, transfer, top1) & candidate

        new_input_ids = torch.where(transfer, preds, input_ids)
        forward_batch.input_ids.copy_(new_input_ids.view(-1))

        return done.tolist()


def _causal_forward(model_runner: ModelRunner, forward_batch: ForwardBatch):
    # The model's default block attention is bidirectional, which is what its
    # CUDA graphs capture, so a causal forward has to run eagerly.
    model = model_runner.model
    model.set_block_attention_causal(True)
    forward_batch.dllm_eager = True
    try:
        return model_runner.forward(forward_batch, pp_proxy_tensors=None)
    finally:
        forward_batch.dllm_eager = False
        model.set_block_attention_causal(False)


Algorithm = HierarchyBlock
