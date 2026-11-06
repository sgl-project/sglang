from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, List, Optional

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs


class HierarchyBlock(DllmAlgorithm):
    """Fast-dLLM v2 hierarchical block decoding.

    Sub-blocks are denoised left to right with bidirectional attention inside
    the block. Every block then gets one causal forward, so the KV left in the
    cache (and inserted into the radix tree) is what an autoregressive pass over
    the same tokens would write. The last position of that causal forward
    predicts the first token of the request's next block; that token is the
    carried per-request state. The first generated token after a prompt tail
    is also predicted causally, so every block starts from an autoregressive
    token as in the reference implementation.
    """

    supported_architectures = ("Fast_dVLMForConditionalGeneration",)
    carries_block_state = True

    @classmethod
    def configure_server_args(cls, server_args: ServerArgs) -> None:
        cfg = resolving_view(server_args)
        if cfg.dllm_fdfo:
            raise ValueError("HierarchyBlock does not support --dllm-fdfo")
        # Denoise and the final KV write use different block masks; a graph
        # captured with one mask cannot replay the other.
        graph_config = with_phase(
            cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
        )
        declare_resolution(server_args, cls.__name__, cuda_graph_config=graph_config)

    @classmethod
    def validate_request(cls, req: Req) -> Optional[str]:
        if req.multimodal_inputs is not None:
            return "HierarchyBlock does not support image or video inputs yet"
        return None

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
        start_list = self._block_start_list(forward_batch)
        if algo_states is not None:
            self._seed_inherited_tokens(forward_batch, algo_states)
        self._seed_after_prompt_tail(model_runner, forward_batch, start_list)

        model = model_runner.model
        if self._denoise_mask(forward_batch).any():
            with _bidirectional_block_attention(model):
                self._denoise(model_runner, forward_batch)

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        full_logits = out.logits_output.full_logits
        vocab_size = full_logits.shape[-1]
        last_logits = full_logits.view(batch_size, self.block_size, vocab_size)[:, -1]
        inherited_tokens = last_logits.argmax(dim=-1).tolist()

        if all(start == self.block_size for start in start_list):
            return out.logits_output, [], None, inherited_tokens, out.can_run_graph

        block_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        next_token_ids_list = [block_ids[i, start_list[i] :] for i in range(batch_size)]
        return (
            out.logits_output,
            next_token_ids_list,
            None,
            inherited_tokens,
            out.can_run_graph,
        )

    def _denoise(self, model_runner: ModelRunner, forward_batch: ForwardBatch) -> None:
        # Each step unmasks at least one token of every request that still has
        # masks, so block_size steps always suffice.
        for _ in range(self.block_size):
            if not self._denoise_mask(forward_batch).any():
                return
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self._unmask_step(forward_batch, out.logits_output.full_logits)

    def _denoise_mask(self, forward_batch: ForwardBatch) -> torch.Tensor:
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        is_mask = input_ids == self.mask_id
        if forward_batch.dllm_prompt_mask is not None:
            is_mask = is_mask & ~forward_batch.dllm_prompt_mask
        return is_mask

    def _seed_inherited_tokens(
        self, forward_batch: ForwardBatch, algo_states: List[Optional[int]]
    ) -> None:
        first_is_mask = self._denoise_mask(forward_batch)[:, 0].tolist()
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        for i, inherited in enumerate(algo_states):
            if first_is_mask[i] and inherited is not None:
                input_ids[i, 0] = inherited

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
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        full_logits = out.logits_output.full_logits
        logits = full_logits.view(forward_batch.batch_size, self.block_size, -1)
        row_ids = torch.tensor(rows, device=logits.device)
        starts = torch.tensor([start_list[i] for i in rows], device=logits.device)
        preds = logits[row_ids, starts - 1].argmax(dim=-1)
        input_ids = forward_batch.input_ids.view(-1, self.block_size)
        input_ids[row_ids, starts] = preds.to(input_ids.dtype)

    def _current_sub_block(self, is_mask: torch.Tensor) -> torch.Tensor:
        batch_size = is_mask.shape[0]
        num_sub_blocks = self.block_size // self.sub_block_size
        sub_blocks = is_mask.view(batch_size, num_sub_blocks, self.sub_block_size)
        first_open = sub_blocks.any(dim=-1).int().argmax(dim=-1)
        sub_ids = torch.arange(num_sub_blocks, device=is_mask.device)
        selected = (sub_ids[None, :] == first_open[:, None])[:, :, None]
        return (sub_blocks & selected).view(batch_size, self.block_size)

    def _unmask_step(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
    ) -> None:
        batch_size = forward_batch.batch_size
        logits = full_logits.view(batch_size, self.block_size, -1)
        # Position i is predicted from the logits at position i - 1.
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        candidates = self._current_sub_block(self._denoise_mask(forward_batch))
        # A predicted mask token would leave its position masked and stall the
        # step, so it falls back to the runner-up token.
        top2 = logits.topk(2, dim=-1).indices
        preds = torch.where(top2[..., 0] == self.mask_id, top2[..., 1], top2[..., 0])
        probs = torch.softmax(logits.float(), dim=-1)
        confidence = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(candidates, confidence, -float("inf"))

        transfer = confidence > self.threshold
        top1 = torch.nn.functional.one_hot(
            confidence.argmax(dim=-1), num_classes=self.block_size
        ).bool()
        transfer = torch.where(transfer.any(dim=-1, keepdim=True), transfer, top1)
        transfer = transfer & candidates

        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        new_input_ids = torch.where(transfer, preds, input_ids)
        forward_batch.input_ids.copy_(new_input_ids.view(-1))


@contextmanager
def _bidirectional_block_attention(model: torch.nn.Module) -> Iterator[None]:
    model.set_block_attention_causal(False)
    try:
        yield
    finally:
        model.set_block_attention_causal(True)


Algorithm = HierarchyBlock
