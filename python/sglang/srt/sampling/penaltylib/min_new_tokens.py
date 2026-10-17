import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils.common import is_pin_memory_available


class BatchedMinNewTokensPenalizer(_BatchedPenalizer):
    """
    Min new tokens penalizer penalizes tokens based on the length of the output.
    """

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.min_new_tokens > 0 for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.min_new_tokens = (
            torch.tensor(
                data=[
                    req.sampling_params.min_new_tokens
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.int32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

        padded_stop_token_ids_cpu = torch.nn.utils.rnn.pad_sequence(
            sequences=[
                torch.tensor(
                    data=[
                        token_id
                        for token_id in (
                            (req.sampling_params.stop_token_ids or set())
                            | (req.eos_token_ids or set())
                            | (req.tokenizer.additional_stop_token_ids or set())
                            | (
                                {req.tokenizer.eos_token_id}
                                if req.tokenizer.eos_token_id is not None
                                else set()
                            )
                        )
                        if token_id is not None
                    ],
                    dtype=torch.int64,
                )
                for req in self.orchestrator.reqs()
            ],
            batch_first=True,
            padding_value=self.orchestrator.vocab_size,
        )
        # The sparse CUDA update uses direct addresses. Reject invalid IDs on
        # the CPU before transfer, as the dense scatter did before this change.
        if torch.any(
            (padded_stop_token_ids_cpu < 0)
            | (padded_stop_token_ids_cpu > self.orchestrator.vocab_size)
        ):
            raise RuntimeError("Stop token ID is out of vocabulary range")
        if pin_memory:
            padded_stop_token_ids_cpu = padded_stop_token_ids_cpu.pin_memory()
        padded_stop_token_ids = padded_stop_token_ids_cpu.to(
            self.orchestrator.device, non_blocking=True
        )
        # Keep only stop-token entries, rather than a batch x vocabulary mask.
        # Padding scatters a zero to token 0, so it never suppresses that token.
        padding = padded_stop_token_ids == self.orchestrator.vocab_size
        self.stop_token_ids = padded_stop_token_ids.masked_fill(padding, 0)
        self.stop_token_penalties = torch.full_like(
            padded_stop_token_ids, float("-inf"), dtype=torch.float32
        ).masked_fill_(padding, 0.0)

        self.len_output_tokens = torch.zeros(
            size=(len(self.orchestrator.reqs()), 1),
            dtype=torch.int32,
            device=self.orchestrator.device,
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        self.len_output_tokens += 1

    def _apply(self, logits: torch.Tensor):
        if logits.is_cuda:
            from sglang.kernels.ops.sampling.min_new_tokens import (
                apply_min_new_tokens_penalties,
            )

            apply_min_new_tokens_penalties(
                logits,
                self.stop_token_ids,
                self.stop_token_penalties,
                self.len_output_tokens,
                self.min_new_tokens,
            )
        else:
            mask = self.len_output_tokens < self.min_new_tokens
            penalties = torch.where(mask, self.stop_token_penalties, 0.0)
            logits.scatter_add_(1, self.stop_token_ids, penalties.to(logits.dtype))

    def _filter(self, keep_indices: torch.Tensor):
        self.min_new_tokens = self.min_new_tokens[keep_indices]
        self.stop_token_ids = self.stop_token_ids[keep_indices]
        self.stop_token_penalties = self.stop_token_penalties[keep_indices]
        self.len_output_tokens = self.len_output_tokens[keep_indices]

    def _merge(self, their: "BatchedMinNewTokensPenalizer"):
        self.min_new_tokens = torch.cat(
            [self.min_new_tokens, their.min_new_tokens], dim=0
        )
        # Independently prepared batches can have different stop-list widths.
        width = max(self.stop_token_ids.shape[1], their.stop_token_ids.shape[1])
        for name in ("stop_token_ids", "stop_token_penalties"):
            ours, theirs = getattr(self, name), getattr(their, name)
            setattr(
                self,
                name,
                torch.cat(
                    [
                        torch.nn.functional.pad(ours, (0, width - ours.shape[1])),
                        torch.nn.functional.pad(theirs, (0, width - theirs.shape[1])),
                    ],
                    dim=0,
                ),
            )
        self.len_output_tokens = torch.cat(
            [self.len_output_tokens, their.len_output_tokens], dim=0
        )

    # Explicit resource cleanup to aid GC and free CUDA memory promptly
    def _teardown(self) -> None:
        for name in (
            "min_new_tokens",
            "stop_token_ids",
            "stop_token_penalties",
            "len_output_tokens",
        ):
            if hasattr(self, name):
                delattr(self, name)
