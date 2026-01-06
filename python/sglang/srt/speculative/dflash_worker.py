import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_draft_model import load_dflash_draft_model
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp=1/pp=1)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self._mask_token_id = self._resolve_mask_token_id()
        self._warned_forced_greedy = False
        self._logged_first_verify = False

        # Load the DFlash draft model (weights are separate from the target model).
        # This is kept as a standalone module (not a TpModelWorker) since its forward
        # is non-causal and differs from standard decoder-only models.
        draft_device = torch.device(target_worker.device)
        draft_dtype = target_worker.model_runner.dtype
        self.draft_model, self.draft_config = load_dflash_draft_model(
            server_args.speculative_draft_model_path,
            device=draft_device,
            dtype=draft_dtype,
        )
        self.block_size = int(getattr(self.draft_config, "block_size", 16))
        if self.tp_rank == 0:
            logger.info(
                "Loaded DFLASH draft model. path=%s, dtype=%s, device=%s, block_size=%s, num_hidden_layers=%s, mask_token_id=%s",
                server_args.speculative_draft_model_path,
                draft_dtype,
                draft_device,
                self.block_size,
                getattr(self.draft_config, "num_hidden_layers", None),
                self._mask_token_id,
            )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # No draft-side pools to clear in the stub implementation.
        return

    def _resolve_mask_token_id(self) -> int:
        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("DFLASH requires tokenizer initialization (skip_tokenizer_init is not supported).")

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            # `convert_tokens_to_ids` can return `None` (or an unk id) depending on tokenizer.
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            mask_token_id = vocab.get("<|MASK|>", None)

        if mask_token_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = tokenizer.convert_tokens_to_ids("<|MASK|>")

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    "<|MASK|>",
                    mask_token_id,
                    len(tokenizer),
                    vocab_size,
                )

        if mask_token_id is None or int(mask_token_id) < 0:
            raise ValueError("DFLASH requires a `<|MASK|>` token id, but it could not be resolved.")

        if mask_token_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={mask_token_id}, vocab_size={vocab_size}. "
                "This likely means `<|MASK|>` requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(mask_token_id)

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch, draft_input: DFlashDraftInput):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise ValueError("DFLASH does not support grammar-constrained decoding yet.")
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if not self._warned_forced_greedy and self.tp_rank == 0:
                logger.warning(
                    "DFLASH currently supports greedy verification only; "
                    "ignoring non-greedy sampling params (e.g. temperature/top_p/top_k) and using argmax."
                )
                self._warned_forced_greedy = True

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError("DFLASH draft state missing target_hidden context features.")
        if len(draft_input.ctx_lens_cpu) != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens_cpu length mismatch: got {len(draft_input.ctx_lens_cpu)} for bs={bs}."
            )

        embed_weight, head_weight = self.target_worker.model_runner.model.get_embed_and_head()

        # Slice ragged target_hidden on CPU for simplicity.
        offsets: List[int] = [0]
        for ln in draft_input.ctx_lens_cpu:
            offsets.append(offsets[-1] + int(ln))

        candidates: List[torch.Tensor] = []
        for i, (req, ctx_len) in enumerate(zip(batch.reqs, draft_input.ctx_lens_cpu, strict=True)):
            start_pos = int(batch.seq_lens_cpu[i].item())
            cache = draft_input.draft_caches[i]
            cache_len = int(cache.get_seq_length())

            if cache_len + int(ctx_len) != start_pos:
                raise RuntimeError(
                    "DFLASH draft cache length mismatch. "
                    f"{cache_len=} + {ctx_len=} != {start_pos=}. "
                    "This can happen if prefix caching is enabled; start with `--disable-radix-cache` for now."
                )

            target_hidden = draft_input.target_hidden[offsets[i] : offsets[i + 1]]
            target_hidden = target_hidden.unsqueeze(0)  # [1, ctx_len, feat]

            block_ids = torch.full(
                (1, self.block_size),
                self._mask_token_id,
                dtype=torch.long,
                device=device,
            )
            block_ids[0, 0] = draft_input.verified_id[i].to(torch.long)

            noise_embedding = F.embedding(block_ids, embed_weight)
            position_ids = torch.arange(
                cache_len,
                start_pos + self.block_size,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            with torch.inference_mode():
                hidden = self.draft_model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache.crop(start_pos)

                draft_hidden = hidden[:, -self.block_size + 1 :, :]
                draft_logits = F.linear(draft_hidden, head_weight)
                draft_tokens = torch.argmax(draft_logits, dim=-1).to(torch.long)

            candidate = torch.cat(
                [block_ids[0, 0].view(1), draft_tokens.view(-1)],
                dim=0,
            )
            candidates.append(candidate)

        draft_tokens = torch.stack(candidates, dim=0)  # [bs, block_size]
        positions = (
            batch.seq_lens.to(torch.long).unsqueeze(1)
            + torch.arange(self.block_size, device=device, dtype=torch.long)[None, :]
        ).flatten()

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.flatten(),
            positions=positions,
            draft_token_num=self.block_size,
        )
        verify_input.prepare_for_verify(batch, self.page_size)

        batch.forward_mode = ForwardMode.TARGET_VERIFY if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError("DFLASH speculative decoding does not support return_logprob yet.")

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if any(len(req.prefix_indices) > 0 for req in batch.reqs):
                raise ValueError(
                    "DFLASH currently does not support radix/prefix cache hits (prefix_indices != 0). "
                    "Start with `--disable-radix-cache` for now."
                )

            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            draft_caches = [self.draft_model.make_cache() for _ in batch.reqs]
            ctx_lens_cpu = model_worker_batch.seq_lens_cpu.tolist()

            batch.spec_info = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens_cpu=ctx_lens_cpu,
                draft_caches=draft_caches,
            )

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )

        # Update draft state for the next iteration.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens_cpu = commit_lens.cpu().tolist()
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
