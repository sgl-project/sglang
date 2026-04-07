import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import (
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import maybe_detect_nan
from sglang.srt.speculative.spectre.spectre_protocol import (
    is_health_check_req as _is_health_check,
)

logger = logging.getLogger(__name__)

_DEFAULT_DRAFT = {
    "draft_tokens": torch.tensor([0], dtype=torch.int64, device="cpu"),
    "draft_logprobs": torch.tensor([0.0], dtype=torch.float32, device="cpu"),
}


class SpectreWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.tp_group = getattr(target_worker.model_runner, "tp_group", None)
        self.tp_size = target_worker.tp_size

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self._cached_tree_structures: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def draft_model_runner(self):
        return None

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_config

    def clear_cache_pool(self):
        self._cached_tree_structures.clear()

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, _ = self.forward_target_extend(batch)
            self._recv_drafts_after_extend(batch, next_token_ids)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            draft_num_tokens = getattr(
                batch, "draft_num_tokens", self.speculative_num_draft_tokens
            )

            if draft_num_tokens == 1 and not batch.forward_mode.is_idle():
                batch_result = self._forward_normal_decode(batch)
                return batch_result

            spec_steps = max(draft_num_tokens - 1, 1)
            spec_info = self.construct_draft_input(batch, draft_num_tokens, spec_steps)

            recv_draft_fn = getattr(batch, "recv_draft_fn", None)
            retry_fn = getattr(batch, "retry_fn", None)
            retry_fail_ratio = getattr(batch, "retry_fail_ratio", 0.0)
            retry_min_count = getattr(batch, "retry_min_count", 4)

            logits_output, verify_output, _, can_run_cuda_graph = self.verify(
                batch,
                spec_info,
                recv_draft_fn=recv_draft_fn,
                retry_fn=retry_fn,
                retry_fail_ratio=retry_fail_ratio,
                retry_min_count=retry_min_count,
            )

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor]]:
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        return (
            batch_result.logits_output,
            batch_result.next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def _recv_drafts_after_extend(
        self, batch: ScheduleBatch, next_token_ids: torch.Tensor
    ):
        recv_draft_fn = getattr(batch, "recv_draft_fn", None)
        if recv_draft_fn is None:
            return

        new_drafts = recv_draft_fn(batch)
        target_tokens = next_token_ids.tolist()
        decoding_reqs = getattr(batch, "decoding_reqs", None) or []
        decoding_req_ids = {
            id(req) for req in decoding_reqs if not _is_health_check(req)
        }

        for i, req in enumerate(batch.reqs):
            if _is_health_check(req):
                continue
            if id(req) not in decoding_req_ids and getattr(req, "is_chunked", 0) > 0:
                continue
            token = target_tokens[i] if i < len(target_tokens) else None
            if token is not None:
                _apply_drafts_to_req(req, token, new_drafts.get(req.rid), skip_d0=True)
            else:
                req.cur_drafts = []
                req.draft_tokens_and_logits = _default_draft()

            req.spec_cnt += 1
            req.len_output_ids = len(req.output_ids) + 1

    def construct_draft_input(
        self,
        batch: ScheduleBatch,
        draft_num_tokens: Optional[int] = None,
        spec_steps: Optional[int] = None,
    ) -> EagleVerifyInput:
        num_draft_tokens = (
            draft_num_tokens
            if draft_num_tokens is not None
            else self.speculative_num_draft_tokens
        )
        spec_steps = (
            spec_steps if spec_steps is not None else self.speculative_num_steps
        )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk, spec_steps, num_draft_tokens
            )

        bs = batch.batch_size()
        device = batch.device
        topk = self.topk

        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        if (
            batch.seq_lens_cpu is None
            or batch.seq_lens_cpu.sum().item() != batch.seq_lens_sum
        ):
            batch.seq_lens_cpu = batch.seq_lens.cpu()

        verified_id_buf = torch.empty(bs, dtype=torch.int64)
        draft_tokens_buf = torch.zeros(bs, spec_steps, dtype=torch.int64)

        for i, req in enumerate(batch.reqs):
            verified_id_buf[i] = (
                req.output_ids[-1]
                if len(req.output_ids) > 0
                else req.origin_input_ids[-1]
            )
            dtl = req.draft_tokens_and_logits
            if dtl is not None:
                dt = dtl.get("draft_tokens")
                if dt is not None:
                    if isinstance(dt, torch.Tensor):
                        n = min(dt.numel(), spec_steps)
                        draft_tokens_buf[i, :n] = dt[:n].cpu() if dt.is_cuda else dt[:n]
                    else:
                        n = min(len(dt), spec_steps)
                        draft_tokens_buf[i, :n] = torch.tensor(
                            dt[:n], dtype=torch.int64
                        )

        verified_id = verified_id_buf.to(device=device, non_blocking=True)
        draft_tokens = draft_tokens_buf.to(device=device, non_blocking=True)

        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                verified_id
            )

        if topk == 1:
            cached_parent, cached_index = self._get_cached_tree_structure(
                num_draft_tokens, spec_steps
            )
            parent_list = cached_parent.unsqueeze(0).expand(bs, -1).contiguous()
            top_scores_index = cached_index.unsqueeze(0).expand(bs, -1).contiguous()
            draft_tokens = draft_tokens[:, : num_draft_tokens - 1].contiguous()
        else:
            parent_list, top_scores_index, draft_tokens = (
                self._construct_tree_structure_general(
                    draft_tokens, bs, device, topk, spec_steps, num_draft_tokens
                )
            )

        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            final_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=batch.seq_lens_sum,
            topk=topk,
            spec_steps=spec_steps,
            num_verify_tokens=num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=final_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=spec_steps,
            topk=topk,
            draft_token_num=num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        recv_draft_fn=None,
        retry_fn=None,
        retry_fail_ratio: float = 0.0,
        retry_min_count: int = 4,
    ):
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = spec_info.spec_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        if self.enable_nan_detection:
            maybe_detect_nan(
                logits_output.next_token_logits,
                "SpectreWorker verify logits",
            )

        torch.cuda.synchronize()
        new_drafts_per_req: dict = {}
        if recv_draft_fn is not None and not batch.forward_mode.is_idle():
            new_drafts_per_req = recv_draft_fn(batch)

        spec_info.hidden_states = logits_output.hidden_states

        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )

        self._post_verify_update_drafts(
            batch,
            res,
            new_drafts_per_req,
            retry_fn=retry_fn,
            retry_fail_ratio=retry_fail_ratio,
            retry_min_count=retry_min_count,
        )

        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _post_verify_update_drafts(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        new_drafts_per_req: dict,
        retry_fn=None,
        retry_fail_ratio: float = 0.0,
        retry_min_count: int = 4,
    ):
        failed_reqs: List = []

        for i, req in enumerate(batch.reqs):
            if _is_health_check(req):
                continue

            drafts = new_drafts_per_req.get(req.rid)

            if drafts is not None:
                draft_token_ids, draft_logprobs = drafts
                verified_tokens = req.output_ids[req.len_output_ids :]
                cur_draft_tokens = list(getattr(req, "cur_drafts", []))
                cur_draft_tokens.append(draft_token_ids[0])

                is_matched, matched_idx = _find_fork_point(
                    verified_tokens, cur_draft_tokens
                )
                req.draft_cnt += len(cur_draft_tokens)
                req.accept_cnt += matched_idx

                if is_matched:
                    req.cur_drafts = list(draft_token_ids[1:])
                    req.draft_tokens_and_logits = _make_draft_dict(
                        draft_token_ids[1:], draft_logprobs[1:]
                    )
                else:
                    req.cur_drafts = []
                    req.draft_tokens_and_logits = _default_draft()
                    failed_reqs.append(req)
            else:
                req.cur_drafts = []
                req.draft_tokens_and_logits = _default_draft()
                failed_reqs.append(req)

            req.spec_cnt += 1
            req.len_output_ids = len(req.output_ids)

        bsz = sum(1 for req in batch.reqs if not _is_health_check(req))
        should_retry = (
            retry_fn is not None
            and failed_reqs
            and bsz > 0
            and bsz > retry_min_count
            and len(failed_reqs) / bsz > retry_fail_ratio
        )
        if should_retry:
            retry_drafts = retry_fn(failed_reqs)
            for req in failed_reqs:
                _apply_drafts_to_req(
                    req,
                    verified_token=-1,
                    drafts=retry_drafts.get(req.rid),
                    skip_d0=False,
                )
                req.spec_cnt += 1

    def _forward_normal_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        bs = batch.batch_size()
        last_token_ids_cpu = [
            req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
            for req in batch.reqs
        ]
        device = batch.seq_lens.device

        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                torch.tensor(last_token_ids_cpu, dtype=torch.int64, device=device)
            )

        batch.input_ids = torch.tensor(
            last_token_ids_cpu, dtype=torch.int32, device=device
        )
        batch.output_ids = None
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_info = None

        if batch.global_num_tokens is not None:
            batch.global_num_tokens = [bs]
        if batch.global_num_tokens_for_logprob is not None:
            batch.global_num_tokens_for_logprob = [bs]

        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        for req in batch.reqs:
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        if batch.orig_seq_lens is not None:
            batch.orig_seq_lens.add_(1)
        batch.seq_lens_sum += bs

        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        recv_draft_fn = getattr(batch, "recv_draft_fn", None)
        new_drafts_per_req: dict = (
            recv_draft_fn(batch) if recv_draft_fn is not None else {}
        )

        next_token_ids_list = batch_result.next_token_ids.tolist()
        for i, req in enumerate(batch.reqs):
            if _is_health_check(req):
                continue
            token = next_token_ids_list[i] if i < len(next_token_ids_list) else None
            if token is not None:
                req.output_ids.append(token)
                if req.grammar is not None and not req.finished():
                    try:
                        req.grammar.accept_token(token)
                    except ValueError:
                        logger.error(
                            f"\033[36m [NormalDecode] grammar.accept_token failed "
                            f"for req {req.rid} token {token} \033[0m"
                        )
            _apply_drafts_to_req(
                req,
                verified_token=token if token is not None else -1,
                drafts=new_drafts_per_req.get(req.rid) if new_drafts_per_req else None,
                skip_d0=True,
            )
            req.spec_cnt += 1
            req.len_output_ids = len(req.output_ids)

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            accept_length_per_req_cpu=[1] * bs,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    def _get_cached_tree_structure(
        self, num_draft_tokens: int, spec_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._cached_tree_structures.get(num_draft_tokens)
        if cached is None:
            device = self.device
            parent_list = torch.arange(
                -1, spec_steps - 1, dtype=torch.int64, device=device
            )
            top_scores_index = torch.arange(
                num_draft_tokens - 1, dtype=torch.int64, device=device
            )
            cached = (parent_list, top_scores_index)
            self._cached_tree_structures[num_draft_tokens] = cached
        return cached

    def _construct_tree_structure_general(
        self,
        all_draft_tokens: torch.Tensor,
        bs: int,
        device: torch.device,
        topk: int,
        spec_steps: int,
        num_draft_tokens: int,
    ):
        score_list, token_list, parents_list = [], [], []
        for i in range(spec_steps):
            if i == 0:
                scores = torch.ones((bs, 1, topk), dtype=torch.float32, device=device)
                tokens = all_draft_tokens[:, 0:1].repeat(1, topk)
                parents = (
                    torch.arange(-1, topk, dtype=torch.int64, device=device)
                    .unsqueeze(0)
                    .repeat(bs, 1)
                )
            else:
                scores = torch.ones(
                    (bs, topk, topk), dtype=torch.float32, device=device
                )
                tokens = all_draft_tokens[:, i : i + 1].repeat(1, topk * topk)
                topk_cs_index = torch.zeros(
                    (bs, topk), dtype=torch.int64, device=device
                )
                parents = topk_cs_index + (topk * topk * (i - 1) + topk)
            score_list.append(scores)
            token_list.append(tokens)
            parents_list.append(parents)

        return organize_draft_results(
            score_list=score_list,
            token_list=token_list,
            parents_list=parents_list,
            num_draft_token=num_draft_tokens,
        )


def _apply_drafts_to_req(
    req,
    verified_token: int,
    drafts: Optional[Tuple[List[int], List[float]]],
    *,
    skip_d0: bool = True,
) -> None:
    if drafts is not None:
        token_ids, logprobs = drafts
        if token_ids:
            if not skip_d0:
                req.cur_drafts = list(token_ids)
                req.draft_tokens_and_logits = _make_draft_dict(token_ids, logprobs)
                return
            if token_ids[0] == verified_token:
                req.cur_drafts = list(token_ids[1:])
                req.draft_tokens_and_logits = _make_draft_dict(
                    token_ids[1:], logprobs[1:]
                )
                return
    req.cur_drafts = []
    req.draft_tokens_and_logits = _default_draft()


def _default_draft() -> dict:
    return {
        "draft_tokens": _DEFAULT_DRAFT["draft_tokens"].clone(),
        "draft_logprobs": _DEFAULT_DRAFT["draft_logprobs"].clone(),
    }


def _make_draft_dict(token_ids, logprobs) -> dict:
    return {
        "draft_tokens": torch.tensor(token_ids, dtype=torch.int64, device="cpu"),
        "draft_logprobs": torch.tensor(logprobs, dtype=torch.float32, device="cpu"),
    }


def _find_fork_point(
    verified_tokens: List[int], draft_tokens: List[int]
) -> Tuple[bool, int]:
    if not verified_tokens or not draft_tokens:
        return False, 0

    min_len = min(len(verified_tokens), len(draft_tokens))
    for i in range(min_len):
        if verified_tokens[i] != draft_tokens[i]:
            return False, i

    if len(draft_tokens) > len(verified_tokens):
        return False, len(verified_tokens)

    return True, len(draft_tokens)
