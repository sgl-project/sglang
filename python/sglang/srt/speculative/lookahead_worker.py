import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.lookahead_cache import LookaheadCache
from sglang.srt.speculative.lookahead_utils import LookaheadVerifyInput
from sglang.srt.speculative.spec_info import SpecInfo, SpeculativeAlgorithm
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LOOKAHEADWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: "TpModelWorker",
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.num_branch_token: int = server_args.speculative_num_draft_tokens
        self.one_branch = server_args.speculative_one_branch
        self.lookahead_cache = None
        if tp_rank == 0:
            self.lookahead_cache = LookaheadCache(
                debug=False, eos_ids=None, gpu_id=tp_rank
            )
            if server_args.speculative_lookahead_path is not None:
                logger.info(
                    f"Load lookahead from: {server_args.speculative_lookahead_path}"
                )
                self.lookahead_cache.load_mem(server_args.speculative_lookahead_path)
        self.rids = {}

    def prepare_for_verify(self, batch: ScheduleBatch):
        bs = len(batch.reqs)
        if bs > 4:
            # if batch size is too large, fallback to normal decode
            batch.spec_algorithm = SpeculativeAlgorithm.NONE
            return None
        (
            seq_lens,
            leaf_nums,
            look_ahead_res,
            drafts,
            tree_mask,
            positions,
            retrive_indexes,
            draft_token_nums,
        ) = ([], [], [], [], [], [], [], [])
        for req in batch.reqs:
            fill_ids = req.origin_input_ids + req.output_ids
            seq_len = len(fill_ids)
            seq_lens.append(seq_len)
            if self.lookahead_cache is not None:
                check_token = fill_ids[-self.num_branch_token :]
                # make total_draft_len 2^n
                total_draft_len = (
                    2 ** ((self.num_branch_token * 4 // bs) - 1).bit_length()
                )
                is_one_branch = (
                    self.one_branch or total_draft_len <= self.num_branch_token
                )
                if is_one_branch:
                    req_drafts, mask, _ = self.lookahead_cache.one_get(
                        check_token,
                        branch_length=self.num_branch_token,
                        idx=self.rids[req.rid],
                    )
                else:
                    req_drafts, mask, _ = self.lookahead_cache.hier_get(
                        check_token,
                        idx=self.rids[req.rid],
                        branch_length=self.num_branch_token,
                        decoding_length=total_draft_len,
                    )
                data = broadcast_pyobj(
                    [req_drafts, mask],
                    self.tp_rank,
                    self.model_runner.tp_group.cpu_group,
                )

            else:
                (req_drafts, mask) = broadcast_pyobj(
                    [],
                    self.tp_rank,
                    self.model_runner.tp_group.cpu_group,
                )
            look_ahead_res.append((req_drafts, mask))
            # number of draft tokens might be different for each req
            draft_token_nums.append(len(req_drafts))

        # check the draft_token_nums all 1 s, if no match just normal decode
        if np.sum(draft_token_nums) == bs:
            batch.spec_algorithm = SpeculativeAlgorithm.NONE
            return None

        cum_draft_token_nums = np.cumsum([0] + draft_token_nums)
        for i, (req_drafts, mask_) in enumerate(look_ahead_res):
            seq_len = seq_lens[i]
            mask = torch.from_numpy(mask_).cuda()
            req_mask = torch.ones(
                (len(req_drafts), seq_len - 1)
            ).cuda()  # TODO: check the new generated token
            req_mask = torch.cat((req_mask, mask), dim=1).to(torch.bool)
            tree_mask.append(req_mask.flatten())

            leaf_mask = mask_[np.argmax(mask_[:, mask_.sum(0) == 1], axis=0), :]
            leaf_num = leaf_mask.shape[0]
            leaf_nums.append(leaf_num)
            row_indices, col_indices = np.nonzero(leaf_mask)
            retrieve_index = [[] for _ in range(leaf_num)]
            for row, col in zip(row_indices, col_indices):
                retrieve_index[row].append(col + cum_draft_token_nums[i])
            for idxs in retrieve_index:
                idxs.extend([-1] * (self.num_branch_token - len(idxs)))

            retrieve_index = torch.tensor(retrieve_index, device="cuda")
            retrive_indexes.append(retrieve_index)
            position = mask.sum(1) + seq_len - 1
            positions.append(position)

            drafts.extend(req_drafts)

        # only one row for each req for one branch case
        leaf_nums = torch.tensor(leaf_nums, device="cuda")
        cum_len = torch.cumsum(leaf_nums, dim=0)
        retrive_cum_len = torch.zeros(
            (leaf_nums.numel() + 1,), dtype=torch.int32, device="cuda"
        )
        retrive_cum_len[1:] = cum_len

        draft_tokens = torch.tensor(drafts, device="cuda")
        self.draft_token_nums = torch.tensor(draft_token_nums, device="cuda")
        retrive_indexes = torch.vstack(retrive_indexes).to(torch.long).cuda()
        positions = torch.cat(positions, axis=0).to(torch.long)
        tree_mask = torch.cat(tree_mask, axis=0)
        batch.spec_algorithm = SpeculativeAlgorithm.LOOKAHEAD
        return LookaheadVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_indexes,
            retrive_cum_len,
            self.draft_token_nums,
        )

    def forward_batch_speculative_generation(self, batch: ScheduleBatch):
        if batch.forward_mode.is_target_verify():
            verify_input = batch.spec_info
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, _ = self.target_worker.forward_batch_generation(
                model_worker_batch, skip_sample=True
            )
            batch.forward_mode = ForwardMode.DECODE
            logits_output, verified_id, accept_length_sum = verify_input.verify(
                batch, logits_output
            )
            return logits_output, verified_id, model_worker_batch, accept_length_sum

        else:
            model_worker_batch = batch.get_model_worker_batch()
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            if self.lookahead_cache is not None:
                next_token_ids_cpu = next_token_ids.tolist()
                for r, token in zip(batch.reqs, next_token_ids_cpu):
                    self.rids[r.rid] = len(self.rids)
                    put_ids = r.fill_ids + [token]
                    self.lookahead_cache.put(
                        put_ids[1:],
                        branch_length=self.num_branch_token * 2,
                        mode="input",
                        idx=self.rids[r.rid],
                    )
            return logits_output, next_token_ids, model_worker_batch, 0

    def finish_request(self, reqs: Union[Req, List[Req]]):
        if not isinstance(reqs, List):
            reqs = [reqs]
        for req in reqs:
            if self.lookahead_cache is not None:
                put_ids = (
                    req.origin_input_ids[-self.num_branch_token :] + req.output_ids
                )
                # update the lookahead_cache after the request is finished, and do the clean up
                self.lookahead_cache.put(
                    put_ids,
                    branch_length=self.num_branch_token * 2,
                    mode="output",
                    idx=self.rids[req.rid],
                    final=True,
                )
                if len(self.rids) >= 1000:
                    self.rids = dict(list(self.rids.items())[-500:])
