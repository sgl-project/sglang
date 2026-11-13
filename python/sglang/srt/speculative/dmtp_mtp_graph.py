"""Exact-B hidden-only MTP graphs over the upstream private runner's KV ids.

This does not alter EAGLE's graph runner, tree width, or sampling path. Each
request has eight execution rows; only accepted rows and its pending token
are committed, with all remaining writes routed to private scratch.
"""

from types import SimpleNamespace

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.speculative.dmtp_runtime import mtp_config, scoped_config
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class DMTPMTPGraphRunner:
    @torch.no_grad()
    def __init__(self, native, backend):
        self.native, self.backend = native, backend
        self.runner = native.worker.model_runner
        self.max_bs = native.owner.server_args.max_running_requests
        self.capture_bs = list(range(1, self.max_bs + 1))
        n = self.max_bs * 8
        weight = native.owner.target_worker.model_runner.model.lm_head.weight
        device = weight.device
        reqs = torch.arange(self.max_bs, device=device, dtype=torch.int64)
        scratch = native.pending_base + native.target_mapping.req_to_token.shape[0]
        slots = torch.cat(
            (
                native.pending_base + reqs[:, None],
                scratch + reqs[:, None] * 7 + torch.arange(7, device=device),
            ),
            dim=1,
        )
        self.buffers = SimpleNamespace(
            input_ids=torch.zeros(n, device=device, dtype=torch.int64),
            req_pool_indices=reqs,
            out_cache_loc=slots.flatten(),
            positions=torch.arange(1, 9, device=device).repeat(self.max_bs),
            mrope_positions=torch.arange(1, 9, device=device).repeat(3, self.max_bs),
            hidden_states=torch.zeros(
                n, weight.shape[1], device=device, dtype=weight.dtype
            ),
            seq_lens=torch.full((self.max_bs,), 8, device=device, dtype=torch.int64),
            extend_seq_lens=torch.full(
                (self.max_bs,), 8, device=device, dtype=torch.int32
            ),
            num_correct_drafts=torch.zeros(
                self.max_bs, device=device, dtype=torch.int32
            ),
            num_accept_tokens=torch.ones(self.max_bs, device=device, dtype=torch.int32),
        )
        self.graphs, self.outputs, self.batches = {}, {}, {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        pool = torch.cuda.graph_pool_handle()
        with (
            scoped_config(**mtp_config(native.owner)),
            get_tp_group().graph_capture(stream=stream),
            model_capture_mode(),
        ):
            for bs in reversed(self.capture_bs):
                batch = self.make_batch(bs)
                self.batches[bs] = batch
                with (
                    torch.cuda.stream(stream),
                    forward_context(
                        ForwardContext(attn_backend=backend), forward_batch=batch
                    ),
                ):
                    backend.init_forward_metadata_out_graph(batch, in_capture=True)
                    for _ in range(3):
                        self.forward(batch)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with (
                    forward_context(
                        ForwardContext(attn_backend=backend), forward_batch=batch
                    ),
                    torch.cuda.graph(graph, pool=pool, stream=stream),
                ):
                    output = self.forward(batch)
                self.graphs[bs], self.outputs[bs] = graph, output
        torch.cuda.current_stream().wait_stream(stream)

    def bind(self, batch):
        ids = self.runner.runner_ids
        ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=None,
            write_virtual=None,
            write_slots=batch.out_cache_loc,
        ).bind(batch, ids)

    def make_batch(self, bs):
        b, n = self.buffers, bs * 8
        spec = EagleDraftExtendInput(
            hidden_states=b.hidden_states[:n],
            num_correct_drafts=b.num_correct_drafts[:bs],
            num_accept_tokens=b.num_accept_tokens[:bs],
        )
        spec.extend_seq_lens_tensor = b.extend_seq_lens[:bs]
        batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=bs,
            input_ids=b.input_ids[:n],
            req_pool_indices=b.req_pool_indices[:bs],
            seq_lens=b.seq_lens[:bs],
            seq_lens_cpu=None,
            seq_lens_sum=None,
            out_cache_loc=b.out_cache_loc[:n],
            positions=b.positions[:n],
            mrope_positions=b.mrope_positions[:, :n],
            extend_num_tokens=n,
            extend_seq_lens=b.extend_seq_lens[:bs],
            extend_seq_lens_cpu=[8] * bs,
            extend_start_loc=torch.arange(
                bs, device=b.input_ids.device, dtype=torch.int32
            )
            * 8,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=spec,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        )
        self.bind(batch)
        return batch

    def forward(self, batch):
        set_dp_buffer_len(
            None, batch.input_ids.numel(), batch.dp_padding_mode.is_max_len(), None
        )
        set_is_extend_in_batch(False)
        output = self.runner.model.forward(batch.input_ids, batch.positions, batch)
        if output.hidden_states is None or output.next_token_logits is not None:
            raise RuntimeError(
                "DMTP MTP graph requires full normalized hidden without logits"
            )
        return output.hidden_states

    def replay(self, batch):
        bs, b = batch.batch_size, self.buffers
        if bs not in self.graphs or batch.input_ids.numel() != bs * 8:
            raise ValueError("DMTP MTP replay requires an exact captured batch")
        n = bs * 8
        for name in (
            "input_ids",
            "req_pool_indices",
            "out_cache_loc",
            "positions",
            "seq_lens",
        ):
            value = getattr(batch, name)
            getattr(b, name)[: value.shape[0]].copy_(value)
        b.mrope_positions[:, :n].copy_(batch.mrope_positions)
        b.hidden_states[:n].copy_(batch.spec_info.hidden_states)
        b.num_correct_drafts[:bs].copy_(batch.spec_info.num_correct_drafts)
        b.num_accept_tokens[:bs].copy_(batch.spec_info.num_accept_tokens)
        target = self.batches[bs]
        with scoped_config(**mtp_config(self.native.owner)):
            self.backend.init_forward_metadata_out_graph(target, in_capture=True)
            self.graphs[bs].replay()
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=self.outputs[bs]
        )
