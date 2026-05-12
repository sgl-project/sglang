from __future__ import annotations

import inspect
import logging

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    DecodeInputBuffers,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    empty_context,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
)

logger = logging.getLogger(__name__)


def dummy_run(
    *,
    batch_size: int,
    is_generation: bool,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
    server_args: ServerArgs,
    attn_backend: object,
    device: str,
    model: torch.nn.Module,
    model_config: ModelConfig,
    req_to_token_pool,
    token_to_kv_pool,
    lora_manager,
    tp_group,
    run_ctx=None,
):
    """Run a dummy forward pass for warmup/profiling."""
    if is_generation:
        capture_forward_mode = ForwardMode.DECODE
    else:
        capture_forward_mode = ForwardMode.EXTEND
    capture_hidden_mode = CaptureHiddenMode.NULL
    num_tokens_per_bs = 1
    if spec_algorithm.is_speculative():
        if is_draft_worker:
            if not spec_algorithm.is_dflash():
                raise RuntimeError("This should not happen")
        capture_forward_mode = ForwardMode.TARGET_VERIFY
        num_tokens_per_bs = server_args.speculative_num_draft_tokens

    if server_args.enable_return_hidden_states:
        capture_hidden_mode = CaptureHiddenMode.FULL

    num_tokens = batch_size * num_tokens_per_bs

    if require_gathered_buffer(server_args):
        attn_tp_size = get_attention_tp_size()
        if attn_tp_size > 1 and num_tokens % attn_tp_size != 0:
            num_tokens = num_tokens // attn_tp_size * attn_tp_size
            batch_size = num_tokens // num_tokens_per_bs

    seq_len_fill_value = attn_backend.get_cuda_graph_seq_len_fill_value()

    if server_args.enable_torch_compile:
        set_torch_compile_config()
        should_disable_torch_compile = not getattr(model, "_can_torch_compile", True)
        if should_disable_torch_compile:
            log_info_on_rank0(
                logger,
                "Transformers backend model reports it is not torch.compile "
                "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
            )
            server_args.enable_torch_compile = False

    # NOTE: aux hidden state capture (eagle3/dflash) is already
    # configured by init_aux_hidden_state_capture() in initialize().

    require_mlp_tp_gather_ = require_mlp_tp_gather(server_args)
    if require_gathered_buffer(server_args):
        assert require_mlp_tp_gather_ or require_attn_tp_gather(server_args)

    buffers: DecodeInputBuffers = DecodeInputBuffers.create(
        device=device,
        max_bs=batch_size,
        max_num_token=num_tokens,
        hidden_size=model_config.hidden_size,
        vocab_size=model_config.vocab_size,
        dtype=model_config.dtype,
        dp_size=server_args.dp_size,
        pp_size=server_args.pp_size,
        is_encoder_decoder=model_config.is_encoder_decoder,
        require_mlp_tp_gather=require_mlp_tp_gather_,
        seq_len_fill_value=seq_len_fill_value,
        encoder_len_fill_value=(
            getattr(model_config.hf_config, "max_source_positions", 0)
            if model_config.is_encoder_decoder
            else 0
        ),
        num_tokens_per_bs=num_tokens_per_bs,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=False,
    )
    buffers.num_token_non_padded[...] = num_tokens

    # For extend mode
    if not is_generation:
        extend_prefix_lens_cpu = [0] * batch_size
        extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
        extend_num_tokens = num_tokens
        extend_seq_lens = torch.full(
            (batch_size,), seq_len_fill_value, dtype=torch.int32, device=device
        )
        extend_prefix_lens = torch.zeros(
            (batch_size,), dtype=torch.int32, device=device
        )
        extend_start_loc = torch.arange(
            0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=device
        )
    else:
        extend_prefix_lens_cpu = None
        extend_seq_lens_cpu = None
        extend_num_tokens = None
        extend_seq_lens = None
        extend_prefix_lens = None
        extend_start_loc = None

    if server_args.pp_size > 1:
        pp_proxy_tensors = PPProxyTensors(
            {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
        )

    if require_mlp_tp_gather_:
        buffers.global_num_tokens_gpu.copy_(
            torch.tensor(
                [num_tokens] * server_args.dp_size,
                dtype=torch.int32,
                device=device,
            )
        )
        buffers.global_num_tokens_for_logprob_gpu.copy_(
            torch.tensor(
                [num_tokens] * server_args.dp_size,
                dtype=torch.int32,
                device=device,
            )
        )
        global_dp_buffer_len = num_tokens * server_args.dp_size
    elif require_attn_tp_gather(server_args):
        buffers.global_num_tokens_gpu.copy_(
            torch.tensor(
                [num_tokens],
                dtype=torch.int32,
                device=device,
            )
        )
        buffers.global_num_tokens_for_logprob_gpu.copy_(
            torch.tensor(
                [num_tokens],
                dtype=torch.int32,
                device=device,
            )
        )
        global_dp_buffer_len = num_tokens
    else:
        global_dp_buffer_len = None

    def get_spec_info():
        spec_info = None
        if spec_algorithm.is_eagle() or spec_algorithm.is_standalone():
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    spec_steps=server_args.speculative_num_steps,
                    topk=server_args.speculative_eagle_topk,
                    draft_token_num=server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
        elif spec_algorithm.is_dflash():
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput

            # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
            spec_info = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=server_args.speculative_num_draft_tokens,
                custom_mask=None,
                capture_hidden_mode=(
                    CaptureHiddenMode.NULL
                    if is_draft_worker
                    else CaptureHiddenMode.FULL
                ),
            )

        elif spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                tree_mask=buffers.custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                draft_token_num=num_tokens_per_bs,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info

    spec_info = get_spec_info()
    if capture_hidden_mode != CaptureHiddenMode.FULL:
        capture_hidden_mode = (
            spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
        )

    if server_args.enable_lora:
        lora_ids = [None] * batch_size
    else:
        lora_ids = None

    forward_batch = ForwardBatch(
        forward_mode=capture_forward_mode,
        batch_size=batch_size,
        input_ids=buffers.input_ids,
        req_pool_indices=buffers.req_pool_indices,
        seq_lens=buffers.seq_lens,
        seq_lens_cpu=buffers.seq_lens_cpu,
        next_token_logits_buffer=buffers.next_token_logits_buffer,
        orig_seq_lens=buffers.seq_lens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        attn_backend=attn_backend,
        out_cache_loc=buffers.out_cache_loc,
        seq_lens_sum=buffers.seq_lens.sum().item(),
        encoder_lens=buffers.encoder_lens,
        return_logprob=False,
        positions=buffers.positions,
        extend_num_tokens=extend_num_tokens,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_start_loc=extend_start_loc,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        global_num_tokens_gpu=buffers.global_num_tokens_gpu,
        global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
        dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        global_dp_buffer_len=global_dp_buffer_len,
        mrope_positions=buffers.mrope_positions,
        spec_algorithm=spec_algorithm,
        spec_info=spec_info,
        capture_hidden_mode=capture_hidden_mode,
        num_token_non_padded=buffers.num_token_non_padded,
        global_forward_mode=capture_forward_mode,
        lora_ids=lora_ids,
    )

    if lora_ids is not None:
        lora_manager.prepare_lora_batch(forward_batch)

    attn_backend.init_forward_metadata(forward_batch)

    def run_once():
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(
            global_dp_buffer_len,
            num_tokens,
            forward_batch.dp_padding_mode.is_max_len(),
        )
        set_is_extend_in_batch(False)

        kwargs = {}
        if (
            server_args.pp_size > 1
            and "pp_proxy_tensors" in inspect.signature(model.forward).parameters
        ):
            kwargs["pp_proxy_tensors"] = PPProxyTensors(
                {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
            )
        if not is_generation:
            kwargs["get_embedding"] = True

        logits_output_or_pp_proxy_tensors = model.forward(
            buffers.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )
        return logits_output_or_pp_proxy_tensors

    torch.get_device_module(device).synchronize()
    tp_group.barrier()
    with torch.inference_mode(), run_ctx or empty_context():
        run_once()
