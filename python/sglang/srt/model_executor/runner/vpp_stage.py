"""Eager logical-stage execution with batch-owned inputs and full token rows."""

from sglang.srt.utils.device_timer import device_timer_ctx


def execute_vpp_stage(
    *, model, attn_backend, forward_batch, stage_id, pp_proxy_tensors, device_timer
):
    layout = model.vpp_layout
    if layout is None:
        raise ValueError("VPP execution requires a model-owned layout")
    if not any(stage.stage_id == stage_id for stage in model.vpp_stages):
        raise ValueError("VPP stage does not belong to this model rank")
    mode = forward_batch.forward_mode
    if not mode.is_extend() or mode.is_mixed():
        raise NotImplementedError("VPP execution supports prefill batches only")
    if forward_batch.global_num_tokens_cpu is not None:
        raise NotImplementedError("VPP execution requires full rows without MLP sync")
    # No eager registry copy: each retained batch keeps its own input/KV-index
    # tensors across virtual passes. Re-plan the shared attention workspace for
    # every pass, since another batch may have used it in between.
    forward_batch.attn_tp_sequence_sharded = False
    prepare = getattr(model, "prepare_forward_batch", None)
    if prepare is not None:
        prepare(forward_batch)
    attn_backend.init_forward_metadata(forward_batch)
    kwargs = {"pp_proxy_tensors": pp_proxy_tensors}
    if stage_id == 0 and forward_batch.input_embeds is not None:
        kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
    if forward_batch.replace_embeds is not None:
        raise NotImplementedError("VPP replacement embeddings are not wired")
    with device_timer_ctx(device_timer, "extend"):
        return model.forward_vpp_stage(
            stage_id,
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )
