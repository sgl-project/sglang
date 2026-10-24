"""
Base executor class for SGLang Diffusion ComfyUI integration.
"""

import hashlib
import logging
import uuid

import torch

try:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
except ImportError as exc:
    # Keep the nodes importable so ComfyUI shows this; fail clearly on first use.
    _RUNTIME_IMPORT_ERROR: ImportError | None = exc
    print(
        f"Error: failed to import the SGLang diffusion runtime: {exc!r}. "
        "Install the diffusion extras with 'pip install sglang[diffusion]'."
    )
else:
    _RUNTIME_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

# SGLDOptions.native_batching, smallest first: "off" sends one worker request
# per row, "cfg_only" one per image (its cond/uncond rows together), "full"
# one for the whole ComfyUI batch.
NATIVE_BATCHING_MODES = ("off", "cfg_only", "full")


class WorkerOutOfMemoryError(RuntimeError):
    """The worker ran out of GPU memory on this request."""


def _is_out_of_memory(error) -> bool:
    """Whether a worker error is an OOM: by type when it is an exception,
    otherwise by the message (the scheduler usually sends a string)."""
    oom_types = tuple(
        t
        for t in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(t, type)
    )
    if oom_types and isinstance(error, oom_types):
        return True
    return "out of memory" in str(error).lower()


def _hash_value(digest, value) -> None:
    """Hash ``value`` into ``digest``, framing each item so values cannot merge."""
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        digest.update(f"T{tensor.dtype}{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(f"D{len(value)}".encode())
        for key, item in value.items():
            _hash_value(digest, key)
            _hash_value(digest, item)
    elif isinstance(value, (list, tuple)):
        digest.update(f"L{len(value)}".encode())
        for item in value:
            _hash_value(digest, item)
    else:
        text = repr(value).encode()
        digest.update(f"V{len(text)}:".encode())
        digest.update(text)


# transformer_options entries with one item per cond chunk; the rest is shared
# apart from "sigmas", which holds one item per row of a single chunk.
_PER_CHUNK_TRANSFORMER_OPTIONS = ("cond_or_uncond", "uuids")


def _slice_batch_row(value, index: int, batch: int):
    """Row ``index`` of a batched ComfyUI argument."""
    return _take_batch_rows(value, slice(index, index + 1), batch)


def _take_batch_rows(value, rows, batch: int, _memo=None):
    """``rows`` (a slice or a list of indices) of a batched ComfyUI argument.

    ComfyUI concatenates every cond-derived kwarg to the full batch, so a
    leading dim of ``batch`` is per-row and 1 is shared; any other size is
    ambiguous and rejected rather than guessed. A tensor repeated in the
    value (one Flux control residual across blocks) maps to one slice.
    """
    if _memo is None:
        _memo = {}
    if torch.is_tensor(value):
        if value.ndim == 0 or value.shape[0] == 1:
            return value
        if value.shape[0] == batch:
            if id(value) not in _memo:
                _memo[id(value)] = (value, value[rows])
            return _memo[id(value)][1]
        raise ValueError(
            f"cannot split a batch of {batch} on a tensor of shape {tuple(value.shape)}"
        )
    if type(value) in (list, tuple):
        return type(value)(_take_batch_rows(item, rows, batch, _memo) for item in value)
    if type(value) is dict:
        return {
            key: _take_batch_rows(item, rows, batch, _memo)
            for key, item in value.items()
        }
    return value


def _cond_uuid_from_transformer_options(options):
    """ComfyUI's per-cond uuid for this call, stable across steps in one run."""
    if not isinstance(options, dict):
        return None
    uuids = options.get("uuids")
    if type(uuids) in (list, tuple) and uuids:
        return tuple(uuids)
    return None


def _slice_transformer_options(options, index: int, batch: int):
    """Shared options pass through; only the per-chunk entries are sliced."""
    if not isinstance(options, dict):
        return options
    sliced = dict(options)
    for key in _PER_CHUNK_TRANSFORMER_OPTIONS:
        value = options.get(key)
        if type(value) in (list, tuple) and value and batch % len(value) == 0:
            # Each chunk holds batch // len rows.
            sliced[key] = type(value)([value[index // (batch // len(value))]])
    # ComfyUI sets sigmas to the timestep before repeating it per chunk.
    sigmas = options.get("sigmas")
    if torch.is_tensor(sigmas) and sigmas.ndim > 0 and sigmas.shape[0] > 1:
        if batch % sigmas.shape[0] != 0:
            raise ValueError(
                f"cannot split a batch of {batch} on sigmas of shape {tuple(sigmas.shape)}"
            )
        row = index % sigmas.shape[0]
        sliced["sigmas"] = sigmas[row : row + 1]
    return sliced


def _cfg_groups(options, batch: int) -> list[list[int]]:
    """Rows of each image: ComfyUI stacks one chunk per cond, each holding
    every image, so image ``j`` is row ``j`` of every chunk."""
    chunks = options.get("cond_or_uncond") if isinstance(options, dict) else None
    if type(chunks) not in (list, tuple) or not chunks or batch % len(chunks):
        raise ValueError(
            "native_batching=cfg_only needs ComfyUI's transformer_options"
            f"['cond_or_uncond'] to split a batch of {batch} into its cond chunks"
        )
    per_chunk = batch // len(chunks)
    return [list(range(image, batch, per_chunk)) for image in range(per_chunk)]


def _cfg_group_count(options, batch: int):
    """Number of cfg_only groups, or None when the batch cannot be grouped."""
    try:
        return len(_cfg_groups(options, batch))
    except ValueError:
        return None


def _group_transformer_options(options, image: int, per_chunk: int):
    """Options for one image's rows: one row per chunk, so only sigmas is sliced."""
    if not isinstance(options, dict):
        return options
    sliced = dict(options)
    sigmas = options.get("sigmas")
    if torch.is_tensor(sigmas) and sigmas.ndim > 0 and sigmas.shape[0] > 1:
        if sigmas.shape[0] != per_chunk:
            raise ValueError(
                f"cannot split {per_chunk} images on sigmas of shape {tuple(sigmas.shape)}"
            )
        sliced["sigmas"] = sigmas[image : image + 1]
    return sliced


def _reject_model_patches(transformer_options) -> None:
    # The DiT runs in the SGLD worker, so ComfyUI block/attention patches would
    # be dropped silently (e.g. MiniMax H3 Fun ControlNet, ModelAttentionBackend).
    opts = transformer_options or {}
    found = [
        f"patches_replace.{name}"
        for name, blocks in (opts.get("patches_replace") or {}).items()
        if blocks
    ]
    found += [
        f"patches.{name}"
        for name, items in (opts.get("patches") or {}).items()
        if items
    ]
    if opts.get("optimized_attention_override") is not None:
        found.append("optimized_attention_override")
    if found:
        raise ValueError(
            "SGLD integrated mode runs the diffusion model in its worker and cannot "
            f"apply ComfyUI model patches ({', '.join(sorted(found))}); remove the "
            "patch nodes (ControlNet, attention backend, block patches) or use the "
            "native ComfyUI loader"
        )


# apply_model kwargs that carry image content (ControlNet residuals, reference latents).
CONTENT_CONDITIONING = ("control", "ref_latents")


def _reject_unapplied_conditioning(adapter, kwargs) -> None:
    dropped = [
        name
        for name in CONTENT_CONDITIONING
        if name not in adapter.applied_conditioning and kwargs.get(name)
    ]
    if dropped:
        raise ValueError(
            f"SGLD integrated mode does not apply {', '.join(dropped)} for this model "
            "(ControlNet / reference-latent nodes); remove those nodes or use the "
            "native ComfyUI loader"
        )


def _check_native_batch(batch: int, timestep, context) -> None:
    """Native batching sends per-row values as-is, so each must have ``batch`` rows."""
    for name, value in (("timestep", timestep), ("context", context)):
        rows = value.shape[0] if torch.is_tensor(value) and value.ndim > 0 else None
        if rows != batch:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value)
            raise ValueError(
                f"native batching needs one {name} row per latent row "
                f"(batch {batch}), got {shape}"
            )


class SGLDiffusionExecutor(torch.nn.Module):
    """Shared ComfyUI DiT-forward executor. Per-model logic lives on the adapter."""

    adapter_cls = None
    # SGLDOptions.native_batching; see NATIVE_BATCHING_MODES.
    native_batching = "off"
    # Lowered for the rest of a sampler run after the worker runs out of memory.
    _run_batching = None

    def __init__(self, generator, model_path, model, config):
        super(SGLDiffusionExecutor, self).__init__()
        self.generator = generator
        self.model_path = model_path
        # Not a submodule: the BaseModel holds this executor as diffusion_model,
        # and registering it back makes state_dict() / .to() recurse forever.
        object.__setattr__(self, "model", model)
        self.dtype = config.unet_config["dtype"]
        self.config = config
        self.loras = []
        self._lora_input = None
        self._sgld_reload = None
        self._ensure_runtime = None
        if self.adapter_cls is None:
            raise TypeError(f"{type(self).__name__} must set adapter_cls")
        self.adapter = self.adapter_cls()
        self.session_id = uuid.uuid4().hex
        self._run_id = 0
        self._sent_conds: set[str] = set()
        self._cond_key_cache: dict[tuple, str] = {}

    def state_dict(self, *args, **kwargs):
        raise RuntimeError(
            "The SGLD worker owns this diffusion model's weights, so ComfyUI weight "
            "nodes (LoraLoader, ModelSave, model merging) cannot use it; load LoRAs "
            "with SGLDLoraLoader"
        )

    @staticmethod
    def should_suppress_logs(timestep):
        """Determine if logs should be suppressed based on timestep value."""
        if torch.is_tensor(timestep):
            return bool((timestep < 1.0).all())
        return bool(timestep < 1.0)

    def set_lora(self, lora_nickname=None, lora_path=None, strength=None, target=None):
        """Set LoRA adapter using SGLang Diffusion API."""
        desired = {
            "lora_nickname": lora_nickname,
            "lora_path": lora_path,
            "strength": strength,
            "target": target,
        }
        if lora_nickname and len(lora_nickname) > 0:
            self.generator.set_lora(
                lora_nickname=lora_nickname,
                lora_path=lora_path,
                strength=strength,
                target=target,
            )

        self._lora_input = desired

    def begin_sampler_run(self) -> None:
        """One ComfyUI ``sampler.sample()`` invocation is one cache lifetime."""
        self._run_id += 1
        self._sent_conds = set()
        self._cond_key_cache = {}
        self._run_batching = None

    def end_sampler_run(self) -> None:
        """Run cache is evicted on the next bind of a newer id for this executor."""

    def sampler_sample_wrapper(self, executor, *args, **kwargs):
        if self._ensure_runtime is not None:
            self._ensure_runtime(self)
        model_wrap = args[0] if args else kwargs["model_wrap"]
        desired = model_wrap.model_patcher.model_options.get("sgld_lora_input")
        if desired != self._lora_input:
            if self._lora_input is not None:
                self.generator.unmerge_lora_weights()
                self._lora_input = None
            if desired is not None:
                self.set_lora(**desired)
        self.begin_sampler_run()
        try:
            return executor(*args, **kwargs)
        finally:
            self.end_sampler_run()

    def comfyui_session_id(self) -> str:
        return f"{self.session_id}:{self._run_id}"

    def _cond_key(self, packed) -> str | None:
        embeds = packed.prompt_embeds
        if not embeds:
            return None
        tensor = embeds[0]
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return None
        # A cond's content is fixed for the life of one sampler run, so with a
        # uuid from ComfyUI we only need to hash it once per run rather than
        # once per step; the memo key still folds in shape/dtype so a stale
        # entry can never be returned for tensors that don't actually match.
        cond_uuid = packed.unpack_ctx.get("comfyui_cond_uuid")
        memo_key = None
        if cond_uuid is not None:
            shape_key = tuple(
                (t.dtype, tuple(t.shape)) if torch.is_tensor(t) else None
                for t in embeds
            )
            memo_key = (cond_uuid, shape_key)
            cached = self._cond_key_cache.get(memo_key)
            if cached is not None:
                return cached
        # Hash everything drop_cached_fields removes: a hit means the worker
        # restores all of it, so a partial key would revive another cond.
        digest = hashlib.blake2b(digest_size=16)
        _hash_value(
            digest,
            (
                embeds,
                packed.pooled_embeds,
                packed.prompt_seq_lens,
                {
                    key: packed.extra_req.get(key)
                    for key in self.adapter.cached_extra_keys
                },
            ),
        )
        key = digest.hexdigest()
        if memo_key is not None:
            self._cond_key_cache[memo_key] = key
        return key

    def _mark_and_maybe_drop(self, packed) -> None:
        key = self._cond_key(packed)
        if key is not None:
            packed.extra_req["comfyui_cond_key"] = key
            if key in self._sent_conds:
                self.adapter.drop_cached_fields(packed)
            else:
                self._sent_conds.add(key)

    def _sampling_params_kwargs(self, packed, timestep) -> dict:
        return {
            "prompt": " ",
            "guidance_scale": packed.guidance_scale,
            "height": packed.height,
            "width": packed.width,
            "num_frames": 1,
            "num_inference_steps": 1,
            "save_output": False,
            "suppress_logs": self.should_suppress_logs(timestep),
        }

    def _execute_packed(self, packed, x, timestep):
        if _RUNTIME_IMPORT_ERROR is not None:
            raise RuntimeError(
                "SGLang diffusion runtime failed to import"
            ) from _RUNTIME_IMPORT_ERROR
        ensure = getattr(self, "_ensure_runtime", None)
        if ensure is not None:
            ensure(self)
        self._mark_and_maybe_drop(packed)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            **self._sampling_params_kwargs(packed, timestep),
        )
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        self.adapter.fill_req(req, packed)
        extra = dict(req.extra or {})
        extra["comfyui_session_id"] = self.comfyui_session_id()
        for key in ("comfyui_cond_key", "comfyui_cache_fp"):
            value = packed.extra_req.get(key)
            if value is not None:
                extra[key] = value
        if packed.control:
            extra["comfyui_control"] = packed.control
        req.extra = extra
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]
        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        if output_batch.error:
            message = f"SGLang Diffusion worker failed: {output_batch.error}"
            if _is_out_of_memory(output_batch.error):
                raise WorkerOutOfMemoryError(message)
            raise RuntimeError(message)
        return self.adapter.unpack(output_batch.noise_pred, packed, x)

    def forward(self, x, timestep, context, **kwargs):
        _reject_model_patches(kwargs.get("transformer_options"))
        _reject_unapplied_conditioning(self.adapter, kwargs)
        batch = int(x.shape[0]) if torch.is_tensor(x) else 1
        if batch == 1:
            return self._forward_one(x, timestep, context, **kwargs)
        groups = _cfg_group_count(kwargs.get("transformer_options"), batch)
        while True:
            mode = self._run_batching or self.native_batching
            if mode == "cfg_only" and groups is None:
                # No cond chunks to group by: send one request per row instead.
                mode = "off"
            try:
                if mode == "full":
                    return self._forward_batch(x, timestep, context, batch, **kwargs)
                if mode == "cfg_only":
                    return self._forward_cfg_groups(
                        x, timestep, context, batch, **kwargs
                    )
                return self._forward_rows(x, timestep, context, batch, **kwargs)
            except WorkerOutOfMemoryError:
                if mode not in ("full", "cfg_only"):
                    raise
                lower = NATIVE_BATCHING_MODES[NATIVE_BATCHING_MODES.index(mode) - 1]
                if lower == "cfg_only" and (groups is None or groups == 1):
                    # cfg_only would resend the same rows (one group) or
                    # cannot group them at all: go straight to per row.
                    lower = "off"
                logger.warning(
                    "SGLD worker ran out of GPU memory with native_batching=%s; "
                    "using %s for the rest of this sampler run",
                    mode,
                    lower,
                )
                self._run_batching = lower

    def _forward_rows(self, x, timestep, context, batch, **kwargs):
        # ComfyUI batches CFG cond/uncond (and batch_size > 1) into one call,
        # but the worker's comfyui path is per-sample: req.timesteps is its
        # schedule and seq lens are per request. Send one request per row.
        return torch.cat(
            [
                self._forward_one(
                    _slice_batch_row(x, i, batch),
                    _slice_batch_row(timestep, i, batch),
                    _slice_batch_row(context, i, batch),
                    _cond_row=i,
                    **{
                        key: (
                            _slice_transformer_options(value, i, batch)
                            if key == "transformer_options"
                            else _slice_batch_row(value, i, batch)
                        )
                        for key, value in kwargs.items()
                    },
                )
                for i in range(batch)
            ]
        )

    def _forward_cfg_groups(self, x, timestep, context, batch, **kwargs):
        """One worker request per image, holding that image's cond/uncond rows."""
        groups = _cfg_groups(kwargs.get("transformer_options"), batch)
        if len(groups) == 1:
            return self._forward_batch(x, timestep, context, batch, **kwargs)
        out = [None] * batch
        for image, rows in enumerate(groups):
            result = self._forward_batch(
                _take_batch_rows(x, rows, batch),
                _take_batch_rows(timestep, rows, batch),
                _take_batch_rows(context, rows, batch),
                len(rows),
                _group=image,
                **{
                    key: (
                        _group_transformer_options(value, image, len(groups))
                        if key == "transformer_options"
                        else _take_batch_rows(value, rows, batch)
                    )
                    for key, value in kwargs.items()
                },
            )
            for position, row in enumerate(rows):
                out[row] = result[position : position + 1]
        return torch.cat(out)

    def _forward_batch(self, x, timestep, context, batch, _group=None, **kwargs):
        """Send ``batch`` rows as one worker request (opt-in, verified models)."""
        _check_native_batch(batch, timestep, context)
        packed = self.adapter.pack(x, timestep, context, **kwargs)
        # The worker walks req.timesteps as the step schedule, so keep that one
        # step and send each row's timestep separately.
        packed.extra_req["comfyui_row_timesteps"] = packed.timesteps
        packed.timesteps = packed.timesteps[:1]
        cond_uuid = _cond_uuid_from_transformer_options(
            kwargs.get("transformer_options")
        )
        packed.unpack_ctx["comfyui_cond_uuid"] = (
            None if cond_uuid is None else (cond_uuid, ("batch", _group))
        )
        return self._execute_packed(packed, x, timestep)

    def _forward_one(self, x, timestep, context, _cond_row=None, **kwargs):
        packed = self.adapter.pack(x, timestep, context, **kwargs)
        cond_uuid = _cond_uuid_from_transformer_options(
            kwargs.get("transformer_options")
        )
        # Rows of one cond chunk share its uuid but can carry different
        # per-row conditioning (e.g. reference latents), so key on the row too.
        # unpack_ctx is executor-local, keeping _execute_packed's signature.
        packed.unpack_ctx["comfyui_cond_uuid"] = (
            None if cond_uuid is None else (cond_uuid, _cond_row)
        )
        return self._execute_packed(packed, x, timestep)
