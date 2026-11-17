"""Resolution + prefix-length alignment for a FIXED system prompt.

The fixed system prompt is shared by every chat request. When `--system-prompt`
is set the value is either the literal prompt text or a path to a UTF-8 file
(an existing path is auto-detected). Everything in this module is inert when
`--system-prompt` is unset: callers must check ``resolve_system_prompt(...)``
for None before calling anything else, so the default code paths stay
byte-identical.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_system_prompt(server_args) -> Optional[str]:
    """Return the fixed system-prompt text, or None if the feature is disabled.

    The literal value is used as-is unless it names an existing file, in which
    case the file is read as UTF-8 (trailing whitespace stripped). Long content
    (large files) is fully supported because we only hold the text in memory.
    """
    value = getattr(server_args, "system_prompt", None)
    if value is None:
        return None
    if os.path.isfile(value):
        try:
            with open(value, "r", encoding="utf-8") as f:
                return f.read().rstrip()
        except OSError as e:
            raise ValueError(
                f"--system-prompt file '{value}' could not be read: {e}"
            ) from e
    return value


def aligned_prefix_len_down(prefix_len: int, chunk: int) -> int:
    """Round a raw prefix token count down to the previous ``chunk`` boundary.

    Linear-attention (SSM / Mamba) state only checkpoints at multiples of
    ``mamba_cache_chunk_size``, so the shared prefix must end exactly on a
    chunk multiple for its state to be reusable. Truncating DOWN (instead of
    padding up) keeps every request's full token stream identical to the
    native render — the removed tail tokens simply stay part of each
    request's private prefill, giving up only their cache reuse.
    """
    if chunk <= 0:
        return prefix_len
    return (prefix_len // chunk) * chunk


def _load_probe_tokenizer(server_args):
    """Load the tokenizer exactly as ``TokenizerManager.init_tokenizer_and_processor``
    will (same helpers, same kwargs), so the startup probe's verdict cannot
    diverge from the runtime probe in serving_chat. A probe that passed on a
    differently-loaded tokenizer would leave the mamba pool sized for the warm
    steady state while every request stays cold."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.utils.hf_transformers import (
        get_tokenizer,
        get_tokenizer_from_processor,
    )

    if ModelConfig.from_server_args(server_args).is_multimodal:
        # Multimodal: the runtime tokenizer is the processor's tokenizer.
        # Replicate get_processor_wrapper's call from the raw record fields —
        # the probe runs BEFORE the config is published, so the get_*()
        # context accessors are not up yet.
        from sglang.srt.utils.hf_transformers.processor import (
            get_processor,
            resolve_image_processor_backend,
        )

        processor = get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            image_processor_backend=resolve_image_processor_backend(server_args),
            tokenizer_backend=server_args.tokenizer_backend,
            model_name=server_args.model_path,
        )
        return get_tokenizer_from_processor(processor)
    return get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        tokenizer_backend=server_args.tokenizer_backend,
    )


def probe_system_prefix_or_disable(server_args) -> None:
    """Early gate for ``--system-prompt``: probe before any subprocess spawns.

    Every downstream consumer — the serving-chat inject/split, the scheduler
    boundary seeding, and (critically) the mamba pool sizing in
    kv_cache_configurator, which forces ratio 1 whenever
    ``server_args.system_prompt`` is set — keys off the flag alone. But the
    feature is only actually live when the tokenizer-side probe succeeds AND
    the system block spans at least one mamba chunk (truncate-down). If the
    probe only failed later at request time, the pool would already be sized
    for the warm steady state (1 slot per request) while every request stays
    cold (3 slots each) — the over-admission crash this gate prevents.

    So run the probe here, in the main process BEFORE schedulers fork; on any
    failure clear ``server_args.system_prompt`` so every consumer sees a
    consistent feature-off state and the server runs the byte-identical native
    path. An unreadable prompt file is a configuration error and keeps
    ``resolve_system_prompt``'s fail-fast ValueError.
    """
    if server_args.system_prompt is None:
        return
    if server_args.skip_tokenizer_init:
        # No tokenizer at runtime either — the serving-chat probe could never
        # run, so disable here to keep pool sizing consistent with behavior.
        logger.error(
            "--system-prompt: skip_tokenizer_init leaves no tokenizer to render "
            "the fixed system prompt; disabling (native path, no prefix reuse)"
        )
        server_args.system_prompt = None
        return
    system_text = resolve_system_prompt(server_args)
    try:
        tokenizer = _load_probe_tokenizer(server_args)
    except Exception as e:
        logger.error(
            "--system-prompt: startup probe could not load the tokenizer (%s); "
            "disabling the fixed system prompt (native path, no prefix reuse)",
            e,
        )
        # Bare assignment is the sanctioned mutation here: the probe runs in
        # Engine init BEFORE the config is published (and before the
        # subprocesses fork), so runtime_context.override() is not available
        # yet and the record is not frozen.
        server_args.system_prompt = None
        return
    # Mirror serving_chat's special-token detection so this probe sees the same
    # encode behavior as the runtime probe.
    try:
        auto_adds_specials = len(tokenizer.encode("")) > 0
    except Exception:
        auto_adds_specials = True
    encode_kwargs = {"add_special_tokens": False} if auto_adds_specials else {}

    # Local import: token_seam imports this module, so a top-level import here
    # would be circular. Tests pre-seed the chunk on the args record; the real
    # record has no such attribute pre-resolution, so fall through to the
    # override (which reads the HF config).
    chunk = getattr(server_args, "mamba_cache_chunk_size", None)
    if chunk is None:
        from sglang.srt.arg_groups.overrides import mamba_cache_chunk_size

        chunk = mamba_cache_chunk_size(server_args)
    from sglang.srt.system_prompt.token_seam import SystemPrefix

    prefix = SystemPrefix(tokenizer, system_text)
    prefix.prepare(chunk, encode_kwargs)
    if not prefix.enabled:
        logger.error(
            "--system-prompt: startup probe rejected the fixed prefix (identity "
            "probe failed, or the system block is shorter than one mamba "
            "chunk=%d); disabling the fixed system prompt (native path, no "
            "prefix reuse)",
            chunk,
        )
        server_args.system_prompt = None
        return
    logger.info(
        "--system-prompt startup probe OK: shared prefix=%d tokens, per-request "
        "tail=%d tokens (chunk=%d); mamba pool may use ratio 1",
        prefix.boundary_len,
        prefix.suffix_len,
        chunk,
    )