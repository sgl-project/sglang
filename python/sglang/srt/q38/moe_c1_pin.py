"""Three MoE cache entries at native path resolution; no forward-path hooks."""
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)
_done = set()

def pin_native_path(result, model_runner):
    path = Path(result)
    if path in _done:
        return result
    # The original native function already resolved model_path, target dtype,
    # global quant/backend/parallel widths, actual HF class, skip_ops and draft
    # quantization. Never substitute a guessed model hash or model-path alias.
    try:
        pins = json.loads(Path(os.environ['Q38_MOE_C1_PIN_FILE']).read_text())
        from flashinfer.autotuner import _collect_metadata
        current = _collect_metadata()
        if current != pins['_metadata']:
            log.warning('Q38 MoE pin SKIPPED: installed metadata differs from R11 seed')
            _done.add(path)
            return result
        data = json.loads(path.read_text()) if path.exists() else {}
        if data.get('_metadata', current) != current:
            log.warning('Q38 MoE pin SKIPPED: existing cache metadata differs: %s', path)
            _done.add(path)
            return result
        data['_metadata'] = current
        # Target and draft load into one singleton. Both files must contain the
        # same three pins, or the later draft load can restore stale target data.
        data.update(pins['target'])
        data.update(pins['draft'])
        tmp = path.with_name(path.name + f'.moe-pin-{os.getpid()}')
        tmp.write_text(json.dumps(data) + '\n')
        os.replace(tmp, path)
        log.info('Q38 MoE pin APPLIED policy=r11-seed-c1-3keys draft=%s path=%s',
                 model_runner.is_draft_worker, path)
        _done.add(path)
    except (OSError, ValueError) as exc:
        # This optional calibration must not prevent a normal service boot.
        log.warning('Q38 MoE pin SKIPPED: %s', exc)
    return result

def register():
    if not os.environ.get('Q38_MOE_C1_PIN_FILE'):
        return
    # Import explicitly so HookRegistry does not rely on package attribute
    # population during frontend versus scheduler import ordering.
    import sglang.srt.model_executor.runner.flashinfer_autotune
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        'sglang.srt.model_executor.runner.flashinfer_autotune.flashinfer_autotune_cache_path',
        pin_native_path, HookType.AFTER)
