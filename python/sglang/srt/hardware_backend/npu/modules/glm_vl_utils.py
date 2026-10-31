"""Concrete-class patch helpers shared by GLM processors."""


def _patch_processor_method(processor, method: str, wrapper, marker: str) -> bool:
    """Patch the actual class, including remote classes; report supported APIs."""
    processor_type = type(processor)
    original = getattr(processor_type, method, None)
    if not callable(original):
        return False
    if getattr(original, marker, False):
        return True

    patched = wrapper(original)
    setattr(patched, marker, True)
    setattr(processor_type, method, patched)
    return True
