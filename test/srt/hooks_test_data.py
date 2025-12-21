import contextvars

# Context-local recorder; default is None
_hook_recorder = contextvars.ContextVar("hook_recorder", default=None)


def set_recorder(recorder):
    """Set the recorder callable for this context; return token for reset."""
    return _hook_recorder.set(recorder)


def reset_recorder(token):
    """Reset the recorder to its previous value."""
    _hook_recorder.reset(token)


def dummy_hook_factory(config):
    """Factory that returns a forward hook capturing a tag from config."""
    tag = config.get("tag", "default")

    def hook(module, inputs, output):
        recorder = _hook_recorder.get()
        if recorder is not None:
            recorder(
                {
                    "module_type": type(module).__name__,
                    "tag": tag,
                    "shape": tuple(output.shape),
                }
            )
        return output

    return hook
