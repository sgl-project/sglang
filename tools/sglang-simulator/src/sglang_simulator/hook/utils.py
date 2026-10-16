def get_obj_from_args(type_name: str, *args, **kwargs):
    for obj in args:
        if type_name == f"{type(obj).__module__}.{type(obj).__name__}":
            return obj
    for obj in kwargs.values():
        if type_name == f"{type(obj).__module__}.{type(obj).__name__}":
            return obj
    return None


def get_hook_target(target, name: str):
    """The method `name` of the hooked class, failing loudly when SGLang moved it."""
    method = getattr(target, name, None)
    if method is None:
        raise RuntimeError(
            f"SGLang Simulator could not hook {target.__module__}."
            f"{target.__qualname__}.{name}. The simulator must be adapted to "
            "this SGLang revision."
        )
    return method
