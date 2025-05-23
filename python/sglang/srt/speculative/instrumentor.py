import inspect
from functools import wraps


def print_shapes(*param_paths: str):
    """
    Decorator for **bound instance methods**.

    Parameters
    ----------
    *param_paths : str
        One or more dotted paths identifying which argument (and optional
        sub-attributes) should have its ``.shape`` printed when the method
        runs.
        • The first segment must be the *argument name* as it appears in the
          method signature.
        • Subsequent segments are traversed with ``getattr``.

    Example
    -------
    @print_shapes("x", "batch.images")
    def forward(self, x, batch): ...
    """

    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # 1️⃣  print the method’s qualified name (class.method)
            print(f"→ {func.__qualname__}")

            # 2️⃣  walk each requested path and print its .shape
            for path in param_paths:
                segments = path.split(".")
                name, attrs = segments[0], segments[1:]

                if name not in bound.arguments:
                    print(f"   [warn] '{name}' not passed")
                    continue

                obj = bound.arguments[name]
                for attr in attrs:
                    try:
                        attr_ = attr[:-1] if attr.endswith("!") else attr
                        obj = getattr(obj, attr_)
                        if obj is None:
                            break
                    except AttributeError:
                        print(f"   [warn] '{path}' missing attr '{attr}'")
                        obj = None
                        break

                if obj is None:
                    # print(f"   [warn] '{path}' is None")
                    continue

                if attr.endswith("!"):
                    shape = None
                else:
                    shape = getattr(obj, "shape", None) if obj is not None else None

                if shape is None:
                    print(f"   {path} = {obj}")
                else:
                    print(f"   {path}.shape = {shape}")
            # run the original method
            return func(*args, **kwargs)

        return wrapper

    return decorator
