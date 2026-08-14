import types


class TritonPlaceholder(types.ModuleType):
    """A dummy Triton module for CPU platforms where triton is not installed.

    This class acts as a no-op placeholder that gracefully handles any
    attribute access, decorator usage, or submodule import at any depth.
    This allows SGLang to boot on platforms like ppc64le without Triton.
    """

    def __init__(self, name="triton"):
        super().__init__(name)
        self.__version__ = "3.2.0"
        self.__path__ = []
        # Must be a str (not None) so that inspect.getmodule/os.path.splitext
        # doesn't crash when iterating sys.modules.
        self.__file__ = f"<{name}-placeholder>"
        self.__spec__ = None
        self.__loader__ = None
        self.__package__ = name
        self.cdiv = lambda a, b: (a + b - 1) // b

    def _dummy_decorator(self, *args, **kwargs):
        """A decorator that passes through the wrapped function unchanged."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    def __getattr__(self, name):
        """Return a new TritonPlaceholder for regular attributes.
        Returns None for dunder attributes to prevent inspect module errors
        (e.g. inspect.getabsfile expects __file__ to be a str or None).
        """
        if name.startswith("__") and name.endswith("__"):
            # Return None for unknown dunder attrs so inspect/os.path don't break
            return None
        
        # If the requested attribute is Capitalized (like Config, JITFunction),
        # return an actual class that can be safely subclassed by PyTorch.
        if name and name[0].isupper():
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                def __getattr__(self, item):
                    return None
            child = DummyClass
        else:
            child = TritonPlaceholder(f"{self.__package__}.{name}")
            
        # Cache it to avoid creating new objects on every access
        object.__setattr__(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        """Allow instances to be called as decorators or class constructors."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    def __contains__(self, item):
        """Support 'x in triton.backends.backends' style checks — always False."""
        return False

    def __iter__(self):
        """Support iteration over placeholder objects — always empty."""
        return iter([])

    def __bool__(self):
        """Evaluate as False in boolean context (e.g. if triton.backends: ...)."""
        return False
