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
        self.cdiv = lambda a, b: (a + b - 1) // b

    def _dummy_decorator(self, *args, **kwargs):
        """A decorator that passes through the wrapped function unchanged."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    def __getattr__(self, name):
        """Return a new TritonPlaceholder for any unknown attribute.
        This handles arbitrary chains like triton.backends.compiler.AttrsDescriptor
        """
        child = TritonPlaceholder(f"triton.{name}")
        # Cache it to avoid creating new objects on every access
        object.__setattr__(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        """Allow instances to be called as decorators or class constructors."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
