import types

class TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "3.2.0"
        self.__path__ = []
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.Config = self._dummy_decorator("Config")
        self.language = TritonLanguagePlaceholder()
        self.cdiv = lambda a, b: (a + b - 1) // b

    def _dummy_decorator(self, name):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f
        return decorator

class TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = None
        self.dtype = None
        self.int64 = None
        self.int32 = None
        self.tensor = None
        self.exp = None
        self.exp2 = None
        self.log = None
        self.log2 = None
        self.where = None

    def __getattr__(self, name):
        # Gracefully handle any arbitrary `tl.*` access
        return None
