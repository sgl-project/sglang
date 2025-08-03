import os
import warnings
from typing import Any


class EnvField:
    def __init__(self, default: Any):
        self.default = default
        # NOTE: we use None to indicate whether the value is set or not
        # If the value is manually set to None, we need mark it as _set_to_none.
        # Always use clear() to reset the value, which leads to the default fallback.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Any:
        value = os.getenv(self.name)
        if self._set_to_none:
            assert value is None
            return None

        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def set(self, value: Any):
        if value is None:
            self._set_to_none = True
            os.environ.pop(self.name, None)
        else:
            self._set_to_none = False
            os.environ[self.name] = str(value)

    def clear(self):
        os.environ.pop(self.name, None)

    @property
    def value(self):
        return self.get()


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class Envs:
    # fmt: off
    SGLANG_TEST_RETRACT = EnvBool(False)
    # fmt: on


envs = Envs()


def _convert_SGL_to_SGLANG():
    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_")
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()

if __name__ == "__main__":
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    print(f"{envs.SGLANG_TEST_RETRACT.value=}")
    envs.SGLANG_TEST_RETRACT.set(None)
    print(f"{envs.SGLANG_TEST_RETRACT.value=}")
    envs.SGLANG_TEST_RETRACT.set(True)
    print(f"{envs.SGLANG_TEST_RETRACT.value=}")
