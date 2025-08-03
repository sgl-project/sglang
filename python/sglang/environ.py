import os
import warnings
from typing import Any


class EnvField:
    def __init__(self, default: Any):
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self, default: Any = None) -> Any:
        value = os.getenv(self.name)
        if value is None:
            return default if default is not None else self.default
        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def set(self, value: Any):
        assert value is not None, "Cannot set None value for environment variable"
        os.environ[self.name] = str(value)

    def clear(self):
        os.environ.pop(self.name, None)

    @property
    def value(self):
        return self.get()


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
    envs.SGLANG_TEST_RETRACT.set(not envs.SGLANG_TEST_RETRACT.value)
    print(f"{envs.SGLANG_TEST_RETRACT.value=}")
