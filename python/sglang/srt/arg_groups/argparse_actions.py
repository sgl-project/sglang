import argparse
import json
import logging

logger = logging.getLogger(__name__)


class LoRAPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        lora_paths = []
        if values:
            assert isinstance(values, list), "Expected a list of LoRA paths."
            for lora_path in values:
                lora_path = lora_path.strip()
                if lora_path.startswith("{") and lora_path.endswith("}"):
                    obj = json.loads(lora_path)
                    assert "lora_path" in obj and "lora_name" in obj, (
                        f"{repr(lora_path)} looks like a JSON str, "
                        "but it does not contain 'lora_name' and 'lora_path' keys."
                    )
                    lora_paths.append(obj)
                else:
                    lora_paths.append(lora_path)

        setattr(namespace, self.dest, lora_paths)


def print_deprecated_warning(message: str):
    logger.warning(f"\033[1;33m{message}\033[0m")


class DeprecatedAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(DeprecatedAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print_deprecated_warning(
            f"The command line argument '{option_string}' is deprecated and will be removed in future versions."
        )


class DeprecatedStoreTrueAction(argparse.Action):
    """Deprecated flag that still stores True and prints a warning."""

    def __init__(
        self,
        option_strings,
        dest,
        new_flag=None,
        nargs=0,
        const=True,
        default=False,
        **kwargs,
    ):
        self.new_flag = new_flag
        super().__init__(
            option_strings, dest, nargs=nargs, const=const, default=default, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        replacement = f" Use '{self.new_flag}' instead." if self.new_flag else ""
        print_deprecated_warning(
            f"'{option_string}' is deprecated and will be removed in a future release.{replacement}"
        )
        setattr(namespace, self.dest, True)


class DeprecatedAliasStoreAction(argparse.Action):
    """Deprecated alias that stores its value and prints a warning."""

    def __init__(self, option_strings, dest, new_flag=None, **kwargs):
        self.new_flag = new_flag
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        replacement = f" Use '{self.new_flag}' instead." if self.new_flag else ""
        print_deprecated_warning(
            f"'{option_string}' is deprecated and will be removed in a future release.{replacement}"
        )
        setattr(namespace, self.dest, values)
