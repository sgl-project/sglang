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


# Retiring a flag comes in four shapes, and which one you need depends on what
# the flag was and what replaced it:
#
#   * the flag is gone and there is no automatic translation
#       -> `DeprecatedAction` with `error_message=`, which stops the launch and
#          names the replacement;
#   * an old boolean whose field survives, possibly renamed
#       -> `DeprecatedStoreTrueAction`;
#   * an old boolean replaced by one *value* of a new valued flag
#       -> `DeprecatedStoreConstAction` with `const_value=`;
#   * an old valued flag replaced by a renamed valued flag
#       -> `DeprecatedAliasStoreAction`.
#
# Only the second has a registration today (`--disable-cuda-graph`). The other
# three are kept because the shapes recur -- this package has retired flags of
# every one of them -- and the fiddly parts (`nargs=0` on a boolean, where the
# const goes, warn-and-continue versus `parser.error`) are what a
# reimplementation gets wrong. Pass `new_flag=` so the warning tells the
# operator what to switch to; that pointer is the whole point.


class DeprecatedAction(argparse.Action):
    """A retired flag with no automatic translation: stop and say so.

    `error_message` should name the replacement, because a bare "unrecognized
    arguments" leaves the operator guessing. Without one it warns and continues,
    which suits a flag that has become a no-op rather than a rename.
    """

    def __init__(self, option_strings, dest, error_message=None, nargs=0, **kwargs):
        self.error_message = error_message
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if self.error_message is not None:
            parser.error(self.error_message)
        print_deprecated_warning(
            f"The command line argument '{option_string}' is deprecated and "
            "will be removed in future versions."
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


class DeprecatedStoreConstAction(argparse.Action):
    """An old boolean whose replacement is one *value* of a valued flag.

    The bool-to-enum migration: the operator passes no value, and the action
    writes the fixed one `const_value` names onto the new field. `nargs=0`
    because the old spelling took no argument.
    """

    def __init__(
        self,
        option_strings,
        dest,
        new_flag=None,
        const_value=None,
        nargs=0,
        default=None,
        **kwargs,
    ):
        self.new_flag = new_flag
        self.const_value = const_value
        super().__init__(option_strings, dest, nargs=nargs, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        replacement = f" Use '{self.new_flag}' instead." if self.new_flag else ""
        print_deprecated_warning(
            f"'{option_string}' is deprecated and will be removed in a future "
            f"release.{replacement}"
        )
        setattr(namespace, self.dest, self.const_value)


class DeprecatedAliasStoreAction(argparse.Action):
    """An old valued flag renamed: keep the value, move it to the new dest."""

    def __init__(self, option_strings, dest, new_flag=None, **kwargs):
        self.new_flag = new_flag
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        replacement = f" Use '{self.new_flag}' instead." if self.new_flag else ""
        print_deprecated_warning(
            f"'{option_string}' is deprecated and will be removed in a future "
            f"release.{replacement}"
        )
        setattr(namespace, self.dest, values)
