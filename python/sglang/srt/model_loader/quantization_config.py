"""Model-declared names for quantization metadata queries.

This module translates names only. Matching rules belong to each config backend.
"""

from collections.abc import Mapping


class CheckpointNames:
    def __init__(self):
        self.rules: dict[tuple[str, str], str] = {}
        self.forward_rules: dict[tuple[str, str], str] = {}

    def add(self, prefix: str, external: str, registered: str):
        key = (prefix, registered)
        forward_key = (prefix, external)
        if key in self.rules and self.rules[key] != external:
            raise ValueError(f"Conflicting checkpoint names for {prefix}.{registered}")
        if (
            forward_key in self.forward_rules
            and self.forward_rules[forward_key] != registered
        ):
            raise ValueError(f"Conflicting checkpoint names for {prefix}.{external}")
        self.rules[key] = external
        self.forward_rules[forward_key] = registered

    def checkpoint_name(self, name: str | None) -> str | None:
        if name is None or not self.rules:
            return name
        parts = name.split(".")
        return ".".join(
            self.rules.get((".".join(parts[:i]), part), part)
            for i, part in enumerate(parts)
        )

    def internal_name(self, name: str) -> str:
        parts = []
        for component in name.split("."):
            parts.append(
                self.forward_rules.get((".".join(parts), component), component)
            )
        return ".".join(parts)


class CheckpointMetadata(Mapping):
    """Read exact metadata keys through a config's current model declarations.

    The source stays unchanged; declarations added during construction are visible
    without copying large expert metadata dictionaries for each layer.
    """

    def __init__(self, source: Mapping, config):
        self.source = source
        self.config = config

    def __getitem__(self, name):
        return self.config.match_layer(name, self.source.__getitem__)

    def __iter__(self):
        names = self.config._checkpoint_names
        if names is None:
            return iter(self.source)
        return (names.internal_name(name) for name in self.source)

    def __len__(self):
        return len(self.source)
