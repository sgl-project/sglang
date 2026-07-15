"""sglang build hooks.

SGLANG_BUILD_RUST_EXTS controls which Rust extensions are built:
  - unset or "all": build every declared Rust extension (the default).
  - "none": build no Rust extensions.
  - comma-separated names: build only extensions whose target matches one of the
    given (case-insensitive) substrings, e.g. "grpc" matches
    "sglang.srt.grpc._core".

This is a build-time environment variable, so it is read directly from
os.environ instead of sglang.srt.environ, which is not available until after the
package has been built.
"""

import os

from setuptools import setup

try:
    from setuptools_rust import build_rust
except ModuleNotFoundError as exc:
    if exc.name != "setuptools_rust":
        raise
    # Alternate platform pyprojects do not declare Rust extensions.
    build_rust = None

_BUILD_RUST_EXTS_ENV = "SGLANG_BUILD_RUST_EXTS"


def _selected_rust_extensions(declared):
    """Return the Rust extensions selected by SGLANG_BUILD_RUST_EXTS.

    `ext.name` is the fully-qualified target (e.g. "sglang.srt.grpc._core") for
    the string-target declarations in pyproject.toml, so comma-separated names
    are matched as case-insensitive substrings of it.
    """
    declared = list(declared)
    raw = os.environ.get(_BUILD_RUST_EXTS_ENV)
    if raw is None:
        return declared

    spec = raw.strip().lower()
    # An empty or whitespace-only value is treated as unset (build everything).
    if not spec or spec == "all":
        return declared
    if spec == "none":
        return []

    tokens = [token.strip() for token in spec.split(",")]
    if not all(tokens):
        raise ValueError(
            f"{_BUILD_RUST_EXTS_ENV}={raw!r} has an empty item; unset it or use "
            "'all', 'none', or a comma-separated list of extension names"
        )

    matched = set()
    unmatched = []
    for token in tokens:
        hits = {ext.name for ext in declared if token in ext.name.lower()}
        if hits:
            matched |= hits
        else:
            unmatched.append(token)
    if unmatched:
        declared_names = sorted(ext.name for ext in declared)
        raise ValueError(
            f"{_BUILD_RUST_EXTS_ENV} matched no declared Rust extension for: "
            f"{unmatched}; declared extensions are {declared_names}"
        )

    return [ext for ext in declared if ext.name in matched]


if build_rust is not None:

    class BuildRust(build_rust):
        """Build only the Rust extensions selected by SGLANG_BUILD_RUST_EXTS."""

        def run(self) -> None:
            rust_extensions = _selected_rust_extensions(self.extensions or [])
            self.extensions = rust_extensions
            self.distribution.rust_extensions = rust_extensions
            if not rust_extensions:
                return
            super().run()

    _cmdclass = {"build_rust": BuildRust}
else:
    _cmdclass = {}


setup(cmdclass=_cmdclass)
