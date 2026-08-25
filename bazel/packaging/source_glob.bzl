"""Source globs for the transitional PEP 517 action."""

_GENERATED_EXCLUDES = [
    "**/*.egg-info/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.so",
    "**/__pycache__/**",
    "**/build/**",
    "**/dist/**",
    "**/target/**",
]

def pep517_source_glob(include):
    """Globs source inputs while excluding routine ignored build products."""
    return native.glob(
        include,
        allow_empty = True,
        exclude = _GENERATED_EXCLUDES,
    )
