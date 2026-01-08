import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--parser-source",
        action="store",
        default="default",
        help="Parser source to use. Options: 'default', 'legacy_v3', 'legacy_vllm' or 'module.path:ClassName'",
    )
    parser.addoption(
        "--parser-mode",
        action="store",
        default="both",
        help="Parser mode: 'both', 'streaming', or 'nonstream'",
    )
