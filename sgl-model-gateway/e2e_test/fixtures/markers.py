"""Marker helper utilities for E2E tests.

This module provides helper functions for extracting values from pytest markers.
"""

from __future__ import annotations

from typing import Any

import pytest


def get_marker_value(
    request: pytest.FixtureRequest,
    marker_name: str,
    arg_index: int = 0,
    default: Any = None,
) -> Any:
    """Get a value from a pytest marker.

    Args:
        request: The pytest fixture request.
        marker_name: Name of the marker to look for.
        arg_index: Index of positional argument to extract.
        default: Default value if marker not found.

    Returns:
        The marker argument value or default.
    """
    marker = request.node.get_closest_marker(marker_name)
    if marker is None:
        return default
    if marker.args and len(marker.args) > arg_index:
        return marker.args[arg_index]
    return default


def get_marker_kwargs(
    request: pytest.FixtureRequest,
    marker_name: str,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get keyword arguments from a pytest marker.

    Args:
        request: The pytest fixture request.
        marker_name: Name of the marker to look for.
        defaults: Default values if marker not found or missing kwargs.

    Returns:
        Dict of keyword arguments merged with defaults.
    """
    result = dict(defaults) if defaults else {}
    marker = request.node.get_closest_marker(marker_name)
    if marker is not None:
        result.update(marker.kwargs)
    return result
