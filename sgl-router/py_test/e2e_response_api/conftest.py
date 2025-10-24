"""
pytest configuration for e2e_response_api tests.

This configures pytest to not collect base test classes that are meant to be inherited.
"""

import pytest  # noqa: F401


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to exclude base test classes.

    Base test classes are meant to be inherited, not run directly.
    We exclude any test that comes from these base classes:
    - StateManagementBaseTest
    - ResponseCRUDBaseTest
    - ConversationCRUDBaseTest
    - MCPTests
    - StateManagementTests
    """
    base_class_names = {
        "StateManagementBaseTest",
        "ResponseCRUDBaseTest",
        "ConversationCRUDBaseTest",
        "MCPTests",
        "StateManagementTests",
    }

    # Filter out tests from base classes
    filtered_items = []
    for item in items:
        # Check if the test's parent class is a base class
        parent_name = item.parent.name if hasattr(item, "parent") else None
        if parent_name not in base_class_names:
            filtered_items.append(item)

    # Update items list
    items[:] = filtered_items
