"""
Factory pattern implementation with decorator-based registration.

This module provides a flexible factory pattern that allows classes to be
registered using the @Factory.register decorator.
"""

import functools
from typing import Any, Callable, Dict, Optional, Type


class Factory:
    def __init__(self):
        """Initialize the factory with an empty registry."""
        self._registry: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        """
        Decorator to register a class with the factory.

        Args:
            name: The name to register the class under.

        Returns:
            The decorator function that registers the class.

        Raises:
            ValueError: If the name is already registered.
        """

        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"Name '{name}' is already registered with class {self._registry[name]}"
                )

            self._registry[name] = cls
            return cls

        return decorator

    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create an instance of the registered class.

        Args:
            name: The name of the class to create.
            *args: Positional arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            An instance of the registered class.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._registry:
            available_names = list(self._registry.keys())
            raise KeyError(
                f"Name '{name}' not found in registry. Available names: {available_names}"
            )

        cls = self._registry[name]
        return cls(*args, **kwargs)
