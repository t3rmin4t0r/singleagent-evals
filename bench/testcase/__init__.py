"""Auto-discovery registry for test cases.

Scans testcase/*/testcase.py for TestCase subclasses. Adding a test case =
creating a directory with testcase.py containing a TestCase subclass.
"""

import importlib
import pkgutil
from pathlib import Path

from bench.testcase.base import TestCase

_registry: dict[str, type[TestCase]] | None = None


def _discover() -> dict[str, type[TestCase]]:
    """Scan sub-packages for TestCase subclasses."""
    registry: dict[str, type[TestCase]] = {}
    package_dir = Path(__file__).resolve().parent

    for info in pkgutil.iter_modules([str(package_dir)]):
        if not info.ispkg:
            continue
        module_path = f"bench.testcase.{info.name}.testcase"
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError:
            continue
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, TestCase)
                and attr is not TestCase
            ):
                instance = attr()
                registry[instance.name] = type(instance)
    return registry


def list_testcases() -> list[str]:
    """Return sorted list of available test case names."""
    global _registry
    if _registry is None:
        _registry = _discover()
    return sorted(_registry.keys())


def get_testcase(name: str) -> TestCase:
    """Get a test case instance by name."""
    global _registry
    if _registry is None:
        _registry = _discover()
    cls = _registry.get(name)
    if cls is None:
        available = ", ".join(list_testcases())
        raise ValueError(f"Unknown test case: {name}. Available: {available}")
    return cls()
