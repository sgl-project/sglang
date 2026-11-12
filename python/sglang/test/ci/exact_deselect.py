import json
import os


def pytest_collection_modifyitems(session, config, items):
    nodes = set(json.loads(os.environ.get("PPU_CI_EXACT_DESELECT", "[]")))
    if nodes:
        items[:] = [item for item in items if item.nodeid not in nodes]
