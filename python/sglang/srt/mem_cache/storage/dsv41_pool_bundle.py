"""Immutable page generations for the actual native DSV4 C1/C2 pool plan.

A manifest is the sole commit point. Bare legacy pool keys never establish a
bundle, and readers keep the selected generation for the entire logical page.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Dsv41PoolBundle:
    names: tuple[str, ...]
    fingerprint: str

    schema = "dsv41-pool-bundle-v1"

    @classmethod
    def from_plan(cls, plan):
        names = tuple(sorted(str(name) for name in plan.pool_names))
        fingerprint = plan.fingerprint
        if not names or len(names) != len(set(names)):
            raise ValueError("A persistent bundle requires unique physical pools")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("A persistent bundle requires a SHA256 pool plan")
        return cls(names, fingerprint)

    def _key(self, *parts):
        descriptor = [self.schema, self.fingerprint, *parts]
        return hashlib.sha256(
            json.dumps(descriptor, separators=(",", ":")).encode()
        ).hexdigest()

    def manifest_key(self, logical_key, namespace):
        return "dsv41b1m_" + self._key(logical_key, namespace)

    def object_key(self, logical_key, namespace, generation, pool):
        if pool not in self.names or not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("Invalid persistent bundle generation or pool")
        return "dsv41b1o_" + self._key(logical_key, namespace, generation, pool)

    def encode_manifest(self, logical_key, generation, objects):
        manifest = dict(
            schema=self.schema,
            fingerprint=self.fingerprint,
            key=logical_key,
            generation=generation,
            objects=objects,
        )
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        if self.decode_manifest(payload, logical_key) is None:
            raise ValueError("Invalid persistent bundle manifest")
        return payload

    def decode_manifest(self, payload, logical_key):
        try:
            manifest = json.loads(payload)
            if (
                set(manifest)
                != {"schema", "fingerprint", "key", "generation", "objects"}
                or manifest["schema"] != self.schema
                or manifest["fingerprint"] != self.fingerprint
                or manifest["key"] != logical_key
                or not re.fullmatch(r"[0-9a-f]{32}", manifest["generation"])
                or set(manifest["objects"]) != set(self.names)
            ):
                return None
            for obj in manifest["objects"].values():
                if (
                    set(obj) != {"size", "sha256"}
                    or type(obj["size"]) is not int
                    or obj["size"] <= 0
                    or not re.fullmatch(r"[0-9a-f]{64}", obj["sha256"])
                ):
                    return None
            return manifest
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
