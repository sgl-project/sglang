"""Select declared AIC datasets; numerical algorithms remain in the AIC SDK."""

import copy
import hashlib
import importlib
import json
from pathlib import Path

from .aic_compat import AICSDK
from .aic_costs import input_identity, validate_quote

_PATH_KEYS = {
    "root",
    "pythonpath",
    "dataset_root",
    "native_index",
    "native_profile_bindings",
    "community_systems_root",
    "legacy_systems_root",
    "path",
    "compat_pythonpath",
}


def load_aic_config(path, *, repository_root=None):
    """Read repository-root or legacy package-parent path-anchored config.

    No Python resolver from the selected checkout is executed. The contract is
    owned by the installed SDK's dataset_config loader for repository configs;
    the package-parent compatibility path is data-only.
    """
    path = Path(path)
    repository = Path(repository_root).resolve(strict=True) if repository_root else None
    if not path.is_absolute() and repository is not None:
        path = repository / path
    path = path.resolve(strict=True)
    value = json.loads(path.read_text())
    anchor = value.get("paths_relative_to")
    if anchor == "repository root":
        if repository is None or not path.is_relative_to(repository):
            raise ValueError("AIC_CONFIG_REPOSITORY_ROOT_REQUIRED")
        # The installed SDK owns the repository publication contract and pins.
        # Do not execute a resolver script found beside arbitrary input JSON.
        loader = importlib.import_module("aiconfigurator_core.sdk.dataset_config")
        return loader.load_config(path, repository_root=repository)
    elif anchor == "SDK_CONFIG.json parent":
        base = path.parent
    else:
        raise ValueError("AIC_CONFIG_PATH_ANCHOR_UNSUPPORTED")

    def resolve(obj):
        if isinstance(obj, list):
            return [resolve(item) for item in obj]
        if not isinstance(obj, dict):
            return obj
        result = {}
        for key, item in obj.items():
            if key in _PATH_KEYS and isinstance(item, str):
                part = Path(item)
                if part.is_absolute() or ".." in part.parts:
                    raise ValueError("AIC_CONFIG_PATH_MUST_BE_CONTAINED")
                target = (base / part).resolve(strict=anchor == "repository root")
                if not target.is_relative_to(base):
                    raise ValueError("AIC_CONFIG_PATH_MUST_BE_CONTAINED")
                result[key] = str(target)
            else:
                result[key] = resolve(item)
        return result

    result = resolve(value)
    for item in result.get("input_files", []):
        payload = Path(item["path"]).read_bytes()
        if (
            len(payload) != item["bytes"]
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise ValueError("AIC_CONFIG_PINNED_INPUT_CHANGED")
    return result


class AICProviders:
    """Resolve explicit core/module/PPU/external/formula requests without fallback.

    Dataset membership and allowed qualifications are separate: a dataset that
    contains conditional records does not itself authorize their consumption.
    """

    def __init__(
        self,
        config_path,
        *,
        api="core",
        allowed_qualifications=("matched",),
        allow_unknown_source=False,
        allow_archives=False,
        repository_root=None,
        external_providers=None,
    ):
        # Extensions are explicitly supplied by trusted Python callers. Input
        # JSON cannot select an import path or instantiate an external SDK.
        self.external_providers = dict(external_providers or {})
        if any(
            not isinstance(name, str) or not name or not callable(resolver)
            for name, resolver in self.external_providers.items()
        ):
            raise ValueError("AIC_EXTERNAL_PROVIDER_REGISTRATION_INVALID")
        path = Path(config_path)
        if not path.is_absolute() and repository_root:
            path = Path(repository_root) / path
        self.path = path.resolve(strict=True)
        self.config = load_aic_config(self.path, repository_root=repository_root)
        self.sdk = AICSDK(api)
        self.policy = dict(
            allowed_qualifications=tuple(allowed_qualifications),
            allow_unknown_source=allow_unknown_source,
        )
        self.allow_archives = allow_archives
        self.cache = {}
        self.inputs = {str(self.path)}
        declared = self.config.get("runtime", {}).get("pythonpath")
        if declared:
            package = importlib.import_module(self.sdk.package)
            if not Path(package.__file__).resolve().is_relative_to(Path(declared)):
                raise ValueError("AIC_RUNTIME_IMPORT_ROOT_MISMATCH")

    def _setting(self, key, request):
        value = self.config
        for part in key.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
        if isinstance(value, dict) and value.get("active") is False:
            if (
                not self.allow_archives
                or request.get("purpose") != "historical_control"
            ):
                raise ValueError("AIC_ARCHIVE_REQUIRES_EXPLICIT_HISTORICAL_CONTROL")
        return value

    def _module(self, dataset, request, *, ppu=False, kv=False):
        if self.sdk.api != "core":
            raise ValueError("AIC_MODULE_PROVIDER_REQUIRES_CORE")
        settings = self._setting(dataset, request)
        key = (dataset, ppu, kv)
        if key not in self.cache:
            if ppu:
                cls = self.sdk.module("ppu_modules").PPUModuleResolver
                selected = dict(
                    settings,
                    allowed_qualifications=self.policy["allowed_qualifications"],
                    allow_conditional_kv_interpolation=kv,
                )
                self.cache[key] = cls.from_config(selected)
            else:
                cls = self.sdk.module("module_costs").ModuleCostDatabase
                self.cache[key] = cls(
                    settings["root"],
                    manifest_sha256=settings["manifest_sha256"],
                    allowed_qualifications=self.policy["allowed_qualifications"],
                )
            self.inputs.add(str(Path(settings["root"]) / "DATASET_MANIFEST.json"))
        return self.cache[key]

    def _database(self, request, plan):
        root = self._setting(plan.get("dataset", "community_systems_root"), request)
        if not isinstance(root, str):
            raise ValueError("AIC_CORE_REQUIRES_SYSTEMS_ROOT")
        version, backend = plan["version"], plan["backend"]
        conditions = request["conditions"]
        if (
            conditions.get("data_version") != version
            or conditions.get("backend") != backend
        ):
            raise ValueError("AIC_CORE_VERSION_OR_BACKEND_BINDING_MISMATCH")
        mode = plan.get("database_mode", "SILICON")
        if mode not in ("SILICON", "HYBRID", "EMPIRICAL", "SOL", "SOL_FULL"):
            raise ValueError("AIC_DATABASE_MODE_INVALID")
        key = (root, request["system"], backend, version, mode)
        if key not in self.cache:
            database = self.sdk.database(
                request["system"],
                backend,
                version,
                systems_paths=[root],
                database_mode=mode,
            )
            if database is None or database.version != version:
                raise ValueError("AIC_REQUESTED_DATABASE_VERSION_UNAVAILABLE")
            self.cache[key] = database
        return self.cache[key], root

    def __call__(self, request):
        request = copy.deepcopy(request)
        plan = request["resolution"]
        provider = plan["provider"]
        if provider == "module":
            db = self._module(plan["dataset"], request)
            quote = db.query(
                request["op"],
                query=request["query"],
                conditions=request["conditions"],
                expected_scope=request["scope"],
            )
        elif provider == "ppu":
            kv = plan.get("allow_conditional_kv_interpolation", False)
            if type(kv) is not bool:
                raise ValueError("AIC_PPU_KV_POLICY_MUST_BE_BOOLEAN")
            db = self._module(plan["dataset"], request, ppu=True, kv=kv)
            if "curve_id" in plan:
                quote = db.curve(
                    plan["curve_id"],
                    plan["coordinate"],
                    conditions=request["conditions"],
                    expected_scope=request["scope"],
                )
            else:
                quote = db.query_attention(
                    query=request["query"],
                    conditions=request["conditions"],
                    expected_scope=request["scope"],
                )
        elif provider == "external":
            name = plan.get("name")
            if not isinstance(name, str) or name not in self.external_providers:
                raise ValueError("AIC_EXTERNAL_PROVIDER_NOT_REGISTERED")
            quote = self.external_providers[name](request)
        elif provider == "formula":
            if self.sdk.api != "core":
                raise ValueError("AIC_FORMULA_PROVIDER_REQUIRES_CORE")
            root = self._setting(plan["dataset"], request)
            if (
                request["conditions"].get("backend") != plan["backend"]
                or request["conditions"].get("data_version") != plan["version"]
            ):
                raise ValueError("AIC_FORMULA_VERSION_OR_BACKEND_BINDING_MISMATCH")
            device = Path(root) / (request["system"] + ".yaml")
            identity = input_identity([device])[0]
            key = ("formula", root, request["system"], plan["backend"], plan["version"])
            if key not in self.cache:
                self.cache[key] = self.sdk.module(
                    "analytical_costs"
                ).CoreFormulaProvider(
                    root,
                    system=request["system"],
                    backend=plan["backend"],
                    version=plan["version"],
                    system_yaml_sha256=identity["sha256"],
                )
            quote = self.cache[key].query(
                request["op"],
                query=request["query"],
                conditions=request["conditions"],
                scope=request["scope"],
                formula=plan["formula"],
                parameters=plan["parameters"],
                assumptions=plan.get("assumptions", []),
            )
        elif provider == "core":
            db, root = self._database(request, plan)
            raw = self.sdk.query(
                db, plan["operation"], systems_path=root, **request["query"]
            )
            source = str(raw["sdk_source"]).lower()
            if source not in ("silicon", "estimated", "sol", "empirical"):
                raise ValueError("AIC_SCALAR_SOURCE_UNKNOWN")
            measured = source == "silicon"
            device = Path(root) / (request["system"] + ".yaml")
            origin = input_identity([device])[0]
            quote = dict(
                schema="aic.cost_quote.v1",
                op=request["op"],
                system=request["system"],
                latency_ms=raw["latency_ms"],
                source="hardware_measurement" if measured else "analytical_assumption",
                method="table_lookup_unknown" if measured else "formula",
                qualification=dict(
                    status="conditional",
                    reasons=[
                        "Scalar SDK does not certify kernel, Graph or serving boundary matching"
                    ],
                ),
                query=request["query"],
                conditions=request["conditions"],
                scope=request["scope"],
                provenance=[origin],
                neighbors=[],
                assumptions=[],
                derivation=dict(raw_sdk_quote=raw),
            )
        else:
            raise ValueError("AIC_PROVIDER_UNSUPPORTED: " + provider)
        # Preserve the SDK contract. Consumer shape aliases are validated by
        # quote_record, never injected into a provider's query/scope.
        quote = validate_quote(quote, **self.policy)
        if quote["op"] != request["op"] or quote["system"] != request["system"]:
            raise ValueError("AIC_PROVIDER_IDENTITY_MISMATCH")
        from .aic_costs import evidence_identities

        self.inputs.update(path for path, _ in evidence_identities(quote))
        return quote
