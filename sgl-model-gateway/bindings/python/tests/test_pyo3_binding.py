"""
Tests that exercise the PyO3 boundary directly (no mocking of `_Router`).

These guard against drift between the Python `RouterArgs` dataclass, the
`Router.from_args` mapping, and the Rust `Router::new` signature in lib.rs.
The tests construct `_Router` without dispatching to remote workers, so they
run quickly and don't require GPU/network.
"""

import pytest
from sglang_router.router import (
    Router,
    backend_from_str,
    build_control_plane_auth_config,
    history_backend_from_str,
    policy_from_str,
    role_from_str,
)
from sglang_router.router_args import RouterArgs
from sglang_router.sglang_router_rs import (
    BackendType,
    HistoryBackendType,
    PolicyType,
    PyApiKeyEntry,
    PyControlPlaneAuthConfig,
    PyJwtConfig,
    PyOracleConfig,
    PyPostgresConfig,
    PyRedisConfig,
    PyRole,
)
from sglang_router.sglang_router_rs import Router as _Router


class TestEnumConversions:
    """All Python ↔ Rust enum conversion helpers cover every variant."""

    def test_policy_from_str_covers_all_variants(self):
        # Mirrors the PolicyType enum in lib.rs. Adding a variant on the Rust
        # side without updating policy_from_str / _POLICY_CHOICES will fail here.
        cases = {
            "random": PolicyType.Random,
            "round_robin": PolicyType.RoundRobin,
            "cache_aware": PolicyType.CacheAware,
            "power_of_two": PolicyType.PowerOfTwo,
            "bucket": PolicyType.Bucket,
            "manual": PolicyType.Manual,
            "consistent_hashing": PolicyType.ConsistentHashing,
            "prefix_hash": PolicyType.PrefixHash,
        }
        for s, expected in cases.items():
            assert policy_from_str(s) == expected

    def test_policy_from_str_none(self):
        assert policy_from_str(None) is None

    def test_backend_from_str(self):
        assert backend_from_str("sglang") == BackendType.Sglang
        assert backend_from_str("openai") == BackendType.Openai
        assert backend_from_str("SGLANG") == BackendType.Sglang
        assert backend_from_str(None) == BackendType.Sglang
        assert backend_from_str(BackendType.Openai) == BackendType.Openai
        with pytest.raises(ValueError, match="Unknown backend"):
            backend_from_str("vllm")

    def test_history_backend_from_str(self):
        assert history_backend_from_str("memory") == HistoryBackendType.Memory
        assert history_backend_from_str("none") == getattr(HistoryBackendType, "None")
        assert history_backend_from_str("oracle") == HistoryBackendType.Oracle
        assert history_backend_from_str("postgres") == HistoryBackendType.Postgres
        assert history_backend_from_str("redis") == HistoryBackendType.Redis
        assert history_backend_from_str(None) == HistoryBackendType.Memory
        assert (
            history_backend_from_str(HistoryBackendType.Redis)
            == HistoryBackendType.Redis
        )
        with pytest.raises(ValueError, match="Unknown history backend"):
            history_backend_from_str("dynamodb")

    def test_role_from_str(self):
        assert role_from_str("admin") == PyRole.Admin
        assert role_from_str("ADMIN") == PyRole.Admin
        assert role_from_str("user") == PyRole.User
        # Unknown roles fall through to User
        assert role_from_str("unknown") == PyRole.User


class TestPyOracleConfig:
    """PyOracleConfig PyO3 validation."""

    def test_defaults(self):
        cfg = PyOracleConfig()
        assert cfg.pool_min == 1
        assert cfg.pool_max == 16
        assert cfg.pool_timeout_secs == 30
        assert cfg.username is None
        assert cfg.password is None
        assert cfg.connect_descriptor is None
        assert cfg.wallet_path is None

    def test_invalid_pool_min_zero(self):
        with pytest.raises(ValueError, match="pool_min must be at least 1"):
            PyOracleConfig(pool_min=0)

    def test_invalid_pool_max_below_min(self):
        with pytest.raises(ValueError, match="pool_max must be >= pool_min"):
            PyOracleConfig(pool_min=5, pool_max=2)

    def test_full_config(self):
        cfg = PyOracleConfig(
            password="secret",
            username="orcl",
            connect_descriptor="dsn",
            wallet_path="/path/to/wallet",
            pool_min=2,
            pool_max=20,
            pool_timeout_secs=45,
        )
        assert cfg.username == "orcl"
        assert cfg.pool_min == 2
        assert cfg.pool_max == 20
        assert cfg.pool_timeout_secs == 45


class TestPyPostgresConfig:
    def test_defaults(self):
        cfg = PyPostgresConfig()
        assert cfg.db_url is None
        assert cfg.pool_max == 16

    def test_with_values(self):
        cfg = PyPostgresConfig(db_url="postgres://localhost/db", pool_max=32)
        assert cfg.db_url == "postgres://localhost/db"
        assert cfg.pool_max == 32


class TestPyRedisConfig:
    def test_defaults(self):
        cfg = PyRedisConfig(url="redis://localhost:6379")
        assert cfg.url == "redis://localhost:6379"
        assert cfg.pool_max == 16
        assert cfg.retention_days == 30

    def test_persistent_retention(self):
        cfg = PyRedisConfig(url="redis://localhost", retention_days=None)
        assert cfg.retention_days is None


class TestPyApiKeyEntry:
    def test_default_role_is_user(self):
        entry = PyApiKeyEntry(id="k1", name="svc", key="secret")
        assert entry.id == "k1"
        assert entry.name == "svc"
        assert entry.key == "secret"
        assert entry.role == PyRole.User

    def test_admin_role(self):
        entry = PyApiKeyEntry(id="k1", name="svc", key="secret", role=PyRole.Admin)
        assert entry.role == PyRole.Admin


class TestPyJwtConfig:
    def test_defaults_have_role_claim(self):
        # role_claim defaults to "roles" matching the smg-auth crate; without
        # this surfaced through PyO3, OIDC role mapping silently breaks.
        cfg = PyJwtConfig(issuer="https://issuer", audience="api")
        assert cfg.issuer == "https://issuer"
        assert cfg.audience == "api"
        assert cfg.role_claim == "roles"
        assert cfg.role_mapping == {}
        assert cfg.jwks_uri is None

    def test_custom_role_claim(self):
        cfg = PyJwtConfig(
            issuer="https://issuer",
            audience="api",
            role_claim="groups",
            role_mapping={"AdminGroup": "admin"},
        )
        assert cfg.role_claim == "groups"
        assert cfg.role_mapping == {"AdminGroup": "admin"}


class TestPyControlPlaneAuthConfig:
    def test_default_audit_enabled(self):
        # PyO3 default mirrors the smg-auth crate: ControlPlaneAuthConfig
        # constructed without arguments has audit_enabled = true.
        cfg = PyControlPlaneAuthConfig()
        assert cfg.audit_enabled is True
        assert cfg.api_keys == []
        assert cfg.jwt is None

    def test_with_jwt_and_keys(self):
        # `PyJwtConfig` doesn't implement Python __eq__, so compare by field.
        jwt = PyJwtConfig(issuer="i", audience="a")
        keys = [PyApiKeyEntry(id="k", name="n", key="s", role=PyRole.Admin)]
        cfg = PyControlPlaneAuthConfig(jwt=jwt, api_keys=keys, audit_enabled=False)
        assert cfg.audit_enabled is False
        assert cfg.jwt is not None
        assert cfg.jwt.issuer == "i"
        assert cfg.jwt.audience == "a"
        assert len(cfg.api_keys) == 1
        assert cfg.api_keys[0].id == "k"


class TestBuildControlPlaneAuthConfig:
    def test_returns_none_when_no_auth(self):
        assert build_control_plane_auth_config({}) is None

    def test_returns_none_when_only_audit_set(self):
        # Audit-only without keys/JWT shouldn't materialize a config object.
        assert (
            build_control_plane_auth_config({"control_plane_audit_enabled": True})
            is None
        )

    def test_audit_default_when_unspecified(self):
        # The Python wrapper has historically defaulted audit_enabled to False
        # when the user doesn't pass control_plane_audit_enabled. Lock that in
        # so a future change can't silently flip it.
        cfg = build_control_plane_auth_config(
            {
                "control_plane_api_keys": [("id1", "Svc", "secret", "admin")],
            }
        )
        assert cfg is not None
        assert cfg.audit_enabled is False
        assert len(cfg.api_keys) == 1
        assert cfg.api_keys[0].role == PyRole.Admin

    def test_jwt_role_claim_threaded_through(self):
        # jwt_role_claim must reach PyJwtConfig — without this the helper
        # silently drops the user's claim name.
        cfg = build_control_plane_auth_config(
            {
                "jwt_issuer": "https://issuer",
                "jwt_audience": "api",
                "jwt_role_claim": "groups",
                "jwt_role_mapping": {"Admins": "admin"},
            }
        )
        assert cfg is not None
        assert cfg.jwt is not None
        assert cfg.jwt.role_claim == "groups"
        assert cfg.jwt.role_mapping == {"Admins": "admin"}

    def test_jwt_default_role_claim(self):
        cfg = build_control_plane_auth_config(
            {"jwt_issuer": "https://issuer", "jwt_audience": "api"}
        )
        assert cfg is not None and cfg.jwt is not None
        assert cfg.jwt.role_claim == "roles"

    def test_warns_when_jwt_incomplete(self, caplog):
        # If the user sets jwt_role_claim/jwks_uri/role_mapping but forgets
        # issuer/audience, the helper drops them silently. Emit a warning so
        # users notice their JWT auth isn't actually enabled.
        with caplog.at_level("WARNING", logger="sglang_router.router"):
            cfg = build_control_plane_auth_config(
                {"jwt_role_claim": "groups", "jwt_role_mapping": {"X": "admin"}}
            )
        assert cfg is None
        assert any(
            "jwt_issuer/jwt_audience missing" in record.message
            for record in caplog.records
        )


class TestParseControlPlaneApiKeys:
    def test_valid(self):
        result = RouterArgs._parse_control_plane_api_keys(
            ["k1:Service Account:admin:secret123", "k2:Read Only:user:secret456"]
        )
        assert result == [
            ("k1", "Service Account", "secret123", "admin"),
            ("k2", "Read Only", "secret456", "user"),
        ]

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid API key format"):
            RouterArgs._parse_control_plane_api_keys(["just-a-key"])

    def test_invalid_role(self):
        with pytest.raises(ValueError, match="Invalid role"):
            RouterArgs._parse_control_plane_api_keys(["id:name:superuser:secret"])

    def test_key_with_colons_preserved(self):
        # The split limit of 4 means the key portion can itself contain colons.
        result = RouterArgs._parse_control_plane_api_keys(
            ["id:name:user:sk-abc:def:ghi"]
        )
        assert result == [("id", "name", "sk-abc:def:ghi", "user")]

    def test_empty(self):
        assert RouterArgs._parse_control_plane_api_keys([]) == []
        assert RouterArgs._parse_control_plane_api_keys(None) == []


class TestParseJwtRoleMapping:
    def test_valid(self):
        result = RouterArgs._parse_jwt_role_mapping(
            ["Gateway.Admin=admin", "Gateway.User=user"]
        )
        assert result == {"Gateway.Admin": "admin", "Gateway.User": "user"}

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid role mapping format"):
            RouterArgs._parse_jwt_role_mapping(["no-equals"])

    def test_invalid_role(self):
        with pytest.raises(ValueError, match="Invalid gateway role"):
            RouterArgs._parse_jwt_role_mapping(["X=superuser"])

    def test_empty(self):
        assert RouterArgs._parse_jwt_role_mapping([]) == {}


class TestRouterFromArgsKitchenSink:
    """End-to-end tests of `Router.from_args(RouterArgs(...))`.

    These instantiate a real PyO3 `_Router` (no mocking) so any drift between
    the Python dataclass fields and the Rust constructor signature surfaces here.
    """

    def test_minimal_regular_mode(self):
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://w1:8000"],
            policy="round_robin",
        )
        router = Router.from_args(args)
        assert isinstance(router._router, _Router)

    def test_pd_mode(self):
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy="power_of_two",
            decode_policy="round_robin",
        )
        router = Router.from_args(args)
        assert isinstance(router._router, _Router)

    def test_all_policies_construct(self):
        # Ensures every PolicyType the binding accepts is reachable through
        # RouterArgs without exploding (e.g. unknown assignment_mode panics).
        for policy in (
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "bucket",
            "manual",
            "consistent_hashing",
            "prefix_hash",
        ):
            args = RouterArgs(
                worker_urls=["http://w1:8000"],
                policy=policy,
                pd_disaggregation=True,
                prefill_urls=[("http://prefill1:8000", None)],
                decode_urls=["http://decode1:8001"],
                prefill_policy=policy,
                decode_policy=policy,
            )
            Router.from_args(args)

    def test_kitchen_sink_passes_every_field(self):
        # Touches every dataclass field that maps directly to a parameter of
        # Rust's Router::new (see lib.rs). PD-mode fields are exercised in
        # test_pd_mode/test_all_policies_construct, and history-backend
        # sub-configs are exercised in test_{oracle,postgres,redis}_history_backend.
        # If you add a field on the Rust side without wiring it through
        # RouterArgs/from_args, _Router(**args_dict) raises
        # TypeError("got an unexpected keyword argument ...").
        args = RouterArgs(
            worker_urls=["http://w1:8000", "http://w2:8000"],
            host="127.0.0.1",
            port=30001,
            policy="cache_aware",
            worker_startup_timeout_secs=60,
            worker_startup_check_interval=5,
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.2,
            eviction_interval_secs=30,
            max_tree_size=2**20,
            max_idle_secs=600,
            assignment_mode="min_load",
            max_payload_size=1024 * 1024,
            bucket_adjust_interval_secs=10,
            dp_aware=True,
            enable_igw=False,
            api_key="key123",
            log_dir="/tmp/router-logs",
            log_level="debug",
            json_log=True,
            service_discovery=False,
            selector={"app": "worker"},
            service_discovery_port=8080,
            service_discovery_namespace="default",
            prefill_selector={"role": "prefill"},
            decode_selector={"role": "decode"},
            bootstrap_port_annotation="custom.io/bootstrap-port",
            prometheus_port=29000,
            prometheus_host="127.0.0.1",
            prometheus_duration_buckets=[0.1, 0.5, 1.0],
            request_id_headers=["x-trace-id"],
            request_timeout_secs=600,
            shutdown_grace_period_secs=30,
            max_concurrent_requests=128,
            queue_size=50,
            queue_timeout_secs=30,
            rate_limit_tokens_per_second=64,
            cors_allowed_origins=["http://localhost:3000"],
            retry_max_retries=2,
            retry_initial_backoff_ms=10,
            retry_max_backoff_ms=1000,
            retry_backoff_multiplier=2.0,
            retry_jitter_factor=0.3,
            disable_retries=False,
            cb_failure_threshold=5,
            cb_success_threshold=2,
            cb_timeout_duration_secs=30,
            cb_window_duration_secs=60,
            disable_circuit_breaker=False,
            health_failure_threshold=2,
            health_success_threshold=1,
            health_check_timeout_secs=3,
            health_check_interval_secs=15,
            health_check_endpoint="/healthz",
            disable_health_check=False,
            model_path="meta-llama/Llama-3-8B",
            tokenizer_path=None,
            chat_template=None,
            tokenizer_cache_enable_l0=True,
            tokenizer_cache_l0_max_entries=1000,
            tokenizer_cache_enable_l1=True,
            tokenizer_cache_l1_max_memory=1024 * 1024,
            reasoning_parser="qwen3",
            tool_call_parser=None,
            mcp_config_path=None,
            backend="sglang",
            history_backend="memory",
            client_cert_path=None,
            client_key_path=None,
            ca_cert_paths=[],
            server_cert_path=None,
            server_key_path=None,
            enable_trace=True,
            otlp_traces_endpoint="otel-collector:4317",
            control_plane_api_keys=[("k1", "svc", "secret", "admin")],
            control_plane_audit_enabled=False,
            jwt_issuer="https://issuer",
            jwt_audience="api",
            jwt_jwks_uri="https://issuer/.well-known/jwks.json",
            jwt_role_claim="groups",
            jwt_role_mapping={"Admins": "admin"},
            pool_idle_timeout_secs=20,
            connect_timeout_secs=5,
            pool_max_idle_per_host=100,
            tcp_keepalive_secs=15,
            enable_wasm=True,
        )
        router = Router.from_args(args)
        assert isinstance(router._router, _Router)
        # Confirm the new fields actually carry the right value into Rust —
        # an isinstance check alone wouldn't catch a typo'd builder call like
        # `.pool_idle_timeout_secs(self.connect_timeout_secs)`.
        assert router._router.pool_idle_timeout_secs == 20
        assert router._router.connect_timeout_secs == 5
        assert router._router.pool_max_idle_per_host == 100
        assert router._router.tcp_keepalive_secs == 15
        assert router._router.enable_wasm is True

    def test_oracle_history_backend(self):
        args = RouterArgs(
            worker_urls=["http://w1:8000"],
            history_backend="oracle",
            oracle_username="user",
            oracle_password="pw",
            oracle_connect_descriptor="dsn",
            oracle_pool_min=2,
            oracle_pool_max=8,
        )
        router = Router.from_args(args)
        assert isinstance(router._router, _Router)

    def test_postgres_history_backend(self):
        args = RouterArgs(
            worker_urls=["http://w1:8000"],
            history_backend="postgres",
            postgres_db_url="postgres://localhost/db",
            postgres_pool_max=8,
        )
        router = Router.from_args(args)
        assert isinstance(router._router, _Router)

    def test_redis_history_backend(self):
        args = RouterArgs(
            worker_urls=["http://w1:8000"],
            history_backend="redis",
            redis_url="redis://localhost:6379",
            redis_pool_max=8,
            redis_retention_days=7,
        )
        router = Router.from_args(args)
        assert isinstance(router._router, _Router)

    def test_redis_persistent_retention(self):
        # redis_retention_days < 0 means persistent (None on the Rust side).
        args = RouterArgs(
            worker_urls=["http://w1:8000"],
            history_backend="redis",
            redis_url="redis://localhost:6379",
            redis_retention_days=-1,
        )
        Router.from_args(args)


class TestBootstrapPortAnnotation:
    """Regression: the wrapper must not silently override user-supplied values."""

    def test_user_value_preserved_through_from_cli_args(self):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(
            [
                "--bootstrap-port-annotation",
                "custom.io/bootstrap-port",
            ]
        )
        assert args.bootstrap_port_annotation == "custom.io/bootstrap-port"

    def test_default_value(self):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args([])
        assert args.bootstrap_port_annotation == "sglang.ai/bootstrap-port"


class TestNewBindingFields:
    """Round-trip checks for fields whose CLI flag, dataclass attribute, and
    PyO3 constructor parameter were historically out of sync."""

    def test_jwt_role_claim_default(self):
        args = RouterArgs()
        assert args.jwt_role_claim == "roles"

    def test_audit_enabled_default_off(self):
        # The Python wrapper defaults audit_enabled to False even though the
        # Rust standalone binary defaults it to True (main.rs:614,
        # disable_audit_logging = false → audit_enabled = true). The
        # divergence is intentional: changing the wrapper default is a
        # behavior change that needs an explicit migration.
        args = RouterArgs()
        assert args.control_plane_audit_enabled is False

    def test_audit_enabled_via_cli_flag(self):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(["--control-plane-audit-enabled"])
        assert args.control_plane_audit_enabled is True

    def test_default_audit_via_cli_is_off(self):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args([])
        assert args.control_plane_audit_enabled is False

    def test_http_pool_defaults(self):
        args = RouterArgs()
        assert args.pool_idle_timeout_secs == 50
        assert args.connect_timeout_secs == 10
        assert args.pool_max_idle_per_host == 500
        assert args.tcp_keepalive_secs == 30

    def test_enable_wasm_default(self):
        args = RouterArgs()
        assert args.enable_wasm is False

    def test_http_pool_via_cli(self):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(
            [
                "--pool-idle-timeout-secs",
                "120",
                "--connect-timeout-secs",
                "20",
                "--pool-max-idle-per-host",
                "200",
                "--tcp-keepalive-secs",
                "45",
                "--enable-wasm",
            ]
        )
        assert args.pool_idle_timeout_secs == 120
        assert args.connect_timeout_secs == 20
        assert args.pool_max_idle_per_host == 200
        assert args.tcp_keepalive_secs == 45
        assert args.enable_wasm is True

    def test_jwt_role_claim_via_cli(self):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(
            [
                "--jwt-issuer",
                "https://issuer",
                "--jwt-audience",
                "api",
                "--jwt-role-claim",
                "groups",
            ]
        )
        assert args.jwt_role_claim == "groups"

    def test_jwt_role_claim_end_to_end(self):
        # Full pipeline: CLI parser → RouterArgs → Router.from_args →
        # `_Router(**args_dict)`. Pins the invariant that `jwt_role_claim` is
        # consumed by `build_control_plane_auth_config` AND popped from
        # args_dict before reaching the Rust constructor (otherwise _Router
        # would raise TypeError on the unknown kwarg).
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(
            [
                "--jwt-issuer",
                "https://issuer",
                "--jwt-audience",
                "api",
                "--jwt-role-claim",
                "groups",
                "--jwt-role-mapping",
                "Admins=admin",
            ]
        )
        router = Router.from_args(args)
        assert isinstance(router._router, _Router)


class TestPolicyChoiceListConsistency:
    """Every policy in the binding's PolicyType must be a CLI choice on every
    policy flag. Catches drift if someone hard-codes a list at one of the three
    argparse `choices=` sites instead of using `_POLICY_CHOICES`."""

    @pytest.mark.parametrize(
        "policy",
        [
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "bucket",
            "manual",
            "consistent_hashing",
            "prefix_hash",
        ],
    )
    def test_main_policy_accepts(self, policy):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(["--policy", policy])
        assert args.policy == policy

    @pytest.mark.parametrize(
        "policy",
        [
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "bucket",
            "manual",
            "consistent_hashing",
            "prefix_hash",
        ],
    )
    def test_prefill_policy_accepts(self, policy):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(
            [
                "--pd-disaggregation",
                "--prefill",
                "http://p:8000",
                "--decode",
                "http://d:8001",
                "--prefill-policy",
                policy,
            ]
        )
        assert args.prefill_policy == policy

    @pytest.mark.parametrize(
        "policy",
        [
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "bucket",
            "manual",
            "consistent_hashing",
            "prefix_hash",
        ],
    )
    def test_decode_policy_accepts(self, policy):
        from sglang_router.launch_router import parse_router_args

        args = parse_router_args(
            [
                "--pd-disaggregation",
                "--prefill",
                "http://p:8000",
                "--decode",
                "http://d:8001",
                "--decode-policy",
                policy,
            ]
        )
        assert args.decode_policy == policy
