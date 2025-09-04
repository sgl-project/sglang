"""
Unit tests for startup sequence logic in sglang_router.

These tests focus on testing the startup sequence logic in isolation,
including router initialization, configuration validation, and startup flow.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace
import logging

from sglang_router.launch_router import RouterArgs, launch_router, setup_logger, policy_from_str
from sglang_router_rs import PolicyType


class TestSetupLogger:
    """Test logger setup functionality."""

    def test_setup_logger_returns_logger(self):
        """Test that setup_logger returns a logger instance."""
        logger = setup_logger()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "router"
        assert logger.level == logging.INFO

    def test_setup_logger_has_handler(self):
        """Test that setup_logger configures a handler."""
        logger = setup_logger()
        
        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)

    def test_setup_logger_has_formatter(self):
        """Test that setup_logger configures a formatter."""
        logger = setup_logger()
        
        handler = logger.handlers[0]
        formatter = handler.formatter
        
        assert formatter is not None
        assert "[Router (Python)]" in formatter._fmt

    def test_setup_logger_multiple_calls(self):
        """Test that multiple calls to setup_logger work correctly."""
        logger1 = setup_logger()
        logger2 = setup_logger()
        
        # Should return the same logger instance
        assert logger1 is logger2


class TestPolicyFromStr:
    """Test policy string to enum conversion in startup context."""

    def test_policy_conversion_in_startup(self):
        """Test policy conversion during startup sequence."""
        # Test all valid policies
        policies = ["random", "round_robin", "cache_aware", "power_of_two"]
        expected_enums = [
            PolicyType.Random,
            PolicyType.RoundRobin,
            PolicyType.CacheAware,
            PolicyType.PowerOfTwo
        ]
        
        for policy_str, expected_enum in zip(policies, expected_enums):
            result = policy_from_str(policy_str)
            assert result == expected_enum

    def test_invalid_policy_in_startup(self):
        """Test handling of invalid policy during startup."""
        with pytest.raises(KeyError):
            policy_from_str("invalid_policy")


class TestRouterInitialization:
    """Test router initialization logic."""

    def test_router_initialization_basic(self):
        """Test basic router initialization."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"],
            policy="cache_aware"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with correct parameters
            mock_router_class.assert_called_once()
            call_args = mock_router_class.call_args
            
            # Check key parameters
            assert call_args[1]['host'] == "127.0.0.1"
            assert call_args[1]['port'] == 30000
            assert call_args[1]['worker_urls'] == ["http://worker1:8000"]
            assert call_args[1]['policy'] == PolicyType.CacheAware
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_pd_mode(self):
        """Test router initialization in PD mode."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000)],
            decode_urls=["http://decode1:8001"],
            policy="power_of_two"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with PD parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['pd_disaggregation'] is True
            assert call_args[1]['prefill_urls'] == [("http://prefill1:8000", 9000)]
            assert call_args[1]['decode_urls'] == ["http://decode1:8001"]
            assert call_args[1]['policy'] == PolicyType.PowerOfTwo
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_service_discovery(self):
        """Test router initialization with service discovery."""
        args = RouterArgs(
            service_discovery=True,
            selector={"app": "worker", "env": "prod"},
            service_discovery_port=8080,
            service_discovery_namespace="default"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with service discovery parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['service_discovery'] is True
            assert call_args[1]['selector'] == {"app": "worker", "env": "prod"}
            assert call_args[1]['service_discovery_port'] == 8080
            assert call_args[1]['service_discovery_namespace'] == "default"
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_retry_config(self):
        """Test router initialization with retry configuration."""
        args = RouterArgs(
            retry_max_retries=3,
            retry_initial_backoff_ms=100,
            retry_max_backoff_ms=10000,
            retry_backoff_multiplier=2.0,
            retry_jitter_factor=0.1,
            disable_retries=False
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with retry parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['retry_max_retries'] == 3
            assert call_args[1]['retry_initial_backoff_ms'] == 100
            assert call_args[1]['retry_max_backoff_ms'] == 10000
            assert call_args[1]['retry_backoff_multiplier'] == 2.0
            assert call_args[1]['retry_jitter_factor'] == 0.1
            assert call_args[1]['disable_retries'] is False
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_circuit_breaker_config(self):
        """Test router initialization with circuit breaker configuration."""
        args = RouterArgs(
            cb_failure_threshold=5,
            cb_success_threshold=2,
            cb_timeout_duration_secs=30,
            cb_window_duration_secs=60,
            disable_circuit_breaker=False
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with circuit breaker parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['cb_failure_threshold'] == 5
            assert call_args[1]['cb_success_threshold'] == 2
            assert call_args[1]['cb_timeout_duration_secs'] == 30
            assert call_args[1]['cb_window_duration_secs'] == 60
            assert call_args[1]['disable_circuit_breaker'] is False
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_rate_limiting_config(self):
        """Test router initialization with rate limiting configuration."""
        args = RouterArgs(
            max_concurrent_requests=512,
            queue_size=200,
            queue_timeout_secs=120,
            rate_limit_tokens_per_second=100
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with rate limiting parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['max_concurrent_requests'] == 512
            assert call_args[1]['queue_size'] == 200
            assert call_args[1]['queue_timeout_secs'] == 120
            assert call_args[1]['rate_limit_tokens_per_second'] == 100
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_health_check_config(self):
        """Test router initialization with health check configuration."""
        args = RouterArgs(
            health_failure_threshold=2,
            health_success_threshold=1,
            health_check_timeout_secs=3,
            health_check_interval_secs=30,
            health_check_endpoint="/healthz"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with health check parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['health_failure_threshold'] == 2
            assert call_args[1]['health_success_threshold'] == 1
            assert call_args[1]['health_check_timeout_secs'] == 3
            assert call_args[1]['health_check_interval_secs'] == 30
            assert call_args[1]['health_check_endpoint'] == "/healthz"
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_prometheus_config(self):
        """Test router initialization with Prometheus configuration."""
        args = RouterArgs(
            prometheus_port=29000,
            prometheus_host="127.0.0.1"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with Prometheus parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['prometheus_port'] == 29000
            assert call_args[1]['prometheus_host'] == "127.0.0.1"
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_cors_config(self):
        """Test router initialization with CORS configuration."""
        args = RouterArgs(
            cors_allowed_origins=["http://localhost:3000", "https://example.com"]
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify router was created with CORS parameters
            call_args = mock_router_class.call_args
            
            assert call_args[1]['cors_allowed_origins'] == ["http://localhost:3000", "https://example.com"]
            
            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()
            
            # Verify return value
            assert result == mock_router_instance

    def test_router_initialization_with_tokenizer_config(self):
        """Test router initialization with tokenizer configuration."""
        # Note: model_path and tokenizer_path are not available in current RouterArgs
        pytest.skip("Tokenizer configuration not available in current implementation")


class TestStartupValidation:
    """Test startup validation logic."""

    def test_pd_mode_validation_during_startup(self):
        """Test PD mode validation during startup."""
        # PD mode without URLs should fail
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=False
        )
        
        with pytest.raises(ValueError, match="PD disaggregation mode requires --prefill"):
            launch_router(args)

    def test_pd_mode_with_service_discovery_validation(self):
        """Test PD mode with service discovery validation during startup."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=True
        )
        
        # Should not raise validation error
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Should create router instance
            mock_router_class.assert_called_once()
            assert result == mock_router_instance

    def test_policy_warning_during_startup(self):
        """Test policy warning during startup in PD mode."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy="power_of_two",
            decode_policy="round_robin"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            with patch('sglang_router.launch_router.logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.getLogger.return_value = mock_logger
                
                result = launch_router(args)
                
                # Should log warning about policy usage
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "Both --prefill-policy and --decode-policy are specified" in warning_call
                
                # Should create router instance
                mock_router_class.assert_called_once()
                assert result == mock_router_instance

    def test_policy_info_during_startup(self):
        """Test policy info logging during startup in PD mode."""
        # Test with only prefill policy specified
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy="power_of_two",
            decode_policy=None
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            with patch('sglang_router.launch_router.logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.getLogger.return_value = mock_logger
                
                result = launch_router(args)
                
                # Should log info about policy usage
                mock_logger.info.assert_called_once()
                info_call = mock_logger.info.call_args[0][0]
                assert "Using --prefill-policy 'power_of_two'" in info_call
                assert "and --policy 'cache_aware'" in info_call
                
                # Should create router instance
                mock_router_class.assert_called_once()
                assert result == mock_router_instance

    def test_policy_info_decode_only_during_startup(self):
        """Test policy info logging during startup with only decode policy specified."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", None)],
            decode_urls=["http://decode1:8001"],
            policy="cache_aware",
            prefill_policy=None,
            decode_policy="round_robin"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            with patch('sglang_router.launch_router.logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.getLogger.return_value = mock_logger
                
                result = launch_router(args)
                
                # Should log info about policy usage
                mock_logger.info.assert_called_once()
                info_call = mock_logger.info.call_args[0][0]
                assert "Using --policy 'cache_aware'" in info_call
                assert "and --decode-policy 'round_robin'" in info_call
                
                # Should create router instance
                mock_router_class.assert_called_once()
                assert result == mock_router_instance


class TestStartupErrorHandling:
    """Test startup error handling logic."""

    def test_router_creation_error_handling(self):
        """Test error handling when router creation fails."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"]
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            # Simulate router creation failure
            mock_router_class.side_effect = Exception("Router creation failed")
            
            with patch('sglang_router.launch_router.logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.getLogger.return_value = mock_logger
                
                with pytest.raises(Exception, match="Router creation failed"):
                    launch_router(args)
                
                # Should log error
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert "Error starting router: Router creation failed" in error_call

    def test_router_start_error_handling(self):
        """Test error handling when router start fails."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000"]
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            # Simulate router start failure
            mock_router_instance.start.side_effect = Exception("Router start failed")
            
            with patch('sglang_router.launch_router.logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.getLogger.return_value = mock_logger
                
                with pytest.raises(Exception, match="Router start failed"):
                    launch_router(args)
                
                # Should log error
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert "Error starting router: Router start failed" in error_call

    def test_validation_error_handling(self):
        """Test error handling when validation fails."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=False
        )
        
        with patch('sglang_router.launch_router.logging') as mock_logging:
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger
            
            with pytest.raises(ValueError, match="PD disaggregation mode requires --prefill"):
                launch_router(args)
            
            # Should log error for validation failures
            mock_logger.error.assert_called_once()


class TestStartupFlow:
    """Test complete startup flow."""

    def test_complete_startup_flow_basic(self):
        """Test complete startup flow for basic configuration."""
        args = RouterArgs(
            host="127.0.0.1",
            port=30000,
            worker_urls=["http://worker1:8000", "http://worker2:8000"],
            policy="cache_aware",
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.5
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify complete flow
            mock_router_class.assert_called_once()
            mock_router_instance.start.assert_called_once()
            assert result == mock_router_instance

    def test_complete_startup_flow_pd_mode(self):
        """Test complete startup flow for PD mode configuration."""
        args = RouterArgs(
            pd_disaggregation=True,
            prefill_urls=[("http://prefill1:8000", 9000), ("http://prefill2:8000", None)],
            decode_urls=["http://decode1:8001", "http://decode2:8001"],
            policy="power_of_two",
            prefill_policy="cache_aware",
            decode_policy="round_robin"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            with patch('sglang_router.launch_router.logging') as mock_logging:
                mock_logger = MagicMock()
                mock_logging.getLogger.return_value = mock_logger
                
                result = launch_router(args)
                
                # Verify complete flow
                mock_router_class.assert_called_once()
                mock_router_instance.start.assert_called_once()
                assert result == mock_router_instance
                
                # Verify policy warning was logged
                mock_logger.warning.assert_called_once()

    def test_complete_startup_flow_with_all_features(self):
        """Test complete startup flow with all features enabled."""
        args = RouterArgs(
            host="0.0.0.0",
            port=30001,
            worker_urls=["http://worker1:8000"],
            policy="round_robin",
            service_discovery=True,
            selector={"app": "worker"},
            service_discovery_port=8080,
            service_discovery_namespace="default",
            dp_aware=True,
            api_key="test-key",
            log_dir="/tmp/logs",
            log_level="debug",
            prometheus_port=29000,
            prometheus_host="0.0.0.0",
            request_id_headers=["x-request-id", "x-trace-id"],
            request_timeout_secs=1200,
            max_concurrent_requests=512,
            queue_size=200,
            queue_timeout_secs=120,
            rate_limit_tokens_per_second=100,
            cors_allowed_origins=["http://localhost:3000"],
            retry_max_retries=3,
            retry_initial_backoff_ms=100,
            retry_max_backoff_ms=10000,
            retry_backoff_multiplier=2.0,
            retry_jitter_factor=0.1,
            cb_failure_threshold=5,
            cb_success_threshold=2,
            cb_timeout_duration_secs=30,
            cb_window_duration_secs=60,
            health_failure_threshold=2,
            health_success_threshold=1,
            health_check_timeout_secs=3,
            health_check_interval_secs=30,
            health_check_endpoint="/healthz"
        )
        
        with patch('sglang_router.launch_router.Router') as mock_router_class:
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance
            
            result = launch_router(args)
            
            # Verify complete flow
            mock_router_class.assert_called_once()
            mock_router_instance.start.assert_called_once()
            assert result == mock_router_instance
            
            # Verify all parameters were passed correctly
            call_args = mock_router_class.call_args
            assert call_args[1]['host'] == "0.0.0.0"
            assert call_args[1]['port'] == 30001
            # When service_discovery is True, worker_urls should be empty
            assert call_args[1]['worker_urls'] == []
            assert call_args[1]['policy'] == PolicyType.RoundRobin
            assert call_args[1]['service_discovery'] is True
            assert call_args[1]['selector'] == {"app": "worker"}
            assert call_args[1]['service_discovery_port'] == 8080
            assert call_args[1]['service_discovery_namespace'] == "default"
            assert call_args[1]['dp_aware'] is True
            assert call_args[1]['api_key'] == "test-key"
            assert call_args[1]['log_dir'] == "/tmp/logs"
            assert call_args[1]['log_level'] == "debug"
            assert call_args[1]['prometheus_port'] == 29000
            assert call_args[1]['prometheus_host'] == "0.0.0.0"
            assert call_args[1]['request_id_headers'] == ["x-request-id", "x-trace-id"]
            assert call_args[1]['request_timeout_secs'] == 1200
            assert call_args[1]['max_concurrent_requests'] == 512
            assert call_args[1]['queue_size'] == 200
            assert call_args[1]['queue_timeout_secs'] == 120
            assert call_args[1]['rate_limit_tokens_per_second'] == 100
            assert call_args[1]['cors_allowed_origins'] == ["http://localhost:3000"]
            assert call_args[1]['retry_max_retries'] == 3
            assert call_args[1]['retry_initial_backoff_ms'] == 100
            assert call_args[1]['retry_max_backoff_ms'] == 10000
            assert call_args[1]['retry_backoff_multiplier'] == 2.0
            assert call_args[1]['retry_jitter_factor'] == 0.1
            assert call_args[1]['cb_failure_threshold'] == 5
            assert call_args[1]['cb_success_threshold'] == 2
            assert call_args[1]['cb_timeout_duration_secs'] == 30
            assert call_args[1]['cb_window_duration_secs'] == 60
            assert call_args[1]['health_failure_threshold'] == 2
            assert call_args[1]['health_success_threshold'] == 1
            assert call_args[1]['health_check_timeout_secs'] == 3
            assert call_args[1]['health_check_interval_secs'] == 30
            assert call_args[1]['health_check_endpoint'] == "/healthz"
