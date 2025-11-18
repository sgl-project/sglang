use super::*;
use crate::core::ConnectionMode;

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    pub fn validate(config: &RouterConfig) -> ConfigResult<()> {
        Self::validate_mode(&config.mode)?;
        Self::validate_policy(&config.policy)?;
        Self::validate_server_settings(config)?;

        if let Some(discovery) = &config.discovery {
            Self::validate_discovery(discovery, &config.mode)?;
        }

        if let Some(metrics) = &config.metrics {
            Self::validate_metrics(metrics)?;
        }

        Self::validate_compatibility(config)?;

        let retry_cfg = config.effective_retry_config();
        let cb_cfg = config.effective_circuit_breaker_config();
        Self::validate_retry(&retry_cfg)?;
        Self::validate_circuit_breaker(&cb_cfg)?;

        if config.history_backend == HistoryBackend::Oracle {
            if config.oracle.is_none() {
                return Err(ConfigError::MissingRequired {
                    field: "oracle".to_string(),
                });
            }
            if let Some(oracle) = &config.oracle {
                Self::validate_oracle(oracle)?;
            }
        }

        Self::validate_tokenizer_cache(&config.tokenizer_cache)?;

        Ok(())
    }

    fn validate_oracle(oracle: &OracleConfig) -> ConfigResult<()> {
        if oracle.username.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "oracle.username".to_string(),
            });
        }

        if oracle.password.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "oracle.password".to_string(),
            });
        }

        if oracle.connect_descriptor.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "oracle_dsn or oracle_tns_alias".to_string(),
            });
        }

        if oracle.pool_min < 1 {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_min".to_string(),
                value: oracle.pool_min.to_string(),
                reason: "Must be at least 1".to_string(),
            });
        }

        if oracle.pool_max < oracle.pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_max".to_string(),
                value: oracle.pool_max.to_string(),
                reason: "Must be >= oracle.pool_min".to_string(),
            });
        }

        if oracle.pool_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_timeout_secs".to_string(),
                value: oracle.pool_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        Ok(())
    }

    fn validate_mode(mode: &RoutingMode) -> ConfigResult<()> {
        match mode {
            RoutingMode::Regular { worker_urls } => {
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
                // Allow empty URLs without service discovery to match legacy behavior
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
            } => {
                // Allow empty URLs even without service discovery to support dynamic worker addition
                // URLs will be validated if provided
                if !prefill_urls.is_empty() {
                    let prefill_url_strings: Vec<String> =
                        prefill_urls.iter().map(|(url, _)| url.clone()).collect();
                    Self::validate_urls(&prefill_url_strings)?;
                }
                if !decode_urls.is_empty() {
                    Self::validate_urls(decode_urls)?;
                }

                for (_url, port) in prefill_urls {
                    if let Some(port) = port {
                        if *port == 0 {
                            return Err(ConfigError::InvalidValue {
                                field: "bootstrap_port".to_string(),
                                value: port.to_string(),
                                reason: "Port must be between 1 and 65535".to_string(),
                            });
                        }
                    }
                }

                if let Some(p_policy) = prefill_policy {
                    Self::validate_policy(p_policy)?;
                }
                if let Some(d_policy) = decode_policy {
                    Self::validate_policy(d_policy)?;
                }
            }
            RoutingMode::OpenAI { worker_urls } => {
                // Allow empty URLs to support dynamic worker addition
                // URLs will be validated if provided
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
            }
        }
        Ok(())
    }

    fn validate_policy(policy: &PolicyConfig) -> ConfigResult<()> {
        match policy {
            PolicyConfig::Random | PolicyConfig::RoundRobin => {}
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold: _,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                if !(0.0..=1.0).contains(cache_threshold) {
                    return Err(ConfigError::InvalidValue {
                        field: "cache_threshold".to_string(),
                        value: cache_threshold.to_string(),
                        reason: "Must be between 0.0 and 1.0".to_string(),
                    });
                }

                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                if *eviction_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "eviction_interval_secs".to_string(),
                        value: eviction_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *max_tree_size == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "max_tree_size".to_string(),
                        value: max_tree_size.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                if *load_check_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_check_interval_secs".to_string(),
                        value: load_check_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::Bucket {
                balance_abs_threshold: _,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                if *bucket_adjust_interval_secs < 1 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be >= 1s".to_string(),
                    });
                }
                if *bucket_adjust_interval_secs >= 4294967296 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be < 4294967296s".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_server_settings(config: &RouterConfig) -> ConfigResult<()> {
        if config.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "port".to_string(),
                value: config.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        if config.max_payload_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_payload_size".to_string(),
                value: config.max_payload_size.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.request_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "request_timeout_secs".to_string(),
                value: config.request_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.queue_size > 0 && config.queue_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "queue_timeout_secs".to_string(),
                value: config.queue_timeout_secs.to_string(),
                reason: "Must be > 0 when queue_size > 0".to_string(),
            });
        }

        if let Some(tokens_per_second) = config.rate_limit_tokens_per_second {
            if tokens_per_second <= 0 {
                return Err(ConfigError::InvalidValue {
                    field: "rate_limit_tokens_per_second".to_string(),
                    value: tokens_per_second.to_string(),
                    reason: "Must be > 0 when specified".to_string(),
                });
            }
        }

        if config.worker_startup_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "worker_startup_timeout_secs".to_string(),
                value: config.worker_startup_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.worker_startup_check_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "worker_startup_check_interval_secs".to_string(),
                value: config.worker_startup_check_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        Ok(())
    }

    fn validate_discovery(discovery: &DiscoveryConfig, mode: &RoutingMode) -> ConfigResult<()> {
        if !discovery.enabled {
            return Ok(());
        }

        if discovery.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "discovery.port".to_string(),
                value: discovery.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        if discovery.check_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "discovery.check_interval_secs".to_string(),
                value: discovery.check_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        match mode {
            RoutingMode::Regular { .. } => {
                if discovery.selector.is_empty() {
                    return Err(ConfigError::ValidationFailed {
                        reason: "Regular mode with service discovery requires a non-empty selector"
                            .to_string(),
                    });
                }
            }
            RoutingMode::PrefillDecode { .. } => {
                if discovery.prefill_selector.is_empty() && discovery.decode_selector.is_empty() {
                    return Err(ConfigError::ValidationFailed {
                        reason: "PD mode with service discovery requires at least one non-empty selector (prefill or decode)".to_string(),
                    });
                }
            }
            RoutingMode::OpenAI { .. } => {
                return Err(ConfigError::ValidationFailed {
                    reason: "OpenAI mode does not support service discovery".to_string(),
                });
            }
        }

        Ok(())
    }

    fn validate_metrics(metrics: &MetricsConfig) -> ConfigResult<()> {
        if metrics.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "metrics.port".to_string(),
                value: metrics.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        if metrics.host.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "metrics.host".to_string(),
                value: metrics.host.clone(),
                reason: "Host cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    fn validate_retry(retry: &RetryConfig) -> ConfigResult<()> {
        if retry.max_retries < 1 {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_retries".to_string(),
                value: retry.max_retries.to_string(),
                reason: "Must be >= 1 (set to 1 to effectively disable retries)".to_string(),
            });
        }
        if retry.initial_backoff_ms == 0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.initial_backoff_ms".to_string(),
                value: retry.initial_backoff_ms.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if retry.max_backoff_ms < retry.initial_backoff_ms {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_backoff_ms".to_string(),
                value: retry.max_backoff_ms.to_string(),
                reason: "Must be >= initial_backoff_ms".to_string(),
            });
        }
        if retry.backoff_multiplier < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.backoff_multiplier".to_string(),
                value: retry.backoff_multiplier.to_string(),
                reason: "Must be >= 1.0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&retry.jitter_factor) {
            return Err(ConfigError::InvalidValue {
                field: "retry.jitter_factor".to_string(),
                value: retry.jitter_factor.to_string(),
                reason: "Must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_circuit_breaker(cb: &CircuitBreakerConfig) -> ConfigResult<()> {
        if cb.failure_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.failure_threshold".to_string(),
                value: cb.failure_threshold.to_string(),
                reason: "Must be >= 1 (set to u32::MAX to effectively disable CB)".to_string(),
            });
        }
        if cb.success_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.success_threshold".to_string(),
                value: cb.success_threshold.to_string(),
                reason: "Must be >= 1".to_string(),
            });
        }
        if cb.timeout_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.timeout_duration_secs".to_string(),
                value: cb.timeout_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if cb.window_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.window_duration_secs".to_string(),
                value: cb.window_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_tokenizer_cache(cache: &TokenizerCacheConfig) -> ConfigResult<()> {
        if cache.enable_l0 && cache.l0_max_entries == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l0_max_entries".to_string(),
                value: cache.l0_max_entries.to_string(),
                reason: "Must be > 0 when L0 cache is enabled".to_string(),
            });
        }

        if cache.enable_l1 && cache.l1_max_memory == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l1_max_memory".to_string(),
                value: cache.l1_max_memory.to_string(),
                reason: "Must be > 0 when L1 cache is enabled".to_string(),
            });
        }

        Ok(())
    }

    fn validate_mtls(config: &RouterConfig) -> ConfigResult<()> {
        if let Some(identity) = &config.client_identity {
            if identity.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "Client identity cannot be empty".to_string(),
                });
            }
        }

        for (idx, ca_cert) in config.ca_certificates.iter().enumerate() {
            if ca_cert.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("CA certificate at index {} cannot be empty", idx),
                });
            }
        }

        Ok(())
    }

    fn validate_compatibility(config: &RouterConfig) -> ConfigResult<()> {
        if config.enable_igw {
            return Ok(());
        }

        if matches!(config.connection_mode, ConnectionMode::Grpc { .. })
            && config.tokenizer_path.is_none()
            && config.model_path.is_none()
        {
            return Err(ConfigError::ValidationFailed {
                reason: "gRPC connection mode requires either --tokenizer-path or --model-path to be specified".to_string(),
            });
        }

        Self::validate_mtls(config)?;

        let has_service_discovery = config.discovery.as_ref().is_some_and(|d| d.enabled);

        if !has_service_discovery {
            if let PolicyConfig::PowerOfTwo { .. } = &config.policy {
                let worker_count = config.mode.worker_count();
                if worker_count < 2 {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: "Power-of-two policy requires at least 2 workers".to_string(),
                    });
                }
            }

            if let RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
            } = &config.mode
            {
                if let Some(PolicyConfig::PowerOfTwo { .. }) = prefill_policy {
                    if prefill_urls.len() < 2 {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: "Power-of-two policy for prefill requires at least 2 prefill workers".to_string(),
                        });
                    }
                }

                if let Some(PolicyConfig::PowerOfTwo { .. }) = decode_policy {
                    if decode_urls.len() < 2 {
                        return Err(ConfigError::IncompatibleConfig {
                            reason:
                                "Power-of-two policy for decode requires at least 2 decode workers"
                                    .to_string(),
                        });
                    }
                }

                // Check bucket for decode
                if let Some(PolicyConfig::Bucket { .. }) = decode_policy {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: "Decode policy should not be allowed to be bucket".to_string(),
                    });
                }
            }
        }

        if has_service_discovery && config.dp_aware {
            return Err(ConfigError::IncompatibleConfig {
                reason: "DP-aware routing is not compatible with service discovery".to_string(),
            });
        }

        Ok(())
    }

    fn validate_urls(urls: &[String]) -> ConfigResult<()> {
        for url in urls {
            if url.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason: "URL cannot be empty".to_string(),
                });
            }

            if !url.starts_with("http://")
                && !url.starts_with("https://")
                && !url.starts_with("grpc://")
            {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.clone(),
                    reason: "URL must start with http://, https://, or grpc://".to_string(),
                });
            }

            match ::url::Url::parse(url) {
                Ok(parsed) => {
                    if parsed.host_str().is_none() {
                        return Err(ConfigError::InvalidValue {
                            field: "worker_url".to_string(),
                            value: url.clone(),
                            reason: "URL must have a valid host".to_string(),
                        });
                    }
                }
                Err(e) => {
                    return Err(ConfigError::InvalidValue {
                        field: "worker_url".to_string(),
                        value: url.clone(),
                        reason: format!("Invalid URL format: {}", e),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_regular_mode() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_empty_worker_urls() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Empty worker URLs are now allowed to match legacy behavior
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_empty_worker_urls_with_service_discovery() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Enable service discovery
        config.discovery = Some(DiscoveryConfig {
            enabled: true,
            selector: vec![("app".to_string(), "test".to_string())]
                .into_iter()
                .collect(),
            ..Default::default()
        });

        // Should pass validation since service discovery is enabled
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_invalid_urls() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["invalid-url".to_string()],
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_cache_aware_thresholds() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                ],
            },
            PolicyConfig::CacheAware {
                cache_threshold: 1.5, // Invalid: > 1.0
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            },
        );

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_cache_aware_single_worker() {
        // Cache-aware with single worker should be allowed (even if not optimal)
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            },
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_pd_mode() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), Some(8081))],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_roundrobin_with_pd_mode() {
        // RoundRobin with PD mode is now supported
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::RoundRobin,
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_cache_aware_with_pd_mode() {
        // CacheAware with PD mode is now supported
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            },
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_power_of_two_with_regular_mode() {
        // PowerOfTwo with Regular mode is now supported
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                ],
            },
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            },
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pd_mode_with_separate_policies() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), None),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000,
                }),
                decode_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                }),
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pd_mode_power_of_two_insufficient_workers() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill1:8000".to_string(), None)], // Only 1 prefill
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                }), // Requires 2+ workers
                decode_policy: None,
            },
            PolicyConfig::Random,
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("prefill requires at least 2"));
        }
    }

    #[test]
    fn test_validate_pd_mode_bucket_policy_restrictions() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), None),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::Bucket {
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    bucket_adjust_interval_secs: 5,
                }),
                decode_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                }),
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(
            result.is_ok(),
            "Prefill policy should be allowed to be bucket"
        );

        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), None),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::Bucket {
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    bucket_adjust_interval_secs: 5,
                }),
                decode_policy: Some(PolicyConfig::Bucket {
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    bucket_adjust_interval_secs: 5,
                }),
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(
            result.is_err(),
            "Decode policy should not be allowed to be bucket"
        );
    }

    #[test]
    fn test_validate_empty_urls_allowed_without_service_discovery() {
        // Test that empty URLs are now allowed in PD mode
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        );

        // Should pass validation even with empty URLs
        assert!(ConfigValidator::validate(&config).is_ok());

        // Test that empty URLs are allowed in Regular mode
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Should pass validation even with empty URLs
        assert!(ConfigValidator::validate(&config).is_ok());

        // Test that empty URLs are allowed in OpenAI mode
        let config = RouterConfig::new(
            RoutingMode::OpenAI {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Should pass validation even with empty URLs
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_grpc_requires_tokenizer() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["grpc://worker:50051".to_string()],
            },
            PolicyConfig::Random,
        );

        // Set connection mode to gRPC without tokenizer config
        config.connection_mode = ConnectionMode::Grpc { port: None };
        config.tokenizer_path = None;
        config.model_path = None;

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("gRPC connection mode requires"));
        }
    }

    #[test]
    fn test_validate_grpc_with_model_path() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["grpc://worker:50051".to_string()],
            },
            PolicyConfig::Random,
        );

        config.connection_mode = ConnectionMode::Grpc { port: None };
        config.model_path = Some("meta-llama/Llama-3-8B".to_string());

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_grpc_with_tokenizer_path() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["grpc://worker:50051".to_string()],
            },
            PolicyConfig::Random,
        );

        config.connection_mode = ConnectionMode::Grpc { port: None };
        config.tokenizer_path = Some("/path/to/tokenizer.json".to_string());

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }
}
