use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use axum::{Router, routing::get};
use tower::ServiceExt;

use super::*;
use crate::message::finish_reason::FinishKind;
use crate::message::response::{CachedTokensDetails, TokenCounts};

fn sample(metrics: &FrontendMetrics, name: &str) -> prometheus::proto::Metric {
    let families = metrics.registry.gather();
    let family = families
        .iter()
        .find(|family| family.get_name() == name)
        .unwrap_or_else(|| panic!("missing {name}: {families:?}"));
    assert_eq!(family.get_metric().len(), 1, "{name}");
    family.get_metric()[0].clone()
}

#[tokio::test]
async fn family_schema_and_zero_series_match_python_registrations() {
    use prometheus::core::Collector;

    let expected: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(include_str!("../../testdata/metrics_python_schema.json")).unwrap();
    let metrics = FrontendMetrics::new(&ServerArgs {
        metrics_config: Some(
            serde_json::json!({
                "labels": {"model_name":"test", "engine_type":"unified", "cluster":"test"},
                "allowed_custom_labels": ["tenant"], "priority_enabled": true,
                "http_labels": {"cluster":"test"}
            })
            .to_string(),
        ),
        ..Default::default()
    })
    .unwrap();
    let zero: std::collections::BTreeSet<_> = metrics
        .registry
        .gather()
        .iter()
        .map(|family| family.get_name().to_owned())
        .collect();
    for (name, schema) in &expected {
        assert_eq!(
            zero.contains(name),
            schema["zero_series"].as_bool().unwrap(),
            "{name}"
        );
    }
    for counter in metrics.counters.values() {
        counter.with_label_values(&vec![""; counter.desc()[0].variable_labels.len()]);
    }
    for histogram in metrics.histograms.values() {
        histogram.with_label_values(&vec![""; histogram.desc()[0].variable_labels.len()]);
    }
    let app = apply_http(
        Router::new().route("/health", get(|| async { "OK" })),
        metrics.clone(),
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let families = metrics.registry.gather();
    // Startup facts come from the Python launch owner, tested with source merging.
    assert_eq!(families.len(), expected.len() - 2);
    for family in families {
        let schema = &expected[family.get_name()];
        assert_eq!(schema["help"], family.get_help());
        assert_eq!(
            schema["type"],
            format!("{:?}", family.get_field_type()).to_lowercase()
        );
        let metric = &family.get_metric()[0];
        let mut labels: Vec<_> = metric
            .get_label()
            .iter()
            .map(|label| label.get_name())
            .collect();
        labels.sort();
        assert_eq!(
            schema["labels"],
            serde_json::json!(labels),
            "{}",
            family.get_name()
        );
        if schema["type"] == "histogram" {
            let buckets: Vec<_> = metric
                .get_histogram()
                .get_bucket()
                .iter()
                .map(|bucket| bucket.get_upper_bound())
                .collect();
            assert_eq!(
                schema["buckets"],
                serde_json::json!(buckets),
                "{}",
                family.get_name()
            );
        }
    }
}

#[test]
fn scheduler_snapshots_count_tokens_and_weight_speculative_intervals_once() {
    let metrics = FrontendMetrics::new(&ServerArgs {
        metrics_config: Some(
            serde_json::json!({
                "allowed_custom_labels": ["tenant"], "priority_enabled": true,
                "bucket_inter_token_latency": [0.1, 0.25, 0.5, 1.0]
            })
            .to_string(),
        ),
        ..Default::default()
    })
    .unwrap();
    let start = metrics.clock.0;
    let request = GenerateRequest {
        started: Some(start),
        received_time: Some(metrics.clock.1 - 2.),
        stream: true,
        priority: Some(4),
        custom_labels: Some(BTreeMap::from([
            ("tenant".into(), "blue".into()),
            ("unlisted".into(), "discard".into()),
        ])),
        sampling_params: crate::message::sampling::SamplingParams {
            regex: Some(".*".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut state = RequestMetrics::new(metrics.clone(), &request).unwrap();
    for (seconds, tokens, finish) in [(1, 0, false), (2, 3, false), (3, 7, true), (4, 7, true)] {
        state.observe_at(
            &ChunkEvent {
                // Deliberately different from the cumulative scheduler count.
                completion_tokens: 99,
                prompt_tokens: 100,
                counts: Arc::new(TokenCounts {
                    generation_tokens: Some(tokens),
                    cached_tokens: 60,
                    cached_tokens_details: Some(Some(CachedTokensDetails {
                        device: 40,
                        host: 20,
                        storage: None,
                        storage_backend: None,
                    })),
                    spec_verify_ct: Some(2),
                    ..Default::default()
                }),
                finish_reason: finish.then(|| FinishKind::Length { length: Some(7) }.into()),
                ..Default::default()
            },
            start + Duration::from_secs(seconds),
        );
    }
    for (name, value) in [
        ("num_requests_total", 1.),
        ("prompt_tokens_total", 100.),
        ("generation_tokens_total", 7.),
        ("spec_verify_calls_total", 2.),
        ("num_so_requests_total", 1.),
    ] {
        let metric = sample(&metrics, &format!("sglang:{name}"));
        assert_eq!(metric.get_counter().get_value(), value);
        let labels: BTreeMap<_, _> = metric
            .get_label()
            .iter()
            .map(|label| (label.get_name(), label.get_value()))
            .collect();
        assert_eq!(labels["tenant"], "blue");
        assert_eq!(labels["priority"], "4");
        assert!(!labels.contains_key("unlisted"));
    }
    for (name, count, sum) in [
        ("time_to_first_token_seconds", 1, 3.),
        ("inter_token_latency_seconds", 7, 2.),
        ("e2e_request_latency_seconds", 1, 5.),
        ("uncached_prompt_tokens_histogram", 1, 40.),
    ] {
        let metric = sample(&metrics, &format!("sglang:{name}"));
        let histogram = metric.get_histogram();
        assert_eq!(histogram.get_sample_count(), count, "{name}");
        assert!((histogram.get_sample_sum() - sum).abs() < 1e-9, "{name}");
    }
    let itl = sample(&metrics, "sglang:inter_token_latency_seconds");
    let buckets = itl.get_histogram().get_bucket();
    assert_eq!(buckets[0].get_cumulative_count(), 0);
    assert_eq!(buckets[1].get_cumulative_count(), 4);
    assert_eq!(buckets[2].get_cumulative_count(), 7);
    let exposition = String::from_utf8(metrics.render().unwrap()).unwrap();
    assert!(exposition.contains("cache_source=\"device\""));
    assert!(exposition.contains("cache_source=\"host\""));
    assert!(!exposition.contains("cache_source=\"total\""));
}

#[test]
fn prefill_suppression_and_unfinished_drop_follow_python_accounting() {
    let metrics = FrontendMetrics::new(&ServerArgs {
        disaggregation_mode: DisaggregationMode::Prefill,
        ..Default::default()
    })
    .unwrap();
    assert!(
        RequestMetrics::new(
            metrics.clone(),
            &GenerateRequest {
                log_metrics: Some(false),
                ..Default::default()
            }
        )
        .is_none()
    );
    let mut state = RequestMetrics::new(metrics.clone(), &GenerateRequest::default()).unwrap();
    state.observe(&ChunkEvent {
        completion_tokens: 3,
        ..Default::default()
    });
    drop(state);
    let metric = sample(&metrics, "sglang:inter_token_latency_seconds");
    assert_eq!(metric.get_histogram().get_sample_count(), 3);
    assert_eq!(metric.get_histogram().get_sample_sum(), 0.);
    let text = String::from_utf8(metrics.render().unwrap()).unwrap();
    assert!(!text.contains("sglang:num_requests_total"));
    assert!(!text.contains("sglang:time_to_first_token_seconds"));
}

#[tokio::test]
async fn http_counts_live_until_body_completion_or_disconnect() {
    let metrics = FrontendMetrics::new(&ServerArgs::default()).unwrap();
    let app = apply_http(
        Router::new().route(
            "/generate",
            get(|| async {
                Body::from_stream(futures::stream::pending::<
                    Result<bytes::Bytes, std::io::Error>,
                >())
            }),
        ),
        metrics.clone(),
    );
    let request = || {
        Request::builder()
            .uri("/generate")
            .header("x-smg-routing-key", "same-session")
            .body(Body::empty())
            .unwrap()
    };
    let first = app.clone().oneshot(request()).await.unwrap();
    let second = app.clone().oneshot(request()).await.unwrap();
    assert_eq!(
        sample(&metrics, "sglang:http_requests_active")
            .get_gauge()
            .get_value(),
        2.
    );
    assert_eq!(
        sample(&metrics, "sglang:routing_keys_active")
            .get_gauge()
            .get_value(),
        1.
    );
    drop(first);
    assert_eq!(
        sample(&metrics, "sglang:http_requests_active")
            .get_gauge()
            .get_value(),
        1.
    );
    assert_eq!(
        sample(&metrics, "sglang:routing_keys_active")
            .get_gauge()
            .get_value(),
        1.
    );
    drop(second);
    assert_eq!(
        sample(&metrics, "sglang:http_requests_active")
            .get_gauge()
            .get_value(),
        0.
    );
    assert_eq!(
        sample(&metrics, "sglang:routing_keys_active")
            .get_gauge()
            .get_value(),
        0.
    );
    assert_eq!(
        sample(&metrics, "sglang:http_requests_total")
            .get_counter()
            .get_value(),
        2.
    );
    assert_eq!(
        sample(&metrics, "sglang:http_responses_total")
            .get_counter()
            .get_value(),
        2.
    );
    for path in ["/random-one", "/random-two"] {
        let response = app
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
    }
    let text = String::from_utf8(metrics.render().unwrap()).unwrap();
    assert!(text.contains("endpoint=\"unmatched\""));
    assert!(!text.contains("random-one"));
}
