#[path = "../src/metrics_aggregator.rs"]
mod metrics_aggregator;

use std::collections::HashMap;
use metrics_aggregator::{aggregate_metrics, MetricPack};

#[test]
fn test_aggregate_simple() {
    let pack1 = MetricPack {
        labels: HashMap::from([("source".to_string(), "worker1".to_string())]),
        metrics_text: r#"
# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{method="post",code="200"} 1027
http_requests_total{method="post",code="400"} 3
"#.to_string(),
    };
    let pack2 = MetricPack {
        labels: HashMap::from([("source".to_string(), "worker2".to_string())]),
        metrics_text: r#"
# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{method="post",code="200"} 500
"#.to_string(),
    };

    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{code="200",method="post",source="worker1"} 1027
http_requests_total{code="400",method="post",source="worker1"} 3
http_requests_total{code="200",method="post",source="worker2"} 500
"#;
    assert_eq!(result.trim(), expected.trim());
}

#[test]
fn test_aggregate_multiple_metrics() {
    let pack1 = MetricPack {
        labels: HashMap::from([("source".to_string(), "w1".to_string())]),
        metrics_text: r#"
# TYPE metric_a gauge
metric_a{dim="x"} 1.0
# TYPE metric_b_total counter
metric_b_total 10
"#.to_string(),
    };
    let pack2 = MetricPack {
        labels: HashMap::from([("source".to_string(), "w2".to_string())]),
        metrics_text: r#"
# TYPE metric_a gauge
metric_a{dim="y"} 2.0
"#.to_string(),
    };

    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# TYPE metric_a gauge
metric_a{dim="x",source="w1"} 1
metric_a{dim="y",source="w2"} 2

# TYPE metric_b_total counter
metric_b_total{source="w1"} 10
"#;
    // Split into lines and sort to handle BTreeMap ordering issues between test environments
    let mut result_lines: Vec<_> = result.trim().lines().map(|l| l.trim()).collect();
    let mut expected_lines: Vec<_> = expected.trim().lines().map(|l| l.trim()).collect();
    result_lines.sort();
    expected_lines.sort();
    assert_eq!(result_lines, expected_lines);
}

#[test]
fn test_empty_input() {
    let result = aggregate_metrics(vec![]).unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_invalid_metrics_are_skipped() {
    let pack1 = MetricPack {
        labels: HashMap::new(),
        metrics_text: "invalid metrics text".to_string(),
    };
    let pack2 = MetricPack {
        labels: HashMap::from([("source".to_string(), "worker1".to_string())]),
        metrics_text: "# TYPE valid_metric gauge\nvalid_metric 123\n".to_string(),
    };
    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# TYPE valid_metric gauge
valid_metric{source="worker1"} 123
"#;
    assert_eq!(result.trim(), expected.trim());
}
