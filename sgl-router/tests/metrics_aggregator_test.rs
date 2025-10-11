#[path = "../src/metrics_aggregator.rs"]
mod metrics_aggregator;

use metrics_aggregator::{aggregate_metrics, MetricPack};

#[test]
fn test_aggregate_simple() {
    let pack1 = MetricPack {
        labels: vec![("source".to_string(), "worker1".to_string())],
        metrics_text: r#"
# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{method="post",code="200"} 1027
http_requests_total{method="post",code="400"} 3
"#.to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "worker2".to_string())],
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
        labels: vec![("source".to_string(), "w1".to_string())],
        metrics_text: r#"
# TYPE metric_a gauge
metric_a{dim="x"} 1.0
# TYPE metric_b_total counter
metric_b_total 10
"#.to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "w2".to_string())],
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
        labels: vec![],
        metrics_text: "invalid metrics text".to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "worker1".to_string())],
        metrics_text: "# TYPE valid_metric gauge\nvalid_metric 123\n".to_string(),
    };
    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = r#"# TYPE valid_metric gauge
valid_metric{source="worker1"} 123
"#;
    assert_eq!(result.trim(), expected.trim());
}

#[test]
fn test_real() {
    let pack1 = MetricPack {
        labels: vec![],
        // https://docs.sglang.ai/references/production_metrics.html
        metrics_text: r###"
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.128902e+06
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.557572e+06
# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.28
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.007507552643049313
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2.3518979474117756e+06
sglang:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.116093850019932e+06
sglang:e2e_request_latency_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:e2e_request_latency_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="0.8",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 866964.5791549598
sglang:time_per_output_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 73.0
sglang:time_per_output_token_seconds_bucket{le="0.015",model_name="meta-llama/Llama-3.1-8B-Instruct"} 382.0
sglang:time_per_output_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_sum{name="generate_request"} 4.514771912145079
sglang:func_latency_seconds_bucket{le="0.05",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.07500000000000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.1125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.16875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="+Inf",name="generate_request"} 14007.0
sglang:func_latency_seconds_count{name="generate_request"} 14007.0
# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 162.0
# HELP sglang:num_used_tokens The number of used tokens
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct"} 123859.0
# HELP sglang:gen_throughput The generate throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct"} 86.50814177726902
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2826.0
"###.to_string(),
    };
    let pack2 = MetricPack {
        labels: vec![("source".to_string(), "worker1".to_string())],
        metrics_text: pack1.metrics_text.clone(),
    };
    let result = aggregate_metrics(vec![pack1, pack2]).unwrap();
    let expected = "TODO";
    assert_eq!(result.trim(), expected.trim());
}

