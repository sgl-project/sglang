use openmetrics_parser::{PrometheusType, PrometheusValue};
use prometheus::Encoder;
use std::collections::HashMap;
use std::string::FromUtf8Error;

pub struct MetricPack {
    pub labels: HashMap<String, String>,
    pub metrics_text: String,
}

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    for metric_pack in metric_packs {
        let exposition =
            match openmetrics_parser::prometheus::parse_prometheus(&metric_pack.metrics_text) {
                Ok(e) => e,
                Err(_) => continue,
            };

        for (_, family) in exposition.families {
            TODO
        }
    }

    encode_metric_families(output_metric_families)
}

fn encode_metric_families(
    metric_families: &[prometheus::proto::MetricFamily],
) -> anyhow::Result<String> {
    let mut buffer = vec![];
    let encoder = prometheus::TextEncoder::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    Ok(String::from_utf8(buffer)?)
}
