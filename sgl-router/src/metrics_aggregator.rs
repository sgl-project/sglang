use openmetrics_parser::{MetricsExposition, PrometheusType, PrometheusValue};
use std::collections::HashMap;
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: HashMap<String, String>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let mut expositions = vec![];

    for metric_pack in metric_packs {
        let exposition =
            match openmetrics_parser::prometheus::parse_prometheus(&metric_pack.metrics_text) {
                Ok(x) => x,
                Err(err) => {
                    warn!(
                        "aggregate_metrics error when parsing text: pack={:?} err={:?}",
                        metric_pack, err
                    );
                    continue;
                }
            };
        let exposition = transform_metrics(exposition, &metric_pack.labels);
        expositions.push(exposition);
    }

    Ok("hi".into())
}

fn transform_metrics(
    mut exposition: PrometheusExposition,
    extra_labels: &HashMap<String, String>,
) -> PrometheusExposition {
    for (_, family) in &mut exposition.families {
        *family = family.with_labels(extra_labels);
    }
    exposition
}
