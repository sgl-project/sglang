use std::collections::HashMap;
use openmetrics_parser::{MetricsExposition, PrometheusType, PrometheusValue};

pub struct MetricPack {
    pub labels: HashMap<String, String>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    for metric_pack in metric_packs {
        let exposition =
            match openmetrics_parser::prometheus::parse_prometheus(&metric_pack.metrics_text) {
                Ok(e) => e,
                Err(err) => {
                    eprintln!("hi err={err:?}");
                    continue;
                },
            };
        eprintln!("exposition={}", exposition);
        let exposition = transform_metrics(exposition);
    }

    Ok("hi".into())
}

fn transform_metrics(mut exposition: PrometheusExposition) -> PrometheusExposition {
    for (_, family) in &mut exposition.families {
        eprintln!("family={}", family);
    }

    exposition
}
