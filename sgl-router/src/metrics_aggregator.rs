use openmetrics_parser::{MetricFamily, MetricsExposition, PrometheusType, PrometheusValue};
use std::collections::hash_map::Entry;
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricsFamily<PrometheusType, PrometheusValue>;

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

    Ok(expositions
        .into_iter()
        .map(|x| format!("{x}"))
        .collect::<Vec<_>>()
        .join("\n"))
}

fn transform_metrics(
    mut exposition: PrometheusExposition,
    extra_labels: &Vec<(String, String)>,
) -> PrometheusExposition {
    for (_, family) in &mut exposition.families {
        *family = family.with_labels(extra_labels.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    exposition
}

fn merge_exposition(a: PrometheusExposition, b: PrometheusExposition) -> PrometheusExposition {
    let mut ans = a;
    for (name, family_b) in b.families.into_iter() {
        ans[name] = if let Some(family_a) = ans.families.remove(&name) {
            merge_family(family_a, family_b)
        } else {
            family_b
        };
    }
    ans
}

fn merge_family(a: PrometheusFamily, b: PrometheusFamily) -> PrometheusFamily {
    TDO
}
