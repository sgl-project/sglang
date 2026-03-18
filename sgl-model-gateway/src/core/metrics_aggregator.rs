use anyhow::ensure;
use openmetrics_parser::{MetricFamily, MetricsExposition, PrometheusType, PrometheusValue};
use tracing::warn;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricFamily<PrometheusType, PrometheusValue>;

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let mut expositions = vec![];
    for metric_pack in metric_packs {
        let metrics_text = &metric_pack.metrics_text;
        // openmetrics_parser doesn't handle colons in metric names; replace with underscores
        let metrics_text = metrics_text.replace(":", "_");

        let exposition = match openmetrics_parser::prometheus::parse_prometheus(&metrics_text) {
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

    let text = try_reduce(expositions.into_iter(), merge_exposition)?
        .map(|x| format!("{x}"))
        .unwrap_or_default();
    Ok(text)
}

fn transform_metrics(
    mut exposition: PrometheusExposition,
    extra_labels: &[(String, String)],
) -> PrometheusExposition {
    for family in exposition.families.values_mut() {
        *family = family.with_labels(extra_labels.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    exposition
}

fn merge_exposition(
    a: PrometheusExposition,
    b: PrometheusExposition,
) -> anyhow::Result<PrometheusExposition> {
    let mut ans = a;
    for (name, family_b) in b.families.into_iter() {
        let family_merged = if let Some(family_a) = ans.families.remove(&name) {
            merge_family(family_a, family_b)?
        } else {
            family_b
        };
        ans.families.insert(name, family_merged);
    }
    Ok(ans)
}

fn merge_family(
    mut a: PrometheusFamily,
    mut b: PrometheusFamily,
) -> anyhow::Result<PrometheusFamily> {
    // When label schemas differ (e.g. PD disaggregation mode where decode workers
    // have a `dp_rank` label but prefill workers do not), pad the missing labels
    // with a default empty-string value so both families share the same schema.
    if a.get_label_names() != b.get_label_names() {
        use std::collections::HashSet;

        let labels_a: HashSet<String> = a.get_label_names().iter().cloned().collect();
        let labels_b: HashSet<String> = b.get_label_names().iter().cloned().collect();

        // Labels present in b but missing in a
        let mut missing_in_a: Vec<String> = labels_b.difference(&labels_a).cloned().collect();
        missing_in_a.sort(); // ensure deterministic order

        // Labels present in a but missing in b
        let mut missing_in_b: Vec<String> = labels_a.difference(&labels_b).cloned().collect();
        missing_in_b.sort(); // ensure deterministic order

        if !missing_in_a.is_empty() {
            a = a.with_labels(missing_in_a.iter().map(|l| (l.as_str(), "")));
        }
        if !missing_in_b.is_empty() {
            b = b.with_labels(missing_in_b.iter().map(|l| (l.as_str(), "")));
        }
    }

    a.with_samples(b.into_iter_samples())
        .map_err(|e| anyhow::anyhow!("failed to merge samples: {e:?}"))
}

fn try_reduce<I, T, E, F>(iterable: I, f: F) -> Result<Option<T>, E>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T, T) -> Result<T, E>,
{
    let mut it = iterable.into_iter();
    let first = match it.next() {
        None => return Ok(None),
        Some(x) => x,
    };

    Ok(Some(it.try_fold(first, f)?))
}
