//! `prometheus_client`'s multiprocess mode, ported: the `MmapedDict` file
//! format (`prometheus_client/mmap_dict.py`), the merge its
//! `MultiProcessCollector` runs on each scrape (`multiprocess.py`), and the
//! text format `generate_latest` renders (`exposition.py`).
//!
//! With metrics enabled, every Python process writes its samples to files under
//! `PROMETHEUS_MULTIPROC_DIR`, and Python's `/metrics` merges those files per
//! scrape. Reading them here serves the same text without taking the GIL.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::hash::Hash;
use std::io;
use std::path::Path;

/// One `MmapedDict` entry: the JSON key (Python's `mmap_key`), the value and
/// the timestamp (read only by `mostrecent` gauges).
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub key: String,
    pub value: f64,
    pub timestamp: f64,
}

/// The used-length `i32` plus four bytes of padding.
const HEADER_LEN: usize = 8;

/// A key's length once padded so the two `f64`s after it are 8-byte aligned.
fn padded_key_len(len: usize) -> usize {
    len + (8 - (len + 4) % 8)
}

fn read_array<const N: usize>(data: &[u8], pos: usize) -> Option<[u8; N]> {
    data.get(pos..pos.checked_add(N)?)?.try_into().ok()
}

/// Decode a `MmapedDict` file (`_read_all_values`): the used length, then
/// entries of an `i32` key length, the UTF-8 key padded with spaces, the value
/// and the timestamp. Errors where Python's reader raises.
pub fn read_entries(data: &[u8]) -> Result<Vec<Entry>, String> {
    let used = read_array::<4>(data, 0)
        .map(u32::from_le_bytes)
        .ok_or("file shorter than its header")? as usize;
    if used > data.len() {
        return Err(format!(
            "used length {used} exceeds the file's {} bytes",
            data.len()
        ));
    }
    let mut entries = Vec::new();
    let mut pos = HEADER_LEN;
    while pos < used {
        let len = read_array::<4>(data, pos)
            .map(u32::from_le_bytes)
            .ok_or("truncated entry")? as usize;
        if len + pos > used {
            return Err("read beyond file size detected, file is corrupted".into());
        }
        pos += 4;
        // Python's check above leaves the key up to four bytes past `used`.
        let key = data.get(pos..pos + len).ok_or("truncated entry")?;
        let key = std::str::from_utf8(key).map_err(|error| format!("key is not UTF-8: {error}"))?;
        pos += padded_key_len(len);
        let (Some(value), Some(timestamp)) =
            (read_array::<8>(data, pos), read_array::<8>(data, pos + 8))
        else {
            return Err("truncated entry".into());
        };
        entries.push(Entry {
            key: key.to_owned(),
            value: f64::from_le_bytes(value),
            timestamp: f64::from_le_bytes(timestamp),
        });
        pos += 16;
    }
    Ok(entries)
}

type Labels = Vec<(String, String)>;

/// One merged sample, labels in the order Python accumulates them.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub name: String,
    pub labels: Labels,
    pub value: f64,
}

/// One merged metric family. `kind` is the file type (`counter`, `gauge`,
/// `histogram`, `summary`).
#[derive(Clone, Debug, PartialEq)]
pub struct Family {
    pub name: String,
    pub help: String,
    pub kind: String,
    pub samples: Vec<Sample>,
}

/// An insertion-ordered map, standing in for Python's ordered `dict`.
struct Ordered<K, V> {
    index: HashMap<K, usize>,
    items: Vec<(K, V)>,
}

impl<K: Hash + Eq + Clone, V> Ordered<K, V> {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            items: Vec::new(),
        }
    }

    fn get_or_insert_with(&mut self, key: K, value: impl FnOnce() -> V) -> &mut V {
        let i = match self.index.get(&key) {
            Some(&i) => i,
            None => {
                self.index.insert(key.clone(), self.items.len());
                self.items.push((key, value()));
                self.items.len() - 1
            }
        };
        &mut self.items[i].1
    }

    fn insert(&mut self, key: K, value: V) {
        match self.index.get(&key) {
            Some(&i) => self.items[i].1 = value,
            None => {
                self.index.insert(key.clone(), self.items.len());
                self.items.push((key, value));
            }
        }
    }
}

struct RawSample {
    name: String,
    labels: Labels,
    value: f64,
    timestamp: f64,
}

struct RawMetric {
    help: String,
    kind: String,
    mode: String,
    samples: Vec<RawSample>,
}

/// `MultiProcessCollector.merge(files, accumulate=True)`: `files` are
/// `(file name, bytes)` in the order Python's glob lists them. A file Python
/// would fail to read is skipped with a warning instead of failing the scrape.
pub fn merge<'a>(files: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> Vec<Family> {
    let mut metrics: Ordered<String, RawMetric> = Ordered::new();
    let mut parsed_keys: HashMap<String, (String, String, Labels, String)> = HashMap::new();
    for (file_name, data) in files {
        let parts: Vec<&str> = file_name.split('_').collect();
        let kind = parts[0];
        let entries = match read_entries(data) {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!(file = file_name, %error, "skipping unreadable metrics file");
                continue;
            }
        };
        for entry in entries {
            let parsed = match parsed_keys.get(&entry.key) {
                Some(parsed) => parsed,
                None => match parse_key(&entry.key) {
                    Some(parsed) => parsed_keys.entry(entry.key.clone()).or_insert(parsed),
                    None => {
                        tracing::warn!(file = file_name, key = entry.key, "skipping bad key");
                        continue;
                    }
                },
            };
            let (metric_name, name, labels, help) = parsed;
            let metric = metrics.get_or_insert_with(metric_name.clone(), || RawMetric {
                help: help.clone(),
                kind: kind.to_owned(),
                mode: String::new(),
                samples: Vec::new(),
            });
            let mut labels = labels.clone();
            if kind == "gauge" {
                metric.mode = parts.get(1).copied().unwrap_or_default().to_owned();
                let pid = parts.get(2).copied().unwrap_or_default();
                labels.push(("pid".into(), pid[..pid.len().saturating_sub(3)].to_owned()));
            }
            metric.samples.push(RawSample {
                name: name.clone(),
                labels,
                value: entry.value,
                timestamp: entry.timestamp,
            });
        }
    }
    metrics
        .items
        .into_iter()
        .map(|(name, metric)| accumulate(name, metric))
        .collect()
}

/// `json.loads(key)` into `(metric_name, name, labels sorted by name, help)`.
fn parse_key(key: &str) -> Option<(String, String, Labels, String)> {
    let (metric_name, name, labels, help): (
        String,
        String,
        serde_json::Map<String, serde_json::Value>,
        String,
    ) = serde_json::from_str(key).ok()?;
    let mut labels: Labels = labels
        .into_iter()
        .map(|(key, value)| match value {
            serde_json::Value::String(value) => (key, value),
            value => (key, value.to_string()),
        })
        .collect();
    labels.sort();
    Some((metric_name, name, labels, help))
}

/// `MultiProcessCollector._accumulate_metrics` for one metric.
fn accumulate(name: String, metric: RawMetric) -> Family {
    let mode = metric.mode.as_str();
    let gauge = metric.kind == "gauge";
    let histogram = metric.kind == "histogram";
    let drop_pid = gauge
        && matches!(
            mode,
            "min"
                | "livemin"
                | "max"
                | "livemax"
                | "sum"
                | "livesum"
                | "mostrecent"
                | "livemostrecent"
        );
    let mut samples: Ordered<Labels, Ordered<(String, Labels), f64>> = Ordered::new();
    let mut timestamps: HashMap<(Labels, String), f64> = HashMap::new();
    let mut buckets: Ordered<Labels, Vec<(f64, f64)>> = Ordered::new();
    for s in metric.samples {
        let labels = if drop_pid {
            s.labels.into_iter().filter(|(k, _)| k != "pid").collect()
        } else {
            s.labels
        };
        if gauge {
            let key = (s.name.clone(), labels.clone());
            match mode {
                "min" | "livemin" | "max" | "livemax" => {
                    let group = samples.get_or_insert_with(labels, Ordered::new);
                    let current = group.get_or_insert_with(key, || s.value);
                    let replace = if mode.ends_with("min") {
                        s.value < *current
                    } else {
                        s.value > *current
                    };
                    if replace {
                        *current = s.value;
                    }
                }
                "sum" | "livesum" => {
                    *samples
                        .get_or_insert_with(labels, Ordered::new)
                        .get_or_insert_with(key, || 0.0) += s.value;
                }
                "mostrecent" | "livemostrecent" => {
                    let current = timestamps
                        .entry((labels.clone(), s.name.clone()))
                        .or_insert(0.0);
                    if *current < s.timestamp {
                        *current = s.timestamp;
                        samples
                            .get_or_insert_with(labels, Ordered::new)
                            .insert(key, s.value);
                    }
                }
                _ => samples
                    .get_or_insert_with(labels, Ordered::new)
                    .insert(key, s.value),
            }
        } else if histogram && let Some((_, le)) = labels.iter().find(|(k, _)| k == "le") {
            let bound = parse_python_float(le);
            let without_le: Labels = labels.iter().filter(|(k, _)| k != "le").cloned().collect();
            let values = buckets.get_or_insert_with(without_le, Vec::new);
            match values
                .iter_mut()
                .find(|(b, _)| b.to_bits() == bound.to_bits())
            {
                Some((_, value)) => *value += s.value,
                None => values.push((bound, s.value)),
            }
        } else {
            *samples
                .get_or_insert_with(labels.clone(), Ordered::new)
                .get_or_insert_with((s.name, labels), || 0.0) += s.value;
        }
    }
    for (labels, mut values) in buckets.items {
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        let group = samples.get_or_insert_with(labels.clone(), Ordered::new);
        let mut acc = 0.0;
        for (bound, value) in values {
            acc += value;
            let mut bucket_labels = labels.clone();
            bucket_labels.push(("le".into(), float_to_go_string(bound)));
            group.insert((format!("{name}_bucket"), bucket_labels), acc);
        }
        group.insert((format!("{name}_count"), labels), acc);
    }
    Family {
        samples: samples
            .items
            .into_iter()
            .flat_map(|(_, group)| group.items)
            .map(|((name, labels), value)| Sample {
                name,
                labels,
                value,
            })
            .collect(),
        name,
        help: metric.help,
        kind: metric.kind,
    }
}

/// Python `float()` of a `le` label (`floatToGoString` output, so `+Inf` too).
fn parse_python_float(s: &str) -> f64 {
    match s {
        "+Inf" | "inf" | "Inf" => f64::INFINITY,
        "-Inf" | "-inf" => f64::NEG_INFINITY,
        _ => s.parse().unwrap_or(f64::NAN),
    }
}

/// `generate_latest`'s text format (0.0.4) for merged families.
pub fn render(families: &[Family]) -> String {
    let mut out = String::new();
    for family in families {
        let (mname, mtype) = match family.kind.as_str() {
            "counter" => (format!("{}_total", family.name), "counter"),
            kind => (family.name.clone(), kind),
        };
        let help = escape_help(&family.help);
        let _ = writeln!(out, "# HELP {} {help}", escape_name(&mname, true));
        let _ = writeln!(out, "# TYPE {} {mtype}", escape_name(&mname, true));
        // OpenMetrics-only samples go into their own gauge after the family.
        let mut om_samples: BTreeMap<&str, Vec<&Sample>> = BTreeMap::new();
        for sample in &family.samples {
            let suffix = ["_created", "_gsum", "_gcount"].into_iter().find(|suffix| {
                sample.name.len() == family.name.len() + suffix.len()
                    && sample.name.starts_with(&family.name)
                    && sample.name.ends_with(suffix)
            });
            match suffix {
                Some(suffix) => om_samples.entry(suffix).or_default().push(sample),
                None => push_sample_line(&mut out, sample),
            }
        }
        for (suffix, samples) in om_samples {
            let name = escape_name(&format!("{}{suffix}", family.name), true);
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} gauge");
            for sample in samples {
                push_sample_line(&mut out, sample);
            }
        }
    }
    out
}

fn push_sample_line(out: &mut String, sample: &Sample) {
    out.push_str(&escape_name(&sample.name, true));
    if !sample.labels.is_empty() {
        let mut labels: Vec<&(String, String)> = sample.labels.iter().collect();
        labels.sort_by(|a, b| a.0.cmp(&b.0));
        out.push('{');
        for (i, (key, value)) in labels.into_iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{}=\"{}\"",
                escape_name(key, false),
                escape_label_value(value)
            );
        }
        out.push('}');
    }
    let _ = writeln!(out, " {}", float_to_go_string(sample.value));
}

fn escape_help(help: &str) -> String {
    help.replace('\\', r"\\").replace('\n', r"\n")
}

fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', r#"\""#)
}

/// The `underscores` escaping of a metric (`colon`) or label name: a rune
/// outside the legacy character set becomes `_`.
fn escape_name(name: &str, colon: bool) -> String {
    name.chars()
        .enumerate()
        .map(|(i, c)| {
            let legacy = c.is_ascii_alphabetic()
                || c == '_'
                || (c.is_ascii_digit() && i > 0)
                || (colon && c == ':');
            if legacy { c } else { '_' }
        })
        .collect()
}

/// `prometheus_client.utils.floatToGoString`.
pub fn float_to_go_string(d: f64) -> String {
    if d == f64::INFINITY {
        return "+Inf".into();
    }
    if d == f64::NEG_INFINITY {
        return "-Inf".into();
    }
    if d.is_nan() {
        return "NaN".into();
    }
    let s = python_float_repr(d);
    // Go switches to exponents sooner than Python.
    match s.find('.') {
        Some(dot) if d > 0.0 && dot > 6 => {
            let mantissa = format!("{}.{}{}", &s[..1], &s[1..dot], &s[dot + 1..]);
            let mantissa = mantissa.trim_end_matches(['0', '.']);
            format!("{mantissa}e+0{}", dot - 1)
        }
        _ => s,
    }
}

/// Python `repr(float)` for a finite value: the shortest round-trip digits,
/// fixed-point for decimal exponents in `[-4, 16)`, scientific otherwise.
fn python_float_repr(d: f64) -> String {
    if d == 0.0 {
        return if d.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` gives the same shortest digits: `-1.25e-5`, `1e16`.
    let sci = format!("{d:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(mantissa) => ("-", mantissa),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        if exp < 0 {
            format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        } else {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{sign}{digits}{}.0", "0".repeat(int_len - digits.len()))
            } else {
                format!("{sign}{}.{}", &digits[..int_len], &digits[int_len..])
            }
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{first}{rest}e{exp_sign}{:02}", exp.abs())
    }
}

/// The files Python's `glob(os.path.join(path, '*.db'))` matches, in
/// directory order, with their contents. A file that vanishes between listing
/// and reading (a dead process's live gauge) is skipped, as Python does.
pub fn read_dir_files(dir: &Path) -> io::Result<Vec<(String, Vec<u8>)>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if name.starts_with('.') || !name.ends_with(".db") {
            continue;
        }
        match std::fs::read(entry.path()) {
            Ok(data) => files.push((name, data)),
            Err(error) => tracing::warn!(file = name, %error, "skipping unreadable metrics file"),
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Fixture {
        files: Vec<FixtureFile>,
        text: String,
    }

    #[derive(serde::Deserialize)]
    struct FixtureFile {
        name: String,
        hex: String,
    }

    fn unhex(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn fixture() -> (Vec<(String, Vec<u8>)>, String) {
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../testdata/prometheus_multiprocess.json"))
                .unwrap();
        let files = fixture
            .files
            .into_iter()
            .map(|file| (file.name, unhex(&file.hex)))
            .collect();
        (files, fixture.text)
    }

    /// Files written by `prometheus_client` itself (several processes, every
    /// metric type and gauge mode, escapes, unicode) merge and render to
    /// exactly what its `MultiProcessCollector` and `generate_latest` produce.
    #[test]
    fn renders_python_multiprocess_files_as_python_does() {
        let (files, text) = fixture();
        let families = merge(
            files
                .iter()
                .map(|(name, data)| (name.as_str(), data.as_slice())),
        );
        assert_eq!(render(&families), text);
    }

    #[test]
    fn corrupt_files_are_skipped() {
        let (files, _) = fixture();
        let (name, data) = &files[0];
        let mut corrupt = data.clone();
        // A key length running past the used length.
        corrupt[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(read_entries(&corrupt).is_err());
        assert!(read_entries(&data[..2]).is_err());
        // A key that passes Python's length check but ends past the file.
        let mut short = vec![0u8; 16];
        short[..4].copy_from_slice(&16u32.to_le_bytes());
        short[8..12].copy_from_slice(&6u32.to_le_bytes());
        assert!(read_entries(&short).is_err());
        assert!(merge([(name.as_str(), corrupt.as_slice())]).is_empty());
    }

    #[test]
    fn float_formatting_matches_python() {
        for (value, repr, go) in [
            (0.0, "0.0", "0.0"),
            (-0.0, "-0.0", "-0.0"),
            (1.0, "1.0", "1.0"),
            (0.1, "0.1", "0.1"),
            (0.0001, "0.0001", "0.0001"),
            (0.00001, "1e-05", "1e-05"),
            (1.5e-7, "1.5e-07", "1.5e-07"),
            (123.456, "123.456", "123.456"),
            (100000.0, "100000.0", "100000.0"),
            (1000000.0, "1000000.0", "1e+06"),
            (1100000.0, "1100000.0", "1.1e+06"),
            (12345678.9, "12345678.9", "1.23456789e+07"),
            (1e15, "1000000000000000.0", "1e+015"),
            (1e16, "1e+16", "1e+16"),
            (1.2345e20, "1.2345e+20", "1.2345e+20"),
            (-2500000.0, "-2500000.0", "-2500000.0"),
            (1e-300, "1e-300", "1e-300"),
        ] {
            assert_eq!(python_float_repr(value), repr, "{value}");
            assert_eq!(float_to_go_string(value), go, "{value}");
        }
        assert_eq!(float_to_go_string(f64::INFINITY), "+Inf");
        assert_eq!(float_to_go_string(f64::NEG_INFINITY), "-Inf");
        assert_eq!(float_to_go_string(f64::NAN), "NaN");
    }

    #[test]
    fn lists_the_files_python_globs() {
        let dir = std::env::temp_dir().join(format!("prom-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (files, _) = fixture();
        std::fs::write(dir.join(&files[0].0), &files[0].1).unwrap();
        std::fs::write(dir.join(".hidden.db"), b"").unwrap();
        std::fs::write(dir.join("counter_1.db.tmp"), b"").unwrap();
        let listed = read_dir_files(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(listed, files[..1]);
    }
}
