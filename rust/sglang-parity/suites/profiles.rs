//! The single catalog of startup scenarios shipped with the API suites.

use std::collections::{BTreeMap, BTreeSet};

use sglang_parity::{ProfileSpec, ResolvedProfile, ServerConfig};

/// Resolve only explicit case bindings, in stable profile-name order.
pub fn resolve<'a>(
    base: &ServerConfig,
    bindings: impl Iterator<Item = &'a Vec<String>>,
) -> Result<Vec<ResolvedProfile>, String> {
    let definitions: BTreeMap<String, ProfileSpec> =
        serde_json::from_str(include_str!("profiles.json")).map_err(|e| e.to_string())?;
    let mut selected = BTreeSet::new();
    for names in bindings {
        let mut unique = BTreeSet::new();
        if names.is_empty() || names.iter().any(|name| !unique.insert(name)) {
            return Err("case profiles must be nonempty and distinct".into());
        }
        selected.extend(names);
    }
    selected
        .into_iter()
        .map(|name| {
            definitions
                .get(name)
                .ok_or_else(|| format!("unknown profile {name:?}"))?
                .resolve(name, base)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_explicit_bindings_are_resolved() {
        let base = serde_json::from_value(serde_json::json!({"model":"fixture"})).unwrap();
        for names in [
            vec![],
            vec!["unknown".into()],
            vec!["default".into(), "default".into()],
        ] {
            assert!(resolve(&base, [&names].into_iter()).is_err());
        }
        let names = vec!["incremental".into()];
        let profiles = resolve(&base, [&names].into_iter()).unwrap();
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].id, "incremental");
        assert!(profiles[0].server.incremental_output());
        let names = vec!["default".into()];
        assert_eq!(
            resolve(&base, [&names].into_iter()).unwrap()[0].server,
            base
        );
    }
}
