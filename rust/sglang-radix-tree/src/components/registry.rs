//! Native factories and per-tree component selections.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};

use super::{ComponentType, FullComponent, MambaComponent, SwaComponent, TreeComponent};
use crate::node::ChildKeyType;
use crate::unified_tree_core::CacheInitParams;

pub type TreeComponentInstance<K> = Arc<dyn TreeComponent<K> + Send + Sync>;

pub struct TreeComponentArgument<'a> {
    pub component_type: ComponentType,
    pub params: &'a CacheInitParams,
    pub is_bigram: bool,
}

/// A native factory supporting both tree key representations.
pub trait TreeComponentFactory: Send + Sync {
    fn create_plain(&self, argument: &TreeComponentArgument<'_>)
    -> TreeComponentInstance<Vec<i64>>;
    fn create_bigram(
        &self,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Vec<(i64, i64)>>;
}

impl<F, C> TreeComponentFactory for F
where
    F: Fn(&TreeComponentArgument<'_>) -> C + Send + Sync,
    C: TreeComponent<Vec<i64>> + TreeComponent<Vec<(i64, i64)>> + Send + Sync + 'static,
{
    fn create_plain(
        &self,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Vec<i64>> {
        Arc::new(self(argument))
    }

    fn create_bigram(
        &self,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Vec<(i64, i64)>> {
        Arc::new(self(argument))
    }
}

pub trait TreeComponentKey: ChildKeyType {
    fn create(
        factory: &dyn TreeComponentFactory,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Self>;
}

impl TreeComponentKey for Vec<i64> {
    fn create(
        factory: &dyn TreeComponentFactory,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Self> {
        factory.create_plain(argument)
    }
}

impl TreeComponentKey for Vec<(i64, i64)> {
    fn create(
        factory: &dyn TreeComponentFactory,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Self> {
        factory.create_bigram(argument)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TreeComponentRegistryError {
    #[error("component factory key must be non-empty")]
    EmptyKey,
    #[error("component factory {0:?} is already registered")]
    DuplicateKey(String),
    #[error("unknown component factory {0:?}")]
    UnknownKey(String),
    #[error("component {0:?} is not enabled")]
    InactiveComponent(ComponentType),
    #[error("duplicate component type {0:?}")]
    DuplicateComponent(ComponentType),
    #[error("component factory {name:?} has type {actual:?}, expected {expected:?}")]
    ComponentTypeMismatch {
        name: String,
        expected: ComponentType,
        actual: ComponentType,
    },
}

type FactoryEntry = (ComponentType, Arc<dyn TreeComponentFactory>);

pub struct TreeComponentRegistry {
    factories: RwLock<HashMap<String, FactoryEntry>>,
}

pub struct TreeComponentFactorySnapshot {
    factories: HashMap<ComponentType, Arc<dyn TreeComponentFactory>>,
}

impl TreeComponentFactorySnapshot {
    /// Invoke this tree's captured factory without holding a registry lock.
    pub fn create<K: TreeComponentKey>(
        &self,
        component_type: ComponentType,
        params: &CacheInitParams,
    ) -> TreeComponentInstance<K> {
        let component = K::create(
            &**self
                .factories
                .get(&component_type)
                .expect("component factory was not selected"),
            &TreeComponentArgument {
                component_type,
                params,
                is_bigram: K::IS_BIGRAM,
            },
        );
        assert_eq!(
            component.component_type(),
            component_type,
            "component factory returned the wrong kind for {component_type:?}"
        );
        component
    }
}

impl Default for TreeComponentRegistry {
    fn default() -> Self {
        let registry = Self {
            factories: RwLock::new(HashMap::new()),
        };
        registry
            .register_tree_component(
                "full_default",
                ComponentType::Full,
                |_: &TreeComponentArgument<'_>| FullComponent,
                false,
            )
            .unwrap();
        registry
            .register_tree_component(
                "swa_default",
                ComponentType::Swa,
                |argument: &TreeComponentArgument<'_>| SwaComponent::new(argument.params),
                false,
            )
            .unwrap();
        registry
            .register_tree_component(
                "mamba_default",
                ComponentType::Mamba,
                |argument: &TreeComponentArgument<'_>| MambaComponent::new(argument.params),
                false,
            )
            .unwrap();
        registry
    }
}

impl TreeComponentRegistry {
    /// Register or explicitly replace a factory for future trees, preserving its kind.
    pub fn register_tree_component(
        &self,
        name: &str,
        component_type: ComponentType,
        factory: impl TreeComponentFactory + 'static,
        replace: bool,
    ) -> Result<(), TreeComponentRegistryError> {
        if name.trim().is_empty() {
            return Err(TreeComponentRegistryError::EmptyKey);
        }
        let factory: Arc<dyn TreeComponentFactory> = Arc::new(factory);
        let previous = {
            let mut factories = self.factories.write().unwrap();
            if let Some((registered_type, _)) = factories.get(name) {
                if !replace {
                    return Err(TreeComponentRegistryError::DuplicateKey(name.to_owned()));
                }
                if *registered_type != component_type {
                    return Err(TreeComponentRegistryError::ComponentTypeMismatch {
                        name: name.to_owned(),
                        expected: *registered_type,
                        actual: component_type,
                    });
                }
            }
            factories.insert(name.to_owned(), (component_type, factory))
        };
        drop(previous);
        Ok(())
    }

    pub fn snapshot(
        &self,
        component_types: &[ComponentType],
        overrides: &HashMap<ComponentType, String>,
    ) -> Result<TreeComponentFactorySnapshot, TreeComponentRegistryError> {
        let mut configured = HashSet::new();
        for &component_type in component_types {
            if !configured.insert(component_type) {
                return Err(TreeComponentRegistryError::DuplicateComponent(
                    component_type,
                ));
            }
        }
        for (component_type, key) in overrides {
            if !configured.contains(component_type) {
                return Err(TreeComponentRegistryError::InactiveComponent(
                    *component_type,
                ));
            }
            if key.trim().is_empty() {
                return Err(TreeComponentRegistryError::EmptyKey);
            }
        }
        let registry = self.factories.read().unwrap();
        let factories = component_types
            .iter()
            .map(|&component_type| {
                let key = overrides
                    .get(&component_type)
                    .map(String::as_str)
                    .unwrap_or_else(|| default_factory_key(component_type));
                let (actual, factory) = registry
                    .get(key)
                    .ok_or_else(|| TreeComponentRegistryError::UnknownKey(key.to_owned()))?;
                if *actual != component_type {
                    return Err(TreeComponentRegistryError::ComponentTypeMismatch {
                        name: key.to_owned(),
                        expected: component_type,
                        actual: *actual,
                    });
                }
                Ok((component_type, Arc::clone(factory)))
            })
            .collect::<Result<_, _>>()?;
        Ok(TreeComponentFactorySnapshot { factories })
    }
}

fn tree_component_registry() -> &'static TreeComponentRegistry {
    static REGISTRY: OnceLock<TreeComponentRegistry> = OnceLock::new();
    REGISTRY.get_or_init(TreeComponentRegistry::default)
}

pub fn register_tree_component(
    name: &str,
    component_type: ComponentType,
    factory: impl TreeComponentFactory + 'static,
    replace: bool,
) -> Result<(), TreeComponentRegistryError> {
    tree_component_registry().register_tree_component(name, component_type, factory, replace)
}

pub fn resolve_tree_component_factories(
    component_types: &[ComponentType],
    overrides: &HashMap<ComponentType, String>,
) -> Result<TreeComponentFactorySnapshot, TreeComponentRegistryError> {
    tree_component_registry().snapshot(component_types, overrides)
}

pub fn default_factory_key(component_type: ComponentType) -> &'static str {
    match component_type {
        ComponentType::Full => "full_default",
        ComponentType::Swa => "swa_default",
        ComponentType::Mamba => "mamba_default",
    }
}

#[cfg(test)]
#[path = "../tests/components/registry.rs"]
mod tests;
