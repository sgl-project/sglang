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

/// A native factory for a tree's key representation.
pub trait TreeComponentFactory<K: ChildKeyType>: Send + Sync {
    fn create(&self, argument: &TreeComponentArgument<'_>) -> TreeComponentInstance<K>;
}

impl<K, F, C> TreeComponentFactory<K> for F
where
    K: ChildKeyType,
    F: Fn(&TreeComponentArgument<'_>) -> C + Send + Sync,
    C: TreeComponent<K> + Send + Sync + 'static,
{
    fn create(&self, argument: &TreeComponentArgument<'_>) -> TreeComponentInstance<K> {
        Arc::new(self(argument))
    }
}

/// A family of factories supporting both concrete tree key representations.
pub trait TreeComponentFactoryFamily:
    TreeComponentFactory<Vec<i64>> + TreeComponentFactory<Vec<(i64, i64)>>
{
}

impl<F> TreeComponentFactoryFamily for F where
    F: TreeComponentFactory<Vec<i64>> + TreeComponentFactory<Vec<(i64, i64)>>
{
}

pub trait TreeComponentKey: ChildKeyType {
    fn create(
        factory: &dyn TreeComponentFactoryFamily,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Self>;
}

impl TreeComponentKey for Vec<i64> {
    fn create(
        factory: &dyn TreeComponentFactoryFamily,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Self> {
        TreeComponentFactory::<Self>::create(factory, argument)
    }
}

impl TreeComponentKey for Vec<(i64, i64)> {
    fn create(
        factory: &dyn TreeComponentFactoryFamily,
        argument: &TreeComponentArgument<'_>,
    ) -> TreeComponentInstance<Self> {
        TreeComponentFactory::<Self>::create(factory, argument)
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
}

pub struct TreeComponentRegistry {
    factories: RwLock<HashMap<String, Arc<dyn TreeComponentFactoryFamily>>>,
}

/// Factory references resolved together for one tree's construction.
pub struct ResolvedTreeComponentFactories {
    factories: HashMap<ComponentType, Arc<dyn TreeComponentFactoryFamily>>,
}

impl ResolvedTreeComponentFactories {
    /// Invoke this tree's selected factory without holding a registry lock.
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
                |_: &TreeComponentArgument<'_>| FullComponent,
                false,
            )
            .unwrap();
        registry
            .register_tree_component(
                "swa_default",
                |argument: &TreeComponentArgument<'_>| SwaComponent::new(argument.params),
                false,
            )
            .unwrap();
        registry
            .register_tree_component(
                "mamba_default",
                |argument: &TreeComponentArgument<'_>| MambaComponent::new(argument.params),
                false,
            )
            .unwrap();
        registry
    }
}

impl TreeComponentRegistry {
    /// Register or explicitly replace a named factory for future trees.
    pub fn register_tree_component(
        &self,
        name: &str,
        factory: impl TreeComponentFactoryFamily + 'static,
        replace: bool,
    ) -> Result<(), TreeComponentRegistryError> {
        if name.trim().is_empty() {
            return Err(TreeComponentRegistryError::EmptyKey);
        }
        let factory: Arc<dyn TreeComponentFactoryFamily> = Arc::new(factory);
        let previous = {
            let mut factories = self.factories.write().unwrap();
            if !replace && factories.contains_key(name) {
                return Err(TreeComponentRegistryError::DuplicateKey(name.to_owned()));
            }
            factories.insert(name.to_owned(), factory)
        };
        drop(previous);
        Ok(())
    }

    pub fn resolve(
        &self,
        component_types: &[ComponentType],
        overrides: &HashMap<ComponentType, String>,
    ) -> Result<ResolvedTreeComponentFactories, TreeComponentRegistryError> {
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
                let factory = registry
                    .get(key)
                    .ok_or_else(|| TreeComponentRegistryError::UnknownKey(key.to_owned()))?;
                Ok((component_type, Arc::clone(factory)))
            })
            .collect::<Result<_, _>>()?;
        Ok(ResolvedTreeComponentFactories { factories })
    }
}

fn tree_component_registry() -> &'static TreeComponentRegistry {
    static REGISTRY: OnceLock<TreeComponentRegistry> = OnceLock::new();
    REGISTRY.get_or_init(TreeComponentRegistry::default)
}

pub fn register_tree_component(
    name: &str,
    factory: impl TreeComponentFactoryFamily + 'static,
    replace: bool,
) -> Result<(), TreeComponentRegistryError> {
    tree_component_registry().register_tree_component(name, factory, replace)
}

pub fn resolve_tree_component_factories(
    component_types: &[ComponentType],
    overrides: &HashMap<ComponentType, String>,
) -> Result<ResolvedTreeComponentFactories, TreeComponentRegistryError> {
    tree_component_registry().resolve(component_types, overrides)
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
