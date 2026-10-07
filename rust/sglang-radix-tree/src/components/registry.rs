//! Native component factories selected by per-cache implementation keys.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};

use super::{
    ComponentInitError, ComponentType, FullComponent, MambaComponent, SwaComponent, TreeComponent,
};
use crate::node::ChildKeyType;
use crate::unified_tree_core::CacheInitParams;

pub type TreeComponentInstance<K> = Arc<dyn TreeComponent<K> + Send + Sync>;

/// Construction arguments for one native component slot.
pub struct TreeComponentArgument<'a> {
    pub component_type: ComponentType,
    pub params: &'a CacheInitParams,
    pub is_bigram: bool,
}

/// A native factory supporting both tree key representations.
pub trait TreeComponentFactory: Send + Sync {
    fn create_plain(
        &self,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<Vec<i64>>, ComponentInitError>;

    fn create_bigram(
        &self,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<Vec<(i64, i64)>>, ComponentInitError>;
}

impl<F, C> TreeComponentFactory for F
where
    F: Fn(&TreeComponentArgument<'_>) -> Result<C, ComponentInitError> + Send + Sync,
    C: TreeComponent<Vec<i64>> + TreeComponent<Vec<(i64, i64)>> + Send + Sync + 'static,
{
    fn create_plain(
        &self,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<Vec<i64>>, ComponentInitError> {
        Ok(Arc::new(self(argument)?))
    }

    fn create_bigram(
        &self,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<Vec<(i64, i64)>>, ComponentInitError> {
        Ok(Arc::new(self(argument)?))
    }
}

pub trait TreeComponentKey: ChildKeyType {
    fn create(
        factory: &dyn TreeComponentFactory,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<Self>, ComponentInitError>;
}

impl TreeComponentKey for Vec<i64> {
    fn create(
        factory: &dyn TreeComponentFactory,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<Self>, ComponentInitError> {
        factory.create_plain(argument)
    }
}

impl TreeComponentKey for Vec<(i64, i64)> {
    fn create(
        factory: &dyn TreeComponentFactory,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<Self>, ComponentInitError> {
        factory.create_bigram(argument)
    }
}

type FactoryEntry = (ComponentType, Arc<dyn TreeComponentFactory>);

pub struct TreeComponentRegistry {
    factories: RwLock<HashMap<String, FactoryEntry>>,
}

pub struct TreeComponentFactorySnapshot {
    entries: Vec<FactoryEntry>,
}

impl TreeComponentFactorySnapshot {
    pub fn component_types(&self) -> Vec<ComponentType> {
        self.entries
            .iter()
            .map(|(component_type, _)| *component_type)
            .collect()
    }

    pub fn validate_configuration(
        &self,
        params: &CacheInitParams,
    ) -> Result<(), ComponentInitError> {
        for (component_type, _) in &self.entries {
            match component_type {
                ComponentType::Swa if params.swa_sliding_window_size == Some(0) => {
                    return Err(ComponentInitError::InvalidConfiguration(
                        "swa_sliding_window_size must be positive",
                    ));
                }
                ComponentType::Mamba if params.mamba_cache_chunk_size == Some(0) => {
                    return Err(ComponentInitError::InvalidConfiguration(
                        "mamba_cache_chunk_size must be positive",
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Invoke a fixed factory selection without holding the registry lock.
    pub fn create_components<K: TreeComponentKey>(
        self,
        params: &CacheInitParams,
    ) -> Result<Vec<TreeComponentInstance<K>>, ComponentInitError> {
        self.entries
            .into_iter()
            .map(|(component_type, factory)| {
                create_from_factory::<K>(
                    &*factory,
                    &TreeComponentArgument {
                        component_type,
                        params,
                        is_bigram: K::IS_BIGRAM,
                    },
                )
            })
            .collect()
    }
}

impl Default for TreeComponentRegistry {
    fn default() -> Self {
        let registry = Self {
            factories: RwLock::new(HashMap::new()),
        };
        registry
            .register_tree_component(
                "full",
                ComponentType::Full,
                |_: &TreeComponentArgument<'_>| Ok(FullComponent),
                false,
            )
            .unwrap();
        registry
            .register_tree_component(
                "swa",
                ComponentType::Swa,
                |argument: &TreeComponentArgument<'_>| {
                    if argument.params.swa_sliding_window_size.is_none() {
                        return Err(ComponentInitError::InvalidConfiguration(
                            "the Swa component requires swa_sliding_window_size",
                        ));
                    }
                    Ok(SwaComponent::new(argument.params))
                },
                false,
            )
            .unwrap();
        registry
            .register_tree_component(
                "mamba",
                ComponentType::Mamba,
                |argument: &TreeComponentArgument<'_>| {
                    if argument.params.mamba_cache_chunk_size.is_none() {
                        return Err(ComponentInitError::InvalidConfiguration(
                            "the Mamba component requires mamba_cache_chunk_size",
                        ));
                    }
                    Ok(MambaComponent::new(argument.params))
                },
                false,
            )
            .unwrap();
        registry
    }
}

impl TreeComponentRegistry {
    /// Register or explicitly replace a factory for subsequently constructed caches.
    pub fn register_tree_component(
        &self,
        name: &str,
        component_type: ComponentType,
        factory: impl TreeComponentFactory + 'static,
        replace: bool,
    ) -> Result<(), ComponentInitError> {
        if name.trim().is_empty() {
            return Err(ComponentInitError::EmptyFactoryKey);
        }
        let factory: Arc<dyn TreeComponentFactory> = Arc::new(factory);
        let previous = {
            let mut factories = self.factories.write().unwrap();
            if let Some((registered_type, _)) = factories.get(name) {
                if !replace {
                    return Err(ComponentInitError::DuplicateFactoryKey(name.to_owned()));
                }
                if *registered_type != component_type {
                    return Err(ComponentInitError::ComponentTypeMismatch {
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

    pub fn registered_tree_components(&self) -> HashMap<String, ComponentType> {
        self.factories
            .read()
            .unwrap()
            .iter()
            .map(|(name, (component_type, _))| (name.clone(), *component_type))
            .collect()
    }

    pub fn snapshot(
        &self,
        factory_keys: &[String],
    ) -> Result<TreeComponentFactorySnapshot, ComponentInitError> {
        let mut configured = HashSet::new();
        let factories = self.factories.read().unwrap();
        let mut entries = Vec::with_capacity(factory_keys.len());
        for key in factory_keys {
            if key.trim().is_empty() {
                return Err(ComponentInitError::EmptyFactoryKey);
            }
            let (component_type, factory) = factories
                .get(key)
                .ok_or_else(|| ComponentInitError::UnknownFactoryKey(key.to_owned()))?;
            if !configured.insert(*component_type) {
                return Err(ComponentInitError::DuplicateComponent(*component_type));
            }
            entries.push((*component_type, Arc::clone(factory)));
        }
        Ok(TreeComponentFactorySnapshot { entries })
    }

    pub fn create_tree_component<K: TreeComponentKey>(
        &self,
        name: &str,
        argument: &TreeComponentArgument<'_>,
    ) -> Result<TreeComponentInstance<K>, ComponentInitError> {
        let snapshot = self.snapshot(&[name.to_owned()])?;
        let (actual, factory) = &snapshot.entries[0];
        if *actual != argument.component_type {
            return Err(ComponentInitError::ComponentTypeMismatch {
                expected: argument.component_type,
                actual: *actual,
            });
        }
        create_from_factory::<K>(&**factory, argument)
    }
}

fn create_from_factory<K: TreeComponentKey>(
    factory: &dyn TreeComponentFactory,
    argument: &TreeComponentArgument<'_>,
) -> Result<TreeComponentInstance<K>, ComponentInitError> {
    if argument.is_bigram != K::IS_BIGRAM {
        return Err(ComponentInitError::KeyModeMismatch);
    }
    if argument.params.page_size == 0 {
        return Err(ComponentInitError::InvalidConfiguration(
            "page_size must be at least 1",
        ));
    }
    let component = K::create(factory, argument)?;
    let actual = component.component_type();
    if actual != argument.component_type {
        return Err(ComponentInitError::ComponentTypeMismatch {
            expected: argument.component_type,
            actual,
        });
    }
    Ok(component)
}

pub(crate) fn tree_component_registry() -> &'static TreeComponentRegistry {
    static REGISTRY: OnceLock<TreeComponentRegistry> = OnceLock::new();
    REGISTRY.get_or_init(TreeComponentRegistry::default)
}

pub fn register_tree_component(
    name: &str,
    component_type: ComponentType,
    factory: impl TreeComponentFactory + 'static,
    replace: bool,
) -> Result<(), ComponentInitError> {
    tree_component_registry().register_tree_component(name, component_type, factory, replace)
}

pub fn create_tree_component<K: TreeComponentKey>(
    name: &str,
    argument: &TreeComponentArgument<'_>,
) -> Result<TreeComponentInstance<K>, ComponentInitError> {
    tree_component_registry().create_tree_component(name, argument)
}

pub fn registered_tree_components() -> HashMap<String, ComponentType> {
    tree_component_registry().registered_tree_components()
}

pub fn default_factory_key(component_type: ComponentType) -> &'static str {
    match component_type {
        ComponentType::Full => "full",
        ComponentType::Swa => "swa",
        ComponentType::Mamba => "mamba",
    }
}

#[cfg(test)]
#[path = "../tests/components/registry.rs"]
mod tests;
