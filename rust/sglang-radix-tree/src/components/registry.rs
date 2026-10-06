//! Named native component factories, independent of Python component registration.

use std::collections::HashMap;
use std::sync::Arc;

use super::{ComponentType, FullComponent, MambaComponent, SwaComponent, TreeComponent};
use crate::node::ChildKeyType;
use crate::unified_tree_core::CacheInitParams;

type ComponentFactory<K> =
    dyn Fn(&CacheInitParams) -> Arc<dyn TreeComponent<K> + Send + Sync> + Send + Sync;

#[derive(Debug, thiserror::Error)]
pub enum ComponentRegistryError {
    #[error("component name must be non-empty")]
    EmptyName,
    #[error("component name {0:?} is already registered")]
    DuplicateName(String),
    #[error("unknown native component name {0:?}")]
    UnknownName(String),
    #[error("component override {0:?} is not enabled")]
    InactiveComponent(ComponentType),
    #[error("duplicate override for component {0:?}")]
    DuplicateOverride(ComponentType),
    #[error("native component {name:?} has type {actual:?}, expected {expected:?}")]
    ComponentTypeMismatch {
        name: String,
        expected: ComponentType,
        actual: ComponentType,
    },
}

pub struct ComponentRegistry<K: ChildKeyType> {
    factories: HashMap<String, (ComponentType, Box<ComponentFactory<K>>)>,
}

impl<K: ChildKeyType> Default for ComponentRegistry<K> {
    fn default() -> Self {
        let mut registry = Self {
            factories: HashMap::new(),
        };
        registry
            .register("full", ComponentType::Full, |_| Arc::new(FullComponent))
            .unwrap();
        registry
            .register("swa", ComponentType::Swa, |params| {
                Arc::new(SwaComponent::new(params))
            })
            .unwrap();
        registry
            .register("mamba", ComponentType::Mamba, |params| {
                Arc::new(MambaComponent::new(params))
            })
            .unwrap();
        registry
    }
}

impl<K: ChildKeyType> ComponentRegistry<K> {
    pub fn register(
        &mut self,
        name: &str,
        component_type: ComponentType,
        factory: impl Fn(&CacheInitParams) -> Arc<dyn TreeComponent<K> + Send + Sync>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), ComponentRegistryError> {
        if name.trim().is_empty() {
            return Err(ComponentRegistryError::EmptyName);
        }
        if self.factories.contains_key(name) {
            return Err(ComponentRegistryError::DuplicateName(name.to_owned()));
        }
        self.factories
            .insert(name.to_owned(), (component_type, Box::new(factory)));
        Ok(())
    }

    pub fn create(
        &self,
        name: &str,
        component_type: ComponentType,
        params: &CacheInitParams,
    ) -> Result<Arc<dyn TreeComponent<K> + Send + Sync>, ComponentRegistryError> {
        let (registered_type, factory) = self
            .factories
            .get(name)
            .ok_or_else(|| ComponentRegistryError::UnknownName(name.to_owned()))?;
        self.check_type(name, component_type, *registered_type)?;
        let component = factory(params);
        self.check_type(name, component_type, component.component_type())?;
        Ok(component)
    }

    fn check_type(
        &self,
        name: &str,
        expected: ComponentType,
        actual: ComponentType,
    ) -> Result<(), ComponentRegistryError> {
        if expected != actual {
            return Err(ComponentRegistryError::ComponentTypeMismatch {
                name: name.to_owned(),
                expected,
                actual,
            });
        }
        Ok(())
    }

    pub fn default_name(component_type: ComponentType) -> &'static str {
        match component_type {
            ComponentType::Full => "full",
            ComponentType::Swa => "swa",
            ComponentType::Mamba => "mamba",
        }
    }

    pub fn registered_names(&self) -> HashMap<String, ComponentType> {
        self.factories
            .iter()
            .map(|(name, (ct, _))| (name.clone(), *ct))
            .collect()
    }
}
