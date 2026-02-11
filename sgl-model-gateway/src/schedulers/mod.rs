// router-level scheduling logic for routing between different routers.
pub mod proportion;
pub mod factory;
pub use factory::SchedulerFactory;
pub use proportion::ProportionScheduler;
use crate::routers::router_manager::RouterId;
use std::fmt::Debug;
use async_trait::async_trait;


#[async_trait]
pub trait SchedulerPolicy: Send + Sync + Debug {
    async fn select_router(
        &self,
        candidate_routers: &[RouterId],
        info: &SelectRouterInfo<'_>,
    ) -> Option<RouterId>;

    fn name(&self) -> &'static str;

    fn needs_request_text(&self) -> bool {
        false // Default: most policies don't need request text
    }

}

#[derive(Debug, Clone, Default)]
pub struct SelectRouterInfo<'a> {
    pub request_text: Option<&'a str>,
    pub tokens: Option<&'a [u32]>,
    pub model_id: Option<&'a str>,
}
