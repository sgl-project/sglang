mod bootstrap;
mod failure;
mod lifecycle;
mod mock;
mod native;
mod state;

pub use bootstrap::{
    BootstrapPort, BootstrapRegistration, ConnectionLifecycle, PairConnection, RuntimeError,
    RuntimeIdentity, bootstrap_decode, bootstrap_prefill,
};
pub use failure::{FailureClass, FailureScope};
pub use lifecycle::{
    FatalPublish, FatalRecord, FatalSource, FirstFatal, FirstFatalSnapshot, RuntimeShutdownOutcome,
    ShutdownMode, ShutdownPhase, ShutdownTracker, WorkerLifecycle,
};
pub use mock::CpuMockBootstrapPort;
pub use native::{NativeBootstrapPort, NativeRegionDescriptor};
pub use state::{
    HeartbeatAction, HeartbeatSnapshot, HeartbeatTracker, LifecycleCoordinator, PairReadiness,
    PairState, RuntimeLifecycle, RuntimeSnapshot,
};
