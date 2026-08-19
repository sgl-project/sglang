//! The [`Runnable`] stage trait — the one contract every pipeline stage
//! (CPU-bound worker or TM router) implements to be spawned by the runtime.

/// A pipeline stage that owns its channel handles + config and runs a blocking
/// loop until its inbox closes. Lets the runtime spawn stages uniformly via
/// `threads::spawn_stage` / `threads::spawn_pool` instead of free `run_*` functions with
/// positional handles. Implemented by every CPU-bound worker and TM router.
pub trait Runnable: Send + 'static {
    fn run(self);
}
