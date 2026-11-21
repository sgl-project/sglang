//! An administrative shutdown enters exactly the launcher's SIGTERM drain.
//! The existing listener-close -> parent notification -> scheduler teardown
//! chain owns the rest; this operation never claims that teardown completed.

use super::CoreHandle;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ShutdownError {
    MissingIdentity,
    DifferentInstance,
    NoOwner,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::handlers::test_utils::Harness;

    #[tokio::test]
    async fn acceptance_keeps_active_work_alive_until_the_existing_drain_finishes() {
        let core = Harness::unbounded(4).handle;
        core.install_shutdown_owner(false);
        let in_flight = core.inner.drain.track();
        assert!(core.request_shutdown(core.owning_instance_id()).unwrap());
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(5), core.drain_ended())
                .await
                .is_err()
        );
        // Dropping the RPC/request object cannot roll back acceptance.
        assert!(!core.request_shutdown(core.owning_instance_id()).unwrap());
        drop(in_flight);
        assert_eq!(
            core.drain_ended().await,
            crate::api_server::core::DrainEnd::Drained
        );
    }

    #[test]
    fn rejected_identity_or_missing_owner_does_not_start_drain() {
        let core = Harness::unbounded(4).handle;
        assert_eq!(
            core.request_shutdown(""),
            Err(ShutdownError::MissingIdentity)
        );
        assert_eq!(
            core.request_shutdown("old-boot"),
            Err(ShutdownError::DifferentInstance)
        );
        assert_eq!(
            core.request_shutdown(core.owning_instance_id()),
            Err(ShutdownError::NoOwner)
        );
        assert!(!core.is_draining());
    }

    #[test]
    fn concurrent_acceptance_is_once_and_does_not_escalate() {
        let core = Harness::unbounded(4).handle;
        core.install_shutdown_owner(false);
        let jobs = (0..16)
            .map(|_| {
                let core = core.clone();
                std::thread::spawn(move || {
                    core.request_shutdown(core.owning_instance_id()).unwrap()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            jobs.into_iter()
                .map(|job| job.join().unwrap())
                .filter(|first| *first)
                .count(),
            1
        );
        assert!(core.is_draining());
        assert!(!core.is_drain_forced());
    }
}

impl CoreHandle {
    pub(crate) fn install_shutdown_owner(&self, force: bool) {
        let _ = self.inner.shutdown_force.set(force);
    }

    pub(crate) fn owning_instance_id(&self) -> &str {
        &self.inner.instance_id
    }

    /// Synchronous acceptance, independent of the RPC reader's lifetime.
    /// The atomic drain transition makes concurrent retries idempotent.
    pub(crate) fn request_shutdown(&self, expected: &str) -> Result<bool, ShutdownError> {
        if expected.is_empty() {
            return Err(ShutdownError::MissingIdentity);
        }
        if expected != self.owning_instance_id() {
            return Err(ShutdownError::DifferentInstance);
        }
        let force = *self
            .inner
            .shutdown_force
            .get()
            .ok_or(ShutdownError::NoOwner)?;
        Ok(self.inner.drain.begin(force))
    }
}
