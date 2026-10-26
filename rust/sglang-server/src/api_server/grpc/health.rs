//! Demand-shared standard gRPC health for the native frontend.
//!
//! Like the custom HealthCheck RPC, this measures scheduler output activity via
//! a synthetic one-token generation. It is not a capacity or drain signal.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::Stream;
use sglang_api_types::api::v1::sglang_service_server::SglangServiceServer;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tonic::server::NamedService;
use tonic::{Request, Response, Status};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_server::Health;
use tonic_health::pb::{HealthCheckRequest, HealthCheckResponse};

use super::service::GrpcService;
use crate::api_server::core::{CoreHandle, HealthStatus};

const REFRESH_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
struct Snapshot {
    status: ServingStatus,
    completed_at: Option<Instant>,
}

impl Snapshot {
    fn fresh(&self, interval: Duration) -> bool {
        self.completed_at.is_some_and(|at| at.elapsed() < interval)
    }
}

#[derive(Default)]
struct Demand {
    count: AtomicUsize,
    changed: Notify,
}

impl Demand {
    fn acquire(self: &Arc<Self>) -> DemandGuard {
        self.count.fetch_add(1, Ordering::AcqRel);
        self.changed.notify_one();
        DemandGuard(self.clone())
    }
}

struct DemandGuard(Arc<Demand>);

impl Drop for DemandGuard {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::AcqRel);
        self.0.changed.notify_one();
    }
}

/// The listener owns this task even when no standard-health clients exist.
/// Aborting drops the sole snapshot sender and any in-flight core probe.
pub(super) struct MonitorGuard(Option<JoinHandle<()>>);

impl MonitorGuard {
    pub(super) async fn stop(mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
            if tokio::time::timeout(Duration::from_secs(1), task)
                .await
                .is_err()
            {
                tracing::warn!("health monitor did not finish aborting within one second");
            }
        }
    }
}

impl Drop for MonitorGuard {
    fn drop(&mut self) {
        if let Some(task) = &self.0 {
            task.abort();
        }
    }
}

#[derive(Clone)]
pub(super) struct StandardHealth {
    snapshots: watch::Receiver<Snapshot>,
    demand: Arc<Demand>,
    interval: Duration,
}

impl StandardHealth {
    pub(super) fn new(core: CoreHandle, timeout: Duration) -> (Self, MonitorGuard) {
        Self::with_interval(core, timeout, REFRESH_INTERVAL)
    }

    fn with_interval(
        core: CoreHandle,
        timeout: Duration,
        interval: Duration,
    ) -> (Self, MonitorGuard) {
        let (sender, snapshots) = watch::channel(Snapshot {
            status: ServingStatus::NotServing,
            completed_at: None,
        });
        let demand = Arc::new(Demand::default());
        let task = tokio::spawn(monitor(core, timeout, interval, demand.clone(), sender));
        (
            Self {
                snapshots,
                demand,
                interval,
            },
            MonitorGuard(Some(task)),
        )
    }

    fn known(service: &str) -> bool {
        service.is_empty() || service == <SglangServiceServer<GrpcService> as NamedService>::NAME
    }
}

async fn monitor(
    core: CoreHandle,
    timeout: Duration,
    interval: Duration,
    demand: Arc<Demand>,
    sender: watch::Sender<Snapshot>,
) {
    loop {
        if demand.count.load(Ordering::Acquire) == 0 {
            demand.changed.notified().await;
            continue;
        }
        let completed_at = sender.borrow().completed_at;
        if let Some(at) = completed_at
            && at.elapsed() < interval
        {
            tokio::select! {
                _ = tokio::time::sleep_until(at + interval) => {}
                _ = demand.changed.notified() => continue,
            }
        }
        if demand.count.load(Ordering::Acquire) == 0 {
            continue;
        }
        // Core's internal timeout starts after admission. Bound the entire
        // future here, including a full intake channel, and skip zero-timeout
        // probes rather than depending on timeout's first-poll ordering.
        let probe = async {
            if timeout.is_zero() {
                return ServingStatus::NotServing;
            }
            match tokio::time::timeout(timeout, core.probe_health(timeout)).await {
                Ok(Ok(HealthStatus::Healthy)) => ServingStatus::Serving,
                _ => ServingStatus::NotServing,
            }
        };
        tokio::pin!(probe);
        loop {
            tokio::select! {
                _ = demand.changed.notified() => {
                    if demand.count.load(Ordering::Acquire) == 0 {
                        break;
                    }
                }
                status = &mut probe => {
                    sender.send_replace(Snapshot { status, completed_at: Some(Instant::now()) });
                    break;
                }
            }
        }
    }
}

fn response(status: ServingStatus) -> HealthCheckResponse {
    HealthCheckResponse {
        status: status as i32,
    }
}

type HealthStream = Pin<Box<dyn Stream<Item = Result<HealthCheckResponse, Status>> + Send>>;

#[tonic::async_trait]
impl Health for StandardHealth {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        if !Self::known(&request.get_ref().service) {
            return Err(Status::not_found("unknown health service"));
        }
        // Subscribe before demand and observe before changed(): a completion
        // between these operations cannot be missed by a stale Check waiter.
        let mut snapshots = self.snapshots.clone();
        if snapshots.has_changed().is_err() {
            return Ok(Response::new(response(ServingStatus::NotServing)));
        }
        let snapshot = *snapshots.borrow_and_update();
        if snapshot.fresh(self.interval) {
            return Ok(Response::new(response(snapshot.status)));
        }
        let _demand = self.demand.acquire();
        loop {
            if snapshots.changed().await.is_err() || snapshots.has_changed().is_err() {
                return Ok(Response::new(response(ServingStatus::NotServing)));
            }
            let snapshot = *snapshots.borrow_and_update();
            if snapshot.fresh(self.interval) {
                return Ok(Response::new(response(snapshot.status)));
            }
        }
    }

    type WatchStream = HealthStream;

    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let known = Self::known(&request.get_ref().service);
        let mut snapshots = self.snapshots.clone();
        // Capture ownership now, not on the stream's first poll. Dropping an
        // unpolled response must release its demand too.
        let demand = known.then(|| self.demand.acquire());
        let interval = self.interval;
        let stream = async_stream::stream! {
            let _demand = demand;
            let closed = snapshots.has_changed().is_err();
            let snapshot = *snapshots.borrow_and_update();
            let mut last = if !known {
                ServingStatus::ServiceUnknown
            } else if !closed && snapshot.fresh(interval) {
                snapshot.status
            } else {
                ServingStatus::NotServing
            };
            yield Ok(response(last));
            if closed { return; }
            loop {
                if snapshots.changed().await.is_err() || snapshots.has_changed().is_err() {
                    if known && last != ServingStatus::NotServing {
                        yield Ok(response(ServingStatus::NotServing));
                    }
                    break;
                }
                let status = snapshots.borrow_and_update().status;
                if known && status != last {
                    last = status;
                    yield Ok(response(status));
                }
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::{CoreConfig, CoreMetadata};
    use crate::message::config::{ModelConfig, ServerArgs};
    use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};
    use futures::StreamExt;

    fn core(
        ready: bool,
        capacity: Option<usize>,
    ) -> (
        CoreHandle,
        flume::Receiver<TmEvent>,
        flume::Receiver<AbortSource>,
    ) {
        core_with_activity(ready, capacity, Default::default())
    }

    fn core_with_activity(
        ready: bool,
        capacity: Option<usize>,
        activity: crate::tokenizer_manager::from_scheduler::ActivityCounter,
    ) -> (
        CoreHandle,
        flume::Receiver<TmEvent>,
        flume::Receiver<AbortSource>,
    ) {
        let (intake_tx, intake_rx) = capacity.map_or_else(flume::unbounded, flume::bounded);
        let (abort_tx, abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: activity,
                startup_ready: ready,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: CoreMetadata::from(&ServerArgs {
                    model_config: ModelConfig {
                        context_len: 4096,
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                metrics: None,
            },
        );
        (core, intake_rx, abort_rx)
    }

    fn request(service: &str) -> Request<HealthCheckRequest> {
        Request::new(HealthCheckRequest {
            service: service.into(),
        })
    }

    #[tokio::test]
    async fn unready_and_zero_timeout_do_not_submit() {
        for (ready, timeout) in [(false, Duration::from_secs(1)), (true, Duration::ZERO)] {
            let (core, intake, _) = core(ready, None);
            let (health, owner) = StandardHealth::new(core, timeout);
            assert_eq!(
                health.check(request("")).await.unwrap().into_inner().status,
                ServingStatus::NotServing as i32
            );
            assert!(intake.try_recv().is_err());
            owner.stop().await;
        }
    }

    #[tokio::test]
    async fn unknown_services_do_not_demand_and_watch_ends_with_owner() {
        let (core, intake, _) = core(true, None);
        let (health, owner) = StandardHealth::new(core, Duration::from_secs(1));
        assert_eq!(
            health.check(request("missing")).await.unwrap_err().code(),
            tonic::Code::NotFound
        );
        let mut stream = health.watch(request("missing")).await.unwrap().into_inner();
        assert_eq!(
            stream.next().await.unwrap().unwrap().status,
            ServingStatus::ServiceUnknown as i32
        );
        assert_eq!(health.demand.count.load(Ordering::Acquire), 0);
        assert!(intake.try_recv().is_err());
        owner.stop().await;
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn unpolled_watch_drop_releases_demand() {
        let (core, intake, _) = core(true, None);
        let (health, owner) = StandardHealth::new(core, Duration::from_secs(1));
        let stream = health.watch(request("")).await.unwrap().into_inner();
        assert_eq!(health.demand.count.load(Ordering::Acquire), 1);
        drop(stream);
        assert_eq!(health.demand.count.load(Ordering::Acquire), 0);
        tokio::task::yield_now().await;
        assert!(intake.try_recv().is_err());
        owner.stop().await;
    }

    #[tokio::test]
    async fn outer_timeout_withdraws_full_intake_submission() {
        let (core, intake, aborts) = core(true, Some(0));
        let (health, owner) = StandardHealth::new(core, Duration::from_millis(10));
        let result = tokio::time::timeout(Duration::from_secs(1), health.check(request("")))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.into_inner().status, ServingStatus::NotServing as i32);
        assert!(
            intake.try_recv().is_err(),
            "timed-out send cannot be admitted later"
        );
        assert!(
            aborts.try_recv().is_err(),
            "unaccepted request needs no scheduler abort"
        );
        owner.stop().await;
    }

    #[tokio::test]
    async fn last_watch_drop_aborts_accepted_probe() {
        let (core, intake, aborts) = core(true, None);
        let (health, owner) = StandardHealth::new(core, Duration::from_secs(10));
        let stream = health.watch(request("")).await.unwrap().into_inner();
        let TmEvent::Intake { request, admission } = intake.recv_async().await.unwrap() else {
            panic!("expected probe")
        };
        assert!(admission.try_accept());
        let rid = request.rid.clone();
        drop(stream);
        let abort = tokio::time::timeout(Duration::from_secs(1), aborts.recv_async())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(abort, AbortSource::Guard(aborted) if aborted == rid));
        assert!(admission.is_cancelled());
        assert_eq!(health.demand.count.load(Ordering::Acquire), 0);
        owner.stop().await;
    }

    #[tokio::test]
    async fn fresh_cache_does_not_demand_and_closed_monitor_overrides_it() {
        let (sender, snapshots) = watch::channel(Snapshot {
            status: ServingStatus::Serving,
            completed_at: Some(Instant::now()),
        });
        let health = StandardHealth {
            snapshots,
            demand: Arc::new(Demand::default()),
            interval: REFRESH_INTERVAL,
        };
        assert_eq!(
            health.check(request("")).await.unwrap().into_inner().status,
            ServingStatus::Serving as i32
        );
        assert_eq!(health.demand.count.load(Ordering::Acquire), 0);
        assert!(
            tokio::time::timeout(Duration::from_millis(1), health.demand.changed.notified())
                .await
                .is_err()
        );
        drop(sender);
        assert_eq!(
            health.check(request("")).await.unwrap().into_inner().status,
            ServingStatus::NotServing as i32
        );
        let mut stream = health.watch(request("")).await.unwrap().into_inner();
        assert_eq!(
            stream.next().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        assert!(stream.next().await.is_none());
    }

    struct DispatcherGuard {
        shutdown: flume::Sender<()>,
        thread: Option<std::thread::JoinHandle<()>>,
    }
    impl Drop for DispatcherGuard {
        fn drop(&mut self) {
            let _ = self.shutdown.send(());
            if let Some(thread) = self.thread.take() {
                thread.join().unwrap();
            }
        }
    }

    async fn accept_probe(intake: &flume::Receiver<TmEvent>) -> crate::message::ids::Rid {
        let event = tokio::time::timeout(Duration::from_secs(2), intake.recv_async())
            .await
            .unwrap()
            .unwrap();
        let TmEvent::Intake { request, admission } = event else {
            panic!("expected probe")
        };
        assert!(admission.try_accept());
        request.rid.clone()
    }

    #[tokio::test]
    async fn tonic_clients_share_scheduler_probe_and_observe_stall_and_recovery() {
        use crate::message::response::frame_health_ack;
        use crate::runtime::Runnable;
        use crate::tokenizer_manager::from_scheduler::Dispatcher;
        use crate::tokenizer_manager::wiring::Senders;
        use tonic_health::pb::{health_client::HealthClient, health_server::HealthServer};
        let activity: crate::tokenizer_manager::from_scheduler::ActivityCounter =
            Default::default();
        let (core, intake, aborts) = core_with_activity(true, None, activity.clone());
        let (frames, from_scheduler) = crate::tokenizer_manager::channel::from_scheduler(8);
        let (shutdown, shutdown_rx) = flume::bounded(1);
        let dispatcher = Dispatcher::new(
            from_scheduler,
            Senders {
                tok_manager_tx: flume::unbounded().0,
                abort_tx: flume::unbounded().0,
                tokenizer_tx: flume::unbounded().0,
                detokenizer_tx: vec![flume::unbounded().0],
            },
            activity,
            shutdown_rx,
        );
        let _dispatcher = DispatcherGuard {
            shutdown,
            thread: Some(std::thread::spawn(move || dispatcher.run())),
        };
        let (health, owner) =
            StandardHealth::with_interval(core, Duration::from_secs(2), Duration::from_secs(1));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(HealthServer::new(health.clone()))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        let client = HealthClient::new(
            tonic::transport::Endpoint::new(format!("http://{addr}"))
                .unwrap()
                .connect()
                .await
                .unwrap(),
        );
        let mut watcher = client
            .clone()
            .watch(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            watcher.message().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        let mut checks = tokio::task::JoinSet::new();
        for i in 0..8 {
            let mut client = client.clone();
            checks.spawn(async move {
                let name = if i % 2 == 0 {
                    ""
                } else {
                    <SglangServiceServer<GrpcService> as NamedService>::NAME
                };
                client
                    .check(HealthCheckRequest {
                        service: name.into(),
                    })
                    .await
                    .unwrap()
                    .into_inner()
                    .status
            });
        }
        tokio::time::timeout(Duration::from_secs(2), async {
            while health.demand.count.load(Ordering::Acquire) != 9 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let first = accept_probe(&intake).await;
        assert!(intake.try_recv().is_err());
        assert!(frames.push(frame_health_ack()));
        while let Some(result) = checks.join_next().await {
            assert_eq!(result.unwrap(), ServingStatus::Serving as i32);
        }
        assert_eq!(
            watcher.message().await.unwrap().unwrap().status,
            ServingStatus::Serving as i32
        );
        assert!(
            matches!(aborts.recv_async().await.unwrap(), AbortSource::Guard(rid) if rid == first)
        );
        let second = accept_probe(&intake).await;
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(3), watcher.message())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .status,
            ServingStatus::NotServing as i32
        );
        assert!(
            matches!(aborts.recv_async().await.unwrap(), AbortSource::Guard(rid) if rid == second)
        );
        let third = accept_probe(&intake).await;
        assert!(frames.push(frame_health_ack()));
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(3), watcher.message())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .status,
            ServingStatus::Serving as i32
        );
        assert!(
            matches!(aborts.recv_async().await.unwrap(), AbortSource::Guard(rid) if rid == third)
        );
        owner.stop().await;
        assert_eq!(
            watcher.message().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        assert!(watcher.message().await.unwrap().is_none());
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn actual_app_registers_both_services_and_both_shutdown_paths_abort_probe() {
        use sglang_api_types::api::v1::{self as api, sglang_service_client::SglangServiceClient};
        use tonic_health::pb::health_client::HealthClient;
        for abrupt in [false, true] {
            let (core, intake, aborts) = core(false, None);
            let args = ServerArgs::default();
            let service = GrpcService::new(
                core.clone(),
                &args,
                Arc::new(crate::api_server::core::HiCacheManager::new(
                    core.clone(),
                    &args,
                )),
            );
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let (shutdown, shutdown_rx) = flume::bounded(1);
            let app = tokio::spawn(super::super::app::serve(listener, service, shutdown_rx));
            let mut health = HealthClient::new(
                tonic::transport::Endpoint::new(format!("http://{addr}"))
                    .unwrap()
                    .connect()
                    .await
                    .unwrap(),
            );
            let mut custom = SglangServiceClient::connect(format!("http://{addr}"))
                .await
                .unwrap();
            assert_eq!(
                custom
                    .health_check(api::HealthCheckRequest {})
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::Unavailable
            );
            assert_eq!(
                health
                    .check(HealthCheckRequest {
                        service: "missing".into()
                    })
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::NotFound
            );
            let mut unknown = health
                .watch(HealthCheckRequest {
                    service: "missing".into(),
                })
                .await
                .unwrap()
                .into_inner();
            assert_eq!(
                unknown.message().await.unwrap().unwrap().status,
                ServingStatus::ServiceUnknown as i32
            );
            assert!(intake.try_recv().is_err());
            core.mark_ready();
            let _watch = health
                .watch(HealthCheckRequest {
                    service: String::new(),
                })
                .await
                .unwrap()
                .into_inner();
            let rid = accept_probe(&intake).await;
            if abrupt {
                app.abort();
            } else {
                shutdown.send(()).unwrap();
            }
            let joined = tokio::time::timeout(Duration::from_secs(2), app)
                .await
                .unwrap();
            if abrupt {
                assert!(joined.unwrap_err().is_cancelled());
            } else {
                joined.unwrap();
            }
            assert!(
                matches!(tokio::time::timeout(Duration::from_secs(1), aborts.recv_async()).await.unwrap().unwrap(), AbortSource::Guard(aborted) if aborted == rid)
            );
            // Tonic may close a connection or deliver EOF; neither may hang forever.
            assert!(
                tokio::time::timeout(Duration::from_secs(1), unknown.message())
                    .await
                    .is_ok()
            );
        }
    }

    #[tokio::test]
    async fn unseen_final_serving_snapshot_cannot_outlive_monitor() {
        let (sender, snapshots) = watch::channel(Snapshot {
            status: ServingStatus::NotServing,
            completed_at: None,
        });
        let health = StandardHealth {
            snapshots,
            demand: Arc::new(Demand::default()),
            interval: REFRESH_INTERVAL,
        };
        let check = health.check(request(""));
        tokio::pin!(check);
        assert!(futures::poll!(&mut check).is_pending());
        let mut stream = health.watch(request("")).await.unwrap().into_inner();
        assert_eq!(
            stream.next().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        let next = stream.next();
        tokio::pin!(next);
        assert!(futures::poll!(&mut next).is_pending());
        sender.send_replace(Snapshot {
            status: ServingStatus::Serving,
            completed_at: Some(Instant::now()),
        });
        drop(sender);
        assert_eq!(
            check.await.unwrap().into_inner().status,
            ServingStatus::NotServing as i32
        );
        assert!(
            next.await.is_none(),
            "closed monitor must not publish its unseen final SERVING"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn stale_checks_refresh_but_idle_cache_hits_do_not_probe() {
        let activity: crate::tokenizer_manager::from_scheduler::ActivityCounter =
            Default::default();
        let (core, intake, aborts) = core_with_activity(false, None, activity.clone());
        let (health, owner) = StandardHealth::new(core.clone(), Duration::from_secs(20));
        assert_eq!(
            health.check(request("")).await.unwrap().into_inner().status,
            ServingStatus::NotServing as i32
        );
        core.mark_ready();
        tokio::time::advance(Duration::from_secs(60)).await;
        assert!(intake.try_recv().is_err(), "no idle periodic probes");
        let checking = health.check(request(""));
        tokio::pin!(checking);
        assert!(futures::poll!(&mut checking).is_pending());
        let rid = accept_probe(&intake).await;
        activity.fetch_add(1, Ordering::Release);
        tokio::time::advance(Duration::from_millis(50)).await;
        assert_eq!(
            checking.await.unwrap().into_inner().status,
            ServingStatus::Serving as i32
        );
        assert!(
            matches!(aborts.recv_async().await.unwrap(), AbortSource::Guard(aborted) if aborted == rid)
        );
        for _ in 0..8 {
            assert_eq!(
                health.check(request("")).await.unwrap().into_inner().status,
                ServingStatus::Serving as i32
            );
        }
        assert_eq!(health.demand.count.load(Ordering::Acquire), 0);
        assert!(intake.try_recv().is_err());
        tokio::time::advance(Duration::from_secs(60)).await;
        assert!(intake.try_recv().is_err());
        let mut watching = health.watch(request("")).await.unwrap().into_inner();
        assert_eq!(
            watching.next().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32,
            "stale initial Watch cannot replay SERVING"
        );
        let _ = accept_probe(&intake).await;
        activity.fetch_add(1, Ordering::Release);
        tokio::time::advance(Duration::from_millis(50)).await;
        assert_eq!(
            watching.next().await.unwrap().unwrap().status,
            ServingStatus::Serving as i32
        );
        tokio::time::advance(Duration::from_millis(4999)).await;
        assert!(intake.try_recv().is_err());
        tokio::time::advance(Duration::from_millis(1)).await;
        let _ = accept_probe(&intake).await;
        // A refresh does not itself publish a NOT_SERVING transition.
        let next = watching.next();
        tokio::pin!(next);
        assert!(futures::poll!(&mut next).is_pending());
        tokio::time::advance(Duration::from_secs(20)).await;
        assert_eq!(
            next.await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        owner.stop().await;
    }

    #[tokio::test]
    async fn disconnected_core_is_not_serving() {
        let (core, intake, _) = core(true, None);
        drop(intake);
        let (health, owner) = StandardHealth::new(core, Duration::from_secs(1));
        assert_eq!(
            health.check(request("")).await.unwrap().into_inner().status,
            ServingStatus::NotServing as i32
        );
        owner.stop().await;
    }

    #[tokio::test]
    async fn last_transport_watch_disconnect_cancels_probe_and_periodic_work() {
        use tonic_health::pb::{health_client::HealthClient, health_server::HealthServer};
        let (core, intake, aborts) = core(true, None);
        let (health, owner) = StandardHealth::with_interval(
            core,
            Duration::from_secs(20),
            Duration::from_millis(100),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(HealthServer::new(health.clone()))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        let channel = tonic::transport::Endpoint::new(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut client = HealthClient::new(channel);
        let mut watcher = client
            .watch(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            watcher.message().await.unwrap().unwrap().status,
            ServingStatus::NotServing as i32
        );
        let rid = accept_probe(&intake).await;
        drop(watcher);
        assert!(
            matches!(tokio::time::timeout(Duration::from_secs(2), aborts.recv_async()).await.unwrap().unwrap(), AbortSource::Guard(aborted) if aborted == rid)
        );
        assert_eq!(health.demand.count.load(Ordering::Acquire), 0);
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(intake.try_recv().is_err());
        owner.stop().await;
        server.abort();
        let _ = server.await;
    }
}
