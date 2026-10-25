/// Generated client implementations.
pub mod sglang_service_client {
    #![allow(
        unused_variables,
        dead_code,
        missing_docs,
        clippy::wildcard_imports,
        clippy::let_unit_value
    )]
    use tonic::codegen::http::Uri;
    use tonic::codegen::*;
    #[derive(Debug, Clone)]
    pub struct SglangServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl SglangServiceClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> SglangServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::Body>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + std::marker::Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + std::marker::Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> SglangServiceClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                    http::Request<tonic::body::Body>,
                    Response = http::Response<
                        <T as tonic::client::GrpcService<tonic::body::Body>>::ResponseBody,
                    >,
                >,
            <T as tonic::codegen::Service<http::Request<tonic::body::Body>>>::Error:
                Into<StdError> + std::marker::Send + std::marker::Sync,
        {
            SglangServiceClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_decoding_message_size(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.inner = self.inner.max_encoding_message_size(limit);
            self
        }
        /// `/generate`: always server-streaming (a unary request is a one-frame
        /// stream); per-batch-item failures ride the stream as GenerateStreamItem
        /// errors, pre-submit failures end the RPC with a gRPC status.
        pub async fn generate(
            &mut self,
            request: impl tonic::IntoRequest<super::GenerateRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::GenerateStreamItem>>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/Generate");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.api.v1.SglangService", "Generate"));
            self.inner.server_streaming(req, path, codec).await
        }
        /// `/health_generate`: fires a 1-token probe and watches the response
        /// heartbeat. UNAVAILABLE on a stalled pipeline.
        pub async fn health_check(
            &mut self,
            request: impl tonic::IntoRequest<super::HealthCheckRequest>,
        ) -> std::result::Result<tonic::Response<super::HealthCheckResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/HealthCheck");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "HealthCheck",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/get_model_info`: static model metadata, no scheduler round-trip.
        pub async fn get_model_info(
            &mut self,
            request: impl tonic::IntoRequest<super::GetModelInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetModelInfoResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/GetModelInfo");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "GetModelInfo",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/server_info`: one control round-trip, shaped through the runtime-metric
        /// allowlist (never the raw server-args dump, which embeds keys).
        pub async fn get_server_info(
            &mut self,
            request: impl tonic::IntoRequest<super::GetServerInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetServerInfoResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/GetServerInfo");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "GetServerInfo",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/hicache/storage-backend`: configuration snapshot and runtime attach/detach.
        pub async fn get_hi_cache_storage(
            &mut self,
            request: impl tonic::IntoRequest<super::GetHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStatus>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/GetHiCacheStorage",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "GetHiCacheStorage",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn attach_hi_cache_storage(
            &mut self,
            request: impl tonic::IntoRequest<super::AttachHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStorageResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/AttachHiCacheStorage",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "AttachHiCacheStorage",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn detach_hi_cache_storage(
            &mut self,
            request: impl tonic::IntoRequest<super::DetachHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStorageResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/DetachHiCacheStorage",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "DetachHiCacheStorage",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/hicache/storage-backend/clear`: clear the storage backend's cached entries.
        pub async fn clear_hi_cache_storage(
            &mut self,
            request: impl tonic::IntoRequest<super::ClearHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStorageResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/ClearHiCacheStorage",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "ClearHiCacheStorage",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// The same static model cards as HTTP `/v1/models`.
        pub async fn list_models(
            &mut self,
            request: impl tonic::IntoRequest<super::ListModelsRequest>,
        ) -> std::result::Result<tonic::Response<super::ListModelsResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/ListModels");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.api.v1.SglangService", "ListModels"));
            self.inner.unary(req, path, codec).await
        }
        /// Decode token IDs locally, without running model inference.
        pub async fn detokenize(
            &mut self,
            request: impl tonic::IntoRequest<super::DetokenizeRequest>,
        ) -> std::result::Result<tonic::Response<super::DetokenizeResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/Detokenize");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.api.v1.SglangService", "Detokenize"));
            self.inner.unary(req, path, codec).await
        }
        /// Cancel by the native rid used by HTTP `/generate` or gRPC Generate, or
        /// cancel all active frontend generations with abort_all. HTTP chat/completion
        /// response IDs are not supported as rid selectors.
        /// Duplicate native IDs cancel all matching generations. A later reuse of the
        /// ID is unaffected. Internal control/detokenization requests are not targeted.
        pub async fn abort(
            &mut self,
            request: impl tonic::IntoRequest<super::AbortRequest>,
        ) -> std::result::Result<tonic::Response<super::AbortResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/Abort");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.api.v1.SglangService", "Abort"));
            self.inner.unary(req, path, codec).await
        }
        /// `/set_internal_state`: override numeric scheduler knobs on every DP rank;
        /// one `updated` flag per rank.
        pub async fn set_internal_state(
            &mut self,
            request: impl tonic::IntoRequest<super::SetInternalStateRequest>,
        ) -> std::result::Result<tonic::Response<super::SetInternalStateResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/SetInternalState",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "SetInternalState",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/start_profile` and `/stop_profile`: a profiler capture on every DP rank.
        pub async fn start_profile(
            &mut self,
            request: impl tonic::IntoRequest<super::StartProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::ProfileResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/StartProfile");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "StartProfile",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn stop_profile(
            &mut self,
            request: impl tonic::IntoRequest<super::StopProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::ProfileResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/StopProfile");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "StopProfile",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/{start,stop,dump}_expert_distribution_record`.
        pub async fn start_expert_distribution_record(
            &mut self,
            request: impl tonic::IntoRequest<super::ExpertDistributionRecordRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExpertDistributionRecordResponse>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/StartExpertDistributionRecord",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "StartExpertDistributionRecord",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn stop_expert_distribution_record(
            &mut self,
            request: impl tonic::IntoRequest<super::ExpertDistributionRecordRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExpertDistributionRecordResponse>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/StopExpertDistributionRecord",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "StopExpertDistributionRecord",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn dump_expert_distribution_record(
            &mut self,
            request: impl tonic::IntoRequest<super::ExpertDistributionRecordRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExpertDistributionRecordResponse>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/DumpExpertDistributionRecord",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "DumpExpertDistributionRecord",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/slow_down`: a per-forward sleep on every DP rank, for testing.
        pub async fn slow_down(
            &mut self,
            request: impl tonic::IntoRequest<super::SlowDownRequest>,
        ) -> std::result::Result<tonic::Response<super::SlowDownResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/SlowDown");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.api.v1.SglangService", "SlowDown"));
            self.inner.unary(req, path, codec).await
        }
        /// `/release_memory_occupation`, `/resume_memory_occupation`.
        pub async fn release_memory_occupation(
            &mut self,
            request: impl tonic::IntoRequest<super::ReleaseMemoryOccupationRequest>,
        ) -> std::result::Result<tonic::Response<super::MemoryOccupationResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/ReleaseMemoryOccupation",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "ReleaseMemoryOccupation",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn resume_memory_occupation(
            &mut self,
            request: impl tonic::IntoRequest<super::ResumeMemoryOccupationRequest>,
        ) -> std::result::Result<tonic::Response<super::MemoryOccupationResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/ResumeMemoryOccupation",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "ResumeMemoryOccupation",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/get_weights_by_name`: one parameter per DP rank.
        pub async fn get_weights_by_name(
            &mut self,
            request: impl tonic::IntoRequest<super::GetWeightsByNameRequest>,
        ) -> std::result::Result<tonic::Response<super::GetWeightsByNameResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/GetWeightsByName",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "GetWeightsByName",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// Weight-update process groups and remote-instance sends. Single-DP-rank
        /// deployments only, as in Python.
        pub async fn init_weights_update_group(
            &mut self,
            request: impl tonic::IntoRequest<super::InitWeightsUpdateGroupRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/InitWeightsUpdateGroup",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "InitWeightsUpdateGroup",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn destroy_weights_update_group(
            &mut self,
            request: impl tonic::IntoRequest<super::DestroyWeightsUpdateGroupRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/DestroyWeightsUpdateGroup",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "DestroyWeightsUpdateGroup",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn init_weights_send_group_for_remote_instance(
            &mut self,
            request: impl tonic::IntoRequest<super::InitWeightsSendGroupForRemoteInstanceRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/InitWeightsSendGroupForRemoteInstance",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "InitWeightsSendGroupForRemoteInstance",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn send_weights_to_remote_instance(
            &mut self,
            request: impl tonic::IntoRequest<super::SendWeightsToRemoteInstanceRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/SendWeightsToRemoteInstance",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "SendWeightsToRemoteInstance",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/parse_function_call`, `/separate_reasoning`: frontend-only parsing.
        pub async fn parse_function_call(
            &mut self,
            request: impl tonic::IntoRequest<super::ParseFunctionCallRequest>,
        ) -> std::result::Result<tonic::Response<super::ParseFunctionCallResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/ParseFunctionCall",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "ParseFunctionCall",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn separate_reasoning(
            &mut self,
            request: impl tonic::IntoRequest<super::SeparateReasoningRequest>,
        ) -> std::result::Result<tonic::Response<super::SeparateReasoningResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/SeparateReasoning",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "SeparateReasoning",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/add_external_corpus`, `/remove_external_corpus`, `/list_external_corpora`.
        pub async fn add_external_corpus(
            &mut self,
            request: impl tonic::IntoRequest<super::AddExternalCorpusRequest>,
        ) -> std::result::Result<tonic::Response<super::AddExternalCorpusResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/AddExternalCorpus",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "AddExternalCorpus",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn remove_external_corpus(
            &mut self,
            request: impl tonic::IntoRequest<super::RemoveExternalCorpusRequest>,
        ) -> std::result::Result<tonic::Response<super::RemoveExternalCorpusResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/RemoveExternalCorpus",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "RemoveExternalCorpus",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn list_external_corpora(
            &mut self,
            request: impl tonic::IntoRequest<super::ListExternalCorporaRequest>,
        ) -> std::result::Result<tonic::Response<super::ListExternalCorporaResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/ListExternalCorpora",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "ListExternalCorpora",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// `/freeze_gc`, `/pause_generation`, `/continue_generation`,
        /// `/pd_role_switch`, `/update_weight_version`.
        pub async fn freeze_gc(
            &mut self,
            request: impl tonic::IntoRequest<super::FreezeGcRequest>,
        ) -> std::result::Result<tonic::Response<super::FreezeGcResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/FreezeGc");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.api.v1.SglangService", "FreezeGc"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn pause_generation(
            &mut self,
            request: impl tonic::IntoRequest<super::PauseGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::GenerationPauseResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/PauseGeneration",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "PauseGeneration",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn continue_generation(
            &mut self,
            request: impl tonic::IntoRequest<super::ContinueGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::GenerationPauseResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/ContinueGeneration",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "ContinueGeneration",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn pd_role_switch(
            &mut self,
            request: impl tonic::IntoRequest<super::PdRoleSwitchRequest>,
        ) -> std::result::Result<tonic::Response<super::PdRoleSwitchResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.api.v1.SglangService/PdRoleSwitch");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "PdRoleSwitch",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn update_weight_version(
            &mut self,
            request: impl tonic::IntoRequest<super::UpdateWeightVersionRequest>,
        ) -> std::result::Result<tonic::Response<super::UpdateWeightVersionResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.api.v1.SglangService/UpdateWeightVersion",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.api.v1.SglangService",
                "UpdateWeightVersion",
            ));
            self.inner.unary(req, path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod sglang_service_server {
    #![allow(
        unused_variables,
        dead_code,
        missing_docs,
        clippy::wildcard_imports,
        clippy::let_unit_value
    )]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with SglangServiceServer.
    #[async_trait]
    pub trait SglangService: std::marker::Send + std::marker::Sync + 'static {
        /// Server streaming response type for the Generate method.
        type GenerateStream: tonic::codegen::tokio_stream::Stream<
                Item = std::result::Result<super::GenerateStreamItem, tonic::Status>,
            > + std::marker::Send
            + 'static;
        /// `/generate`: always server-streaming (a unary request is a one-frame
        /// stream); per-batch-item failures ride the stream as GenerateStreamItem
        /// errors, pre-submit failures end the RPC with a gRPC status.
        async fn generate(
            &self,
            request: tonic::Request<super::GenerateRequest>,
        ) -> std::result::Result<tonic::Response<Self::GenerateStream>, tonic::Status>;
        /// `/health_generate`: fires a 1-token probe and watches the response
        /// heartbeat. UNAVAILABLE on a stalled pipeline.
        async fn health_check(
            &self,
            request: tonic::Request<super::HealthCheckRequest>,
        ) -> std::result::Result<tonic::Response<super::HealthCheckResponse>, tonic::Status>;
        /// `/get_model_info`: static model metadata, no scheduler round-trip.
        async fn get_model_info(
            &self,
            request: tonic::Request<super::GetModelInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetModelInfoResponse>, tonic::Status>;
        /// `/server_info`: one control round-trip, shaped through the runtime-metric
        /// allowlist (never the raw server-args dump, which embeds keys).
        async fn get_server_info(
            &self,
            request: tonic::Request<super::GetServerInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetServerInfoResponse>, tonic::Status>;
        /// `/hicache/storage-backend`: configuration snapshot and runtime attach/detach.
        async fn get_hi_cache_storage(
            &self,
            request: tonic::Request<super::GetHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStatus>, tonic::Status>;
        async fn attach_hi_cache_storage(
            &self,
            request: tonic::Request<super::AttachHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStorageResponse>, tonic::Status>;
        async fn detach_hi_cache_storage(
            &self,
            request: tonic::Request<super::DetachHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStorageResponse>, tonic::Status>;
        /// `/hicache/storage-backend/clear`: clear the storage backend's cached entries.
        async fn clear_hi_cache_storage(
            &self,
            request: tonic::Request<super::ClearHiCacheStorageRequest>,
        ) -> std::result::Result<tonic::Response<super::HiCacheStorageResponse>, tonic::Status>;
        /// The same static model cards as HTTP `/v1/models`.
        async fn list_models(
            &self,
            request: tonic::Request<super::ListModelsRequest>,
        ) -> std::result::Result<tonic::Response<super::ListModelsResponse>, tonic::Status>;
        /// Decode token IDs locally, without running model inference.
        async fn detokenize(
            &self,
            request: tonic::Request<super::DetokenizeRequest>,
        ) -> std::result::Result<tonic::Response<super::DetokenizeResponse>, tonic::Status>;
        /// Cancel by the native rid used by HTTP `/generate` or gRPC Generate, or
        /// cancel all active frontend generations with abort_all. HTTP chat/completion
        /// response IDs are not supported as rid selectors.
        /// Duplicate native IDs cancel all matching generations. A later reuse of the
        /// ID is unaffected. Internal control/detokenization requests are not targeted.
        async fn abort(
            &self,
            request: tonic::Request<super::AbortRequest>,
        ) -> std::result::Result<tonic::Response<super::AbortResponse>, tonic::Status>;
        /// `/set_internal_state`: override numeric scheduler knobs on every DP rank;
        /// one `updated` flag per rank.
        async fn set_internal_state(
            &self,
            request: tonic::Request<super::SetInternalStateRequest>,
        ) -> std::result::Result<tonic::Response<super::SetInternalStateResponse>, tonic::Status>;
        /// `/start_profile` and `/stop_profile`: a profiler capture on every DP rank.
        async fn start_profile(
            &self,
            request: tonic::Request<super::StartProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::ProfileResponse>, tonic::Status>;
        async fn stop_profile(
            &self,
            request: tonic::Request<super::StopProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::ProfileResponse>, tonic::Status>;
        /// `/{start,stop,dump}_expert_distribution_record`.
        async fn start_expert_distribution_record(
            &self,
            request: tonic::Request<super::ExpertDistributionRecordRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExpertDistributionRecordResponse>,
            tonic::Status,
        >;
        async fn stop_expert_distribution_record(
            &self,
            request: tonic::Request<super::ExpertDistributionRecordRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExpertDistributionRecordResponse>,
            tonic::Status,
        >;
        async fn dump_expert_distribution_record(
            &self,
            request: tonic::Request<super::ExpertDistributionRecordRequest>,
        ) -> std::result::Result<
            tonic::Response<super::ExpertDistributionRecordResponse>,
            tonic::Status,
        >;
        /// `/slow_down`: a per-forward sleep on every DP rank, for testing.
        async fn slow_down(
            &self,
            request: tonic::Request<super::SlowDownRequest>,
        ) -> std::result::Result<tonic::Response<super::SlowDownResponse>, tonic::Status>;
        /// `/release_memory_occupation`, `/resume_memory_occupation`.
        async fn release_memory_occupation(
            &self,
            request: tonic::Request<super::ReleaseMemoryOccupationRequest>,
        ) -> std::result::Result<tonic::Response<super::MemoryOccupationResponse>, tonic::Status>;
        async fn resume_memory_occupation(
            &self,
            request: tonic::Request<super::ResumeMemoryOccupationRequest>,
        ) -> std::result::Result<tonic::Response<super::MemoryOccupationResponse>, tonic::Status>;
        /// `/get_weights_by_name`: one parameter per DP rank.
        async fn get_weights_by_name(
            &self,
            request: tonic::Request<super::GetWeightsByNameRequest>,
        ) -> std::result::Result<tonic::Response<super::GetWeightsByNameResponse>, tonic::Status>;
        /// Weight-update process groups and remote-instance sends. Single-DP-rank
        /// deployments only, as in Python.
        async fn init_weights_update_group(
            &self,
            request: tonic::Request<super::InitWeightsUpdateGroupRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>;
        async fn destroy_weights_update_group(
            &self,
            request: tonic::Request<super::DestroyWeightsUpdateGroupRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>;
        async fn init_weights_send_group_for_remote_instance(
            &self,
            request: tonic::Request<super::InitWeightsSendGroupForRemoteInstanceRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>;
        async fn send_weights_to_remote_instance(
            &self,
            request: tonic::Request<super::SendWeightsToRemoteInstanceRequest>,
        ) -> std::result::Result<tonic::Response<super::WeightGroupResponse>, tonic::Status>;
        /// `/parse_function_call`, `/separate_reasoning`: frontend-only parsing.
        async fn parse_function_call(
            &self,
            request: tonic::Request<super::ParseFunctionCallRequest>,
        ) -> std::result::Result<tonic::Response<super::ParseFunctionCallResponse>, tonic::Status>;
        async fn separate_reasoning(
            &self,
            request: tonic::Request<super::SeparateReasoningRequest>,
        ) -> std::result::Result<tonic::Response<super::SeparateReasoningResponse>, tonic::Status>;
        /// `/add_external_corpus`, `/remove_external_corpus`, `/list_external_corpora`.
        async fn add_external_corpus(
            &self,
            request: tonic::Request<super::AddExternalCorpusRequest>,
        ) -> std::result::Result<tonic::Response<super::AddExternalCorpusResponse>, tonic::Status>;
        async fn remove_external_corpus(
            &self,
            request: tonic::Request<super::RemoveExternalCorpusRequest>,
        ) -> std::result::Result<tonic::Response<super::RemoveExternalCorpusResponse>, tonic::Status>;
        async fn list_external_corpora(
            &self,
            request: tonic::Request<super::ListExternalCorporaRequest>,
        ) -> std::result::Result<tonic::Response<super::ListExternalCorporaResponse>, tonic::Status>;
        /// `/freeze_gc`, `/pause_generation`, `/continue_generation`,
        /// `/pd_role_switch`, `/update_weight_version`.
        async fn freeze_gc(
            &self,
            request: tonic::Request<super::FreezeGcRequest>,
        ) -> std::result::Result<tonic::Response<super::FreezeGcResponse>, tonic::Status>;
        async fn pause_generation(
            &self,
            request: tonic::Request<super::PauseGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::GenerationPauseResponse>, tonic::Status>;
        async fn continue_generation(
            &self,
            request: tonic::Request<super::ContinueGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::GenerationPauseResponse>, tonic::Status>;
        async fn pd_role_switch(
            &self,
            request: tonic::Request<super::PdRoleSwitchRequest>,
        ) -> std::result::Result<tonic::Response<super::PdRoleSwitchResponse>, tonic::Status>;
        async fn update_weight_version(
            &self,
            request: tonic::Request<super::UpdateWeightVersionRequest>,
        ) -> std::result::Result<tonic::Response<super::UpdateWeightVersionResponse>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct SglangServiceServer<T> {
        inner: Arc<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
        max_decoding_message_size: Option<usize>,
        max_encoding_message_size: Option<usize>,
    }
    impl<T> SglangServiceServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
                max_decoding_message_size: None,
                max_encoding_message_size: None,
            }
        }
        pub fn with_interceptor<F>(inner: T, interceptor: F) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
        /// Limits the maximum size of a decoded message.
        ///
        /// Default: `4MB`
        #[must_use]
        pub fn max_decoding_message_size(mut self, limit: usize) -> Self {
            self.max_decoding_message_size = Some(limit);
            self
        }
        /// Limits the maximum size of an encoded message.
        ///
        /// Default: `usize::MAX`
        #[must_use]
        pub fn max_encoding_message_size(mut self, limit: usize) -> Self {
            self.max_encoding_message_size = Some(limit);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for SglangServiceServer<T>
    where
        T: SglangService,
        B: Body + std::marker::Send + 'static,
        B::Error: Into<StdError> + std::marker::Send + 'static,
    {
        type Response = http::Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<std::result::Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            match req.uri().path() {
                "/sglang.api.v1.SglangService/Generate" => {
                    #[allow(non_camel_case_types)]
                    struct GenerateSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::ServerStreamingService<super::GenerateRequest>
                        for GenerateSvc<T>
                    {
                        type Response = super::GenerateStreamItem;
                        type ResponseStream = T::GenerateStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GenerateRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::generate(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GenerateSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/HealthCheck" => {
                    #[allow(non_camel_case_types)]
                    struct HealthCheckSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::HealthCheckRequest>
                        for HealthCheckSvc<T>
                    {
                        type Response = super::HealthCheckResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::HealthCheckRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::health_check(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = HealthCheckSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/GetModelInfo" => {
                    #[allow(non_camel_case_types)]
                    struct GetModelInfoSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::GetModelInfoRequest>
                        for GetModelInfoSvc<T>
                    {
                        type Response = super::GetModelInfoResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetModelInfoRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::get_model_info(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GetModelInfoSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/GetServerInfo" => {
                    #[allow(non_camel_case_types)]
                    struct GetServerInfoSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::GetServerInfoRequest>
                        for GetServerInfoSvc<T>
                    {
                        type Response = super::GetServerInfoResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetServerInfoRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::get_server_info(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GetServerInfoSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/GetHiCacheStorage" => {
                    #[allow(non_camel_case_types)]
                    struct GetHiCacheStorageSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::GetHiCacheStorageRequest>
                        for GetHiCacheStorageSvc<T>
                    {
                        type Response = super::HiCacheStatus;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetHiCacheStorageRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::get_hi_cache_storage(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GetHiCacheStorageSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/AttachHiCacheStorage" => {
                    #[allow(non_camel_case_types)]
                    struct AttachHiCacheStorageSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::AttachHiCacheStorageRequest>
                        for AttachHiCacheStorageSvc<T>
                    {
                        type Response = super::HiCacheStorageResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::AttachHiCacheStorageRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::attach_hi_cache_storage(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = AttachHiCacheStorageSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/DetachHiCacheStorage" => {
                    #[allow(non_camel_case_types)]
                    struct DetachHiCacheStorageSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::DetachHiCacheStorageRequest>
                        for DetachHiCacheStorageSvc<T>
                    {
                        type Response = super::HiCacheStorageResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::DetachHiCacheStorageRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::detach_hi_cache_storage(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = DetachHiCacheStorageSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/ClearHiCacheStorage" => {
                    #[allow(non_camel_case_types)]
                    struct ClearHiCacheStorageSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ClearHiCacheStorageRequest>
                        for ClearHiCacheStorageSvc<T>
                    {
                        type Response = super::HiCacheStorageResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ClearHiCacheStorageRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::clear_hi_cache_storage(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ClearHiCacheStorageSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/ListModels" => {
                    #[allow(non_camel_case_types)]
                    struct ListModelsSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::ListModelsRequest> for ListModelsSvc<T> {
                        type Response = super::ListModelsResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ListModelsRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::list_models(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ListModelsSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/Detokenize" => {
                    #[allow(non_camel_case_types)]
                    struct DetokenizeSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::DetokenizeRequest> for DetokenizeSvc<T> {
                        type Response = super::DetokenizeResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::DetokenizeRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::detokenize(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = DetokenizeSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/Abort" => {
                    #[allow(non_camel_case_types)]
                    struct AbortSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::AbortRequest> for AbortSvc<T> {
                        type Response = super::AbortResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::AbortRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut =
                                async move { <T as SglangService>::abort(&inner, request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = AbortSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/SetInternalState" => {
                    #[allow(non_camel_case_types)]
                    struct SetInternalStateSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::SetInternalStateRequest>
                        for SetInternalStateSvc<T>
                    {
                        type Response = super::SetInternalStateResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::SetInternalStateRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::set_internal_state(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = SetInternalStateSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/StartProfile" => {
                    #[allow(non_camel_case_types)]
                    struct StartProfileSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::StartProfileRequest>
                        for StartProfileSvc<T>
                    {
                        type Response = super::ProfileResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StartProfileRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::start_profile(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StartProfileSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/StopProfile" => {
                    #[allow(non_camel_case_types)]
                    struct StopProfileSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::StopProfileRequest>
                        for StopProfileSvc<T>
                    {
                        type Response = super::ProfileResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StopProfileRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::stop_profile(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StopProfileSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/StartExpertDistributionRecord" => {
                    #[allow(non_camel_case_types)]
                    struct StartExpertDistributionRecordSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ExpertDistributionRecordRequest>
                        for StartExpertDistributionRecordSvc<T>
                    {
                        type Response = super::ExpertDistributionRecordResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ExpertDistributionRecordRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::start_expert_distribution_record(
                                    &inner, request,
                                )
                                .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StartExpertDistributionRecordSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/StopExpertDistributionRecord" => {
                    #[allow(non_camel_case_types)]
                    struct StopExpertDistributionRecordSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ExpertDistributionRecordRequest>
                        for StopExpertDistributionRecordSvc<T>
                    {
                        type Response = super::ExpertDistributionRecordResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ExpertDistributionRecordRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::stop_expert_distribution_record(
                                    &inner, request,
                                )
                                .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = StopExpertDistributionRecordSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/DumpExpertDistributionRecord" => {
                    #[allow(non_camel_case_types)]
                    struct DumpExpertDistributionRecordSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ExpertDistributionRecordRequest>
                        for DumpExpertDistributionRecordSvc<T>
                    {
                        type Response = super::ExpertDistributionRecordResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ExpertDistributionRecordRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::dump_expert_distribution_record(
                                    &inner, request,
                                )
                                .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = DumpExpertDistributionRecordSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/SlowDown" => {
                    #[allow(non_camel_case_types)]
                    struct SlowDownSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::SlowDownRequest> for SlowDownSvc<T> {
                        type Response = super::SlowDownResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::SlowDownRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::slow_down(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = SlowDownSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/ReleaseMemoryOccupation" => {
                    #[allow(non_camel_case_types)]
                    struct ReleaseMemoryOccupationSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ReleaseMemoryOccupationRequest>
                        for ReleaseMemoryOccupationSvc<T>
                    {
                        type Response = super::MemoryOccupationResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ReleaseMemoryOccupationRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::release_memory_occupation(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ReleaseMemoryOccupationSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/ResumeMemoryOccupation" => {
                    #[allow(non_camel_case_types)]
                    struct ResumeMemoryOccupationSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ResumeMemoryOccupationRequest>
                        for ResumeMemoryOccupationSvc<T>
                    {
                        type Response = super::MemoryOccupationResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ResumeMemoryOccupationRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::resume_memory_occupation(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ResumeMemoryOccupationSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/GetWeightsByName" => {
                    #[allow(non_camel_case_types)]
                    struct GetWeightsByNameSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::GetWeightsByNameRequest>
                        for GetWeightsByNameSvc<T>
                    {
                        type Response = super::GetWeightsByNameResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetWeightsByNameRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::get_weights_by_name(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = GetWeightsByNameSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/InitWeightsUpdateGroup" => {
                    #[allow(non_camel_case_types)]
                    struct InitWeightsUpdateGroupSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::InitWeightsUpdateGroupRequest>
                        for InitWeightsUpdateGroupSvc<T>
                    {
                        type Response = super::WeightGroupResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::InitWeightsUpdateGroupRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::init_weights_update_group(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = InitWeightsUpdateGroupSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/DestroyWeightsUpdateGroup" => {
                    #[allow(non_camel_case_types)]
                    struct DestroyWeightsUpdateGroupSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::DestroyWeightsUpdateGroupRequest>
                        for DestroyWeightsUpdateGroupSvc<T>
                    {
                        type Response = super::WeightGroupResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::DestroyWeightsUpdateGroupRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::destroy_weights_update_group(&inner, request)
                                    .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = DestroyWeightsUpdateGroupSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/InitWeightsSendGroupForRemoteInstance" => {
                    #[allow(non_camel_case_types)]
                    struct InitWeightsSendGroupForRemoteInstanceSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<
                            super::InitWeightsSendGroupForRemoteInstanceRequest,
                        > for InitWeightsSendGroupForRemoteInstanceSvc<T>
                    {
                        type Response = super::WeightGroupResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                super::InitWeightsSendGroupForRemoteInstanceRequest,
                            >,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::init_weights_send_group_for_remote_instance(
                                    &inner, request,
                                )
                                .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = InitWeightsSendGroupForRemoteInstanceSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/SendWeightsToRemoteInstance" => {
                    #[allow(non_camel_case_types)]
                    struct SendWeightsToRemoteInstanceSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::SendWeightsToRemoteInstanceRequest>
                        for SendWeightsToRemoteInstanceSvc<T>
                    {
                        type Response = super::WeightGroupResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::SendWeightsToRemoteInstanceRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::send_weights_to_remote_instance(
                                    &inner, request,
                                )
                                .await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = SendWeightsToRemoteInstanceSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/ParseFunctionCall" => {
                    #[allow(non_camel_case_types)]
                    struct ParseFunctionCallSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ParseFunctionCallRequest>
                        for ParseFunctionCallSvc<T>
                    {
                        type Response = super::ParseFunctionCallResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ParseFunctionCallRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::parse_function_call(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ParseFunctionCallSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/SeparateReasoning" => {
                    #[allow(non_camel_case_types)]
                    struct SeparateReasoningSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::SeparateReasoningRequest>
                        for SeparateReasoningSvc<T>
                    {
                        type Response = super::SeparateReasoningResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::SeparateReasoningRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::separate_reasoning(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = SeparateReasoningSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/AddExternalCorpus" => {
                    #[allow(non_camel_case_types)]
                    struct AddExternalCorpusSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::AddExternalCorpusRequest>
                        for AddExternalCorpusSvc<T>
                    {
                        type Response = super::AddExternalCorpusResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::AddExternalCorpusRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::add_external_corpus(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = AddExternalCorpusSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/RemoveExternalCorpus" => {
                    #[allow(non_camel_case_types)]
                    struct RemoveExternalCorpusSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::RemoveExternalCorpusRequest>
                        for RemoveExternalCorpusSvc<T>
                    {
                        type Response = super::RemoveExternalCorpusResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::RemoveExternalCorpusRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::remove_external_corpus(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = RemoveExternalCorpusSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/ListExternalCorpora" => {
                    #[allow(non_camel_case_types)]
                    struct ListExternalCorporaSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ListExternalCorporaRequest>
                        for ListExternalCorporaSvc<T>
                    {
                        type Response = super::ListExternalCorporaResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ListExternalCorporaRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::list_external_corpora(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ListExternalCorporaSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/FreezeGc" => {
                    #[allow(non_camel_case_types)]
                    struct FreezeGcSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::FreezeGcRequest> for FreezeGcSvc<T> {
                        type Response = super::FreezeGcResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::FreezeGcRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::freeze_gc(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = FreezeGcSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/PauseGeneration" => {
                    #[allow(non_camel_case_types)]
                    struct PauseGenerationSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::PauseGenerationRequest>
                        for PauseGenerationSvc<T>
                    {
                        type Response = super::GenerationPauseResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::PauseGenerationRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::pause_generation(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = PauseGenerationSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/ContinueGeneration" => {
                    #[allow(non_camel_case_types)]
                    struct ContinueGenerationSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ContinueGenerationRequest>
                        for ContinueGenerationSvc<T>
                    {
                        type Response = super::GenerationPauseResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ContinueGenerationRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::continue_generation(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ContinueGenerationSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/PdRoleSwitch" => {
                    #[allow(non_camel_case_types)]
                    struct PdRoleSwitchSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::PdRoleSwitchRequest>
                        for PdRoleSwitchSvc<T>
                    {
                        type Response = super::PdRoleSwitchResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::PdRoleSwitchRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::pd_role_switch(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = PdRoleSwitchSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/sglang.api.v1.SglangService/UpdateWeightVersion" => {
                    #[allow(non_camel_case_types)]
                    struct UpdateWeightVersionSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::UpdateWeightVersionRequest>
                        for UpdateWeightVersionSvc<T>
                    {
                        type Response = super::UpdateWeightVersionResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::UpdateWeightVersionRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::update_weight_version(&inner, request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = UpdateWeightVersionSvc(inner);
                        let codec = tonic_prost::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            )
                            .apply_max_message_size_config(
                                max_decoding_message_size,
                                max_encoding_message_size,
                            );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => Box::pin(async move {
                    let mut response = http::Response::new(tonic::body::Body::default());
                    let headers = response.headers_mut();
                    headers.insert(
                        tonic::Status::GRPC_STATUS,
                        (tonic::Code::Unimplemented as i32).into(),
                    );
                    headers.insert(
                        http::header::CONTENT_TYPE,
                        tonic::metadata::GRPC_CONTENT_TYPE,
                    );
                    Ok(response)
                }),
            }
        }
    }
    impl<T> Clone for SglangServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
                max_decoding_message_size: self.max_decoding_message_size,
                max_encoding_message_size: self.max_encoding_message_size,
            }
        }
    }
    /// Generated gRPC service name
    pub const SERVICE_NAME: &str = "sglang.api.v1.SglangService";
    impl<T> tonic::server::NamedService for SglangServiceServer<T> {
        const NAME: &'static str = SERVICE_NAME;
    }
}
