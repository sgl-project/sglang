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
        /// SGLang-native RPCs (typed proto)
        pub async fn text_generate(
            &mut self,
            request: impl tonic::IntoRequest<super::TextGenerateRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::TextGenerateResponse>>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/TextGenerate",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "TextGenerate",
            ));
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn generate(
            &mut self,
            request: impl tonic::IntoRequest<super::GenerateRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::GenerateResponse>>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Generate");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "Generate",
            ));
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn text_embed(
            &mut self,
            request: impl tonic::IntoRequest<super::TextEmbedRequest>,
        ) -> std::result::Result<tonic::Response<super::TextEmbedResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/TextEmbed");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "TextEmbed",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn embed(
            &mut self,
            request: impl tonic::IntoRequest<super::EmbedRequest>,
        ) -> std::result::Result<tonic::Response<super::EmbedResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Embed");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.runtime.v1.SglangService", "Embed"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn classify(
            &mut self,
            request: impl tonic::IntoRequest<super::ClassifyRequest>,
        ) -> std::result::Result<tonic::Response<super::ClassifyResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Classify");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "Classify",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn tokenize(
            &mut self,
            request: impl tonic::IntoRequest<super::TokenizeRequest>,
        ) -> std::result::Result<tonic::Response<super::TokenizeResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Tokenize");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "Tokenize",
            ));
            self.inner.unary(req, path, codec).await
        }
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
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Detokenize");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "Detokenize",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn health_check(
            &mut self,
            request: impl tonic::IntoRequest<super::HealthCheckRequest>,
        ) -> std::result::Result<tonic::Response<super::HealthCheckResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/HealthCheck",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "HealthCheck",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn watch_engine_state(
            &mut self,
            request: impl tonic::IntoRequest<super::WatchEngineStateRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::EngineStateSnapshot>>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/WatchEngineState",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "WatchEngineState",
            ));
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn get_model_info(
            &mut self,
            request: impl tonic::IntoRequest<super::GetModelInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetModelInfoResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/GetModelInfo",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "GetModelInfo",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn get_server_info(
            &mut self,
            request: impl tonic::IntoRequest<super::GetServerInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetServerInfoResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/GetServerInfo",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "GetServerInfo",
            ));
            self.inner.unary(req, path, codec).await
        }
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
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/ListModels");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "ListModels",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn get_load(
            &mut self,
            request: impl tonic::IntoRequest<super::GetLoadRequest>,
        ) -> std::result::Result<tonic::Response<super::GetLoadResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/GetLoad");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "GetLoad",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn abort(
            &mut self,
            request: impl tonic::IntoRequest<super::AbortRequest>,
        ) -> std::result::Result<tonic::Response<super::AbortResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Abort");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.runtime.v1.SglangService", "Abort"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn flush_cache(
            &mut self,
            request: impl tonic::IntoRequest<super::FlushCacheRequest>,
        ) -> std::result::Result<tonic::Response<super::FlushCacheResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/FlushCache");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "FlushCache",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn pause_generation(
            &mut self,
            request: impl tonic::IntoRequest<super::PauseGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::PauseGenerationResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/PauseGeneration",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "PauseGeneration",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn continue_generation(
            &mut self,
            request: impl tonic::IntoRequest<super::ContinueGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::ContinueGenerationResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/ContinueGeneration",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "ContinueGeneration",
            ));
            self.inner.unary(req, path, codec).await
        }
        /// OpenAI-compatible RPCs (JSON pass-through)
        pub async fn chat_complete(
            &mut self,
            request: impl tonic::IntoRequest<super::OpenAiRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::OpenAiStreamChunk>>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/ChatComplete",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "ChatComplete",
            ));
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn complete(
            &mut self,
            request: impl tonic::IntoRequest<super::OpenAiRequest>,
        ) -> std::result::Result<
            tonic::Response<tonic::codec::Streaming<super::OpenAiStreamChunk>>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Complete");
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "Complete",
            ));
            self.inner.server_streaming(req, path, codec).await
        }
        pub async fn open_ai_embed(
            &mut self,
            request: impl tonic::IntoRequest<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/OpenAIEmbed",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "OpenAIEmbed",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn open_ai_classify(
            &mut self,
            request: impl tonic::IntoRequest<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/OpenAIClassify",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "OpenAIClassify",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn score(
            &mut self,
            request: impl tonic::IntoRequest<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Score");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.runtime.v1.SglangService", "Score"));
            self.inner.unary(req, path, codec).await
        }
        pub async fn rerank(
            &mut self,
            request: impl tonic::IntoRequest<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/sglang.runtime.v1.SglangService/Rerank");
            let mut req = request.into_request();
            req.extensions_mut()
                .insert(GrpcMethod::new("sglang.runtime.v1.SglangService", "Rerank"));
            self.inner.unary(req, path, codec).await
        }
        /// Admin/Ops RPCs
        pub async fn start_profile(
            &mut self,
            request: impl tonic::IntoRequest<super::StartProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::StartProfileResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/StartProfile",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "StartProfile",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn stop_profile(
            &mut self,
            request: impl tonic::IntoRequest<super::StopProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::StopProfileResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/StopProfile",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "StopProfile",
            ));
            self.inner.unary(req, path, codec).await
        }
        pub async fn update_weights_from_disk(
            &mut self,
            request: impl tonic::IntoRequest<super::UpdateWeightsRequest>,
        ) -> std::result::Result<tonic::Response<super::UpdateWeightsResponse>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::unknown(format!("Service was not ready: {}", e.into()))
            })?;
            let codec = tonic_prost::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/UpdateWeightsFromDisk",
            );
            let mut req = request.into_request();
            req.extensions_mut().insert(GrpcMethod::new(
                "sglang.runtime.v1.SglangService",
                "UpdateWeightsFromDisk",
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
        /// SGLang-native RPCs (typed proto)
        async fn text_generate(
            &self,
            request: tonic::Request<super::TextGenerateRequest>,
        ) -> std::result::Result<
            tonic::Response<BoxStream<super::TextGenerateResponse>>,
            tonic::Status,
        > {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn generate(
            &self,
            request: tonic::Request<super::GenerateRequest>,
        ) -> std::result::Result<tonic::Response<BoxStream<super::GenerateResponse>>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn text_embed(
            &self,
            request: tonic::Request<super::TextEmbedRequest>,
        ) -> std::result::Result<tonic::Response<super::TextEmbedResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn embed(
            &self,
            request: tonic::Request<super::EmbedRequest>,
        ) -> std::result::Result<tonic::Response<super::EmbedResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn classify(
            &self,
            request: tonic::Request<super::ClassifyRequest>,
        ) -> std::result::Result<tonic::Response<super::ClassifyResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn tokenize(
            &self,
            request: tonic::Request<super::TokenizeRequest>,
        ) -> std::result::Result<tonic::Response<super::TokenizeResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn detokenize(
            &self,
            request: tonic::Request<super::DetokenizeRequest>,
        ) -> std::result::Result<tonic::Response<super::DetokenizeResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn health_check(
            &self,
            request: tonic::Request<super::HealthCheckRequest>,
        ) -> std::result::Result<tonic::Response<super::HealthCheckResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn watch_engine_state(
            &self,
            request: tonic::Request<super::WatchEngineStateRequest>,
        ) -> std::result::Result<
            tonic::Response<BoxStream<super::EngineStateSnapshot>>,
            tonic::Status,
        > {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn get_model_info(
            &self,
            request: tonic::Request<super::GetModelInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetModelInfoResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn get_server_info(
            &self,
            request: tonic::Request<super::GetServerInfoRequest>,
        ) -> std::result::Result<tonic::Response<super::GetServerInfoResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn list_models(
            &self,
            request: tonic::Request<super::ListModelsRequest>,
        ) -> std::result::Result<tonic::Response<super::ListModelsResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn get_load(
            &self,
            request: tonic::Request<super::GetLoadRequest>,
        ) -> std::result::Result<tonic::Response<super::GetLoadResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn abort(
            &self,
            request: tonic::Request<super::AbortRequest>,
        ) -> std::result::Result<tonic::Response<super::AbortResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn flush_cache(
            &self,
            request: tonic::Request<super::FlushCacheRequest>,
        ) -> std::result::Result<tonic::Response<super::FlushCacheResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn pause_generation(
            &self,
            request: tonic::Request<super::PauseGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::PauseGenerationResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn continue_generation(
            &self,
            request: tonic::Request<super::ContinueGenerationRequest>,
        ) -> std::result::Result<tonic::Response<super::ContinueGenerationResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        /// OpenAI-compatible RPCs (JSON pass-through)
        async fn chat_complete(
            &self,
            request: tonic::Request<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<BoxStream<super::OpenAiStreamChunk>>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn complete(
            &self,
            request: tonic::Request<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<BoxStream<super::OpenAiStreamChunk>>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn open_ai_embed(
            &self,
            request: tonic::Request<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn open_ai_classify(
            &self,
            request: tonic::Request<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn score(
            &self,
            request: tonic::Request<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn rerank(
            &self,
            request: tonic::Request<super::OpenAiRequest>,
        ) -> std::result::Result<tonic::Response<super::OpenAiResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        /// Admin/Ops RPCs
        async fn start_profile(
            &self,
            request: tonic::Request<super::StartProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::StartProfileResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn stop_profile(
            &self,
            request: tonic::Request<super::StopProfileRequest>,
        ) -> std::result::Result<tonic::Response<super::StopProfileResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
        async fn update_weights_from_disk(
            &self,
            request: tonic::Request<super::UpdateWeightsRequest>,
        ) -> std::result::Result<tonic::Response<super::UpdateWeightsResponse>, tonic::Status>
        {
            Err(tonic::Status::unimplemented("Not yet implemented"))
        }
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
                "/sglang.runtime.v1.SglangService/TextGenerate" => {
                    #[allow(non_camel_case_types)]
                    struct TextGenerateSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::ServerStreamingService<super::TextGenerateRequest>
                        for TextGenerateSvc<T>
                    {
                        type Response = super::TextGenerateResponse;
                        type ResponseStream = BoxStream<super::TextGenerateResponse>;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::TextGenerateRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::text_generate(&inner, request).await
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
                        let method = TextGenerateSvc(inner);
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
                "/sglang.runtime.v1.SglangService/Generate" => {
                    #[allow(non_camel_case_types)]
                    struct GenerateSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::ServerStreamingService<super::GenerateRequest>
                        for GenerateSvc<T>
                    {
                        type Response = super::GenerateResponse;
                        type ResponseStream = BoxStream<super::GenerateResponse>;
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
                "/sglang.runtime.v1.SglangService/TextEmbed" => {
                    #[allow(non_camel_case_types)]
                    struct TextEmbedSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::TextEmbedRequest> for TextEmbedSvc<T> {
                        type Response = super::TextEmbedResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::TextEmbedRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::text_embed(&inner, request).await
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
                        let method = TextEmbedSvc(inner);
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
                "/sglang.runtime.v1.SglangService/Embed" => {
                    #[allow(non_camel_case_types)]
                    struct EmbedSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::EmbedRequest> for EmbedSvc<T> {
                        type Response = super::EmbedResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::EmbedRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut =
                                async move { <T as SglangService>::embed(&inner, request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = EmbedSvc(inner);
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
                "/sglang.runtime.v1.SglangService/Classify" => {
                    #[allow(non_camel_case_types)]
                    struct ClassifySvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::ClassifyRequest> for ClassifySvc<T> {
                        type Response = super::ClassifyResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ClassifyRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::classify(&inner, request).await
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
                        let method = ClassifySvc(inner);
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
                "/sglang.runtime.v1.SglangService/Tokenize" => {
                    #[allow(non_camel_case_types)]
                    struct TokenizeSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::TokenizeRequest> for TokenizeSvc<T> {
                        type Response = super::TokenizeResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::TokenizeRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::tokenize(&inner, request).await
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
                        let method = TokenizeSvc(inner);
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
                "/sglang.runtime.v1.SglangService/Detokenize" => {
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
                "/sglang.runtime.v1.SglangService/HealthCheck" => {
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
                "/sglang.runtime.v1.SglangService/WatchEngineState" => {
                    #[allow(non_camel_case_types)]
                    struct WatchEngineStateSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::ServerStreamingService<super::WatchEngineStateRequest>
                        for WatchEngineStateSvc<T>
                    {
                        type Response = super::EngineStateSnapshot;
                        type ResponseStream = BoxStream<super::EngineStateSnapshot>;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::WatchEngineStateRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::watch_engine_state(&inner, request).await
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
                        let method = WatchEngineStateSvc(inner);
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
                "/sglang.runtime.v1.SglangService/GetModelInfo" => {
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
                "/sglang.runtime.v1.SglangService/GetServerInfo" => {
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
                "/sglang.runtime.v1.SglangService/ListModels" => {
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
                "/sglang.runtime.v1.SglangService/GetLoad" => {
                    #[allow(non_camel_case_types)]
                    struct GetLoadSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::GetLoadRequest> for GetLoadSvc<T> {
                        type Response = super::GetLoadResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetLoadRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::get_load(&inner, request).await
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
                        let method = GetLoadSvc(inner);
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
                "/sglang.runtime.v1.SglangService/Abort" => {
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
                "/sglang.runtime.v1.SglangService/FlushCache" => {
                    #[allow(non_camel_case_types)]
                    struct FlushCacheSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::FlushCacheRequest> for FlushCacheSvc<T> {
                        type Response = super::FlushCacheResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::FlushCacheRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::flush_cache(&inner, request).await
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
                        let method = FlushCacheSvc(inner);
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
                "/sglang.runtime.v1.SglangService/PauseGeneration" => {
                    #[allow(non_camel_case_types)]
                    struct PauseGenerationSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::PauseGenerationRequest>
                        for PauseGenerationSvc<T>
                    {
                        type Response = super::PauseGenerationResponse;
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
                "/sglang.runtime.v1.SglangService/ContinueGeneration" => {
                    #[allow(non_camel_case_types)]
                    struct ContinueGenerationSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::UnaryService<super::ContinueGenerationRequest>
                        for ContinueGenerationSvc<T>
                    {
                        type Response = super::ContinueGenerationResponse;
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
                "/sglang.runtime.v1.SglangService/ChatComplete" => {
                    #[allow(non_camel_case_types)]
                    struct ChatCompleteSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::ServerStreamingService<super::OpenAiRequest>
                        for ChatCompleteSvc<T>
                    {
                        type Response = super::OpenAiStreamChunk;
                        type ResponseStream = BoxStream<super::OpenAiStreamChunk>;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::OpenAiRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::chat_complete(&inner, request).await
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
                        let method = ChatCompleteSvc(inner);
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
                "/sglang.runtime.v1.SglangService/Complete" => {
                    #[allow(non_camel_case_types)]
                    struct CompleteSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService>
                        tonic::server::ServerStreamingService<super::OpenAiRequest>
                        for CompleteSvc<T>
                    {
                        type Response = super::OpenAiStreamChunk;
                        type ResponseStream = BoxStream<super::OpenAiStreamChunk>;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::OpenAiRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::complete(&inner, request).await
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
                        let method = CompleteSvc(inner);
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
                "/sglang.runtime.v1.SglangService/OpenAIEmbed" => {
                    #[allow(non_camel_case_types)]
                    struct OpenAIEmbedSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::OpenAiRequest> for OpenAIEmbedSvc<T> {
                        type Response = super::OpenAiResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::OpenAiRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::open_ai_embed(&inner, request).await
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
                        let method = OpenAIEmbedSvc(inner);
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
                "/sglang.runtime.v1.SglangService/OpenAIClassify" => {
                    #[allow(non_camel_case_types)]
                    struct OpenAIClassifySvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::OpenAiRequest> for OpenAIClassifySvc<T> {
                        type Response = super::OpenAiResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::OpenAiRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::open_ai_classify(&inner, request).await
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
                        let method = OpenAIClassifySvc(inner);
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
                "/sglang.runtime.v1.SglangService/Score" => {
                    #[allow(non_camel_case_types)]
                    struct ScoreSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::OpenAiRequest> for ScoreSvc<T> {
                        type Response = super::OpenAiResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::OpenAiRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut =
                                async move { <T as SglangService>::score(&inner, request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = ScoreSvc(inner);
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
                "/sglang.runtime.v1.SglangService/Rerank" => {
                    #[allow(non_camel_case_types)]
                    struct RerankSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::OpenAiRequest> for RerankSvc<T> {
                        type Response = super::OpenAiResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::OpenAiRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut =
                                async move { <T as SglangService>::rerank(&inner, request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let max_decoding_message_size = self.max_decoding_message_size;
                    let max_encoding_message_size = self.max_encoding_message_size;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let method = RerankSvc(inner);
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
                "/sglang.runtime.v1.SglangService/StartProfile" => {
                    #[allow(non_camel_case_types)]
                    struct StartProfileSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::StartProfileRequest>
                        for StartProfileSvc<T>
                    {
                        type Response = super::StartProfileResponse;
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
                "/sglang.runtime.v1.SglangService/StopProfile" => {
                    #[allow(non_camel_case_types)]
                    struct StopProfileSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::StopProfileRequest>
                        for StopProfileSvc<T>
                    {
                        type Response = super::StopProfileResponse;
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
                "/sglang.runtime.v1.SglangService/UpdateWeightsFromDisk" => {
                    #[allow(non_camel_case_types)]
                    struct UpdateWeightsFromDiskSvc<T: SglangService>(pub Arc<T>);
                    impl<T: SglangService> tonic::server::UnaryService<super::UpdateWeightsRequest>
                        for UpdateWeightsFromDiskSvc<T>
                    {
                        type Response = super::UpdateWeightsResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::UpdateWeightsRequest>,
                        ) -> Self::Future {
                            let inner = Arc::clone(&self.0);
                            let fut = async move {
                                <T as SglangService>::update_weights_from_disk(&inner, request)
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
                        let method = UpdateWeightsFromDiskSvc(inner);
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
    pub const SERVICE_NAME: &str = "sglang.runtime.v1.SglangService";
    impl<T> tonic::server::NamedService for SglangServiceServer<T> {
        const NAME: &'static str = SERVICE_NAME;
    }
}
