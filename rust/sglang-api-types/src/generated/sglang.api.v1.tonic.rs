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
