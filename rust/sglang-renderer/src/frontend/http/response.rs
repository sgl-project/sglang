//! HTTP SSE framing for typed OpenAI response streams.

use super::error::error_payload;
use crate::ResponseError;
use axum::response::{
    IntoResponse, Response,
    sse::{Event, Sse},
};
use futures::{Stream, StreamExt};
use std::convert::Infallible;

pub(super) fn sse_response<T, S, F>(chunks: S, serialize: F) -> Response
where
    T: Send + 'static,
    S: Stream<Item = Result<T, ResponseError>> + Send + 'static,
    F: Fn(T) -> String + Send + 'static,
{
    let events = async_stream::stream! {
        futures::pin_mut!(chunks);
        while let Some(chunk) = chunks.next().await {
            let data = match chunk {
                Ok(chunk) => serialize(chunk),
                Err(error) => {
                    let status = super::error::response_status(&error);
                    error_payload(status, error.message).to_string()
                }
            };
            yield Ok::<_, Infallible>(Event::default().data(data));
        }
        // An error may be followed by the protocol's final usage chunk.
        // Only this transport owns the SSE terminator.
        yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
    };
    Sse::new(events).into_response()
}
