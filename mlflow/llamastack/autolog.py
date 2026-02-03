import json
import logging
from typing import Any, AsyncIterator, Iterator

import mlflow.llamastack
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.llamastack.constant import FLAVOR_NAME
from mlflow.llamastack.utils.chat_schema import set_span_chat_attributes
from mlflow.tracing.constant import (
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
    SpanAttributeKey,
    TokenUsageKey,
)
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import safe_patch

_logger = logging.getLogger(__name__)


def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    """
    Enables (or disables) and configures autologging from LlamaStack to MLflow.

    Args:
        disable: If ``True``, disables the LlamaStack autologging integration. If ``False``,
            enables the LlamaStack autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            LlamaStack that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during LlamaStack
            autologging. If ``False``, show all events and warnings during LlamaStack
            autologging.
        log_traces: If ``True``, traces are logged for LlamaStack models. If ``False``, no traces
            are collected during inference. Default to ``True``.
    """
    _autolog(
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        log_traces=log_traces,
    )


autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    from llama_stack_client.resources.chat.completions import (
        AsyncCompletions as AsyncCompletionsResource,
    )
    from llama_stack_client.resources.chat.completions import (
        Completions as CompletionsResource,
    )

    safe_patch(FLAVOR_NAME, CompletionsResource, "create", patched_call)
    safe_patch(FLAVOR_NAME, AsyncCompletionsResource, "create", async_patched_call)


def patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.llamastack.FLAVOR_NAME)

    if config.log_traces:
        span = _start_span(self, kwargs)

    try:
        result = original(self, *args, **kwargs)
    except Exception as e:
        if config.log_traces:
            _end_span_on_exception(span, e)
        raise

    if config.log_traces:
        _end_span_on_success(span, kwargs, result)

    return result


async def async_patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.llamastack.FLAVOR_NAME)

    if config.log_traces:
        span = _start_span(self, kwargs)

    try:
        result = await original(self, *args, **kwargs)
    except Exception as e:
        if config.log_traces:
            _end_span_on_exception(span, e)
        raise

    if config.log_traces:
        _end_span_on_success(span, kwargs, result)

    return result


def _start_span(instance: Any, inputs: dict[str, Any]) -> LiveSpan:
    # Record input parameters to attributes (exclude messages from attributes)
    attributes = {k: v for k, v in inputs.items() if k != "messages"}
    attributes[SpanAttributeKey.MESSAGE_FORMAT] = "llamastack"

    return start_span_no_context(
        name=instance.__class__.__name__,
        span_type=SpanType.CHAT_MODEL,
        inputs=inputs,
        attributes=attributes,
    )


def _end_span_on_success(span: LiveSpan, inputs: dict[str, Any], result: Any):
    try:
        # Check if result is a streaming response
        from llama_stack_client._streaming import AsyncStream, Stream
    except ImportError:
        # Fallback if streaming classes are not available
        Stream = None
        AsyncStream = None

    if Stream and isinstance(result, Stream):
        # Sync streaming
        def _stream_output_logging_hook(stream: Iterator) -> Iterator:
            output = []
            for i, chunk in enumerate(stream):
                _add_span_event(span, i, chunk)
                output.append(chunk)
                yield chunk
            _process_last_chunk(span, chunk, inputs, output)

        result._iterator = _stream_output_logging_hook(result._iterator)
    elif AsyncStream and isinstance(result, AsyncStream):
        # Async streaming
        async def _stream_output_logging_hook(stream: AsyncIterator) -> AsyncIterator:
            output = []
            async for chunk in stream:
                _add_span_event(span, len(output), chunk)
                output.append(chunk)
                yield chunk
            _process_last_chunk(span, chunk, inputs, output)

        result._iterator = _stream_output_logging_hook(result._iterator)
    else:
        # Non-streaming response
        try:
            set_span_chat_attributes(span, inputs, result)
            span.end(outputs=result)
        except Exception as e:
            _logger.warning(f"Encountered unexpected error when ending trace: {e}", exc_info=True)


def _process_last_chunk(
    span: LiveSpan, chunk: Any, inputs: dict[str, Any], output: list[Any]
) -> None:
    try:
        if not output:
            output = None
        elif output[0].object == "chat.completion.chunk":
            # Reconstruct a completion object from streaming chunks
            output = _reconstruct_completion_from_stream(output)
            # Set usage information on span if available
            if usage := getattr(chunk, "usage", None):
                usage_dict = {
                    TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
                    TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
                    TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
                }
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)

        _end_span_on_success(span, inputs, output)
    except Exception as e:
        _logger.warning(
            f"Encountered unexpected error when autologging processes the chunks in response: {e}"
        )


def _reconstruct_completion_from_stream(chunks: list[Any]) -> Any:
    """
    Reconstruct a completion object from streaming chunks.
    LlamaStack follows OpenAI-compatible format.
    """
    try:
        from llama_stack_client.types.chat_completion import ChatCompletion
        from llama_stack_client.types.chat_completion_message import ChatCompletionMessage
        from llama_stack_client.types.completion_choice import CompletionChoice
    except ImportError:
        _logger.debug("Failed to import LlamaStack types for stream reconstruction")
        return chunks

    # Build the message from chunks
    def _extract_content(chunk: Any) -> str:
        if not chunk.choices:
            return ""
        content = chunk.choices[0].delta.content
        return content or ""

    message = ChatCompletionMessage(
        role="assistant",
        content="".join(map(_extract_content, chunks)),
    )

    # Extract metadata from the last chunk
    last_chunk = chunks[-1]
    finish_reason = "stop"
    if choices := getattr(last_chunk, "choices", None):
        if chunk_choice := choices[0]:
            finish_reason = getattr(chunk_choice, "finish_reason") or finish_reason

    choice = CompletionChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )

    # Build the completion object
    return ChatCompletion(
        id=last_chunk.id,
        choices=[choice],
        created=last_chunk.created,
        model=last_chunk.model,
        object="chat.completion",
        usage=last_chunk.usage,
    )


def _end_span_on_exception(span: LiveSpan, e: Exception):
    try:
        span.add_event(SpanEvent.from_exception(e))
        span.end(status=SpanStatusCode.ERROR)
    except Exception as inner_e:
        _logger.warning(f"Encountered unexpected error when ending trace: {inner_e}")


def _add_span_event(span: LiveSpan, index: int, chunk: Any):
    span.add_event(
        SpanEvent(
            name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=index),
            attributes={STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(chunk, cls=TraceJSONEncoder)},
        )
    )
