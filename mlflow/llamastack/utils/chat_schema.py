import logging
from typing import Any

from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import set_span_model_attribute

_logger = logging.getLogger(__name__)


def set_span_chat_attributes(span: LiveSpan, inputs: dict[str, Any], output: Any):
    """Set chat-related attributes on the span from inputs and output."""
    # Set model name if available
    set_span_model_attribute(span, inputs)

    # Extract and set usage information if available
    if usage := _parse_usage(output):
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)


def _parse_usage(output: Any) -> dict[str, Any] | None:
    """
    Parse token usage information from LlamaStack response objects.

    Args:
        output: The response object from LlamaStack API calls

    Returns:
        A dictionary containing token usage information.
    """
    if output is None:
        return None

    # Handle LlamaStack ChatCompletion response (OpenAI-compatible format)
    try:
        from llama_stack_client.types.chat_completion import ChatCompletion

        if isinstance(output, ChatCompletion) and (usage := output.usage):
            return {
                TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
            }
    except ImportError:
        _logger.debug("Failed to import ChatCompletion type from llama_stack_client")

    return None
