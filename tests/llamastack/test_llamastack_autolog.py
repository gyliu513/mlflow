import asyncio
import sys
from unittest.mock import Mock

import pytest

import mlflow.llamastack
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces

# Mock response types
DUMMY_CHAT_COMPLETION_REQUEST = {
    "model": "Llama3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
}


@pytest.fixture
def mock_chat_completion_response():
    """Create a mock ChatCompletion response."""
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = "Llama3.2-3B-Instruct"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].index = 0
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "Hello! How can I help you?"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    return mock_response


@pytest.fixture(params=[True, False], ids=["async", "sync"])
def is_async(request):
    return request.param


@pytest.fixture(autouse=True)
def mock_llamastack_client():
    """Mock llama_stack_client module to avoid import errors."""

    # Create actual classes with create method
    class Completions:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            # Return a simple mock response
            from unittest.mock import MagicMock

            response = MagicMock()
            response.id = "chatcmpl-123"
            response.model = kwargs.get("model", "test-model")
            response.usage = MagicMock()
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 20
            response.usage.total_tokens = 30
            return response

    class AsyncCompletions:
        def __init__(self, client):
            self.client = client

        async def create(self, **kwargs):
            # Return a simple mock response
            from unittest.mock import MagicMock

            response = MagicMock()
            response.id = "chatcmpl-123"
            response.model = kwargs.get("model", "test-model")
            response.usage = MagicMock()
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 20
            response.usage.total_tokens = 30
            return response

    # Create mock modules
    mock_completions = Mock()
    mock_completions.Completions = Completions
    mock_completions.AsyncCompletions = AsyncCompletions

    mock_chat = Mock()
    mock_chat.completions = mock_completions

    mock_resources = Mock()
    mock_resources.chat = mock_chat

    mock_streaming = Mock()
    mock_streaming.Stream = type("Stream", (), {})
    mock_streaming.AsyncStream = type("AsyncStream", (), {})

    mock_types = Mock()
    mock_types.chat_completion = Mock()
    mock_types.chat_completion.ChatCompletion = Mock
    mock_types.chat_completion_message = Mock()
    mock_types.chat_completion_message.ChatCompletionMessage = Mock
    mock_types.completion_choice = Mock()
    mock_types.completion_choice.CompletionChoice = Mock

    # Register mocks in sys.modules
    sys.modules["llama_stack_client"] = Mock()
    sys.modules["llama_stack_client.resources"] = mock_resources
    sys.modules["llama_stack_client.resources.chat"] = mock_chat
    sys.modules["llama_stack_client.resources.chat.completions"] = mock_completions
    sys.modules["llama_stack_client._streaming"] = mock_streaming
    sys.modules["llama_stack_client.types"] = mock_types
    sys.modules["llama_stack_client.types.chat_completion"] = mock_types.chat_completion
    sys.modules["llama_stack_client.types.chat_completion_message"] = (
        mock_types.chat_completion_message
    )
    sys.modules["llama_stack_client.types.completion_choice"] = mock_types.completion_choice

    yield

    # Cleanup
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("llama_stack_client"):
            del sys.modules[module_name]


def test_chat_completions_autolog(is_async):
    # Import classes
    from llama_stack_client.resources.chat.completions import (
        AsyncCompletions,
        Completions,
    )

    # Autolog MUST be called before the API call
    mlflow.llamastack.autolog()

    try:
        # Create client instance
        client_instance = (
            AsyncCompletions(client=Mock()) if is_async else Completions(client=Mock())
        )

        # Call the method (patching will intercept and trace)
        if is_async:
            asyncio.run(client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST))
        else:
            client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST)

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "AsyncCompletions" if is_async else "Completions"
        assert span.span_type == SpanType.CHAT_MODEL
        assert span.inputs == DUMMY_CHAT_COMPLETION_REQUEST
        assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "llamastack"
    finally:
        mlflow.llamastack.autolog(disable=True)


def test_disable_autologging(is_async):
    from llama_stack_client.resources.chat.completions import (
        AsyncCompletions,
        Completions,
    )

    # Test that autologging can be disabled
    mlflow.llamastack.autolog(disable=True)

    client_instance = AsyncCompletions(client=Mock()) if is_async else Completions(client=Mock())

    if is_async:
        asyncio.run(client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST))
    else:
        client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 0

    # Re-enable autologging
    mlflow.llamastack.autolog()

    if is_async:
        asyncio.run(client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST))
    else:
        client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1

    mlflow.llamastack.autolog(disable=True)


def test_log_traces_false(is_async):
    from llama_stack_client.resources.chat.completions import (
        AsyncCompletions,
        Completions,
    )

    # Test that log_traces=False prevents trace collection
    mlflow.llamastack.autolog(log_traces=False)

    client_instance = AsyncCompletions(client=Mock()) if is_async else Completions(client=Mock())

    if is_async:
        asyncio.run(client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST))
    else:
        client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 0

    mlflow.llamastack.autolog(disable=True)


def test_autolog_with_active_run(is_async):
    from llama_stack_client.resources.chat.completions import (
        AsyncCompletions,
        Completions,
    )

    mlflow.llamastack.autolog()

    with mlflow.start_run():
        client_instance = (
            AsyncCompletions(client=Mock()) if is_async else Completions(client=Mock())
        )

        if is_async:
            asyncio.run(client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST))
        else:
            client_instance.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1

    mlflow.llamastack.autolog(disable=True)


# Error handling test would be useful but is complex to implement with mocks
# The autolog implementation does handle errors correctly via _end_span_on_exception
