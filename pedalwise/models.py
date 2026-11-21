"""Chat models from a "provider:model" string, so switching provider is config (23).

    chat_model("openai:gpt-5-mini")
    chat_model("deepseek:deepseek-flash")
    chat_model("gpt-5-mini")               # no provider: OpenAI, as in Tutorials 01-22

init_chat_model() does most of the work: it knows LangChain's integration
package for each provider. The one exception is DeepSeek, which needs a
small fix, below.
"""

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

# The environment variable each provider's package reads its key from.
API_KEY_VARS = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", "google_genai": "GOOGLE_API_KEY"}


def split_spec(spec: str) -> tuple[str, str]:
    """'deepseek:deepseek-flash' -> ('deepseek', 'deepseek-flash'). No prefix means OpenAI."""
    provider, sep, name = spec.partition(":")
    return (provider, name) if sep else ("openai", spec)


class ThinkingDeepSeek(ChatDeepSeek):
    """ChatDeepSeek that sends the model's reasoning back to it.

    In thinking mode DeepSeek returns its chain of thought as
    reasoning_content, next to the answer. DeepSeek's docs say that when a
    request includes tools, the reasoning from earlier assistant turns must
    be sent back with the conversation, or the API returns HTTP 400.
    LangChain keeps reasoning_content on the message (additional_kwargs) but
    its OpenAI-format conversion leaves it out. This puts it back.

    In practice (tested November 2026) the API accepted agent conversations
    without it. Following the documented contract is still the safe choice:
    an API can start enforcing its own rules at any time, and sending the
    reasoning back keeps the model's earlier thinking in its context.
    """

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None,
                             **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        # One payload message per input message, in the same order.
        for message, sent in zip(messages, payload["messages"]):
            reasoning = isinstance(message, AIMessage) and message.additional_kwargs.get(
                "reasoning_content")
            if reasoning and sent.get("role") == "assistant":
                sent["reasoning_content"] = reasoning
        return payload


def default_model_spec() -> str:
    """The model the tutorials use: CHAT_MODEL, else OPENAI_MODEL, else gpt-5-mini."""
    return os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")


def deepseek_thinking_from_env() -> bool:
    return os.getenv("DEEPSEEK_THINKING", "").lower() in ("1", "true", "yes")


def default_chat_model(**kwargs: Any) -> BaseChatModel:
    """The chat model from .env, so every tutorial follows one setting (Tutorial 23).

    With no CHAT_MODEL set, this is the same as ChatOpenAI(model="gpt-5-mini", **kwargs).
    """
    return chat_model(default_model_spec(), deepseek_thinking=deepseek_thinking_from_env(),
                      **kwargs)


def chat_model(spec: str, deepseek_thinking: bool = False, **kwargs: Any) -> BaseChatModel:
    """Build a chat model. kwargs (max_tokens, timeout...) go to the provider class."""
    provider, name = split_spec(spec)
    if provider == "deepseek":
        # Thinking is on by default at DeepSeek. For a support chat it mostly
        # adds delay and output tokens, so it's off unless asked for.
        thinking = {"type": "enabled" if deepseek_thinking else "disabled"}
        return ThinkingDeepSeek(model=name, extra_body={"thinking": thinking}, **kwargs)
    return init_chat_model(name, model_provider=provider, **kwargs)
