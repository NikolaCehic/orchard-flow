"""Provider adapters for OrchardFlow agent nodes.

The adapters intentionally avoid importing vendor SDKs or validating live
credentials at import time. Local tests can use FakeProvider, while production
callers can pass a transport into the OpenAI or Anthropic adapters later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Callable, Mapping, MutableMapping, Protocol


ProviderTransport = Callable[[Mapping[str, Any], Mapping[str, str]], Mapping[str, Any]]


class ProviderConfigurationError(RuntimeError):
    """Raised when a real provider adapter has no usable credentials."""


class ProviderRuntimeError(RuntimeError):
    """Raised when a provider adapter cannot execute a configured request."""


class ProviderClient(Protocol):
    """Minimal interface consumed by agent nodes."""

    provider_name: str

    @property
    def is_configured(self) -> bool:
        """Return whether the provider can make a real external request."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> "ProviderResponse":
        """Generate a response for an agent node."""


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    model: str
    content: str
    confidence: float = 1.0
    usage: Mapping[str, int] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


PLACEHOLDER_SECRET_VALUES = {
    "",
    "placeholder",
    "changeme",
    "change-me",
    "todo",
    "none",
    "null",
    "your_api_key",
    "your-api-key",
    "your_openai_api_key",
    "your-anthropic-api-key",
    "sk-placeholder",
    "sk-...",
}


def is_placeholder_secret(value: str | None) -> bool:
    """Return True for missing or obviously non-live secrets."""

    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in PLACEHOLDER_SECRET_VALUES:
        return True
    return (
        "placeholder" in normalized
        or normalized.startswith("your_")
        or normalized.startswith("your-")
        or normalized.endswith("_here")
        or normalized.endswith("-here")
    )


class BaseProviderAdapter:
    provider_name = "base"
    api_key_env = ""
    model_env = ""
    default_model = "configured-by-env"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        transport: ProviderTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or self.default_model
        self.transport = transport

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        transport: ProviderTransport | None = None,
    ) -> "BaseProviderAdapter":
        env = os.environ if environ is None else environ
        return cls(
            api_key=env.get(cls.api_key_env),
            model=env.get(cls.model_env, cls.default_model),
            transport=transport,
        )

    @property
    def is_configured(self) -> bool:
        return not is_placeholder_secret(self.api_key)

    def config_summary(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider_name,
            "api_key_env": self.api_key_env,
            "model_env": self.model_env,
            "model": self.model,
            "configured": self.is_configured,
            "placeholder_safe": True,
        }

    def ensure_configured(self) -> None:
        if not self.is_configured:
            raise ProviderConfigurationError(
                f"{self.provider_name} adapter requires {self.api_key_env}; "
                "the current value is missing or placeholder-only."
            )

    def build_payload(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def parse_response(self, payload: Mapping[str, Any]) -> ProviderResponse:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        self.ensure_configured()
        if self.transport is None:
            raise ProviderRuntimeError(
                f"{self.provider_name} adapter is configured but no transport was supplied; "
                "local workflow tests should use FakeProvider."
            )
        request = self.build_payload(prompt, system=system, temperature=temperature)
        headers = self.build_headers()
        return self.parse_response(self.transport(request, headers))

    def build_headers(self) -> Mapping[str, str]:
        return {"authorization": f"Bearer {self.api_key}"}


class OpenAIProvider(BaseProviderAdapter):
    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    default_model = "openai-model-from-env"

    def build_payload(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> Mapping[str, Any]:
        messages: list[Mapping[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

    def parse_response(self, payload: Mapping[str, Any]) -> ProviderResponse:
        choices = payload.get("choices") or []
        content = ""
        if choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message") or {}
                if isinstance(message, Mapping):
                    content = str(message.get("content", ""))
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            content=content,
            usage=dict(usage),
            raw=payload,
        )


class AnthropicProvider(BaseProviderAdapter):
    provider_name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    model_env = "ANTHROPIC_MODEL"
    default_model = "anthropic-model-from-env"

    def build_headers(self) -> Mapping[str, str]:
        return {
            "x-api-key": str(self.api_key),
            "anthropic-version": "2023-06-01",
        }

    def build_payload(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }
        if system:
            payload["system"] = system
        return payload

    def parse_response(self, payload: Mapping[str, Any]) -> ProviderResponse:
        blocks = payload.get("content") or []
        content_parts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, Mapping) and block.get("type") == "text":
                    content_parts.append(str(block.get("text", "")))
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            content="\n".join(part for part in content_parts if part),
            usage=dict(usage),
            raw=payload,
        )


@dataclass
class FakeProvider:
    """Deterministic provider for local tests and placeholder-key workflows."""

    provider_name: str = "fake"
    model: str = "fake-local-model"
    response_prefix: str = "fake-response"
    confidence: float = 0.95
    calls: list[Mapping[str, Any]] = field(default_factory=list)

    @property
    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
            }
        )
        return ProviderResponse(
            provider=self.provider_name,
            model=self.model,
            content=f"{self.response_prefix}: {prompt}",
            confidence=self.confidence,
            raw={"fake": True},
        )


__all__ = [
    "AnthropicProvider",
    "BaseProviderAdapter",
    "FakeProvider",
    "OpenAIProvider",
    "ProviderClient",
    "ProviderConfigurationError",
    "ProviderResponse",
    "ProviderRuntimeError",
    "is_placeholder_secret",
]
