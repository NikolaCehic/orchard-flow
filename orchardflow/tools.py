"""Tool registry with schema metadata and local rate-limit enforcement."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Deque, Mapping


ToolHandler = Callable[[Mapping[str, Any]], Any]


class ToolRegistryError(RuntimeError):
    """Base class for tool registry failures."""


class ToolAlreadyRegistered(ToolRegistryError):
    """Raised when registering a duplicate tool name."""


class ToolNotFound(ToolRegistryError):
    """Raised when a caller asks for an unknown tool."""


class ToolRateLimitExceeded(ToolRegistryError):
    """Raised when a tool exceeds its configured call budget."""


class ToolValidationError(ToolRegistryError):
    """Raised when tool arguments do not match basic schema constraints."""


@dataclass(frozen=True)
class RateLimit:
    max_calls: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    schema: Mapping[str, Any]
    rate_limit: RateLimit
    handler: ToolHandler
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Register tools by name and enforce per-tool sliding-window limits."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._calls: dict[str, Deque[float]] = defaultdict(deque)

    def register(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        handler: ToolHandler,
        rate_limit: RateLimit,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolDefinition:
        if name in self._tools:
            raise ToolAlreadyRegistered(f"Tool {name!r} is already registered")
        if not name.strip():
            raise ToolValidationError("Tool name cannot be empty")
        definition = ToolDefinition(
            name=name,
            schema=dict(schema),
            rate_limit=rate_limit,
            handler=handler,
            description=description,
            metadata=dict(metadata or {}),
        )
        self._tools[name] = definition
        return definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFound(f"Tool {name!r} is not registered") from exc

    def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in sorted(self._tools)]

    def can_call(self, name: str, *, now: float | None = None) -> bool:
        tool = self.get(name)
        current_time = time.time() if now is None else now
        history = self._active_history(tool.name, tool.rate_limit, current_time)
        return len(history) < tool.rate_limit.max_calls

    def check_rate_limit(self, name: str, *, now: float | None = None) -> None:
        tool = self.get(name)
        current_time = time.time() if now is None else now
        history = self._active_history(tool.name, tool.rate_limit, current_time)
        if len(history) >= tool.rate_limit.max_calls:
            raise ToolRateLimitExceeded(
                f"Tool {name!r} exceeded {tool.rate_limit.max_calls} calls "
                f"per {tool.rate_limit.window_seconds:g} seconds"
            )

    def call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        now: float | None = None,
    ) -> Any:
        tool = self.get(name)
        args = dict(arguments or {})
        self.validate_arguments(tool, args)
        current_time = time.time() if now is None else now
        self.check_rate_limit(name, now=current_time)
        self._calls[name].append(current_time)
        return tool.handler(args)

    def remaining_calls(self, name: str, *, now: float | None = None) -> int:
        tool = self.get(name)
        current_time = time.time() if now is None else now
        history = self._active_history(name, tool.rate_limit, current_time)
        return max(tool.rate_limit.max_calls - len(history), 0)

    def validate_arguments(self, tool: ToolDefinition, arguments: Mapping[str, Any]) -> None:
        required = tool.schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolValidationError(
                f"Tool {tool.name!r} missing required arguments: {', '.join(missing)}"
            )

        if tool.schema.get("additionalProperties") is False:
            properties = tool.schema.get("properties", {})
            unexpected = [key for key in arguments if key not in properties]
            if unexpected:
                raise ToolValidationError(
                    f"Tool {tool.name!r} received unexpected arguments: "
                    f"{', '.join(unexpected)}"
                )

    def _active_history(
        self,
        name: str,
        rate_limit: RateLimit,
        now: float,
    ) -> Deque[float]:
        history = self._calls[name]
        while history and now - history[0] >= rate_limit.window_seconds:
            history.popleft()
        return history


def build_default_tool_registry() -> ToolRegistry:
    """Create a minimal safe registry for local graph exercises."""

    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Return the supplied text without external side effects.",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        rate_limit=RateLimit(max_calls=30, window_seconds=60),
        handler=lambda args: {"text": args["text"]},
    )
    return registry


__all__ = [
    "RateLimit",
    "ToolAlreadyRegistered",
    "ToolDefinition",
    "ToolNotFound",
    "ToolRateLimitExceeded",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolValidationError",
    "build_default_tool_registry",
]
