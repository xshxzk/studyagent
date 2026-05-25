from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def print_tool_result(result: ToolResult) -> None:
    print(f"success: {result.success}")
    print(f"content: {result.content}")

    if result.data:
        print(f"data: {result.data}")

    if result.error:
        print(f"error: {result.error}")
