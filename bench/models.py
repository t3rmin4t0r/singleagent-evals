"""Data models for benchmark framework."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


def _generate_run_id() -> str:
    """Generate a UUIDv7 (time-based, sortable)."""
    return str(uuid.uuid7())


@dataclass
class ToolDef:
    """A tool definition: schema + callable."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]


@dataclass
class FileUpload:
    """Record of one file upload to API."""

    filename: str
    original_checksum: str
    updated_checksum: str
    file_id: str


@dataclass
class ToolCall:
    """Record of one tool invocation."""

    name: str
    arguments: dict[str, Any]
    result: str
    api_request_id: str | None = None


@dataclass
class RunRecord:
    """Complete record of a benchmark run."""

    provider: str
    model: str
    testcase: str = ""
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tool_calls: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    file_uploads: list[FileUpload] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    error: str | None = None
    followup_response: str = ""
    run_id: str = field(default_factory=_generate_run_id)
    api_request_ids: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out

    def summary(self) -> str:
        lines = [
            f"Provider: {self.provider} ({self.model})",
            f"Testcase: {self.testcase}",
            f"Tool calls: {self.total_tool_calls}",
            f"Tokens: {self.total_tokens_in:,} in / {self.total_tokens_out:,} out / {self.total_tokens:,} total",
            f"Wall time: {self.wall_time_seconds:.1f}s",
            f"Output files: {len(self.output_files)}",
        ]
        if self.error:
            lines.append(f"Error: {self.error}")
        for f in self.output_files:
            lines.append(f"  - {f}")
        return "\n".join(lines)
