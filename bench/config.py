"""Constants and run configuration."""

from dataclasses import dataclass

# Tool result truncation limit (unified across all test cases)
MAX_TOOL_RESULT_LENGTH = 15_000

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "claude": "claude-sonnet-4-5-20250929",
    "openai": "gpt-5.1",
}

# Default follow-up prompt when test case doesn't override
DEFAULT_FOLLOWUP_PROMPT = "please reverify your analysis again."


@dataclass(frozen=True)
class RunConfig:
    """Immutable configuration for a benchmark run."""

    testcase: str
    provider: str
    model: str
    followup: bool = False
    iterations: int = 1
    max_turns: int = 30
    output_root: str = "output"
    debug: bool = False
    text: bool = False
    version: str = "v2"

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")
