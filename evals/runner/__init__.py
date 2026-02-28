"""Runner factory."""

from evals.runner.base import BaseRunner


def create_runner(provider: str) -> BaseRunner:
    """Create a runner for the given provider."""
    if provider == "claude":
        from evals.runner.claude import ClaudeRunner
        return ClaudeRunner()
    if provider == "openai":
        from evals.runner.openai import OpenAIRunner
        return OpenAIRunner()
    raise ValueError(f"Unknown provider: {provider}. Available: claude, openai")
