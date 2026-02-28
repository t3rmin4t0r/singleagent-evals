"""Runner factory."""

from bench.runner.base import BaseRunner


def create_runner(provider: str) -> BaseRunner:
    """Create a runner for the given provider."""
    if provider == "claude":
        from bench.runner.claude import ClaudeRunner
        return ClaudeRunner()
    if provider == "openai":
        from bench.runner.openai import OpenAIRunner
        return OpenAIRunner()
    raise ValueError(f"Unknown provider: {provider}. Available: claude, openai")
