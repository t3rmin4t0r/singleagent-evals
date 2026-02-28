# Single-Agent Evals

A framework for evaluating LLM tool-calling capabilities on domain-specific financial tasks.

## Overview

This framework tests how well LLMs can use tools to complete realistic financial reconciliation tasks. It measures:

- **Accuracy**: Does the model produce correct results?
- **Tool usage**: How efficiently does the model use available tools?
- **Robustness**: Can the model handle messy real-world data?

## Supported Providers

- **Claude** (Anthropic) - claude-sonnet-4-5-20250929, claude-opus-4-5-20251101
- **OpenAI** - gpt-5.1

## Installation

Requires Python 3.14+.

```bash
# Clone the repo
git clone https://github.com/t3rmin4t0r/singleagent-evals.git
cd singleagent-evals

# Install with uv
uv sync
```

## Usage

```bash
# List available test cases
uv run evals --list

# Run a test case
uv run evals -t reconciliation_complex -p claude

# Run with OpenAI
uv run evals -t reconciliation_yearly -p openai

# Run both providers
uv run evals -t reconciliation -p both

# Run multiple iterations
uv run evals -t reconciliation_complex -p claude -n 5

# Enable follow-up verification prompt
uv run evals -t reconciliation_yearly -p claude -f
```

## Test Cases

| Test Case | Description | Invoices | Vendors | Months |
|-----------|-------------|----------|---------|--------|
| `reconciliation` | Basic Q1 reconciliation | 9 | 3 | 3 |
| `reconciliation_complex` | Q1 with intentional discrepancies | 19 | 6 | 4 |
| `reconciliation_extended` | H1 with duplicates | 22 | 6 | 6 |
| `reconciliation_yearly` | Full year reconciliation | 84 | 7 | 12 |
| `reconciliation_adversarial` | Mismatched vendor names | 84 | 7 | 12 |

## How It Works

1. **Input**: The model receives an Excel expense report and PDF invoices
2. **Task**: Reconcile invoice amounts against QBO expense data
3. **Tools**: The model uses `save_reconciliation` to record matched pairs
4. **Verification**: Results are compared against a golden baseline

## Output

Each run produces:
- `run_record.json` - Metadata (tokens, time, tool calls)
- `trace.jsonl` - Detailed trace with all tool calls
- `reconciliation_output.csv` - The model's reconciliation results
- `debug.log` - Debug output (when not using --debug flag)

## Environment Variables

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## CLI Options

```
--testcase, -t    Test case name
--provider, -p    LLM provider (claude, openai, both)
--model, -m       Override default model
--followup, -f    Send follow-up verification prompt
--iterations, -n  Number of iterations (default: 1)
--max-turns       Max tool-call turns (default: 30)
--debug           Enable debug output
--text            Convert files to text instead of uploading
--output-root     Output directory (default: output)
--list            List available test cases
```
