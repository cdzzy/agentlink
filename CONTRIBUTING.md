# Contributing to agentlink

Thanks for your interest in contributing!

## How to help

1. **Report bugs** ? open an issue with steps to reproduce, expected vs actual behavior.
2. **Suggest features** ? open an issue tagged `enhancement`. Read the roadmap first to avoid duplicates.
3. **Submit PRs** ? fork, branch from `main`, and open a PR. CI must pass.

## Development

```bash
npm install    # or pip install -e ".[dev]"
npm test       # or pytest tests/ -v
```

## Guidelines

- Keep the core zero-dependency (or minimal).
- Every new feature needs tests.
- Update `CHANGELOG.md` for user-facing changes.
- Follow the existing code style (lint runs in CI).
