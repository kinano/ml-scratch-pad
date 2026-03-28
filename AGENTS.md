# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A scratch pad for ML learning experiments using Python with scikit-learn, numpy, and matplotlib.

## Setup & Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run a script
uv run python src/main.py
uv run python src/plot_classifier_comparison.py

# Run with a specific Python version (pinned in .python-version)
uv run --python $(cat .python-version) python src/main.py
```

## Structure

- `src/` — experiment scripts; each file is typically a standalone script exploring a concept
- `pyproject.toml` — dependencies (scikit-learn, numpy, matplotlib)
- `.python-version` — Python version pin for uv
