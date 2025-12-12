"""Prompt definitions for MCP."""

from __future__ import annotations

from typing import Callable, Dict

# central registry for prompts: name -> function
PROMPT_REGISTRY: Dict[str, Callable] = {}


def prompt(name: str | None = None) -> Callable:
    """Decorator to tag and register a function as an MCP prompt with an optional name."""

    def decorator(func: Callable) -> Callable:
        prompt_name = name or func.__name__
        PROMPT_REGISTRY[prompt_name] = func
        setattr(func, "__mcp_prompt_name__", prompt_name)
        return func

    return decorator


@prompt("model_lookup")
def model_lookup(task: str, limit: int = 5) -> str:
    """Generate a simple prompt for looking up models by task."""
    return f"Find up to {limit} MindSpore models relevant to task: {task}"
