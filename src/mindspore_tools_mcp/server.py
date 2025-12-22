"""MCP server for MindSpore model registry."""

from __future__ import annotations

import inspect

from mcp.server.fastmcp import FastMCP

from mindspore_tools_mcp import prompt as prompt_module
from mindspore_tools_mcp import resource as resource_module
from mindspore_tools_mcp import tools


def register_module_functions(mcp: FastMCP, module) -> None:
    """Auto-register public functions in the tools module as MCP tools."""
    for _, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__module__ != module.__name__:    # 过滤非本模块函数
            continue
        if fn.__name__.startswith("_"):     # 过滤私有函数
            continue
        # print(f"[REGISTER TOOL] {fn.__name__}")  # 临时调试
        mcp.add_tool(fn)


def register_module_resources(mcp: FastMCP, module) -> None:
    """Auto-register resources, preferring module registry if present."""
    registry = getattr(module, "RESOURCE_REGISTRY", None)
    if isinstance(registry, dict):
        for uri, fn in registry.items():
            mcp.resource(uri)(fn)
        return
    # fallback: attribute tagging
    for _, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__module__ != module.__name__:
            continue
        uri = getattr(fn, "__mcp_resource_uri__", None)
        if not uri:
            continue
        mcp.resource(uri)(fn)


def register_module_prompts(mcp: FastMCP, module) -> None:
    """Auto-register prompts, preferring module registry if present."""
    registry = getattr(module, "PROMPT_REGISTRY", None)
    if isinstance(registry, dict):
        for name, fn in registry.items():
            mcp.prompt(name)(fn)
        return
    # fallback: attribute tagging
    for _, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__module__ != module.__name__:
            continue
        prompt_name = getattr(fn, "__mcp_prompt_name__", None)
        if not prompt_name:
            continue
        mcp.prompt(prompt_name)(fn)


def create_server() -> FastMCP:
    mcp = FastMCP("MindSpore Models")

    # auto register tools from tools.py (e.g., list_models, get_model_info)
    register_module_functions(mcp, tools)
    # auto register resources and prompts
    register_module_resources(mcp, resource_module)
    register_module_prompts(mcp, prompt_module)

    return mcp


if __name__ == "__main__":
    server = create_server()
    server.run(transport="stdio")
