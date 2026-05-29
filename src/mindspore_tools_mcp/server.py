"""MCP server for MindSpore model registry and development tools."""

from __future__ import annotations

import inspect

from mcp.server.fastmcp import FastMCP

from mindspore_tools_mcp import prompt as prompt_module
from mindspore_tools_mcp import resource as resource_module
from mindspore_tools_mcp import tools
from mindspore_tools_mcp import msutils_tools  # 新增: msutils 工具封装
from mindspore_tools_mcp import linter_tools  # 代码评分器（纯标准库，无额外依赖）

# 以下模块有额外依赖，使用延迟导入
# - template_tools: 需要 mindspore
# - api_tools: 需要 pydantic + mindspore
_LAZY_MODULES = {
    "template_tools": "mindspore_tools_mcp.template_tools",
    "api_tools": "mindspore_tools_mcp.api_tools",
}


def _try_register_module(mcp: FastMCP, module_name: str) -> None:
    """尝试导入并注册工具模块，缺少依赖时跳过并打印警告。"""
    try:
        mod = __import__(module_name, fromlist=["register_tools"])
        register_module_functions(mcp, mod)
    except ImportError as e:
        print(f"[WARN] 跳过 {module_name}: {e}")
        print(f"       请安装对应依赖以启用相关工具。")
        print(f"       例如: pip install mindspore pydantic")


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
    
    # auto register msutils tools (AI安全、数据处理、训练工具等)
    register_module_functions(mcp, msutils_tools)
    
    # auto register linter tools (代码评分器，纯标准库，无额外依赖)
    register_module_functions(mcp, linter_tools)
    
    # auto register template tools (训练模板生成器) - 需要 mindspore
    _try_register_module(mcp, _LAZY_MODULES["template_tools"])
    
    # auto register api_examples tools (API 示例生成器) - 需要 pydantic + mindspore
    _try_register_module(mcp, _LAZY_MODULES["api_tools"])
    
    # auto register resources and prompts
    register_module_resources(mcp, resource_module)
    register_module_prompts(mcp, prompt_module)

    return mcp


if __name__ == "__main__":
    print("Starting MindSpore Models MCP server...")
    print("  - Model registry: list_models, recommend_models, compare_models...")
    print("  - msutils tools: adversarial_attack, lr_scheduler, callbacks...")
    print("  - Linter: lint_mindspore_code, get_lint_rules, compare_code_snippets...")
    print("  - Templates: generate_training_template, get_available_options... (需要 mindspore)")
    print("  - API Examples: get_api_examples, search_apis... (需要 pydantic + mindspore)")
    server = create_server()
    server.run(transport="stdio")
