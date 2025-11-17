# Repository Guidelines

## Project Structure & Module Organization
仓库保持极简：`src/` 存放运行时代码，`main.py` 启动 MCP 工具链，后续功能如 `handlers/`、`templates/` 也应置于其中；`tests/` 一致镜像源码结构，例如 `tests/test_data_loader.py` 对应 `src/data_loader.py`。`data/` 建议保存模型/数据集 JSON 清单与 `samples/` 示例脚本，长文档放在 `docs/`。请避免把生成产物或大型二进制提交到 Git。

## Build, Test, and Development Commands
全部工作流使用 uv：
- `uv sync` 安装或更新 `.venv/` 中的依赖。
- `uv run python -m src.main` 在本地运行 MCP 入口，验证 CLI/Server 行为。
- `uv run pytest` 运行完整测试，可附加节点如 `uv run pytest tests/test_data_loader.py::test_cache` 进行定向验证。
- `uv run python -m compileall src` 发布前快速检查语法错误。

## Coding Style & Naming Conventions
目标 Python 3.11，统一 4 空格缩进，公共函数写类型注解并使用简洁 Google 风格 docstring。模块与文件命名保持 `snake_case`，类使用 `PascalCase`，常量 `UPPER_SNAKE_CASE`。Jinja2 模板放在 `src/templates/`，以场景命名，如 `inference.py.j2`。导入顺序遵循「标准库-三方-本地」，优先 `pathlib` 而非裸字符串，函数保持短小且显式返回。

## Testing Guidelines
Pytest 为默认框架，遵循 Arrange-Act-Assert 结构并使用描述性名称（如 `test_list_models_filters_by_task`）。测试数据优先来自 `data/models.json` 或 `tests/fixtures/` 构造的轻量字典，禁止联网。重点覆盖筛选逻辑、缓存、模板渲染，合入前力求语句覆盖率 ≥80%。快照或 golden 测试应读取 `data/samples/`，保持期望集中管理。

## Commit & Pull Request Guidelines
当前无历史，建议采用 Conventional Commits（例如 `feat(data): add gnn registry`）并一次只解决单一问题。提交底部可引用相关 Issue（`Refs: #12`）。PR 需包含：背景/目标、主要修改点、验证方式（命令或截图）、以及对 `mcp.json`、`models.json` 等契约的影响说明。即便是自提 PR 也要请求复审，并等待 CI/linters 全绿后再合并。

## Security & Configuration Tips
切勿在版本库中存放密钥；通过 `config.py` 读取环境变量。对外暴露的 JSON 输入需严格校验，避免未受控数据进 MCP 资源。同步外部模型中心时优先使用只读令牌，并遵守 `.gitignore`，确保 `.venv/` 和缓存不被提交。