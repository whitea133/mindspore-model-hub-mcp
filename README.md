# mindspore-model-hub-mcp

基于 MCP 的 MindSpore 开发工具包：提供官方模型清单的规范化查询/筛选，预留 PyTorch ↔ MindSpore 互转能力的扩展空间。

## 特性
- 官方模型清单：`data/mindspore_official_models.json` 统一字段（group/category/task/suite/links/metrics 等）。
- MCP 工具：`list_models` 支持 group/category/task/suite/关键词过滤；`get_model_info` 返回单个模型详情。
- MCP 资源：`mindspore://models/official` 暴露全量清单，便于客户端缓存或本地筛选。
- 抓取脚本：`scripts/update_model_list.py` 从官方页面生成最新 JSON。

## 目录结构
```
mindspore-model-hub-mcp/
├─ data/                      # 官方模型数据
│  └─ mindspore_official_models.json
├─ scripts/                   # 辅助脚本
│  ├─ update_model_list.py    # 抓取并生成模型 JSON
│  └─ __init__.py
├─ src/
│  └─ mindspore_mcp/          # 代码主体
│     ├─ server.py            # MCP 服务器（自动注册 tools/resources/prompts）
│     ├─ tools.py             # list_models / get_model_info
│     ├─ resource.py          # 资源定义
│     ├─ prompt.py            # prompt 定义
│     ├─ backup_server.py
│     ├─ config.py
│     └─ __init__.py
├─ tests/
├─ pyproject.toml
└─ uv.lock / README.md ...
```

## 快速开始（下载压缩包方式）
1) 下载项目压缩包并解压到本地路径（例如 `E:/CodeProject/mindspore-model-hub-mcp`），进入该目录。
2) 安装依赖：
```bash
uv sync
```
3) 客户端配置（示例，需按实际路径调整）：
```jsonc
// cline_mcp_settings.json 片段
"mindspore_model_hub_mcp": {
  "command": "uv",
  "args": [
    "--directory",
    "E:/CodeProject/mindspore-model-hub-mcp",
    "run",
    "python",
    "-m",
    "mindspore_mcp.server"
  ],
  "autoApprove": []
}
```
4) 调用示例：
   - 工具：`list_models(task="text-generation")`、`get_model_info("llama2")`
   - 资源：读取 `mindspore://models/official` 获取全量清单
5) 更新数据（可选）：
```bash
uv run python scripts/update_model_list.py
```

可选：如果需要在 IDE/测试中直接 `import mindspore_mcp`，可执行一次 `uv pip install -e .` 进行可编辑安装。

## 使用提示
- 客户端如不支持 `read_resource`，可用工具接口获取数据；支持资源的客户端可读取全量 JSON 后本地过滤，避免大 token 消耗。
- 可编辑安装后代码改动会实时生效；已有进程需重启/重载才会读取新代码。
