# mindspore-model-hub-mcp

基于 MCP 的 MindSpore 开发工具包：提供官方模型清单的规范化查询/筛选，并预留跨框架（如 PyTorch ↔ MindSpore）转换能力的扩展空间。

## 特性
- 官方模型清单：`data/mindspore_official_models.json` 按统一字段整理（group/category/task/suite/links/metrics 等）。
- MCP 工具：`list_models` 支持 group/category/task/suite/关键词过滤，`get_model_info` 返回单个模型详情。
- MCP 资源：`mindspore://models/official` 暴露全量清单，便于客户端缓存或本地筛选。
- 抓取脚本：`scripts/update_model_list.py` 从官方页面生成最新 JSON。

## 目录结构
```
mindspore-model-hub-mcp/
├─ data/                       # 官方模型数据
│   └─ mindspore_official_models.json
├─ scripts/                    # 辅助脚本
│   ├─ update_model_list.py    # 抓取并生成模型 JSON
│   └─ __init__.py
├─ src/
│   └─ mindspore_mcp/          # 代码主体
│       ├─ server.py           # MCP 服务器，自动注册 tools/resources/prompts
│       ├─ tools.py            # list_models / get_model_info
│       ├─ resource.py         # 资源定义（官方清单）
│       ├─ prompt.py           # 示例 prompt 定义
│       ├─ backup_server.py    # 备用入口
│       ├─ config.py           # 预留配置
│       └─ __init__.py
├─ tests/                      # 测试
├─ pyproject.toml              # 包配置（setuptools，src 布局）
└─ uv.lock / README.md ...
```

## 使用
1) 安装依赖并可编辑安装
```bash
uv pip install -e .
```
2) 运行 MCP 服务器
```bash
uv run python -m mindspore_mcp.server
```
3) 调用示例
   - 工具：`list_models(task="text-generation")`，`get_model_info("llama2")`
   - 资源：读取 `mindspore://models/official` 获取全量清单
4) 更新数据
```bash
uv run python scripts/update_model_list.py
```

## 开发提示
- 代码改动后可编辑安装会实时生效。
- 客户端若不支持 `read_resource`，可用工具接口获取数据；支持资源的客户端可直接读取全量 JSON 再本地过滤，避免大 token 消耗。
