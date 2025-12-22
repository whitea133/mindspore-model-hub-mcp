# mindspore-tools-mcp

基于 MCP 的 MindSpore 模型与 API 映射工具包：提供官方模型清单的标准化查询，并内置 PyTorch → MindSpore API 映射与代码翻译辅助。

## 功能特性
- 官方模型检索：`list_models` 支持按 group/category/task/suite 或关键词过滤；`get_model_info` 返回单模型详情。
- 资源暴露：`mindspore://models/official` 提供完整模型清单；`mindspore://opmap/...` 资源暴露 PyTorch→MindSpore API 映射（全量/分 section、consistent/diff）。
- 映射工具：`query_op_mapping` 支持 section 过滤与模糊匹配；`translate_pytorch_code` 自动替换一致 API，并为差异项输出提示。
- 数据脚本：`scripts/update_model_list.py` 更新官方模型 JSON，`scripts/fetch_api_mapping.py` 抓取并刷新 API 映射。

## 目录结构
```
mindspore-tools-mcp/
├─ data/                      # 官方模型与 API 映射数据
│  ├─ convert/                # 分 section 的映射分片
│  ├─ mindspore_official_models.json
│  ├─ pytorch_ms_api_mapping_consistent.json
│  └─ pytorch_ms_api_mapping_diff.json
├─ scripts/                   # 数据/映射更新脚本
│  ├─ update_model_list.py
│  └─ fetch_api_mapping.py
├─ src/
│  └─ mindspore_tools_mcp/    # MCP 服务
│     ├─ server.py            # MCP 入口，自动注册 tools/resources/prompts
│     ├─ tools.py             # list_models/get_model_info/query_op_mapping/translate_pytorch_code 等
│     ├─ resource.py          # MCP 资源定义
│     ├─ prompt.py            # prompt 注册
│     ├─ backup_server.py
│     └─ main.py
├─ tests/
│  └─ test.py                 # smoke check
├─ pyproject.toml
└─ uv.lock
```

## 快速开始
1. 安装依赖
   ```bash
   uv sync
   ```
2. 启动 MCP 服务（stdio）
   ```bash
   uv run python -m mindspore_tools_mcp.server
   ```
3. 客户端配置示例（需按本地路径调整）
   ```jsonc
   // cline_mcp_settings.json 片段
   "mindspore_tools_mcp": {
     "command": "uv",
     "args": [
       "--directory",
       "E:/CodeProject/mindspore-tools-mcp",
       "run",
       "python",
       "-m",
       "mindspore_tools_mcp.server"
     ],
     "autoApprove": []
   }
   ```
4. 调用示例
   - 工具：`list_models(task="text-generation")`、`get_model_info("llama2")`、`query_op_mapping("torch.addmm")`、`translate_pytorch_code("import torch; torch.addmm(...)")`
   - 资源：读取 `mindspore://models/official` 或 `mindspore://opmap/pytorch/consistent`
5. 更新数据（可选）
   ```bash
   uv run python scripts/update_model_list.py       # 刷新官方模型清单
   uv run python scripts/fetch_api_mapping.py       # 刷新 API 映射数据
   ```

## 额外说明
- 若客户端不支持 `read_resource`，可通过工具接口获取同样数据并在本地缓存过滤。
- 如需在 IDE/测试中直接引用，执行一次 `uv pip install -e .` 进行可编辑安装。

