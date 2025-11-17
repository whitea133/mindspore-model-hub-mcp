# MINDSPORE-MODEL-HUB-MCP

暂时的项目结构：
  mindspore-model-hub-mcp/
  ├─ pyproject.toml          # uv 管理依赖 & 脚本；包含 mcp、fastapi 等运行依赖
  ├─ uv.lock                 # uv 自动生成的锁文件
  ├─ README.md               # 说明项目目标、MVP 功能、使用步骤
  ├─ mcp.json                # MCP server 元数据（名称、端点、工具/资源定义）
  ├─ src/
  │  ├─ __init__.py
  │  ├─ main.py              # 入口：初始化 uv 虚拟环境、注册 MCP 工具
  │  ├─ config.py            # 配置读取（数据路径、缓存设置）
  │  ├─ data_loader.py       # 读取 data/models.json、提供缓存接口
  │  ├─ handlers/
  │  │   ├─ list_models.py   # 实现 `list_models` 过滤逻辑
  │  └─ templates/
  │      ├─ inference.py.j2
  │      └─ finetune.py.j2
  ├─ data/
  │  ├─ models.json          # 模型清单：名称/任务/指标/下载链接/标签
  │  ├─ datasets.json        # 数据集元信息（可选）
  │  └─ samples/             # 某些固定脚本或资源
  │  ├─ fetch_model_zoo.py   # 从官方资源抓取/更新模型数据
  │  └─ validate_data.py
  └─ docs/
     ├─ api.md               # MCP 接口说明（参数/返回体）
     └─ roadmap.md           # 后续扩展计划

- 聚合模型元数据：汇总 Transformer、CV、GNN 等模型的名称、任务类型、适用数据集、MindSpore 版本、权重下载地址、维护者等
字段，形成结构化清单。
- 筛选与检索接口：提供类似 list_models(task="text-classification")、list_models(dataset="COCO") 的调用，快速获得匹配模
型列表；支持 tags、关键词等扩展过滤。
- 模型详情查询：get_model_details(name="BERT_BASE_MS") 可返回单个模型的完整信息，包括依赖、论文链接、指标来源、贡献
者等。
- 自动生成示例：generate_sample(name="YOLOv5_MS", scenario="inference") 会根据模板 + 元数据拼出可运行的 MindSpore 脚本
（加载权重、构建网络、推理或 fine-tune），帮助用户“一键复制即用”。
- MCP 集成：以上能力通过 MCP 资源/工具暴露，可被 Codex CLI、聊天机器人或 IDE 插件调用，方便展示和演示。

首版计划覆盖 MindSpore 社区最常用的模型（NLP/CV/GNN 各数个），随着贡献者提交新的模型和 benchmark 数据逐步扩展，并计划
引入评分、贡献者排行榜和自动部署脚本等增值能力。