# mindspore-tools-mcp

<div align="center">

![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Ready-purple?style=flat-square)
![Tests](https://img.shields.io/github/actions/workflow/status/whitea133/mindspore-tools-mcp/ci.yml?branch=main&label=Tests&style=flat-square)
**基于 MCP 的 MindSpore 开发工具套件**

提供官方模型清单查询、智能模型推荐、PyTorch→MindSpore API 映射、以及 **AI 安全、数据处理、训练工具** 等开发必备功能。

> 📦 29 个 MCP 工具，一键连接 Claude / Cline / Cursor 等 AI 客户端

<img src="docs/mcp_present.gif" width="100%" alt="Claude Code MCP 连接演示">

</div>

---

## 📑 目录

- [📖 文档](#-文档)
- [✨ 功能特性](#-功能特性)
- [📦 目录结构](#-目录结构)
- [🚀 快速开始](#-快速开始)
- [📖 API 参考](#-api-参考)
- [💡 使用示例](#-使用示例)
- [📚 msutils 工具库](#-api-示例生成器模块-api_examples)
- [🔍 代码评分器](#-代码评分器-linter)
- [📚 API 示例生成器](#-api-示例生成器模块-api_examples-1)
- [🤝 贡献](#-贡献)
- [📝 许可](#-许可)

---

## 📖 文档

| 文档 | 说明 |
|------|------|
| [🚀 快速上手](docs/quickstart.html) | 3 分钟配置 Cline，含实机演示 |
| [📖 场景化使用指南](docs/USAGE_SCENARIOS.md) | 5 个真实场景，从模型选型到代码审查的完整演示 |
| [🏗️ 架构文档](docs/architecture.md) | 项目整体架构、模块关系和设计思路 |
| [📋 使用示例](examples/) | 17 个独立示例脚本，可直接运行 |

---

## ✨ 功能特性

### 🔍 官方模型检索

- `list_models` - 支持按 group/category/task/suite 或关键词过滤
- `get_model_info` - 返回单模型详情
- `mindspore://models/official` - 资源端点提供完整模型清单

### 🤖 智能模型推荐

- `recommend_models` - 根据自然语言描述智能推荐合适的模型
  - 支持任务描述: "图像分类"、"文本生成"、"OCR"、"推荐系统" 等
  - 支持硬件约束: ascend、gpu、cpu
  - 提供推荐理由和匹配分数
- `compare_models` - 对比多个模型，帮助选择最适合的模型

### 🗺️ API 映射工具

- `query_op_mapping` - 支持 section 过滤与模糊匹配
- `diagnose_translation` - 检查 PyTorch→MindSpore 代码翻译是否完整
- `mindspore://opmap/...` - 资源暴露 PyTorch→MindSpore API 映射

### 🛡️ AI 安全工具 (NEW!)

- `generate_adversarial_attack` - 生成对抗攻击配置和示例代码
  - 支持: FGSM、PGD、DeepFool、CW、JSMA
- `evaluate_model_robustness` - 评估模型鲁棒性配置

### 📊 数据处理工具 (NEW!)

- `create_data_augmentation_pipeline` - 创建数据增强流水线
  - 支持: 图像分类、目标检测、语义分割、NLP

### 🔧 训练工具 (NEW!)

- `get_lr_scheduler` - 获取学习率调度器配置
  - 支持: 余弦退火、阶梯衰减、多项式衰减、One Cycle
- `get_training_callbacks` - 获取训练回调函数配置
  - 支持: 检查点保存、早停、TensorBoard、梯度裁剪

### 📈 分析工具 (NEW!)

- `compute_model_complexity` - 计算模型 FLOPs、参数量、内存占用

### 🌐 分布式训练 (NEW!)

- `setup_distributed_training` - 配置分布式训练

### 🚀 部署工具 (NEW!)

- `quantize_model` - 模型量化配置
  - 支持: 动态量化、静态量化、量化感知训练
- `convert_model_format` - 模型格式转换配置

### 🔍 代码评分器 (NEW!)

- `lint_mindspore_code` - MindSpore 代码质量评分
  - 4 个维度评分: 性能、兼容性、最佳实践、可维护性
  - 26 条检查规则，自动发现问题并给出建议
- `get_lint_rules` - 获取所有检查规则列表
- `compare_code_snippets` - 对比两个代码片段的质量

### 🚂 训练模板生成器 (NEW!)

- `generate_training_template` - 一键生成完整训练脚本
  - 支持: ResNet、LeNet、MobileNet 等模型
  - 支持: CIFAR-10、CIFAR-100、ImageNet 数据集
  - 支持: Ascend、GPU、CPU 硬件
  - 包含: 数据加载、模型定义、训练配置、回调函数
- `get_available_options` - 获取所有可用选项
- `generate_quick_start` - 快速生成入门级训练脚本

### 📊 数据脚本

- `scripts/update_model_list.py` - 更新官方模型 JSON
- `scripts/fetch_api_mapping.py` - 抓取并刷新 API 映射

---

## 📦 目录结构

```
mindspore-tools-mcp/
├── data/                              # 官方模型与 API 映射数据
│   ├── convert/                       # 分 section 的映射分片
│   ├── mindspore_official_models.json
│   ├── pytorch_ms_api_mapping_consistent.json
│   └── pytorch_ms_api_mapping_diff.json
│
├── scripts/                           # 数据/映射更新脚本
│   ├── update_model_list.py
│   └── fetch_api_mapping.py
│
├── src/
│   └── mindspore_tools_mcp/           # MCP 服务
│       ├── server.py                   # MCP 入口
│       ├── tools.py                    # 模型检索工具
│       ├── msutils_tools.py            # 🆕 msutils MCP 工具封装
│       ├── linter_tools.py             # 🆕 代码评分器 MCP 封装
│       ├── template_tools.py           # 🆕 训练模板生成器封装
│       ├── msutils/                    # 🆕 MindSpore 开发工具库
│       │   ├── data/                   # 数据处理
│       │   ├── train/                  # 训练工具
│       │   ├── security/               # AI 安全
│       │   ├── eval/                   # 评估指标
│       │   ├── nlp/                    # NLP 工具
│       │   ├── distributed/            # 分布式训练
│       │   ├── deploy/                 # 部署工具
│       │   └── analysis/               # 分析可视化
│       ├── linter/                     # 🆕 代码评分器核心
│       │   ├── __init__.py
│       │   ├── rules.py                 # 检查规则定义
│       │   ├── checker.py               # 检查逻辑
│       │   └── formatter.py             # 报告格式化
│       ├── templates/                  # 🆕 训练模板生成器
│       │   ├── __init__.py
│       │   └── generator.py             # 模板生成逻辑
│       ├── api_examples/                 # 🆕 API 示例生成器
│       │   ├── __init__.py              # 核心查询逻辑
│       │   ├── registry.py              # API 注册表（34个API）
│       │   └── searcher.py              # 搜索逻辑
│       ├── resource.py                   # 资源定义
│       └── prompt.py                   # Prompt 注册
│
├── examples/                          # 🆕 使用示例
├── tests/                             # 测试文件
├── pyproject.toml                     # 项目配置
└── uv.lock                            # 依赖锁文件
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
git clone https://github.com/whitea133/mindspore-tools-mcp.git
cd mindspore-tools-mcp
uv sync
```

### 2️⃣ 配置 MCP 客户端

将下面的 `your-path` 替换为你的实际项目路径（如 `E:/CodeProject/mindspore-tools-mcp`），然后按你使用的客户端配置：

<!-- tabs:start -->

#### **Claude Code (CLI)**

项目级配置（在项目根目录创建 `.mcp.json`）：

```json
{
  "mcpServers": {
    "mindspore_tools_mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "your-path",
        "run",
        "python",
        "-m",
        "mindspore_tools_mcp.server"
      ]
    }
  }
}
```

也可以用命令行一步添加：

```bash
claude mcp add mindspore_tools_mcp -- uv --directory your-path run python -m mindspore_tools_mcp.server
```

> 验证：启动 `claude` 后输入 `/mcp` 查看连接状态。

#### **Claude Desktop**

编辑配置文件（路径：`%APPDATA%\Claude\claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "mindspore_tools_mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "your-path",
        "run",
        "python",
        "-m",
        "mindspore_tools_mcp.server"
      ]
    }
  }
}
```

> 配置后需完全退出 Claude Desktop（包括托盘图标）再重新打开。

#### **Cline (VS Code)**

编辑 Cline 配置文件（路径：`~/.cline/cline_mcp_settings.json`）：

```json
{
  "mindspore_tools_mcp": {
    "command": "uv",
    "args": [
      "--directory",
      "your-path",
      "run",
      "python",
      "-m",
      "mindspore_tools_mcp.server"
    ],
    "autoApprove": []
  }
}
```

#### **Cursor**

编辑项目级配置（路径：`your-project/.cursor/mcp.json`）：

```json
{
  "mcpServers": {
    "mindspore_tools_mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "your-path",
        "run",
        "python",
        "-m",
        "mindspore_tools_mcp.server"
      ]
    }
  }
}
```

<!-- tabs:end -->

### 3️⃣ 开始使用

配置完成后，重启你的 AI 客户端，即可在对话中直接调用所有工具：

```
💬 你：推荐一个适合在 Ascend 上做图像分类的模型

🤖 AI：[自动调用 recommend_models 工具]
    推荐 ResNet50 — 精度 76.1% | 参数 25.6M | 原生支持 Ascend...

💬 你：帮我生成 ResNet50 在 CIFAR-10 上的训练脚本

🤖 AI：[自动调用 generate_training_template 工具]
    ✅ 已生成 train_resnet50_cifar10.py，包含完整训练流程...
```

---

## 📖 API 参考

### 模型检索工具

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `list_models` | 列出模型 | `list_models(task="text-generation")` |
| `get_model_info` | 获取模型详情 | `get_model_info("llama2")` |
| `recommend_models` | 智能模型推荐 | `recommend_models("图像分类")` |
| `compare_models` | 对比模型 | `compare_models(["resnet50", "vit"])` |
| `query_op_mapping` | 查询 API 映射 | `query_op_mapping("torch.add")` |
| `diagnose_translation` | 诊断代码翻译 | `diagnose_translation(py_code, ms_code)` |

### AI 安全工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `generate_adversarial_attack` | 生成对抗攻击配置 | `generate_adversarial_attack("fgsm", epsilon=0.1)` |
| `evaluate_model_robustness` | 评估模型鲁棒性 | `evaluate_model_robustness(model_config)` |

### 数据处理工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `create_data_augmentation_pipeline` | 创建数据增强流水线 | `create_data_augmentation_pipeline("image_classification")` |

### 训练工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `get_lr_scheduler` | 获取学习率调度器 | `get_lr_scheduler("cosine_annealing", total_epochs=100)` |
| `get_training_callbacks` | 获取训练回调配置 | `get_training_callbacks(["checkpoint", "early_stopping"])` |

### 分析工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `compute_model_complexity` | 计算模型复杂度 | `compute_model_complexity("resnet50")` |

### 分布式训练 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `setup_distributed_training` | 配置分布式训练 | `setup_distributed_training(num_gpus=8)` |

### 部署工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `quantize_model` | 模型量化配置 | `quantize_model("dynamic", precision="int8")` |
| `convert_model_format` | 模型格式转换 | `convert_model_format("pytorch", "mindspore")` |

### 代码评分器 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `lint_mindspore_code` | 代码质量评分 | `lint_mindspore_code(code, level="all")` |
| `get_lint_rules` | 获取检查规则 | `get_lint_rules(category="performance")` |
| `compare_code_snippets` | 对比代码质量 | `compare_code_snippets(code_a, code_b)` |

### 训练模板生成器 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `generate_training_template` | 生成训练脚本 | `generate_training_template(model="resnet50")` |
| `get_available_options` | 获取可用选项 | `get_available_options()` |
| `generate_quick_start` | 快速入门脚本 | `generate_quick_start("beginner")` |

### API 示例生成器 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `get_api_examples` | 获取 API 完整示例 | `get_api_examples("nn.Conv2d")` |
| `search_apis` | 搜索相关 API | `search_apis("卷积")` |
| `list_api_categories` | 列出所有 API 分类 | `list_api_categories()` |
| `get_related_apis` | 获取相关 API 列表 | `get_related_apis("nn.Conv2d")` |
| `get_quick_reference` | 快速 API 参考（简洁版） | `get_quick_reference("nn.Conv2d")` |

### 资源列表

| 资源名 | 说明 |
|--------|------|
| `mindspore://models/official` | 官方模型完整清单 |
| `mindspore://opmap/pytorch/consistent` | 一致 API 映射 |
| `mindspore://opmap/pytorch/diff` | 差异 API 映射 |

### Prompt 列表

| Prompt 名 | 说明 |
|-----------|------|
| `model_lookup` | 按任务查找模型 |
| `model_recommend` | 智能模型推荐 |
| `model_compare` | 模型对比 |
| `migration_guide` | 迁移指南 |
| `performance_optimize` | 性能优化建议 |

---

## 💡 使用示例

### 智能模型推荐

```python
# 推荐"图像分类"模型
recommend_models("图像分类", limit=3)
# 返回: resnet18, resnet34, resnet50 (带推荐理由和性能指标)

# 推荐支持 Ascend 的文本生成模型
recommend_models("文本生成大模型", hardware="ascend", limit=3)
# 返回: llama2, qwen, baichuan2

# OCR 模型推荐
recommend_models("OCR文字识别", limit=5)
# 返回: dbnet_resnet18, dbnet_resnet50, ...
```

### AI 安全对抗攻击 🆕

```python
# 生成 FGSM 攻击配置
generate_adversarial_attack("fgsm", epsilon=0.1)
# 返回: 攻击配置 + MindSpore 示例代码

# 生成 PGD 攻击配置
generate_adversarial_attack("pgd", epsilon=0.3, num_iterations=40)
# 返回: 迭代攻击配置 + 代码示例
```

### 数据增强流水线 🆕

```python
# 创建图像分类增强流水线
create_data_augmentation_pipeline("image_classification")
# 返回: 增强方法列表 + MindSpore Dataset 代码

# 创建 NLP 数据增强
create_data_augmentation_pipeline("nlp", augmentations=["RandomDelete", "SynonymReplace"])
```

### 学习率调度 🆕

```python
# 余弦退火调度
get_lr_scheduler("cosine_annealing", total_epochs=100, warmup_epochs=5)
# 返回: 调度器配置 + 学习率曲线 + MindSpore 代码
```

### 模型复杂度分析 🆕

```python
# 分析 ResNet50 复杂度
compute_model_complexity("resnet50")
# 返回: FLOPs=4.12G, Params=25.6M, Memory=98MB
```

### 模型量化 🆕

```python
# 动态量化配置
quantize_model("dynamic", precision="int8")
# 返回: 量化配置 + MindSpore 代码 + 预期加速比

# 静态量化配置
quantize_model("static", precision="int8", calibration_dataset_size=100)
```

### 代码评分器 🆕

```python
# 检查代码质量
result = lint_mindspore_code(code)
# 返回: score=85, grade="B", dimensions={...}, issues=[...]

# 对比两个代码
compare_code_snippets(good_code, bad_code)
# 返回: winner, score差异, 问题差异
```

### 训练模板生成器 🆕

```python
# 生成完整训练脚本
result = generate_training_template(
    task="image_classification",
    model="resnet50",
    dataset="cifar10",
    hardware="Ascend"
)
# 返回: script, filename, config
# 保存到文件后即可运行

# 快速入门
result = generate_quick_start("beginner")
# 生成适合新手的简单训练脚本
```

### API 示例生成器 🆕

```python
# 获取 API 完整示例
get_api_examples("nn.Conv2d")
# 返回: 描述、签名、参数说明、多个示例代码、相关 API、官方文档链接

# 搜索相关 API
search_apis("卷积")
# 返回: nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Conv2dTranspose 等

# 列出所有 API 分类
list_api_categories()
# 返回: 神经网络层(11)、损失函数(3)、优化器(3)、数据处理(2)等

# 获取相关 API
get_related_apis("nn.Conv2d")
# 返回: nn.Conv1d, nn.Conv3d, nn.Dense, nn.BatchNorm2d

# 快速参考（简洁版）
get_quick_reference("nn.Conv2d")
# 返回: 描述 + 签名 + 一个基础示例
```

---

## 📚 API 示例生成器模块 (api_examples)

`msutils` 是集成在项目中的 MindSpore 开发工具库，包含以下模块：

| 模块 | 功能 | 文件数 | 代码行数 |
|------|------|--------|---------|
| `data/` | 数据增强、加载器、变换 | 4 | 1,173 |
| `train/` | 回调函数、学习率调度器 | 3 | 976 |
| `security/` | 对抗攻击、防御、鲁棒性评估 | 4 | 855 |
| `eval/` | 评估指标 | 2 | 516 |
| `nlp/` | 文本增强、分词器 | 3 | 849 |
| `distributed/` | DDP 工具 | 2 | 398 |
| `deploy/` | 模型转换、量化 | 3 | 315 |
| `analysis/` | 复杂度分析、可视化 | 3 | 454 |
| **总计** | | **25** | **5,664** |

---

## 🔍 代码评分器 (Linter)

`linter` 模块提供 MindSpore 代码质量评分功能，帮助开发者发现代码问题并给出改进建议。

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| ⚡ 性能 | 30% | 循环内创建对象、未使用混合精度等 |
| 🔌 兼容性 | 25% | PyTorch API 使用、废弃 API、Ascend 优化 |
| ✨ 最佳实践 | 25% | 随机种子、学习率调度、检查点、梯度裁剪 |
| 🔧 可维护性 | 20% | 函数长度、代码重复、文档字符串 |

### 检查规则

| 类别 | 规则数 | 示例 |
|------|--------|------|
| 性能规则 | 6 | 循环内创建优化器、未启用 AMP |
| 兼容性规则 | 5 | 使用 torch API、废弃 API |
| 最佳实践 | 7 | 未设置种子、未使用检查点 |
| 可维护性 | 5 | 函数过长、缺少文档 |
| **总计** | **26** | |

### 评分等级

| 等级 | 分数范围 | 说明 |
|------|---------|------|
| A | 90-100 | 优秀，代码质量很高 |
| B | 80-89 | 良好，有少量可改进之处 |
| C | 70-79 | 一般，建议优化 |
| D | 60-69 | 较差，需要改进 |
| F | 0-59 | 不合格，存在严重问题 |

---

## 📚 API 示例生成器模块 (api_examples)

`api_examples` 模块提供 MindSpore 常用 API 的示例代码和使用说明，帮助开发者快速学习和使用 MindSpore。

### 支持的 API 分类

| 分类 | 数量 | 示例 API |
|------|------|---------|
| 神经网络层 (nn) | 11 | Conv2d, BatchNorm2d, Dense, ReLU, LSTM, Embedding |
| 损失函数 (loss) | 3 | CrossEntropyLoss, BCEWithLogitsLoss, MSELoss |
| 优化器 (optim) | 3 | Adam, AdamW, SGD |
| 数据处理 (dataset) | 2 | MnistDataset, CIFAR-10 |
| 训练 (train) | 3 | Model, TrainOneStepCell, DynamicLossScaleUpdateCell |
| 学习率调度 (lr) | 2 | cosine_decay_lr, step_lr |
| 回调函数 (callbacks) | 3 | Callback, TimeMonitor, ModelCheckpoint |
| 网络单元 (cell) | 2 | Cell, SequentialCell |
| 模型加载保存 (common) | 2 | load_checkpoint, save_checkpoint |
| 算子 (ops) | 3 | cat, stack, Reshape |
| 分布式训练 (distributed) | 1 | DistributedSampler |
| **总计** | **34** | |

### 特色功能

- **精确匹配**: 输入 `nn.Conv2d` 返回完整信息
- **模糊搜索**: 输入 `卷积` 自动匹配所有相关 API
- **多示例**: 每个 API 提供 1-3 个使用示例
- **相关 API**: 自动关联相关模块（如 Conv2d → BatchNorm2d）
- **官方文档**: 每个 API 提供官方文档链接

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

> 💡 本项目欢迎 MindSpore 官方社区收录，如有意向请通过 [Issue](https://github.com/whitea133/mindspore-tools-mcp/issues) 联系。

---

## 📝 许可

MIT License - 详见 LICENSE 文件

---

## 👤 作者

- **GitHub**: [@whitea133](https://github.com/whitea133)
- **邮箱**: 1309848726@qq.com

---

## 🙏 致谢

- [MindSpore](https://www.mindspore.cn/) - 华为全场景深度学习框架
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐**

</div>