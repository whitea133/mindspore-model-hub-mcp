# Changelog

> 所有重要的项目更新都会记录在此文件中。

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，
版本管理遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

---

## [Unreleased] - 开发中

### 📝 文档更新
- 新增 [场景化使用指南](docs/USAGE_SCENARIOS.md)（5 个真实场景完整演示）
- 新增 [架构文档](docs/architecture.md)
- README 优化：增加目录导航、文档链接区块
- 合并 README_CN.md → README.md，统一单文件
- examples/README.md 增加场景文档跳转链接
- pyproject.toml requires-python 收紧至 >=3.10

### ✅ 测试扩展
- 新增 ResNet50 完整训练示例（train + evaluate_robustness）
- 测试用例扩充至 129 个

### 🛠️ 持续优化
- 代码评分器（Linter）4 维度、26 条规则完善
- msutils 工具库持续扩展（25 模块 / 5,664 行）
- API 示例注册表扩充至 34 个 API

---

## [1.0.0] - 2026-05-03

### 🎉 正式发布

#### 🔍 模型检索工具（核心）
| 工具名 | 说明 |
|--------|------|
| `list_models` | 按 group/category/task/suite 或关键词过滤 MindSpore 官方模型 |
| `get_model_info` | 获取单模型的完整信息（参数规模、精度指标、适用场景） |
| `recommend_models` | 自然语言智能推荐模型，支持 Ascend/GPU/CPU 硬件约束 |
| `compare_models` | 对比多个模型，帮助选择最适合的模型 |

#### 🗺️ API 映射工具
| 工具名 | 说明 |
|--------|------|
| `query_op_mapping` | 查询 PyTorch API 对应的 MindSpore 实现，支持模糊匹配 |
| `diagnose_translation` | 检查 PyTorch→MindSpore 代码翻译的完整性和正确性 |

#### 🛡️ AI 安全工具
| 工具名 | 说明 |
|--------|------|
| `generate_adversarial_attack` | 生成对抗攻击配置，支持 FGSM/PGD/DeepFool/CW/JSMA 五种攻击方法 |
| `evaluate_model_robustness` | 评估模型鲁棒性，生成评估配置和示例代码 |

#### 📊 数据处理工具
| 工具名 | 说明 |
|--------|------|
| `create_data_augmentation_pipeline` | 创建数据增强流水线，支持图像分类、目标检测、语义分割、NLP |

#### 🔧 训练工具
| 工具名 | 说明 |
|--------|------|
| `get_lr_scheduler` | 获取学习率调度器配置（余弦退火/阶梯衰减/多项式/One Cycle） |
| `get_training_callbacks` | 获取训练回调函数配置（检查点/早停/TensorBoard/梯度裁剪） |

#### 📈 模型分析工具
| 工具名 | 说明 |
|--------|------|
| `compute_model_complexity` | 计算模型 FLOPs、参数量、内存占用 |

#### 🌐 分布式训练
| 工具名 | 说明 |
|--------|------|
| `setup_distributed_training` | 配置分布式训练（DDP 多卡训练） |

#### 🚀 部署工具
| 工具名 | 说明 |
|--------|------|
| `quantize_model` | 模型量化配置（动态量化/静态量化/量化感知训练） |
| `convert_model_format` | 模型格式转换（PyTorch ↔ MindSpore ↔ ONNX） |

#### 🔍 代码评分器（Linter）
| 工具名 | 说明 |
|--------|------|
| `lint_mindspore_code` | MindSpore 代码质量评分（4 维度、26 条规则） |
| `get_lint_rules` | 获取所有检查规则详情 |
| `compare_code_snippets` | 对比两个代码片段的质量差异 |

#### 🚂 训练模板生成器
| 工具名 | 说明 |
|--------|------|
| `generate_training_template` | 一键生成完整训练脚本（模型+数据集+配置+回调） |
| `get_available_options` | 获取所有可用模型/数据集/硬件选项 |
| `generate_quick_start` | 生成适合新手的入门级训练脚本 |

#### 📖 API 示例生成器
| 工具名 | 说明 |
|--------|------|
| `get_api_examples` | 获取 MindSpore API 完整示例（描述/签名/参数/多示例代码） |
| `search_apis` | 模糊搜索相关 API（输入"卷积"自动匹配所有卷积相关 API） |
| `list_api_categories` | 列出所有 API 分类和数量 |
| `get_related_apis` | 获取相关 API 列表（如 Conv2d → BatchNorm2d、Dense） |
| `get_quick_reference` | 快速 API 参考（简洁版，适合快速查阅） |

#### 📚 MCP 资源
- `mindspore://models/official` - 官方模型完整清单（300+ 模型）
- `mindspore://opmap/pytorch/consistent` - 一致 API 映射
- `mindspore://opmap/pytorch/diff` - 差异 API 映射

#### 💬 MCP Prompt
- `model_lookup` - 按任务查找模型
- `model_recommend` - 智能模型推荐
- `model_compare` - 模型对比
- `migration_guide` - PyTorch→MindSpore 迁移指南
- `performance_optimize` - 性能优化建议

---

## [0.1.0] - 2026-03-20

### 🔧 初始开发版本

- MCP 服务框架搭建
- 模型列表加载与基础检索功能
- API 映射数据初始化
- 工具函数基础封装

---

## 更新日志编写规范

每次发布新版本时，请按以下格式更新本文件：

```markdown
## [X.Y.Z] - YYYY-MM-DD

### 🚀 新增功能
- 功能描述

### 🐛 修复问题
- 问题描述

### 🛠️ 优化改进
- 改进描述

### 📝 文档更新
- 文档描述
```

---

## 贡献者

- **whitea133** (张俊杰) - 暨南大学 · 网络空间安全学院

---

*此 Changelog 格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)*