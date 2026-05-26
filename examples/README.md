# MindSpore Tools MCP 使用示例

本文档提供 `mindspore-tools-mcp` 的完整使用示例，涵盖所有 MCP 工具的实际应用场景。

---

> 💡 更多真实场景演示（模型选型、PyTorch 迁移、对抗攻击、训练流程、代码审查）请查看 **[📖 场景化使用指南](../docs/USAGE_SCENARIOS.md)**

---

## 📁 示例文件列表

### 模型检索示例
| 文件 | 说明 |
|------|------|
| `01_model_discovery.py` | 如何发现和选择合适的 MindSpore 模型 |
| `02_model_recommendation.py` | 智能模型推荐系统使用 |
| `03_model_comparison.py` | 多模型对比分析 |

### API 迁移示例
| 文件 | 说明 |
|------|------|
| `04_pytorch_to_mindspore.py` | PyTorch → MindSpore 代码转换 |
| `05_api_mapping_checker.py` | API 映射查询与验证 |

### AI 安全示例
| 文件 | 说明 |
|------|------|
| `06_adversarial_attack.py` | 对抗攻击配置与生成 |
| `07_robustness_evaluation.py` | 模型鲁棒性评估 |

### 训练工具示例
| 文件 | 说明 |
|------|------|
| `08_data_augmentation.py` | 数据增强流水线配置 |
| `09_lr_scheduler.py` | 学习率调度器使用 |
| `10_training_callbacks.py` | 训练回调函数配置 |

### 部署优化示例
| 文件 | 说明 |
|------|------|
| `11_model_quantization.py` | 模型量化配置 |
| `12_model_conversion.py` | 模型格式转换 |
| `13_distributed_training.py` | 分布式训练配置 |

### 代码评分器示例 🆕
| 文件 | 说明 |
|------|------|
| `15_code_linter.py` | MindSpore 代码质量评分 |

### API 示例生成器示例 🆕
| 文件 | 说明 |
|------|------|
| `17_api_examples.py` | MindSpore API 示例查询 |

### 训练模板生成器示例 🆕
| 文件 | 说明 |
|------|------|
| `16_training_template.py` | 一键生成训练脚本 |

### API 示例生成器示例 🆕
| 文件 | 说明 |
|------|------|
| `17_api_examples.py` | MindSpore API 示例查询 |

### 分析工具示例
| 文件 | 说明 |
|------|------|
| `14_model_complexity.py` | 模型复杂度分析 |

---

## 🚀 快速开始

### 方式一：通过 MCP 客户端调用

```json
// MCP 客户端配置 (如 Cline)
{
  "mindspore_tools_mcp": {
    "command": "uv",
    "args": [
      "--directory",
      "E:/CodeProject/mindspore-tools-mcp",
      "run",
      "python",
      "-m",
      "mindspore_tools_mcp.server"
    ]
  }
}
```

### 方式二：直接导入使用

```python
from mindspore_tools_mcp import tools
from mindspore_tools_mcp import msutils_tools

# 模型查询
models = tools.list_models(task="image-classification")
print(models)

# 对抗攻击配置
attack_config = msutils_tools.generate_adversarial_attack("fgsm", epsilon=0.1)
print(attack_config['code_example'])
```

---

## 💡 常见工作流

### 工作流 1：选择模型 → 训练 → 部署

```
1. recommend_models("图像分类")     → 推荐模型
2. compute_model_complexity()       → 分析复杂度
3. create_data_augmentation_pipeline()  → 配置数据增强
4. get_lr_scheduler()              → 学习率配置
5. get_training_callbacks()        → 回调配置
6. quantize_model()                → 模型量化
7. convert_model_format()          → 导出部署
```

### 工作流 2：PyTorch 迁移

```
1. query_op_mapping("torch.xxx")    → 查找 API 映射
2. diagnose_translation(py_code, ms_code)  → 检查翻译完整性
3. compute_model_complexity()       → 对比模型大小
```

### 工作流 3：AI 安全研究

```
1. recommend_models("图像分类")     → 选择目标模型
2. generate_adversarial_attack("pgd")  → 生成攻击配置
3. evaluate_model_robustness()      → 评估鲁棒性
4. quantize_model()                → 部署防御后的模型
```

---

**开始探索示例 → [01_model_discovery.py](01_model_discovery.py)**
