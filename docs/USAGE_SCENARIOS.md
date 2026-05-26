# mindspore-tools-mcp 场景化使用指南

> 本文通过 5 个真实场景，演示如何使用 mindspore-tools-mcp 的 29 个 MCP 工具完成从模型选型到部署的全流程任务。

---

## 场景 1：模型选型 — "我该用哪个 MindSpore 模型？"

### 背景

MindSpore 生态覆盖 260+ 个模型，横跨大语言模型、计算机视觉、OCR、强化学习、推荐系统、科学计算六大领域。面对这么多选择，开发者往往不知道该从何入手。

本场景演示如何通过 MCP 工具，用自然语言快速锁定目标模型。

---

### 步骤 1：用自然语言描述需求

向 AI 助手发送一条自然语言查询：

```
我需要做一个图像分类任务，数据集是 CIFAR-10，用 Ascend GPU 训练，
推荐一个精度高且参数量适中的模型。
```

MCP 工具 `model_lookup` 会在 MindSpore 模型库中进行语义匹配，返回最相关的候选模型。

### 步骤 2：获取推荐列表

调用 `recommend_models` 工具，指定任务类型和硬件约束：

```json
{
  "name": "recommend_models",
  "arguments": {
    "task": "图像分类",
    "hardware": "ascend",
    "criteria": "精度优先"
  }
}
```

**返回示例**（部分）：

| 模型 | 套件 | Top-1 精度 | 参数量 | 硬件 |
|------|------|-----------|--------|------|
| ResNet-152 | MindCV | 78.3% | 60.2M | Ascend/GPU |
| EfficientNet-B7 | MindCV | 80.4% | 66.0M | Ascend/GPU |
| SwinTransformer-Base | MindCV | 83.5% | 88.0M | Ascend/GPU |
| ConvNeXt-Base | MindCV | 83.8% | 88.6M | Ascend/GPU |

### 步骤 3：按领域浏览模型

如果想浏览某一领域下的所有模型，使用 `list_models`：

```json
{
  "name": "list_models",
  "arguments": {
    "group": "mindcv",
    "task": "image-classification"
  }
}
```

**返回**：MindCV 套件下所有图像分类模型，包含 VGG、ResNet、DenseNet、MobileNet、EfficientNet、ViT、SwinTransformer、ConvNeXt 等系列。

### 步骤 4：查看模型详情

选中某个模型后，调用 `get_model_info` 获取完整信息：

```json
{
  "name": "get_model_info",
  "arguments": {
    "model_name": "ResNet-50"
  }
}
```

**返回**：模型卡片链接、配置文件路径、变体列表（ResNet-18/34/50/101/152）、训练数据集、支持的硬件、精度指标。

### 步骤 5：多模型对比

使用 `model_compare` 对比多个候选模型：

```json
{
  "name": "model_compare",
  "arguments": {
    "models": ["ResNet-50", "MobileNet-V3", "EfficientNet-B0"],
    "metrics": ["accuracy", "params", "flops"]
  }
}
```

**对比结果**：

| 模型 | Top-1 精度 | 参数量 | FLOPs |
|------|-----------|--------|-------|
| EfficientNet-B0 | 77.1% | 5.3M | 0.4B |
| ResNet-50 | 76.1% | 25.6M | 4.1B |
| MobileNet-V3 | 75.2% | 5.4M | 0.2B |

> **结论**：如果追求精度，选 EfficientNet-B0；如果追求速度，选 MobileNet-V3；如果需要成熟生态，选 ResNet-50。

---

### 可用领域速查

| 领域套件 | 说明 | 代表模型 |
|---------|------|---------|
| **MindCV** | 计算机视觉 | ResNet, EfficientNet, ViT, SwinTransformer, ConvNeXt |
| **MindFormers** | 大语言模型 | LLaMA, GLM, Qwen, DeepSeek, InternLM, Baichuan2 |
| **MindOCR** | 文字识别 | DBNet, SVTR, CRNN, RARE |
| **MindRL** | 强化学习 | DQN, PPO, SAC, TD3 |
| **MindRec** | 推荐系统 | Wide&Deep, DCN |
| **MindScience** | 科学计算 | PINNs, DeepONet, MEGA-Fold, FNO |

---

### 涉及的 MCP 工具

| 工具 | 功能 |
|------|------|
| `model_lookup` | 自然语言模型查询 |
| `recommend_models` | 智能模型推荐 |
| `list_models` | 按条件列出模型 |
| `get_model_info` | 获取模型详情 |
| `model_compare` | 多模型对比分析 |

---
## 场景 2：PyTorch → MindSpore 迁移 — "我的 PyTorch 代码怎么搬到 MindSpore？"

### 背景

从 PyTorch 迁移到 MindSpore 是许多开发者的痛点：API 名称不同、张量操作差异、自动求导机制不同、训练循环写法不同。手动逐行翻译不仅耗时，还容易踩坑。

本场景演示如何用 MCP 工具实现**半自动迁移**：先查映射表，再自动诊断翻译错误，最后生成迁移指南。

---

### 步骤 1：查询 API 映射

遇到不熟悉的 PyTorch API 时，调用 `query_op_mapping`：

```json
{
  "name": "query_op_mapping",
  "arguments": {
    "pytorch_api": "torch.nn.CrossEntropyLoss"
  }
}
```

**返回**：

```json
{
  "pytorch": "torch.nn.CrossEntropyLoss",
  "mindspore": "mindspore.nn.CrossEntropyLoss",
  "note": "参数一致，可直接替换。注意 MindSpore 的 label 默认不需要 one-hot 编码。",
  "examples": [
    "# PyTorch\ncriterion = torch.nn.CrossEntropyLoss()\nloss = criterion(output, target)",
    "# MindSpore\ncriterion = mindspore.nn.CrossEntropyLoss()\nloss = criterion(output, target)"
  ]
}
```

**常用映射速查**：

| PyTorch | MindSpore | 差异说明 |
|---------|-----------|---------|
| `torch.tensor` | `mindspore.Tensor` | 一致 |
| `torch.nn.Module` | `mindspore.nn.Cell` | 基类名不同 |
| `torch.optim.Adam` | `mindspore.nn.Adam` | 在 nn 包下 |
| `torch.utils.data.DataLoader` | `mindspore.dataset.GeneratorDataset` | 接口完全不同 |
| `torch.nn.functional.relu` | `mindspore.ops.functional.relu` | 路径不同 |
| `model.train()` | `model.set_train()` | 方法名不同 |
| `model.eval()` | `model.set_train(False)` | 无 eval 方法 |
| `torch.no_grad()` | `mindspore.ops.composite.no_grad()` | 上下文管理器不同 |
| `torch.cuda.is_available()` | `mindspore.set_context(device_target='GPU')` | 硬件设置方式不同 |
| `torch.save(model)` | `mindspore.save_checkpoint(model)` | 保存 API 不同 |

---

### 步骤 2：翻译代码并诊断

把你翻译好的 MindSpore 代码提交给 `diagnose_translation` 进行自动诊断：

```json
{
  "name": "diagnose_translation",
  "arguments": {
    "mindspore_code": "import mindspore as ms\nfrom mindspore import nn, ops\n\nms.set_context(device_target='GPU')\n\nclass SimpleNet(nn.Cell):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Dense(784, 256)\n        self.fc2 = nn.Dense(256, 10)\n        self.relu = ops.ReLU()\n    def construct(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        return x\n\nmodel = SimpleNet()\nmodel.train()\noptimizer = nn.Adam(model.trainable_params(), lr=0.001)"
  }
}
```

**返回诊断报告**：

```
✅ 导入正确：mindspore, nn, ops
✅ 上下文设置正确
✅ nn.Cell 继承正确
⚠️ 第17行: model.train() 应改为 model.set_train()
⚠️ 第19行: nn.Adam 在 MindSpore 中需要通过 nn.optim.Adam 或 nn.Adam 使用，
   参数传递方式为 nn.Adam(params=params, learning_rate=0.001)

问题总数：0 错误 | 2 警告
建议修复率：100%
```

---

### 步骤 3：获取迁移指南

使用 Prompt 模式获取完整的迁移指导：

```
请使用 mindspore-tools-mcp 的 migration_guide 功能，
帮我将以下 PyTorch ResNet-50 图像分类代码迁移到 MindSpore。
```

**返回**：
- 完整的 MindSpore 版本 ResNet-50 实现
- 数据加载部分从 DataLoader 改为 GeneratorDataset
- 训练循环从 PyTorch 风格改为 MindSpore Model.fit 或手动循环
- 学习率调度器的对应写法
- 常见陷阱提示

---

### 完整迁移示例：一个 CIFAR-10 分类器

**迁移前（PyTorch）**：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**迁移后（MindSpore）**：

```python
import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import vision, transforms
from mindspore.dataset import Cifar10Dataset

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, pad_mode='pad', padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(32 * 16 * 16, 128)
        self.fc2 = nn.Dense(128, 10)
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
```

**关键差异对照**：

| 差异点 | PyTorch | MindSpore |
|--------|---------|-----------|
| 基类 | `nn.Module` | `nn.Cell` |
| 前向方法 | `forward()` | `construct()` |
| 池化 | `nn.MaxPool2d(2, 2)` | `nn.MaxPool2d(kernel_size=2, stride=2)` |
| 展平 | `x.view(-1)` | `nn.Flatten()` |
| 可训练参数 | `model.parameters()` | `model.trainable_params()` |
| 学习率 | `lr=0.001` | `learning_rate=0.001` |
| padding 模式 | 默认 zero padding | `pad_mode='pad'` 显式指定 |

---

### 涉及的 MCP 工具

| 工具 | 功能 |
|------|------|
| `query_op_mapping` | 查询 PyTorch ↔ MindSpore API 映射 |
| `diagnose_translation` | 诊断翻译后的 MindSpore 代码 |
| `migration_guide`（Prompt） | 生成完整迁移指南和示例代码 |

---
## 场景 3：对抗攻击研究 — "我的模型抗不抗打？"

### 背景

AI 安全是模型上线前必须验证的环节。一个看似 95% 精度的分类器，可能被几像素的扰动骗得体无完肤。MindSpore 官方提供了对抗攻击工具包，但配置复杂、上手门槛高。

本场景演示如何用 MCP 工具**一键生成对抗攻击代码**并**评估模型鲁棒性**，让安全验证变得像调 API 一样简单。

---

### 步骤 1：选择攻击方法并生成代码

调用 `generate_adversarial_attack` 生成对抗攻击代码：

```json
{
  "name": "generate_adversarial_attack",
  "arguments": {
    "attack_method": "FGSM",
    "target_model": "ResNet-50",
    "dataset": "CIFAR-10",
    "epsilon": 0.031
  }
}
```

**返回**：可直接运行的完整 Python 代码，包含：
- 模型加载与预处理
- FGSM 攻击实现（梯度符号 + epsilon 扰动）
- 攻击前后对比（原图 → 对抗样本 → 预测结果）
- 可视化输出

---

### 步骤 2：支持的攻击方法速查

| 攻击方法 | 类型 | 原理 | 适用场景 |
|---------|------|------|---------|
| **FGSM** | 白盒 | 梯度符号方向加扰动 | 快速基线测试 |
| **PGD** | 白盒 | 多步迭代 FGSM | 强攻击，评估鲁棒性上限 |
| **MI-FGSM** | 白盒 | 动量迭代 FGSM | 绕过部分防御 |
| **C&W** | 白盒 | 优化最小扰动 | 高质量对抗样本 |
| **DeepFool** | 白盒 | 迭代逼近决策边界 | 最小距离对抗样本 |
| **UAP** | 白盒 | 通用对抗扰动 | 批量攻击 |

---

### 步骤 3：完整 FGSM 攻击示例

以下是通过 MCP 工具生成的代码（精简版）：

```python
import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.dataset as ds
import numpy as np

# 加载预训练模型
model = MyResNet50()
model.set_train(False)

# FGSM 攻击
def fgsm_attack(model, image, label, epsilon=0.031):
    """FGSM: x' = x + epsilon * sign(∇_x J(θ, x, y))"""
    image_grad_fn = ms.ops.GradOperation(get_by_list=False)
    
    def compute_loss(img):
        logits = model(img)
        return nn.CrossEntropyLoss()(logits, label)
    
    grad = image_grad_fn(compute_loss)(image)
    perturbed_image = image + epsilon * ops.sign(grad)
    return ops.clip_by_value(perturbed_image, 0.0, 1.0)

# 执行攻击
test_loader = create_cifar10_loader(batch_size=1)
clean_acc = 0
attack_acc = 0
total = 0

for images, labels in test_loader:
    # 干净样本准确率
    pred = model(images).asnumpy().argmax(axis=1)
    clean_acc += (pred == labels.asnumpy()).sum()
    
    # 对抗样本准确率
    adv_images = fgsm_attack(model, images, labels, epsilon=0.031)
    adv_pred = model(adv_images).asnumpy().argmax(axis=1)
    attack_acc += (adv_pred == labels.asnumpy()).sum()
    total += len(labels)

print(f"干净样本准确率: {clean_acc/total:.2%}")
print(f"FGSM 攻击后准确率: {attack_acc/total:.2%}")
print(f"精度下降: {(clean_acc-attack_acc)/total:.2%}")
```

**典型输出**：

```
干净样本准确率: 93.52%
FGSM 攻击后准确率: 41.28%
精度下降: 52.24%
```

---

### 步骤 4：评估模型鲁棒性

使用 `evaluate_model_robustness` 进行全面的鲁棒性评估：

```json
{
  "name": "evaluate_model_robustness",
  "arguments": {
    "model_name": "ResNet-50",
    "dataset": "CIFAR-10",
    "attacks": ["FGSM", "PGD-10", "MI-FGSM"],
    "epsilons": [0.01, 0.031, 0.063]
  }
}
```

**返回鲁棒性报告**：

| 攻击方法 | ε=0.01 | ε=0.031 | ε=0.063 |
|---------|--------|---------|---------|
| FGSM | 85.3% | 41.3% | 12.7% |
| PGD-10 | 78.1% | 22.5% | 5.2% |
| MI-FGSM | 80.7% | 25.8% | 6.9% |

> **结论**：该模型对 FGSM（ε=0.01）有一定鲁棒性，但在 PGD-10（ε=0.031）下精度骤降至 22.5%，**建议进行对抗训练加固**。

---

### 步骤 5：对抗训练加固

除了攻击工具，mindspore-tools-mcp 还集成了防御方法：

```python
from msutils.defense import AdversarialTraining

# 在训练循环中加入对抗训练
def train_with_adv(model, loader, optimizer, epsilon=0.031):
    for data, target in loader:
        # 生成对抗样本
        adv_data = fgsm_attack(model, data, target, epsilon)
        
        # 用对抗样本训练
        logits = model(adv_data)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
```

**对抗训练前后对比**：

| 指标 | 普通训练 | 对抗训练 |
|------|---------|---------|
| 干净样本精度 | 93.5% | 89.2% |
| PGD-10（ε=0.031） | 22.5% | 58.7% |
| 精度-鲁棒性权衡 | ❌ 脆弱 | ✅ 均衡 |

---

### 涉及的 MCP 工具

| 工具 | 功能 |
|------|------|
| `generate_adversarial_attack` | 生成对抗攻击代码（支持 FGSM/PGD/MI-FGSM 等） |
| `evaluate_model_robustness` | 全面评估模型鲁棒性 |
| msutils `defense` 模块 | 对抗训练、防御蒸馏等防御方法 |

---
## 场景 4：完整训练流程 — "从零开始训练一个 MindSpore 模型"

### 背景

搭建一个完整的 MindSpore 训练流程涉及大量样板代码：数据加载、数据增强、学习率调度、损失函数、优化器、回调函数、模型保存、断点续训……每个环节都有 MindSpore 特有的 API 写法。

本场景演示如何用 MCP 工具**一键生成完整训练脚本**，从数据处理到模型部署，告别重复劳动。

---

### 步骤 1：生成训练模板

调用 `generate_training_template`，描述你的需求：

```json
{
  "name": "generate_training_template",
  "arguments": {
    "task": "image-classification",
    "model": "ResNet-50",
    "dataset": "CIFAR-10",
    "hardware": "GPU",
    "features": ["data_augmentation", "lr_scheduler", "checkpoint", "early_stopping", "mixed_precision"]
  }
}
```

**返回**：一个完整的、可直接运行的 Python 训练脚本，包含以下模块：

```
📄 train_resnet50_cifar10.py
├── 数据加载与增强（AutoAugment / RandomCrop / Normalize）
├── 模型定义（ResNet-50，预训练权重可选）
├── 损失函数（CrossEntropyLoss + Label Smoothing）
├── 优化器（AdamW + Weight Decay）
├── 学习率调度（CosineDecay + WarmUp）
├── 混合精度训练（AMP Level O2）
├── 回调函数（ModelCheckpoint / LossMonitor / EarlyStopping）
├── 断点续训（load_checkpoint）
└── 评估与指标（Accuracy / Top-5 Accuracy）
```

---

### 步骤 2：配置学习率调度

使用 `get_lr_scheduler` 获取学习率调度配置：

```json
{
  "name": "get_lr_scheduler",
  "arguments": {
    "scheduler": "cosine_decay",
    "total_epochs": 200,
    "warmup_epochs": 5,
    "base_lr": 0.001,
    "min_lr": 1e-6
  }
}
```

**返回代码**：

```python
from mindspore import nn

lr_scheduler = nn.cosine_decay_lr(
    min_lr=1e-6,
    max_lr=0.001,
    total_steps=200 * steps_per_epoch,
    step_per_epoch=steps_per_epoch,
    warmup_steps=5 * steps_per_epoch
)
```

**常用调度器对比**：

| 调度器 | 特点 | 适用场景 |
|--------|------|---------|
| `cosine_decay` | 平滑下降到最小值 | 通用，推荐首选 |
| `exponential_decay` | 指数衰减 | 简单快速实验 |
| `piecewise_constant` | 分段常数 | 精细控制学习阶段 |
| `polynomial_decay` | 多项式衰减 | 迁移学习微调 |
| `warmup + cosine` | 预热 + 余弦 | 大模型训练 |

---

### 步骤 3：获取数据增强策略

使用 `get_data_augmentation` 自动配置数据增强流水线：

```json
{
  "name": "get_data_augmentation",
  "arguments": {
    "dataset": "CIFAR-10",
    "task": "classification",
    "level": "advanced"
  }
}
```

**返回**：

```python
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

# 训练集增强
train_transforms = [
    vision.RandomCrop(32, padding=4),
    vision.RandomHorizontalFlip(prob=0.5),
    vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
    vision.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    vision.HWC2CHW()
]

# 验证集（仅归一化）
val_transforms = [
    vision.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    vision.HWC2CHW()
]
```

---

### 步骤 4：一键启动训练

生成的脚本支持多种启动方式：

```bash
# 基础训练
python train_resnet50_cifar10.py --epochs 200 --batch_size 128

# 混合精度训练（加速 1.5-2x）
python train_resnet50_cifar10.py --epochs 200 --batch_size 128 --amp

# 多卡训练（Ascend）
python train_resnet50_cifar10.py --epochs 200 --device_num 8

# 断点续训
python train_resnet50_cifar10.py --resume checkpoint/resnet50-epoch_100.ckpt

# 分布式训练
mpirun -n 4 python train_resnet50_cifar10.py --distribute
```

---

### 步骤 5：监控训练过程

通过回调函数实时监控：

```python
from mindspore import Model, Callback

class TrainingMonitor(Callback):
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        print(f"[Epoch {epoch}] Loss: {loss:.4f}")

monitor = TrainingMonitor()

model = Model(network, loss_fn, optimizer, metrics={"accuracy": nn.Accuracy()})
model.train(epochs, dataset, callbacks=[monitor, time_monitor, ckpt_cb])
```

**典型训练曲线**：

```
[Epoch   1] Loss: 2.1456  Acc: 18.3%
[Epoch  10] Loss: 1.2341  Acc: 56.7%
[Epoch  50] Loss: 0.4523  Acc: 84.2%
[Epoch 100] Loss: 0.2134  Acc: 91.8%
[Epoch 150] Loss: 0.1234  Acc: 93.5%
[Epoch 200] Loss: 0.0876  Acc: 94.2%  ← 最佳
```

---

### 完整训练脚本核心结构

```python
import mindspore as ms
from mindspore import nn, ops, Model, Tensor
from mindspore.train import LossMonitor, ModelCheckpoint, CheckpointConfig

# 1. 配置
ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')

# 2. 数据
train_dataset = create_dataset('cifar10', 'train', batch_size=128)
eval_dataset = create_dataset('cifar10', 'eval', batch_size=128)

# 3. 模型
network = ResNet50(num_classes=10)

# 4. 损失 + 优化器
loss_fn = nn.CrossEntropyLoss(smoothing=0.1)
optimizer = nn.AdamW(network.trainable_params(), learning_rate=lr_scheduler, weight_decay=5e-4)

# 5. 混合精度
network = ms.amp.auto_mixed_precision(network, amp_level='O2')

# 6. 回调
ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * 10,
                                keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix='resnet50', directory='./checkpoint',
                              config=ckpt_config)

# 7. 训练
model = Model(network, loss_fn, optimizer, metrics={'accuracy': nn.Accuracy()})
model.train(200, train_dataset,
            callbacks=[LossMonitor(0.01), ckpoint_cb],
            dataset_sink_mode=True)

# 8. 评估
result = model.eval(eval_dataset)
print(f"Eval Accuracy: {result['accuracy']:.2%}")
```

---

### 涉及的 MCP 工具

| 工具 | 功能 |
|------|------|
| `generate_training_template` | 一键生成完整训练脚本 |
| `get_lr_scheduler` | 获取学习率调度器配置 |
| `get_data_augmentation` | 获取数据增强策略 |
| `get_optimizer_config` | 获取优化器配置 |
| msutils `train` 模块 | 训练辅助函数和回调 |
| msutils `data` 模块 | 数据处理工具 |

---
## 场景 5：代码质量审查 — "我的 MindSpore 代码写得好不好？"

### 背景

写完一段 MindSpore 代码后，难免有疑虑：API 用法是否规范？有没有性能陷阱？代码风格是否符合社区惯例？手动逐行检查费时且容易遗漏。

本场景演示如何用 MCP 的 Linter 工具对 MindSpore 代码进行**自动评分、问题检测和改进建议**，像"代码审查官"一样帮你把关。

---

### 步骤 1：提交代码审查

调用 `lint_mindspore_code` 对你的代码进行全面检查：

```json
{
  "name": "lint_mindspore_code",
  "arguments": {
    "code": "import mindspore as ms\nfrom mindspore import nn\n\nms.set_context(device_target='GPU', mode=ms.PYNATIVE_MODE)\n\nclass MyModel(nn.Cell):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(3, 64, 3)\n        self.bn = nn.BatchNorm2d(64)\n        self.fc = nn.Dense(64*28*28, 10)\n\n    def construct(self, x):\n        x = self.conv(x)\n        x = self.bn(x)\n        x = ms.ops.relu(x)\n        x = ms.ops.max_pool2d(x, 2)\n        x = x.view(-1, 64*28*28)\n        x = self.fc(x)\n        return x\n\nnet = MyModel()\nnet.train()\nfor epoch in range(100):\n    for data, label in train_loader:\n        logits = net(data)\n        loss = nn.CrossEntropyLoss()(logits, label)\n        loss.backward()"
  }
}
```

---

### 步骤 2：获取审查报告

**返回完整的代码审查报告**：

```
═══════════════════════════════════════
  MindSpore 代码质量审查报告
  综合评分：62/100 ⚠️ 需改进
═══════════════════════════════════════

✅ 通过项 (3)
  [OK] 导入规范：正确使用 mindspore 命名空间
  [OK] nn.Cell 继承：正确继承 nn.Cell 基类
  [OK] 模型结构：Conv2d → BN → ReLU → Pool 布局合理

❌ 错误项 (3)
  [ERR-001] 第 4 行: PYNATIVE_MODE 用于训练效率较低，建议使用 GRAPH_MODE
  [ERR-002] 第 17 行: net.train() 已废弃，应使用 net.set_train()
  [ERR-003] 第 22 行: MindSpore 不支持 loss.backward()，
            应使用 nn.WithLossCell + nn.TrainOneStepCell 或自定义 GradOperation

⚠️ 警告项 (4)
  [WARN-001] 第 9 行: Conv2d 未指定 pad_mode，默认为 'valid'，
             建议显式指定 pad_mode='same' 或 'pad'
  [WARN-002] 第 14 行: ms.ops.relu() 建议用 nn.ReLU() 包装为 Cell，
             可获得更好的图优化效果
  [WARN-003] 第 17 行: x.view() 在 GRAPH_MODE 下不可用，
             应使用 x.reshape() 或 nn.Flatten()
  [WARN-004] 第 10 行: BatchNorm2d 在推理时需要调用 model.set_train(False)，
             确保更新 running_mean/running_var

💡 优化建议 (2)
  [OPT-001] 建议：在 Conv2d 和 BN 之间没有激活函数，
             Conv-BN-ReLU 是更好的实践顺序
  [OPT-002] 建议：训练循环建议封装为 Model.train()，
             支持 dataset_sink_mode 数据下沉加速

═══════════════════════════════════════
```

---

### 步骤 3：查看自动修复后的代码

审查工具不仅发现问题，还给出修复后的代码：

```python
import mindspore as ms
from mindspore import nn, ops

ms.set_context(device_target='GPU', mode=ms.GRAPH_MODE)  # [FIX] PYNATIVE → GRAPH

class MyModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, pad_mode='same')   # [FIX] 显式 pad_mode
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()                                # [FIX] 用 Cell 包装
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 16 * 16, 10)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)      # [FIX] Cell 调用代替 ops 函数
        x = self.pool(x)
        x = self.flatten(x)    # [FIX] reshape → Flatten
        x = self.fc(x)
        return x

# [FIX] 使用 Model API 进行训练
net = MyModel()
loss_fn = nn.WithLossCell(net, nn.CrossEntropyLoss())
train_net = nn.TrainOneStepCell(loss_fn, nn.Adam(net.trainable_params(), lr=0.001))
train_net.set_train()  # [FIX] train() → set_train()

model = ms.Model(train_net)
model.train(100, train_dataset, callbacks=[ms.LossMonitor()])
```

---

### 步骤 4：代码质量评分维度

Linter 从 6 个维度对代码进行评分：

| 维度 | 权重 | 说明 |
|------|------|------|
| **API 规范** | 25% | 是否使用正确的 MindSpore API，避免 PyTorch 混用 |
| **性能优化** | 20% | 是否利用图模式、数据下沉、混合精度等加速 |
| **代码风格** | 15% | 命名规范、注释质量、代码组织 |
| **错误处理** | 15% | 异常处理、边界条件、类型安全 |
| **最佳实践** | 15% | Conv-BN-ReLU 顺序、权重初始化、学习率选择 |
| **可维护性** | 10% | 模块化程度、代码复用、配置管理 |

**评分等级**：

| 分数 | 等级 | 说明 |
|------|------|------|
| 90-100 | 🟢 优秀 | 可直接用于生产 |
| 75-89 | 🔵 良好 | 少量改进即可上线 |
| 60-74 | 🟡 一般 | 存在问题，建议修复 |
| 0-59 | 🔴 待改进 | 需要较大重构 |

---

### 步骤 5：获取代码格式化

Linter 还支持自动格式化代码风格：

```json
{
  "name": "format_mindspore_code",
  "arguments": {
    "code": "import mindspore as ms;from mindspore import nn;x=ms.Tensor([1,2,3])",
    "style": "pep8"
  }
}
```

**返回**：

```python
import mindspore as ms
from mindspore import nn

x = ms.Tensor([1, 2, 3])
```

---

### 常见问题速查

| 问题 | 现象 | 修复 |
|------|------|------|
| `PYNATIVE_MODE` 训练慢 | 训练速度仅为 GRAPH 的 1/3 | 切换 `ms.GRAPH_MODE` |
| `loss.backward()` 报错 | RuntimeError | 改用 `nn.WithLossCell` + `GradOperation` |
| `model.train()` 无效 | 模型没切换训练状态 | 改用 `model.set_train()` |
| `x.view()` 报错 | GRAPH_MODE 不支持 | 改用 `x.reshape()` 或 `nn.Flatten()` |
| `torch.xxx` 混用 | 部分函数可运行但性能差 | 全部替换为 MindSpore API |
| 忘记 `set_train(False)` | BN 的 running stats 不更新 | 推理前调用 `set_train(False)` |

---

### 涉及的 MCP 工具

| 工具 | 功能 |
|------|------|
| `lint_mindspore_code` | 全面代码审查（评分 + 问题检测 + 修复建议） |
| `format_mindspore_code` | 代码自动格式化 |

---

## 总结

| 场景 | 核心工具 | 解决的问题 |
|------|---------|-----------|
| 1. 模型选型 | recommend_models, model_compare | 260+ 模型中快速找到最合适的 |
| 2. PyTorch 迁移 | query_op_mapping, diagnose_translation | 降低迁移门槛，减少踩坑 |
| 3. 对抗攻击 | generate_adversarial_attack, evaluate_model_robustness | 一键生成攻击代码，评估模型安全 |
| 4. 训练流程 | generate_training_template, get_lr_scheduler | 一键生成完整训练脚本 |
| 5. 代码审查 | lint_mindspore_code, format_mindspore_code | 自动检查代码质量，输出改进建议 |

> 💡 **提示**：所有工具均通过 MCP 协议接入，可在 Claude Desktop、Cursor、VS Code 等支持 MCP 的客户端中直接调用，也可以通过 `examples/` 目录下的独立脚本单独使用。

---

*本文档随项目持续更新，欢迎提交 Issue 和 PR 共同完善。*
