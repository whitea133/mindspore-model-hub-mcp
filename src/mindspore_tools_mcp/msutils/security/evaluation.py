"""
msutils.security.evaluation - 鲁棒性评估

评估模型在对抗攻击下的鲁棒性
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def evaluate_robustness(
    model,
    test_dataset,
    attack=None,
    attack_params: Optional[Dict] = None,
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    评估模型鲁棒性
    
    Args:
        model: 待评估模型
        test_dataset: 测试数据集
        attack: 攻击方法，默认 FGSM
        attack_params: 攻击参数
        num_samples: 评估样本数
    
    Returns:
        鲁棒性评估结果
    
    Example:
        >>> from mindspore_tools_mcp.msutils.security import evaluate_robustness, FGSM
        >>> attack = FGSM(model, epsilon=0.03)
        >>> results = evaluate_robustness(model, test_dataset, attack)
        >>> print(f"Clean accuracy: {results['clean_acc']:.2%}")
        >>> print(f"Adversarial accuracy: {results['adversarial_acc']:.2%}")
    """
    if attack_params is None:
        attack_params = {'epsilon': 0.03}
    
    # 如果没有指定攻击，使用 FGSM
    if attack is None:
        from mindspore_tools_mcp.msutils.security.attacks import FGSM
        attack = FGSM(model, **attack_params)
    
    # 评估清洁样本准确率
    clean_correct = 0
    total = 0
    
    for i, batch in enumerate(test_dataset):
        if i >= num_samples:
            break
        
        images, labels = batch
        images = images.asnumpy() if hasattr(images, 'asnumpy') else images
        labels = labels.asnumpy() if hasattr(labels, 'asnumpy') else labels
        
        # 清洁样本预测
        predictions = model(images)
        if hasattr(predictions, 'asnumpy'):
            predictions = predictions.asnumpy()
        predicted_labels = np.argmax(predictions, axis=1)
        
        clean_correct += np.sum(predicted_labels == labels)
        total += len(labels)
    
    clean_acc = clean_correct / total if total > 0 else 0
    
    # 评估对抗样本准确率
    adv_correct = 0
    total = 0
    
    for i, batch in enumerate(test_dataset):
        if i >= num_samples:
            break
        
        images, labels = batch
        images = images.asnumpy() if hasattr(images, 'asnumpy') else images
        labels = labels.asnumpy() if hasattr(labels, 'asnumpy') else labels
        
        # 生成对抗样本
        adversarial_images = attack.generate(images, labels)
        
        # 对抗样本预测
        predictions = model(adversarial_images)
        if hasattr(predictions, 'asnumpy'):
            predictions = predictions.asnumpy()
        predicted_labels = np.argmax(predictions, axis=1)
        
        adv_correct += np.sum(predicted_labels == labels)
        total += len(labels)
    
    adv_acc = adv_correct / total if total > 0 else 0
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'robustness_improvement': adv_acc - clean_acc,
        'num_samples': total
    }


def auto_attack(
    model,
    test_dataset,
    attacks: Optional[List] = None,
    epsilon: float = 0.03,
    num_samples: int = 1000
) -> Dict:
    """
    自动攻击评估
    
    使用多种攻击方法评估模型鲁棒性
    
    Args:
        model: 待评估模型
        test_dataset: 测试数据集
        attacks: 攻击方法列表，默认 [FGSM, PGD, BIM]
        epsilon: 扰动大小
        num_samples: 评估样本数
    
    Returns:
        各攻击方法的评估结果
    
    Example:
        >>> from mindspore_tools_mcp.msutils.security import auto_attack
        >>> results = auto_attack(model, test_dataset)
        >>> for attack_name, metrics in results.items():
        ...     print(f"{attack_name}: {metrics['adversarial_accuracy']:.2%}")
    """
    # 默认攻击方法
    if attacks is None:
        from mindspore_tools_mcp.msutils.security.attacks import FGSM, PGD, BIM
        attacks = [
            ('FGSM', FGSM(model, epsilon=epsilon)),
            ('PGD', PGD(model, epsilon=epsilon, alpha=epsilon/3, steps=10)),
            ('BIM', BIM(model, epsilon=epsilon))
        ]
    
    results = {}
    
    for attack_name, attack in attacks:
        print(f"Evaluating {attack_name}...")
        
        result = evaluate_robustness(
            model=model,
            test_dataset=test_dataset,
            attack=attack,
            num_samples=num_samples
        )
        
        results[attack_name] = result
        
        print(f"  Clean: {result['clean_accuracy']:.2%}")
        print(f"  {attack_name}: {result['adversarial_accuracy']:.2%}")
    
    return results


def perturbation_analysis(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    attack,
    epsilon_range: List[float] = None
) -> Dict:
    """
    扰动分析
    
    分析不同扰动大小对模型的影响
    
    Args:
        model: 待评估模型
        images: 输入图像
        labels: 真实标签
        attack: 攻击方法
        epsilon_range: 扰动大小范围
    
    Returns:
        各扰动大小下的准确率
    
    Example:
        >>> from msutils.security.attacks import FGSM
        >>> attack = FGSM(model)
        >>> results = perturbation_analysis(model, images, labels, attack)
        >>> print(results)
    """
    if epsilon_range is None:
        epsilon_range = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
    
    results = {'epsilon': [], 'accuracy': []}
    
    for epsilon in epsilon_range:
        # 设置攻击参数
        attack.epsilon = epsilon
        
        # 生成对抗样本
        adversarial_images = attack.generate(images, labels)
        
        # 预测
        predictions = model(adversarial_images)
        if hasattr(predictions, 'asnumpy'):
            predictions = predictions.asnumpy()
        predicted_labels = np.argmax(predictions, axis=1)
        
        # 计算准确率
        accuracy = np.mean(predicted_labels == labels)
        
        results['epsilon'].append(epsilon)
        results['accuracy'].append(accuracy)
    
    return results


def compute_adversarial_distance(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 10
) -> Dict:
    """
    计算对抗距离
    
    衡量从原始样本到决策边界的距离
    
    Args:
        model: 待评估模型
        images: 输入图像
        labels: 真实标签
        num_classes: 类别数
    
    Returns:
        对抗距离统计
    
    Example:
        >>> results = compute_adversarial_distance(model, images, labels)
        >>> print(f"Mean distance: {results['mean']:.4f}")
    """
    from mindspore import Tensor
    import mindspore.ops as ops
    
    distances = []
    
    for i in range(len(images)):
        image = images[i:i+1]
        label = labels[i]
        
        # 计算到其他类别的距离
        image_ms = Tensor(image.astype(np.float32))
        
        # 获取模型输出
        output = model(image_ms)
        output = output.asnumpy()[0] if hasattr(output, 'asnumpy') else np.array(output).flatten()
        
        # 计算 logit 差异
        target_logit = output[label]
        other_logits = np.delete(output, label)
        
        # 最小对抗距离（简化版本）
        min_distance = np.min(target_logit - other_logits)
        distances.append(min_distance)
    
    distances = np.array(distances)
    
    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'median': np.median(distances)
    }


def certify_robustness(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    radii: List[float] = None
) -> Dict:
    """
    认证鲁棒性
    
    认证模型在给定半径内的鲁棒性
    
    Args:
        model: 待评估模型
        images: 输入图像
        labels: 真实标签
        radii: 认证半径列表
    
    Returns:
        各半径下的认证准确率
    
    Example:
        >>> results = certify_robustness(model, images, labels)
        >>> print(f"Certified accuracy at r=0.1: {results[0.1]:.2%}")
    """
    if radii is None:
        radii = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    results = {}
    
    for radius in radii:
        certified = 0
        total = len(images)
        
        for i in range(total):
            image = images[i]
            label = labels[i]
            
            # 简化版认证：检查对抗距离
            from mindspore import Tensor
            output = model(Tensor(image.astype(np.float32)))
            output = output.asnumpy()[0] if hasattr(output, 'asnumpy') else np.array(output).flatten()
            predicted = np.argmax(output)
            
            if predicted == label:
                # 计算到决策边界的距离
                target_logit = output[label]
                other_max = np.max(np.delete(output, label))
                distance = target_logit - other_max
                
                if distance >= radius:
                    certified += 1
        
        results[radius] = certified / total if total > 0 else 0
    
    return results


# 导出所有函数
__all__ = [
    'evaluate_robustness',
    'auto_attack',
    'perturbation_analysis',
    'compute_adversarial_distance',
    'certify_robustness'
]
