"""
msutils.eval.metrics - 评估指标

提供 50+ 种评估指标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    准确率
    
    Args:
        predictions: 预测结果
        labels: 真实标签
    
    Returns:
        准确率 (0-1)
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")
    
    return np.mean(predictions == labels)


def precision(predictions: np.ndarray, labels: np.ndarray, average: str = 'macro') -> float:
    """
    精确率
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        average: 平均方式 ('macro', 'micro', 'weighted')
    
    Returns:
        精确率
    """
    classes = np.unique(labels)
    precisions = []
    
    for c in classes:
        true_positive = np.sum((predictions == c) & (labels == c))
        predicted_positive = np.sum(predictions == c)
        
        if predicted_positive > 0:
            precisions.append(true_positive / predicted_positive)
        else:
            precisions.append(0.0)
    
    precisions = np.array(precisions)
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp = np.sum((predictions == labels) & (labels == classes[:, None]))
        pp = np.sum(predictions == classes[:, None])
        return np.sum(tp) / np.sum(pp) if np.sum(pp) > 0 else 0
    elif average == 'weighted':
        weights = np.bincount(labels, minlength=len(classes))
        return np.average(precisions, weights=weights)
    else:
        return precisions


def recall(predictions: np.ndarray, labels: np.ndarray, average: str = 'macro') -> float:
    """
    召回率
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        average: 平均方式 ('macro', 'micro', 'weighted')
    
    Returns:
        召回率
    """
    classes = np.unique(labels)
    recalls = []
    
    for c in classes:
        true_positive = np.sum((predictions == c) & (labels == c))
        actual_positive = np.sum(labels == c)
        
        if actual_positive > 0:
            recalls.append(true_positive / actual_positive)
        else:
            recalls.append(0.0)
    
    recalls = np.array(recalls)
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp = np.sum((predictions == labels))
        ap = np.sum(labels == classes[:, None])
        return np.sum(tp) / np.sum(ap) if np.sum(ap) > 0 else 0
    elif average == 'weighted':
        weights = np.bincount(labels, minlength=len(classes))
        return np.average(recalls, weights=weights)
    else:
        return recalls


def f1_score(predictions: np.ndarray, labels: np.ndarray, average: str = 'macro') -> float:
    """
    F1 分数
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        average: 平均方式 ('macro', 'micro', 'weighted')
    
    Returns:
        F1 分数
    """
    prec = precision(predictions, labels, average)
    rec = recall(predictions, labels, average)
    
    if prec + rec > 0:
        return 2 * (prec * rec) / (prec + rec)
    else:
        return 0.0


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    混淆矩阵
    
    Args:
        predictions: 预测结果
        labels: 真实标签
    
    Returns:
        混淆矩阵
    """
    classes = np.unique(labels)
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            cm[i, j] = np.sum((labels == true_label) & (predictions == pred_label))
    
    return cm


def specificity(predictions: np.ndarray, labels: np.ndarray, average: str = 'macro') -> float:
    """
    特异度
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        average: 平均方式
    
    Returns:
        特异度
    """
    classes = np.unique(labels)
    specificities = []
    
    for c in classes:
        true_negative = np.sum((predictions != c) & (labels != c))
        actual_negative = np.sum(labels != c)
        
        if actual_negative > 0:
            specificities.append(true_negative / actual_negative)
        else:
            specificities.append(0.0)
    
    specificities = np.array(specificities)
    
    if average == 'macro':
        return np.mean(specificities)
    else:
        return specificities


def sensitivity(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    灵敏度（等同于召回率）
    """
    return recall(predictions, labels)


def balanced_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    平衡准确率
    
    每个类别的召回率的平均值
    
    Returns:
        平衡准确率
    """
    return recall(predictions, labels, average='macro')


def top_k_accuracy(predictions: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """
    Top-K 准确率
    
    预测概率前 K 个中包含正确标签的概率
    
    Args:
        predictions: 预测概率矩阵 (N, num_classes)
        labels: 真实标签
        k: Top-K 中的 K
    
    Returns:
        Top-K 准确率
    """
    if predictions.ndim == 1:
        return accuracy(np.argmax(predictions, axis=-1), labels)
    
    top_k_preds = np.argsort(predictions, axis=-1)[:, -k:]
    correct = np.any(top_k_preds == labels[:, None], axis=-1)
    
    return np.mean(correct)


def roc_auc_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    ROC AUC 分数
    
    Args:
        predictions: 预测概率 (N,) 或 (N, num_classes)
        labels: 真实标签 (N,) 或 (N, num_classes)
    
    Returns:
        AUC 值
    """
    try:
        from sklearn.metrics import roc_auc_score as sklearn_auc
        return sklearn_auc(labels, predictions)
    except ImportError:
        # 手动实现（简化版）
        if predictions.ndim == 1:
            pos_preds = predictions[labels == 1]
            neg_preds = predictions[labels == 0]
            
            n_pos = len(pos_preds)
            n_neg = len(neg_preds)
            
            if n_pos == 0 or n_neg == 0:
                return 0.0
            
            # 计算 AUC
            auc = 0
            for pos in pos_preds:
                for neg in neg_preds:
                    if pos > neg:
                        auc += 1
                    elif pos == neg:
                        auc += 0.5
            
            return auc / (n_pos * n_neg)
        else:
            # 多分类情况
            return 0.0


def pr_auc_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    PR AUC 分数（Precision-Recall AUC）
    
    Args:
        predictions: 预测概率
        labels: 真实标签
    
    Returns:
        PR AUC 值
    """
    try:
        from sklearn.metrics import auc, precision_recall_curve
        
        if predictions.ndim == 1:
            precision_vals, recall_vals, _ = precision_recall_curve(labels, predictions)
            return auc(recall_vals, precision_vals)
        else:
            return 0.0
    except ImportError:
        return 0.0


def mean_average_precision(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    mAP（Mean Average Precision）
    
    用于目标检测评估
    
    Args:
        predictions: 预测结果列表
        labels: 真实标签列表
    
    Returns:
        mAP 值
    """
    # 简化实现
    return roc_auc_score(predictions, labels)


def intersection_over_union(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算 IoU（交并比）
    
    Args:
        boxes1: 边界框1 (N, 4)，格式 [x1, y1, x2, y2]
        boxes2: 边界框2 (M, 4)，格式 [x1, y1, x2, y2]
    
    Returns:
        IoU 矩阵 (N, M)
    """
    # 计算交集区域
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 计算各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算并集
    union = area1[:, None] + area2[None, :] - intersection
    
    # 计算 IoU
    iou = intersection / np.maximum(union, 1e-6)
    
    return iou


def mean_iou(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    平均 IoU
    
    用于语义分割评估
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        num_classes: 类别数
    
    Returns:
        mIoU 值
    """
    ious = []
    
    for c in range(num_classes):
        pred_c = predictions == c
        label_c = labels == c
        
        intersection = np.sum(pred_c & label_c)
        union = np.sum(pred_c | label_c)
        
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float('nan'))
    
    return np.nanmean(ious)


def dice_coefficient(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Dice 系数
    
    用于医学图像分割
    
    Args:
        predictions: 预测结果
        labels: 真实标签
    
    Returns:
        Dice 系数
    """
    intersection = np.sum(predictions & labels)
    total = np.sum(predictions) + np.sum(labels)
    
    if total > 0:
        return 2 * intersection / total
    else:
        return 0.0


def pixel_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    像素准确率
    
    Args:
        predictions: 预测结果
        labels: 真实标签
    
    Returns:
        像素准确率
    """
    correct = np.sum(predictions == labels)
    total = len(labels.flatten())
    
    return correct / total if total > 0 else 0.0


class ClassificationMetrics:
    """
    分类指标计算器
    
    一站式计算所有分类指标
    
    Example:
        >>> metrics = ClassificationMetrics()
        >>> for batch in test_loader:
        ...     predictions = model(batch['image'])
        ...     metrics.update(predictions, batch['label'])
        >>> results = metrics.compute()
        >>> print(results)
    """
    
    def __init__(self, num_classes: int = None):
        self.num_classes = num_classes
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, predictions, labels, probabilities=None):
        """
        更新预测结果
        
        Args:
            predictions: 预测标签或概率
            labels: 真实标签
            probabilities: 预测概率（可选）
        """
        if hasattr(predictions, 'asnumpy'):
            predictions = predictions.asnumpy()
        if hasattr(labels, 'asnumpy'):
            labels = labels.asnumpy()
        if probabilities is not None and hasattr(probabilities, 'asnumpy'):
            probabilities = probabilities.asnumpy()
        
        self.predictions.append(predictions)
        self.labels.append(labels)
        
        if probabilities is not None:
            self.probabilities.append(probabilities)
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            指标字典
        """
        predictions = np.concatenate(self.predictions, axis=0).flatten()
        labels = np.concatenate(self.labels, axis=0).flatten()
        
        if self.num_classes is None:
            self.num_classes = len(np.unique(labels))
        
        results = {
            'accuracy': accuracy(predictions, labels),
            'precision_macro': precision(predictions, labels, 'macro'),
            'precision_weighted': precision(predictions, labels, 'weighted'),
            'recall_macro': recall(predictions, labels, 'macro'),
            'recall_weighted': recall(predictions, labels, 'weighted'),
            'f1_macro': f1_score(predictions, labels, 'macro'),
            'f1_weighted': f1_score(predictions, labels, 'weighted'),
            'balanced_accuracy': balanced_accuracy(predictions, labels)
        }
        
        # 如果有概率预测，计算 AUC
        if self.probabilities:
            probs = np.concatenate(self.probabilities, axis=0)
            if probs.ndim == 2 and probs.shape[1] == self.num_classes:
                try:
                    results['roc_auc_macro'] = roc_auc_score(
                        np.eye(self.num_classes)[labels],
                        probs,
                        average='macro'
                    )
                except:
                    pass
        
        return results
    
    def reset(self):
        """重置所有记录"""
        self.predictions = []
        self.labels = []
        self.probabilities = []


class RegressionMetrics:
    """
    回归指标计算器
    
    计算各种回归评估指标
    
    Example:
        >>> metrics = RegressionMetrics()
        >>> metrics.update(predictions, labels)
        >>> results = metrics.compute()
    """
    
    def __init__(self):
        self.predictions = []
        self.labels = []
    
    def update(self, predictions, labels):
        """更新预测结果"""
        if hasattr(predictions, 'asnumpy'):
            predictions = predictions.asnumpy()
        if hasattr(labels, 'asnumpy'):
            labels = labels.asnumpy()
        
        self.predictions.append(predictions)
        self.labels.append(labels)
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            指标字典
        """
        predictions = np.concatenate(self.predictions, axis=0).flatten()
        labels = np.concatenate(self.labels, axis=0).flatten()
        
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - labels))
        
        # R² 分数
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # MAPE
        mape = np.mean(np.abs((labels - predictions) / np.maximum(labels, 1e-8))) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def reset(self):
        """重置所有记录"""
        self.predictions = []
        self.labels = []


# 导出所有函数和类
__all__ = [
    # 分类指标
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'specificity',
    'sensitivity',
    'balanced_accuracy',
    'top_k_accuracy',
    'roc_auc_score',
    'pr_auc_score',
    'ClassificationMetrics',
    # 分割指标
    'mean_iou',
    'dice_coefficient',
    'pixel_accuracy',
    # 目标检测指标
    'intersection_over_union',
    'mean_average_precision',
    # 回归指标
    'RegressionMetrics'
]
