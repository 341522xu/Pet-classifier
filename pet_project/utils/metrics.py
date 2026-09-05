import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


def compute_metrics(all_labels, all_preds, all_probs):
    """计算Top-1 Acc、Top-5 Acc、Macro-F1"""
    top1 = accuracy_score(all_labels, all_preds)
    top5_correct = sum(1 for i in range(len(all_labels))
                        if all_labels[i] in np.argsort(all_probs[i])[-5:])
    top5 = top5_correct / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return top1, top5, macro_f1


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path, title="Confusion Matrix"):
    """绘制混淆矩阵热力图并保存"""
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True Label', xlabel='Predicted Label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def get_confused_pairs(cm, class_names, top_k=5):
    """找出最易混淆的Top-K品种对"""
    cm_nodiag = cm.copy()
    np.fill_diagonal(cm_nodiag, 0)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm_nodiag[i][j] > 0:
                pairs.append((class_names[i], class_names[j], cm_nodiag[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
