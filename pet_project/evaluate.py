import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import get_dataloaders
from models.model import get_model
from utils.metrics import compute_metrics, plot_confusion_matrix, get_confused_pairs
from utils.gradcam import plot_gradcam


@torch.no_grad()
def inference(model, loader, device):
    """在测试集上推理，返回预测、标签和概率"""
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, pred = outputs.max(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def pick_samples(model, loader, device):
    """从测试集选1张预测正确+1张预测错误的图"""
    correct_img, correct_label, correct_pred = None, None, None
    wrong_img, wrong_label, wrong_pred = None, None, None
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(images.size(0)):
                if correct_img is None and preds[i] == labels[i]:
                    correct_img = images[i:i+1]
                    correct_label = labels[i].item()
                    correct_pred = preds[i].item()
                if wrong_img is None and preds[i] != labels[i]:
                    wrong_img = images[i:i+1]
                    wrong_label = labels[i].item()
                    wrong_pred = preds[i].item()
                if correct_img is not None and wrong_img is not None:
                    return correct_img, correct_label, correct_pred, wrong_img, wrong_label, wrong_pred
    return correct_img, correct_label, correct_pred, wrong_img, wrong_label, wrong_pred


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--title", type=str, default="Confusion Matrix")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    _, _, test_loader, class_names = get_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = get_model(num_classes=37, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # 推理
    all_preds, all_labels, all_probs = inference(model, test_loader, device)
    top1, top5, macro_f1 = compute_metrics(all_labels, all_preds, all_probs)
    print(f"Top-1 Acc: {top1*100:.2f}%")
    print(f"Top-5 Acc: {top5*100:.2f}%")
    print(f"Macro-F1:  {macro_f1:.4f}")

    # 混淆矩阵
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm = plot_confusion_matrix(all_labels, all_preds, class_names, cm_path, title=args.title)
    print(f"Confusion matrix saved to: {cm_path}")

    # 最易混淆品种对
    pairs = get_confused_pairs(cm, class_names, top_k=5)
    print("\nTop-5 confused pairs:")
    for true_name, pred_name, cnt in pairs:
        print(f"  {true_name} -> {pred_name}: {cnt}")

    # Grad-CAM
    correct_img, correct_label, correct_pred, wrong_img, wrong_label, wrong_pred = pick_samples(model, test_loader, device)
    if correct_img is not None and wrong_img is not None:
        gradcam_path = os.path.join(args.output_dir, "gradcam.png")
        plot_gradcam(correct_img, correct_label, correct_pred,
                     wrong_img, wrong_label, wrong_pred,
                     class_names, model, gradcam_path)
        print(f"\nGrad-CAM saved to: {gradcam_path}")


if __name__ == "__main__":
    main()
