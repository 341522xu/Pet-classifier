import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM热力图生成器"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


def denormalize(tensor):
    """把归一化后的张量还原成可显示的图片"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def plot_gradcam(correct_img, correct_label, correct_pred,
                 wrong_img, wrong_label, wrong_pred,
                 class_names, model, save_path):
    """生成2x2的Grad-CAM对比图并保存"""
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    cam_correct, _ = grad_cam.generate(correct_img, correct_pred)
    cam_wrong, _ = grad_cam.generate(wrong_img, wrong_pred)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    img_c = denormalize(correct_img)
    axes[0, 0].imshow(img_c)
    axes[0, 0].set_title(f'Correct: {class_names[correct_label]}\n(Pred: {class_names[correct_pred]})', fontsize=10)
    axes[0, 0].axis('off')
    axes[0, 1].imshow(img_c)
    axes[0, 1].imshow(cam_correct, cmap='jet', alpha=0.5)
    axes[0, 1].set_title('Grad-CAM', fontsize=10)
    axes[0, 1].axis('off')

    img_w = denormalize(wrong_img)
    axes[1, 0].imshow(img_w)
    axes[1, 0].set_title(f'Wrong: True={class_names[wrong_label]}\nPred={class_names[wrong_pred]}', fontsize=10)
    axes[1, 0].axis('off')
    axes[1, 1].imshow(img_w)
    axes[1, 1].imshow(cam_wrong, cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Grad-CAM', fontsize=10)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
