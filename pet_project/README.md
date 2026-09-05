# 基于深度学习的牛津宠物细粒度分类

**姓名**：xuqiang
**学号**：W125301182  
**GitHub**：https://github.com/xuqiang/pet-classifier

## 项目简介

本项目基于 ResNet-18 预训练模型，在 Oxford-IIIT Pet 数据集（37类猫狗品种，约7349张图片）上完成细粒度图像分类任务。通过两组单变量消融实验（Label Smoothing 和 MixUp）验证正则化方法对细粒度分类的效果，并使用混淆矩阵和 Grad-CAM 热力图进行模型分析。

## 数据集

- **Oxford-IIIT Pet**：37个品种（25种狗 + 12种猫），约7349张图片
- **划分方式**：70% 训练 / 15% 验证 / 15% 测试，分层抽样（Stratified Split）
- **数据增强**：训练集 Resize(256) + RandomCrop(224) + RandomHorizontalFlip + Normalize；验证/测试集 Resize(256) + CenterCrop(224) + Normalize

## 环境依赖

```bash
pip install -r requirements.txt
