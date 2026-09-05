import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split


class TransformSubset(Dataset):
    """给子集套不同transform的包装类"""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, target = self.dataset[self.indices[i]]
        return self.transform(img), target


def get_dataloaders(root="./data", batch_size=32, num_workers=2, seed=42):
    """加载Oxford-IIIT Pet，按70/15/15分层划分，返回三个DataLoader"""
    # 加载trainval和test两部分，合并成完整数据集
    ds_trainval = datasets.OxfordIIITPet(root=root, download=True, split="trainval")
    ds_test = datasets.OxfordIIITPet(root=root, download=True, split="test")
    dataset = ConcatDataset([ds_trainval, ds_test])
    all_labels = np.array(list(ds_trainval._labels) + list(ds_test._labels))
    valid_idx = np.where(all_labels != -1)[0]

    # 70/15/15 分层划分
    train_val, test_idx = train_test_split(
        valid_idx, test_size=0.15, stratify=all_labels[valid_idx], random_state=seed)
    train_idx, val_idx = train_test_split(
        train_val, test_size=0.15/0.85, stratify=all_labels[train_val], random_state=seed)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TransformSubset(dataset, train_idx, train_transform)
    val_ds   = TransformSubset(dataset, val_idx,   eval_transform)
    test_ds  = TransformSubset(dataset, test_idx,  eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    class_names = ds_trainval.classes
    return train_loader, val_loader, test_loader, class_names
