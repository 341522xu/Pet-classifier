import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_dataloaders
from models.model import get_model


def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False, alpha=0.2):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha, device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            _, pred = outputs.max(1)
            correct += pred.eq(labels_a).sum().item()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet Training")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--save_name", type=str, default="best_model")
    parser.add_argument("--log_dir", type=str, default="runs/exp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, _ = get_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = get_model(num_classes=37, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=args.use_mixup, alpha=args.mixup_alpha)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/{args.save_name}.pth")

        print(f"Epoch [{epoch+1:2d}/{args.epochs}] "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} | "
              f"Best:{best_val_acc:.4f} | {time.time()-t0:.1f}s")

    writer.close()
    print(f"\nDone! Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: checkpoints/{args.save_name}.pth")


if __name__ == "__main__":
    main()
