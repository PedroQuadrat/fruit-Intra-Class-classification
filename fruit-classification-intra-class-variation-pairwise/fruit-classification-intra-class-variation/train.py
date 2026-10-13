import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class PairDataset(Dataset):
    """Generate same-class and different-class pairs from class-labeled folders."""
    def __init__(self, root, transform, indices=None, seed=42, pairs_per_image=2):
        self.root = Path(root)
        self.transform = transform
        self.rng = random.Random(seed)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        self.class_names = [p.name for p in class_dirs]
        self.samples = []
        for label, folder in enumerate(class_dirs):
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() in extensions:
                    self.samples.append((path, label))
        if len(self.class_names) < 2:
            raise ValueError(f"Need at least two class folders under {root}")
        if not self.samples:
            raise ValueError(f"No image files found under {root}")

        # Restrict the image list if a split is supplied.
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        self.by_class = {}
        for idx, (_, label) in enumerate(self.samples):
            self.by_class.setdefault(label, []).append(idx)
        self.valid_labels = [label for label, ids in self.by_class.items() if ids]
        if len(self.valid_labels) < 2:
            raise ValueError("Both train and validation split need images from at least two classes.")
        self.pairs_per_image = pairs_per_image

    def __len__(self):
        return len(self.samples) * self.pairs_per_image

    def __getitem__(self, index):
        first_idx = index % len(self.samples)
        first_path, first_label = self.samples[first_idx]
        rng = random.Random(index + self.rng.randrange(10**9))

        positive = rng.random() < 0.5 and len(self.by_class[first_label]) > 1
        if positive:
            options = [i for i in self.by_class[first_label] if i != first_idx]
            second_idx = rng.choice(options)
            target = 0.0  # same class => embeddings should be close
        else:
            other_labels = [lab for lab in self.valid_labels if lab != first_label]
            second_label = rng.choice(other_labels)
            second_idx = rng.choice(self.by_class[second_label])
            target = 1.0  # different class => embeddings should be far apart

        second_path, _ = self.samples[second_idx]
        try:
            image_a = Image.open(first_path).convert("RGB")
            image_b = Image.open(second_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not read image pair: {first_path}, {second_path}") from exc
        return self.transform(image_a), self.transform(image_b), torch.tensor(target, dtype=torch.float32)


class SiameseEncoder(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        input_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, image):
        vector = self.projection(self.backbone(image))
        return nn.functional.normalize(vector, p=2, dim=1)


def contrastive_loss(distance, target, margin=1.0):
    # target 0: same class; target 1: different class
    same_loss = (1.0 - target) * distance.pow(2)
    different_loss = target * torch.clamp(margin - distance, min=0.0).pow(2)
    return (same_loss + different_loss).mean()


def collect_distances(model, loader, device):
    distances, targets = [], []
    model.eval()
    with torch.inference_mode():
        for image_a, image_b, target in loader:
            image_a, image_b = image_a.to(device), image_b.to(device)
            emb_a, emb_b = model(image_a), model(image_b)
            dist = torch.linalg.vector_norm(emb_a - emb_b, dim=1)
            distances.extend(dist.cpu().numpy().tolist())
            targets.extend(target.numpy().tolist())
    return np.asarray(distances), np.asarray(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/fruits"))
    parser.add_argument("--output", type=Path, default=Path("models/siamese_fruit.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pairs-per-image", type=int, default=2)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Dataset not found: {args.data_dir}. See docs/dataset.md.")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_paths, labels, class_names = [], [], []
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for folder in sorted(p for p in args.data_dir.iterdir() if p.is_dir()):
        class_names.append(folder.name)
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in extensions:
                image_paths.append(path)
                labels.append(len(class_names) - 1)
    if len(class_names) < 2:
        raise SystemExit("Create at least two class folders, e.g. banana_prata/ and banana_dagua/.")
    if min([labels.count(i) for i in range(len(class_names))], default=0) < 2:
        raise SystemExit("Each class needs at least two images to generate same-class pairs.")

    all_indices = np.arange(len(image_paths))
    train_idx, val_idx = train_test_split(
        all_indices, test_size=args.val_fraction, random_state=args.seed, stratify=labels
    )

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # PairDataset uses the original global list, so provide a dedicated split-aware wrapper.
    class SplitPairDataset(Dataset):
        def __init__(self, indices, transform, seed):
            self.indices = list(indices)
            self.transform = transform
            self.rng = random.Random(seed)
            self.by_class = {}
            for global_idx in self.indices:
                self.by_class.setdefault(labels[global_idx], []).append(global_idx)
            self.classes = list(self.by_class)
        def __len__(self):
            return len(self.indices) * args.pairs_per_image
        def __getitem__(self, index):
            first_idx = self.indices[index % len(self.indices)]
            first_label = labels[first_idx]
            rng = random.Random(self.rng.randrange(10**9) + index)
            same = rng.random() < 0.5 and len(self.by_class[first_label]) > 1
            if same:
                candidates = [i for i in self.by_class[first_label] if i != first_idx]
                second_idx = rng.choice(candidates)
                target = 0.0
            else:
                other_labels = [lab for lab in self.classes if lab != first_label]
                second_label = rng.choice(other_labels)
                second_idx = rng.choice(self.by_class[second_label])
                target = 1.0
            def load(idx):
                with Image.open(image_paths[idx]) as im:
                    return self.transform(im.convert("RGB"))
            return load(first_idx), load(second_idx), torch.tensor(target, dtype=torch.float32)

    train_set = SplitPairDataset(train_idx, train_transform, args.seed)
    val_set = SplitPairDataset(val_idx, eval_transform, args.seed + 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SiameseEncoder(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_balanced = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for image_a, image_b, target in train_loader:
            image_a, image_b, target = image_a.to(device), image_b.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            emb_a, emb_b = model(image_a), model(image_b)
            distance = torch.linalg.vector_norm(emb_a - emb_b, dim=1)
            loss = contrastive_loss(distance, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_distances, val_targets = collect_distances(model, val_loader, device)
        # Choose threshold that maximizes validation balanced accuracy.
        candidates = np.unique(np.quantile(val_distances, np.linspace(0, 1, 101)))
        best_threshold, best_balanced = 0.5, -1.0
        for threshold in candidates:
            pred_diff = (val_distances >= threshold).astype(int)
            positive = val_targets == 1
            negative = val_targets == 0
            tpr = pred_diff[positive].mean() if positive.any() else 0
            tnr = (pred_diff[negative] == 0).mean() if negative.any() else 0
            balanced = (tpr + tnr) / 2
            if balanced > best_balanced:
                best_balanced, best_threshold = balanced, float(threshold)
        val_pred = (val_distances >= best_threshold).astype(int)
        val_acc = float((val_pred == val_targets).mean())
        train_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_pair_acc={val_acc:.3f} | val_balanced_acc={best_balanced:.3f} "
            f"| threshold={best_threshold:.4f}"
        )

        if best_balanced > best_val_balanced:
            best_val_balanced = best_balanced
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "embedding_dim": 128,
                "distance_threshold": best_threshold,
                "architecture": "siamese_efficientnet_b0",
            }, args.output)
            print(f"Saved checkpoint: {args.output}")

    print("Training complete. Use an independent, specimen-separated test set for final scientific metrics.")


if __name__ == "__main__":
    main()
