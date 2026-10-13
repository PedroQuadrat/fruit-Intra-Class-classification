import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torchvision import models, transforms


class SiameseEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
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
        return nn.functional.normalize(self.projection(self.backbone(image)), p=2, dim=1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate same-class vs different-class image pairs."
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with one subfolder per class, ideally an independent test set.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/siamese_fruit.pth"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=10000,
                        help="Maximum number of pairs to evaluate (0 means all pairs).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"Test dataset folder not found: {args.data_dir}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = payload["class_names"]
    image_size = args.image_size or int(payload.get("image_size", 224))
    threshold = float(payload["distance_threshold"])

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths, labels = [], []
    for label, class_name in enumerate(class_names):
        folder = args.data_dir / class_name
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder in test data: {folder}")
        class_paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in extensions)
        if not class_paths:
            raise SystemExit(f"No images found in class folder: {folder}")
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))

    if len(class_names) < 2:
        raise SystemExit("The checkpoint must contain at least two classes.")

    # Build candidate pairs with ground truth: 0 = same class, 1 = different class.
    all_pairs = []
    for i, j in itertools.combinations(range(len(paths)), 2):
        target = 0 if labels[i] == labels[j] else 1
        all_pairs.append((i, j, target))
    if args.max_pairs > 0 and len(all_pairs) > args.max_pairs:
        rng = random.Random(args.seed)
        all_pairs = rng.sample(all_pairs, args.max_pairs)

    model = SiameseEncoder(payload.get("embedding_dim", 128))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Encode each image once, then compare embeddings for every selected pair.
    embeddings = []
    with torch.inference_mode():
        for path in paths:
            try:
                with Image.open(path) as im:
                    tensor = transform(im.convert("RGB")).unsqueeze(0)
            except Exception as exc:
                raise RuntimeError(f"Could not read image: {path}") from exc
            embeddings.append(model(tensor).squeeze(0))
    embeddings = torch.stack(embeddings)

    y_true, y_pred = [], []
    for i, j, target in all_pairs:
        distance = torch.linalg.vector_norm(embeddings[i] - embeddings[j]).item()
        prediction = 1 if distance >= threshold else 0
        y_true.append(target)
        y_pred.append(prediction)

    if not y_true:
        raise SystemExit("No image pairs were generated.")
    print(f"Images: {len(paths)} | Pairs evaluated: {len(y_true)}")
    print(f"Decision threshold: {threshold:.6f}")
    print(f"Pair accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print("Class labels: 0 = same class, 1 = different classes")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["same_class", "different_class"],
                                zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; labels [0, 1]):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
