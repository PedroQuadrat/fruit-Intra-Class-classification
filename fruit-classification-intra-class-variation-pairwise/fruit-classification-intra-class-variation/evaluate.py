import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch import nn


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained fruit classifier.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with class subfolders; ideally an independent test set.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/fruit_classifier.pth"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = payload["class_names"]
    size = int(payload.get("image_size", 224))
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if dataset.classes != class_names:
        raise SystemExit(
            f"Class folders {dataset.classes} do not match checkpoint classes {class_names}."
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(1)
            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true class, columns=predicted class):")
    print(confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names)))))


if __name__ == "__main__":
    main()
