# Architecture: pairwise fruit image comparison

## Objective

Given two images, estimate whether their labels correspond to the same known variety or different known varieties. This is a pairwise similarity task, not merely single-image classification.

## Model

A Siamese-style encoder shares the same EfficientNet-B0 feature extractor and projection head between both images. Each image becomes a normalized embedding vector. Euclidean distance between vectors is the comparison score.

## Training

- Positive pair: two images from the same class, target `0`.
- Negative pair: images from different classes, target `1`.
- Contrastive loss encourages positive pairs to have smaller distances and negative pairs to exceed a margin.
- A distance threshold is selected using validation pairs to maximize balanced accuracy.

The threshold is a dataset-dependent operating point, not a universal value. The current starter script uses a random image-level split for convenience; a specimen/session-level split is needed for defensible scientific evaluation.

## Inference

The Streamlit app preprocesses two images identically, computes embeddings, measures their Euclidean distance, and compares the distance with the saved threshold.

## Research extensions

- Compare with a simple pretrained-feature cosine-distance baseline.
- Test ResNet-50, EfficientNet-B0, and MobileNetV3.
- Add Grad-CAM or other explanations for visual inspection.
- Evaluate on new capture sessions and unseen individual fruits.
- Add an abstention/unknown result when the distance is outside a validated operating range.
