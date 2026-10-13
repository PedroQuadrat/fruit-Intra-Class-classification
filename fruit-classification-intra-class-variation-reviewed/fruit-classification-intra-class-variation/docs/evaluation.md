# Evaluation protocol

Do not claim that the model distinguishes banana varieties until it has been tested on independent, correctly labeled images.

## Report

- Number of unique fruit specimens and images per class.
- How labels were confirmed.
- Pair construction and split protocol.
- Pair accuracy, balanced accuracy, precision, recall, F1-score, and confusion matrix.
- False-positive pairs (different varieties predicted same) and false-negative pairs (same variety predicted different).
- Performance under different lighting, backgrounds, ripeness stages, and capture devices.

## Recommended protocol

1. Define labels and collect data from multiple individual fruits.
2. Divide by specimen or capture session into train, validation, and test partitions.
3. Train on the training partition.
4. Select the distance threshold using validation data only.
5. Evaluate once on a held-out test set.
6. Compare with a pretrained-feature baseline and alternative encoders.
7. Save the code version, random seed, model checkpoint, and dataset manifest.

A similarity score or embedding distance is not automatically a calibrated probability. Report the decision threshold and its validation protocol.
