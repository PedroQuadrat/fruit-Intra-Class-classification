# Dataset guidance

## Required labels

Use one folder per confirmed variety, such as:

```text
data/fruits/
├── banana_prata/
└── banana_dagua/
```

Folder labels must reflect verified variety labels, not ripeness or background. Common names can vary regionally; document the identification criteria and how labels were confirmed.

## Public dataset caveat

Public banana image datasets often target ripeness stages, segmentation, or generic banana recognition. Such data may help initialize experiments, but it cannot be assumed to distinguish banana Prata from banana d'água unless the varieties are explicitly labeled and represented. Inspect the dataset documentation, license, collection conditions, and class labels before use.

Possible starting points for exploration (check current availability and license):
- Fruits-360: https://github.com/Horea94/Fruit-Images-Dataset
- Cavendish banana dataset: https://hydralab.rmutsv.app/dataset/cavendish-banana-dataset/

These are not asserted to contain the target Brazilian varieties. They may be unsuitable as the main dataset for this specific research question.

## Capture and leakage control

If no suitable labeled dataset exists, collect your own images. Photograph many individual fruits from multiple angles, backgrounds, lighting conditions, maturity stages, and cameras. Include multiple independent specimens per class.

For evaluation, split by physical fruit or capture session, not just by image. Photos of the same fruit must not be spread across training and test sets. Otherwise the model may memorize the specimen, background, or capture setup.

## Starter code limitation

The training script currently uses a random image-level split to make the prototype easier to run. Replace this with a specimen/session-aware split before reporting scientific results.
