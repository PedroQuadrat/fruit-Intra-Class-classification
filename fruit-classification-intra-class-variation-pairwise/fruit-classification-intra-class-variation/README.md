# Fruit Classification with Intra-Class Variation

Starter repository for a Python computer-vision application that classifies visually similar fruit categories (initial example: banana Prata vs. banana d'água).

> **Research prototype:** this repository provides a baseline pipeline, not a validated classifier. You must collect and label representative images, train the model, and evaluate it on images not used during training before relying on predictions.

## Goals

- Train an image classifier using transfer learning (EfficientNet-B0).
- Run predictions through a small Streamlit web app.
- Keep code, setup instructions, dataset guidance, and evaluation notes in one repository.
- Make it possible to add classes beyond bananas later.

## Stack

- Python 3.10 or 3.11
- PyTorch / Torchvision
- Pillow
- Streamlit
- scikit-learn (evaluation metrics)
- Git / GitHub

No paid software is required. A GPU is helpful for training but not mandatory for a small first experiment. Google Colab is an optional cloud alternative.

## Repository layout

```text
fruit-classification-intra-class-variation/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── README.md
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   └── evaluation.md
└── data/
    └── .gitkeep
```

The trained model is saved to `models/fruit_classifier.pth` (not committed to Git by default).

## 1. Set up

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installing PyTorch fails, use the selector at https://pytorch.org/get-started/locally/ to choose the command suitable for your operating system and CUDA setup.

## 2. Prepare the dataset

Arrange images into class folders. The folder names become the class labels:

```text
data/fruits/
├── banana_prata/
│   ├── image_001.jpg
│   └── image_002.jpg
└── banana_dagua/
    ├── image_001.jpg
    └── image_002.jpg
```

The training script splits each class into training and validation subsets. Preferably, capture multiple individual fruits, backgrounds, lighting conditions, cameras, angles, and maturity stages. Do not put near-duplicate photos of the same fruit in both subsets; that can inflate validation scores. See [`docs/dataset.md`](docs/dataset.md).

## 3. Train

```bash
python train.py --data-dir data/fruits --output models/fruit_classifier.pth --epochs 10
```

The first run may download pretrained EfficientNet weights. An internet connection is required for that initial download unless the weights are already cached. You can override the default image size and validation fraction; see `python train.py --help`.

## 4. Launch the app

After training has produced the checkpoint:

```bash
streamlit run app.py
```

Upload a photo in the browser. The app displays the predicted class and model scores. Scores are not guaranteed to be calibrated probabilities and should not be treated as certainty.

## 5. Evaluate

```bash
python evaluate.py --data-dir data/fruits --checkpoint models/fruit_classifier.pth
```

Report per-class precision, recall, F1-score, confusion matrix, and the test-set composition. For a credible final evaluation, set aside a test set before model development rather than repeatedly tuning against the validation set. See [`docs/evaluation.md`](docs/evaluation.md).

## Suggested Git commands

```bash
git init
git add .
git commit -m "Initial fruit classification prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fruit-classification-intra-class-variation.git
git push -u origin main
```

Create the empty GitHub repository first, then replace the example remote URL with your own.

## Limitations and next steps

- The project does not include a fruit dataset or a trained model.
- Performance on banana varieties is an empirical question; visual differences may be subtle and affected by ripeness, cultivar, lighting, and camera.
- A classifier always chooses among its known classes unless an explicit unknown-class/rejection strategy is added.
- Next steps: collect data, define labeling rules, train a baseline, inspect errors, compare architectures, and document measured results.
