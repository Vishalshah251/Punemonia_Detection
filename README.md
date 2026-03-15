# Pneumonia Detection from Chest X-Rays — CNN

Binary classification (**NORMAL vs PNEUMONIA**) using a custom CNN built with TensorFlow/Keras.

## Project Structure

```
pneumonia-xray-cnn/
├── data/raw/chest_xray/        # Dataset (train / val / test)
├── models/                     # Saved .keras checkpoints
├── notebooks/
│   └── Cnn_Training.ipynb      # End-to-end interactive pipeline
├── results/plots/              # Training curves & confusion matrix
└── src/
    ├── config/config.py        # Paths & hyperparameters
    ├── data/tf_dataset.py      # ImageDataGenerator pipeline
    ├── models/cnn_regularized.py
    ├── evaluation/metrics.py
    └── training/train_regularized.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Notebook** (recommended):
```bash
jupyter notebook notebooks/Cnn_Training.ipynb
```

**Script**:
```bash
python -m src.training.train_regularized
```

## Architecture

| Block | Layers |
|-------|--------|
| 1–4   | Conv2D (32→256) + BatchNorm + MaxPool |
| Head  | GlobalAveragePooling → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → sigmoid |

- Optimizer: Adam  
- Loss: Binary Cross-Entropy  
- Callbacks: EarlyStopping · ReduceLROnPlateau · ModelCheckpoint  
- Target: **≥ 85% validation accuracy**
# Punemonia_Detection
