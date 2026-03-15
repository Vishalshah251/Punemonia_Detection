import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.config.config import RESULTS_DIR, CLASSES
import os


def plot_training_curves(history):
    """Plot accuracy and loss curves side-by-side and save to results/plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[metric[0]], label=f"Train {title}")
        ax.plot(history.history[metric[1]], label=f"Val {title}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")


def evaluate_model(model, test_gen):
    """Print classification report and plot confusion matrix."""
    test_gen.reset()
    y_true = test_gen.classes

    preds = model.predict(test_gen, verbose=1)
    y_pred = (preds.ravel() >= 0.5).astype(int)

    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved → {path}")
