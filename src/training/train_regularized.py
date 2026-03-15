"""
train_regularized.py
Run from project root:  python -m src.training.train_regularized
"""
import tensorflow as tf
from src.config.config import EPOCHS, LR, MODEL_PATH
from src.data.tf_dataset import get_generators
from src.models.cnn_regularized import build_model
from src.evaluation.metrics import plot_training_curves, evaluate_model


def main():
    # ── GPU check ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus if gpus else 'None — using CPU'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = get_generators()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # ── Curves + Evaluation ───────────────────────────────────────────────────
    plot_training_curves(history)

    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    evaluate_model(model, test_gen)

    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
