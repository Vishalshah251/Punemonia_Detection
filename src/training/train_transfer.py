"""
train_transfer.py
Run from project root:  python -m src.training.train_transfer

Two-phase training:
  Phase 1 — frozen base  : fast head warm-up   (TL_EPOCHS_FROZEN epochs)
  Phase 2 — fine-tuning  : top layers unfrozen (TL_EPOCHS_FINETUNE epochs)
"""
import tensorflow as tf
from src.config.config import (
    TL_EPOCHS_FROZEN, TL_EPOCHS_FINETUNE,
    TL_LR_FROZEN, TL_LR_FINETUNE, TL_MODEL_PATH,
)
from src.data.tf_dataset import get_generators
from src.models.transfer_learning import build_transfer_model
from src.evaluation.metrics import plot_training_curves, evaluate_model


def _callbacks(monitor="val_loss", model_path=TL_MODEL_PATH, patience_es=5, patience_lr=3):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience_es,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=patience_lr,
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]


def main():
    # ── GPU check ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus if gpus else 'None — using CPU'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = get_generators()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Train head only (base frozen)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Frozen base: training head only")
    print("="*60)

    model = build_transfer_model(trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(TL_LR_FROZEN),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TL_EPOCHS_FROZEN,
        callbacks=_callbacks(),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tune top layers of base with very low LR
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2 — Fine-tuning top MobileNetV2 layers")
    print("="*60)

    # Rebuild with top layers unfrozen, reuse trained head weights
    model_ft = build_transfer_model(trainable_base=True)
    model_ft.set_weights(model.get_weights())   # carry over Phase 1 weights

    model_ft.compile(
        optimizer=tf.keras.optimizers.Adam(TL_LR_FINETUNE),  # much lower LR
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history_phase2 = model_ft.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TL_EPOCHS_FINETUNE,
        callbacks=_callbacks(),
    )

    # ── Merge histories for a single curve plot ────────────────────────────
    class MergedHistory:
        def __init__(self, h1, h2):
            self.history = {
                k: h1.history[k] + h2.history[k]
                for k in h1.history
            }

    plot_training_curves(MergedHistory(history_phase1, history_phase2))

    # ── Test evaluation ───────────────────────────────────────────────────
    test_loss, test_acc = model_ft.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    evaluate_model(model_ft, test_gen)

    print(f"\nFine-tuned model saved → {TL_MODEL_PATH}")


if __name__ == "__main__":
    main()
