from tensorflow.keras import Sequential, layers
from src.config.config import IMAGE_SIZE


def build_model() -> Sequential:
    """
    4-block CNN:
      Conv2D → BatchNorm → MaxPool  (×4, doubling filters 32→256)
    Followed by Dense head with Dropout for regularisation.
    Input: (224, 224, 1)  Output: sigmoid scalar (binary classification)
    """
    model = Sequential([
        # ── Block 1 ──────────────────────────────────────────────────────────
        layers.Conv2D(32, 3, padding="same", activation="relu",
                      input_shape=(*IMAGE_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # ── Block 2 ──────────────────────────────────────────────────────────
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # ── Block 3 ──────────────────────────────────────────────────────────
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # ── Block 4 ──────────────────────────────────────────────────────────
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # ── Classifier head ──────────────────────────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="pneumonia_cnn")

    return model
