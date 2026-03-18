from tensorflow.keras import Sequential, Model, layers, Input
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


# ── CBAM Attention Blocks ─────────────────────────────────────────────────────

def channel_attention(x, ratio=8):
    """Squeeze-and-Excitation style channel attention."""
    filters = x.shape[-1]
    # Squeeze: both avg and max pool across spatial dims
    avg = layers.GlobalAveragePooling2D()(x)
    mx  = layers.GlobalMaxPooling2D()(x)
    # Shared MLP
    dense1 = layers.Dense(filters // ratio, activation="relu")
    dense2 = layers.Dense(filters, activation="sigmoid")
    avg_out = dense2(dense1(avg))
    max_out = dense2(dense1(mx))
    scale = layers.Add()([avg_out, max_out])                  # (batch, filters)
    scale = layers.Reshape((1, 1, filters))(scale)            # broadcast-ready
    return layers.Multiply()([x, scale])


def spatial_attention(x):
    """Spatial attention: highlight where to look."""
    # Channel-wise avg and max → concat → conv → sigmoid mask
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    mx  = layers.Lambda(lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg, mx])           # (H, W, 2)
    mask = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(concat)
    return layers.Multiply()([x, mask])


def cbam_block(x, ratio=8):
    """Full CBAM: channel attention → spatial attention."""
    x = channel_attention(x, ratio)
    x = spatial_attention(x)
    return x


# ── CNN + CBAM Attention Model ────────────────────────────────────────────────

def build_attention_model() -> Model:
    """
    Same 4-block CNN but with a CBAM attention block inserted after Block 3
    and Block 4. Uses the Functional API so GradCAM can target named layers.
    Input: (224, 224, 1)  Output: sigmoid scalar
    """
    import tensorflow as tf

    inputs = Input(shape=(*IMAGE_SIZE, 1), name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3 + CBAM
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)                   # ← attention after block 3
    x = layers.MaxPooling2D()(x)

    # Block 4 + CBAM  (GradCAM target layer: "conv4")
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)                   # ← attention after block 4
    x = layers.MaxPooling2D()(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs, name="pneumonia_cnn_cbam")
