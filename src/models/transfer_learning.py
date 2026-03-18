"""
transfer_learning.py
MobileNetV2 pretrained on ImageNet, adapted for grayscale binary classification.

Two-phase strategy:
  Phase 1 — base frozen:   train only the custom head (fast convergence)
  Phase 2 — fine-tuning:   unfreeze top layers of base with a very low LR
"""
import tensorflow as tf
from tensorflow.keras import Model, layers
from src.config.config import IMAGE_SIZE, TL_UNFREEZE_FROM


def _grayscale_to_rgb():
    """Repeat grayscale channel 3× so MobileNetV2 (RGB) accepts the input."""
    return layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1),
                         name="grayscale_to_rgb")


def build_transfer_model(trainable_base: bool = False) -> Model:
    """
    Build MobileNetV2 transfer learning model.

    Args:
        trainable_base: False → Phase 1 (head only).
                        True  → Phase 2 (top layers unfrozen).
    Returns:
        Compiled-ready Keras Model.
    """
    inputs = layers.Input(shape=(*IMAGE_SIZE, 1), name="input")

    # Convert grayscale → RGB for MobileNetV2
    x = _grayscale_to_rgb()(inputs)

    # MobileNetV2 pretrained on ImageNet — exclude top classifier
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    if trainable_base:
        # Phase 2: unfreeze layers from TL_UNFREEZE_FROM onward
        for layer in base.layers[:TL_UNFREEZE_FROM]:
            layer.trainable = False
        for layer in base.layers[TL_UNFREEZE_FROM:]:
            layer.trainable = True
    else:
        # Phase 1: freeze entire base
        base.trainable = False

    x = base(x, training=trainable_base)  # training=True enables BN updates during fine-tune

    # Custom classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="drop_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs, outputs, name="pneumonia_mobilenetv2")
