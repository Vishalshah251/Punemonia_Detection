import numpy as np
import tensorflow as tf


def compute_gradcam(model, img_array, layer_name: str) -> np.ndarray:
    """
    Compute GradCAM heatmap for a single preprocessed image.

    Args:
        model:      Trained Keras model (Functional API).
        img_array:  np.ndarray of shape (1, H, W, C), values in [0, 1].
        layer_name: Name of the target conv layer (e.g. 'conv4').

    Returns:
        heatmap: np.ndarray of shape (H', W') normalised to [0, 1].
    """
    # Sub-model that outputs (target conv activations, final prediction)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        # For binary classification use the single output neuron
        loss = predictions[:, 0]

    # Gradients of the class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)                 # (1, H', W', C)

    # Global average pool the gradients → importance weight per channel
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))           # (C,)

    # Weighted combination of feature maps
    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)   # (H', W')

    # ReLU + normalise to [0, 1]
    cam = tf.nn.relu(cam).numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    return cam
