from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config.config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE


def get_generators():
    """Return (train_gen, val_gen, test_gen) using ImageDataGenerator."""

    # Augmentation only on training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="binary",
    )

    train_gen = train_datagen.flow_from_directory(TRAIN_DIR, shuffle=True,  **common)
    val_gen   = eval_datagen.flow_from_directory(VAL_DIR,   shuffle=False, **common)
    test_gen  = eval_datagen.flow_from_directory(TEST_DIR,  shuffle=False, **common)

    return train_gen, val_gen, test_gen
