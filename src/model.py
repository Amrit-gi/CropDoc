"""MobileNetV2 Transfer Learning Classifier Architecture for CropDoc.

Defines model building, transfer learning backbone loading, compilation, and optimization.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Tuple, List, Optional
from config import IMAGE_SIZE, INPUT_SHAPE, LEARNING_RATE, DEFAULT_CLASS_NAMES


def build_model(
    num_classes: int = len(DEFAULT_CLASS_NAMES),
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
    trainable_layers: int = 20
) -> tf.keras.Model:
    """Builds and compiles a MobileNetV2 transfer learning model.

    Args:
        num_classes: Number of output disease classes.
        input_shape: Input image tensor shape (height, width, channels).
        learning_rate: Adam optimizer learning rate.
        trainable_layers: Number of top convolutional layers to unfreeze for fine-tuning.

    Returns:
        Compiled Keras Model instance.
    """
    # 1. Base MobileNetV2 backbone pretrained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model layers initially, then optionally unfreeze top layers
    base_model.trainable = True
    if trainable_layers > 0:
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    else:
        base_model.trainable = False

    # 2. Classification Head Architecture
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Preprocessing scaling for MobileNetV2 [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CropDoc_MobileNetV2")

    # 3. Compile Model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model
