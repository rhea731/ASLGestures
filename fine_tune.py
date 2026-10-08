import tensorflow as tf
from tensorflow import keras

DATASET_DIR = "asl_alphabet_train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Classes:", train_dataset.class_names)


# Load the Stage 1 model
model = keras.models.load_model("asl_model_stage1.keras")

model.summary()


# Find the MobileNetV3Small inside our model
base_model = model.layers[2]

# Unfreeze the MobileNet model
base_model.trainable = True

# Keep the early layers frozen
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False


# Show which layers are being fine-tuned
print("\nTrainable layers:")

for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)


# Recompile with a very small learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Fine-tune the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)


# Save the fine-tuned model
model.save("asl_model_finetuned.keras")

print("\nFine-tuned model saved as asl_model_finetuned.keras")
