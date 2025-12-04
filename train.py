import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =====================================================
# Configurations
# =====================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6
SEED = 42

DATASET_PATH = r"C:\Users\ksrak\OneDrive\Desktop\FINAL_YEAR_PROJECT\EXPLAINABLE_AI_BRAIN_TUMOR\DATASET\archive"
SAVE_DIR    = r"C:\Users\ksrak\OneDrive\Desktop\FINAL_YEAR_PROJECT\EXPLAINABLE_AI_BRAIN_TUMOR\CODES\saved_models_resnet"
PLOTS_DIR   = r"C:\Users\ksrak\OneDrive\Desktop\FINAL_YEAR_PROJECT\EXPLAINABLE_AI_BRAIN_TUMOR\CODES\plots_resnet"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# Data Generators
# =====================================================
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dirnassnetectory(
    os.path.join(DATASET_PATH, "Training"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "Validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes

# =====================================================
# Build ResNet50 Model
# =====================================================
def build_resnet(num_classes):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(base.input, out, name="resnet50_head")

resnet_model = build_resnet(num_classes)

# =====================================================
# Compile & Train
# =====================================================
ckpt_path = os.path.join(SAVE_DIR, "best_model.keras")
callbacks = [
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]

resnet_model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

print("\n=== Training ResNet50 Model ===")
history = resnet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final snapshot
final_path = os.path.join(SAVE_DIR, "final_model.keras")
resnet_model.save(final_path)
print(f"\nSaved ResNet50 model to: {final_path}")
print(f"Best checkpoint saved to: {ckpt_path}")

# =====================================================
# Plot Accuracy & Loss
# =====================================================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("ResNet50 Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "resnet50_accuracy.png"), dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("ResNet50 Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "resnet50_loss.png"), dpi=150, bbox_inches="tight")
plt.show()