# train.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ==========================
# Dataset Paths
# ==========================
TRAIN_DIR = "dataset/train/train"
TEST_DIR = "dataset/test/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# ==========================
# Load VGG16 Base Model
# ==========================
vgg_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Freeze base layers initially
for layer in vgg_base.layers:
    layer.trainable = False

# ==========================
# Custom Classifier Head
# ==========================
x = GlobalAveragePooling2D()(vgg_base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

# 17-class output
output = Dense(17, activation="softmax")(x)

model = Model(inputs=vgg_base.input, outputs=output)

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# Data Generators
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==========================
# Callbacks
# ==========================
os.makedirs("model", exist_ok=True)

checkpoint = ModelCheckpoint(
    "model/nail_disease_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==========================
# Train Model
# ==========================
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print("✅ Model training completed and saved")
