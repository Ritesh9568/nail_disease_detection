# model.py
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Input image shape
input_shape = (224, 224, 3)

# Load VGG16 base model
vgg_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=input_shape)
)

# Freeze base model layers
for layer in vgg_base.layers:
    layer.trainable = False

# Custom classifier head
x = GlobalAveragePooling2D()(vgg_base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

# ✅ Multi-class output (17 diseases)
output = Dense(17, activation="softmax")(x)

# Final model
model = Model(inputs=vgg_base.input, outputs=output)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
