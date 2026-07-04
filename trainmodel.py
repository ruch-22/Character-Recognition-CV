from function import *

import os
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Create label mapping
label_map = {label: num for num, label in enumerate(actions)}

# ===========================
# Load Dataset
# ===========================

features = []
labels = []

for action in actions:
    for image_num in range(no_sequences):

        npy_path = os.path.join(
            DATA_PATH,
            action,
            f"{image_num}.npy"
        )

        if os.path.exists(npy_path):

            keypoints = np.load(npy_path)

            features.append(keypoints)
            labels.append(label_map[action])

x = np.array(features)
y = to_categorical(labels)

print("\nDataset Loaded Successfully")
print("----------------------------")
print("Input Shape :", x.shape)
print("Output Shape:", y.shape)
print("Classes     :", actions)
print("----------------------------\n")

# ===========================
# Train-Test Split
# ===========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ===========================
# TensorBoard Logs
# ===========================

log_dir = "Logs"
tb_callback = TensorBoard(log_dir=log_dir)

# ===========================
# Early Stopping
# ===========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# ===========================
# Build Neural Network
# ===========================

model = Sequential([

    Dense(
        128,
        activation="relu",
        input_shape=(63,)
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        len(actions),
        activation="softmax"
    )

])

# ===========================
# Compile Model
# ===========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===========================
# Model Summary
# ===========================

model.summary()

# ===========================
# Train Model
# ===========================

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[
        tb_callback,
        early_stop
    ]
)

# ===========================
# Evaluate Model
# ===========================

loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ===========================
# Save Model
# ===========================

model.save("model.keras")

print("\nModel saved successfully as model.keras")