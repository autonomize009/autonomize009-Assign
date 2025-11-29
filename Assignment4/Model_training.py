import numpy as np
import tensorflow as tf
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

# function to create frames from the data
def frame(x, frame_len, hop_len):

    assert x.shape == (len(x), 6)
    assert x.shape[0] >= frame_len
    assert hop_len >= 1

    n_frames = 1 + (x.shape[0] - frame_len) // hop_len
    shape = (n_frames, frame_len, x.shape[1])
    strides = ((hop_len * x.strides[0],) + x.strides)
    return np.lib.stride_tricks.as_strided(x, shape = shape, strides = strides)

# function to build model based on provided hyperparameters value
def build_model(filters1, filters2, dense_units, dropout):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=filters1, kernel_size=3, activation='relu',
                               input_shape=(26, 6)),
        tf.keras.layers.Conv1D(filters=filters2, kernel_size=3, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

x_recordings = []
y_recordings = []

# labels for different classes of motion
labels = ['bicep_curl', 'hammer_curl', 'wrist_rotation']
base_dir = Path(__file__).resolve().parent # To get the base directory of this python script

for i, label in enumerate(labels):
    # iterates over each files corresponding to each class
    filename = base_dir / f"{label}.csv" 
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    features = data[:, 0:6] # first 6 columns are the input features
    x_recordings.append(features)
    y_recordings.append(i)

x_frames = []
y_frames = []

# creating frame overlapping
for i, recording in enumerate(x_recordings):
    frames = frame(recording, 26, 13)  # 50% overlap
    x_frames.append(frames)
    y_frames.append(np.full(frames.shape[0], y_recordings[i]))

x_frames = np.concatenate(x_frames)
y_frames = np.concatenate(y_frames)
print("x_frames shape:", x_frames.shape) # (total_frames, 26, 6)
print("y_frames shape:", y_frames.shape) # (total_frames,)

# Data normalization
acc = x_frames[:, :, 0:3] # ax, ay, az in mg
gyro = x_frames[:, :, 3:6] # gx, gy, gz in mdps

acc_norm = acc / 2000.0 # full-scale is 2g
gyro_norm = gyro / 2000000.0 # full-scale is 2000dps

x_frames_norm = np.concatenate([acc_norm, gyro_norm], axis = -1)

# train/val/test dataset split at 60/20/20 ratio
x_train, x_temp, y_train, y_temp = train_test_split(
    x_frames_norm, y_frames, test_size = 0.4, random_state = 22, stratify = y_frames
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size = 0.50, random_state = 22, stratify = y_temp
)

print("Training frame samples:", x_train.shape)
print("Validation frame samples:", x_val.shape)
print("Testing frame samples:", x_test.shape)

# model tuning and training
# Hyperparameter grid to try
hyperparams_list = [
    {"filters1": 16, "filters2": 8,  "dense_units": 64,  "dropout": 0.5},
    {"filters1": 32, "filters2": 16, "dense_units": 64,  "dropout": 0.5},
    {"filters1": 16, "filters2": 8,  "dense_units": 128, "dropout": 0.5},
]

best_val_acc = -np.inf
best_model = None
best_history = None
best_config = None

num_epochs = 40

for idx, hp in enumerate(hyperparams_list):
    print(f"Training config {idx+1}/{len(hyperparams_list)}: {hp}")
    model = build_model(**hp)
    history = model.fit(x_train, y_train, epochs = num_epochs, validation_data = (x_val, y_val), verbose = 1)
    # Use best validation accuracy of this run
    val_acc = max(history.history["val_accuracy"])
    print(f"Config {hp} best val_accuracy = {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_history = history
        best_config = hp

print("\nBest hyperparameters based on validation accuracy:")
print(best_config)
print(f"Best val_accuracy: {best_val_acc:.4f}")

# Evaluate best model on test set
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose = 2)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

best_model.summary()

# loss and accuracy curves for training and validation dataset
plt.figure()
plt.plot(best_history.history['loss'], label='train_loss')
plt.plot(best_history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss (Best Config)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(best_history.history['accuracy'], label='train_accuracy')
plt.plot(best_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy (Best Config)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plotting confusion matrix on test dataset
Y_pred = best_model.predict(x_test)
y_pred = np.argmax(Y_pred, axis = 1)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot = True, xticklabels = labels,  yticklabels = labels, cmap = plt.cm.Blues, fmt = 'd', cbar = False)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix (Best Config)')
plt.show()

# plotting ROC curve
num_classes = 3
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

plt.figure()
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], Y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Test Set, Best Config)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Saving the best model into ‘model.h5’
model_filename = base_dir / f"model.h5" 
best_model.save(model_filename)
print("Saved best model to model.h5")

