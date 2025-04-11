import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Bidirectional, LSTM, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History

def extract_frames(
    video_path: str,
    size: Tuple[int, int] = (224, 224),
    sequence_length: int = 16,
) -> Optional[List[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None # Video not found or cannot be opened
    
    # print(f"Extracting frames from <{video_path}>")
    indices = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, sequence_length, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break # End of video or frame not found
        frame = cv2.resize(frame, size)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    # print(f"Extracted {len(frames)} frames.")
    return frames if len(frames) == sequence_length else None # Not enough frames extracted

def build_dataset(
    root_dir: str,
    classes: List[str] = ["NonViolence", "Violence"],
    size: Tuple[int, int] = (224, 224),
    sequence_length: int = 16,
    max_videos_per_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for i, class_name in enumerate(classes):
        class_path = os.path.join(root_dir, class_name)
        
        # print(f"\nProcessing class: {class_name}")
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        # print(f"Found {len(files)} videos.")

        if max_videos_per_class is not None:
            files = files[:max_videos_per_class]
        
        for video in tqdm(files, desc=f"{class_name[:10]}..."):
            video_path = os.path.join(class_path, video)
            frames = extract_frames(video_path, size=size, sequence_length=sequence_length)
            if frames: # Only add if we have enough frames
                X.append(np.stack(frames))
                y.append(i)
        
        # print(f"{class_name}: Added {len(y) - sum(y if i==0 else 0)} samples")

    return np.array(X, dtype=np.float32), np.array(y)

# *********************************************************************
root_dir = "Real Life Violence Dataset"
classes = ["NonViolence", "Violence"]
size = (224, 224)
sequence_length = 16
max_videos_per_class = 1000
X, y = build_dataset(root_dir, classes, size, sequence_length, max_videos_per_class)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
# *********************************************************************

def build_model(
    input_shape: Tuple[int, int, int, int] = (16, 224, 224, 3),
    base_cnn_trainable: bool = False,
    lstm_units: int = 224,
    dense_units: int = 32,
    dropout_rate: float = 0.15,
    l2_reg: float = 5e-3
) -> Model:
    inputs = Input(shape=input_shape)

    # MobileNetV2 pre-trained
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', alpha=0.5)
    base_cnn.trainable = base_cnn_trainable

    x = TimeDistributed(base_cnn)(inputs)
    x = Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg))(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )

    return model

def plot_metrics(history: History, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)

    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
        plt.close()

    if 'loss' in history.history and 'val_loss' in history.history:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ["NonViolence", "Violence"],
    output_dir: str = "plots",
    filename: str = "confusion_matrix.png"
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    y_pred_bin = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred_bin)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ["NonViolence", "Violence"],
    output_dir: str = "plots",
    filename: str = "classification_report.png",
    include_avg: bool = False
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    y_pred_bin = (y_pred > 0.5).astype(int)

    report = classification_report(
        y_true,
        y_pred_bin,
        target_names=class_names,
        output_dict=True
    )

    df_report = pd.DataFrame(report).T

    if not include_avg:
        df_report = df_report.loc[class_names]

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f")
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# *********************************************************************
model = build_model()

# callbacks = [
#     EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True)
# ]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    # callbacks=callbacks,
    verbose=1
)

model.save("last_model.keras")

val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(X_test, y_test, verbose=0)
print(f"\nEvaluación en Test Set:")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {val_acc:.4f}")
print(f"AUC: {val_auc:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")

y_pred = model.predict(X_test)

plot_metrics(history)
plot_confusion_matrix(y_test, y_pred, class_names=classes)
plot_classification_report(y_test, y_pred, class_names=classes)
# *********************************************************************
