import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def preprocess_data(df, target_col):
    print(f"Preprocessing for {target_col}...")
    # 1. Drop useless columns
    cols_to_drop = ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 2. Handle infinite values (common in network traffic data)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 3. Handle missing values (Impute with median)
    df = df.fillna(df.median(numeric_only=True))
    
    # 4. Clean labels for Label2 (case sensitivity and typos)
    if target_col == 'Label2':
        df[target_col] = df[target_col].str.upper().str.replace('-TRANSFER', '-TRANSFER').str.replace('-STREAMING', '-STREAMING')
        # Specific fixes based on EDA
        df[target_col] = df[target_col].replace({
            'VIDEO-STREAMING': 'VIDEO-STREAMING',
            'VIDEO-STREAMING': 'VIDEO-STREAMING', # Just to be safe
            'FILE-TRANSFER': 'FILE-TRANSFER',
            'AUDIO-STREAMING': 'AUDIO-STREAMING'
        })
    
    # 5. Handle categorical features (Protocol)
    if 'Protocol' in df.columns:
        df = pd.get_dummies(df, columns=['Protocol'])
    
    # 6. Separate features and target
    X = df.drop(columns=['Label1', 'Label2'])
    y = df[target_col]
    
    # 7. Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # 8. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, num_classes, le

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_experiment(X, y, num_classes, le, label_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = build_model(X.shape[1], num_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"Training {label_name} model...")
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=128, callbacks=[early_stop], verbose=0)
    
    # Evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{label_name} Accuracy: {acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {label_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'/home/ubuntu/{label_name.replace(" ", "_")}_cm.png')
    plt.close()
    
    return acc, report

if __name__ == "__main__":
    df = pd.read_csv('/home/ubuntu/upload/Darknet.csv')
    
    # Experiment 1: Label 1
    X1, y1, c1, le1 = preprocess_data(df.copy(), 'Label1')
    acc1, report1 = run_experiment(X1, y1, c1, le1, "Label 1 Traffic Type")
    
    # Experiment 2: Label 2
    X2, y2, c2, le2 = preprocess_data(df.copy(), 'Label2')
    acc2, report2 = run_experiment(X2, y2, c2, le2, "Label 2 Activity Type")
    
    # Save results to a text file for the report
    with open('/home/ubuntu/model_results.txt', 'w') as f:
        f.write(f"Label 1 Accuracy: {acc1:.4f}\n")
        f.write("Label 1 Classification Report:\n")
        f.write(report1)
        f.write("\n" + "="*50 + "\n")
        f.write(f"Label 2 Accuracy: {acc2:.4f}\n")
        f.write("Label 2 Classification Report:\n")
        f.write(report2)
