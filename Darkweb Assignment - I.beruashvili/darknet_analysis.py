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

def preprocess_data(df, target_col):
    # 1. Drop useless columns (Flow ID, Timestamp, etc. if they exist)
    cols_to_drop = ['Flow ID', 'Timestamp', 'Source IP', 'Destination IP']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 2. Handle missing values (Impute with median)
    df = df.fillna(df.median(numeric_only=True))
    
    # 3. Handle categorical features (e.g., Protocol)
    if 'Protocol' in df.columns:
        df = pd.get_dummies(df, columns=['Protocol'])
    
    # 4. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 5. Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # 6. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, num_classes, le

def build_improved_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

def train_and_evaluate(X, y, num_classes, label_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = build_improved_model(X.shape[1], num_classes)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print(f"\n--- Training Model for {label_name} ---")
    history = model.fit(
        X_train, y_train, 
        validation_split=0.2, 
        epochs=100, 
        batch_size=64, 
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluation
    y_pred = model.predict(X_test)
    if num_classes > 2:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int)
        
    print(f"\nClassification Report for {label_name}:")
    print(classification_report(y_test, y_pred_classes))
    
    return model, history

# Main Execution Flow (Template)
if __name__ == "__main__":
    print("Darknet Traffic Analysis Script Initialized.")
    print("Note: Ensure 'darknet.csv' is in the same directory.")
    
    # try:
    #     df = pd.read_csv('darknet.csv')
    #     
    #     # Assignment 2: Label 1 (Traffic Type)
    #     X1, y1, classes1, le1 = preprocess_data(df, 'Label1')
    #     model1, hist1 = train_and_evaluate(X1, y1, classes1, "Traffic Type (Label 1)")
    #     
    #     # Assignment 3: Label 2 (Activity Type)
    #     X2, y2, classes2, le2 = preprocess_data(df, 'Label2')
    #     model2, hist2 = train_and_evaluate(X2, y2, classes2, "Activity Type (Label 2)")
    #     
    # except FileNotFoundError:
    #     print("Error: darknet.csv not found. Please provide the dataset to run the full analysis.")
