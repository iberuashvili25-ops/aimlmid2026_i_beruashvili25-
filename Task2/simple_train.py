import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

print("=== SIMPLE SPAM DETECTOR TRAINING ===")

# 1. Load your data
df = pd.read_csv('i_beruashvili25_38765.csv')
print(f"✅ Loaded {len(df)} emails")

# 2. Split features (X) and target (y)
# Assuming last column is target
X = df.iloc[:, :-1]  # All columns except last
y = df.iloc[:, -1]   # Last column only

print(f"✅ Features: {X.shape[1]}, Target: {y.name}")

# 3. Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"✅ Training set: {len(X_train)} emails")
print(f"✅ Testing set: {len(X_test)} emails")

# 4. Train model
print("\n🏋️ Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model trained!")

# 5. Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 RESULTS:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Confusion Matrix:")
print(f"         Predicted")
print(f"         No Spam  Spam")
print(f"Actual 0  {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       1  {cm[1,0]:4d}    {cm[1,1]:4d}")

# 6. Save model
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n💾 Model saved as 'spam_model.pkl'")

# 7. Show important features
print("\n🔍 Top features for spam detection:")
feature_names = X.columns.tolist()
coefficients = model.coef_[0]

# Sort by importance
important = sorted(zip(feature_names, coefficients), 
                   key=lambda x: abs(x[1]), reverse=True)[:5]

for feature, coef in important:
    direction = "SPAM" if coef > 0 else "NOT SPAM"
    print(f"  {feature}: {coef:.4f} ({direction})")