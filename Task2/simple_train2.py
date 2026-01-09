import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import re
import numpy as np

# ========== EMAIL PARSER FUNCTION ==========
def extract_features_from_email(email_text, feature_names):
    """
    Convert raw email text to same features as CSV
    Returns: List of feature values in same order as training data
    """
    text_lower = email_text.lower()
    
    # Extract features - these should match your CSV columns
    features_dict = {
        'word_count': len(email_text.split()),
        'char_count': len(email_text),
        'link_count': len(re.findall(r'http[s]?://', email_text)),
        'free_count': text_lower.count('free'),
        'win_count': text_lower.count('win') + text_lower.count('winner'),
        'urgent_count': text_lower.count('urgent'),
        'money_count': len(re.findall(r'\$|dollar|money|cash|price', text_lower)),
        'exclamation_count': email_text.count('!'),
        'uppercase_ratio': sum(1 for c in email_text if c.isupper()) / max(1, len(email_text)),
        'url_count': len(re.findall(r'www\.|\.com|\.net|\.org', text_lower)),
        'click_count': text_lower.count('click'),
        'offer_count': text_lower.count('offer') + text_lower.count('deal'),
        'prize_count': text_lower.count('prize') + text_lower.count('reward'),
        'limited_count': text_lower.count('limited') + text_lower.count('exclusive'),
        'bonus_count': text_lower.count('bonus') + text_lower.count('extra'),
        'guarantee_count': text_lower.count('guarantee') + text_lower.count('guaranteed'),
        'risk_free_count': text_lower.count('risk') + text_lower.count('risk-free'),
        'number_count': len(re.findall(r'\d+', email_text)),
        'special_char_count': len(re.findall(r'[!@#$%^&*()_+=|<>?{}\[\]~-]', email_text)),
        'question_count': email_text.count('?'),
        'allcaps_count': sum(1 for word in email_text.split() if len(word) > 1 and word.isupper()),
        'avg_word_length': np.mean([len(word) for word in email_text.split()]) if email_text.split() else 0,
        'has_attachment': int('attachment' in text_lower or 'attach' in text_lower or '.pdf' in text_lower),
        'has_free_money': int('free money' in text_lower or 'free cash' in text_lower),
        'has_click_here': int('click here' in text_lower or 'click below' in text_lower),
        'has_urgent_action': int('urgent action' in text_lower or 'act now' in text_lower),
        'has_dear_friend': int('dear friend' in text_lower or 'dear sir' in text_lower),
    }
    
    # Return in same order as training features
    return [features_dict.get(col, 0) for col in feature_names]

def classify_email(email_text, model, feature_names):
    """
    Classify a single email
    """
    features = extract_features_from_email(email_text, feature_names)
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    return {
        'prediction': 'SPAM' if prediction == 1 else 'LEGITIMATE',
        'spam_probability': probability[1],
        'confidence': max(probability)
    }

# ========== MAIN TRAINING CODE ==========
print("=== SIMPLE SPAM DETECTOR TRAINING ===")

# 1. Load your data
df = pd.read_csv('i_beruashvili25_38765.csv')
print(f"✅ Loaded {len(df)} emails")

# 2. Split features (X) and target (y)
# Assuming last column is target
feature_names = df.columns.tolist()[:-1]  # All column names except last
X = df.iloc[:, :-1]  # All columns except last
y = df.iloc[:, -1]   # Last column only

print(f"✅ Features: {len(feature_names)} columns")
print(f"✅ Target column: '{df.columns[-1]}'")
print(f"✅ Class counts: Spam={sum(y)}, Legit={len(y)-sum(y)}")

# 3. Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"✅ Training set: {len(X_train)} emails (70%)")
print(f"✅ Testing set: {len(X_test)} emails (30%)")

# 4. Train model
print("\n🏋️ Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained!")

# 5. Check accuracy on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 TEST RESULTS (30% unseen data):")
print(f"Accuracy: {accuracy:.2%}")
print(f"Confusion Matrix:")
print(f"         Predicted")
print(f"         Legit  Spam")
print(f"Actual 0  {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       1  {cm[1,0]:4d}    {cm[1,1]:4d}")

print(f"\n📈 Performance Metrics:")
print(f"Correct predictions: {cm[0,0] + cm[1,1]} / {len(y_test)}")
print(f"False Positives (legit marked as spam): {cm[0,1]}")
print(f"False Negatives (spam missed): {cm[1,0]}")

# 6. Show important features
print("\n🔍 TOP 5 FEATURES FOR SPAM DETECTION:")
coefficients = model.coef_[0]

# Sort by absolute value (most important)
important = sorted(zip(feature_names, coefficients), 
                   key=lambda x: abs(x[1]), reverse=True)[:10]

print("\nMost important features (positive = spam indicator):")
for i, (feature, coef) in enumerate(important[:5], 1):
    direction = "SPAM indicator" if coef > 0 else "LEGIT indicator"
    print(f"  {i}. {feature}: {coef:.4f} ({direction})")

print("\nFull coefficient list:")
for feature, coef in zip(feature_names, coefficients):
    if abs(coef) > 0.1:  # Only show meaningful coefficients
        direction = "→ SPAM" if coef > 0 else "→ LEGIT"
        print(f"  {feature:20s} {coef:7.4f} {direction}")

# 7. Save everything
save_data = {
    'model': model,
    'feature_names': feature_names,
    'accuracy': accuracy,
    'confusion_matrix': cm,
    'coefficients': coefficients
}

with open('spam_model.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print(f"\n💾 Model saved to 'spam_model.pkl'")

# 8. TEST THE EMAIL PARSER
print("\n" + "="*50)
print("TESTING EMAIL PARSER")
print("="*50)

# Test with example emails
test_spam = "WIN FREE PRIZE! Click http://example.com URGENT offer! $1000 CASH!!!"
test_legit = "Hello team, meeting tomorrow at 10 AM. Please bring reports."

print("\n🧪 Test 1 - Spam email:")
print(f"Text: '{test_spam[:50]}...'")
result = classify_email(test_spam, model, feature_names)
print(f"Result: {result['prediction']}")
print(f"Spam probability: {result['spam_probability']:.2%}")

print("\n🧪 Test 2 - Legitimate email:")
print(f"Text: '{test_legit}'")
result = classify_email(test_legit, model, feature_names)
print(f"Result: {result['prediction']}")
print(f"Spam probability: {result['spam_probability']:.2%}")

# 9. Interactive testing option
print("\n" + "="*50)
print("INTERACTIVE TESTING")
print("="*50)
print("Want to test your own email? (y/n)")
response = input().strip().lower()

if response == 'y':
    print("\nPaste your email text (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    
    email_text = "\n".join(lines[:-1])
    
    if email_text.strip():
        result = classify_email(email_text, model, feature_names)
        print(f"\n🎯 Classification: {result['prediction']}")
        print(f"   Spam probability: {result['spam_probability']:.2%}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Show what features were found
        features = extract_features_from_email(email_text, feature_names)
        print(f"\n📊 Features extracted (non-zero):")
        for name, value in zip(feature_names, features):
            if value != 0:
                print(f"   {name}: {value}")
    else:
        print("No email text provided.")

print("\n" + "="*50)
print("NEXT STEPS:")
print("="*50)
print("1. Create visualizations: python create_visualizations.py")
print("2. Run full test: python test_emails.py")
print("3. Quick classify: python quick_classify.py \"Your email here\"")