import pickle
import sys

def extract_features_from_email(email_text, feature_names):
    """
    Copy of the parser function - same as in simple_train.py
    """
    import re
    import numpy as np
    
    text_lower = email_text.lower()
    
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
    
    return [features_dict.get(col, 0) for col in feature_names]

# Main script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_classify.py \"Your email text here\"")
        print("Example: python quick_classify.py \"Win free money now!\"")
        sys.exit(1)
    
    email_text = ' '.join(sys.argv[1:])
    
    try:
        # Load the trained model
        with open('spam_model.pkl', 'rb') as f:
            save_data = pickle.load(f)
        
        model = save_data['model']
        feature_names = save_data['feature_names']
        
        print(f"✅ Model loaded (Accuracy: {save_data['accuracy']:.2%})")
        print(f"📧 Classifying: {email_text[:100]}...")
        
        # Extract features
        features = extract_features_from_email(email_text, feature_names)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        print(f"\n🎯 RESULT: {'SPAM' if prediction == 1 else 'LEGITIMATE'}")
        print(f"   Spam probability: {probability[1]:.2%}")
        print(f"   Legitimate probability: {probability[0]:.2%}")
        
        # Show top features found
        print(f"\n📊 Features found in this email:")
        nonzero_features = [(name, value) for name, value in zip(feature_names, features) if value != 0]
        
        if nonzero_features:
            for name, value in nonzero_features[:10]:  # Show first 10 non-zero
                print(f"   {name}: {value}")
        else:
            print("   No spam indicators found")
            
    except FileNotFoundError:
        print("❌ Error: No trained model found!")
        print("   First run: python simple_train.py")