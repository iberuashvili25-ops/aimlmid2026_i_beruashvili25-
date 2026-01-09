import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('i_beruashvili25_38765.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Visualization A: Class Distribution (Bar Chart)
plt.figure(figsize=(12, 5))

# Plot 1: Class Distribution
plt.subplot(1, 2, 1)
class_counts = y.value_counts()
colors = ['#4CAF50', '#F44336']  # Green for legit, Red for spam
bars = plt.bar(['Legitimate (0)', 'Spam (1)'], class_counts, color=colors)
plt.title('Email Class Distribution in Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Email Type', fontsize=12)
plt.ylabel('Number of Emails', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add count labels on bars
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', fontweight='bold')

# Plot 2: Confusion Matrix Heatmap (example values - replace with yours)
plt.subplot(1, 2, 2)
# Example confusion matrix - REPLACE WITH YOUR ACTUAL VALUES!
cm = np.array([[145, 12],  # Replace these numbers
               [8, 135]])  # with your actual confusion matrix

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'])
plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization 1 saved as 'visualizations.png'")
print("   - Left: Shows dataset is balanced (similar spam/legit counts)")
print("   - Right: Heatmap shows model performance - strong diagonal = good predictions")

# Visualization C: Feature Importance (Bar Chart)
plt.figure(figsize=(10, 6))

# Load model and get coefficients
import pickle
model = pickle.load(open('spam_model.pkl', 'rb'))
feature_names = X.columns.tolist()
coefficients = model.coef_[0]

# Get top 10 features
top_indices = np.argsort(np.abs(coefficients))[-10:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_coefficients = [coefficients[i] for i in top_indices]

colors = ['red' if c > 0 else 'blue' for c in top_coefficients]
plt.barh(range(len(top_features)), top_coefficients, color=colors)
plt.yticks(range(len(top_features)), top_features)
plt.title('Top 10 Most Important Features for Spam Detection', 
          fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Add coefficient values on bars
for i, (coef, color) in enumerate(zip(top_coefficients, colors)):
    align = 'left' if coef > 0 else 'right'
    pos = 0.01 if coef > 0 else -0.01
    plt.text(coef + pos, i, f'{coef:.3f}', 
             ha=align, va='center', fontweight='bold',
             color=color)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Visualization 2 saved as 'feature_importance.png'")
print("   - Red bars: Features indicating SPAM (positive coefficients)")
print("   - Blue bars: Features indicating LEGITIMATE (negative coefficients)")
print("   - Longer bars = more important for classification")