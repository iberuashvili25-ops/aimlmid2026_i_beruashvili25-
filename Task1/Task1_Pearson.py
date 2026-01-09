import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# My data from assignment
x = np.array([-10.00, -7.90, -5.10, -3.50, -1.50, 1.90, 3.00, 5.00, 7.00, 8.50])
y = np.array([6.50, 5.90, 3.60, 4.50, 1.70, 0.50, -2.20, -3.00, -4.00, -5.50])

# Calculate correlation
r, p_value = pearsonr(x, y)
print(f"Pearson correlation coefficient (r): {r:.3f}")
print(f"P-value: {p_value:.3f}")

# Interpretation
if abs(r) < 0.3:
    strength = "weak"
elif abs(r) < 0.7:
    strength = "moderate"
else:
    strength = "strong"

direction = "negative" if r < 0 else "positive"
print(f"\nInterpretation: {strength} {direction} correlation")

if p_value < 0.05:
    print("Statistically significant at 95% confidence level")
else:
    print("NOT statistically significant at 95% confidence level")

# Create visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', s=80, alpha=0.7, label='Data points')

# Add trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, label=f'Trendline (r={r:.3f})')

plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values', fontsize=12)
plt.title(f'Scatter Plot with Pearson Correlation\nr = {r:.3f}, p = {p_value:.3f}', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()