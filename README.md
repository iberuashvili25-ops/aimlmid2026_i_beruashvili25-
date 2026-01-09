Midterm Exam - AI for Cybersecurity
Student: Irakli Beruashvili
Date: January 2026

Task 1: Find Pearson's Correlation Coefficient
Data Points from the Graph
I collected these (x,y) coordinates by hovering over the blue dots:

X values: [-10.00, -7.90, -5.10, -3.50, -1.50, 1.90, 3.00, 5.00, 7.00, 8.50]
Y values: [6.50, 5.90, 3.60, 4.50, 1.70, 0.50, -2.20, -3.00, -4.00, -5.50]

Calculation Results

Pearson's r: -0.989
P-value: 0.000 (less than 0.001)

Interpretation: Very strong negative correlation that is statistically significant.

Visualization

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/71a4d1e2-04a3-425b-b4f6-4cbde8f21032" />

The scatter plot shows all points closely follow a downward trend line, confirming the strong negative correlation.




Task 2: Spam email detection

The main goal of this task is to develop one Python console application for email classification within spam and legitimate classes. 
1. Dataset
File: i_beruashvili25_38765.csv (uploaded to repository)

2. Model Training
Logistic Regression trained on 70% of data

Model code: simple_train.py
Top features for spam detection:

urgent_count (strongest spam indicator)
free_count
link_count
winner_count
word_count

3. Model Validation
Accuracy: 93.33% on 30% test data
Confusion Matrix:

text
         Predicted
         Legitimate  Spam
Actual 0     145      12
       1       8     135
4. Email Parser
The application (simple_train.py) can:
Extract features from raw email text
Classify emails as spam or legitimate
Show classification confidence
5. Example Spam Email
text
WIN FREE PRIZE! Click http://example.com URGENT $1000 CASH!!!
Why spam: Contains spam keywords, link, urgency, money references.

6. Example Legitimate Email
text
Hello team, meeting tomorrow at 10 AM. Please bring reports.
Why legitimate: Normal business language, no spam indicators.

7. Visualizations
Visualization A: Class Distribution & Confusion Matrix Heatmap
Shows balanced dataset and model performance

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/a44c41ff-c867-46c3-8b4b-06093ac604e7" />

Visualization B: Feature Importance
Shows which features most indicate spam vs legitimate

<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/f3d9a138-08b1-496b-8967-1eb432e5eb6e" />

