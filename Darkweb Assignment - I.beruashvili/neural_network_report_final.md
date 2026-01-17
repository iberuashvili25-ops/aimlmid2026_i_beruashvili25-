# AI and ML for Cybersecurity: Darknet Traffic Analysis Report

## Part 1: Mathematical Derivation of Weight Updates

### 1.1 Introduction
This section provides the mathematical foundation for training feedforward neural networks using the **Softmax activation function** and **Categorical Cross-Entropy** loss. This combination is optimal for multi-class classification tasks like network traffic categorization.

### 1.2 Notation and Definitions
*   **Logits ($z_k$):** $z_k = \sum_j W_{kj} a_j + b_k$
*   **Softmax ($p_k$):** $p_k = \frac{\exp(z_k)}{\sum_m \exp(z_m)}$
*   **Cross-Entropy Loss ($L$):** $L = -\sum_k y_k \ln(p_k)$
*   **Learning Rate:** $\mu$

### 1.3 Output Layer Derivation
Using the chain rule, the gradient of the loss with respect to the logits simplifies elegantly:
$$\frac{\partial L}{\partial z_k} = p_k - y_k$$
This represents the "error" at the output layer. The weight update for the output layer is then:
$$\Delta W_{kj} = -\mu (p_k - y_k) a_j$$
In matrix form: **$\Delta W = -\mu (p - y) a^T$**

### 1.4 Hidden Layer Backpropagation
The error $\delta_j$ at a hidden neuron $j$ is calculated by propagating the output error backward:
$$\delta_j = \phi'(v_j) \sum_k (p_k - y_k) W_{kj}$$
Where $\phi'(v_j)$ is the derivative of the hidden layer's activation function (e.g., ReLU or Sigmoid). The weight update for the hidden layer is:
$$\Delta w_{ji} = -\mu \delta_j x_i$$

---

## Part 2: Practical Model Improvement (Assignment 2: Traffic Type Prediction)

### 2.1 Baseline Analysis and Improvement Strategy
The initial baseline model achieved a low accuracy of approximately 70%. The improvement strategy focused on addressing common neural network weaknesses: feature scaling, handling missing values (imputation), and implementing a robust architecture with **BatchNormalization** and **Dropout**.

### 2.2 Model Architecture
A deep neural network with three hidden layers (256, 128, 64 neurons) was used, incorporating BatchNormalization and Dropout (0.3 and 0.2) to stabilize training and prevent overfitting. The output layer used Softmax activation for the four traffic types (Non-Tor, NonVPN, Tor, VPN).

### 2.3 Results and Analysis (Label 1)
The improved model achieved a high overall accuracy of **96.20%**, significantly surpassing the baseline and meeting the assignment's goal.

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **0.9620** |
| Macro Avg F1-Score | 0.92 |
| Weighted Avg F1-Score | 0.96 |

**Classification Report (Label 1)**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Non-Tor | 0.99 | 1.00 | 1.00 | 22089 |
| NonVPN | 0.90 | 0.86 | 0.88 | 4773 |
| Tor | 0.97 | 0.86 | 0.91 | 278 |
| VPN | 0.87 | 0.90 | 0.88 | 4584 |

The model demonstrates near-perfect performance in identifying **Non-Tor** traffic. The slightly lower recall for **Tor** traffic (0.86) suggests a small number of Tor flows are being misclassified, likely as NonVPN or VPN, which is a critical area for further investigation in a cybersecurity context.

**Confusion Matrix (Label 1)**
[Image: /home/ubuntu/Label_1_Traffic_Type_cm.png]

---

## Part 3: Activity Type Prediction (Assignment 3: Multi-Class Classification)

### 3.1 Model Configuration
The same preprocessing and architecture were applied to predict the specific activity type (Label 2). The output layer was configured with Softmax for the eight distinct activity classes identified in the dataset.

### 3.2 Results and Analysis (Label 2)
The model achieved an overall accuracy of **82.86%** for activity type prediction. While lower than the traffic type model, this is a strong result for a multi-class problem with imbalanced classes.

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **0.8286** |
| Macro Avg F1-Score | 0.63 |
| Weighted Avg F1-Score | 0.81 |

**Classification Report (Label 2)**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| AUDIO-STREAMING | 0.79 | 0.77 | 0.78 | 4270 |
| BROWSING | 0.86 | 0.96 | 0.91 | 9292 |
| CHAT | 0.56 | 0.83 | 0.67 | 2326 |
| EMAIL | 0.71 | 0.18 | 0.29 | 1229 |
| FILE-TRANSFER | 0.70 | 0.62 | 0.66 | 2237 |
| P2P | 0.97 | 0.99 | 0.98 | 9704 |
| VIDEO-STREAMING | 0.61 | 0.40 | 0.48 | 1953 |
| VOIP | 0.79 | 0.16 | 0.27 | 713 |

The model performs exceptionally well on **P2P** and **BROWSING** traffic. However, the low recall for **EMAIL** (0.18) and **VOIP** (0.16) indicates that these activities are frequently misclassified as other types, likely due to their lower representation in the dataset and the similarity of their flow characteristics to other classes. This highlights the need for class-weighting or oversampling techniques in future work.

**Confusion Matrix (Label 2)**
[Image: /home/ubuntu/Label_2_Activity_Type_cm.png]

## Conclusion
The project successfully fulfilled all three assignment requirements. The theoretical derivation of the backpropagation algorithm was completed, and the practical implementation demonstrated a significant improvement in traffic classification accuracy from the baseline 70% to **96.20%** for traffic type and **82.86%** for activity type. The analysis of the confusion matrices provides clear direction for future model refinement, particularly for low-recall classes like Email and VOIP.
