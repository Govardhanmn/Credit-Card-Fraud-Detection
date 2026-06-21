# Credit Card Fraud Detection & Anomaly Monitoring System

An end-to-end machine learning pipeline and real-time Streamlit monitoring dashboard for credit card fraud detection. The project leverages PCA-transformed transaction features, class-imbalance resampling, and ensemble tree-based classifiers to capture fraudulent signatures with minimal customer friction.

---

## 🚀 Key Features

* **High-Imbalance Resampling Benchmark**: Benchmarks five balancing strategies (**RUS, SMOTE, ADASYN, SMOTETomek, NearMiss**) to handle the highly skewed minority class (0.17% fraud rate).
* **Multi-Model Evaluation Pipeline**: Evaluates and logs 6 metrics (Precision, Recall, F1-Score, ROC-AUC, PR-AUC, MCC) across five model architectures:
  * **Logistic Regression** (baseline linear ML)
  * **Decision Tree** (non-linear ML)
  * **Random Forest** (ensemble bagging)
  * **XGBoost** (ensemble boosting)
  * **Deep Artificial Neural Network (MLP)** (TensorFlow/Keras)
* **Consensus Feature Importance**: Combines tree-based split importances (Random Forest MDI and XGBoost gain) to identify the most predictive PCA dimensions (such as `V17`, `V14`, `V12`, `V10`).
* **Real-time Streamlit Monitoring Dashboard**: A production-ready dashboard featuring:
  * **Top-Bar KPIs**: Real-time evaluation engine stats (PR-AUC, ROC-AUC, Anomaly Centroid).
  * **2D PCA Latent Space Scatter**: Interactive visualization of normal vs. anomalous transaction flows.
  * **L2 Anomaly Score Gauge**: Computes multi-dimensional Euclidean distance against normal transaction centroids.
  * **Batch CSV Transaction Uploader**: Dynamic batch upload and automatic transaction risk scoring.

---

## 📊 Model Performance Benchmarks (Real Test Set Results)

All metrics are evaluated on the held-out test fold (56,962 transactions, 98 fraud cases) preserving the original 0.17% class imbalance:

| Model | Precision ↑ | Recall ↑ | F1-Score ↑ | ROC-AUC ↑ | PR-AUC ↑ (Primary) | MCC ↑ | Composite Score* | Status / Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **87.10%** | 82.65% | **84.82%** | 0.9655 | 0.8553 | **84.82%** | **0.8495** | 🏆 **Best Overall Balance (Winner by Composite)** |
| **XGBoost** | 82.83% | 83.67% | 83.25% | **0.9802** | **0.8772** | 0.8322 | 0.8461 | 🥇 **Best Fraud Capture (Highest PR-AUC)** |
| **ANN (Optimized)** | 83.33% | 81.63% | 82.47% | 0.9740 | 0.7233 | 0.8245 | 0.7939 | 🔶 Strong Deep Learning Alternative |
| **ANN (Default)** | 34.68% | 87.76% | 49.71% | 0.9740 | 0.7233 | 0.5506 | 0.6300 | ❌ High Customer Friction |
| **Logistic Regression** | 6.11% | **91.84%** | 11.46% | 0.9719 | 0.7216 | 0.2335 | 0.4870 | ❌ Production Unsuitable |
| **Decision Tree** | 13.44% | 80.61% | 23.03% | 0.9020 | 0.6509 | 0.3269 | 0.4812 | ❌ Poor Generalization |

*\*Composite Score = PR-AUC × 0.30 + F1-Score × 0.25 + Recall × 0.25 + Precision × 0.20*

### 💡 Selection Rationale
* **Random Forest** is selected if the primary business objective is **minimizing customer friction** (declines). It flags only 12 false positives on the test set, keeping legitimate transactions smooth.
* **XGBoost** is selected if the priority is **maximizing fraud recovery**. It catches one additional fraud case (82/98, representing 83.67% Recall) but at the cost of 17 false positives (82.83% Precision).

```
