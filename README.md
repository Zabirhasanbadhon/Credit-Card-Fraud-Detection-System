# 🛡️ CREDIT CARD FRAUD DETECTION SYSTEM
### Intelligent Financial Transaction Risk Analysis Using Machine Learning

<p align="center">
  A powerful Python-based desktop application that detects potentially fraudulent credit card transactions using machine learning, advanced data preprocessing, and statistical analysis.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/GUI-Tkinter-2C5F2D?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Database-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Data%20Visualization-Matplotlib-11557C?style=flat-square" />
  <img src="https://img.shields.io/badge/Reports-CSV%20%7C%20PDF-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Models-7%20Supervised%20%2B%20Isolation%20Forest-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey?style=flat-square" />
</p>

---

## 🚀 Overview

The **Credit Card Fraud Detection System** is a machine learning-powered desktop application designed to identify potentially fraudulent financial transactions.

Developed using Python, Tkinter, Scikit-learn, SQLite, Matplotlib, and ReportLab, the application provides an interactive environment for transaction risk assessment, batch fraud detection, model evaluation, and fraud review management.

The system trains and compares multiple supervised machine learning classifiers using the publicly available ULB Credit Card Fraud Detection dataset.

It supports individual transaction analysis, bulk CSV processing, historical transaction tracking, interactive analytics, and professional CSV/PDF report generation.

### 🎯 Main Objective

To develop an intelligent fraud-screening and decision-support application that demonstrates how machine learning can identify suspicious transaction patterns in highly imbalanced financial datasets.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🏠 Interactive Dashboard | Overview of transaction analysis and fraud statistics |
| 💳 Single Transaction Detection | Analyze individual transactions for potential fraud |
| 📂 Batch CSV Detection | Process thousands of transactions from CSV files |
| 🧠 Multiple ML Algorithms | Train and compare seven supervised classifiers |
| 🔍 Anomaly Detection | Evaluate Isolation Forest separately |
| ⚖️ Imbalance Handling | Class weights, under-sampling, over-sampling, and SMOTE |
| 🎯 Probability Calibration | Validation-based calibration of fraud probability estimates |
| 📊 Threshold Optimization | Select classification thresholds using F-beta |
| 🚨 Fraud Review Queue | Review flagged transactions and manage review statuses |
| 🗄️ SQLite Database | Store and manage transaction prediction history |
| 📈 Advanced Analytics | Explore 11 interactive analytical charts |
| 📉 Model Performance | Evaluate classification and prediction metrics |
| 📄 PDF Reports | Generate detailed fraud analysis reports |
| 📑 CSV Export | Export transaction results and historical records |
| 🔎 History Management | Search, filter, sort, delete, and clear records |
| 🔐 Privacy-Focused | Keep transaction data and processing local |
| ⚙️ Custom Settings | Configure model and risk assessment preferences |

---

## ⚙️ How It Works

The application follows a structured machine learning pipeline to analyze credit card transactions.

```text
          CREDIT CARD DATASET
                   |
                   ▼
          DATA PREPROCESSING
                   |
                   ▼
           STRATIFIED SPLIT
                   |
                   ▼
        CLASS IMBALANCE HANDLING
                   |
                   ▼
        TRAIN MULTIPLE ML MODELS
                   |
                   ▼
         MODEL COMPARISON
                   |
                   ▼
       VALIDATION & CALIBRATION
                   |
                   ▼
        THRESHOLD OPTIMIZATION
                   |
                   ▼
          MODEL SELECTION
                   |
                   ▼
          FINAL TESTING
                   |
                   ▼
        TRANSACTION DETECTION
                   |
             ┌─────┴─────┐
             ▼           ▼
         LOW RISK     HIGH RISK
             |           |
             ▼           ▼
          HISTORY    REVIEW QUEUE
```

## 🧠 Machine Learning Models

The system compares seven supervised machine learning algorithms and evaluates an unsupervised anomaly detection model separately.

### Supervised Algorithms

| Algorithm | Description |
|---|---|
| Logistic Regression | Linear classification for fraud detection |
| Decision Tree | Rule-based transaction classification |
| Random Forest | Ensemble learning using multiple decision trees |
| K-Nearest Neighbors (KNN) | Classification based on neighboring observations |
| Support Vector Machine (SVM) | Classification using an RBF kernel |
| Gradient Boosting | Sequential ensemble learning |
| Extra Trees | Randomized decision-tree ensemble |

### Unsupervised Algorithm

**Isolation Forest**

Used separately for anomaly detection and evaluation. It is not automatically selected as the primary supervised classifier.

---

## 📊 Machine Learning Methodology

The project uses a carefully structured training and evaluation process.

| Step | Methodology |
|---|---|
| Data Cleaning | Remove exact duplicate rows |
| Missing Values | Median imputation inside the pipeline |
| Feature Scaling | RobustScaler for Time and Amount |
| Data Splitting | Stratified 60% / 20% / 20% |
| Random State | 42 |
| Imbalance Handling | Class weights, under-sampling, over-sampling, and SMOTE |
| Probability Calibration | Sigmoid calibration using validation data |
| Threshold Optimization | F-beta score maximization |
| Model Selection | 50% PR-AUC + 50% F1 on validation |
| Final Evaluation | Untouched test dataset |
| Time-Aware Evaluation | Additional chronological train/test robustness check |

### Data Splitting

The dataset is divided into three subsets:

- **60% Training:** Used to train machine learning models.
- **20% Validation:** Used for calibration, threshold selection, and model selection.
- **20% Testing:** Used for final performance evaluation.

The splitting process uses stratification to preserve the fraud-to-legitimate transaction ratio.

### Model Selection Strategy

The best supervised model is selected using:

\[
Score = 0.5 \times PR\text{-}AUC + 0.5 \times F1
\]

This approach prioritizes fraud detection performance rather than relying on accuracy alone.

### Evaluation Metrics

The system calculates:

- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- False Positive Rate
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report

---

## 📈 Understanding Precision, Recall & PR-AUC

Fraud detection requires more than simply achieving high accuracy.

### 🎯 Precision

Of all transactions flagged as potentially fraudulent, how many were actually fraudulent?

Low precision means more false alarms.

### 🔍 Recall

Of all actual fraudulent transactions, how many did the model successfully identify?

Low recall means more missed fraud cases.

### 📊 PR-AUC

Precision-Recall Area Under the Curve summarizes the trade-off between precision and recall across different classification thresholds.

It is especially useful for highly imbalanced datasets such as credit card fraud detection.

**Why not accuracy alone?**

With only approximately 0.172% fraudulent transactions, a model that predicts every transaction as legitimate can achieve approximately 99.83% accuracy while detecting zero fraud cases.

That is why this project emphasizes precision, recall, F1, and PR-AUC.

---

## 🚨 Fraud Review Queue

The application includes a dedicated Fraud Review Queue inside the **Transaction History** page.

Flagged transactions can be reviewed independently of their machine learning predictions.

### Review Statuses

| Status | Description |
|---|---|
| Pending Review | Transaction is awaiting manual review |
| Reviewed | Transaction has been examined |
| Confirmed Fraud | Reviewer has confirmed fraudulent activity |
| False Positive | Reviewer has determined the flag was incorrect |

Review decisions are stored separately from model predictions.

This distinction ensures that human review outcomes do not silently overwrite machine learning classifications.

---

## 📊 Application Modules

### 🏠 1. Dashboard

Displays an overview of transaction activity, fraud statistics, and system information.

### 💳 2. Single Transaction Detection

Allows users to manually enter transaction details and obtain a fraud-risk assessment.

Features include:

- Time and Amount input.
- V1–V28 feature input.
- Real legitimate/fraud example loading from the local dataset.
- Fraud classification.
- Calibrated probability estimate.
- Risk band.
- Model information.

### 📂 3. Batch CSV Detection

Analyzes multiple transactions from a CSV file.

Features include:

- CSV file browsing.
- Data validation and preview.
- Batch transaction scoring.
- Processing in chunks of 20,000 rows.
- Invalid-row tracking.
- CSV result export.

Required columns:

```text
Time, V1, V2, V3, ..., V28, Amount
```

An optional `Class` column is preserved separately and is never used as a prediction feature.

### 🕘 4. Transaction History

Provides local transaction history management.

- Search and filtering.
- Sorting.
- Transaction deletion.
- Clear history.
- CSV export.
- Fraud Review Queue.

### 📈 5. Analytics

Includes 11 charts for analyzing transaction patterns, fraud distribution, and model-related insights.

### 🧠 6. Model Performance

Displays machine learning evaluation metrics and model comparison results.

### 📄 7. Reports

Generates downloadable reports in:

- CSV format.
- PDF format.

Available report types:

- Full analysis report.
- Flagged transactions only.
- Model performance report.

### ⚙️ 8. Settings & About

Provides application configuration, model retraining options, risk threshold settings, and project information.

---

## 🎯 Risk Classification

The application uses configurable risk bands to categorize transaction scores.

| Risk Band | Condition |
|---|---|
| 🔴 High | Score is greater than or equal to the fraud threshold |
| 🟡 Medium | Score is greater than or equal to the configured threshold ratio |
| 🟢 Low | Score is below the medium-risk boundary |

The default medium-risk ratio is `0.5`.

These bands are configurable through Settings.

**Important:** Risk bands are operational score categories, not independently validated guarantees of actual fraud probability.

---

## ▶️ Run the Application

After installing the dependencies and training the models:

```bash
python main.py
```

The Tkinter desktop application will launch.

---

## 📂 Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   ├── *.pkl
│   ├── model_metrics.json
│   └── model_config.json
│
├── database/
│   └── SQLite database
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── threshold_optimizer.py
│   ├── prediction.py
│   ├── batch_detection.py
│   ├── database.py
│   ├── analytics.py
│   ├── report_generator.py
│   ├── validation.py
│   └── app_context.py
│
├── gui/
│   ├── styles.py
│   ├── dashboard.py
│   ├── single_detection.py
│   ├── batch_detection.py
│   ├── history.py
│   ├── analytics_page.py
│   ├── model_performance.py
│   ├── reports.py
│   └── settings.py
│
├── assets/
│   ├── icon.png
│   └── images/
│
├── reports/
├── logs/
│
└── tests/
```

### 📌 Important Directories

| Directory | Purpose |
|---|---|
| `src/` | Machine learning, preprocessing, database, and business logic |
| `gui/` | Tkinter interface and application pages |
| `dataset/` | Input credit card transaction dataset |
| `models/` | Trained machine learning models and configuration |
| `database/` | Local SQLite database |
| `assets/` | Application icons and screenshots |
| `reports/` | Generated CSV and PDF reports |
| `logs/` | Application logs |
| `tests/` | Unit and integration tests |

The Fraud Review Queue is available as the second tab inside Transaction History.

---

## 🔐 Privacy & Security

Privacy is an important part of this project.

- All processing takes place locally.
- No transaction information is sent to external servers.
- No card numbers, CVVs, PINs, or customer names are requested or stored.
- V1–V28 feature values are not stored by default.
- Feature-value storage can be enabled through application settings.
- Transaction history can be deleted through the application.
- SQLite is used for local database management.

**Security Note:** Pickle-based model files can execute code when loaded. Only load model files generated by your own trusted training process. Never load untrusted model files.

---

## ⚠️ Disclaimer

**This application is a fraud-screening and decision-support tool, not a banking or payment-security system.**

Machine learning predictions are automated risk assessments based on patterns learned from historical training data.

- A flagged transaction is not confirmed fraud.
- An unflagged transaction is not guaranteed legitimate.
- Fraud probability is an estimate, not a guarantee.
- The application does not block transactions.
- It does not contact banks or payment providers.
- It does not transmit transaction data anywhere.

Do not use this application as the sole basis for real financial decisions.

---

## 🔮 Future Improvements

- [ ] Time-aware cross-validation.
- [ ] Cost-sensitive threshold optimization using real fraud-loss and review costs.
- [ ] SHAP-style model explanations, with limitations due to anonymized features.
- [ ] Real-time model drift monitoring.
- [ ] Machine learning model versioning.
- [ ] Advanced fraud pattern visualization.
- [ ] Multi-user audit logging.
- [ ] Installable Windows executable using PyInstaller.
- [ ] Enhanced PDF reporting and analytics.
- [ ] Improved desktop UI and user experience.

---

## 👨‍💻 Developer

**ZaBir HaSan**

<p align="center">
  <strong>🛡️ CREDIT CARD FRAUD DETECTION SYSTEM</strong>
  <br />
  <em>Detect Risk. Analyze Patterns. Support Better Decisions.</em>
  <br /><br />
  Built with ❤️ using Python, Tkinter & Machine Learning.
</p>

---
